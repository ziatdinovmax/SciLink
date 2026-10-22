"""The fan-out's memory rules and the provider circuit breaker: the capacity
plan (an estimate never refuses a branch), the guard that cancels the
heaviest of several running branches and reruns it alone, and the breaker
that holds admission for a failing model without touching running work.
"""

import json
import os
import threading
import time

import numpy as np
import pytest

from scilink.agents.meta_agent import fanout as fo
from scilink.agents.meta_agent.meta_orchestrator import MetaMode
from scilink.wrappers import llm_limiter


# --------------------------------------------------------------- the estimate

def test_estimate_item_is_the_input_based_estimate_else_the_mode_floor(tmp_path):
    np.save(tmp_path / "a.npy", np.zeros((1000, 1000)))
    item = {"mode": "analysis", "data_path": str(tmp_path / "a.npy")}
    assert fo.estimate_item(item) == fo._branch_mem_estimate(item)
    assert fo.estimate_item({"mode": "planning"}) == fo.MODE_MEM_FLOOR["planning"]


def test_an_estimate_never_refuses_a_capacity_plan(tmp_path, monkeypatch):
    # the fan-out's rule: an input-based estimate is coarse by design and
    # never refuses what the machine may be able to run
    monkeypatch.setattr(fo, "_branch_mem_estimate", lambda b: 40e9)
    item = {"mode": "analysis", "label": "guess", "data_path": str(tmp_path / "missing.npy")}
    plan = fo.plan_capacity([item], {"total": 16e9, "available": 8e9}, max_workers=3)
    assert plan["refused"] == [] and plan["run"] == [item]
    assert plan["run"][0]["_mem_est"] == 40e9 and plan["estimated_bytes"] == 40e9
    assert plan["together"] is False and plan["workers"] == 1


# --------------------------------------------------------------- the breaker

def test_the_breaker_trips_on_a_failing_model_and_a_success_does_not_close_it(monkeypatch):
    monkeypatch.setattr(llm_limiter, "BREAKER_HOLD_S", 5.0)
    for _ in range(llm_limiter.BREAKER_FAILURES - 1):
        llm_limiter.note_provider_failure("m")
    assert llm_limiter.provider_tripped("m") is None
    llm_limiter.note_provider_failure("m")
    assert 0 < llm_limiter.provider_tripped("m") <= 5.0
    assert llm_limiter.provider_tripped() is not None and llm_limiter.provider_tripped("other") is None
    # the retry policy records under the wrapper's PREFIXED name; admission
    # asks with the meta's name, which may be bare: one key
    llm_limiter.reset_breaker()
    for _ in range(llm_limiter.BREAKER_FAILURES):
        llm_limiter.note_provider_failure("anthropic/claude-opus-4-6")
    assert llm_limiter.provider_tripped("claude-opus-4-6") is not None
    assert llm_limiter.provider_tripped("anthropic/claude-opus-4-6") is not None
    assert llm_limiter.provider_tripped("gemini-2.0-flash") is None
    llm_limiter.reset_breaker()
    for _ in range(llm_limiter.BREAKER_FAILURES):
        llm_limiter.note_provider_failure("m")
    llm_limiter.note_provider_ok("m")
    assert llm_limiter.provider_tripped("m") is not None     # the hold runs its course
    llm_limiter.reset_breaker()
    # a brown-out: failures beside successes still trip (half the calls)...
    for ok in [False, True] * llm_limiter.BREAKER_FAILURES:
        llm_limiter.note_provider_ok("m") if ok else llm_limiter.note_provider_failure("m")
    assert llm_limiter.provider_tripped("m") is not None
    llm_limiter.reset_breaker()
    # ... a healthy run with a few failures does not
    for i in range(60):
        llm_limiter.note_provider_failure("m") if i % 10 == 0 else llm_limiter.note_provider_ok("m")
    assert llm_limiter.provider_tripped("m") is None


def test_an_open_breaker_holds_admission_for_that_model_not_running_work(monkeypatch):
    monkeypatch.setattr(llm_limiter, "BREAKER_HOLD_S", 1.0)
    monkeypatch.setattr(fo, "_available_memory", lambda: 8e9)
    for _ in range(llm_limiter.BREAKER_FAILURES):
        llm_limiter.note_provider_failure("m")
    admitted = threading.Event()

    def admit():
        fo._admit_branch("t:breaker", 1e6, "held one", "m")
        admitted.set()
    t = threading.Thread(target=admit, daemon=True)
    t.start()
    assert not admitted.wait(0.3)                # held while the breaker is open for its model
    fo._admit_branch("t:other", 1e6, "another model's item", "other")   # not held
    fo._release_branch("t:other")
    assert admitted.wait(3.0)                    # admitted once the hold ran out
    fo._release_branch("t:breaker")


def test_the_retry_policy_feeds_the_breaker(monkeypatch):
    from scilink.wrappers.litellm_wrapper import call_with_retries
    llm_limiter.reset_breaker()
    monkeypatch.setattr(llm_limiter, "BREAKER_FAILURES", 2)
    monkeypatch.setattr("scilink.wrappers.litellm_wrapper._backoff_s", lambda attempt: 0.0)

    class Throttled(Exception):
        status_code = 429
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        if calls["n"] <= 2:
            raise Throttled("slow down")
        return "ok"
    assert call_with_retries(call, 3, model="m") == "ok"
    assert calls["n"] == 3
    assert llm_limiter.provider_tripped("m") is not None  # two failures of three calls tripped it
    llm_limiter.reset_breaker()
    calls["n"] = -10
    with pytest.raises(Throttled):
        call_with_retries(call, 1, model="m")
    assert llm_limiter.provider_tripped("m") is not None


# ----------------------------------------------------------------- fan-out

def _fanout_meta(tmp_path):
    from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent
    return MetaOrchestratorAgent(base_dir=str(tmp_path / "fm"), api_key="sk-dummy", meta_mode=MetaMode.AUTONOMOUS)


def _fanout_fakes(monkeypatch, behaviours):
    """A fake branch child keyed by its primary dataset, and the gate / fusion
    model answers (tests/test_fanout_branch_cancellation.py's shape); a
    behaviour is a list consumed one run at a time."""
    import re

    def fake_child(orch, base_dir, restore=False):
        class C:
            def run_task(self, task, context=None, autonomy=None):
                m = re.search(r"PRIMARY dataset for THIS analysis: (\S+)", task)
                queue = behaviours.get(m.group(1) if m else "", ["good"])
                beh = queue.pop(0) if len(queue) > 1 else queue[0]
                if beh in ("chatty", "slow"):
                    for _ in range(400 if beh == "chatty" else 60):
                        print(f"{beh} branch still working...")
                        time.sleep(0.05)
                return {"status": "success", "summary": f"{beh} ok", "key_findings": ["finding"],
                        "files_produced": []}
        return C()

    def fake_llm(orch, prompt, extra_parts=None):
        if "SCRIPT CONTRACT" in prompt or "AUDITING a computed" in prompt:
            return {"verdict": "accept", "issues": [], "refinement_instructions": "", "method": "qualitative",
                    "rationale": "r", "script": ""}
        if "complementary measurements of ONE system" in prompt:
            return {"detailed_analysis": "fused", "scientific_claims": []}
        return {"verdict": "complementary", "confidence": 0.9, "rationale": "r", "join_axis": "T",
                "join_type": "shared_parameter_axis", "fanout_set": list(behaviours),
                "redundant_clusters": [], "unrelated": [], "excluded_notes": ""}
    monkeypatch.setattr(fo, "_make_ephemeral_analysis_child", fake_child)
    monkeypatch.setattr(fo, "_llm_json", fake_llm)
    monkeypatch.setattr(fo, "_FANOUT_POLL_S", 0.2)


def test_a_fan_out_branch_estimate_never_refuses_it_and_every_branch_runs(tmp_path, monkeypatch):
    import scilink.agents.exp_agents.analysis_orchestrator  # noqa: F401
    paths = [str(tmp_path / f"{n}.npy") for n in "ABC"]
    for p in paths:
        np.save(p, np.zeros((8, 8)))
    _fanout_fakes(monkeypatch, {p: ["good"] for p in paths})
    monkeypatch.setattr(fo, "machine_memory", lambda: {"total": 8e9, "available": 6e9})
    monkeypatch.setattr(fo, "_available_memory", lambda: 6e9)
    # the input-based estimate never refuses a fan-out branch, however large:
    # branches that do not fit together wait for each other instead
    monkeypatch.setattr(fo, "_branch_mem_estimate", lambda b: 40e9)
    ag = _fanout_meta(tmp_path)
    out = json.loads(ag._run_fanout([{"data_path": p, "task": f"Analyze {p}", "label": os.path.basename(p)}
                                     for p in paths]))
    assert out["status"] == "success" and out["branches_run"] == 3
    assert "not_started" not in out
    assert sorted(e["label"] for e in ag._delegation_ledger if e.get("fanout")) == ["A.npy", "B.npy", "C.npy"]
    assert not fo._mem_running


def test_the_fan_out_guard_cancels_the_heaviest_branch_and_reruns_it_alone(tmp_path, monkeypatch):
    import scilink.agents.exp_agents.analysis_orchestrator  # noqa: F401
    A, B = (str(tmp_path / "A.npy"), str(tmp_path / "B.npy"))
    np.save(A, np.zeros((8, 8)))
    np.save(B, np.zeros((64, 64)))
    _fanout_fakes(monkeypatch, {A: ["slow"], B: ["chatty", "good"]})
    # B's estimate is the larger: it is the one the guard picks
    monkeypatch.setattr(fo, "_branch_mem_estimate",
                        lambda b: 2e9 if str(b.get("data_path", "")).endswith("B.npy") else 1e9)
    low = {"on": False}
    monkeypatch.setattr(fo, "_available_memory", lambda: 1e8 if low["on"] else 8e9)
    threading.Timer(1.0, lambda: low.update(on=True)).start()
    real = fo.guard_memory

    def guard(*a, **k):
        f = real(*a, **k)
        if f is not None:
            low["on"] = False
        return f
    monkeypatch.setattr(fo, "guard_memory", guard)
    ag = _fanout_meta(tmp_path)
    t0 = time.monotonic()
    out = json.loads(ag._run_fanout([{"data_path": p, "task": f"Analyze {p}", "label": os.path.basename(p)}
                                     for p in (A, B)]))
    assert out["status"] == "success" and out["branches_with_output"] == 2, out
    b = next(e for e in ag._delegation_ledger if e.get("label") == "B.npy")
    assert b["memory_pressure"] and b["memory_retried"] and b["retries"] == 1
    assert b["status"] == "success" and "memory_pressure" in b["retry_of_error"]
    assert b["late_result"]["status"] == "cancelled"        # the first attempt wound down
    assert time.monotonic() - t0 < 25.0
    assert not fo._mem_running
