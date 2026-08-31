"""Offline tests for cooperative fan-out branch cancellation

Before this, an over-budget branch was merely ABANDONED: its subprocesses
were killed and its ledger entry closed, but the thread kept burning API
quota and held its memory-admission slot until run_task returned — a
runaway branch could block *other* branches from admission. Now the budget
path also fires a per-branch stop event; a thread-aware stdout/stderr guard
raises ``AgentStoppedError`` inside that branch thread on its next print,
so the branch aborts cooperatively and releases its slot.

  conda run -n scilink python -m pytest tests/test_fanout_branch_cancellation.py -v
"""
import json
import os
import sys
import threading
import time

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

import numpy as np
import pytest

import scilink.agents.meta_agent.fanout as fo
from scilink.ui.output_capture import AgentStoppedError


# ---------------------------------------------------------------- unit level


def test_thread_stop_stream_raises_only_for_stopped_thread(capsys):
    stream = fo._ThreadStopStream(sys.stdout)
    stream.write("fine before registration\n")

    ev = threading.Event()
    fo._register_branch_stop(ev)
    try:
        stream.write("fine while event unset\n")
        ev.set()
        with pytest.raises(AgentStoppedError):
            stream.write("must raise\n")
    finally:
        fo._unregister_branch_stop()
    # After unregistration the same thread writes freely again.
    stream.write("fine after unregistration\n")


def test_other_threads_unaffected_by_a_stopped_branch():
    ev = threading.Event()
    ev.set()
    errors = []

    def branch():
        fo._register_branch_stop(ev)
        try:
            stream = fo._ThreadStopStream(sys.stdout)
            try:
                stream.write("branch write\n")
                errors.append("branch write did NOT raise")
            except AgentStoppedError:
                pass
        finally:
            fo._unregister_branch_stop()

    t = threading.Thread(target=branch)
    t.start()
    t.join()
    # The coordinator thread keeps writing through its own guard unharmed.
    fo._ThreadStopStream(sys.stdout).write("coordinator write\n")
    assert not errors


def test_stop_guard_install_is_idempotent():
    old_out, old_err = sys.stdout, sys.stderr
    try:
        fo._ensure_stop_guard_installed()
        first_out = sys.stdout
        assert isinstance(first_out, fo._ThreadStopStream)
        fo._ensure_stop_guard_installed()
        assert sys.stdout is first_out, "guard stacked a second layer"
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def test_user_stop_propagates_uncaught(tmp_path):
    """AgentStoppedError NOT caused by the branch's own budget stop (i.e. a
    user-initiated stop) must stay unrecoverable and propagate."""
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent)
    meta = MetaOrchestratorAgent(api_key="sk-dummy", base_dir=str(tmp_path))

    class UserStoppedChild:
        def run_task(self, task, context=None, autonomy=None):
            raise AgentStoppedError("user clicked Stop")

    orig = fo._make_ephemeral_analysis_child
    fo._make_ephemeral_analysis_child = lambda orch, base_dir: UserStoppedChild()
    try:
        own_event = threading.Event()   # never set: not a budget cancel
        with pytest.raises(AgentStoppedError):
            fo._run_one_branch(meta, {"data_path": str(tmp_path),
                                      "task": "t", "label": "L"},
                               [], {"index": 0}, None, None, own_event)
    finally:
        fo._make_ephemeral_analysis_child = orig
    # The branch's admission slot was still released on the way out.
    assert not fo._mem_running


# ---------------------------------------------------------- integration level


def _install_fakes(behaviors):
    """Fake child + gate/fusion LLM, mirroring tests/test_wallclock_budget.py."""
    def fake_child(orch, base_dir):
        class C:
            def run_task(self, task, context=None, autonomy=None):
                import re
                m = re.search(r"PRIMARY dataset for THIS analysis: (\S+)", task)
                beh = behaviors.get(m.group(1) if m else "", "good")
                if beh == "chatty":
                    # Prints in a loop: cancellable on any print. Runs ~20 s
                    # if NOT cancelled — the test's wall clock is the proof
                    # that cancellation actually stopped the thread.
                    for _ in range(400):
                        print("chatty branch still working...")
                        time.sleep(0.05)
                return {"status": "success", "summary": f"{beh} ok",
                        "key_findings": ["finding"], "files_produced": []}
        return C()
    fo._make_ephemeral_analysis_child = fake_child

    def fake_llm(orch, prompt, extra_parts=None):
        if "SCRIPT CONTRACT" in prompt or "AUDITING a computed" in prompt:
            return {"verdict": "accept", "issues": [],
                    "refinement_instructions": "", "method": "qualitative",
                    "rationale": "r", "script": ""}
        if "complementary measurements of ONE system" in prompt:
            return {"detailed_analysis": "fused", "scientific_claims": []}
        return {"verdict": "complementary", "confidence": 0.9, "rationale": "r",
                "join_axis": "T", "join_type": "shared_parameter_axis",
                "fanout_set": list(behaviors), "redundant_clusters": [],
                "unrelated": [], "excluded_notes": ""}
    fo._llm_json = fake_llm


def test_over_budget_branch_is_cancelled_not_abandoned(tmp_path):
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent, MetaMode)
    # Pre-warm the worker-side lazy import (see test_wallclock_budget.py).
    import scilink.agents.exp_agents.analysis_orchestrator  # noqa: F401

    paths = [str(tmp_path / f"{n}.npy") for n in "AB"]
    for p in paths:
        np.save(p, np.zeros((8, 8)))
    A, B = paths
    behaviors = {A: "good", B: "chatty"}

    ag = MetaOrchestratorAgent(base_dir=str(tmp_path), api_key="sk-dummy",
                               meta_mode=MetaMode.AUTONOMOUS)
    orig_child = fo._make_ephemeral_analysis_child
    orig_llm = fo._llm_json
    orig_poll = fo._FANOUT_POLL_S
    _install_fakes(behaviors)
    fo._FANOUT_POLL_S = 0.2
    try:
        t0 = time.monotonic()
        out = json.loads(ag._run_fanout(
            [{"data_path": p, "task": f"Analyze {p}",
              "label": os.path.basename(p)} for p in paths],
            branch_time_budget_s=0.5))
        assert out.get("status") == "success"
        assert out.get("branches_timed_out") == 1

        cancelled = [e for e in ag._delegation_ledger
                     if e.get("fanout") and e.get("timed_out")]
        assert len(cancelled) == 1
        # The ledger verdict is unchanged from #358: degraded, budget error.
        assert cancelled[0]["status"] == "error"
        assert "wall-clock budget" in (cancelled[0].get("error") or "")

        # THE new behavior: the thread actually stopped. Its cancellation is
        # recorded as a late result well before the ~20 s it would have run.
        deadline = time.monotonic() + 5.0
        while (cancelled[0].get("late_result") is None
               and time.monotonic() < deadline):
            time.sleep(0.1)
        late = cancelled[0].get("late_result")
        assert late is not None, "cancelled thread never wound down"
        assert late.get("status") == "cancelled"
        assert time.monotonic() - t0 < 10.0, (
            "cancelled branch kept running to completion")

        # ... and it released its memory-admission slot on the way out.
        assert not fo._mem_running
    finally:
        fo._make_ephemeral_analysis_child = orig_child
        fo._llm_json = orig_llm
        fo._FANOUT_POLL_S = orig_poll


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
