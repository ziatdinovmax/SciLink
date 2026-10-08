"""A swarm analysis item carries its own depth (#756), by the rule a single
delegation and a fan-out branch use: ``profile``, ``targets``,
``time_budget_s`` read by one helper, recorded on the ledger entry, passed to
the child's ``run_task`` on the thread and the process placement, kept on a
memory-guard rerun and on a re-run after a retraction. An unknown profile is
refused with its reason before anything runs.

The workers are stand-ins; the meta, its ledger and the coordinator are real.
No LLM calls.
"""

import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

import scilink.agents.meta_agent.fanout as fo
from scilink.agents.meta_agent import placements, swarm
from scilink.agents.meta_agent.meta_orchestrator import MetaMode, MetaOrchestratorAgent

DEPTH = {"profile": "extract", "targets": ["band position"], "time_budget_s": 120.0}


class Worker:
    """Records the depth each call received; ``until_stopped`` keeps it
    running (printing, which is where a Stop lands) until it is cancelled."""

    calls: list = []

    def __init__(self, mode, base_dir, until_stopped=lambda task: False):
        self.mode, self.base_dir, self.until_stopped = mode, Path(base_dir), until_stopped

    def run_task(self, task, context=None, autonomy=None, **depth):
        Worker.calls.append({"mode": self.mode, "task": task, **depth})
        deadline = time.time() + (1.0 if task == "light" else 0.1)
        while time.time() < deadline or self.until_stopped(task):
            print(".", end="", flush=True)
            time.sleep(0.05)
            if time.time() > deadline + 20:
                break
        return {"status": "success", "summary": f"{task} done", "key_findings": [],
                "files_produced": [], "suggested_followups": [], "warnings": []}


@pytest.fixture()
def meta(tmp_path, monkeypatch):
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SCILINK_SWARM_PLACEMENT", "thread")
    monkeypatch.setattr(swarm, "_POLL_S", 0.05)
    monkeypatch.setattr(swarm.fo, "_available_memory", lambda: 8e9)
    monkeypatch.setattr(swarm, "_memory", lambda: {"total": 16e9, "available": 8e9})
    Worker.calls = []
    m = MetaOrchestratorAgent(base_dir=str(tmp_path / "meta"), model_name="anthropic/claude-sonnet-4-5",
                              meta_mode=MetaMode.AUTONOMOUS, launch_dir=str(tmp_path))
    m._enable_human_feedback = False
    return m


def _workers(monkeypatch, until_stopped=lambda task: False):
    def build(orch, mode, base_dir, **kw):
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        return Worker(mode, base_dir, until_stopped)
    monkeypatch.setattr(swarm, "build_child", build)


def _depth_of(call):
    return {k: v for k, v in call.items() if k in fo._BRANCH_DEPTH_KEYS}


def test_an_items_depth_reaches_its_worker_and_its_entry(meta, monkeypatch):
    _workers(monkeypatch)
    res = json.loads(swarm.run_swarm(meta, [
        {"mode": "analysis", "task": "fit A", "label": "A", **DEPTH},
        {"mode": "analysis", "task": "fit B", "label": "B"},
        {"mode": "planning", "task": "plan C", "label": "C", "profile": "quick"}]))
    assert res["status"] == "success" and res["not_started"] == []
    by_task = {c["task"]: c for c in Worker.calls}
    assert _depth_of(by_task["fit A"]) == DEPTH
    # without depth the call is what it was; a planning item has none to take
    assert set(by_task["fit B"]) == {"mode", "task"}
    assert set(by_task["plan C"]) == {"mode", "task"}
    ledger = {e["task"]: e for e in meta._delegation_ledger}
    assert ledger["fit A"]["depth"] == DEPTH
    assert "depth" not in ledger["fit B"] and "depth" not in ledger["plan C"]


def test_a_memory_guard_rerun_keeps_the_items_depth(meta, monkeypatch):
    monkeypatch.setitem(swarm._MODE_MEM_FLOOR, "analysis", 2e9)        # the heavy one
    runs = {"heavy": 0}

    def until_stopped(task):
        return task == "heavy" and runs["heavy"] == 1
    _workers(monkeypatch, until_stopped)
    real_run_task = Worker.run_task

    def counting(self, task, **kw):
        if task == "heavy":
            runs["heavy"] += 1
        return real_run_task(self, task, **kw)
    monkeypatch.setattr(Worker, "run_task", counting)
    low = {"on": False}
    monkeypatch.setattr(swarm, "_memory", lambda: {"total": 16e9, "available": 1e8 if low["on"] else 8e9})
    threading.Timer(0.4, lambda: low.update(on=True)).start()
    real_guard = swarm._guard_memory

    def guard(*args):
        cancelled = real_guard(*args)
        if cancelled is not None:
            low["on"] = False
        return cancelled
    monkeypatch.setattr(swarm, "_guard_memory", guard)
    res = json.loads(swarm.run_swarm(meta, [
        {"mode": "planning", "task": "light", "label": "light"},
        {"mode": "analysis", "task": "heavy", "label": "heavy", **DEPTH}]))
    assert [(r["label"], r["status"]) for r in res["results"]][1:] == [("heavy", "cancelled"),
                                                                      ("heavy", "success")]
    heavy = [_depth_of(c) for c in Worker.calls if c["task"] == "heavy"]
    assert heavy == [DEPTH, DEPTH]


def test_the_process_worker_runs_under_the_items_depth(meta, monkeypatch, tmp_path):
    item = {"mode": "analysis", "task": "fit A", "label": "A", "depth": dict(DEPTH)}
    spec = placements.item_spec(meta, item, "fit A", tmp_path / "w", "AUTONOMOUS")
    assert spec["depth"] == DEPTH
    json.dumps(spec)                                          # plain data: it crosses a process
    from scilink.agents.meta_agent import workers
    monkeypatch.setattr(workers, "build_child", lambda host, mode, base_dir, **kw: Worker(mode, base_dir))
    out = placements.run_item(spec)
    assert out["result"]["status"] == "success" and _depth_of(Worker.calls[-1]) == DEPTH
    # an item with none: the call is what it was
    Worker.calls.clear()
    placements.run_item(placements.item_spec(meta, {"mode": "analysis", "task": "t", "label": "B"},
                                             "t", tmp_path / "w2", "AUTONOMOUS"))
    assert set(Worker.calls[-1]) == {"mode", "task"}


# ── one rule on all three paths ──────────────────────────────────────────

@pytest.mark.parametrize("given, applied", [
    (DEPTH, DEPTH),
    ({"profile": "quick", "targets": "band position", "time_budget_s": "90"},
     {"profile": "quick", "targets": ["band position"], "time_budget_s": 90.0}),
    ({"profile": None, "targets": [], "time_budget_s": 0}, {}),
])
def test_the_same_depth_reaches_the_child_on_every_path(meta, monkeypatch, tmp_path, given, applied):
    # a single delegation
    meta._get_analysis_child = lambda: Worker("analysis", tmp_path / "d")
    meta._delegate("analysis", "fit D", None, None, "D", depth=dict(given))
    # a fan-out branch
    paths = []
    for n in "AB":
        paths.append(str(tmp_path / f"{n}.npy"))
        np.save(paths[-1], np.zeros((4, 4)))
    monkeypatch.setattr(fo, "_llm_json", lambda orch, prompt, extra_parts=None: {
        "verdict": "complementary", "confidence": 0.9, "rationale": "r", "join_axis": "T",
        "join_type": "shared_parameter_axis", "fanout_set": list(paths), "redundant_clusters": [],
        "unrelated": [], "excluded_notes": ""})
    monkeypatch.setattr(fo, "_make_ephemeral_analysis_child",
                        lambda orch, base_dir, restore=False: Worker("analysis", base_dir))
    meta._run_fanout([{"data_path": p, "task": f"fit {Path(p).stem}", "label": Path(p).stem, **given}
                      for p in paths])
    # a swarm item
    _workers(monkeypatch)
    swarm.run_swarm(meta, [{"mode": "analysis", "task": "fit S", "label": "S", **given},
                           {"mode": "analysis", "task": "fit T", "label": "T", **given}])
    seen = {c["task"]: _depth_of(c) for c in Worker.calls}
    assert set(seen) >= {"fit D", "fit S", "fit T"} and any(t.startswith("Analyze") or "fit A" in t
                                                           for t in seen)
    assert all(d == applied for d in seen.values()), seen


@pytest.mark.parametrize("bad, why", [
    ({"profile": "thorogh"}, "thorogh"),
    ({"targets": [3]}, "targets"),
    ({"time_budget_s": -5}, "time_budget_s"),
])
def test_bad_depth_is_refused_with_its_reason_before_anything_runs(meta, monkeypatch, tmp_path, bad, why):
    meta._get_analysis_child = lambda: Worker("analysis", tmp_path / "d")
    out = json.loads(meta._delegate("analysis", "fit D", None, None, "D", depth=bad))
    assert out["status"] == "error" and why in out["message"] and meta._delegation_ledger == []
    out = json.loads(meta._run_fanout([{"data_path": str(tmp_path), "task": f"t{n}", "label": n, **bad}
                                       for n in "AB"]))
    assert out["status"] == "error" and why in out["message"]
    _workers(monkeypatch)
    res = json.loads(swarm.run_swarm(meta, [{"mode": "analysis", "task": "x", "label": "x", **bad},
                                            {"mode": "planning", "task": "p", "label": "p"},
                                            {"mode": "planning", "task": "q", "label": "q"}]))
    [refused] = res["not_started"]
    assert refused["label"] == "x" and why in refused["reason"]
    assert all(c["mode"] == "planning" for c in Worker.calls)


def test_a_subscription_with_bad_depth_is_refused_when_declared():
    from scilink.agents.meta_agent import reactions
    ok, refused = reactions.normalize_subscriptions([
        {"on": {"kind": "claim"}, "enqueue": {"mode": "analysis", "task": "t", "profile": "fast"}},
        {"on": {"kind": "claim"}, "enqueue": {"mode": "analysis", "task": "t", "profile": "quick"}}])
    assert len(ok) == 1 and ok[0]["enqueue"]["profile"] == "quick"
    assert refused[0]["subscription"] == 1 and "fast" in refused[0]["reason"]


def test_a_rerun_after_a_retraction_starts_from_the_items_depth_and_pattern(meta, monkeypatch, tmp_path):
    from scilink.agents.meta_agent import board as board_mod
    _workers(monkeypatch)
    data = tmp_path / "series"
    data.mkdir()
    claim = meta.board.post(kind="claim", author={"worker": "w", "delegation_index": 90, "mode": "analysis"},
                            subject="s", payload={"text": "a claim"}, status="verified")["finding_id"]
    swarm.run_swarm(meta, [
        {"mode": "analysis", "task": "fit A", "label": "A", "data_path": str(data), "pattern": "a_*.csv",
         "subject": "s", "reads_board": {}, **DEPTH},
        {"mode": "planning", "task": "plan", "label": "P"}])
    entry = next(e for e in meta._delegation_ledger if e["task"].startswith("fit A"))
    assert entry["pattern"] == "a_*.csv" and entry["depth"] == DEPTH and claim in entry["reads"]
    out = board_mod.retract_and_report(meta, claim, "wrong")
    [item] = [i for i in out["rerun_items"] if i["context"]["reruns_delegation"] == entry["index"]]
    assert item["pattern"] == "a_*.csv" and _depth_of(item) == DEPTH
    # and it is an item run_swarm takes as it is
    ok, refused = swarm.normalize_items([item, {"mode": "planning", "task": "p", "label": "p"}])
    assert refused == [] and ok[0]["depth"] == DEPTH


def test_the_swarm_tool_offers_an_items_depth(meta):
    params = next(t for t in meta.tools.openai_schemas
                  if t["function"]["name"] == "run_swarm")["function"]["parameters"]
    item = params["properties"]["work_items"]["items"]["properties"]
    assert item["profile"]["enum"] == ["thorough", "quick", "extract"]
    assert {"targets", "time_budget_s", "pattern"} <= set(item)
