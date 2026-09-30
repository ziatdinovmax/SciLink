"""run_swarm: several delegations of any mode at once, on ephemeral workers,
with a capacity plan, a question queue, a memory guard and per-item budgets.

The workers here are stand-ins for the mode orchestrators (their LLM turn is
what the other suites test); the meta, its ledger and the coordinator are real.
"""

import json
import threading
import time
from pathlib import Path

import pytest

from scilink import hitl, tracing
from scilink.agents.meta_agent import swarm
from scilink.agents.meta_agent.meta_orchestrator import MetaMode, MetaOrchestratorAgent
from scilink.usage import UsageLedger


class FakeWorker:
    """Runs one item: optionally asks a question, optionally until stopped."""

    live = 0
    peak = 0
    lock = threading.Lock()

    def __init__(self, mode, base_dir, behaviour):
        self.mode, self.base_dir, self.behaviour = mode, Path(base_dir), behaviour

    def run_task(self, task, context=None, autonomy=None):
        with FakeWorker.lock:
            FakeWorker.live += 1
            FakeWorker.peak = max(FakeWorker.peak, FakeWorker.live)
        try:
            b = self.behaviour(task)
            answer = None
            if b.get("ask"):
                answer = hitl.request_human_feedback("\nApprove? ", kind="review_plan", default="")
            if b.get("llm"):
                tracing.note_llm_call(prompt_tokens=b["llm"], completion_tokens=1, model="m")
            deadline = time.time() + b.get("seconds", 0.2)
            while time.time() < deadline or b.get("until_stopped"):
                print(".", end="", flush=True)          # where a Stop lands
                time.sleep(0.05)
                if b.get("until_stopped") and time.time() > deadline + 20:
                    break
            out = self.base_dir / "out.txt"
            out.write_text(task)
            return {"status": "success", "summary": f"{self.mode}: {task} ({answer})",
                    "key_findings": [f"finding of {task}"], "files_produced": [str(out)],
                    "suggested_followups": [], "warnings": [], "autonomy": getattr(autonomy, "name", None)}
        finally:
            with FakeWorker.lock:
                FakeWorker.live -= 1


@pytest.fixture(scope="module", autouse=True)
def _mode_modules():
    """A real worker imports its mode's orchestrator while it is built; the
    stand-ins do not, so the first swarm would pay those imports one at a
    time under the import lock. Import them once up front."""
    import scilink.agents.exp_agents.analysis_orchestrator  # noqa: F401
    import scilink.agents.planning_agents.planning_orchestrator  # noqa: F401
    import scilink.agents.sim_agents.simulation_orchestrator  # noqa: F401


@pytest.fixture()
def meta(tmp_path, monkeypatch):
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(swarm, "_POLL_S", 0.05)
    # Admission and the guard read the machine's memory; the tests fix it.
    monkeypatch.setattr(swarm.fo, "_available_memory", lambda: 8e9)
    monkeypatch.setattr(swarm, "_memory", lambda: {"total": 16e9, "available": 8e9})
    FakeWorker.live = FakeWorker.peak = 0
    m = MetaOrchestratorAgent(base_dir=str(tmp_path / "meta"), model_name="anthropic/claude-sonnet-4-5",
                              meta_mode=MetaMode.AUTONOMOUS, launch_dir=str(tmp_path))
    m._enable_human_feedback = False
    return m


def _workers(monkeypatch, behaviour=lambda task: {}):
    built = []

    def build(orch, mode, base_dir, **kw):
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        w = FakeWorker(mode, base_dir, behaviour)
        built.append(w)
        return w
    monkeypatch.setattr(swarm, "build_child", build)
    return built


ITEMS = [{"mode": "analysis", "task": "fit A7", "label": "Raman A7", "subject": "TiO2 A7"},
         {"mode": "simulation", "task": "anatase cell", "label": "anatase cell"},
         {"mode": "planning", "task": "purity check", "label": "purity plan"}]


def test_items_of_every_mode_run_at_once_as_delegations(meta, monkeypatch):
    built = _workers(monkeypatch, lambda task: {"seconds": 0.4})
    res = json.loads(swarm.run_swarm(meta, ITEMS))
    assert res["status"] == "success" and res["not_started"] == []
    assert FakeWorker.peak == 3
    assert [r["mode"] for r in res["results"]] == ["analysis", "simulation", "planning"]
    assert all(r["status"] == "success" and r["files_produced"] == 1 for r in res["results"])
    ledger = meta._delegation_ledger
    assert [e["mode"] for e in ledger] == ["analysis", "simulation", "planning"]
    assert {e["swarm"] for e in ledger} == {res["swarm_id"]}
    assert all(e["parallel_group"] == res["swarm_id"] and not e.get("fanout") for e in ledger)
    assert ledger[0]["subject"] == "TiO2 A7"
    # every worker in its own directory under swarm/
    dirs = sorted(w.base_dir.relative_to(Path(meta.base_dir)).as_posix() for w in built)
    assert dirs == ["swarm/01_raman_a7", "swarm/02_anatase_cell", "swarm/03_purity_plan"]
    # autonomous meta: every worker autonomous; nothing private left on the entries
    assert all(not any(k in e for k in ("_item",)) for e in ledger)


def test_the_checkpoint_holds_the_swarm(meta, monkeypatch):
    _workers(monkeypatch)
    swarm.run_swarm(meta, ITEMS[:2])
    ck = json.loads((Path(meta.base_dir) / "checkpoint.json").read_text())
    assert [e.get("swarm") is not None for e in ck["delegation_ledger"]] == [True, True]


def test_bad_items_are_refused_with_reasons(meta, monkeypatch):
    _workers(monkeypatch)
    res = json.loads(swarm.run_swarm(meta, [{"mode": "optimize", "task": "x", "label": "a"},
                                            {"mode": "analysis", "task": " ", "label": "b"},
                                            {"mode": "planning", "task": "ok", "label": "c"},
                                            {"mode": "planning", "task": "ok too", "label": "d"}]))
    assert [r["label"] for r in res["results"]] == ["c", "d"]
    reasons = {r["label"]: r["reason"] for r in res["not_started"]}
    assert "mode must be one of" in reasons["a"] and reasons["b"] == "empty task"


def test_one_item_is_not_a_swarm(meta, monkeypatch):
    built = _workers(monkeypatch)
    res = json.loads(swarm.run_swarm(meta, [{"mode": "planning", "task": "t", "label": "only"}]))
    assert res["status"] == "error" and "at least two" in res["message"] and built == []


def test_more_items_than_the_limit_are_not_started(meta, monkeypatch):
    _workers(monkeypatch)
    items = [{"mode": "planning", "task": f"t{i}", "label": f"p{i}"} for i in range(swarm.SWARM_MAX_ITEMS + 2)]
    res = json.loads(swarm.run_swarm(meta, items))
    assert len(res["results"]) == swarm.SWARM_MAX_ITEMS
    assert len(res["not_started"]) == 2 and "limit" in res["not_started"][0]["reason"]


def test_an_item_larger_than_the_machine_is_not_started(meta, monkeypatch, tmp_path):
    _workers(monkeypatch)
    big = tmp_path / "cube.npy"
    big.write_bytes(b"\0" * 2_000_000)                   # x6 = 12 MB estimated
    monkeypatch.setattr(swarm, "_memory", lambda: {"total": 1.51e9 + 5e6, "available": 1e9})
    monkeypatch.setitem(swarm._MODE_MEM_FLOOR, "planning", 1e6)
    res = json.loads(swarm.run_swarm(meta, [
        {"mode": "analysis", "task": "fit", "label": "big cube", "data_path": str(big)},
        {"mode": "planning", "task": "plan", "label": "small plan"}]))
    assert [r["label"] for r in res["results"]] == ["small plan"]
    assert "this machine has" in res["not_started"][0]["reason"]


def test_the_capacity_plan_says_whether_items_fit_together():
    items = [{"mode": "planning", "label": "a"}, {"mode": "simulation", "label": "b"}]
    fits = swarm.capacity_plan([dict(i) for i in items], {"total": 16e9, "available": 8e9})
    tight = swarm.capacity_plan([dict(i) for i in items], {"total": 16e9, "available": 2e9})
    assert fits["together"] and not tight["together"]
    assert "wait for others" in swarm._memory_line(tight)


# ------------------------------------------------------------------ autopilot

class Person:
    def __init__(self, confirm="y"):
        self.confirm, self.asked = confirm, []

    def ask(self, req):
        self.asked.append(req)
        if req.kind == "confirm":
            return self.confirm
        return "ok " + req.origin.get("branch_label", "?")


@pytest.fixture()
def attended(meta, monkeypatch):
    meta._enable_human_feedback = True
    meta.meta_mode = MetaMode.AUTOPILOT
    person = Person()
    hitl.set_default_channel(person)
    yield person
    hitl.set_default_channel(None)


def test_a_declined_swarm_runs_nothing(meta, attended, monkeypatch):
    built = _workers(monkeypatch)
    attended.confirm = "n"
    res = json.loads(swarm.run_swarm(meta, ITEMS))
    assert res["status"] == "declined" and built == [] and meta._delegation_ledger == []
    (gate,) = attended.asked
    assert gate.subject["title"] == "Launch this swarm?"
    assert any(b.get("label", "").startswith("🐝 Items") for b in gate.subject["blocks"])


def test_each_workers_question_reaches_the_person_and_comes_back_to_it(meta, attended, monkeypatch):
    _workers(monkeypatch, lambda task: {"ask": True, "seconds": 0.1})
    res = json.loads(swarm.run_swarm(meta, ITEMS[:2]))
    summaries = {r["label"]: r["summary"] for r in res["results"]}
    assert summaries == {"Raman A7": "analysis: fit A7 (ok Raman A7)",
                         "anatase cell": "simulation: anatase cell (ok anatase cell)"}
    heads = sorted(r.prompt.split("]")[0] for r in attended.asked if r.kind != "confirm")
    assert heads == ["\n[worker: Raman A7 · TiO2 A7", "\n[worker: anatase cell"]


# ------------------------------------------------------------------ guards

def test_memory_pressure_cancels_the_largest_item_and_runs_it_again_alone(meta, monkeypatch):
    runs = {"n": 0}
    monkeypatch.setitem(swarm._MODE_MEM_FLOOR, "analysis", 2e9)     # the heavy one

    def behaviour(task):
        if task == "heavy":
            runs["n"] += 1
            return {"until_stopped": runs["n"] == 1, "seconds": 0.3}
        return {"seconds": 1.0}
    _workers(monkeypatch, behaviour)
    low = {"on": False}
    monkeypatch.setattr(swarm, "_memory", lambda: {"total": 16e9, "available": 1e8 if low["on"] else 8e9})
    threading.Timer(0.4, lambda: low.update(on=True)).start()
    real_guard = swarm._guard_memory

    def guard(*args):
        cancelled = real_guard(*args)
        if cancelled is not None:
            low["on"] = False                # cancelling the heavy item freed the memory
        return cancelled
    monkeypatch.setattr(swarm, "_guard_memory", guard)
    res = json.loads(swarm.run_swarm(meta, [
        {"mode": "planning", "task": "light", "label": "light"},
        {"mode": "analysis", "task": "heavy", "label": "heavy"}]))
    by = [(r["label"], r["status"]) for r in res["results"]]
    assert by == [("light", "success"), ("heavy", "cancelled"), ("heavy", "success")]
    assert "memory_pressure" in res["results"][1]["error"]
    assert runs["n"] == 2


def test_an_item_over_its_budget_is_cancelled(meta, monkeypatch, capsys):
    _workers(monkeypatch, lambda task: {"until_stopped": task == "slow", "seconds": 0.1})
    res = json.loads(swarm.run_swarm(meta, [
        {"mode": "planning", "task": "slow", "label": "slow one"},
        {"mode": "planning", "task": "fast", "label": "fast one"}], item_time_budget_s=0.5))
    status = {r["label"]: r["status"] for r in res["results"]}
    assert status == {"slow one": "error", "fast one": "success"}
    assert "swarm item 'slow one' exceeded its wall-clock budget" in capsys.readouterr().out


def test_usage_is_charged_to_each_item_under_the_coordinators_session(meta, monkeypatch, tmp_path):
    _workers(monkeypatch, lambda task: {"llm": 100})
    led = UsageLedger(tmp_path / "usage.jsonl")
    tracing.set_usage_sink(led.record)
    tracing.bind_session("meta-session")
    try:
        swarm.run_swarm(meta, ITEMS[:2])
    finally:
        tracing.set_usage_sink(None)
        tracing.bind_session(None)
    s = led.summary()
    assert set(s["by_worker"]) == {"meta-session/swarm:01_raman_a7", "meta-session/swarm:02_anatase_cell"}
    assert s["by_session"] == {"meta-session": {"calls": 2, "prompt_tokens": 200, "completion_tokens": 2}}


def test_the_meta_offers_run_swarm(meta):
    names = [t["function"]["name"] for t in meta.tools.openai_schemas]
    assert "run_swarm" in names
    assert "FRESH agent" in meta.messages[0]["content"]


def test_an_item_running_alone_is_never_cancelled_by_the_guard(meta, monkeypatch, capsys):
    """The guard is against several items overcommitting together. Once one
    item is alone it runs on, even with memory still low, and a cancelled
    item's rerun waits for the cancelled worker to end before it starts."""
    monkeypatch.setitem(swarm._MODE_MEM_FLOOR, "analysis", 2e9)
    runs = {"heavy": 0}

    def behaviour(task):
        if task == "heavy":
            runs["heavy"] += 1
            return {"until_stopped": runs["heavy"] == 1, "seconds": 0.3}
        return {"seconds": 0.6}
    _workers(monkeypatch, behaviour)
    monkeypatch.setattr(swarm, "_memory", lambda: {"total": 16e9, "available": 1e8})   # low throughout
    res = json.loads(swarm.run_swarm(meta, [
        {"mode": "planning", "task": "light", "label": "light"},
        {"mode": "analysis", "task": "heavy", "label": "heavy"}]))
    assert [(r["label"], r["status"]) for r in res["results"]] == [
        ("light", "success"), ("heavy", "cancelled"), ("heavy", "success")]
    assert capsys.readouterr().out.count("cancelling '") == 1


def test_budgets_keep_firing_while_a_question_is_on_screen(meta, attended, monkeypatch):
    """The person takes 2 s over one worker's question; another item with a
    0.5 s budget must be cancelled meanwhile, not after the answer."""
    marks = {}
    real_ask = attended.ask

    def slow_ask(req):
        if req.kind != "confirm":
            time.sleep(2.0)
            marks["answered_at"] = time.time()
        return real_ask(req)
    attended.ask = slow_ask
    real_cancel = swarm.fo._cancel_overdue_branches

    def cancel(*args, **kwargs):
        before = {id(e) for e in args[2].values() if e.get("timed_out")}
        real_cancel(*args, **kwargs)
        if any(e.get("timed_out") and id(e) not in before for e in args[2].values()):
            marks.setdefault("cancelled_at", time.time())
    monkeypatch.setattr(swarm.fo, "_cancel_overdue_branches", cancel)
    _workers(monkeypatch, lambda task: {"ask": task == "asks", "until_stopped": task == "slow",
                                        "seconds": 0.2})
    res = json.loads(swarm.run_swarm(meta, [
        {"mode": "planning", "task": "asks", "label": "asker"},
        {"mode": "planning", "task": "slow", "label": "slow one"}], item_time_budget_s=0.5))
    status = {r["label"]: r["status"] for r in res["results"]}
    assert status == {"asker": "success", "slow one": "error"}
    assert marks["cancelled_at"] < marks["answered_at"]


def test_an_unanswered_worker_question_is_a_warning_on_its_result(meta, attended, monkeypatch):
    class Nobody:
        def ask(self, req):
            return "y" if req.kind == "confirm" else (_ for _ in ()).throw(AssertionError("asked"))
    monkeypatch.setattr(swarm, "question_timeout_s", lambda: 0.3)
    hitl.set_default_channel(Nobody())
    _workers(monkeypatch, lambda task: {"ask": True, "seconds": 0.1})
    res = json.loads(swarm.run_swarm(meta, ITEMS[:2]))
    entry = meta._delegation_ledger[0]
    assert any("unattended" in w for w in entry.get("warnings") or []), entry.get("warnings")


def test_a_stop_on_the_persons_channel_stops_the_swarm(meta, attended, monkeypatch):
    from scilink.ui.output_capture import AgentStoppedError

    class StopsMidway:
        def ask(self, req):
            if req.kind == "confirm":
                return "y"
            raise AgentStoppedError("Agent stopped by user")
    hitl.set_default_channel(StopsMidway())
    _workers(monkeypatch, lambda task: {"ask": True, "seconds": 0.1})
    with pytest.raises(AgentStoppedError):
        swarm.run_swarm(meta, ITEMS[:2])
    deadline = time.time() + 5                      # the workers wind down and let go
    while swarm.fo._mem_running and time.time() < deadline:
        time.sleep(0.05)
    assert not swarm.fo._mem_running


def test_a_stop_during_cleanup_still_releases_the_memory_reservation(meta, monkeypatch):
    from scilink.ui.output_capture import AgentStoppedError

    class Stubborn(FakeWorker):
        _mcp_connections = {"srv": object()}

        def disconnect_mcp_server(self, name):
            raise AgentStoppedError("Agent stopped by user")   # a log line raising on Stop
    built = []

    def build(orch, mode, base_dir, **kw):
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        w = Stubborn(mode, base_dir, lambda task: {})
        built.append(w)
        return w
    monkeypatch.setattr(swarm, "build_child", build)
    res = json.loads(swarm.run_swarm(meta, ITEMS[:2]))
    assert all(r["status"] == "success" for r in res["results"])
    assert not swarm.fo._mem_running


def test_a_rerun_is_given_up_when_the_cancelled_worker_never_ends(meta, monkeypatch, capsys):
    monkeypatch.setitem(swarm._MODE_MEM_FLOOR, "analysis", 2e9)
    monkeypatch.setattr(swarm, "SWARM_DRAIN_TIMEOUT_S", 0.6)

    def behaviour(task):
        if task == "hung":
            return {"hung": True}
        return {"seconds": 0.3}
    real_run = FakeWorker.run_task

    def run_task(self, task, context=None, autonomy=None):
        if self.behaviour(task).get("hung"):
            time.sleep(3.0)                     # never prints, never notices the cancel
            return {"status": "success", "summary": "late", "key_findings": [],
                    "files_produced": [], "suggested_followups": [], "warnings": []}
        return real_run(self, task, context, autonomy)
    monkeypatch.setattr(FakeWorker, "run_task", run_task)
    _workers(monkeypatch, behaviour)
    low = {"on": False}
    monkeypatch.setattr(swarm, "_memory", lambda: {"total": 16e9, "available": 1e8 if low["on"] else 8e9})
    threading.Timer(0.15, lambda: low.update(on=True)).start()
    t0 = time.time()
    res = json.loads(swarm.run_swarm(meta, [
        {"mode": "planning", "task": "light", "label": "light"},
        {"mode": "analysis", "task": "hung", "label": "hung"}]))
    assert time.time() - t0 < 2.5                               # did not wait the worker out
    assert [(r["label"], r["status"]) for r in res["results"]] == [("light", "success"), ("hung", "cancelled")]
    assert res["not_started"] and "rerun abandoned" in res["not_started"][0]["reason"]
    assert "giving up the rerun" in capsys.readouterr().out
