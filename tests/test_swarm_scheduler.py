"""Swarm stage 4, the local scheduler: the worker contract and its process
placement, the measured table per item class, the token budget with
reservation, the provider circuit breaker, the guard that reads what a
process worker holds, the cluster executor's default cancel, and the
fan-out's share of the refusal and the guard.

The process placement is exercised with a REAL child process on a stand-in
target in this module (``child_target``, importable in the child because
pytest puts this directory on ``sys.path``, which the child inherits); the
child's side of the real target (``placements.run_item``) is exercised
in-process with the worker constructor stubbed, as ``tests/test_run_swarm.py``
stubs it. A live swarm closes the loop on a real analysis item.
"""

import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from scilink import tracing
from scilink.agents.meta_agent import fanout as fo
from scilink.agents.meta_agent import peaks, placements, swarm
from scilink.agents.meta_agent.meta_orchestrator import MetaMode, MetaOrchestratorAgent
from scilink.agents.meta_agent.placements import LocalProcess, WorkerHandle
from scilink.ui.output_capture import AgentStoppedError
from scilink.utils.log_context import is_cancelled, register_cancel, unregister_cancel
from scilink.wrappers import llm_limiter

TARGET = f"{__name__}:child_target"


# ------------------------------------------------------------ the stand-in child

USAGE = {"m": {"calls": 2, "prompt_tokens": 100, "completion_tokens": 10, "seconds": 0.5}}


def child_target(spec):
    """What the child runs instead of ``run_item``: behaviour read from the
    task text. Returns the shape ``run_item`` returns."""
    import json as _json
    import subprocess
    task = str(spec.get("task") or "")
    base = Path(spec.get("base_dir") or ".")
    base.mkdir(parents=True, exist_ok=True)
    (base / placements.WORKER_USAGE).write_text(_json.dumps(USAGE))     # as the real sink keeps it
    marker = os.environ.get("SCILINK_TEST_KILL_ONCE_MARKER")
    if "kill once" in task and marker and not Path(marker).exists():
        Path(marker).write_text("killed")
        os.kill(os.getpid(), 9)
    if "kill" in task and "once" not in task:
        os.kill(os.getpid(), 9)
    if "raise" in task:
        raise RuntimeError("the target failed")
    if "grandchild" in task:
        # a grandchild in its own session (as a generated script runs), and
        # the worker busy in one long C call, where no SIGTERM handler runs
        import hashlib
        g = subprocess.Popen(["sleep", "300"], start_new_session=True)
        (base / "grandchild.pid").write_text(str(g.pid))
        hashlib.pbkdf2_hmac("sha256", b"x", b"y", 300_000_000)
    for i in range(3):
        print(f"worker line {i}", flush=True)
    held = None
    if "hold" in task:
        held = bytearray(300 * (1 << 20))          # 300 MB the sampler must see
        held[::4096] = b"x" * len(held[::4096])     # touched: resident, not reserved
    seconds = 6.0 if "slow" in task else 0.2
    end = time.time() + seconds
    while time.time() < end:
        time.sleep(0.05)
    del held
    return {"result": {"status": "success", "summary": f"did {task}", "key_findings": [f"finding of {task}"],
                       "files_produced": [], "suggested_followups": [], "warnings": []},
            "usage": {"m": {"calls": 2, "prompt_tokens": 100, "completion_tokens": 10, "seconds": 0.5}},
            "unattended": 1}


def _spec(task, **extra):
    import tempfile
    return {"mode": "analysis", "task": task, "label": task, "base_dir": tempfile.mkdtemp(prefix="swarm-item-"),
            "autonomy": "AUTONOMOUS", "host": {}, **extra}


# ------------------------------------------------------- the worker contract

def test_a_process_worker_returns_its_result_usage_and_measured_peak():
    pl = LocalProcess(target=TARGET)
    h = pl.submit(_spec("plain"))
    assert pl.poll(h)["state"] in ("queued", "running")
    final = pl.wait(h, poll_s=0.05)
    assert final["state"] == "done" and h.result["summary"] == "did plain"
    assert h.usage == {"m": {"calls": 2, "prompt_tokens": 100, "completion_tokens": 10, "seconds": 0.5}}
    assert h.unattended == 1
    assert final["peak_rss_bytes"] and final["peak_rss_bytes"] > 10e6      # a fresh interpreter


def test_the_sampler_sees_what_the_child_holds():
    pl = LocalProcess(target=TARGET)
    h = pl.submit(_spec("hold slow"))
    pl.wait(h, poll_s=0.05)
    assert h.state == "done"
    assert h.peak_rss_bytes > 250e6, h.peak_rss_bytes


def test_a_killed_worker_nobody_cancelled_is_out_of_memory_and_a_failed_target_is_failed():
    pl = LocalProcess(target=TARGET)
    killed, failed = pl.submit(_spec("kill")), pl.submit(_spec("raise"))
    assert pl.wait(killed, poll_s=0.05)["state"] == "out_of_memory"
    assert "SIGKILL" in killed.stop_reason
    assert killed.usage == USAGE                 # what it spent before the kill, from its file
    assert pl.wait(failed, poll_s=0.05)["state"] == "failed"
    assert "the target failed" in failed.stop_reason


def test_cancel_ends_the_child_and_is_idempotent():
    pl = LocalProcess(target=TARGET)
    h = pl.submit(_spec("slow"))
    deadline = time.time() + 10
    while h.thread_id is None and time.time() < deadline:
        time.sleep(0.05)
    time.sleep(1.5)                       # the child is up and sleeping
    t0 = time.time()
    pl.cancel(h)
    pl.cancel(h)
    final = pl.wait(h, poll_s=0.05)
    assert final["state"] == "cancelled" and time.time() - t0 < 5.0
    assert h.stop_reason == "cancelled"           # the cancel's reason, not the signal's


def test_the_waiting_threads_own_cancel_cancels_the_worker():
    """The item thread waits; its cancel (a budget, the guard, a Stop) is
    what ``wait`` polls — no one has to call ``cancel`` from outside."""
    pl = LocalProcess(target=TARGET)
    ev = threading.Event()
    out = {}

    def item_thread():
        register_cancel(ev)
        try:
            h = pl.submit(_spec("slow"))
            out["final"] = pl.wait(h, poll_s=0.05)
        finally:
            unregister_cancel()
    t = threading.Thread(target=item_thread)
    t.start()
    time.sleep(2.0)
    ev.set()
    t.join(10)
    assert out["final"]["state"] == "cancelled"


def test_the_handle_states_are_the_contracts():
    assert placements.STATES == ("queued", "running", "done", "failed", "cancelled",
                                 "out_of_memory", "interrupted")
    assert WorkerHandle({}, "process").snapshot() == {"state": "queued", "result": None,
                                                      "peak_rss_bytes": None, "stop_reason": None}


# ------------------------------------------------------- the child's own side

def test_run_item_builds_the_child_from_the_host_spec_and_reports_usage_and_defaults(tmp_path, monkeypatch):
    from scilink import executors, hitl
    from scilink.agents.meta_agent import workers
    seen = {}

    class Child:
        def run_task(self, task, context=None, autonomy=None):
            seen["autonomy"] = autonomy.name
            tracing.note_llm_call(prompt_tokens=7, completion_tokens=3, model="m1")
            tracing.note_llm_call(prompt_tokens=1, completion_tokens=1, model="m2")
            seen["answer"] = hitl.request_human_feedback("ok? ", kind="confirm", default="n")
            return {"status": "success", "summary": task, "key_findings": [], "files_produced": [Path(".")]}

    def build(host, mode, base_dir, label=None):
        seen["host"] = host
        seen["mode"], seen["base_dir"], seen["label"] = mode, base_dir, label
        host._propagate_extensions_to_child(Child())
        return Child()
    monkeypatch.setattr(workers, "build_child", build)
    monkeypatch.setattr(workers, "release_child", lambda c: seen.setdefault("released", True))
    monkeypatch.setattr(executors, "_GLOBAL_SANDBOX_APPROVED", False)
    monkeypatch.delenv("UNSAFE_EXECUTION_OK", raising=False)
    spec = {"mode": "analysis", "task": "fit", "context": {"a": 1}, "label": "L", "base_dir": str(tmp_path / "w"),
            "autonomy": "AUTONOMOUS", "sandbox_approved": True,
            "host": {"model_name": "anthropic/x", "base_url": None, "file_roots": [str(tmp_path)],
                     "extensions": [{"kind": "skill", "skill_path": "/no/such/skill.md"}]}}
    out = placements.run_item(spec)
    assert out["result"]["status"] == "success" and out["result"]["files_produced"] == ["."]   # plain data
    assert out["usage"] == {"m1": {"calls": 1, "prompt_tokens": 7, "completion_tokens": 3, "seconds": 0.0},
                            "m2": {"calls": 1, "prompt_tokens": 1, "completion_tokens": 1, "seconds": 0.0}}
    assert out["unattended"] == 1 and seen["answer"] == "n"
    assert seen["autonomy"] == "AUTONOMOUS" and seen["mode"] == "analysis" and seen["label"] == "Swarm: L"
    assert seen["host"].model_name == "anthropic/x" and seen["host"].api_key is None
    assert [str(r) for r in seen["host"].path_fence.roots] == [str(tmp_path)]
    assert executors._GLOBAL_SANDBOX_APPROVED and os.environ["UNSAFE_EXECUTION_OK"] == "true"
    assert seen["released"] and tracing._usage_sink is None


def test_the_placement_rule(monkeypatch):
    monkeypatch.delenv(placements.PLACEMENT_ENV, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")

    class Orch:
        api_key = "k"
        _shared_extensions = []
    heavy = {"mode": "analysis", "data_path": "/d"}
    assert placements.placement_for(Orch(), heavy, attended=False)[0] == "process"
    assert placements.placement_for(Orch(), heavy, attended=True) == (
        "thread", "the swarm is attended: its questions need the person's channel")
    assert placements.placement_for(Orch(), {"mode": "planning"}, attended=False)[0] == "thread"
    assert placements.placement_for(Orch(), {"mode": "analysis"}, attended=False)[0] == "thread"
    tools = Orch()
    tools._shared_extensions = [{"kind": "tools", "schemas": [], "factory": lambda: None}]
    assert "custom tools" in placements.placement_for(tools, heavy, attended=False)[1]
    foreign = Orch()
    foreign.api_key = "not-in-the-environment"
    assert "not in the environment" in placements.placement_for(foreign, heavy, attended=False)[1]
    monkeypatch.setenv(placements.PLACEMENT_ENV, "thread")
    assert placements.placement_for(Orch(), heavy, attended=False) == ("thread", "SCILINK_SWARM_PLACEMENT=thread")
    monkeypatch.setenv(placements.PLACEMENT_ENV, "process")
    assert placements.placement_for(Orch(), heavy, attended=True)[0] == "process"


def test_the_host_spec_carries_no_secret_and_only_inheritable_extensions():
    class Fence:
        roots = [Path("/a"), Path("/b")]

    class Orch:
        api_key = "SECRET"
        model_name, base_url, embedding_model, embedding_base_url = "m", "http://p", "e", None
        knowledge_dir = "/kb"
        path_fence = Fence()
        futurehouse_api_key = "FH"
        _shared_extensions = [{"kind": "skill", "skill_path": Path("/s.md")},
                              {"kind": "tools", "schemas": [], "factory": print},
                              {"kind": "mcp", "server_name": "srv", "url": "http://x", "headers": {"h": 1}}]
    spec = placements.host_spec(Orch())
    assert "SECRET" not in json.dumps(spec) and "FH" not in json.dumps(spec)
    assert spec["file_roots"] == ["/a", "/b"] and spec["knowledge_dir"] == "/kb"
    assert [e["kind"] for e in spec["extensions"]] == ["skill", "mcp"]
    assert spec["extensions"][1] == {"kind": "mcp", "server_name": "srv", "url": "http://x", "headers": {"h": 1}}


# ------------------------------------------------------------ the measured table

def test_the_item_class_is_computed_from_the_spec_alone(tmp_path):
    np.save(tmp_path / "a.npy", np.zeros((300, 300)))           # 720 kB
    np.save(tmp_path / "b.npy", np.zeros((1000, 1000)))         # 8 MB
    (tmp_path / "c.csv").write_text("1,2\n")
    assert peaks.item_class({"mode": "analysis", "data_path": str(tmp_path / "a.npy")}) == "analysis:array:1MB:1"
    assert peaks.item_class({"mode": "analysis", "data_path": str(tmp_path)}) == "analysis:array:8MB:few"
    assert peaks.item_class({"mode": "analysis", "data_path": str(tmp_path / "c.csv")}) == "analysis:curve:1MB:1"
    assert peaks.item_class({"mode": "planning"}) == "planning"
    assert peaks.item_class({"mode": "analysis", "data_path": str(tmp_path / "missing")}) == "analysis:nodata"


def test_the_table_keeps_the_max_per_class_and_sizes_with_headroom(tmp_path):
    path = tmp_path / "measured_items.json"
    assert peaks.measured("analysis:array:8MB:1", path) is None
    assert peaks.record("analysis:array:8MB:1", peak_rss_bytes=2e9, tokens=1000, path=path)["runs"] == 1
    row = peaks.record("analysis:array:8MB:1", peak_rss_bytes=1e9, tokens=3000, path=path)
    assert row["peak_rss_bytes"] == 2e9 and row["tokens"] == 3000 and row["runs"] == 2
    assert peaks.peak_estimate("analysis:array:8MB:1", path) == 2e9 * peaks.PEAK_HEADROOM
    assert peaks.token_estimate("analysis:array:8MB:1", path) == 3000
    assert peaks.record("x", path=path) == {}                   # nothing to record: a no-op
    assert peaks.measured("x", path) is None
    assert peaks.record("planning", tokens=50, path=path)["peak_rss_bytes"] is None


def test_estimate_item_prefers_the_measured_peak(tmp_path, monkeypatch):
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    np.save(tmp_path / "a.npy", np.zeros((1000, 1000)))
    item = {"mode": "analysis", "data_path": str(tmp_path / "a.npy")}
    est = fo.estimate_item(item)
    assert est == fo._branch_mem_estimate(item) and not item["_mem_measured"]
    peaks.record(item["_mem_class"], peak_rss_bytes=3e9)
    assert fo.estimate_item(item) == 3e9 * peaks.PEAK_HEADROOM and item["_mem_measured"]
    assert fo.estimate_item({"mode": "planning"}) == fo.MODE_MEM_FLOOR["planning"]


# --------------------------------------------------------------- the breaker

def test_the_breaker_trips_on_a_failing_model_and_a_success_does_not_close_it(monkeypatch):
    monkeypatch.setattr(llm_limiter, "BREAKER_HOLD_S", 5.0)
    for _ in range(llm_limiter.BREAKER_FAILURES - 1):
        llm_limiter.note_provider_failure("m")
    assert llm_limiter.provider_tripped("m") is None
    llm_limiter.note_provider_failure("m")
    assert 0 < llm_limiter.provider_tripped("m") <= 5.0
    assert llm_limiter.provider_tripped() is not None and llm_limiter.provider_tripped("other") is None
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


# -------------------------------------------------------- the cluster's cancel

def test_the_cluster_executor_cancels_its_job_on_the_threads_own_cancel(tmp_path):
    from test_cluster_executor import FakeConn, FakeScheduler, _RC
    from scilink.agents.sim_agents.cluster_executor import ClusterExecutor
    conn = FakeConn()
    sched = FakeScheduler(conn, terminal_after=9999)
    ex = ClusterExecutor(conn, scheduler=sched, poll_interval=0.05, timeout=10_000)
    assert ex.cancel_check is is_cancelled
    ev = threading.Event()
    out = {}

    def item():
        register_cancel(ev)
        try:
            out["r"] = ex.run({"in.lj": "S"}, "lmp -in in.lj", str(tmp_path / "m"))
        except AgentStoppedError as exc:
            out["stop"] = exc
        finally:
            unregister_cancel()
    t = threading.Thread(target=item)
    t.start()
    time.sleep(0.2)
    ev.set()
    t.join(10)
    # the job is cancelled AND the caller is ended (so a campaign loop does
    # not critique and resubmit); a caller's own hook keeps the result
    assert "r" not in out and isinstance(out["stop"], AgentStoppedError)
    assert sched.cancelled == ["12345"] and (tmp_path / "m" / _RC).read_text() == "cancelled"
    # with the cancel already set, nothing is submitted at all
    out.clear()
    t = threading.Thread(target=item)
    t.start()
    t.join(10)
    assert isinstance(out["stop"], AgentStoppedError) and sched.cancelled == ["12345"]


def test_is_cancelled_reads_the_threads_cancel():
    assert not is_cancelled()
    ev = threading.Event()
    register_cancel(ev)
    try:
        assert not is_cancelled()
        ev.set()
        assert is_cancelled()
    finally:
        unregister_cancel()


# ------------------------------------------------------------ the swarm itself

@pytest.fixture()
def meta(tmp_path, monkeypatch):
    import scilink.agents.exp_agents.analysis_orchestrator  # noqa: F401 - the worker-side import
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv(placements.PLACEMENT_ENV, raising=False)
    monkeypatch.setattr(swarm, "_POLL_S", 0.05)
    monkeypatch.setattr(swarm.fo, "_available_memory", lambda: 8e9)
    monkeypatch.setattr(swarm, "_memory", lambda: {"total": 16e9, "available": 8e9})
    monkeypatch.setattr(swarm, "LocalProcess", lambda: LocalProcess(target=TARGET))
    m = MetaOrchestratorAgent(base_dir=str(tmp_path / "meta"), model_name="anthropic/claude-sonnet-4-5",
                              meta_mode=MetaMode.AUTONOMOUS, launch_dir=str(tmp_path))
    m._enable_human_feedback = False
    return m


def _cube(tmp_path, name="cube.npy", shape=(100, 100)):
    p = tmp_path / name
    np.save(p, np.zeros(shape))
    return str(p)


def _threads(monkeypatch):
    """The thread placement's stand-in workers (as tests/test_run_swarm.py)."""
    from test_run_swarm import FakeWorker

    def build(orch, mode, base_dir, **kw):
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        return FakeWorker(mode, base_dir, lambda task: {"llm": 100})
    monkeypatch.setattr(swarm, "build_child", build)


def test_an_analysis_item_with_data_runs_in_its_own_process_and_is_measured(meta, monkeypatch, tmp_path):
    _threads(monkeypatch)
    cube = _cube(tmp_path)
    res = json.loads(swarm.run_swarm(meta, [
        {"mode": "analysis", "task": "plain", "label": "the cube", "data_path": cube},
        {"mode": "planning", "task": "plan", "label": "the plan"}]))
    by = {r["label"]: r for r in res["results"]}
    assert by["the cube"]["status"] == "success" and by["the cube"]["placement"] == "process"
    assert by["the plan"]["status"] == "success" and by["the plan"]["placement"] == "thread"
    assert by["the cube"]["peak_rss_bytes"] > 10e6 and by["the plan"]["peak_rss_bytes"] is None
    # the child's usage is charged to the item's worker tag, and the
    # unanswered question is the same warning a thread item gets
    assert by["the cube"]["tokens"] == 110 and by["the plan"]["tokens"] == 101
    assert any("took their defaults in the worker process" in w for w in by["the cube"]["warnings"])
    assert res["tokens"] == {"max_tokens": None, "spent": 211}
    entry = next(e for e in meta._delegation_ledger if e.get("label") == "the cube")
    assert entry["placement_reason"] == "an analysis item with data, nobody attending"
    assert entry["worker_state"] == "done"
    # the class was measured: memory from the process, tokens from both
    cls = peaks.item_class({"mode": "analysis", "data_path": cube})
    row = peaks.measured(cls)
    assert row["runs"] == 1 and row["peak_rss_bytes"] == by["the cube"]["peak_rss_bytes"] and row["tokens"] == 110
    row_p = peaks.measured("planning")
    assert row_p["tokens"] == 101 and row_p["peak_rss_bytes"] is None
    # ... and the next plan of the class is sized from it
    plan = swarm.capacity_plan([{"mode": "analysis", "task": "again", "label": "again", "data_path": cube, "slug": "again"}],
                               {"total": 16e9, "available": 8e9}, orch=meta)
    it = plan["run"][0]
    assert it["_mem_measured"] and it["_mem_est"] == row["peak_rss_bytes"] * peaks.PEAK_HEADROOM
    assert "measured" in swarm._mem_tag(it) and "own process" in json.dumps(swarm.swarm_plan_subject(plan, False))


def test_a_worker_killed_by_the_system_runs_again_alone_once(meta, monkeypatch, tmp_path):
    _threads(monkeypatch)
    monkeypatch.setenv("SCILINK_TEST_KILL_ONCE_MARKER", str(tmp_path / "killed.marker"))
    res = json.loads(swarm.run_swarm(meta, [
        {"mode": "analysis", "task": "kill once", "label": "heavy", "data_path": _cube(tmp_path)},
        {"mode": "planning", "task": "plan", "label": "light"}]))
    by = [(r["label"], r["status"]) for r in res["results"]]
    assert by == [("heavy", "error"), ("light", "success"), ("heavy", "success")]
    assert "out_of_memory" in res["results"][0]["error"] and "SIGKILL" in res["results"][0]["error"]
    # the rerun is settled on its own: the killed attempt spent nothing
    assert [r["tokens"] for r in res["results"]] == [110, 101, 110] and res["tokens"]["spent"] == 321
    first = next(e for e in meta._delegation_ledger if e.get("label") == "heavy")
    assert first["worker_state"] == "out_of_memory" and first.get("out_of_memory")


def test_the_guard_cancels_the_process_that_holds_the_most_not_the_largest_estimate(meta, monkeypatch, tmp_path):
    """Two process items of the same class: the estimates tie, but one holds
    300 MB. Under pressure the guard reads what each worker was sampled at
    and cancels that one — then reruns it alone."""
    _threads(monkeypatch)
    low = {"on": False}
    monkeypatch.setattr(swarm, "_memory", lambda: {"total": 16e9, "available": 1e8 if low["on"] else 8e9})
    real_guard = swarm._guard_memory

    def guard(*args):
        cancelled = real_guard(*args)
        if cancelled is not None:
            low["on"] = False
        return cancelled
    monkeypatch.setattr(swarm, "_guard_memory", guard)
    threading.Timer(3.5, lambda: low.update(on=True)).start()
    runs = {"hold": 0}
    orig = placements.LocalProcess.submit

    def submit(self, spec):
        if "hold" in spec["task"]:
            runs["hold"] += 1
            if runs["hold"] == 2:
                spec = {**spec, "task": "plain rerun"}      # the rerun need not hold
        return orig(self, spec)
    monkeypatch.setattr(placements.LocalProcess, "submit", submit)
    res = json.loads(swarm.run_swarm(meta, [
        {"mode": "analysis", "task": "slow", "label": "lean", "data_path": _cube(tmp_path, "a.npy")},
        {"mode": "analysis", "task": "hold slow", "label": "holder", "data_path": _cube(tmp_path, "b.npy")}]))
    by = [(r["label"], r["status"]) for r in res["results"]]
    assert by == [("lean", "success"), ("holder", "cancelled"), ("holder", "success")], res["results"]
    assert "memory_pressure" in res["results"][1]["error"] and "this item held" in res["results"][1]["error"]


def test_the_token_budget_refuses_what_the_remainder_cannot_cover_and_settles_on_what_was_spent(meta, monkeypatch):
    _threads(monkeypatch)
    items = [{"mode": "planning", "task": "a", "label": "a"}, {"mode": "planning", "task": "b", "label": "b"},
             {"mode": "simulation", "task": "c", "label": "c"}]
    # 200k reserved by each of the first two planning items; the third needs 200k more
    res = json.loads(swarm.run_swarm(meta, items, budget={"max_tokens": 450_000}))
    assert [r["label"] for r in res["results"]] == ["a", "b"]
    assert res["not_started"] == [{"label": "c", "reason": (
        "would exceed the swarm's token budget: 450,000 in all, 0 spent, 400,000 reserved by running "
        "items, 200,000 needed")}]
    assert res["tokens"] == {"max_tokens": 450_000, "spent": 202}
    # measured: a planning item of this host spends 101, so the next swarm
    # reserves that, and a budget of 150 admits one planning item, not two
    assert peaks.token_estimate("planning") == 101
    res = json.loads(swarm.run_swarm(meta, items[:2], budget={"max_tokens": 150}))
    assert [r["label"] for r in res["results"]] == ["a"]
    assert "101 needed" in res["not_started"][0]["reason"]


def test_the_token_budget_arithmetic_is_what_the_reaction_path_uses(monkeypatch, tmp_path):
    """``react`` admits a fired item through the same ``TokenBudget.admit``
    as the initial items: reservation first, settlement on what was spent."""
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    b = swarm.TokenBudget(300_000)
    a, c = {"mode": "planning", "label": "a"}, {"mode": "planning", "label": "c"}
    assert b.admit(a) is None and b.reserved == 200_000
    assert "200,000 reserved by running items, 200,000 needed" in b.admit(c)
    b.settle(a, 1_234)
    assert (b.spent, b.reserved) == (1_234, 0)
    assert b.admit(c) is None                      # the remainder covers it now
    assert swarm.TokenBudget(0).admit({"mode": "analysis", "label": "x"}) is None   # no cap: never refused
    assert b.summary() == {"max_tokens": 300_000, "spent": 1_234}


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


def test_a_fan_out_branch_larger_than_the_host_is_not_started_and_the_rest_run(tmp_path, monkeypatch):
    import scilink.agents.exp_agents.analysis_orchestrator  # noqa: F401
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    paths = [str(tmp_path / f"{n}.npy") for n in "ABC"]
    for p in paths:
        np.save(p, np.zeros((8, 8)))
    _fanout_fakes(monkeypatch, {p: ["good"] for p in paths})
    monkeypatch.setattr(fo, "machine_memory", lambda: {"total": 8e9, "available": 6e9})
    # the input-based estimate never refuses a fan-out branch; a MEASURED class does
    cls = {p: peaks.item_class({"mode": "analysis", "data_path": p}) for p in paths}
    assert len(set(cls.values())) == 1          # same files, one class: measure through a copy
    monkeypatch.setattr(fo, "_branch_mem_estimate", lambda b: 40e9)
    peaks.record("B-class", peak_rss_bytes=40e9)
    real_class = peaks.item_class
    monkeypatch.setattr(peaks, "item_class", lambda it: "B-class" if str(it.get("data_path", "")).endswith("B.npy") else real_class(it))
    ag = _fanout_meta(tmp_path)
    out = json.loads(ag._run_fanout([{"data_path": p, "task": f"Analyze {p}", "label": os.path.basename(p)}
                                     for p in paths]))
    assert out["status"] == "success" and out["branches_run"] == 2
    assert out["not_started"] == [{"label": "B.npy", "reason": "needs about 40.0 GB (measured for its class); this machine has 8.0 GB in all"}]
    assert sorted(e["label"] for e in ag._delegation_ledger if e.get("fanout")) == ["A.npy", "C.npy"]
    # with one branch left there is nothing to fan out
    monkeypatch.setattr(peaks, "item_class", lambda it: "B-class" if not str(it.get("data_path", "")).endswith("A.npy") else real_class(it))
    ag2 = _fanout_meta(tmp_path / "second")
    out = json.loads(ag2._run_fanout([{"data_path": p, "task": f"Analyze {p}", "label": os.path.basename(p)}
                                      for p in paths]))
    assert out["status"] == "declined" and out["reason"] == "does_not_fit_host" and len(out["not_started"]) == 2


def test_the_fan_out_guard_cancels_the_heaviest_branch_and_reruns_it_alone(tmp_path, monkeypatch):
    import scilink.agents.exp_agents.analysis_orchestrator  # noqa: F401
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    A, B = (str(tmp_path / "A.npy"), str(tmp_path / "B.npy"))
    np.save(A, np.zeros((8, 8)))
    np.save(B, np.zeros((64, 64)))                       # the larger estimate
    _fanout_fakes(monkeypatch, {A: ["slow"], B: ["chatty", "good"]})
    # B's class was measured at 2 GB once: it is the one the guard picks
    peaks.record(peaks.item_class({"mode": "analysis", "data_path": B}), peak_rss_bytes=2e9)
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


# ---------------------------------------------------- after the review (round 1)

def test_cancel_ends_a_grandchild_in_its_own_session(tmp_path):
    """A generated script leads its own session, and the worker's SIGTERM
    handler does not run while the worker sits in one long C call: the
    tree kill snapshots the descendants and kills the survivors."""
    pl = LocalProcess(target=TARGET)
    h = pl.submit(_spec("grandchild slow"))
    base = Path(h.spec["base_dir"])
    deadline = time.time() + 20
    while not (base / "grandchild.pid").exists() and time.time() < deadline:
        time.sleep(0.1)
    gpid = int((base / "grandchild.pid").read_text())
    import psutil
    assert psutil.pid_exists(gpid)
    time.sleep(0.5)
    pl.cancel(h)
    assert pl.wait(h, poll_s=0.05)["state"] == "cancelled"
    deadline = time.time() + 5
    while psutil.pid_exists(gpid) and psutil.Process(gpid).status() != "zombie" and time.time() < deadline:
        time.sleep(0.1)
    assert not psutil.pid_exists(gpid) or psutil.Process(gpid).status() == "zombie"


def test_the_workers_console_reaches_the_turn_and_its_log(capsys):
    pl = LocalProcess(target=TARGET)
    h = pl.submit(_spec("plain"))
    pl.wait(h, poll_s=0.05)
    out = capsys.readouterr().out
    assert all(f"worker line {i}" in out for i in range(3)), out[-500:]
    log = (Path(h.spec["base_dir"]) / placements.WORKER_LOG).read_text()
    assert "worker line 2" in log


def test_a_key_the_child_would_not_find_keeps_the_item_a_thread(monkeypatch):
    monkeypatch.delenv(placements.PLACEMENT_ENV, raising=False)
    monkeypatch.delenv("SCILINK_API_KEY", raising=False)
    heavy = {"mode": "analysis", "data_path": "/d"}

    class Proxy:
        api_key, base_url, _shared_extensions = "pk", "http://proxy", []
    assert "SCILINK_API_KEY" in placements.placement_for(Proxy(), heavy, attended=False)[1]
    monkeypatch.setenv("SCILINK_API_KEY", "pk")
    assert placements.placement_for(Proxy(), heavy, attended=False)[0] == "process"
    monkeypatch.setenv("OTHER_KEY", "pk")
    monkeypatch.setenv("SCILINK_API_KEY", "a-different-one")
    assert placements.placement_for(Proxy(), heavy, attended=False)[0] == "thread"

    class Direct:
        api_key, base_url, _shared_extensions = None, None, []
        embedding_api_key, futurehouse_api_key = "ek", None
    assert "embedding API key" in placements.placement_for(Direct(), heavy, attended=False)[1]
    monkeypatch.setenv("OPENAI_API_KEY", "ek")
    assert placements.placement_for(Direct(), heavy, attended=False)[0] == "process"


def test_a_measured_class_is_refused_on_its_raw_peak_and_admitted_with_headroom(tmp_path, monkeypatch):
    np.save(tmp_path / "a.npy", np.zeros((1000, 1000)))
    item = {"mode": "analysis", "task": "t", "label": "ran here", "data_path": str(tmp_path / "a.npy")}
    peaks.record(peaks.item_class(item), peak_rss_bytes=12.5e9)        # a run that succeeded on this machine
    plan = fo.plan_capacity([dict(item)], {"total": 16e9, "available": 8e9}, max_workers=3)
    assert plan["refused"] == [] and plan["run"][0]["_mem_peak"] == 12.5e9
    assert plan["run"][0]["_mem_est"] == 12.5e9 * peaks.PEAK_HEADROOM     # admission plans with headroom
    peaks.record(peaks.item_class(item), peak_rss_bytes=15e9)
    plan = fo.plan_capacity([dict(item)], {"total": 16e9, "available": 8e9}, max_workers=3)
    assert "15.0 GB (measured for its class)" in plan["refused"][0]["reason"]
    # the fan-out's rule: an estimate never refuses, a measurement does
    monkeypatch.setattr(fo, "_branch_mem_estimate", lambda b: 40e9)
    assert fo.plan_capacity([{"mode": "analysis", "label": "guess", "data_path": str(tmp_path / "missing.npy")}],
                            {"total": 16e9, "available": 8e9}, max_workers=3, refuse_unmeasured=False)["refused"] == []
    assert peaks.forget(peaks.item_class(item)) and peaks.measured(peaks.item_class(item)) is None
    assert not peaks.forget("never-seen")


def test_the_class_names_the_series_replay_workers(tmp_path, monkeypatch):
    for i in range(6):
        np.save(tmp_path / f"cube_{i}.npy", np.zeros((10, 10, 10)))
    item = {"mode": "analysis", "data_path": str(tmp_path)}
    monkeypatch.delenv("SCILINK_HS_SERIES_WORKERS", raising=False)
    monkeypatch.delenv("SCILINK_MAX_WORKERS", raising=False)
    one = peaks.item_class(item)
    monkeypatch.setenv("SCILINK_HS_SERIES_WORKERS", "4")
    four = peaks.item_class(item)
    assert four == one + ":w4" and ":w" not in one
    assert peaks.item_class({**item, "series_workers": 2}) == one + ":w2"
    monkeypatch.setenv("SCILINK_HS_SERIES_WORKERS", "16")
    assert peaks.item_class(item) == one + ":w5"              # never more than the units to replay
    assert ":w" not in peaks.item_class({"mode": "analysis", "data_path": str(tmp_path / "cube_0.npy")})


def test_a_cancelled_items_spend_is_charged_when_its_thread_ends(meta, monkeypatch):
    """A thread item cancelled on its budget winds down cooperatively and
    may keep calling the model: five calls of 11,000 tokens, cancelled after
    the second, are charged in full — to the budget and the ledger."""
    class Talker:
        def __init__(self, mode, base_dir):
            self.base_dir = Path(base_dir)

        def run_task(self, task, context=None, autonomy=None):
            for _ in range(5):
                tracing.note_llm_call(prompt_tokens=10_000, completion_tokens=1_000, model="m")
                time.sleep(0.3)                  # no print: the cancel is never noticed
            return {"status": "success", "summary": task, "key_findings": ["f"], "files_produced": []}

    def build(orch, mode, base_dir, **kw):
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        return Talker(mode, base_dir)
    monkeypatch.setattr(swarm, "build_child", build)
    res = json.loads(swarm.run_swarm(meta, [{"mode": "planning", "task": "a", "label": "a"},
                                            {"mode": "planning", "task": "b", "label": "b"}],
                                     item_time_budget_s=0.5, budget={"max_tokens": 1_000_000}))
    assert all(r["status"] == "error" for r in res["results"])
    entries = [e for e in meta._delegation_ledger if e.get("label") in ("a", "b")]
    deadline = time.time() + 10
    while any(e.get("tokens") != 55_000 for e in entries) and time.time() < deadline:
        time.sleep(0.1)
    assert [e["tokens"] for e in entries] == [55_000, 55_000]
    assert all(tracing.worker_usage(f"swarm:{e['index']:02d}_{e['label']}")["calls"] == 0 for e in entries)


def test_the_real_run_item_runs_in_a_real_child_on_a_failing_model(tmp_path, monkeypatch):
    """Pins the spec's pickling, the child's imports and the result shape:
    the real ``run_item`` builds a real analysis worker, whose one model
    call fails on a model that does not exist (no tokens spent)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("SCILINK_LLM_RETRIES", "0")
    spec = {"mode": "analysis", "task": "say hello", "context": None, "label": "real", "autonomy": "AUTONOMOUS",
            "base_dir": str(tmp_path / "real"), "budget_s": 60, "sandbox_approved": False,
            "host": {"model_name": "anthropic/claude-no-such-model-0", "base_url": None,
                     "embedding_model": None, "embedding_api_key": None, "embedding_base_url": None,
                     "futurehouse_api_key": None, "knowledge_dir": None, "file_roots": [str(tmp_path)],
                     "extensions": []}}
    pl = LocalProcess()
    h = pl.submit(spec)
    final = pl.wait(h, poll_s=0.2)
    assert final["state"] == "done", (final, h.stop_reason)
    assert isinstance(h.result, dict) and h.result.get("status") in ("error", "success")
    assert set(h.result) >= {"status", "summary", "key_findings", "files_produced", "warnings"}
    assert (tmp_path / "real" / placements.WORKER_LOG).exists()
    assert final["peak_rss_bytes"] > 1e8


def slow_replay(spec):
    time.sleep(60)
    return {"index": spec["index"], "result": {"status": "success"}}


def test_a_cancelled_series_ends_its_replay_children_within_seconds(monkeypatch):
    """The replay pool's wait reads the series' own cancel (an item's budget,
    the guard, a Stop) and kills the children it recorded, instead of
    letting the cancel land only when the next replay finishes."""
    import logging
    from scilink.agents.exp_agents.controllers import hyperspectral_series as hs
    monkeypatch.setattr(hs, "_REPLAY_TARGET", f"{__name__}:slow_replay")
    monkeypatch.delenv("SCILINK_HS_SERIES_POOL", raising=False)
    ev, out = threading.Event(), {}

    def series_item():
        register_cancel(ev)
        try:
            pool = hs.ReplayPool(2, logging.getLogger("t"))
            pool.submit({"index": 0})
            pool.submit({"index": 1})
            t0 = time.monotonic()
            try:
                pool.collect()
            except AgentStoppedError:
                out["after_s"] = time.monotonic() - t0
            out["children"] = [p.pid for p in __import__("psutil").Process().children(recursive=True)
                               if any("child_process" in a for a in (p.cmdline() or []))]
        finally:
            unregister_cancel()
    t = threading.Thread(target=series_item)
    t.start()
    time.sleep(3.0)                              # the children are up
    ev.set()
    t.join(30)
    assert out.get("after_s") is not None and out["after_s"] < 12, out
    assert out["children"] == [], out
