"""A thread's own cancel reaches the waits SciLink owns: the cluster
executor's job, a hyperspectral series' replay children, and the process
tree kill that never snapshots a reaped process.
"""

import threading
import time

from scilink.ui.output_capture import AgentStoppedError
from scilink.utils.log_context import is_cancelled, register_cancel, unregister_cancel


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


# ---------------------------------------------------- the series' replay pool

def slow_replay(spec):
    time.sleep(60)
    return {"index": spec["index"], "result": {"status": "success"}}


def test_a_cancelled_series_ends_its_replay_children_within_seconds(monkeypatch):
    """The replay pool's wait reads the series' own cancel (a budget, the
    guard, a Stop) and kills the children it recorded, instead of letting
    the cancel land only when the next replay finishes."""
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
            import psutil

            def live_children():
                out_ = []
                for c in psutil.Process().children(recursive=True):
                    try:                             # a dying child is a zombie for a moment
                        if c.status() != psutil.STATUS_ZOMBIE and any(
                                "child_process" in a for a in (c.cmdline() or [])):
                            out_.append(c.pid)
                    except psutil.Error:
                        continue
                return out_
            deadline = time.monotonic() + 5
            while live_children() and time.monotonic() < deadline:
                time.sleep(0.2)
            out["children"] = live_children()
        finally:
            unregister_cancel()
    t = threading.Thread(target=series_item)
    t.start()
    time.sleep(3.0)                              # the children are up
    ev.set()
    t.join(30)
    assert out.get("after_s") is not None and out["after_s"] < 12, out
    assert out["children"] == [], out


# ------------------------------------------------------- the process tree kill

def test_a_reaped_pid_is_not_snapshotted(monkeypatch):
    import subprocess
    from scilink import executors
    proc = subprocess.Popen([__import__("sys").executable, "-c", "pass"])
    proc.wait()
    called = {"n": 0}

    def descendants(pid):
        called["n"] += 1
        return []
    monkeypatch.setattr(executors, "_descendants_of", descendants)
    executors._kill_process_tree(proc)
    assert called["n"] == 0
