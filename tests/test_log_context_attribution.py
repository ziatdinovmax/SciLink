"""Background threads must inherit their session's stdout route (#627).

The web app's per-thread stdout router only forwards a REGISTERED thread's
writes to the session stream; a bare ``threading.Thread`` / ``pool.submit``
worker's prints stay on the server console. ``start_attributed_thread`` and
``attributed_to_current`` are the helpers that make a helper thread a child
of the calling session for exactly the duration of its work.

  conda run -n scilink python -m pytest tests/test_log_context_attribution.py -v
"""
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from scilink.server import stdout_router as sr
from scilink.ui.output_capture import AgentStoppedError
from scilink.utils import log_context as lc


def _routed_session():
    """A routing stream over a fake console plus a capture registered for
    the CALLING thread — the shape of one web-app turn."""
    console = io.StringIO()
    stream = sr._RoutingStream(console)
    cap = sr.RoutedCapture(tag="")
    return console, stream, cap


def test_bare_thread_is_dropped_from_the_session_stream():
    """The bug as observed: an unregistered thread's write reaches the
    console only."""
    console, stream, cap = _routed_session()
    with cap:
        t = threading.Thread(target=stream.write, args=("orphan\n",))
        t.start()
        t.join()
    assert "orphan" in console.getvalue()
    assert "orphan" not in cap.getvalue()


def test_start_attributed_thread_routes_into_the_session_capture():
    console, stream, cap = _routed_session()
    with cap:
        t = lc.start_attributed_thread(stream.write, args=("heartbeat\n",),
                                       name="hb")
        t.join(5)
    assert "heartbeat" in cap.getvalue()
    assert "heartbeat" in console.getvalue()   # still tees to the console


def test_attributed_thread_unregisters_after_its_target_returns():
    """No stale mapping survives the thread (thread ids are reused)."""
    seen = {}

    def target():
        seen["tid"] = threading.get_ident()
        seen["during"] = lc.effective_thread(threading.get_ident())

    t = lc.start_attributed_thread(target)
    t.join(5)
    assert seen["during"] == threading.get_ident()
    assert lc.effective_thread(seen["tid"]) == seen["tid"]


def test_attributed_thread_swallows_a_session_stop():
    """A helper stopped through the routed stream ends quietly — no
    'Exception in thread' traceback on the console."""
    hooked = []
    orig_hook = threading.excepthook
    threading.excepthook = lambda a: hooked.append(a)
    try:
        console, stream, cap = _routed_session()
        with cap:
            cap.request_stop()
            t = lc.start_attributed_thread(stream.write, args=("tick\n",))
            t.join(5)
    finally:
        threading.excepthook = orig_hook
    assert not hooked
    assert "tick" not in cap.getvalue()


def test_attributed_to_current_scopes_registration_to_each_call():
    """A pool worker is a child of the session only while it runs the
    wrapped callable; afterwards the reused thread is unattributed again."""
    parent = threading.get_ident()
    tids = {}

    def job():
        tids["tid"] = threading.get_ident()
        return lc.effective_thread(threading.get_ident())

    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(lc.attributed_to_current(job)).result() == parent
        # The SAME worker thread, unwrapped: no leftover attribution.
        assert pool.submit(job).result() == tids["tid"]


def test_attributed_to_current_propagates_exceptions():
    def boom():
        raise ValueError("branch failed")

    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(ValueError, match="branch failed"):
            pool.submit(lc.attributed_to_current(boom)).result()


def test_attributed_to_current_receives_the_session_stop():
    """The whole point for fan-out branches: a user Stop must reach the
    worker (the Streamlit-era global capture did; the router only reaches
    attributed threads)."""
    console, stream, cap = _routed_session()
    with cap:
        cap.request_stop()
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(lc.attributed_to_current(stream.write), "x\n")
            with pytest.raises(AgentStoppedError):
                fut.result(5)


def test_effective_thread_follows_a_worker_chain():
    """A best-of-N candidate inside a fan-out branch (worker of a worker)
    resolves to the chat thread, so its output reaches the session too."""
    chat = threading.get_ident()
    seen = {}

    def candidate():
        seen["candidate"] = lc.effective_thread(threading.get_ident())

    def branch():
        seen["branch"] = lc.effective_thread(threading.get_ident())
        t = lc.start_attributed_thread(candidate)
        t.join(5)

    t = lc.start_attributed_thread(branch)
    t.join(5)
    assert seen == {"branch": chat, "candidate": chat}


def test_effective_thread_chain_is_bounded():
    """A self-referencing (stale) entry must not spin forever."""
    tid = 424242
    with lc._LOCK:
        lc._WORKERS[tid] = (tid, "", False)
    try:
        assert lc.effective_thread(tid) == tid
    finally:
        with lc._LOCK:
            lc._WORKERS.pop(tid, None)


def test_plain_register_worker_is_unchanged():
    """The existing one-level registration keeps its exact behavior."""
    parent = 777
    lc.register_worker(parent, "cand_00", prefix=True)
    try:
        me = threading.get_ident()
        assert lc.effective_thread(me) == parent
        assert lc.is_concise_fanout_worker(me)
    finally:
        lc.unregister_worker()
    assert lc.effective_thread(threading.get_ident()) == threading.get_ident()


# ---------------------------------------------------------------- stop persistence


@pytest.fixture
def logging_enabled():
    """Some test modules call ``logging.disable(logging.CRITICAL)`` at import
    time, which short-circuits ``logger.info`` before the record factory —
    the hook these tests exercise. Lift it for the test, restore after."""
    import logging
    prev = logging.root.manager.disable
    logging.disable(logging.NOTSET)
    try:
        yield
    finally:
        logging.disable(prev)


def _blocked_worker(stream, gate, results, logger_name="test.stop.persist"):
    """An attributed worker parked in a 'long model call' (the gate) that
    prints and logs only after being released."""
    import logging

    def body():
        gate.wait(10)
        try:
            stream.write("late print\n")
            results["print"] = "wrote"
        except AgentStoppedError:
            results["print"] = "raised"
        try:
            logging.getLogger(logger_name).info("late log record")
            results["log"] = "emitted"
        except AgentStoppedError:
            results["log"] = "raised"
    return body


def test_stop_persists_for_workers_after_the_route_is_gone(logging_enabled):
    """Live failure: a fan-out branch inside a long model call at Stop time
    next printed AFTER the coordinator had exited (route removed, turn log
    handler removed), found nothing to raise on, and kept running. The
    stop must outlive the route for every thread attributed to the turn —
    on the print path and on the logging path."""
    import logging
    logging.getLogger("test.stop.persist").setLevel(logging.INFO)
    console, stream, cap = _routed_session()
    gate, results = threading.Event(), {}
    with cap:
        t = lc.start_attributed_thread(_blocked_worker(stream, gate, results))
        cap.request_stop()
    assert not sr._ROUTES                      # the turn is gone...
    gate.set()                                 # ...now the worker wakes up
    t.join(5)
    assert results == {"print": "raised", "log": "raised"}
    assert not lc._STOPPED_PARENTS, "flag must clear with the last straggler"
    assert not lc._WORKERS


def test_worker_attributed_after_the_stop_is_stopped_too(logging_enabled):
    """A queued fan-out branch admitted after Stop (seen live: the second
    branch was still waiting for memory headroom) must not start narrating."""
    import logging
    logging.getLogger("test.stop.persist").setLevel(logging.INFO)
    console, stream, cap = _routed_session()
    with cap:
        wrapped = lc.attributed_to_current(lambda: None)   # captured pre-stop
        cap.request_stop()
    gate, results = threading.Event(), {}
    gate.set()
    late = lc.attributed_to_current(_blocked_worker(stream, gate, results))
    t = threading.Thread(target=lambda: (late(), None))
    t.start(); t.join(5)
    assert results == {"print": "raised", "log": "raised"}
    assert not lc._STOPPED_PARENTS


def test_stop_does_not_touch_unrelated_threads_or_a_fresh_turn():
    import logging
    console, stream, cap = _routed_session()
    with cap:
        lc.start_attributed_thread(lambda: None).join(5)
        cap.request_stop()
    # An unrelated (never attributed) thread writes and logs freely.
    out = {}
    def other():
        stream.write("unrelated\n")
        logging.getLogger("test.stop.other").warning("unrelated record")
        out["ok"] = True
    t = threading.Thread(target=other); t.start(); t.join(5)
    assert out.get("ok")
    # A fresh turn on the SAME thread id starts clean.
    cap2 = sr.RoutedCapture(tag="")
    with cap2:
        res = {}
        lc.start_attributed_thread(
            lambda: res.setdefault("w", stream.write("fresh\n"))).join(5)
    assert res.get("w") == 6
    assert "fresh" in cap2.getvalue()
    assert not lc._STOPPED_PARENTS


def test_streamlit_capture_stop_also_persists_for_workers(logging_enabled):
    """The Streamlit OutputCapture shares the gap (its TeeStream is
    restored when the turn exits); same wiring, same guarantee."""
    from scilink.ui.output_capture import OutputCapture
    cap = OutputCapture()
    gate, results = threading.Event(), {}
    with cap:
        t = lc.start_attributed_thread(_blocked_worker(sr._RoutingStream(io.StringIO()), gate, results))
        cap.request_stop()
    gate.set(); t.join(5)
    assert results == {"print": "raised", "log": "raised"}
    assert not lc._STOPPED_PARENTS
