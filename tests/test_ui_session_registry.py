"""UI session registry: reattach a live run after a browser disconnect.

The registry is plain Python (dict state), so the disconnect scenario is
simulated directly: session A syncs, its "browser" vanishes (nothing to do —
the registry is process-scoped), a fresh state B attaches and must receive
the SAME live objects the background thread mutates.
"""
import threading

from scilink.ui import registry
from scilink.ui.state import ChatTask, FeedbackRequest


def setup_function(_):
    registry._SESSIONS.clear()


def _state(sdir="s1", **over):
    st = {"session_dir": sdir, "agent": object(), "agent_config": {"model": "m"},
          "chat_messages": [{"role": "user", "content": "hi"}],
          "chat_task": ChatTask(), "known_images": set(), "app_mode": "meta"}
    st.update(over)
    return st


def test_sync_and_reattach_share_live_objects():
    a = _state()
    a["chat_task"].is_running = True
    registry.sync_from(a)

    b = {}  # fresh browser session after the disconnect
    assert registry.attach_to(b, "s1")
    assert b["agent"] is a["agent"]
    assert b["chat_task"] is a["chat_task"]
    assert b["chat_messages"] is a["chat_messages"]
    assert b["agent_initialized"] is True and b["session_dir"] == "s1"

    # the background thread finishes while only B is attached
    a["chat_task"].result = "done"
    a["chat_task"].is_running = False
    assert b["chat_task"].result == "done"


def test_sync_tracks_reassigned_task():
    a = _state()
    registry.sync_from(a)
    fresh = ChatTask(is_running=True)
    a["chat_task"] = fresh                # new turn starts
    registry.sync_from(a)                 # next script run syncs
    b = {}
    registry.attach_to(b, "s1")
    assert b["chat_task"] is fresh


def test_status_of_all_states():
    t = ChatTask()
    assert registry.status_of({"chat_task": t}) == "idle"
    t.is_running = True
    assert registry.status_of({"chat_task": t}) == "running"
    t.feedback_request = FeedbackRequest(prompt="ok?")
    assert registry.status_of({"chat_task": t}) == "awaiting input"
    t.is_running = False
    t.feedback_request = None
    t.result = "r"
    assert registry.status_of({"chat_task": t}) == "finished"
    assert registry.status_of({}) == "idle"


def test_live_entries_sorted_and_described():
    registry.sync_from(_state("s_old"))
    registry.sync_from(_state("s_new", chat_messages=[{}, {}, {}]))
    entries = registry.live_entries()
    assert [e["session_dir"] for e in entries] == ["s_new", "s_old"]
    assert entries[0]["n_messages"] == 3
    assert entries[0]["config"] == {"model": "m"}


def test_unregister_and_missing_attach():
    registry.sync_from(_state())
    registry.unregister("s1")
    assert registry.live_entries() == []
    assert registry.attach_to({}, "s1") is False
    registry.unregister(None)  # no-op, no raise


def test_pending_feedback_survives_reattach():
    a = _state()
    a["chat_task"].is_running = True
    req = FeedbackRequest(prompt="approve the plan?")
    a["chat_task"].feedback_request = req
    registry.sync_from(a)
    b = {}
    registry.attach_to(b, "s1")
    # B answers; the blocked agent thread (waiting on the SAME event) wakes
    got = {}
    waiter = threading.Thread(target=lambda: got.setdefault("r", (req.event.wait(2), req.response)))
    waiter.start()
    b["chat_task"].feedback_request.response = "yes"
    b["chat_task"].feedback_request.event.set()
    waiter.join()
    assert got["r"] == (True, "yes")


def test_sync_without_session_dir_is_noop():
    registry.sync_from({"agent": object()})
    assert registry.live_entries() == []
