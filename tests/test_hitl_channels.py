"""Phase-1 channel tests: QueueChannel handshake and the UI channel contract."""

import threading
import time

import pytest

from scilink.hitl import (
    FeedbackRequest,
    QueueChannel,
    set_default_channel,
    set_thread_channel,
)


class StubChannel:
    def __init__(self, answer=""):
        self.answer = answer
        self.requests = []

    def ask(self, req):
        self.requests.append(req)
        return self.answer


@pytest.fixture(autouse=True)
def _reset_channels():
    yield
    set_default_channel(None)
    set_thread_channel(None)


def test_queue_channel_serves_worker_through_coordinator_channel():
    qch = QueueChannel()
    coordinator = StubChannel(answer="approved")
    results = {}

    def worker():
        set_thread_channel(qch)
        results["answer"] = qch.ask(FeedbackRequest(
            prompt="\nYour choice: ", kind="review_plan",
            origin={"branch_label": "raman"},
        ))

    t = threading.Thread(target=worker)
    t.start()
    # Coordinator loop: poll until the request lands, then serve it.
    deadline = time.time() + 5
    served = 0
    while served == 0 and time.time() < deadline:
        served = qch.serve_pending(through=coordinator)
        time.sleep(0.01)
    t.join(timeout=5)
    assert served == 1
    assert results["answer"] == "approved"
    (req,) = coordinator.requests
    # Branch label is prefixed onto the served prompt.
    assert req.prompt.startswith("\n[branch: raman]")
    assert req.origin["branch_label"] == "raman"


def test_queue_channel_serves_multiple_serially():
    qch = QueueChannel()
    coordinator = StubChannel(answer="")
    threads = [
        threading.Thread(target=lambda i=i: qch.ask(FeedbackRequest(
            prompt=f"p{i}", origin={"branch_label": f"b{i}"})))
        for i in range(3)
    ]
    for t in threads:
        t.start()
    deadline = time.time() + 5
    total = 0
    while total < 3 and time.time() < deadline:
        total += qch.serve_pending(through=coordinator)
        time.sleep(0.01)
    for t in threads:
        t.join(timeout=5)
    assert total == 3
    assert len(coordinator.requests) == 3


def test_queue_channel_timeout_returns_default():
    qch = QueueChannel(timeout_s=0.05)
    answer = qch.ask(FeedbackRequest(prompt="p", default="n"))
    assert answer == "n"  # nobody served → default, no hang


def test_queue_channel_unblocks_worker_when_serving_raises():
    qch = QueueChannel()

    class StopChannel:
        def ask(self, req):
            raise KeyboardInterrupt

    results = {}

    def worker():
        results["answer"] = qch.ask(FeedbackRequest(prompt="p", default="skip"))

    t = threading.Thread(target=worker)
    t.start()
    time.sleep(0.05)
    with pytest.raises(KeyboardInterrupt):
        qch.serve_pending(through=StopChannel())
    t.join(timeout=5)
    assert results["answer"] == "skip"


def test_ui_channel_handshake_carries_kind_and_options():
    """The UI channel publishes a structured request to the task slot and
    returns the widget's response after the event fires."""
    streamlit = pytest.importorskip("streamlit")  # noqa: F841
    from scilink.ui.app import _UIChannel

    class Cap:
        def getvalue(self):
            return "captured context"

    class Task:
        stopped = False
        feedback_request = None

    task = Task()
    ch = _UIChannel(task, Cap())
    out = {}

    def agent_thread():
        out["answer"] = ch.ask(FeedbackRequest(
            prompt="\nYour choice: ", kind="bestofn_select",
            options=["1", "2"], origin={"stage": "bestofn_join"},
        ))

    t = threading.Thread(target=agent_thread)
    t.start()
    deadline = time.time() + 5
    while task.feedback_request is None and time.time() < deadline:
        time.sleep(0.01)
    req = task.feedback_request
    assert req is not None
    assert req.kind == "bestofn_select"
    assert req.options == ["1", "2"]
    assert req.origin["stage"] == "bestofn_join"
    assert req.context == "captured context"
    req.response = "2"
    req.event.set()
    t.join(timeout=5)
    assert out["answer"] == "2"
    assert task.feedback_request is None


def test_ui_channel_metadata_cache_keys_on_origin_filename():
    pytest.importorskip("streamlit")
    from scilink.ui.app import _UIChannel

    class Cap:
        def getvalue(self):
            return ""

    class Task:
        stopped = False
        feedback_request = None

    task = Task()
    ch = _UIChannel(task, Cap())

    def answer_first():
        deadline = time.time() + 5
        while task.feedback_request is None and time.time() < deadline:
            time.sleep(0.01)
        task.feedback_request.response = "XRD powder pattern"
        task.feedback_request.event.set()

    t = threading.Thread(target=answer_first)
    t.start()
    req = FeedbackRequest(prompt="\n> Context: ", kind="dataset_description",
                          origin={"filename": "run1.csv"})
    assert ch.ask(req) == "XRD powder pattern"
    t.join(timeout=5)
    # Second ask for the same file auto-replies from the cache — no publish.
    req2 = FeedbackRequest(prompt="\n> Context: ", kind="dataset_description",
                           origin={"filename": "run1.csv"})
    assert ch.ask(req2) == "XRD powder pattern"
    assert task.feedback_request is None
