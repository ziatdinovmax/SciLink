"""Phase-2 durability tests: feedback log, pending sidecar, timeout policy."""

import json
import threading
import time
from pathlib import Path

import pytest

from scilink.hitl import (
    FeedbackRequest,
    QueueChannel,
    request_human_feedback,
    set_default_channel,
    set_thread_channel,
    set_thread_feedback_log,
    get_thread_feedback_log,
    use_feedback_log,
)


class StubChannel:
    def __init__(self, answer=""):
        self.answer = answer

    def ask(self, req):
        return self.answer


@pytest.fixture(autouse=True)
def _reset():
    yield
    set_default_channel(None)
    set_thread_channel(None)
    set_thread_feedback_log(None)


def _records(log: Path):
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_log_records_asked_and_answered(tmp_path):
    log = tmp_path / "feedback_log.jsonl"
    set_default_channel(StubChannel(answer="looks good"))
    with use_feedback_log(log):
        out = request_human_feedback(
            "p: ", kind="review_plan", default="", origin={"stage": "s"})
    assert out == "looks good"
    recs = _records(log)
    assert [r["event"] for r in recs] == ["asked", "answered"]
    assert recs[0]["kind"] == "review_plan"
    assert recs[0]["id"] == recs[1]["id"]
    assert recs[1]["answer"] == "looks good"
    assert "elapsed_s" in recs[1]
    # pending sidecar cleared after the answer
    assert not (tmp_path / "pending_question.json").exists()


def test_pending_sidecar_exists_while_blocked(tmp_path):
    log = tmp_path / "feedback_log.jsonl"
    sidecar = tmp_path / "pending_question.json"
    observed = {}

    class InspectingChannel:
        def ask(self, req):
            observed["pending"] = json.loads(sidecar.read_text())
            return "ok"

    set_default_channel(InspectingChannel())
    with use_feedback_log(log):
        request_human_feedback("p: ", kind="confirm", origin={"stage": "x"})
    assert observed["pending"]["kind"] == "confirm"
    assert observed["pending"]["origin"] == {"stage": "x"}
    assert not sidecar.exists()


def test_interrupted_prompt_is_recorded_and_sidecar_cleared(tmp_path):
    log = tmp_path / "feedback_log.jsonl"

    class EOFChannel:
        def ask(self, req):
            raise EOFError

    set_default_channel(EOFChannel())
    with use_feedback_log(log):
        with pytest.raises(EOFError):
            request_human_feedback("p: ", kind="review_fit")
    recs = _records(log)
    assert [r["event"] for r in recs] == ["asked", "interrupted"]
    assert recs[1]["error"] == "EOFError"
    assert not (tmp_path / "pending_question.json").exists()


def test_no_binding_means_no_files(tmp_path, monkeypatch):
    set_default_channel(StubChannel(answer="x"))
    monkeypatch.chdir(tmp_path)
    assert request_human_feedback("p: ") == "x"
    assert list(tmp_path.iterdir()) == []


def test_nested_rebinding_restores_caller(tmp_path):
    """The run_task pattern: child chat rebinds; caller binding restored."""
    meta_log = tmp_path / "meta" / "feedback_log.jsonl"
    child_log = tmp_path / "child" / "feedback_log.jsonl"
    set_default_channel(StubChannel(answer=""))

    set_thread_feedback_log(str(meta_log))            # meta chat entry
    prev = get_thread_feedback_log()                  # run_task saves
    set_thread_feedback_log(str(child_log))           # child chat entry
    request_human_feedback("child prompt: ", kind="review_plan")
    set_thread_feedback_log(prev)                     # run_task finally
    request_human_feedback("meta prompt: ", kind="confirm")

    assert [r["kind"] for r in _records(child_log)
            if r["event"] == "asked"] == ["review_plan"]
    assert [r["kind"] for r in _records(meta_log)
            if r["event"] == "asked"] == ["confirm"]


def test_queue_channel_timeout_policy_with_log(tmp_path):
    """An unattended queued prompt times out to its default AND the log
    still records the answered event with the default answer."""
    log = tmp_path / "feedback_log.jsonl"
    set_thread_channel(QueueChannel(timeout_s=0.05))
    with use_feedback_log(log):
        out = request_human_feedback("p: ", kind="confirm", default="n")
    assert out == "n"
    recs = _records(log)
    assert recs[-1]["event"] == "answered"
    assert recs[-1]["answer"] == "n"


def test_thread_isolated_logs(tmp_path):
    set_default_channel(StubChannel(answer=""))
    log_a = tmp_path / "a.jsonl"
    log_b = tmp_path / "b.jsonl"

    def worker():
        set_thread_feedback_log(str(log_b))
        request_human_feedback("wp: ", kind="review_fit")

    set_thread_feedback_log(str(log_a))
    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=5)
    request_human_feedback("mp: ", kind="review_plan")
    assert [r["kind"] for r in _records(log_a)
            if r["event"] == "asked"] == ["review_plan"]
    assert [r["kind"] for r in _records(log_b)
            if r["event"] == "asked"] == ["review_fit"]
