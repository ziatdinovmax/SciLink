"""Questions from concurrent workers reach the right worker, say who is asking,
and never hold a worker forever.

Several workers of one fan-out or swarm can stop at a gate at once. Each
question is tagged with the worker and the subject it works on, parked on one
queue and served to the person one at a time; each answer goes back to the
worker that asked. A worker waits at most ``question_timeout_s()`` and then
goes on with the gate's own default, the question is withdrawn so nobody is
asked something no one waits for, and the worker's feedback log says so.
"""

import json
import threading
import time

import pytest

from scilink import hitl
from scilink.hitl import (FeedbackRequest, QueueChannel, WorkerChannel, question_timeout_s,
                          request_human_feedback, set_thread_channel, use_feedback_log)


class ByAsker:
    """A person who answers each worker by name, and remembers what was asked."""

    def __init__(self, delay=0.0):
        self.prompts, self.delay = [], delay

    def ask(self, req):
        self.prompts.append(req.prompt)
        time.sleep(self.delay)
        return "answer for " + req.origin["branch_label"]


def _serve_until(qch, person, n, timeout=5):
    deadline, served = time.time() + timeout, 0
    while served < n and time.time() < deadline:
        served += qch.serve_pending(through=person)
        time.sleep(0.01)
    return served


def _wait_for(pred, timeout=5):
    deadline = time.time() + timeout
    while not pred() and time.time() < deadline:
        time.sleep(0.01)
    assert pred()


def _worker(qch, label, subject, answers, log=None):
    def run():
        set_thread_channel(WorkerChannel(qch, label, subject=subject, kind="worker"))
        try:
            with use_feedback_log(log):
                answers[label] = request_human_feedback("\nApprove the plan? ",
                                                        kind="review_plan", default="")
        finally:
            set_thread_channel(None)
    return threading.Thread(target=run)


def test_two_workers_asking_at_once_each_get_their_own_answer(tmp_path):
    qch, answers = QueueChannel(timeout_s=10), {}
    threads = [_worker(qch, "raman-A7", "TiO2 batch A7", answers),
               _worker(qch, "dft-anatase", "anatase cell", answers)]
    [t.start() for t in threads]
    _wait_for(lambda: len(qch.pending()) == 2)

    waiting = qch.pending()
    assert {(w["worker"], w["subject"]) for w in waiting} == {
        ("raman-A7", "TiO2 batch A7"), ("dft-anatase", "anatase cell")}
    assert all(w["kind"] == "review_plan" and not w["being_answered"] for w in waiting)

    person = ByAsker()
    assert _serve_until(qch, person, 2) == 2
    [t.join(5) for t in threads]
    assert answers == {"raman-A7": "answer for raman-A7", "dft-anatase": "answer for dft-anatase"}
    assert sorted(p.split("]")[0] for p in person.prompts) == [
        "\n[worker: dft-anatase · anatase cell", "\n[worker: raman-A7 · TiO2 batch A7"]
    assert qch.pending() == []


def test_an_unanswered_worker_goes_on_with_the_default_and_the_question_is_withdrawn(tmp_path):
    qch, answers = QueueChannel(timeout_s=0.2), {}
    log = tmp_path / "w1" / "feedback_log.jsonl"
    t = _worker(qch, "w1", None, answers, log=log)
    t.start()
    t.join(5)
    assert answers == {"w1": ""}
    assert qch.pending() == []
    person = ByAsker()
    assert qch.serve_pending(through=person) == 0 and person.prompts == []
    events = [json.loads(line)["event"] for line in log.read_text().splitlines()]
    assert events == ["asked", "timed_out", "answered"]
    assert not (tmp_path / "w1" / "pending_question.json").exists()


def test_an_answer_that_comes_after_the_worker_gave_up_is_not_used(capsys):
    qch, answers = QueueChannel(timeout_s=0.2), {}
    t = _worker(qch, "slow", None, answers)
    t.start()
    _wait_for(lambda: qch.pending())
    person = ByAsker(delay=0.5)            # still reading when the worker gives up
    assert qch.serve_pending(through=person) == 1
    t.join(5)
    assert answers == {"slow": ""}
    assert "was not used" in capsys.readouterr().out


def test_a_worker_answered_in_time_is_not_timed_out(tmp_path):
    qch, answers = QueueChannel(timeout_s=2), {}
    log = tmp_path / "feedback_log.jsonl"
    t = _worker(qch, "quick", None, answers, log=log)
    t.start()
    _wait_for(lambda: qch.pending())
    assert qch.serve_pending(through=ByAsker()) == 1
    t.join(5)
    assert answers == {"quick": "answer for quick"}
    assert "timed_out" not in log.read_text()


def test_a_fanout_branch_keeps_its_prompt_label():
    qch = QueueChannel(timeout_s=5)
    person = ByAsker()
    out = {}

    def branch():
        out["a"] = WorkerChannel(qch, "raman").ask(FeedbackRequest(prompt="\nYour choice: "))

    t = threading.Thread(target=branch)
    t.start()
    _wait_for(lambda: qch.pending())
    qch.serve_pending(through=person)
    t.join(5)
    assert person.prompts == ["\n[branch: raman]\nYour choice: "]


@pytest.mark.parametrize("raw,expected", [
    (None, hitl.QUESTION_TIMEOUT_S), ("", hitl.QUESTION_TIMEOUT_S), ("90", 90.0),
    ("0", None), ("none", None), ("-5", None), ("soon", hitl.QUESTION_TIMEOUT_S)])
def test_the_question_timeout_comes_from_the_environment(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("SCILINK_QUESTION_TIMEOUT_S", raising=False)
    else:
        monkeypatch.setenv("SCILINK_QUESTION_TIMEOUT_S", raw)
    assert question_timeout_s() == expected


def test_fanout_parks_branch_questions_with_the_timeout():
    import inspect
    from scilink.agents.meta_agent import fanout
    src = inspect.getsource(fanout)
    assert "QueueChannel(timeout_s=question_timeout_s())" in src
    assert "WorkerChannel(queue_channel" in src
