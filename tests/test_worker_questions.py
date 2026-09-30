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
    qch, answers = QueueChannel(timeout_s=0.3), {}
    t = _worker(qch, "slow", None, answers)
    t.start()
    _wait_for(lambda: qch.pending())
    person = ByAsker(delay=1.0)            # far past the clock the showing started
    assert qch.serve_pending(through=person) == 1
    t.join(5)
    assert answers == {"slow": ""}
    assert "was not used" in capsys.readouterr().out


def test_a_question_being_read_gets_a_full_clock_of_its_own(tmp_path):
    """The worker waited almost the whole timeout in the queue; once shown, the
    person still has the whole timeout to answer."""
    qch, answers = QueueChannel(timeout_s=0.6), {}
    log = tmp_path / "feedback_log.jsonl"
    t = _worker(qch, "patient", None, answers, log=log)
    t.start()
    _wait_for(lambda: qch.pending())
    time.sleep(0.45)                       # queued behind others
    person = ByAsker(delay=0.4)            # 0.45 + 0.4 > 0.6, but within 0.6 of being shown
    assert qch.serve_pending(through=person) == 1
    t.join(5)
    assert answers == {"patient": "answer for patient"}
    assert "timed_out" not in log.read_text()


def test_the_gate_can_tell_a_timeout_from_an_answer(tmp_path):
    qch, answers = QueueChannel(timeout_s=0.2), {}
    log = tmp_path / "feedback_log.jsonl"
    seen = {}

    def run():
        set_thread_channel(WorkerChannel(qch, "w", kind="worker"))
        try:
            with use_feedback_log(log):
                answers["w"] = request_human_feedback("\nApprove? ", kind="approve_or_revise", default="")
                seen["timed_out"] = hitl.last_question_timed_out()
        finally:
            set_thread_channel(None)
    t = threading.Thread(target=run)
    t.start()
    t.join(5)
    assert answers == {"w": ""} and seen["timed_out"] is True
    answered = [json.loads(l) for l in log.read_text().splitlines()][-1]
    assert answered["event"] == "answered" and answered.get("unattended") is True
    # an answered question leaves the marker clear
    qch2 = QueueChannel(timeout_s=5)
    t2 = _worker(qch2, "ok", None, answers)
    t2.start()
    _wait_for(lambda: qch2.pending())
    qch2.serve_pending(through=ByAsker())
    t2.join(5)


def test_a_cancelled_worker_withdraws_its_question_and_stops():
    from scilink.utils.log_context import register_cancel, unregister_cancel
    from scilink.ui.output_capture import AgentStoppedError
    qch, out = QueueChannel(timeout_s=30), {}
    stop = threading.Event()

    def run():
        register_cancel(stop)
        try:
            WorkerChannel(qch, "w").ask(FeedbackRequest(prompt="p", default="d"))
            out["result"] = "returned"
        except AgentStoppedError:
            out["result"] = "stopped"
        finally:
            unregister_cancel()
    t = threading.Thread(target=run)
    t.start()
    _wait_for(lambda: qch.pending())
    stop.set()
    t.join(5)
    assert out["result"] == "stopped" and qch.pending() == []
    assert qch.serve_pending(through=ByAsker()) == 0


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
    ("0", None), ("none", None), ("-5", hitl.QUESTION_TIMEOUT_S), ("soon", hitl.QUESTION_TIMEOUT_S),
    ("inf", hitl._QUESTION_TIMEOUT_MAX_S), ("1e10", hitl._QUESTION_TIMEOUT_MAX_S)])
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


# ------------------------------------------------------------ the planning gate

def test_a_timed_out_plan_review_is_not_recorded_as_a_human_decision(tmp_path):
    """A swarm worker's plan gate that nobody answers gets the default, which
    reads as Enter. The planner must not stamp that as human approval: the
    plan stays a draft the agent may revise."""
    from types import SimpleNamespace
    from scilink.agents.planning_agents.base_agent import BaseAgent
    from scilink.agents.planning_agents.planning_agent import PlanningAgent

    a = PlanningAgent.__new__(PlanningAgent)
    BaseAgent.__init__(a, str(tmp_path))
    a.agent_type = "planning"
    a.state = {"plan_history": []}
    plan = {"iteration": 1, "stage": "Initial", "proposed_experiments": [{"name": "x"}]}

    hitl._thread_local.last_timed_out = True
    try:
        a._stamp_human_review(plan, "accepted")
    finally:
        hitl._thread_local.last_timed_out = False
    assert "human_review" not in plan
    assert plan["unattended_gate"]["would_have_been"] == "accepted"

    a._stamp_human_review(plan, "accepted")            # a real Enter, later
    assert plan["human_review"]["status"] == "accepted" and "unattended_gate" not in plan
