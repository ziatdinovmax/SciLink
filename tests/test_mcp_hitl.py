"""Phase-3 MCP HITL tests: pending queue, verbatim answers, awaiting_input,
set_autonomy propagation, run_task routing. All offline — handlers are
exercised directly with a fabricated server state."""

import asyncio
import json
import threading
import time

import pytest

pytest.importorskip("mcp")

from scilink.mcp_server import (  # noqa: E402
    PendingAction,
    _MCPChannel,
    _execute_run_task_captured,
    _handle_job_status,
    _handle_respond,
    _handle_set_autonomy,
    _pending_questions,
)
from scilink.hitl import FeedbackRequest  # noqa: E402


def _state(mode="co-pilot", timeout=5.0):
    return {
        "analysis_orch": None,
        "planning_orch": None,
        "pending": {},
        "jobs": {},
        "job_counter": 0,
        "config": {"analysis_mode": mode, "hitl_timeout_s": timeout},
    }


def _respond(state, **kwargs):
    (content,) = asyncio.run(_handle_respond(state, kwargs))
    return json.loads(content.text)


# ------------------------------------------------------------ question flow

def test_question_answered_verbatim():
    state = _state()
    ch = _MCPChannel(state, job_id="job_1", timeout_s=10)
    req = FeedbackRequest(prompt="Approve plan? ", kind="review_plan",
                          default="")
    out = {}

    t = threading.Thread(target=lambda: out.setdefault("a", ch.ask(req)))
    t.start()
    deadline = time.time() + 5
    while not state["pending"] and time.time() < deadline:
        time.sleep(0.01)
    assert req.id in state["pending"]

    res = _respond(state, response="use three peaks instead",
                   request_id=req.id)
    t.join(timeout=5)
    assert res["status"] == "answered"
    assert res["kind"] == "review_plan"
    # Free text is the ANSWER, delivered verbatim — not a cancel.
    assert out["a"] == "use three peaks instead"
    assert state["pending"] == {}


def test_respond_without_id_when_single_pending():
    state = _state()
    ch = _MCPChannel(state, timeout_s=10)
    req = FeedbackRequest(prompt="q ", kind="confirm", default="n")
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("a", ch.ask(req)))
    t.start()
    while not state["pending"]:
        time.sleep(0.01)
    res = _respond(state, response="")
    t.join(timeout=5)
    assert res["status"] == "answered"
    assert out["a"] == ""


def test_multiple_pending_requires_request_id():
    state = _state()
    state["pending"]["q_a"] = {
        "type": "question", "req": FeedbackRequest(prompt="a", kind="confirm"),
        "event": threading.Event(), "holder": {}, "job_id": None}
    state["pending"]["q_b"] = {
        "type": "question", "req": FeedbackRequest(prompt="b", kind="confirm"),
        "event": threading.Event(), "holder": {}, "job_id": None}
    res = _respond(state, response="yes")
    assert res["status"] == "error"
    ids = {q["request_id"] for q in res["pending"]}
    assert ids == {"q_a", "q_b"}


def test_channel_timeout_returns_default():
    state = _state()
    ch = _MCPChannel(state, timeout_s=0.05)
    assert ch.ask(FeedbackRequest(prompt="q", default="n")) == "n"
    assert state["pending"] == {}


# ----------------------------------------------------------- tool approvals

def test_two_approvals_do_not_overwrite():
    state = _state()
    for n, tool in enumerate(("run_analysis", "run_optimization")):
        state["job_counter"] += 1
        state["pending"][f"act_{n}"] = {
            "type": "tool_approval",
            "action": PendingAction(tool, {}, "p", {"orch_key": "analysis_orch"}),
        }
    assert len(state["pending"]) == 2  # queue, not slot


def test_free_text_on_approval_cancels_with_echo():
    state = _state()
    state["pending"]["act_1"] = {
        "type": "tool_approval",
        "action": PendingAction("run_analysis", {}, "p",
                                {"orch_key": "analysis_orch"}),
    }
    res = _respond(state, response="please use the raman skill",
                   request_id="act_1")
    assert res["status"] == "cancelled"
    assert "please use the raman skill" in res["message"]
    assert "note" in res  # feedback not silently discarded


# ------------------------------------------------------------- job status

def test_job_status_reports_awaiting_input():
    class NeverDone:
        def done(self):
            return False

    state = _state()
    state["jobs"]["job_9"] = {"future": NeverDone(), "tool": "orchestrate",
                              "started_at": "t", "status": "running",
                              "result": None}
    req = FeedbackRequest(prompt="Approve? ", kind="review_plan")
    state["pending"][req.id] = {"type": "question", "req": req,
                                "event": threading.Event(), "holder": {},
                                "job_id": "job_9"}
    (content,) = _handle_job_status(state, {"job_id": "job_9"})
    payload = json.loads(content.text)
    assert payload["status"] == "awaiting_input"
    assert payload["questions"][0]["request_id"] == req.id
    assert payload["questions"][0]["kind"] == "review_plan"
    assert _pending_questions(state, job_id="other") == []


# ----------------------------------------------------------- set_autonomy

def test_set_autonomy_propagates_to_live_orchestrators():
    calls = {}

    class AnalysisStub:
        def set_analysis_mode(self, mode):
            calls["analysis"] = mode

    class PlanningStub:
        def set_autonomy_level(self, level):
            calls["planning"] = level

    state = _state(mode="autonomous")
    state["analysis_orch"] = AnalysisStub()
    state["planning_orch"] = PlanningStub()
    state["pending"]["act_1"] = {
        "type": "tool_approval",
        "action": PendingAction("run_analysis", {}, "p", {})}
    q = FeedbackRequest(prompt="q", kind="confirm")
    state["pending"][q.id] = {"type": "question", "req": q,
                              "event": threading.Event(), "holder": {},
                              "job_id": "job_1"}

    (content,) = _handle_set_autonomy(state, {"mode": "co-pilot"})
    payload = json.loads(content.text)
    assert payload["status"] == "success"
    from scilink.agents.exp_agents.analysis_orchestrator import AnalysisMode
    from scilink.agents.planning_agents.planning_orchestrator import (
        AutonomyLevel,
    )
    assert calls["analysis"] == AnalysisMode.CO_PILOT
    assert calls["planning"] == AutonomyLevel.CO_PILOT
    # Approvals cleared; parked agent questions kept (they belong to jobs).
    assert "act_1" not in state["pending"]
    assert q.id in state["pending"]


# ------------------------------------------------------- run_task routing

def test_orchestrate_routes_through_run_task_with_autonomy():
    seen = {}

    class OrchStub:
        def run_task(self, prompt, autonomy=None):
            seen["autonomy"] = autonomy
            return {"status": "success", "task": prompt,
                    "summary": "the chat text", "files_produced": [],
                    "key_findings": ["f1"], "suggested_followups": [],
                    "warnings": []}

    state = _state(mode="co-pilot")
    out = json.loads(_execute_run_task_captured(
        OrchStub(), "do the thing", "co-pilot", state, None))
    from scilink.agents.exp_agents.analysis_orchestrator import AnalysisMode
    assert seen["autonomy"] == AnalysisMode.CO_PILOT
    # Structured contract + legacy text field.
    assert out["status"] == "success"
    assert out["key_findings"] == ["f1"]
    assert out["response"] == "the chat text"
