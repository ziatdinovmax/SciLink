"""Meta orchestrator over MCP — offline tests (handlers with stub orchs)."""

import asyncio
import json
import logging
import threading
import time

import pytest

pytest.importorskip("mcp")

from scilink.mcp_server import (  # noqa: E402
    _execute_meta_chat_captured,
    _handle_orchestrate_meta,
    _handle_set_autonomy,
    _new_job_lines,
)
from scilink.agents.meta_agent.meta_orchestrator import MetaMode  # noqa: E402


def _state(mode="autopilot"):
    return {"analysis_orch": None, "planning_orch": None, "meta_orch": None,
            "pending": {}, "jobs": {}, "job_counter": 0,
            "config": {"analysis_mode": mode, "hitl_timeout_s": 5}}


class MetaStub:
    def __init__(self, reply="fused findings", resting=MetaMode.AUTOPILOT):
        self.meta_mode = resting
        self.reply = reply
        self.seen = []

    def set_meta_mode(self, mode):
        self.meta_mode = mode

    def chat(self, prompt):
        self.seen.append((prompt, self.meta_mode))
        logging.getLogger("Meta").info("delegating to analysis specialist")
        print("fan-out launched")
        return self.reply


def test_meta_chat_sets_and_restores_mode_and_maps_autonomy():
    stub = MetaStub(resting=MetaMode.AUTONOMOUS)
    prev = logging.getLogger().level
    logging.getLogger().setLevel(logging.INFO)
    try:
        lines = _new_job_lines()
        out = json.loads(_execute_meta_chat_captured(
            stub, "investigate", "autopilot", _state(), "job_1", lines))
    finally:
        logging.getLogger().setLevel(prev)
    # autopilot serving -> AUTOPILOT for the call; resting mode restored.
    assert stub.seen[0][1] == MetaMode.AUTOPILOT
    assert stub.meta_mode == MetaMode.AUTONOMOUS
    assert out == {"status": "success", "response": "fused findings"}
    joined = "\n".join(lines)
    assert "delegating to analysis specialist" in joined  # logging narration
    assert "fan-out launched" in joined                   # stdout tee


def test_meta_chat_autonomous_maps_to_autonomous():
    stub = MetaStub(resting=MetaMode.AUTOPILOT)
    _execute_meta_chat_captured(stub, "go", "autonomous", _state(), None)
    assert stub.seen[0][1] == MetaMode.AUTONOMOUS
    assert stub.meta_mode == MetaMode.AUTOPILOT


def test_meta_chat_error_reply_maps_to_error_status():
    stub = MetaStub(reply="❌ Error: something broke")
    out = json.loads(_execute_meta_chat_captured(
        stub, "go", "autonomous", _state(), None))
    assert out["status"] == "error"
    assert "something broke" in out["response"]


def test_orchestrate_meta_requires_meta_mode():
    state = _state()
    (content,) = asyncio.run(_handle_orchestrate_meta(
        state, {"prompt": "x"}, None))
    payload = json.loads(content.text)
    assert payload["status"] == "error"
    assert "--mode meta" in payload["message"]


def test_orchestrate_meta_background_job_with_log_buffer():
    class Executor:
        def submit(self, fn, *args):
            import concurrent.futures
            f = concurrent.futures.Future()
            f.set_result(fn(*args))
            return f

    state = _state()
    state["meta_orch"] = MetaStub()
    (content,) = asyncio.run(_handle_orchestrate_meta(
        state, {"prompt": "investigate", "background": True}, Executor()))
    payload = json.loads(content.text)
    assert payload["status"] == "started"
    job = state["jobs"][payload["job_id"]]
    assert job["tool"] == "orchestrate_meta"
    assert job["log_lines"] is not None
    assert json.loads(job["future"].result())["status"] == "success"


def test_set_autonomy_propagates_to_meta():
    state = _state(mode="autonomous")
    state["meta_orch"] = MetaStub(resting=MetaMode.AUTONOMOUS)
    (content,) = _handle_set_autonomy(state, {"mode": "autopilot"})
    assert json.loads(content.text)["status"] == "success"
    assert state["meta_orch"].meta_mode == MetaMode.AUTOPILOT
    _handle_set_autonomy(state, {"mode": "autonomous"})
    assert state["meta_orch"].meta_mode == MetaMode.AUTONOMOUS
