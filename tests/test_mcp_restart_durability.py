"""MCP server restart durability.

A server restarted on the same --session-dir must resume the campaign:
granular tool calls checkpoint the owning orchestrator, and construction
restores the checkpoint when one exists. In --mode both the analysis and
planning orchestrators get their own nested base dirs (they shared one
checkpoint.json before). No LLM: the scalarizer is stubbed.
"""
import contextlib
import io
import json
import os
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("mcp")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

from scilink.mcp_server import _init_orchestrators, _execute_tool_captured  # noqa: E402

L6 = dict(T=[30.0, 30.0, 90.0, 90.0, 50.0, 78.0],
          t=[10.0, 50.0, 10.0, 50.0, 35.0, 18.0],
          Y=[8.56, 3.88, 22.47, 10.81, 48.83, 61.1])
CSV = "T,t,Y\n" + "\n".join(f"{a},{b},{c}" for a, b, c in
                            zip(L6["T"], L6["t"], L6["Y"])) + "\n"


def _config(session_dir, mode):
    return {"api_key": "sk-dummy", "model_name": None, "base_url": None,
            "mode": mode, "session_dir": session_dir,
            "analysis_mode": "autonomous", "futurehouse_api_key": None,
            "hitl_timeout_s": 5.0}


def _boot(session_dir, mode="plan"):
    state = {"analysis_orch": None, "planning_orch": None, "meta_orch": None,
             "pending": {}, "jobs": {}, "job_counter": 0,
             "config": _config(session_dir, mode)}
    with contextlib.redirect_stdout(io.StringIO()):
        _init_orchestrators(state, state["config"])
    orch = state["planning_orch"]
    orch.scalarizer.scalarize = lambda **kw: {
        "status": "success", "metrics": dict(L6), "source_script": None,
        "column_roles": {"inputs": ["T", "t"], "targets": ["Y"]},
        "passthrough": True, "error": None}
    orch.bo.run_optimization_loop = lambda **kw: {
        "status": "success", "next_parameters": {"T": 60.0, "t": 25.0}, "strategy": {}}
    return state


def test_granular_calls_checkpoint_and_restart_resumes():
    with tempfile.TemporaryDirectory() as tmp:
        session = str(Path(tmp) / "s")
        csv = Path(tmp) / "seed.csv"
        csv.write_text(CSV)
        st1 = _boot(session)
        out = json.loads(_execute_tool_captured(
            st1["planning_orch"].tools, "analyze_file",
            {"file_path": str(csv), "extraction_goal": "x",
             "inputs": ["T", "t"], "targets": ["Y"]}, state=st1))
        assert out["status"] == "success" and out["rows_added"] == 6
        # the granular call left a checkpoint behind
        assert (Path(session) / "checkpoint.json").exists()

        # "restart": a fresh server on the same dir
        st2 = _boot(session)
        p2 = st2["planning_orch"]
        assert p2.expected_input_columns == ["T", "t"]
        assert p2.expected_target_columns == ["Y"]
        assert not list(Path(session).glob("stale_campaign_*"))
        bo = json.loads(_execute_tool_captured(
            p2.tools, "run_optimization", {"parallel_capable": False}, state=st2))
        assert bo["status"] == "success", bo
        # and the dedup ledger still knows the file (no re-ingest by accident)
        again = json.loads(_execute_tool_captured(
            p2.tools, "analyze_file",
            {"file_path": str(csv), "extraction_goal": "x",
             "inputs": ["T", "t"], "targets": ["Y"]}, state=st2))
        assert again.get("rows_added") == 0


def test_mode_both_nests_orchestrators():
    with tempfile.TemporaryDirectory() as tmp:
        session = str(Path(tmp) / "s")
        st = _boot(session, mode="both")
        a, p = st["analysis_orch"], st["planning_orch"]
        assert Path(a.base_dir).resolve() == (Path(session) / "analysis").resolve()
        assert Path(p.base_dir).resolve() == (Path(session) / "planning").resolve()
        assert a.checkpoint_path != p.checkpoint_path


def test_single_mode_layout_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        session = str(Path(tmp) / "s")
        st = _boot(session, mode="plan")
        assert Path(st["planning_orch"].base_dir).resolve() == Path(session).resolve()


# ── Issue #469: jobs + pending questions survive a restart ────────────────

import threading
import time

from scilink.mcp_server import (  # noqa: E402
    _MCPChannel, _handle_job_status, _handle_job_result, _handle_respond,
    _register_job, _restore_jobs, _persist_jobs, _new_job_lines,
    _JOBS_STORE_NAME,
)


def _mkstate(session_dir):
    return {"analysis_orch": None, "planning_orch": None, "meta_orch": None,
            "pending": {}, "jobs": {}, "job_counter": 0,
            "interrupted_questions": {}, "_persist_lock": threading.Lock(),
            "config": _config(session_dir, "plan")}


def _call(handler, state, args):
    import asyncio as _a
    r = handler(state, args)
    if _a.iscoroutine(r):
        r = _a.run(r)
    return json.loads(r[0].text)


def test_finished_job_survives_restart():
    import concurrent.futures
    with tempfile.TemporaryDirectory() as tmp:
        st = _mkstate(tmp)
        with concurrent.futures.ThreadPoolExecutor(1) as ex:
            fut = ex.submit(lambda: json.dumps({"status": "success", "answer": 42}))
            lines = _new_job_lines(); lines.extend(["did a thing", "done"])
            _register_job(st, "job_x_001", {"future": fut, "tool": "orchestrate_analysis",
                                            "started_at": "x", "status": "running",
                                            "result": None, "log_lines": lines})
            fut.result(); time.sleep(0.3)   # done-callback persists the completion
        assert (Path(tmp) / _JOBS_STORE_NAME).exists()
        st2 = _mkstate(tmp)
        _restore_jobs(st2)
        assert st2["jobs"]["job_x_001"]["status"] == "completed"
        out = _call(_handle_job_status, st2, {"job_id": "job_x_001"})
        assert out["status"] == "completed" and "did a thing" in out.get("log_tail", "")
        res = _call(_handle_job_result, st2, {"job_id": "job_x_001"})
        assert res == {"status": "success", "answer": 42}
        assert st2["job_counter"] >= st["job_counter"]


def test_running_job_becomes_interrupted_with_hint():
    with tempfile.TemporaryDirectory() as tmp:
        st = _mkstate(tmp)
        lines = _new_job_lines(); lines.append("step 3/15 ...")
        st["jobs"]["job_y_001"] = {"future": None, "tool": "orchestrate_meta",
                                   "started_at": "y", "status": "running",
                                   "result": None, "log_lines": lines}
        _persist_jobs(st)
        st2 = _mkstate(tmp)
        _restore_jobs(st2)
        out = _call(_handle_job_status, st2, {"job_id": "job_y_001"})
        assert out["status"] == "interrupted" and "re-issue" in out["message"].lower()
        res = _call(_handle_job_result, st2, {"job_id": "job_y_001"})
        assert res["status"] == "interrupted" and "step 3/15" in res["log_tail"]


def test_parked_question_persisted_and_respond_after_restart_explains():
    class _Req:
        id, kind, prompt, options, origin, default = ("q_1", "review_plan",
                                                      "OK?", [], {}, "")
    with tempfile.TemporaryDirectory() as tmp:
        st = _mkstate(tmp)
        st["jobs"]["job_z_001"] = {"future": None, "tool": "orchestrate_analysis",
                                   "started_at": "z", "status": "running",
                                   "result": None, "log_lines": _new_job_lines()}
        ch = _MCPChannel(st, job_id="job_z_001", timeout_s=0.4)
        t = threading.Thread(target=ch.ask, args=(_Req(),)); t.start()
        time.sleep(0.15)
        onwire = json.loads((Path(tmp) / _JOBS_STORE_NAME).read_text())
        assert onwire["pending_questions"] and onwire["pending_questions"][0]["request_id"] == "q_1"
        # 'restart' while the question is parked
        st2 = _mkstate(tmp)
        _restore_jobs(st2)
        t.join()
        ans = _call(_handle_respond, st2, {"request_id": "q_1", "response": ""})
        assert ans["status"] == "interrupted" and "restart" in ans["message"].lower()
        # and an unknown id still errors as before
        ans = _call(_handle_respond, st2, {"request_id": "nope", "response": ""})
        assert ans["status"] == "error"


def test_live_state_unaffected_and_no_store_without_writes():
    with tempfile.TemporaryDirectory() as tmp:
        st = _mkstate(tmp)
        _restore_jobs(st)                     # no file: no-op
        out = _call(_handle_job_status, st, {"job_id": "missing"})
        assert out["status"] == "error" and "Unknown job_id" in out["message"]
