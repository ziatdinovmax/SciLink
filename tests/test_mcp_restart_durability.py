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
