"""The scalarizer's per-file "confirm these columns" prompt is off in every
autonomy mode. It sat behind the automated checks (runtime, row-count trap,
plot self-check, schema verification), fired only on the codegen path, and
blocked a delegated meta turn on a terminal keypress. analyze_file and
analyze_batch must never ask for it; the scalarizer's own default is off."""
import contextlib
import inspect
import io
import json
import os
from pathlib import Path

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

from scilink.agents.planning_agents.planning_orchestrator import (
    PlanningOrchestratorAgent, AutonomyLevel,
)
from scilink.agents.planning_agents.scalarizer_agent import ScalarizerAgent

CSV = "T,t,Y\n30,10,8.5\n90,10,22.4\n50,35,48.8\n"


def test_scalarizer_default_is_off():
    assert inspect.signature(ScalarizerAgent.scalarize).parameters[
        "enable_human_review"].default is False


@pytest.mark.parametrize("level", [AutonomyLevel.CO_PILOT, AutonomyLevel.AUTOPILOT,
                                   AutonomyLevel.AUTONOMOUS])
def test_analyze_file_and_batch_never_request_review(tmp_path, level):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "a.csv").write_text(CSV)
    (data_dir / "b.csv").write_text(CSV)
    with contextlib.redirect_stdout(io.StringIO()):
        o = PlanningOrchestratorAgent(
            base_dir=str(tmp_path / "s"), api_key="sk-dummy",
            autonomy_level=level, data_dir=str(data_dir))
    calls = []

    def fake(**kw):
        calls.append(kw)
        return {"status": "success", "source_script": None, "passthrough": True,
                "metrics": {"T": [30.0, 90.0, 50.0], "t": [10.0, 10.0, 35.0],
                            "Y": [8.5, 22.4, 48.8]},
                "column_roles": {"inputs": ["T", "t"], "targets": ["Y"]}, "error": None}
    o.scalarizer.scalarize = fake
    with contextlib.redirect_stdout(io.StringIO()):
        out = json.loads(o.tools.execute_tool(
            "analyze_file", file_path=str(data_dir / "a.csv"),
            extraction_goal="x", inputs=["T", "t"], targets=["Y"]))
        assert out["status"] == "success"
        o.tools.execute_tool("analyze_batch", file_paths=[str(data_dir / "b.csv")],
                             extraction_goal="x", inputs=["T", "t"], targets=["Y"])
    assert calls, "scalarize was not called"
    assert all(kw.get("enable_human_review") is False for kw in calls), calls
