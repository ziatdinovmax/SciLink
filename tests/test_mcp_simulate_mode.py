"""Simulation orchestrator over MCP — offline tests (handlers with stub orch).

Mirrors tests/test_mcp_meta_mode.py: exercises the `--mode simulate` wiring
(orchestrate routing, autonomy resolution, background jobs, set_autonomy
propagation) without a real server or LLM.
"""

import asyncio
import json

import pytest

pytest.importorskip("mcp")

from scilink.mcp_server import (  # noqa: E402
    _execute_run_task_captured,
    _handle_orchestrate,
    _handle_set_autonomy,
)
from scilink.agents.sim_agents.simulation_orchestrator import (  # noqa: E402
    SimulationMode,
)


def _state(mode="autonomous"):
    return {"analysis_orch": None, "planning_orch": None,
            "simulation_orch": None, "meta_orch": None,
            "pending": {}, "jobs": {}, "job_counter": 0,
            "config": {"analysis_mode": mode, "hitl_timeout_s": 5}}


class SimStub:
    """Stands in for SimulationOrchestratorAgent's run_task contract."""

    def __init__(self, resting=SimulationMode.AUTONOMOUS):
        self.simulation_mode = resting
        self.seen = []

    def set_simulation_mode(self, mode):
        self.simulation_mode = mode

    def run_task(self, prompt, autonomy=None, **kw):
        self.seen.append((prompt, autonomy))
        return {"status": "success", "summary": "built + ran the calculation",
                "files_produced": ["POSCAR"], "key_findings": []}


# The autonomy branch in _execute_run_task_captured resolves the enum from the
# orchestrator's module name, so the stub must look like it lives there.
SimStub.__module__ = "scilink.agents.sim_agents.simulation_orchestrator"


class _InlineExecutor:
    def submit(self, fn, *args):
        import concurrent.futures
        f = concurrent.futures.Future()
        f.set_result(fn(*args))
        return f


def test_run_task_captured_resolves_simulation_mode():
    stub = SimStub()
    out = json.loads(_execute_run_task_captured(stub, "build TiO2", "autopilot"))
    # autopilot serving -> SimulationMode.AUTOPILOT passed to run_task
    assert stub.seen[0][1] == SimulationMode.AUTOPILOT
    assert out["status"] == "success"
    assert out["response"] == "built + ran the calculation"   # summary -> response


def test_orchestrate_simulation_requires_simulate_mode():
    state = _state()
    (content,) = asyncio.run(_handle_orchestrate(
        state, "scilink_orchestrate_simulation", {"prompt": "x"}, None))
    payload = json.loads(content.text)
    assert payload["status"] == "error"
    assert "--mode simulate" in payload["message"]      # not "--mode both"


def test_orchestrate_simulation_background_job():
    state = _state()
    state["simulation_orch"] = SimStub()
    (content,) = asyncio.run(_handle_orchestrate(
        state, "scilink_orchestrate_simulation",
        {"prompt": "relax Cu fcc", "background": True}, _InlineExecutor()))
    payload = json.loads(content.text)
    assert payload["status"] == "started"
    job = state["jobs"][payload["job_id"]]
    assert job["tool"] == "orchestrate_simulation"
    assert json.loads(job["future"].result())["status"] == "success"


def test_set_autonomy_propagates_to_simulation():
    state = _state(mode="autonomous")
    state["simulation_orch"] = SimStub(resting=SimulationMode.AUTONOMOUS)
    (content,) = _handle_set_autonomy(state, {"mode": "co-pilot"})
    assert json.loads(content.text)["status"] == "success"
    assert state["simulation_orch"].simulation_mode == SimulationMode.CO_PILOT
    _handle_set_autonomy(state, {"mode": "autonomous"})
    assert state["simulation_orch"].simulation_mode == SimulationMode.AUTONOMOUS


def test_simulate_mode_registers_sim_tools_end_to_end(tmp_path):
    """create_server(mode='simulate') builds the real sim orchestrator and lists
    its tools with clean scilink_ names — standalone (no analysis/planning leak),
    orchestrate + control tools present, long tools background-capable."""
    import mcp.types as types
    from scilink.mcp_server import create_server

    srv = create_server(api_key="sk-dummy", mode="simulate",
                        session_dir=str(tmp_path))
    srv.eager_init()          # constructs the sim orchestrator + tool_map
    handler = srv.request_handlers[types.ListToolsRequest]
    res = asyncio.run(handler(types.ListToolsRequest(method="tools/list")))
    tools = res.root.tools
    names = {t.name for t in tools}

    # core simulation surface, clean prefix, the orchestrate tool
    assert {"scilink_run_simulation", "scilink_generate_structure",
            "scilink_route_simulation", "scilink_orchestrate_simulation",
            "scilink_analyze_output", "scilink_list_available_software"} <= names
    assert all(not n.startswith("scilink_sim_") for n in names)   # no collision prefix
    # generic control/job tools come for free
    assert {"scilink_job_status", "scilink_job_result",
            "scilink_respond"} <= names
    # standalone: analysis/planning tool sets are NOT registered
    assert "scilink_analyze_file" not in names
    assert "scilink_run_optimization" not in names
    # a long sim tool got the optional background param injected
    rs = next(t for t in tools if t.name == "scilink_run_simulation")
    assert "background" in (rs.inputSchema.get("properties") or {})
