"""
Hermetic test that the `route_simulation` tool is registered on the
SimulationOrchestrator and dispatches to SimulationRouter correctly.

We don't construct a full SimulationOrchestratorAgent (it does heavy
setup: chat history, mode validation, MP cache, etc.). Instead we
build a minimal shim object that quacks like the orchestrator from
the tools' perspective, instantiate SimulationOrchestratorTools, and
exercise the routing tool via execute_tool().
"""

import json
from unittest.mock import MagicMock
from pathlib import Path

import pytest

from scilink.agents.sim_agents.simulation_orchestrator_tools import (
    SimulationOrchestratorTools,
)
from scilink.utils.available_software import AvailableSoftware


class _MockOrch:
    """Minimal shim that satisfies the attributes the tools look up."""
    def __init__(self, model, tmp_path):
        self.model = model
        self.api_key = "test"
        self.base_url = None
        self.model_name = "test-model"
        self.mp_api_key = None
        self.futurehouse_api_key = None
        self.hpc_connection = None
        self.hpc_scheduler = None
        self.base_dir = tmp_path
        self.structures_dir = tmp_path / "structures"
        self.structures_dir.mkdir(exist_ok=True)
        self.generated_structures = []
        self.default_calc_params = {}
        from scilink.agents.sim_agents.simulation_orchestrator import (
            SimulationMode,
        )
        self.simulation_mode = SimulationMode.CO_PILOT
        self.routing_decision = None


def _model_returning(payload: str):
    model = MagicMock()
    resp = MagicMock()
    resp.text = payload
    model.generate_content.return_value = resp
    return model


def test_route_simulation_tool_is_registered(tmp_path):
    """The tool appears in the orchestrator's tool registry."""
    orch = _MockOrch(model=MagicMock(), tmp_path=tmp_path)
    tools = SimulationOrchestratorTools(orch)

    assert "route_simulation" in tools.functions_map
    schema_names = [
        s["function"]["name"] for s in tools.openai_schemas
    ]
    assert "route_simulation" in schema_names


def test_route_simulation_tool_returns_decision(tmp_path, monkeypatch):
    """Calling the tool returns a JSON string with the routing decision."""
    # Mock SimulationRouter's available_software so the tool doesn't
    # touch ~/.scilink/available_software.yaml during the test.
    test_avail = AvailableSoftware()
    test_avail.set("periodic_dft", "vasp", True, source="test")
    monkeypatch.setattr(
        "scilink.utils.available_software.AvailableSoftware.auto",
        classmethod(lambda cls, *a, **kw: test_avail),
    )

    decision_payload = json.dumps({
        "scale": "periodic_dft",
        "engine": "vasp",
        "reasoning": "Cu slab calls for periodic DFT; VASP available.",
        "alternatives": [],
    })
    orch = _MockOrch(model=_model_returning(decision_payload), tmp_path=tmp_path)
    tools = SimulationOrchestratorTools(orch)

    out = tools.execute_tool(
        "route_simulation",
        user_goal="Relax a Cu(111) slab",
        system_description="metallic surface, 16 atoms",
    )

    parsed = json.loads(out)
    assert parsed["scale"] == "periodic_dft"
    assert parsed["engine"] == "vasp"
    assert "reasoning" in parsed
    # The orchestrator's routing_decision state was stashed
    assert orch.routing_decision is not None
    assert orch.routing_decision["scale"] == "periodic_dft"


def test_route_simulation_tool_propagates_no_candidates_error(tmp_path, monkeypatch):
    """If no candidates exist, the tool returns the router's error JSON."""
    empty_avail = AvailableSoftware()
    monkeypatch.setattr(
        "scilink.utils.available_software.AvailableSoftware.auto",
        classmethod(lambda cls, *a, **kw: empty_avail),
    )

    orch = _MockOrch(model=MagicMock(), tmp_path=tmp_path)
    tools = SimulationOrchestratorTools(orch)

    out = tools.execute_tool(
        "route_simulation",
        user_goal="anything",
    )
    parsed = json.loads(out)
    assert parsed["scale"] is None
    assert parsed["engine"] is None
    assert "error" in parsed


def test_route_simulation_tool_required_args(tmp_path):
    """The tool's schema requires `user_goal`."""
    orch = _MockOrch(model=MagicMock(), tmp_path=tmp_path)
    tools = SimulationOrchestratorTools(orch)

    schema = next(
        s for s in tools.openai_schemas
        if s["function"]["name"] == "route_simulation"
    )
    required = schema["function"]["parameters"]["required"]
    assert "user_goal" in required
    assert "system_description" not in required  # optional
