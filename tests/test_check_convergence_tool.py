"""Tests for the check_observable_convergence orchestrator tool.

Unit tests for convergence-flag scanning and tool-level integration
with a monkeypatched SimulationAnalysisAgent.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Convergence-flag scanning helpers
# ---------------------------------------------------------------------------

# The flag vocabulary must match the tool implementation.
_CONVERGENCE_FLAG_KEYS = frozenset({
    "plateau_reached", "converged", "linear_regime", "extreme_narrowing",
})


def _check_flags(results: dict) -> dict:
    """Pure-function mirror of the tool's convergence-flag scanning logic.

    Given a ``run_analysis`` results dict, return
    ``{"converged": bool, "unconverged": [str], "properties": dict}``.
    """
    properties = {}
    unconverged = []
    for prop, result in results.items():
        if result.get("status") == "error":
            properties[prop] = {"status": "error"}
            continue

        flag_key = None
        flag_value = None
        for key in _CONVERGENCE_FLAG_KEYS:
            if key in result:
                flag_key = key
                flag_value = result[key]
                break

        prop_converged = flag_value if flag_key is not None else True
        properties[prop] = {"converged": bool(prop_converged)}
        if flag_key is not None:
            properties[prop]["convergence_flag"] = flag_key
            properties[prop]["flag_value"] = flag_value
        if not prop_converged:
            unconverged.append(prop)

    return {
        "converged": len(unconverged) == 0,
        "unconverged": unconverged,
        "properties": properties,
    }


# ---------------------------------------------------------------------------
# Unit tests: convergence-flag scanning (pure, no LLM)
# ---------------------------------------------------------------------------

class TestCheckFlags:
    def test_all_converged(self):
        results = {
            "shear_viscosity": {
                "status": "success", "value": 0.89, "units": "mPa·s",
                "plateau_reached": True,
            },
        }
        out = _check_flags(results)
        assert out["converged"] is True
        assert out["unconverged"] == []
        assert out["properties"]["shear_viscosity"]["converged"] is True

    def test_one_unconverged(self):
        results = {
            "shear_viscosity": {
                "status": "success", "value": 0.89, "units": "mPa·s",
                "plateau_reached": False,
            },
        }
        out = _check_flags(results)
        assert out["converged"] is False
        assert out["unconverged"] == ["shear_viscosity"]
        assert out["properties"]["shear_viscosity"]["converged"] is False
        assert out["properties"]["shear_viscosity"]["convergence_flag"] == "plateau_reached"

    def test_no_flag_means_converged(self):
        results = {
            "band_gap": {
                "status": "success", "value": 1.1, "units": "eV",
            },
        }
        out = _check_flags(results)
        assert out["converged"] is True
        assert out["unconverged"] == []
        assert out["properties"]["band_gap"]["converged"] is True
        assert "convergence_flag" not in out["properties"]["band_gap"]

    def test_mixed_flag_names(self):
        results = {
            "shear_viscosity": {
                "status": "success", "value": 0.89, "units": "mPa·s",
                "plateau_reached": True,
            },
            "t1_time": {
                "status": "success", "value": 2.3, "units": "ps",
                "extreme_narrowing": False,
            },
        }
        out = _check_flags(results)
        assert out["converged"] is False
        assert out["unconverged"] == ["t1_time"]
        assert out["properties"]["shear_viscosity"]["converged"] is True
        assert out["properties"]["t1_time"]["converged"] is False

    def test_empty_results(self):
        out = _check_flags({})
        assert out["converged"] is True
        assert out["unconverged"] == []
        assert out["properties"] == {}

    def test_error_property_skipped(self):
        results = {
            "shear_viscosity": {
                "status": "error", "message": "script crashed",
            },
            "diffusion": {
                "status": "success", "value": 2.3e-9, "units": "m²/s",
                "converged": True,
            },
        }
        out = _check_flags(results)
        assert out["converged"] is True
        assert out["unconverged"] == []
        assert out["properties"]["shear_viscosity"]["status"] == "error"

    def test_multiple_unconverged(self):
        results = {
            "shear_viscosity": {
                "status": "success", "value": 0.89, "units": "mPa·s",
                "plateau_reached": False,
            },
            "diffusion": {
                "status": "success", "value": 2.3e-9, "units": "m²/s",
                "converged": False,
            },
        }
        out = _check_flags(results)
        assert out["converged"] is False
        assert set(out["unconverged"]) == {"shear_viscosity", "diffusion"}


# ---------------------------------------------------------------------------
# Tool integration tests (monkeypatched SimulationAnalysisAgent)
# ---------------------------------------------------------------------------

def _make_fake_orch(**overrides):
    """Create a minimal mock orchestrator with LLM config."""
    orch = MagicMock()
    orch.api_key = overrides.get("api_key", "test-key")
    orch.base_url = overrides.get("base_url", None)
    orch.model_name = overrides.get("model_name", "claude-opus-4-6")
    orch.base_dir = overrides.get("base_dir", "/tmp/test")
    orch.mp_api_key = None
    orch.futurehouse_api_key = None
    orch.hpc_connection = None
    orch.hpc_scheduler = None
    orch.generated_structures = []
    orch.default_calc_params = {}
    orch.routing_decision = {}
    orch.active_skill_and_domain = MagicMock(return_value=(None, None))
    return orch


def _build_tools(orch):
    """Build the tool registry; returns the tools instance."""
    from scilink.agents.sim_agents.simulation_orchestrator_tools import (
        SimulationOrchestratorTools,
    )
    return SimulationOrchestratorTools(orch)


class TestToolIntegration:
    def test_tool_registered(self):
        orch = _make_fake_orch()
        tools = _build_tools(orch)
        assert "check_observable_convergence" in tools.functions_map

    def test_tool_returns_converged(self, tmp_path):
        orch = _make_fake_orch()
        tools = _build_tools(orch)

        analysis_result = {
            "status": "success",
            "results": {
                "shear_viscosity": {
                    "status": "success", "value": 0.89, "units": "mPa·s",
                    "plateau_reached": True,
                    "verification": {"plausible": True, "reasoning": "ok"},
                },
            },
            "skills_used": ["viscosity_greenkubo"],
            "data_kinds": ["thermo_log"],
        }

        with patch(
            "scilink.agents.sim_agents.simulation_analysis_agent"
            ".SimulationAnalysisAgent"
        ) as MockAgent:
            mock_instance = MagicMock()
            mock_instance.run_analysis.return_value = analysis_result
            MockAgent.return_value = mock_instance

            raw = tools.functions_map["check_observable_convergence"](
                output_dir=str(tmp_path), research_goal="shear viscosity",
            )

        out = json.loads(raw)
        assert out["status"] == "success"
        assert out["converged"] is True
        assert out["unconverged"] == []
        assert out["properties"]["shear_viscosity"]["converged"] is True

    def test_tool_returns_unconverged(self, tmp_path):
        orch = _make_fake_orch()
        tools = _build_tools(orch)

        analysis_result = {
            "status": "success",
            "results": {
                "shear_viscosity": {
                    "status": "success", "value": 0.89, "units": "mPa·s",
                    "plateau_reached": False,
                    "verification": {"plausible": False,
                                     "reasoning": "not converged"},
                },
            },
            "skills_used": ["viscosity_greenkubo"],
            "data_kinds": ["thermo_log"],
        }

        with patch(
            "scilink.agents.sim_agents.simulation_analysis_agent"
            ".SimulationAnalysisAgent"
        ) as MockAgent:
            mock_instance = MagicMock()
            mock_instance.run_analysis.return_value = analysis_result
            MockAgent.return_value = mock_instance

            raw = tools.functions_map["check_observable_convergence"](
                output_dir=str(tmp_path), research_goal="shear viscosity",
            )

        out = json.loads(raw)
        assert out["status"] == "success"
        assert out["converged"] is False
        assert out["unconverged"] == ["shear_viscosity"]
        # Diagnostic-only: the tool reports the finding + per-property
        # evidence, and never prescribes a remedy.
        assert "recommendation" not in out
        assert out["properties"]["shear_viscosity"]["convergence_flag"] == "plateau_reached"

    def test_tool_with_dft_output_no_flags(self, tmp_path):
        orch = _make_fake_orch()
        tools = _build_tools(orch)

        analysis_result = {
            "status": "success",
            "results": {
                "band_gap": {
                    "status": "success", "value": 1.1, "units": "eV",
                    "verification": {"plausible": True, "reasoning": "ok"},
                },
            },
            "skills_used": ["band_structure"],
            "data_kinds": ["dft_output"],
        }

        with patch(
            "scilink.agents.sim_agents.simulation_analysis_agent"
            ".SimulationAnalysisAgent"
        ) as MockAgent:
            mock_instance = MagicMock()
            mock_instance.run_analysis.return_value = analysis_result
            MockAgent.return_value = mock_instance

            raw = tools.functions_map["check_observable_convergence"](
                output_dir=str(tmp_path), research_goal="band gap",
            )

        out = json.loads(raw)
        assert out["converged"] is True
        assert out["unconverged"] == []
        assert "recommendation" not in out

    def test_tool_handles_analysis_error(self, tmp_path):
        orch = _make_fake_orch()
        tools = _build_tools(orch)

        with patch(
            "scilink.agents.sim_agents.simulation_analysis_agent"
            ".SimulationAnalysisAgent"
        ) as MockAgent:
            mock_instance = MagicMock()
            mock_instance.run_analysis.return_value = {
                "status": "error",
                "message": "no recognized output",
                "results": {},
            }
            MockAgent.return_value = mock_instance

            raw = tools.functions_map["check_observable_convergence"](
                output_dir=str(tmp_path), research_goal="viscosity",
            )

        out = json.loads(raw)
        assert out["status"] == "error"

    def test_tool_handles_exception(self, tmp_path):
        orch = _make_fake_orch()
        tools = _build_tools(orch)

        with patch(
            "scilink.agents.sim_agents.simulation_analysis_agent"
            ".SimulationAnalysisAgent"
        ) as MockAgent:
            MockAgent.side_effect = RuntimeError("boom")

            raw = tools.functions_map["check_observable_convergence"](
                output_dir=str(tmp_path), research_goal="viscosity",
            )

        out = json.loads(raw)
        assert out["status"] == "error"
        assert "boom" in out["message"]
