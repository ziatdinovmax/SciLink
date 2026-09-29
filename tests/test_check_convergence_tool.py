"""Tests for the check_observable_convergence orchestrator tool.

Covers the strict three-state classifier (`_classify_convergence`) directly
and the tool end-to-end with a monkeypatched SimulationAnalysisAgent. The
classifier is the single source of truth — there is no separate mirror to
drift.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scilink.agents.sim_agents.simulation_orchestrator_tools import (  # noqa: E402
    _classify_convergence, _parse_flag_bool,
)


# ---------------------------------------------------------------------------
# _parse_flag_bool — strict boolean coercion of model-written flags
# ---------------------------------------------------------------------------

class TestParseFlagBool:
    def test_real_bools_pass_through(self):
        assert _parse_flag_bool(True) is True
        assert _parse_flag_bool(False) is False

    def test_bool_strings_any_case(self):
        assert _parse_flag_bool("true") is True
        assert _parse_flag_bool("True") is True
        assert _parse_flag_bool("FALSE") is False
        assert _parse_flag_bool("  false  ") is False

    def test_unparseable_is_none(self):
        # The classic bug: bool("False") is True. These must NOT coerce.
        for v in ("False?", "maybe", "yes", "", 0, 1, None, 0.0):
            assert _parse_flag_bool(v) is None, v


# ---------------------------------------------------------------------------
# _classify_convergence — converged / not_converged / not_assessed
# ---------------------------------------------------------------------------

class TestClassify:
    def _state(self, result):
        return _classify_convergence(result)["state"]

    def test_flag_true_converged(self):
        assert self._state({"value": 0.89, "plateau_reached": True}) == "converged"

    def test_flag_false_not_converged(self):
        assert self._state({"value": 0.89, "plateau_reached": False}) == "not_converged"

    def test_no_flag_no_verdict_not_assessed(self):
        assert self._state({"value": 1.1, "units": "eV"}) == "not_assessed"

    def test_no_flag_plausible_true_not_assessed(self):
        assert self._state({
            "value": 1.1, "verification": {"plausible": True},
        }) == "not_assessed"

    def test_string_true_flag_converged(self):
        assert self._state({"value": 1.0, "plateau_reached": "true"}) == "converged"

    # --- Maxim's review table (each row was reported as converged) ---

    def test_row1_no_flag_implausible_not_converged(self):
        # value:-0.4, no flag, verification.plausible:false
        assert self._state({
            "value": -0.4, "verification": {"plausible": False,
                                            "reasoning": "negative"},
        }) == "not_converged"

    def test_row2_string_false_flag_not_converged(self):
        # plateau_reached:"False" (a string) — bool("False") is True, so this
        # must be parsed strictly.
        assert self._state({"value": 0.5, "plateau_reached": "False"}) == "not_converged"

    def test_row3_mixed_flags_not_converged_deterministic(self):
        # converged:true + plateau_reached:false — every flag is read, so the
        # false one wins regardless of dict/hash order. Run repeatedly to make
        # the determinism explicit.
        result = {"value": 0.5, "converged": True, "plateau_reached": False}
        for _ in range(20):
            assert self._state(dict(result)) == "not_converged"

    # --- other edge cases ---

    def test_plausible_false_overrides_true_flag(self):
        assert self._state({
            "value": 0.5, "plateau_reached": True,
            "verification": {"plausible": False},
        }) == "not_converged"

    def test_unparseable_only_flag_not_assessed(self):
        assert self._state({"value": 0.5, "plateau_reached": "maybe"}) == "not_assessed"

    def test_true_plus_unparseable_not_assessed(self):
        # A true flag can't be trusted when another present flag is unreadable.
        assert self._state({
            "value": 0.5, "converged": True, "plateau_reached": "maybe",
        }) == "not_assessed"

    def test_all_flags_true_converged(self):
        assert self._state({
            "value": 0.5, "converged": True, "plateau_reached": True,
        }) == "converged"

    def test_evidence_records_flags_and_verification(self):
        ev = _classify_convergence({
            "value": 0.89, "units": "mPa·s", "plateau_reached": False,
            "verification": {"plausible": False, "reasoning": "x"},
        })
        assert ev["value"] == 0.89 and ev["units"] == "mPa·s"
        assert ev["verification"]["reasoning"] == "x"
        assert ev["flags"]["plateau_reached"]["parsed"] is False
        assert ev["flags"]["plateau_reached"]["raw"] is False


# ---------------------------------------------------------------------------
# Tool integration tests (monkeypatched SimulationAnalysisAgent)
# ---------------------------------------------------------------------------

def _make_fake_orch(**overrides):
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
    from scilink.agents.sim_agents.simulation_orchestrator_tools import (
        SimulationOrchestratorTools,
    )
    return SimulationOrchestratorTools(orch)


def _run_tool(tmp_path, results, research_goal="viscosity"):
    """Invoke the tool with a monkeypatched analysis agent returning ``results``."""
    orch = _make_fake_orch()
    tools = _build_tools(orch)
    analysis_result = {
        "status": "success", "results": results,
        "skills_used": ["viscosity_greenkubo"], "data_kinds": ["thermo_log"],
    }
    with patch(
        "scilink.agents.sim_agents.simulation_analysis_agent"
        ".SimulationAnalysisAgent"
    ) as MockAgent:
        MockAgent.return_value.run_analysis.return_value = analysis_result
        raw = tools.functions_map["check_observable_convergence"](
            output_dir=str(tmp_path), research_goal=research_goal,
        )
    return json.loads(raw)


class TestToolIntegration:
    def test_tool_registered(self):
        tools = _build_tools(_make_fake_orch())
        assert "check_observable_convergence" in tools.functions_map

    def test_tool_returns_converged(self, tmp_path):
        out = _run_tool(tmp_path, {
            "shear_viscosity": {
                "status": "success", "value": 0.89, "units": "mPa·s",
                "plateau_reached": True,
                "verification": {"plausible": True, "reasoning": "ok"},
            },
        })
        assert out["status"] == "success"
        assert out["converged"] is True
        assert out["unconverged"] == [] and out["not_assessed"] == []
        assert out["properties"]["shear_viscosity"]["state"] == "converged"

    def test_tool_returns_unconverged(self, tmp_path):
        out = _run_tool(tmp_path, {
            "shear_viscosity": {
                "status": "success", "value": 0.89, "units": "mPa·s",
                "plateau_reached": False,
                "verification": {"plausible": False, "reasoning": "no plateau"},
            },
        })
        assert out["converged"] is False
        assert out["unconverged"] == ["shear_viscosity"]
        assert out["properties"]["shear_viscosity"]["state"] == "not_converged"
        # Diagnostic-only: never prescribes a remedy.
        assert "recommendation" not in out

    def test_tool_row1_no_flag_implausible(self, tmp_path):
        out = _run_tool(tmp_path, {
            "shear_viscosity": {
                "status": "success", "value": -0.4, "units": "mPa·s",
                "verification": {"plausible": False, "reasoning": "negative"},
            },
        })
        assert out["converged"] is False
        assert out["unconverged"] == ["shear_viscosity"]
        assert out["properties"]["shear_viscosity"]["state"] == "not_converged"

    def test_tool_row2_string_false_flag(self, tmp_path):
        out = _run_tool(tmp_path, {
            "shear_viscosity": {
                "status": "success", "value": 0.5, "units": "mPa·s",
                "plateau_reached": "False",
            },
        })
        assert out["converged"] is False
        assert out["unconverged"] == ["shear_viscosity"]

    def test_tool_row3_mixed_flags(self, tmp_path):
        out = _run_tool(tmp_path, {
            "shear_viscosity": {
                "status": "success", "value": 0.5, "units": "mPa·s",
                "converged": True, "plateau_reached": False,
            },
        })
        assert out["converged"] is False
        assert out["unconverged"] == ["shear_viscosity"]

    def test_tool_dft_no_flags_not_assessed(self, tmp_path):
        out = _run_tool(tmp_path, {
            "band_gap": {
                "status": "success", "value": 1.1, "units": "eV",
                "verification": {"plausible": True, "reasoning": "ok"},
            },
        }, research_goal="band gap")
        # Nothing was found not-converged, but the property carried no
        # convergence signal, so it is surfaced as not_assessed — not silently
        # reported converged.
        assert out["converged"] is True
        assert out["unconverged"] == []
        assert out["not_assessed"] == ["band_gap"]
        assert out["properties"]["band_gap"]["state"] == "not_assessed"
        assert "recommendation" not in out

    def test_tool_error_property_surfaced(self, tmp_path):
        out = _run_tool(tmp_path, {
            "shear_viscosity": {"status": "error", "message": "script crashed"},
            "diffusion": {
                "status": "success", "value": 2.3e-9, "units": "m²/s",
                "converged": True,
            },
        })
        assert out["properties"]["shear_viscosity"]["state"] == "error"
        assert out["properties"]["diffusion"]["state"] == "converged"

    def test_tool_handles_analysis_error(self, tmp_path):
        orch = _make_fake_orch()
        tools = _build_tools(orch)
        with patch(
            "scilink.agents.sim_agents.simulation_analysis_agent"
            ".SimulationAnalysisAgent"
        ) as MockAgent:
            MockAgent.return_value.run_analysis.return_value = {
                "status": "error", "message": "no recognized output", "results": {},
            }
            raw = tools.functions_map["check_observable_convergence"](
                output_dir=str(tmp_path), research_goal="viscosity",
            )
        assert json.loads(raw)["status"] == "error"

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
        assert out["status"] == "error" and "boom" in out["message"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
