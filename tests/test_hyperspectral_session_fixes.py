"""Regression tests for the four grid-STS session defects (2026-07-22).

1. METHOD WIRING — the LLM-selected decomposition method is written to the
   per-run ``state["settings"]`` (a copy), but the unmixing controllers read
   their constructor settings dict, so PCA/ICA selection never reached the
   unmixer (labels said PCA, sklearn ran NMF) since 96de3d18.
2. AXIS PRECONDITION — the deliberately-required physical axis-2 range
   (fcb77007) was enforced only at interpretation/plot time, after
   preprocessing + the full component test loop had burned compute.
3. SPECTRAL-IMAGING HEURISTIC — grid spectroscopy (STS) prose matched none
   of the datacube keywords, so schema conformance never demanded the axis.
4. SNR GUARDS — absolute 1e-9 epsilons (meant as division guards) declared
   amperes-scale data "signalless" (same class as the #381 mask bug).
"""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from scilink.agents.exp_agents.controllers import hyperspectral_controllers as hc
from scilink.agents.exp_agents.metadata_converter import (
    _describes_spectral_imaging,
    check_schema_conformance,
)
from scilink.agents.exp_agents.preprocess import HyperspectralPreprocessingAgent


LOGGER = logging.getLogger("test")

AXIS_OK = {
    "axis_spec": {
        "axis_2": {"name": "Bias", "units": "V", "start": 1.0, "end": -1.0},
        "signal_is_nonnegative": False,
    }
}


# ---------------------------------------------------------------------------
# 1. Method wiring: state settings (LLM choice) reach the unmixer
# ---------------------------------------------------------------------------

@pytest.fixture
def recorded_methods(monkeypatch):
    """Stub the unmixing tool and record the method each call receives."""
    seen = []

    def fake_unmix(hspy_data, n_comp, settings, logger):
        seen.append(settings.get("method"))
        h, w, e = hspy_data.shape
        return np.zeros((n_comp, e)), np.zeros((h, w, n_comp)), 1.0

    monkeypatch.setattr(hc.tools, "run_spectral_unmixing", fake_unmix)
    monkeypatch.setattr(
        hc.tools, "create_nmf_summary_plot",
        lambda *a, **k: b"img",
    )
    return seen


def _constructor_settings():
    # What the agent passes at pipeline construction: default method.
    return {"method": "nmf", "min_auto_components": 2, "max_auto_components": 3,
            "output_dir": "unused"}


def _state_with_selected_method(method="pca"):
    # What GetInitialComponentParamsController produces at runtime.
    return {
        "hspy_data": np.random.rand(4, 4, 8),
        "system_info": dict(AXIS_OK),
        "settings": {"method": method, "normalize": True},
        "initial_n_components": 2,
        "iteration_title": "t",
    }


def test_component_test_loop_uses_llm_selected_method(recorded_methods, tmp_path):
    ctrl = hc.RunComponentTestLoopController(LOGGER, _constructor_settings())
    ctrl.settings["output_dir"] = str(tmp_path)
    state = ctrl.execute(_state_with_selected_method("pca"))
    assert recorded_methods and set(recorded_methods) == {"pca"}
    assert state["component_test_errors"]


def test_final_unmixing_uses_llm_selected_method(recorded_methods):
    ctrl = hc.RunFinalSpectralUnmixingController(LOGGER, _constructor_settings())
    state = _state_with_selected_method("pca")
    state["final_n_components"] = 2
    ctrl.execute(state)
    assert recorded_methods == ["pca"]


def test_controllers_fall_back_to_constructor_settings(recorded_methods):
    # A state without "settings" (defensive path) still works.
    ctrl = hc.RunFinalSpectralUnmixingController(LOGGER, _constructor_settings())
    state = _state_with_selected_method()
    del state["settings"]
    state["final_n_components"] = 2
    ctrl.execute(state)
    assert recorded_methods == ["nmf"]


# ---------------------------------------------------------------------------
# 2. Axis precondition: fail before compute, not after decomposition
# ---------------------------------------------------------------------------

def _make_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    from scilink.agents.exp_agents.hyperspectral_analysis_agent import (
        HyperspectralAnalysisAgent,
    )
    return HyperspectralAnalysisAgent(
        api_key="dummy", base_url="http://localhost:1",
        output_dir=str(tmp_path / "out"), enable_human_feedback=False,
    )


class _SentinelController:
    """Records whether the iteration pipeline was ever entered."""
    def __init__(self):
        self.entered = False

    def execute(self, state):
        self.entered = True
        state["error_dict"] = {"error": "sentinel-reached"}
        return state


def test_missing_axis_range_fails_before_pipeline(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path, monkeypatch)
    sentinel = _SentinelController()
    agent.iteration_pipeline = [sentinel]
    agent.synthesis_pipeline = []

    cube = tmp_path / "cube.npy"
    np.save(cube, np.random.rand(4, 4, 8))

    result_json, error_dict = agent._run_analysis_pipeline(
        data_path=str(cube), system_info={}, instruction_prompt="x",
    )
    assert result_json is None
    assert "Missing physical axis range" in error_dict["error"]
    assert "no analysis was performed" in error_dict["details"]
    assert sentinel.entered is False  # nothing was computed


def test_valid_axis_range_proceeds_into_pipeline(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path, monkeypatch)
    sentinel = _SentinelController()
    agent.iteration_pipeline = [sentinel]
    agent.synthesis_pipeline = []

    cube = tmp_path / "cube.npy"
    np.save(cube, np.random.rand(4, 4, 8))

    _, error_dict = agent._run_analysis_pipeline(
        data_path=str(cube), system_info=dict(AXIS_OK), instruction_prompt="x",
    )
    assert sentinel.entered is True
    assert error_dict == {"error": "sentinel-reached"}


# ---------------------------------------------------------------------------
# 2b. Final-N selection proceeds with partial visual evidence
# ---------------------------------------------------------------------------

def _final_select_controller():
    model = SimpleNamespace(generate_content=lambda **k: "resp")
    parse_fn = lambda resp: ({"final_components": 3, "reasoning": "r"}, None)
    return hc.GetFinalComponentSelectionController(
        model, LOGGER, generation_config=None, safety_settings=None,
        parse_fn=parse_fn,
    )


def test_final_selection_runs_with_elbow_only():
    ctrl = _final_select_controller()
    state = {"initial_n_components": 5, "component_test_range": [2, 3, 4],
             "elbow_plot_bytes": b"elbow", "component_test_visuals": []}
    ctrl.execute(state)
    # Pre-fix: visuals missing => silent fallback to the initial estimate (5).
    assert state["final_n_components"] == 3


def test_final_selection_falls_back_when_no_evidence():
    ctrl = _final_select_controller()
    state = {"initial_n_components": 5, "component_test_range": [2, 3, 4],
             "elbow_plot_bytes": None, "component_test_visuals": []}
    ctrl.execute(state)
    assert state["final_n_components"] == 5


# ---------------------------------------------------------------------------
# 3. Spectral-imaging heuristic recognizes grid spectroscopy
# ---------------------------------------------------------------------------

def test_grid_spectroscopy_is_spectral_imaging():
    md = {"experiment_type": "Spectroscopy",
          "experiment": {"technique": "Grid Spectroscopy (STS)"},
          "sample": {"material": "TaS2"}}
    assert _describes_spectral_imaging(md)
    ok, issues = check_schema_conformance(md)
    assert not ok
    assert any("resolvable" in i for i in issues)


def test_plain_1d_spectroscopy_is_not_spectral_imaging():
    md = {"experiment_type": "Spectroscopy",
          "experiment": {"technique": "Raman spectroscopy"},
          "sample": {"material": "Si"}}
    assert not _describes_spectral_imaging(md)


# ---------------------------------------------------------------------------
# 4. SNR guards are scale-invariant
# ---------------------------------------------------------------------------

def _snr(stats):
    host = SimpleNamespace(logger=LOGGER)
    return HyperspectralPreprocessingAgent._calculate_snr(host, stats)


def test_snr_nonzero_on_amperes_scale_data():
    # The real session's scale: p50 ~2.5e-12 sat under the old 1e-9 floor
    # and reported "SNR is 0.0 (no variance or signal)".
    stats = {"p99": 1.42e-11, "p50": 2.47e-12, "std": 1.0e-12}
    snr, reasoning = _snr(stats)
    assert snr > 1.0
    assert "no variance" not in reasoning


def test_snr_is_scale_invariant():
    stats = {"p99": 1.42e-11, "p50": 2.47e-12, "std": 1.0e-12}
    scaled = {k: v * 1e13 for k, v in stats.items()}
    assert _snr(stats)[0] == pytest.approx(_snr(scaled)[0])


def test_snr_zero_only_for_constant_data():
    snr, reasoning = _snr({"p99": 0.0, "p50": 0.0, "std": 0.0})
    assert snr == 0.0
    assert "no variance" in reasoning
