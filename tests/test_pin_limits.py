"""Pins that are not failures (#761, the corrective part), and the saved
fit's space.

- A pure lineshape limit is not a pin: a mixing fraction at exactly 0 or 1,
  recognised by its [0, 1] bounds as well as by its name, and one width of a
  two-width lineshape at its floor while the other width carries the line.
  A collapsed single width, an amplitude capped at 1 and a real floor still
  pin.
- A centre held at an end of the measured axis is a band peaking outside the
  range: reported with no value, like a secondary pin, and the fit of the
  spectrum is not called degenerate.
- ``fit.npy`` is saved in the same space as ``data.npy``, baseline included
  (the recomputed R² and the residual diagnostics read it against the data)."""

import json
import logging
from pathlib import Path

import numpy as np

from scilink.skills._shared.curve_fitting_tools import validate_bound_pinning
from scilink.agents.exp_agents.controllers import curve_fitting_controllers as cc


def _pins(params, bounds):
    return [(p["component"], p["parameter"]) for p in validate_bound_pinning(params, bounds)]


def test_pure_lineshape_limits_are_not_pins_and_real_ones_still_are():
    assert _pins({"p": {"fL": 0.9999, "mix": 0.0}}, {"p": {"fL": [0, 1], "mix": [0, 1]}}) == []
    assert _pins({"p": {"amplitude": 1.0}}, {"p": {"amplitude": [0, 1]}}) == [("p", "amplitude")]   # a real ceiling
    voigt = {"p": {"center": 500.0, "sigma": 1e-12, "gamma": 6.0}}
    assert _pins(voigt, {"p": {"sigma": [1e-12, 50], "gamma": [0.1, 50]}}) == []          # a pure Lorentzian
    spike = {"p": {"center": 500.0, "fwhm": 1e-3}}
    assert _pins(spike, {"p": {"fwhm": [1e-3, 50]}}) == [("p", "fwhm")]                    # one width, collapsed
    floor = {"p": {"sigma": 2.5, "gamma": 10.0}}
    assert _pins(floor, {"p": {"sigma": [2.5, 50], "gamma": [0.1, 50]}}) == [("p", "sigma")]  # a real floor
    assert _pins({"p": {"k": 1.0000001e-06}}, {"p": {"k": [1e-06, 50.0]}}) == [("p", "k")]   # a floor, not a width


def test_a_centre_at_the_end_of_the_axis_is_beyond_the_range():
    stats = {"x_range": [374.1, 4000.0]}
    pins = [{"component": "edge", "parameter": "center", "value": 374.1, "bound": 374.1, "side": "lower"},
            {"component": "b2", "parameter": "center", "value": 700.0, "bound": 700.0, "side": "upper"},
            {"component": "b3", "parameter": "fwhm", "value": 4000.0, "bound": 4000.0, "side": "upper"}]
    keep, beyond = cc._beyond_axis(pins, stats)
    assert [p["component"] for p in keep] == ["b2", "b3"] and [p["component"] for p in beyond] == ["edge"]
    assert cc._beyond_axis(pins, {}) == (pins, [])


X = np.linspace(374.1, 4000.0, 400)


def _controller(tmp_path, corrections):
    c = cc.UnifiedSeriesProcessingController.__new__(cc.UnifiedSeriesProcessingController)
    c.logger = logging.getLogger("t761"); c.output_dir = Path(tmp_path); c.executor = object()
    c._extract_extra_operands = lambda state, p: None
    c._extra_operand_block = lambda state: ""
    c._compute_statistics = lambda cd: {"n_points": X.size, "x_range": [float(X[0]), float(X[-1])]}
    c._should_escalate_timeout_model = lambda *a, **k: False
    c._correct_script = lambda state, script, err: (corrections.append(err) or (script + "\n# fixed", "x"))
    return c


def test_through_the_fit_loop_an_edge_centre_is_a_caveat_not_a_degenerate_fit(tmp_path, monkeypatch):
    params = {"edge": {"center": float(X[0]), "amplitude": 0.4, "fwhm": 120.0},
              "main": {"center": 1400.0, "amplitude": 0.8, "fwhm": 60.0}}
    bounds = {"edge": {"center": [float(X[0]), 500.0]}, "main": {"center": [1300.0, 1500.0]}}
    run = {"status": "success", "visualization_path": "viz.png", "visualization_bytes": b"", "exec": {},
           "stdout": "FIT_RESULTS_JSON:" + json.dumps({"model_type": "2PV", "parameters": params, "bounds": bounds,
                                                        "fit_quality": {"r_squared": 0.99}})}
    monkeypatch.setattr(cc, "stage_and_run_adaptive", lambda *a, **k: run)
    corrections = []
    res = _controller(tmp_path, corrections)._fit_single_spectrum(
        {}, np.column_stack([X, np.ones_like(X)]), "s.csv", "s", 0, base_script="first")
    assert res["success"] and "pinned_at_bound" not in res and corrections == []
    assert [p["component"] for p in res["secondary_pins"]] == ["edge"]
    assert res["parameters"]["edge"]["center"] is None                       # not a measurement
    assert res["parameters"]["main"]["center"] == 1400.0


def test_the_fit_is_saved_in_the_datas_space():
    from scilink.agents.exp_agents import instruct as I
    assert "add back any baseline or background you subtracted" in I.FITTING_SCRIPT_INSTRUCTIONS
    assert "baseline included" in I.FITTING_SCRIPT_CORRECTION_INSTRUCTIONS
