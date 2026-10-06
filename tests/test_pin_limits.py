"""Pins that are not failures (#761, the corrective part), and the saved
fit's space.

- A pure lineshape limit is not a pin: a mixing fraction at exactly 0 or 1,
  recognised by its [0, 1] bounds as well as by its name, and one width of a
  two-width lineshape at its floor while the other width carries the line.
  A collapsed single width, an amplitude capped at 1 and a real floor still
  pin.
- A centre held at an end of the measured axis is a band peaking outside the
  range. With no targets declared, the whole component is reported with no
  value (half a profile measures nothing), like a secondary pin, and the fit
  is not called degenerate; a declared target stays a pin, as on main.
- The lineshape exemptions are the curve agent's (opt-in): the hyperspectral
  scalar check keeps main's rule.
- ``fit.npy`` is saved in the same space as ``data.npy``, baseline included
  (the recomputed R² and the residual diagnostics read it against the data)."""

import json
import logging
from pathlib import Path

import numpy as np

from scilink.skills._shared.curve_fitting_tools import validate_bound_pinning
from scilink.agents.exp_agents.controllers import curve_fitting_controllers as cc


def _pins(params, bounds):
    return [(p["component"], p["parameter"]) for p in validate_bound_pinning(params, bounds, lineshape_limits=True)]


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
    # a Gaussian/Lorentzian pair of ONE line, by name: these are pure lineshapes
    assert _pins({"p": {"fwhm_g": 1e-6, "fwhm_l": 8.0}}, {"p": {"fwhm_g": [1e-6, 50], "fwhm_l": [0.1, 50]}}) == []
    assert _pins({"c": {"p1_sigma": 1e-6, "p1_gamma": 6.0}}, {"c": {"p1_sigma": [1e-6, 50], "p1_gamma": [0.1, 50]}}) == []
    # ... and these are not (#764 review): another peak's width, the other side of an asymmetric
    # profile, a centre on a normalised axis, a floor that is not small beside the carrying width
    assert _pins({"c": {"p1_sigma": 1e-3, "p2_gamma": 6.0}}, {"c": {"p1_sigma": [1e-3, 50], "p2_gamma": [0.1, 50]}}) \
        == [("c", "p1_sigma")]
    assert _pins({"p": {"sigma_left": 1e-3, "sigma_right": 8.0}},
                 {"p": {"sigma_left": [1e-3, 50], "sigma_right": [0.1, 50]}}) == [("p", "sigma_left")]
    assert _pins({"p": {"center": 1.0}}, {"p": {"center": [0, 1]}}) == [("p", "center")]
    assert _pins({"p": {"sigma": 2.5, "gamma": 30.0}}, {"p": {"sigma": [2.5, 50], "gamma": [0.1, 50]}}) == [("p", "sigma")]
    # a skewness railed at a negative bound is not a width at zero (#764 re-review)
    assert _pins({"p": {"gamma": -10.0, "sigma": 150.0}}, {"p": {"gamma": [-10, 10], "sigma": [1, 500]}}) == [("p", "gamma")]
    assert _pins({"p": {"gamma": -1.0, "sigma": 50.0}}, {"p": {"gamma": [-1, 1], "sigma": [1, 500]}}) == [("p", "gamma")]


def test_the_lineshape_exemptions_are_curve_only():
    """The shared rule is unchanged for the hyperspectral scalar check: a
    [0, 1] scalar at 1 is still a failed fit there (#764 review)."""
    from scilink.agents.exp_agents.controllers import hyperspectral_controllers as hc
    verdict, why = hc._check_scalar(1.0, "fraction", [0, 1], {})
    assert verdict == "failed" and "upper bound" in why
    assert validate_bound_pinning({"p": {"fL": 1.0}}, {"p": {"fL": [0, 1]}}) != []     # off by default


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
    # none of the edge band's values is a measurement: its width and area come from half a profile
    assert all(v is None for v in res["parameters"]["edge"].values())
    assert res["parameters"]["main"]["center"] == 1400.0


def test_a_declared_targets_edge_centre_stays_a_pin(tmp_path, monkeypatch):
    """The plan asked for that band and its maximum is not in the data: a
    degenerate fit, as on main (#742; #764 review). Not repaired in the loop."""
    params = {"edge": {"center": float(X[0]), "amplitude": 0.4, "fwhm": 120.0},
              "main": {"center": 1400.0, "amplitude": 0.8, "fwhm": 60.0}}
    bounds = {"edge": {"center": [float(X[0]), 500.0]}, "main": {"center": [1300.0, 1500.0]}}
    run = {"status": "success", "visualization_path": "viz.png", "visualization_bytes": b"", "exec": {},
           "stdout": "FIT_RESULTS_JSON:" + json.dumps({"model_type": "2PV", "parameters": params, "bounds": bounds,
                                                        "targets": ["edge"], "fit_quality": {"r_squared": 0.99}})}
    monkeypatch.setattr(cc, "stage_and_run_adaptive", lambda *a, **k: run)
    corrections = []
    res = _controller(tmp_path, corrections)._fit_single_spectrum(
        {}, np.column_stack([X, np.ones_like(X)]), "s.csv", "s", 0, base_script="first")
    assert corrections == []                                                 # no repair can move it into the data
    assert [p["component"] for p in res["pinned_at_bound"]] == ["edge"]


def test_a_declared_non_target_edge_band_is_emptied_too(tmp_path, monkeypatch):
    """Targets declared, the edge band not among them: a secondary pin, and
    none of its values is a measurement, as with no targets (#764 re-review)."""
    params = {"edge": {"center": float(X[0]), "amplitude": 0.4, "fwhm": 120.0},
              "main": {"center": 1400.0, "amplitude": 0.8, "fwhm": 60.0}}
    bounds = {"edge": {"center": [float(X[0]), 500.0]}, "main": {"center": [1300.0, 1500.0]}}
    run = {"status": "success", "visualization_path": "viz.png", "visualization_bytes": b"", "exec": {},
           "stdout": "FIT_RESULTS_JSON:" + json.dumps({"model_type": "2PV", "parameters": params, "bounds": bounds,
                                                        "targets": ["main"], "fit_quality": {"r_squared": 0.99}})}
    monkeypatch.setattr(cc, "stage_and_run_adaptive", lambda *a, **k: run)
    res = _controller(tmp_path, [])._fit_single_spectrum(
        {}, np.column_stack([X, np.ones_like(X)]), "s.csv", "s", 0, base_script="first")
    assert res["success"] and "pinned_at_bound" not in res
    assert all(v is None for v in res["parameters"]["edge"].values())
    assert res["parameters"]["main"] == {"center": 1400.0, "amplitude": 0.8, "fwhm": 60.0}


def test_the_fit_is_saved_in_the_datas_space():
    from scilink.agents.exp_agents import instruct as I
    assert "add back any baseline or background you subtracted" in I.FITTING_SCRIPT_INSTRUCTIONS
    assert "baseline included" in I.FITTING_SCRIPT_CORRECTION_INSTRUCTIONS
