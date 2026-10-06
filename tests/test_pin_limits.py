"""Pins that are not failures (#761, the corrective part), and the saved
fit's space.

- A pure lineshape limit is not a pin: a mixing fraction at exactly 0 or 1,
  recognised by its [0, 1] bounds as well as by its name, and one width of a
  two-width lineshape at its floor while the other width carries the line.
  A collapsed single width, an amplitude capped at 1 and a real floor still
  pin.
- A centre held at an end of the measured axis is a band peaking outside the
  range: NOT MEASURED. The whole component is reported with no value (half a
  profile measures nothing), target or not, and the other bands stand; only a
  fit with no declared target left measured stays a pin.
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
            {"component": "b3", "parameter": "fwhm", "value": 4000.0, "bound": 4000.0, "side": "upper"},
            # a bound widened past the data: the centre is outside the axis
            {"component": "past", "parameter": "center", "value": 365.0, "bound": 365.0, "side": "lower"}]
    bounds = {"past": {"center": [365.0, 500.0]}}                            # its window reaches into the data
    keep, beyond = cc._beyond_axis(pins, stats, bounds)
    assert [p["component"] for p in keep] == ["b2", "b3"] and [p["component"] for p in beyond] == ["edge", "past"]
    # a bound past the data on a parameter that is not a position on the axis is a failed fit
    # (#764 re-review): a relative shift, a log-space mu, an offset
    for par, rng in (("pos_shift", [-5.0, 5.0]), ("mu", [-3.0, 3.0]), ("x0", [-1.0, 1.0])):
        pin = {"component": "b", "parameter": par, "value": rng[0], "bound": rng[0], "side": "lower"}
        assert cc._beyond_axis([pin], stats, {"b": {par: rng}}) == ([pin], []), par
    far = [{"component": "past", "parameter": "center", "value": 300.0, "bound": 300.0, "side": "lower"}]
    assert cc._beyond_axis(far, stats) == (far, [])                          # no bounds: only at the axis end
    assert cc._beyond_axis(far, stats, {"past": {"center": [300.0, 500.0]}})[1] == [dict(far[0], reason="centre beyond the measured axis")]
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


def _edge_run(tmp_path, monkeypatch, targets, edge_extra=None, edge_bounds=None, only_edge=False):
    params = {"edge": {"center": float(X[0]), "center_err": 0.3, "amplitude": 0.4, "fwhm": 120.0, **(edge_extra or {})},
              "main": {"center": 1400.0, "amplitude": 0.8, "fwhm": 60.0}}
    bounds = {"edge": {"center": [float(X[0]), 500.0], **(edge_bounds or {})}, "main": {"center": [1300.0, 1500.0]}}
    if only_edge:
        params.pop("main"); bounds.pop("main")
    fr = {"model_type": "2PV", "parameters": params, "bounds": bounds, "fit_quality": {"r_squared": 0.99}}
    if targets is not None:
        fr["targets"] = targets
    run = {"status": "success", "visualization_path": "viz.png", "visualization_bytes": b"", "exec": {},
           "stdout": "FIT_RESULTS_JSON:" + json.dumps(fr)}
    monkeypatch.setattr(cc, "stage_and_run_adaptive", lambda *a, **k: run)
    corrections = []
    res = _controller(tmp_path, corrections)._fit_single_spectrum(
        {}, np.column_stack([X, np.ones_like(X)]), "s.csv", "s", 0, base_script="first")
    return res, corrections


def test_an_edge_band_is_not_measured_and_the_other_bands_stand(tmp_path, monkeypatch):
    """A band peaking beyond the axis is reported as not measured, every one
    of its values empty, whether or not it is a declared target; the fit of
    the other bands stands and nothing is repaired."""
    for targets in (None, ["edge", "main"], ["main"]):
        res, corrections = _edge_run(tmp_path, monkeypatch, targets)
        assert res["success"] and "pinned_at_bound" not in res and corrections == [], targets
        assert [p["component"] for p in res["not_measured"]] == ["edge"], targets
        assert "secondary_pins" not in res, targets
        assert all(v is None for v in res["parameters"]["edge"].values()), targets
        assert res["parameters"]["main"] == {"center": 1400.0, "amplitude": 0.8, "fwhm": 60.0}, targets
        assert any(c.startswith("Not measured: edge") for c in res["caveats"]), targets


def test_a_not_measured_bands_other_pins_go_with_it(tmp_path, monkeypatch):
    res, _ = _edge_run(tmp_path, monkeypatch, ["edge", "main"],
                       edge_extra={"fwhm": 300.0}, edge_bounds={"fwhm": [1.0, 300.0]})
    assert "pinned_at_bound" not in res and [p["component"] for p in res["not_measured"]] == ["edge"]


def test_a_fit_whose_every_target_is_beyond_the_axis_stays_a_pin(tmp_path, monkeypatch):
    """Nothing the plan asked for was measured: not a verified fit. The
    targets are read as split_pins_by_targets reads them: declared names that
    are fitted components, else every fitted component (#764 re-review)."""
    res, corrections = _edge_run(tmp_path, monkeypatch, ["edge"])
    assert corrections == []                                                 # no repair can move it into the data
    assert [p["component"] for p in res["pinned_at_bound"]] == ["edge"]
    assert "not_measured" not in res
    # a declared name that is no fitted component does not count as measured
    res, _ = _edge_run(tmp_path, monkeypatch, ["edge", "edge_area_ratio"])
    assert [p["component"] for p in res["pinned_at_bound"]] == ["edge"] and "not_measured" not in res
    # the only band beyond the axis, with typo-only targets or none
    for targets in (["D band"], None):
        res, _ = _edge_run(tmp_path, monkeypatch, targets, only_edge=True)
        assert [p["component"] for p in res["pinned_at_bound"]] == ["edge"], targets
        assert "not_measured" not in res, targets


def test_an_off_axis_parameter_railed_outside_the_axis_is_still_repaired(tmp_path, monkeypatch):
    """Through the fit loop: a relative shift, a log-space mu and an offset at
    a bound outside the axis pin and are repaired, as before (#764 re-review)."""
    for par, rng in (("pos_shift", [-5.0, 5.0]), ("mu", [-3.0, 3.0]), ("x0", [-1.0, 1.0])):
        res, corrections = _edge_run(tmp_path, monkeypatch, None, edge_extra={"center": 600.0, par: rng[0]},
                                     edge_bounds={"center": [500.0, 700.0], par: rng})
        assert corrections, par                                              # repaired
        assert [(p["component"], p["parameter"]) for p in res["pinned_at_bound"]] == [("edge", par)], par
        assert "not_measured" not in res, par


def test_the_single_spectrum_synthesis_is_told_why_a_value_is_empty():
    sent = {}

    class Model:
        def generate_content(self, contents, **kw):
            sent["prompt"] = "\n".join(c for c in contents if isinstance(c, str))
            raise RuntimeError("stop after the prompt")

    syn = cc.UnifiedCurveSynthesisController(Model(), logging.getLogger("t764"), None, None,
                                             lambda r: ({}, None), "", "out")
    caveat = "Not measured: edge (centre held at the axis end 374.1). Those bands peak beyond the measured axis."
    state = {"original_plot_bytes": b"", "fit_results": {"parameters": {"edge": {"center": None}}, "fit_quality": {}},
             "series_results": [{"caveats": [caveat]}], "system_info": {}}
    try:
        syn._synthesize_single_spectrum(state)
    except Exception:
        pass
    assert "## Fit caveats" in sent["prompt"] and caveat in sent["prompt"]


def test_a_follower_with_a_band_beyond_the_axis_verifies_with_a_caveat_flag(tmp_path, monkeypatch):
    """Through the series: the unit carries a non-refit `not_measured` flag,
    is not relaxed and refit, and verifies on the bands it measured."""
    import test_series_verdict_path as sv
    from scilink.agents.exp_agents._verification_record import analysis_verdict

    class EdgeExecutor(sv.FakeExecutor):
        def execute_script(self, script, working_dir=None, timeout=None, **kw):
            res = super().execute_script(script, working_dir=working_dir, timeout=timeout, **kw)
            name = self.calls[-1][0] if self.calls else None
            if res.get("status") != "success":
                return res
            out = json.loads(res["stdout"].split("FIT_RESULTS_JSON:", 1)[1])
            lo = float(sv.X[0])
            out["parameters"]["edge"] = {"center": lo if name == "spectrum_0001" else 160.0, "amplitude": 0.3, "fwhm": 50.0}
            out["bounds"] = {"edge": {"center": [lo, 300.0]}, "peak_1": {"fwhm": [1.0, 40.0]}}
            out["targets"] = ["peak_1", "edge"]
            res["stdout"] = "FIT_RESULTS_JSON:" + json.dumps(out)
            return res

    follower_r2 = {"spectrum_0001": 0.97, "spectrum_0002": 0.96}
    ex = EdgeExecutor(follower_r2)
    state, _ = sv.run_series(tmp_path, monkeypatch, names=sv.NAMES, anchors={"spectrum_0000": sv.OK},
                             follower_r2=follower_r2, executor=ex)
    raw = {u["name"]: u for u in state["series_results"]}
    results = sv.compile_results(tmp_path, state)
    by_name = {u["name"]: u for u in results["individual_results"]}
    flags = {f["name"]: f["reason"] for f in (state.get("flagged_spectra") or [])}
    row = raw["spectrum_0001"]
    assert [n for n, _ in ex.calls].count("spectrum_0001") == 1             # nothing relaxed or refit
    assert not row.get("pinned_at_bound") and row["not_measured"][0]["component"] == "edge"
    assert all(v is None for v in row["parameters"]["edge"].values())
    assert row["parameters"]["peak_1"]["center"] == 144.0                    # the other bands stand
    assert flags.get("spectrum_0001") == "not_measured"
    assert by_name["spectrum_0001"]["unit_verdict"]["verified"] is True
    assert analysis_verdict(results)["verified"] is True


def test_the_fit_is_saved_in_the_datas_space():
    from scilink.agents.exp_agents import instruct as I
    assert "add back any baseline or background you subtracted" in I.FITTING_SCRIPT_INSTRUCTIONS
    assert "baseline included" in I.FITTING_SCRIPT_CORRECTION_INSTRUCTIONS
