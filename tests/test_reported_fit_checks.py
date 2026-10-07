"""What a curve fit reports is checked against the fit and the data (#762).

- Code generation is told the layout of ``data.npy`` (from the array staged)
  and that a script replayed on other spectra must not choose a column, a
  window or anything else by matching one spectrum's values: in the
  generation, correction and conformance prompts and the bank's adapt
  contract.
- A saved fit that follows the x axis, or a replay further from the data
  than the data's own spread where its anchor was not below a flat line, is
  not a fit of this data: the ladder is told so, and a unit that still shows it fails (refit-eligible),
  never verified on its self-reported R². A saved fit below that line on its
  own is no error (a peaks-only fit saved without its baseline, a windowed
  model evaluated outside its window).
- A band whose REPORTED centre lies outside the measured axis is not
  measured; a fit left with no target measured is withheld, never failed.
- A band width wider than the whole axis and the uncertainties of a
  degenerate fit are reported as no value, with the reason.
"""

import json
import logging
from pathlib import Path

import numpy as np

from scilink.agents.exp_agents.controllers import curve_fitting_controllers as cc
from scilink.agents.exp_agents._verification_record import analysis_verdict

X = np.linspace(374.1, 4000.0, 400)
Y = np.exp(-0.5 * ((X - 1400.0) / 30.0) ** 2) + 0.2


# -- the contract ---------------------------------------------------------

def test_the_layout_is_stated_from_the_array_staged():
    assert "column 0 is x, column 1 is y" in cc._data_layout(np.zeros((50, 2)))
    assert "row 0 is x, row 1 is y" in cc._data_layout(np.zeros((2, 50)))
    assert "x is the point index" in cc._data_layout(np.zeros(50))
    # the same rule as the arrays are read by
    for a in (np.zeros((50, 2)), np.zeros((2, 50)), np.zeros(50)):
        assert cc._extract_xy(a) is not None


def test_the_rule_is_in_every_prompt_that_writes_or_judges_a_script():
    from types import SimpleNamespace
    from scilink.agents.exp_agents.instruct import (
        FITTING_SCRIPT_INSTRUCTIONS, FITTING_SCRIPT_CORRECTION_INSTRUCTIONS, PLAN_CONFORMANCE_CHECK_INSTRUCTIONS)
    prompts = []

    class _Model:
        def generate_content(self, contents, **kw):
            prompts.append(contents if isinstance(contents, str) else json.dumps(contents, default=str))
            return SimpleNamespace(text=json.dumps({"script": "np.load('data.npy')", "diagnosis": "d"}))
    ctrl = cc.UnifiedSeriesProcessingController(
        model=_Model(), logger=logging.getLogger("t762"), generation_config=None, safety_settings=None,
        parse_fn=lambda r: (json.loads(r.text), None), executor=object(),
        script_instructions=FITTING_SCRIPT_INSTRUCTIONS, correction_instructions=FITTING_SCRIPT_CORRECTION_INSTRUCTIONS,
        quality_instructions="", output_dir="/tmp", plot_fn=lambda d, i: b"p", r2_threshold=0.95, parallel_workers=1)
    helper = getattr(ctrl, "_fitting_helper", ctrl)
    state = {"locked_fitting_config": {"physical_model": "one Gaussian"}}
    stats = {"n_points": 50, "x_range": (0.0, 1.0), "y_range": (0.0, 1.0)}
    helper._generate_fitting_script(state, "data.npy", stats, data_layout=cc._data_layout(np.zeros((2, 50))))
    helper._generate_fitting_script(state, "data.npy", stats)                 # a caller that names no layout
    helper._correct_script(state, "print(0)", "boom")
    assert "Layout: shape (2, N): row 0 is x, row 1 is y" in prompts[0]
    assert "Layout: shape (N, 2): column 0 is x, column 1 is y" in prompts[1]
    for p in prompts[:2]:
        assert "never choose a column, a window or" in p and "describe this spectrum only" in p
    assert "never choosing a column or a window by matching one spectrum's values" in prompts[2]
    assert "choose a column\nor a window by matching one spectrum's own values" in PLAN_CONFORMANCE_CHECK_INSTRUCTIONS


# -- the saved fit is a fit of THIS data -----------------------------------

def test_a_saved_fit_of_the_x_axis_is_not_a_fit_of_the_data():
    data = np.column_stack([X, Y])
    why, r2 = cc._saved_fit_mismatch(data, X.copy())
    assert why and "follows the x axis" in why and r2 < 0
    # level means of x windows follow x too
    steps = np.where(X < 2000, X[X < 2000].mean(), X[X >= 2000].mean())
    assert "follows the x axis" in cc._saved_fit_mismatch(data, steps)[0]
    # a level far off the data that tracks nothing — a replay that read the
    # wrong thing — is caught by the replay rule against its anchor only
    level = np.full_like(Y, 60.0)
    assert cc._saved_fit_mismatch(data, level)[0] is None
    why, _ = cc._saved_fit_mismatch(data, level, anchor_r2=0.34)
    assert why and "on its anchor" in why
    # a level fitted to a flat, noisy control: just below zero where its
    # anchor was just above is no evidence (seen live)
    rng = np.random.default_rng(0)
    noise = np.column_stack([X, 0.5 * rng.standard_normal(X.size)])
    near = np.full(X.size, 0.2)
    _, r2 = cc._saved_fit_mismatch(noise, near)
    assert -1.0 < r2 < 0 and cc._saved_fit_mismatch(noise, near, anchor_r2=0.021)[0] is None
    # the (2, N) layout is read the same way
    assert "follows the x axis" in cc._saved_fit_mismatch(data.T, X.copy())[0]


def test_a_saved_fit_below_a_flat_line_alone_is_no_error():
    data = np.column_stack([X, Y])
    peaks_only = Y - 0.2 - 3.0                         # saved without its baseline, offset
    why, r2 = cc._saved_fit_mismatch(data, peaks_only)
    assert why is None and r2 < 0
    # a windowed recipe: its anchor scored below zero too, so the follower is held to nothing new
    assert cc._saved_fit_mismatch(data, peaks_only, anchor_r2=-4.0)[0] is None
    # a healthy fit, and a fit saved in descending order
    assert cc._saved_fit_mismatch(data, Y.copy(), anchor_r2=0.9)[0] is None
    assert cc._saved_fit_mismatch(np.column_stack([X[::-1], Y[::-1]]), Y.copy(), anchor_r2=0.9)[0] is None
    # a fit that cannot be aligned keeps the old path
    assert cc._saved_fit_mismatch(data, Y[:-3]) == (None, None)


class _FitWriter:
    """Prints the fit JSON and saves ``fit.npy`` as the script text says: a
    script containing ``AXIS`` saves the x axis as its fit."""
    timeout = 30

    def __init__(self):
        self.calls = []

    def execute_script(self, script, working_dir=None, timeout=None, **kw):
        wd = Path(working_dir)
        self.calls.append(script)
        d = np.load(wd / "data.npy")
        x, y = cc._extract_xy(d)
        np.save(wd / "fit.npy", np.asarray(x if "AXIS" in script else y, float))
        (wd / "visualization.png").write_bytes(b"png")
        out = {"model_type": "m", "fit_quality": {"r_squared": 0.99},
               "parameters": {"peak_1": {"center": 1400.0, "amplitude": 1.0, "fwhm": 70.0}}}
        return {"status": "success", "stdout": "FIT_RESULTS_JSON:" + json.dumps(out), "stderr": "", "message": ""}


def _ctrl(tmp_path, ex, corrected):
    import test_series_verdict_path as sv
    ctrl = sv._controller(tmp_path, ex)
    errors = []
    ctrl._correct_script = lambda state, script, err: (errors.append(err) or corrected, "fixed")
    ctrl._check_plan_conformance = lambda state, script: None
    ctrl._generate_fitting_script = lambda *a, **k: "AXIS np.load('data.npy')"
    return ctrl, errors


def test_the_ladder_is_told_and_a_repair_is_kept(tmp_path):
    ex = _FitWriter()
    ctrl, errors = _ctrl(tmp_path, ex, corrected="np.load('data.npy')[:, 1]  # fixed")
    res = ctrl._fit_single_spectrum(state={"system_info": {}}, curve_data=np.column_stack([X, Y]),
                                    data_path="d.npy", spectrum_name="spectrum_0000", spectrum_idx=0)
    assert res["success"] and res["script"] == "np.load('data.npy')[:, 1]  # fixed"
    assert len(errors) == 1 and errors[0].startswith("THE SAVED FIT IS NOT A FIT OF THIS DATA")
    assert "never by matching one spectrum's values" in errors[0]
    assert res["saved_fit_r2"] > 0.99


def test_a_unit_that_still_fits_the_axis_fails(tmp_path):
    ex = _FitWriter()
    ctrl, errors = _ctrl(tmp_path, ex, corrected="AXIS still")
    res = ctrl._fit_single_spectrum(state={"system_info": {}}, curve_data=np.column_stack([X, Y]),
                                    data_path="d.npy", spectrum_name="spectrum_0000", spectrum_idx=0)
    assert not res["success"] and res["kind"] == "saved_fit_mismatch"
    assert "follows the x axis" in res["error"]
    # a stale fit.npy from an earlier attempt is not read as this attempt's
    class NoFit(_FitWriter):
        def execute_script(self, script, working_dir=None, timeout=None, **kw):
            res = super().execute_script(script, working_dir=working_dir, timeout=timeout, **kw)
            (Path(working_dir) / "fit.npy").unlink()
            return res
    d = Path(tmp_path) / "b" / "spectrum_0000"
    d.mkdir(parents=True)
    np.save(d / "fit.npy", X)
    import os
    os.utime(d / "fit.npy", (1, 1))
    _nofit = NoFit()
    _real = _nofit.execute_script

    def _keep_stale(script, working_dir=None, **k):
        out = _real(script, working_dir=working_dir, **k)
        np.save(Path(working_dir) / "fit.npy", X)
        os.utime(Path(working_dir) / "fit.npy", (1, 1))
        return out
    _nofit.execute_script = _keep_stale
    ctrl2, errors2 = _ctrl(tmp_path / "b", _nofit, corrected="fixed")
    res = ctrl2._fit_single_spectrum(state={"system_info": {}}, curve_data=np.column_stack([X, Y]),
                                     data_path="d.npy", spectrum_name="spectrum_0000", spectrum_idx=0,
                                     base_script="no fit saved")
    assert res["success"] and errors2 == []


def test_a_follower_whose_replay_does_not_fit_its_data_is_failed_not_verified(tmp_path, monkeypatch):
    """Through the series: the anchor's saved-fit R² reaches the follower,
    whose replay lies far from its own data; it fails, is
    refit-eligible, and the series is not verified on its self-report."""
    import test_series_verdict_path as sv
    real_canned = sv._canned_anchor
    monkeypatch.setattr(sv, "_canned_anchor", lambda *a, **k: {**real_canned(*a, **k), "saved_fit_r2": 0.34})

    class Ex(sv.FakeExecutor):
        def execute_script(self, script, working_dir=None, timeout=None, **kw):
            res = super().execute_script(script, working_dir=working_dir, timeout=timeout, **kw)
            wd = Path(working_dir)
            y = cc._extract_xy(np.load(wd / "data.npy"))[1]
            bad = "spectrum_0002" in wd.as_posix()
            np.save(wd / "fit.npy", np.full_like(y, 60.0) if bad else y)   # a level far off this unit's data
            return res

    follower_r2 = {"spectrum_0001": 0.97, "spectrum_0002": 0.97, "spectrum_0003": 0.97}
    ex = Ex(follower_r2)
    state, _ = sv.run_series(tmp_path, monkeypatch, names=sv.NAMES + ["spectrum_0003"],
                             anchors={"spectrum_0000": sv.OK}, follower_r2=follower_r2, executor=ex)
    raw = {u["name"]: u for u in state["series_results"]}
    assert raw["spectrum_0001"]["success"] and raw["spectrum_0001"]["unit_verdict"]["verified"]
    bad = raw["spectrum_0002"]
    assert not bad["success"] and bad.get("kind") == "saved_fit_mismatch" and "on its anchor" in bad["error"]
    assert bad["parameters"] == {} and bad["unit_verdict"]["decided_by"] == "excluded"   # no wrong numbers
    flags = {f["name"]: f["reason"] for f in state.get("flagged_spectra") or []}
    assert flags.get("spectrum_0002") == "fit_failed"                          # a refit reason
    results = sv.compile_results(tmp_path, state)
    by_name = {u["name"]: u for u in results["individual_results"]}
    assert by_name["spectrum_0002"]["kind"] == "saved_fit_mismatch"


def test_without_an_anchor_reference_a_follower_is_judged_as_before(tmp_path, monkeypatch):
    """No anchor saved fit (an older recipe, a canned anchor): a replay below
    a flat line is not failed — only the axis rule applies."""
    import test_series_verdict_path as sv

    class Ex(sv.FakeExecutor):
        def execute_script(self, script, working_dir=None, timeout=None, **kw):
            res = super().execute_script(script, working_dir=working_dir, timeout=timeout, **kw)
            y = cc._extract_xy(np.load(Path(working_dir) / "data.npy"))[1]
            np.save(Path(working_dir) / "fit.npy", np.full_like(y, 60.0))
            return res
    follower_r2 = {"spectrum_0001": 0.97, "spectrum_0002": 0.97}
    state, _ = sv.run_series(tmp_path, monkeypatch, names=sv.NAMES, anchors={"spectrum_0000": sv.OK},
                             follower_r2=follower_r2, executor=Ex(follower_r2))
    assert all(u["success"] for u in state["series_results"])
    assert analysis_verdict(sv.compile_results(tmp_path, state))["verified"]


# -- the reported values ----------------------------------------------------

def _controller(tmp_path, corrections):
    c = cc.UnifiedSeriesProcessingController.__new__(cc.UnifiedSeriesProcessingController)
    c.logger = logging.getLogger("t762"); c.output_dir = Path(tmp_path); c.executor = object()
    c._extract_extra_operands = lambda state, p: None
    c._extra_operand_block = lambda state: ""
    c._compute_statistics = lambda cd: {"n_points": X.size, "x_range": [float(X[0]), float(X[-1])]}
    c._should_escalate_timeout_model = lambda *a, **k: False
    c._correct_script = lambda state, script, err: (corrections.append(err) or (script + "\n# fixed", "x"))
    return c


def _run(tmp_path, monkeypatch, params, targets=None, bounds=None, fq=None, base_script="first"):
    fr = {"model_type": "m", "parameters": params, "fit_quality": {"r_squared": 0.99, **(fq or {})}}
    if targets is not None:
        fr["targets"] = targets
    if bounds:
        fr["bounds"] = bounds
    run = {"status": "success", "visualization_path": "viz.png", "visualization_bytes": b"", "exec": {},
           "stdout": "FIT_RESULTS_JSON:" + json.dumps(fr)}
    monkeypatch.setattr(cc, "stage_and_run_adaptive", lambda *a, **k: run)
    corrections = []
    res = _controller(tmp_path, corrections)._fit_single_spectrum(
        {}, np.column_stack([X, Y]), "s.csv", "s", 0, base_script=base_script)
    return res, corrections


def test_a_centre_reported_outside_the_axis_is_not_measured_and_the_other_bands_stand(tmp_path, monkeypatch):
    params = {"low": {"center": 96.5, "center_err": 0.2, "amplitude": 0.4, "fwhm": 50.0},
              "main": {"center": 1400.0, "amplitude": 0.8, "fwhm": 70.0}}
    res, corrections = _run(tmp_path, monkeypatch, params, targets=["low", "main"])
    assert res["success"] and corrections == [] and "pinned_at_bound" not in res
    assert [(p["component"], p["value"]) for p in res["not_measured"]] == [("low", 96.5)]
    assert all(v is None for v in res["parameters"]["low"].values())
    assert res["parameters"]["main"]["center"] == 1400.0
    assert any("low (centre 96.5 outside the measured axis)" in c for c in res["caveats"])
    assert "no_target_measured" not in res["fit_quality"]


def test_a_width_reported_as_the_centre_withholds_a_fit_that_measured_nothing(tmp_path, monkeypatch):
    """Every band's centre carries its FWHM (arguments shifted by one): the
    fit is fine, the report is not. Nothing asked for was measured: the
    values are no value and the fit is withheld, not failed or repaired."""
    params = {f"b{i}": {"center": w, "fwhm": w, "amplitude": 1.0} for i, w in enumerate((6.0, 14.0, 40.0))}
    res, corrections = _run(tmp_path, monkeypatch, params, targets=["b0", "b1", "b2"])
    assert res["success"] and corrections == []
    assert sorted(p["component"] for p in res["not_measured"]) == ["b0", "b1", "b2"]
    assert res["fit_quality"]["no_target_measured"] == ["b0", "b1", "b2"]
    from scilink.agents.exp_agents._verification_record import _unit_verdict
    res["quality_history"] = {"approved": True, "final_r2": 0.99, "verification_iterations": [{}]}
    bad = _unit_verdict(res, where="")
    assert bad and not bad["verified"] and "no target measured" in bad["reason"]
    # the feature table gains no column for it (lists are not scalars)
    from scilink.agents.exp_agents.feature_table import _flatten_scalars
    assert not any("no_target" in k for k in _flatten_scalars(res["fit_quality"], "fit_"))


def test_names_that_are_not_a_bands_position_are_not_read(tmp_path, monkeypatch):
    """Measured on the saved fits: a unit suffix (another unit), a derived
    quantity, an ambiguous ``mu`` (a level's mean), a position with no width
    — none is held to the axis."""
    params = {"p": {"center_Hz": -1215.0, "fwhm": 30.0, "amplitude": 1.0},
              "d": {"center_separation": 13.3, "fwhm": 5.0},
              "level_A": {"mu": -26.1, "sigma": 0.4},
              "edge": {"center": 50.0, "height": 1.0},
              "main": {"center": 1400.0, "amplitude": 0.8, "fwhm": 70.0}}
    res, _ = _run(tmp_path, monkeypatch, params)
    assert "not_measured" not in res and res["parameters"]["level_A"]["mu"] == -26.1


def test_a_width_wider_than_the_axis_is_no_value_and_zero_is_left_alone(tmp_path, monkeypatch):
    params = {"flat": {"center": 1300.0, "fwhm": 74427.0, "fwhm_err": 9.0, "amplitude": 0.1},
              "voigt": {"center": 1400.0, "sigma": 20.0, "gamma": 0.0, "amplitude": 0.8}}
    res, _ = _run(tmp_path, monkeypatch, params)
    assert res["parameters"]["flat"]["fwhm"] is None and res["parameters"]["flat"]["fwhm_err"] is None
    assert res["parameters"]["flat"]["center"] == 1300.0                      # the rest of the band stands
    assert res["parameters"]["voigt"]["gamma"] == 0.0                         # a lineshape limit
    assert [(p["component"], p["parameter"]) for p in res["withheld_values"]] == [("flat", "fwhm")]
    assert any("a width wider than the measured axis" in c for c in res["caveats"])


def test_a_degenerate_fits_uncertainties_are_no_value(tmp_path, monkeypatch):
    params = {"main": {"center": 1400.0, "center_err": 0.001, "amplitude": 0.8, "amplitude_err": 0.01,
                       "fwhm": 40.0, "fwhm_err": 0.1}}
    res, _ = _run(tmp_path, monkeypatch, params, bounds={"main": {"fwhm": [1.0, 40.0]}})
    assert [p["parameter"] for p in res["pinned_at_bound"]] == ["fwhm"]
    assert all(res["parameters"]["main"][k] is None for k in ("center_err", "amplitude_err", "fwhm_err"))
    assert res["parameters"]["main"]["center"] == 1400.0
    assert res["withheld_values"][0]["parameter"] == "uncertainties"
    assert any("every uncertainty (a degenerate fit's covariance is not a precision)" in c for c in res["caveats"])
    # an undegenerate fit keeps them
    params["main"]["fwhm"] = 30.0
    res, _ = _run(tmp_path, monkeypatch, params, bounds={"main": {"fwhm": [1.0, 40.0]}})
    assert res["parameters"]["main"]["center_err"] == 0.001 and "withheld_values" not in res


def test_a_secondary_pin_withholds_its_components_uncertainties(tmp_path, monkeypatch):
    params = {"main": {"center": 1400.0, "center_err": 0.5, "amplitude": 0.8, "fwhm": 70.0},
              "background": {"c0": 0.2, "c0_err": 0.01, "slope": 1.0, "slope_err": 0.3}}
    res, _ = _run(tmp_path, monkeypatch, params, targets=["main"], bounds={"background": {"slope": [-1.0, 1.0]}})
    assert [p["component"] for p in res["secondary_pins"]] == ["background"]
    assert res["parameters"]["background"]["c0_err"] is None and res["parameters"]["background"]["c0"] == 0.2
    assert res["parameters"]["main"]["center_err"] == 0.5


def test_the_caveat_flags_are_non_refit_and_say_why(tmp_path, monkeypatch):
    assert "withheld_values" in cc.CAVEAT_FLAGS
    import test_series_verdict_path as sv

    class Ex(sv.FakeExecutor):
        def execute_script(self, script, working_dir=None, timeout=None, **kw):
            res = super().execute_script(script, working_dir=working_dir, timeout=timeout, **kw)
            out = json.loads(res["stdout"].split("FIT_RESULTS_JSON:", 1)[1])
            out["parameters"]["peak_1"]["fwhm"] = 5000.0                         # wider than the 700-wide axis
            res["stdout"] = "FIT_RESULTS_JSON:" + json.dumps(out)
            return res
    follower_r2 = {"spectrum_0001": 0.97, "spectrum_0002": 0.97}
    ex = Ex(follower_r2)
    state, _ = sv.run_series(tmp_path, monkeypatch, names=sv.NAMES, anchors={"spectrum_0000": sv.OK},
                             follower_r2=follower_r2, executor=ex)
    flags = {f["name"]: f for f in state.get("flagged_spectra") or []}
    assert flags["spectrum_0001"]["reason"] == "withheld_values"
    assert "a width wider than the measured axis" in flags["spectrum_0001"]["recommendation"]
    assert [n for n, _ in ex.calls].count("spectrum_0001") == 1                 # nothing refit
    assert analysis_verdict(sv.compile_results(tmp_path, state))["verified"]


def test_the_planning_ingestion_names_the_new_caveat():
    from scilink.agents.planning_agents.orchestrator_tools import OrchestratorTools
    assert "withheld_values" in OrchestratorTools._EMPTY_VALUE_REASONS


# -- a reuse of the recipe is held to its anchor too ------------------------

def test_the_recipe_carries_its_anchors_saved_fit(tmp_path, monkeypatch):
    """The anchor's saved-fit R² goes with the recipe (its certification
    reference, which every copy carries), so a later reuse is held to it."""
    import test_series_verdict_path as sv
    real_canned = sv._canned_anchor
    monkeypatch.setattr(sv, "_canned_anchor", lambda *a, **k: {**real_canned(*a, **k), "saved_fit_r2": 0.34})
    follower_r2 = {"spectrum_0001": 0.97, "spectrum_0002": 0.97}
    state, _ = sv.run_series(tmp_path, monkeypatch, names=sv.NAMES, anchors={"spectrum_0000": sv.OK},
                             follower_r2=follower_r2)
    recs = list((state.get("locked_recipes") or {}).values())
    assert recs and recs[0]["saved_fit_r2"] == 0.34
    assert recs[0]["certification_reference"]["saved_fit_r2"] == 0.34


def test_a_reuse_replay_is_held_to_the_recipe_it_replays(tmp_path):
    import test_series_verdict_path as sv
    from types import SimpleNamespace
    ctrl = sv._controller(tmp_path, object())
    seen = []
    ctrl._fit_single_spectrum = lambda **kw: seen.append(kw.get("anchor_saved_fit_r2")) or {"success": False}
    ctx = SimpleNamespace(state={}, data=np.column_stack([X, Y]), data_path="d", item_name="s", item_idx=0)
    ctrl._run_reuse_candidate(ctx, "script", "prior", None, 1, 1, reference={"saved_fit_r2": 0.6})
    ctrl._run_reuse_candidate(ctx, "script", "prior", None, 1, 1)               # an older recipe: no reference
    assert seen == [0.6, None]


def test_the_anchors_saved_fit_is_read_in_any_layout(tmp_path):
    """An array series stages each unit as (2, N): the anchor's saved-fit R²
    is read as its followers' replays are, or the replay rule is silently off."""
    for data in (np.column_stack([X, Y]), np.vstack([X, Y])):
        ex = _FitWriter()
        ctrl, _ = _ctrl(tmp_path / str(data.shape), ex, corrected="np.load('data.npy')")
        res = ctrl._fit_single_spectrum(state={"system_info": {}}, curve_data=data, data_path="d.npy",
                                        spectrum_name="spectrum_0000", spectrum_idx=0,
                                        base_script="np.load('data.npy')")
        assert res["success"] and res["saved_fit_r2"] > 0.99, data.shape
