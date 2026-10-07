"""What a curve fit reports is checked against the fit and the data (#762).

- Code generation is told where x and y sit in ``data.npy`` (from the array
  staged), and every prompt that writes, repairs or judges a script that a
  replay runs carries the rule: never pick a column, a window or anything
  else by matching a value read off one spectrum. A repair (which never sees
  the generation prompt) is told the layout too.
- A saved fit that follows the x axis, or a replay that does not follow its
  data even up to a constant offset (R² < -1) where its anchor was not below
  zero, is not a fit of this data: the ladder is told, and a unit that still
  shows it fails (refit-eligible), never verified on its self-reported R². A
  saved fit below a flat line on its own is no error (a peaks-only fit saved
  without its baseline, a rising background along a series, a windowed
  model evaluated outside its window, a level fitted to a flat control).
- A band whose REPORTED centre lies outside the measured axis is not
  measured; a fit left with no target measured is withheld — by its own
  gate, as a follower and as a reuse — never failed.
- A band width wider than the whole axis and the uncertainties of a
  degenerate fit (and of a secondary pin's component) are reported as no
  value, with the reason.
- Where nothing fires, the fit is reported exactly as the script printed it.
"""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scilink.agents.exp_agents.controllers import curve_fitting_controllers as cc
from scilink.agents.exp_agents._verification_record import analysis_verdict, unit_verdict_for

X = np.linspace(200.0, 2000.0, 400)
Y = np.exp(-0.5 * ((X - 900.0) / 30.0) ** 2) + 0.2
GOOD = "np.load('data.npy')  # reads x and y by position"
AXIS = "np.load('data.npy')  # AXIS: fits the x column"


# -- the contract ---------------------------------------------------------

def test_the_layout_is_stated_from_the_array_staged():
    assert "column 0 is x, column 1 is y" in cc._data_layout(np.zeros((50, 2)))
    assert "row 0 is x, row 1 is y" in cc._data_layout(np.zeros((2, 50)))
    assert "x is the point index" in cc._data_layout(np.zeros(50))
    # a shape no reader recognises names no layout
    assert cc._data_layout(np.zeros((50, 3))) == "an array of shape (50, 3), as staged"
    assert cc._extract_xy(np.zeros((50, 3))) is None


class _Model:
    """Records every prompt; answers with ``reply`` (a dict) as JSON."""

    def __init__(self, reply=None):
        self.prompts, self.reply = [], reply or {"script": GOOD, "diagnosis": "d"}

    def generate_content(self, contents=None, **kw):
        c = contents if contents is not None else kw.get("contents")
        self.prompts.append(c if isinstance(c, str) else json.dumps(c, default=str))
        return SimpleNamespace(text=json.dumps(self.reply))


def _real_controller(tmp_path, executor, model):
    from scilink.agents.exp_agents.instruct import (FITTING_SCRIPT_INSTRUCTIONS,
                                                    FITTING_SCRIPT_CORRECTION_INSTRUCTIONS)
    return cc.UnifiedSeriesProcessingController(
        model=model, logger=logging.getLogger("t762"), generation_config=None, safety_settings=None,
        parse_fn=lambda r: (json.loads(r.text), None), executor=executor,
        script_instructions=FITTING_SCRIPT_INSTRUCTIONS, correction_instructions=FITTING_SCRIPT_CORRECTION_INSTRUCTIONS,
        quality_instructions="", output_dir=str(tmp_path), plot_fn=lambda d, i: b"p", r2_threshold=0.95,
        parallel_workers=1)


def test_the_rule_is_in_every_prompt_that_writes_or_judges_a_script(tmp_path):
    from scilink.agents.exp_agents.instruct import PLAN_CONFORMANCE_CHECK_INSTRUCTIONS
    model = _Model()
    ctrl = _real_controller(tmp_path, object(), model)
    state = {"locked_fitting_config": {"physical_model": "one Gaussian"}}
    stats = {"n_points": 2, "x_range": (0.0, 1.0), "y_range": (0.0, 1.0)}
    ctrl._generate_fitting_script(state, "data.npy", stats, data_layout=cc._data_layout(np.zeros((2, 50))))
    ctrl._generate_fitting_script(state, "data.npy", stats)          # a caller naming no layout, two points
    ctrl._correct_script(state, "print(0)", "boom")
    assert "Layout: shape (2, N): row 0 is x, row 1 is y" in model.prompts[0]
    assert "Layout: shape (N, 2): column 0 is x, column 1 is y" in model.prompts[1]
    for p in model.prompts[:2]:
        assert "never pick a column, a window or\n   anything else by matching a value read off this spectrum" in p
    assert "never picking a column or a window by matching a value read off one spectrum" in model.prompts[2]
    assert "pick a column\nor a window by matching a value read off one spectrum" in PLAN_CONFORMANCE_CHECK_INSTRUCTIONS


def test_the_bank_adapt_prompt_carries_the_layout_and_the_rule(tmp_path):
    from scilink.agents.exp_agents._qc_engine import QCItemContext
    model = _Model({"edits": [], "model_family_kept": True, "rationale": "fits as is"})
    ctrl = _real_controller(tmp_path, object(), model)
    ctrl._fit_single_spectrum = lambda **kw: {"success": False}
    for data, layout in ((np.vstack([X, Y]), "shape (2, N): row 0 is x"), (np.column_stack([X, Y]), "shape (N, 2)")):
        state = {"_bank_exemplar": {"score": 0.99, "record": {"working_script": GOOD}},
                 "locked_fitting_config": {}, "system_info": {}, "data_statistics": {}}
        ctx = QCItemContext(state=state, data=data, data_path="d", item_name="s", item_idx=0)
        ctrl._try_bank_edit_adapt(ctx)
        assert f"data.npy layout: {layout}" in model.prompts[-1]
        assert "reading x and y from their fixed positions in data.npy" in model.prompts[-1]


# -- the saved fit is a fit of THIS data -----------------------------------

def test_a_saved_fit_of_the_x_axis_is_not_a_fit_of_the_data():
    data = np.column_stack([X, Y])
    why, _ = cc._saved_fit_mismatch(data, X.copy())
    assert why and "follows the x axis" in why
    steps = np.where(X < 1000, X[X < 1000].mean(), X[X >= 1000].mean())   # level means of x windows
    assert "follows the x axis" in cc._saved_fit_mismatch(data, steps)[0]
    assert "follows the x axis" in cc._saved_fit_mismatch(data.T, X.copy())[0]      # (2, N) read the same way


def test_a_replay_that_does_not_follow_its_data_up_to_a_constant_is_caught():
    data = np.column_stack([X, Y])
    wrong = Y.mean() - 3.0 * (Y - Y.mean())          # a shape the data does not have, whatever the offset
    why, r2 = cc._saved_fit_mismatch(data, wrong)
    assert why is None and r2 < -1                  # alone, no evidence
    why, _ = cc._saved_fit_mismatch(data, wrong, anchor_r2=0.6)
    assert why and "even up to a constant offset" in why and "on its anchor" in why


def test_an_offset_or_a_flat_control_is_no_evidence():
    peak = np.exp(-0.5 * ((X - 900.0) / 30.0) ** 2)
    rng = np.random.default_rng(0)
    for bg in (np.full_like(X, 0.5), np.full_like(X, 2.0), 0.004 * (X - X[0])):
        data = np.column_stack([X, peak + bg + 0.01 * rng.standard_normal(X.size)])
        assert cc._saved_fit_mismatch(data, peak, anchor_r2=0.96)[0] is None   # a peaks-only recipe
    noise = np.column_stack([X, 0.5 * rng.standard_normal(X.size)])
    assert cc._saved_fit_mismatch(noise, np.full(X.size, 0.2), anchor_r2=0.02)[0] is None   # a flat control
    data = np.column_stack([X, Y])
    assert cc._saved_fit_mismatch(data, Y - 3.0, anchor_r2=-4.0)[0] is None          # a windowed anchor
    assert cc._saved_fit_mismatch(data, Y.copy(), anchor_r2=0.9)[0] is None
    assert cc._saved_fit_mismatch(np.column_stack([X[::-1], Y[::-1]]), Y.copy(), anchor_r2=0.9)[0] is None
    assert cc._saved_fit_mismatch(data, Y[:-3]) == (None, None)                     # cannot be aligned


class _FitWriter:
    """Prints the fit JSON and saves ``fit.npy`` as the script says: AXIS
    saves the x column, NOFIT saves nothing, anything else saves y."""
    timeout = 30

    def __init__(self, params=None):
        self.calls, self.params = [], params

    def execute_script(self, script, working_dir=None, timeout=None, **kw):
        wd = Path(working_dir)
        self.calls.append(script)
        x, y = cc._extract_xy(np.load(wd / "data.npy"))
        if "NOFIT" not in script:
            np.save(wd / "fit.npy", np.asarray(x if "AXIS" in script else y, float))
        (wd / "visualization.png").write_bytes(b"png")
        out = {"model_type": "m", "fit_quality": {"r_squared": 0.99},
               "parameters": self.params or {"peak_1": {"center": 900.0, "amplitude": 1.0, "fwhm": 70.0}}}
        return {"status": "success", "stdout": "FIT_RESULTS_JSON:" + json.dumps(out), "stderr": "", "message": ""}


def _fit(tmp_path, ex, model, data, *, first=AXIS, base_script=None, state=None, **kw):
    ctrl = _real_controller(tmp_path, ex, model)
    ctrl._generate_fitting_script = lambda *a, **k: first
    ctrl._check_plan_conformance = lambda state, script: None
    return ctrl._fit_single_spectrum(state={"system_info": {}, "locked_fitting_config": {}, **(state or {})},
                                     curve_data=data, data_path="d.npy", spectrum_name="spectrum_0000",
                                     spectrum_idx=0, base_script=base_script, **kw)


def test_a_repair_through_the_real_correction_prompt_is_told_the_layout(tmp_path):
    for data, layout in ((np.column_stack([X, Y]), "shape (N, 2): column 0 is x"),
                         (np.vstack([X, Y]), "shape (2, N): row 0 is x, row 1 is y")):
        model = _Model({"script": GOOD, "diagnosis": "read column 1"})
        res = _fit(tmp_path / str(data.shape), _FitWriter(), model, data)
        assert res["success"] and res["script"] == GOOD and res["saved_fit_r2"] > 0.99
        corr = [p for p in model.prompts if "Fix this failed script" in p]
        assert len(corr) == 1 and "THE SAVED FIT IS NOT A FIT OF THIS DATA" in corr[0]
        assert "never by matching a value read off one spectrum" in corr[0]
        assert f"data.npy layout: {layout}" in corr[0]


def test_a_unit_that_still_fits_the_axis_fails_and_a_strict_replay_calls_no_model(tmp_path):
    model = _Model({"script": AXIS + " again", "diagnosis": "d"})
    res = _fit(tmp_path / "a", _FitWriter(), model, np.column_stack([X, Y]))
    assert not res["success"] and res["kind"] == "saved_fit_mismatch" and "follows the x axis" in res["error"]
    model = _Model()
    res = _fit(tmp_path / "b", _FitWriter(), model, np.column_stack([X, Y]), base_script=AXIS,
               state={"_strict_replay": True})
    assert not res["success"] and res["kind"] == "saved_fit_mismatch" and model.prompts == []


def test_a_fit_npy_this_run_did_not_write_is_never_read(tmp_path):
    """In the loop and after it: a fit left by an earlier attempt or an
    earlier call is not this script's fit."""
    d = Path(tmp_path) / "a" / "spectrum_0000"
    d.mkdir(parents=True)
    np.save(d / "fit.npy", X)                                                  # an earlier call's fit
    model = _Model()
    res = _fit(tmp_path / "a", _FitWriter(), model, np.column_stack([X, Y]), base_script="x = np.load('data.npy')  # NOFIT")
    assert res["success"] and model.prompts == [] and "saved_fit_r2" not in res and not res.get("residual_diagnostics")
    # a rejected x-axis attempt, then a repair that saves no fit.npy: the rejected fit is not read
    model = _Model({"script": "x = np.load('data.npy')  # NOFIT", "diagnosis": "d"})
    res = _fit(tmp_path / "b", _FitWriter(), model, np.column_stack([X, Y]))
    assert res["success"] and "saved_fit_r2" not in res and not res.get("residual_diagnostics")


def test_a_follower_whose_replay_does_not_fit_its_data_is_failed_not_verified(tmp_path, monkeypatch):
    """Through the series: the anchor's saved-fit R² reaches the follower,
    whose replay does not follow its own data; it fails, is refit-eligible,
    and reports no number."""
    import test_series_verdict_path as sv
    real_canned = sv._canned_anchor
    monkeypatch.setattr(sv, "_canned_anchor", lambda *a, **k: {**real_canned(*a, **k), "saved_fit_r2": 0.6})

    class Ex(sv.FakeExecutor):
        def execute_script(self, script, working_dir=None, timeout=None, **kw):
            res = super().execute_script(script, working_dir=working_dir, timeout=timeout, **kw)
            wd = Path(working_dir)
            y = cc._extract_xy(np.load(wd / "data.npy"))[1]
            bad = "spectrum_0002" in wd.as_posix()
            np.save(wd / "fit.npy", y.mean() - 3.0 * (y - y.mean()) if bad else y)
            return res

    follower_r2 = {"spectrum_0001": 0.97, "spectrum_0002": 0.97, "spectrum_0003": 0.97}
    state, _ = sv.run_series(tmp_path, monkeypatch, names=sv.NAMES + ["spectrum_0003"],
                             anchors={"spectrum_0000": sv.OK}, follower_r2=follower_r2, executor=Ex(follower_r2))
    raw = {u["name"]: u for u in state["series_results"]}
    assert raw["spectrum_0001"]["success"] and raw["spectrum_0001"]["unit_verdict"]["verified"]
    bad = raw["spectrum_0002"]
    assert not bad["success"] and bad.get("kind") == "saved_fit_mismatch" and "on its anchor" in bad["error"]
    assert bad["parameters"] == {} and bad["unit_verdict"]["decided_by"] == "excluded"
    flags = {f["name"]: f["reason"] for f in state.get("flagged_spectra") or []}
    assert flags.get("spectrum_0002") == "fit_failed"                          # a refit reason


def test_a_peaks_only_recipe_under_a_rising_background_stays_mains_path(tmp_path, monkeypatch):
    """The anchor's saved fit leaves out a small background; the followers'
    background rises. Nothing fires: no correction, every unit verified."""
    import test_series_verdict_path as sv
    real_canned = sv._canned_anchor
    monkeypatch.setattr(sv, "_canned_anchor", lambda *a, **k: {**real_canned(*a, **k), "saved_fit_r2": 0.96})
    peak = np.exp(-0.5 * ((sv.X - 144) / 6) ** 2)
    rises = {"spectrum_0001": 0.5, "spectrum_0002": 2.0, "spectrum_0003": 4.0}
    real_spectrum = sv._spectrum
    monkeypatch.setattr(sv, "_spectrum", lambda i: np.c_[sv.X, real_spectrum(i)[:, 1]
                                                         + rises.get(f"spectrum_{i:04d}", 0.0)])

    class Ex(sv.FakeExecutor):
        def execute_script(self, script, working_dir=None, timeout=None, **kw):
            res = super().execute_script(script, working_dir=working_dir, timeout=timeout, **kw)
            np.save(Path(working_dir) / "fit.npy", peak)
            return res
    follower_r2 = {n: 0.97 for n in rises}
    ex = Ex(follower_r2)
    state, _ = sv.run_series(tmp_path, monkeypatch, names=sv.NAMES + ["spectrum_0003"],
                             anchors={"spectrum_0000": sv.OK}, follower_r2=follower_r2, executor=ex)
    assert all(u["success"] and u["unit_verdict"]["verified"] for u in state["series_results"])
    assert [n for n, _ in ex.calls] == list(rises)                             # one run each, nothing repaired


def test_the_recipe_carries_its_anchors_saved_fit(tmp_path, monkeypatch):
    import test_series_verdict_path as sv
    real_canned = sv._canned_anchor
    monkeypatch.setattr(sv, "_canned_anchor", lambda *a, **k: {**real_canned(*a, **k), "saved_fit_r2": 0.5})
    state, _ = sv.run_series(tmp_path, monkeypatch, names=sv.NAMES, anchors={"spectrum_0000": sv.OK},
                             follower_r2={"spectrum_0001": 0.97, "spectrum_0002": 0.97})
    recs = list((state.get("locked_recipes") or {}).values())
    assert recs and recs[0]["saved_fit_r2"] == 0.5
    assert recs[0]["certification_reference"]["saved_fit_r2"] == 0.5


# -- a reuse, through the real qc_try_reuse ----------------------------------

def _board_copy(tmp_path, saved_fit_r2, script):
    """A recipe as the swarm board copies it: the script and its sidecar,
    carrying the run's certification reference."""
    import test_board_copy_certification as bc
    ref = {"kind": "curve", "drift_state": None, "x_range": None, "identity": None,
           **({"saved_fit_r2": saved_fit_r2} if saved_fit_r2 is not None else {})}
    _, _, copy = bc._post_series_copy(tmp_path, ref)
    copy.write_text(script)
    return copy


def _reuse(tmp_path, ex, model, copy, *, source=None):
    from scilink.agents.exp_agents._qc_engine import QCItemContext
    out = tmp_path / "new"
    out.mkdir(parents=True, exist_ok=True)
    ctrl = _real_controller(out, ex, model)
    state = {"num_spectra": 1, "is_single_spectrum": True, "system_info": {}, "locked_fitting_config": {},
             "prior_analysis_paths": [str(copy)], "reuse_locked_script": True, "_reuse_candidates": []}
    ctx = QCItemContext(state=state, data=np.column_stack([X, Y]), data_path=str(out / "new.txt"),
                        item_name="spectrum_0000", item_idx=0, reuse_script=copy.read_text(),
                        reuse_source=source or f"prior: {copy.name}")
    return ctrl.qc_try_reuse(ctx)


class _Wrong(_FitWriter):
    """A script marked WRONG saves a fit of a shape the data does not have."""

    def execute_script(self, script, working_dir=None, timeout=None, **kw):
        res = super().execute_script(script, working_dir=working_dir, timeout=timeout, **kw)
        if "WRONG" in script:
            y = cc._extract_xy(np.load(Path(working_dir) / "data.npy"))[1]
            np.save(Path(working_dir) / "fit.npy", y.mean() - 3.0 * (y - y.mean()))
        return res


def test_a_reuse_is_held_to_the_recipe_it_replays(tmp_path):
    """A board copy carrying its anchor's saved-fit R²: the replay that does
    not follow the new data is sent to the ladder; with no reference (or a
    script-bank cold start, which no prior run stands behind) it is not."""
    wrong = "np.load('data.npy')  # WRONG shape"
    model = _Model()
    res = _reuse(tmp_path / "a", _Wrong(), model, _board_copy(tmp_path / "a", 0.6, wrong))
    corr = [p for p in model.prompts if "Fix this failed script" in p]
    assert corr and "on its anchor" in corr[0] and res["script"] == GOOD
    for sub, r2, source in (("b", None, None), ("c", 0.6, "script_bank:abc")):
        model = _Model()
        res = _reuse(tmp_path / sub, _Wrong(), model, _board_copy(tmp_path / sub, r2, wrong), source=source)
        assert model.prompts == [] and res["script"] == wrong, sub


def test_a_reuse_that_measured_nothing_is_not_verified(tmp_path):
    """A band reported outside the axis on new data under an old recipe:
    the replay gate passed, the values are no value, and neither a single
    reuse nor a series anchor that replayed it is verified."""
    ex = _FitWriter(params={"peak_1": {"center": 50.0, "amplitude": 1.0, "fwhm": 70.0}})
    res = _reuse(tmp_path, ex, _Model(), _board_copy(tmp_path, None, GOOD))
    assert res["reuse_validity"]["verdict"] == "good"
    assert all(v is None for v in res["parameters"]["peak_1"].values())
    assert res["fit_quality"]["no_target_measured"] == ["peak_1"]
    full = {"status": "success", "fit_quality": res["fit_quality"], "reuse_validity": res["reuse_validity"]}
    v = analysis_verdict(full)
    assert not v["verified"] and "no target measured" in v["reason"]
    uv = unit_verdict_for({**res, "success": True})
    assert not uv["verified"] and uv["decided_by"] == "replay_gate" and "no target measured" in uv["reason"]


def test_a_follower_that_measured_nothing_is_not_verified():
    unit = {"success": True, "fitted_from": "locked_script",
            "fit_quality": {"no_target_measured": ["peak_1"], "not_measured": [{"component": "peak_1"}]}}
    uv = unit_verdict_for(unit, recipe={"unit": "a", "verdict": {"verified": True}})
    assert not uv["verified"] and "no target measured" in uv["reason"]


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


def test_where_nothing_fires_the_fit_is_reported_as_printed(tmp_path, monkeypatch):
    params = {"main": {"center": 900.0, "center_err": 0.4, "amplitude": 0.8, "fwhm": 70.0, "fwhm_err": 1.0},
              "background": {"c0": 0.2, "c0_err": 0.01},
              "rms_error": 0.03}
    res, corrections = _run(tmp_path, monkeypatch, json.loads(json.dumps(params)), targets=["main"])
    assert res["success"] and corrections == [] and res["parameters"] == params
    assert set(res["fit_quality"]) == {"r_squared"}
    assert not any(k in res for k in ("caveats", "not_measured", "withheld_values", "pinned_at_bound"))


def test_a_centre_reported_outside_the_axis_is_not_measured_and_the_other_bands_stand(tmp_path, monkeypatch):
    params = {"low": {"center": 50.0, "center_err": 0.2, "amplitude": 0.4, "fwhm": 50.0},
              "main": {"center": 900.0, "amplitude": 0.8, "fwhm": 70.0}}
    res, corrections = _run(tmp_path, monkeypatch, params, targets=["low", "main"])
    assert res["success"] and corrections == [] and "pinned_at_bound" not in res
    assert [(p["component"], p["value"]) for p in res["not_measured"]] == [("low", 50.0)]
    assert all(v is None for v in res["parameters"]["low"].values())
    assert res["parameters"]["main"]["center"] == 900.0
    assert any("low (centre 50 outside the measured axis)" in c for c in res["caveats"])
    assert "no_target_measured" not in res["fit_quality"]


def test_a_width_reported_as_the_centre_withholds_a_fit_that_measured_nothing(tmp_path, monkeypatch):
    """Every band's centre carries its width (a reporting call shifted by
    one): the fit is fine, the report is not. The values are no value and
    the fit is withheld, not failed or repaired."""
    params = {f"b{i}": {"center": w, "fwhm": w, "amplitude": 1.0} for i, w in enumerate((5.0, 12.0, 30.0))}
    res, corrections = _run(tmp_path, monkeypatch, params, targets=["b0", "b1", "b2"])
    assert res["success"] and corrections == []
    assert sorted(p["component"] for p in res["not_measured"]) == ["b0", "b1", "b2"]
    assert res["fit_quality"]["no_target_measured"] == ["b0", "b1", "b2"]
    from scilink.agents.exp_agents._verification_record import _unit_verdict
    res["quality_history"] = {"approved": True, "final_r2": 0.99, "verification_iterations": [{}]}
    bad = _unit_verdict(res, where="")
    assert bad and not bad["verified"] and "no target measured" in bad["reason"]
    from scilink.agents.exp_agents.feature_table import _flatten_scalars
    assert not any("no_target" in k for k in _flatten_scalars(res["fit_quality"], "fit_"))


def test_both_band_readers_agree(tmp_path, monkeypatch):
    """A background and the only band, named by ``cen``, outside the axis:
    nothing asked for was measured, as for ``center``."""
    params = {"background": {"c0": 0.2, "c1": 1e-4}, "band": {"cen": 50.0, "sigma": 20.0, "amplitude": 1.0}}
    res, _ = _run(tmp_path, monkeypatch, params)
    assert res["fit_quality"]["no_target_measured"] == ["band"]


def test_names_that_are_not_a_bands_position_are_not_read(tmp_path, monkeypatch):
    """A unit suffix (another unit), a derived quantity, an ambiguous ``mu``
    (a level's mean), a position with no width — none is held to the axis."""
    params = {"p": {"center_mm": -5.0, "fwhm": 30.0, "amplitude": 1.0},
              "d": {"center_separation": 7.0, "fwhm": 5.0},
              "level_A": {"mu": -20.0, "sigma": 0.4},
              "edge": {"center": 50.0, "height": 1.0},
              "main": {"center": 900.0, "amplitude": 0.8, "fwhm": 70.0}}
    res, _ = _run(tmp_path, monkeypatch, params)
    assert "not_measured" not in res and res["parameters"]["level_A"]["mu"] == -20.0


def test_a_width_wider_than_the_axis_is_no_value_and_zero_is_left_alone(tmp_path, monkeypatch):
    params = {"flat": {"center": 800.0, "fwhm": 1e5, "fwhm_err": 9.0, "amplitude": 0.1},
              "voigt": {"center": 900.0, "sigma": 20.0, "gamma": 0.0, "amplitude": 0.8}}
    res, _ = _run(tmp_path, monkeypatch, params)
    assert res["parameters"]["flat"]["fwhm"] is None and res["parameters"]["flat"]["fwhm_err"] is None
    assert res["parameters"]["flat"]["center"] == 800.0
    assert res["parameters"]["voigt"]["gamma"] == 0.0
    assert [(p["component"], p["parameter"]) for p in res["withheld_values"]] == [("flat", "fwhm")]
    assert any("a width wider than the measured axis" in c for c in res["caveats"])


def test_a_degenerate_fits_uncertainties_are_no_value_and_its_metrics_stay(tmp_path, monkeypatch):
    params = {"main": {"center": 900.0, "center_err": 0.001, "amplitude": 0.8, "amplitude_stderr_rel": 0.01,
                       "fwhm": 40.0, "fwhm_ci95": 0.1, "phase_uncertainty_rad": 0.2, "plateau_std": 0.3},
              "rms_error": 0.03, "fit_error": 0.1, "mean_abs_error": 0.02}
    res, _ = _run(tmp_path, monkeypatch, params, bounds={"main": {"fwhm": [1.0, 40.0]}})
    assert [p["parameter"] for p in res["pinned_at_bound"]] == ["fwhm"]
    m = res["parameters"]["main"]
    assert all(m[k] is None for k in ("center_err", "amplitude_stderr_rel", "fwhm_ci95", "phase_uncertainty_rad"))
    assert m["center"] == 900.0 and m["plateau_std"] == 0.3                    # a scatter is a measurement
    assert (res["parameters"]["rms_error"], res["parameters"]["fit_error"]) == (0.03, 0.1)   # fit metrics
    assert any("the reported uncertainties (a degenerate fit's covariance is not a precision)" in c
               for c in res["caveats"])
    params["main"]["fwhm"] = 30.0                                              # not degenerate: kept
    res, _ = _run(tmp_path, monkeypatch, params, bounds={"main": {"fwhm": [1.0, 40.0]}})
    assert res["parameters"]["main"]["center_err"] == 0.001 and "withheld_values" not in res


def test_a_secondary_pin_withholds_its_components_uncertainties_on_the_record(tmp_path, monkeypatch):
    params = {"main": {"center": 900.0, "center_err": 0.5, "amplitude": 0.8, "fwhm": 70.0},
              "background": {"c0": 0.2, "c0_err": 0.01, "slope": 1.0, "slope_err": 0.3}}
    res, _ = _run(tmp_path, monkeypatch, params, targets=["main"], bounds={"background": {"slope": [-1.0, 1.0]}})
    assert [p["component"] for p in res["secondary_pins"]] == ["background"]
    assert res["parameters"]["background"]["c0_err"] is None and res["parameters"]["background"]["c0"] == 0.2
    assert res["parameters"]["main"]["center_err"] == 0.5
    assert {(p["component"], p["parameter"]) for p in res["withheld_values"]} == {("background", "uncertainties")}
    assert any("background's uncertainties" in c for c in res["caveats"])


def test_the_caveat_flags_are_non_refit_and_a_secondary_pin_keeps_its_flag(tmp_path, monkeypatch):
    assert "withheld_values" in cc.CAVEAT_FLAGS
    import test_series_verdict_path as sv

    class Ex(sv.FakeExecutor):
        def execute_script(self, script, working_dir=None, timeout=None, **kw):
            res = super().execute_script(script, working_dir=working_dir, timeout=timeout, **kw)
            out = json.loads(res["stdout"].split("FIT_RESULTS_JSON:", 1)[1])
            out["parameters"]["peak_1"]["fwhm"] = 5000.0                       # wider than the 700-wide axis
            if "spectrum_0002" in Path(working_dir).as_posix():                # and a secondary pin
                out["parameters"]["background"] = {"c0": 1.0}
                out["bounds"] = {"background": {"c0": [-1.0, 1.0]}}
                out["targets"] = ["peak_1"]
            res["stdout"] = "FIT_RESULTS_JSON:" + json.dumps(out)
            return res
    follower_r2 = {"spectrum_0001": 0.97, "spectrum_0002": 0.97}
    ex = Ex(follower_r2)
    state, _ = sv.run_series(tmp_path, monkeypatch, names=sv.NAMES, anchors={"spectrum_0000": sv.OK},
                             follower_r2=follower_r2, executor=ex)
    flags = {f["name"]: f for f in state.get("flagged_spectra") or []}
    assert flags["spectrum_0001"]["reason"] == "withheld_values"
    assert "a width wider than the measured axis" in flags["spectrum_0001"]["recommendation"]
    assert flags["spectrum_0002"]["reason"] == "secondary_pin"
    assert [n for n, _ in ex.calls].count("spectrum_0001") == 1                 # nothing refit
    assert analysis_verdict(sv.compile_results(tmp_path, state))["verified"]


def test_the_planning_ingestion_names_the_new_caveat():
    from scilink.agents.planning_agents.orchestrator_tools import OrchestratorTools
    assert "withheld_values" in OrchestratorTools._EMPTY_VALUE_REASONS
