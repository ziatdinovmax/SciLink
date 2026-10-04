"""The board's verdict on series runs, through the REAL series path.

Every round of the #702 review found a case the hand-built fixtures could
not: the verdict reconstructed, after the fact, which recipe each unit
replayed. This module drives the real machinery — `UnifiedSeriesProcessingController.execute`
(the real `_fit_single_spectrum` for every follower), `_detect_outliers`,
`stamp_profile`, `AdaptiveRefitController.execute` with its consistency pass,
and `CurveFittingAgent._compile_results` — stubbing only the anchor/refit QC
loop (canned results) and the script executor (a fake that prints the fit
JSON it is asked for). What `individual_results` carries is therefore what
the code writes, and `analysis_verdict` is judged on it.
"""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from scilink.agents.exp_agents._verification_record import analysis_verdict, series_anchor_unit
from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
    AdaptiveRefitController, UnifiedSeriesProcessingController)

THRESHOLD = 0.95
X = np.linspace(100, 800, 200)


def _spectrum(seed: int) -> np.ndarray:
    y = np.exp(-0.5 * ((X - 144) / 6) ** 2) + 0.01 * np.random.default_rng(seed).standard_normal(X.size)
    return np.c_[X, y]


class FakeExecutor:
    """Runs no script: prints the fit JSON the harness assigned to the unit
    (by the working directory's spectrum index) and drops a visualization."""
    timeout = 30

    def __init__(self, r2_by_name):
        self.r2_by_name = r2_by_name
        self.calls = []

    def execute_script(self, script, working_dir=None, timeout=None, **kw):
        wd = Path(working_dir)
        name = next((n for n in self.r2_by_name if n in wd.as_posix()), None)
        self.calls.append((name, script))
        r2 = self.r2_by_name.get(name, 0.99)
        if r2 is None:                                   # this unit's script fails to run
            return {"status": "error", "stdout": "", "stderr": "Traceback: boom", "message": "script failed: boom"}
        (wd / "visualization.png").write_bytes(b"png")
        model = f"model of {script}"
        out = {"model_type": model, "parameters": {"peak_1": {"center": 144.0, "amplitude": 1.0, "fwhm": 12.0}},
               "fit_quality": {"r_squared": r2, "rmse": 0.01}}
        return {"status": "success", "stdout": "FIT_RESULTS_JSON:" + json.dumps(out), "stderr": "", "message": ""}


def _canned_anchor(name, idx, *, r2, approved, script, warning=None, judge_warning=None, unverified=False,
                   failed=False, reused=None, pinned=None):
    if reused:                                   # a locked-script reuse: no QC record, a replay-gate verdict
        return {"index": idx, "name": name, "data_path": f"stack_index_{idx}", "success": True, "error": None,
                "model_type": f"model of {script}", "parameters": {"peak_1": {"center": 144.0}},
                "fit_quality": {"r_squared": r2, "rmse": 0.01}, "visualization_path": None,
                "visualization_bytes": None, "statistics": {}, "script": script, "script_errors": [],
                "reuse_validity": {"reused": True, "source": "prior", "r_squared": r2, "threshold": THRESHOLD,
                                   "verdict": reused, "message": "reused"}}
    if failed:                                   # the QC loop produced nothing
        return {"index": idx, "name": name, "data_path": f"stack_index_{idx}", "success": False,
                "error": "all attempts failed", "parameters": {}, "fit_quality": {}, "script": None,
                "script_errors": []}
    qh = {"final_r2": r2, "threshold": THRESHOLD, "approved": approved,
          "verification_iterations": [{"r_squared": r2, "annealing_level": 0}],
          "alternative_models": [], "script_errors": [], "judge_reasoning": None}
    if approved:
        qh["approved_by"] = "verifier"
    if unverified:
        qh.update({"approved": False, "unverified": True, "stopped_by": "time_budget"})
    res = {"index": idx, "name": name, "data_path": f"stack_index_{idx}", "success": True, "error": None,
           "model_type": f"model of {script}", "parameters": {"peak_1": {"center": 144.0}},
           "fit_quality": {"r_squared": r2, "rmse": 0.01}, "visualization_path": None,
           "visualization_bytes": None, "statistics": {}, "script": script, "script_errors": [],
           "quality_history": qh}
    if warning:
        res["quality_warning"] = warning
    if judge_warning:
        res["judge_warning"] = judge_warning
    if pinned:
        # what the controller writes for a fit with a parameter at its bound
        # (#592), approved or not: the pins and a "Degenerate fit" warning
        from scilink.skills._shared.curve_fitting_tools import describe_pinned
        res["pinned_at_bound"] = pinned
        res["fit_quality"] = {**res["fit_quality"], "pinned_at_bound": pinned}
        res["quality_warning"] = ("Degenerate fit: " + describe_pinned(pinned)
                                  + " — the extracted value is not trustworthy")
    return res


def _controller(tmp_path, executor):
    return UnifiedSeriesProcessingController(
        model=MagicMock(), logger=logging.getLogger("series_path"), generation_config=None,
        safety_settings=None, parse_fn=lambda r: (json.loads(r.text), None), executor=executor,
        script_instructions="", correction_instructions="", quality_instructions="",
        output_dir=str(tmp_path), plot_fn=lambda data, info: b"plot", r2_threshold=THRESHOLD,
        parallel_workers=1)


def _refitter(tmp_path, executor):
    return AdaptiveRefitController(
        model=MagicMock(), logger=logging.getLogger("series_path.refit"), generation_config=None,
        safety_settings=None, parse_fn=lambda r: (json.loads(r.text), None), executor=executor,
        script_instructions="", correction_instructions="", quality_instructions="",
        output_dir=str(tmp_path), plot_fn=lambda data, info: b"plot", r2_threshold=THRESHOLD)


def run_series(tmp_path, monkeypatch, *, names, anchors, follower_r2, refits=None, regimes=None,
               max_series_refits=None, fresh_script="FRESH: np.load('data.npy')", cold_start=None,
               state_extra=None, executor=None):
    """Drive the real series + refit path. ``anchors``: name -> canned anchor
    kwargs (the QC loop's output); ``follower_r2``: name -> the R² the
    replayed script "achieves"; ``refits``: name -> canned refit kwargs (the
    refit QC loop's output); ``regimes``: list of index lists."""
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    executor = executor if executor is not None else FakeExecutor(follower_r2)
    ctrl = _controller(tmp_path, executor)

    def fake_best_of_n(state, curve_data, data_path, spectrum_name, spectrum_idx, **kw):
        spec = anchors[spectrum_name]
        # the real QC loop stages the unit's data as spectrum_NNNN/data.npy; the canned one does too
        d = Path(tmp_path) / f"spectrum_{spectrum_idx:04d}"
        d.mkdir(parents=True, exist_ok=True)
        try:
            np.save(d / "data.npy", np.asarray(curve_data, dtype=float))
        except Exception:  # noqa: BLE001 - a canned anchor whose data is not an array
            pass
        return _canned_anchor(spectrum_name, spectrum_idx, **spec)
    monkeypatch.setattr(ctrl, "_fit_with_quality_control_best_of_n", fake_best_of_n)
    # a follower with no base script (its anchor failed) generates fresh code:
    # the model is a mock, so hand it a script string
    monkeypatch.setattr(ctrl, "_generate_fitting_script", lambda *a, **k: fresh_script)
    monkeypatch.setattr(ctrl, "_check_plan_conformance", lambda state, script: None)

    state = {"num_spectra": len(names), "is_single_spectrum": False,
             "spectrum_stack": np.stack([_spectrum(i) for i in range(len(names))]),
             "locked_fitting_config": {"physical_model": "M1"}, "system_info": {},
             "spectrum_names": list(names)}
    if regimes:
        state["series_analysis_plan"] = {"regimes": [
            {"name": f"R{k + 1}", "spectrum_indices": idxs} for k, idxs in enumerate(regimes)]}
        state["regime_configs"] = {i: {"physical_model": "M1"} for idxs in regimes for i in idxs}
    if max_series_refits is not None:
        state["max_series_refits"] = max_series_refits
    if cold_start:                       # a script-bank cold start: the anchor is asked to reuse it
        state["_cold_start_reuse"] = {"script": cold_start, "id": "bank-1"}
    state.update(state_extra or {})
    state = ctrl.execute(state)
    # the stack names units spectrum_NNNN; map the caller's names onto them
    # (the controller names from the stack, so anchors/follower_r2 use those)
    refitter = _refitter(tmp_path, executor)
    if refits:
        def fake_refit(state, curve_data, data_path, spectrum_name, spectrum_idx, **kw):
            spec = refits.get(spectrum_name)
            if spec is None:
                return {"success": False, "error": "no refit canned", "index": spectrum_idx, "name": spectrum_name}
            return _canned_anchor(spectrum_name, spectrum_idx, **spec)
        monkeypatch.setattr(refitter._fitting_helper, "_fit_with_quality_control", fake_refit)
    else:
        monkeypatch.setattr(refitter._fitting_helper, "_fit_with_quality_control",
                            lambda *a, **k: {"success": False, "error": "no refit", "index": 0, "name": ""})
    state = refitter.execute(state)
    return state, executor


def compile_results(tmp_path, state):
    """The agent's real _compile_results on a bare agent."""
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    agent = object.__new__(CurveFittingAgent)
    agent.output_dir = Path(tmp_path)
    agent.logger = logging.getLogger("series_path.agent")
    agent._validate_scientific_claims = lambda claims: claims
    agent._maybe_stage_t2_solutions = lambda state: []
    agent._bank_series_scripts = lambda state: None
    state.setdefault("synthesis_result", {"detailed_analysis": "d", "scientific_claims": [{"claim": "c"}]})
    state.setdefault("task_mode", "fitting")
    agent._save_fitting_scripts(state)          # as analyze() does before compiling
    return agent._compile_results(state)


NAMES = ["spectrum_0000", "spectrum_0001", "spectrum_0002"]
OK = {"r2": 0.98, "approved": True, "script": "M1"}
SALVAGED = {"r2": 0.80, "approved": False, "script": "M1", "warning": "R² = 0.8000 below threshold 0.95"}


def _units(results):
    return [(u["name"], u.get("role"), u.get("adaptively_refitted", False), u.get("fitted_from"),
             round((u.get("fit_quality") or {}).get("r_squared") or 0.0, 2),
             bool((u.get("quality_history") or {}).get("approved")),
             u.get("regime")) for u in results["individual_results"]]


def test_clean_series_verifies_through_the_real_path(tmp_path, monkeypatch):
    state, ex = run_series(tmp_path, monkeypatch, names=NAMES, anchors={"spectrum_0000": OK},
                           follower_r2={"spectrum_0001": 0.97, "spectrum_0002": 0.96})
    results = compile_results(tmp_path, state)
    units = _units(results)
    assert units[0][1] == "anchor" and all(u[3] == "locked_script" for u in units[1:]), units
    assert [n for n, _ in ex.calls] == ["spectrum_0001", "spectrum_0002"]   # the followers replayed M1
    v = analysis_verdict(results)
    assert v["verified"], (v, units)
    assert series_anchor_unit(results) == "spectrum_0000"


def test_salvaged_anchor_refit_to_approved_does_not_vouch_for_followers(tmp_path, monkeypatch):
    """Round 5's M1/M2 case, through the real refit path."""
    state, ex = run_series(tmp_path, monkeypatch, names=NAMES, anchors={"spectrum_0000": SALVAGED},
                           follower_r2={"spectrum_0001": 0.82, "spectrum_0002": 0.84},
                           refits={"spectrum_0000": {"r2": 0.97, "approved": True, "script": "M2"},
                                   "spectrum_0001": {"r2": 0.81, "approved": False, "script": "M2"},
                                   "spectrum_0002": {"r2": 0.81, "approved": False, "script": "M2"}})
    results = compile_results(tmp_path, state)
    units = _units(results)
    assert units[0][2] is True and units[0][4] == 0.97, units          # the anchor was refit to M2
    assert all(u[2] is False for u in units[1:]), units                 # the followers stayed on M1
    v = analysis_verdict(results)
    assert not v["verified"], (v, units)
    assert "spectrum_0001" in v["reason"] or "spectrum_0000" in v["reason"]


def test_two_regimes_each_judge_their_own_recipe(tmp_path, monkeypatch):
    """Round 6: regime 1 salvaged-then-refit (followers on M1) must block even
    when regime 2 is clean and refit to approved; regime 1 clean with regime
    2 salvaged-then-refit (its followers refit too) must verify and never
    blame regime 1."""
    names = [f"spectrum_{i:04d}" for i in range(6)]
    regimes = [[0, 1, 2], [3, 4, 5]]
    # case A: regime 1 launders, regime 2 clean → not verified
    state, _ = run_series(tmp_path / "a", monkeypatch, names=names, regimes=regimes,
                          anchors={"spectrum_0000": SALVAGED, "spectrum_0003": {"r2": 0.93, "approved": True, "script": "M3"}},
                          follower_r2={"spectrum_0001": 0.82, "spectrum_0002": 0.83, "spectrum_0004": 0.97, "spectrum_0005": 0.97},
                          refits={"spectrum_0000": {"r2": 0.97, "approved": True, "script": "M2"},
                                  "spectrum_0003": {"r2": 0.97, "approved": True, "script": "M4"},
                                  "spectrum_0001": {"r2": 0.81, "approved": False, "script": "M2"},
                                  "spectrum_0002": {"r2": 0.81, "approved": False, "script": "M2"}})
    results = compile_results(tmp_path / "a", state)
    v = analysis_verdict(results)
    assert not v["verified"], (v, _units(results))
    assert "spectrum_0004" not in v["reason"] and "spectrum_0005" not in v["reason"], v
    # case B: regime 1 clean, regime 2 salvaged then everything in it refit to approved → verified
    state, _ = run_series(tmp_path / "b", monkeypatch, names=names, regimes=regimes,
                          anchors={"spectrum_0000": OK, "spectrum_0003": {**SALVAGED, "script": "M3"}},
                          follower_r2={"spectrum_0001": 0.97, "spectrum_0002": 0.97, "spectrum_0004": 0.82, "spectrum_0005": 0.83},
                          refits={"spectrum_0003": {"r2": 0.97, "approved": True, "script": "M4"},
                                  "spectrum_0004": {"r2": 0.97, "approved": True, "script": "M4"},
                                  "spectrum_0005": {"r2": 0.97, "approved": True, "script": "M4"}})
    results = compile_results(tmp_path / "b", state)
    v = analysis_verdict(results)
    assert v["verified"], (v, _units(results))


def test_cut_and_failed_anchors_through_the_real_path(tmp_path, monkeypatch):
    # a budget-cut anchor: unverified, followers replay it → not verified, blamed on the anchor
    state, _ = run_series(tmp_path / "cut", monkeypatch, names=NAMES,
                          anchors={"spectrum_0000": {"r2": 0.80, "approved": False, "script": "M1", "unverified": True}},
                          follower_r2={"spectrum_0001": 0.97, "spectrum_0002": 0.97})
    v = analysis_verdict(compile_results(tmp_path / "cut", state))
    assert not v["verified"] and v["reason"].startswith("verification did not finish") and "spectrum_0000" in v["reason"], v
    # a FAILED anchor: no base script, the followers are fresh code with no verifier
    state, ex = run_series(tmp_path / "failed", monkeypatch, names=NAMES,
                           anchors={"spectrum_0000": {"r2": 0.0, "approved": False, "script": None, "failed": True}},
                           follower_r2={"spectrum_0001": 0.99, "spectrum_0002": 0.99})
    results = compile_results(tmp_path / "failed", state)
    units = _units(results)
    assert all(u[3] == "fresh_code" for u in units[1:]), units
    v = analysis_verdict(results)
    assert not v["verified"] and "without a locked recipe" in v["reason"], v


def test_fresh_code_followers_are_stamped_on_the_parallel_path(tmp_path, monkeypatch):
    """The parallel drain (`SCILINK_CURVE_FIT_WORKERS` > 1) stamps the same
    verdicts as the serial loop."""
    executor = FakeExecutor({"spectrum_0001": 0.97, "spectrum_0002": 0.97})
    ctrl = _controller(tmp_path, executor)
    ctrl.parallel_workers = 2
    monkeypatch.setattr(ctrl, "_fit_with_quality_control_best_of_n",
                        lambda state, curve_data, data_path, spectrum_name, spectrum_idx, **kw:
                        _canned_anchor(spectrum_name, spectrum_idx, **OK))
    state = {"num_spectra": 3, "is_single_spectrum": False, "spectrum_stack": np.stack([_spectrum(i) for i in range(3)]),
             "locked_fitting_config": {"physical_model": "M1"}, "system_info": {}}
    state = ctrl.execute(state)
    uv = [r.get("unit_verdict") for r in state["series_results"]]
    assert uv[0]["own_gate"] and uv[0]["verified"]
    assert all(u and u["verified"] and u.get("recipe_of") == "spectrum_0000" for u in uv[1:]), uv


def test_refit_outcomes_through_the_real_path(tmp_path, monkeypatch):
    # salvaged anchor, then the anchor AND its followers refit to approved fits → verified
    state, _ = run_series(tmp_path / "all", monkeypatch, names=NAMES, anchors={"spectrum_0000": SALVAGED},
                          follower_r2={"spectrum_0001": 0.82, "spectrum_0002": 0.84},
                          refits={n: {"r2": 0.97, "approved": True, "script": "M2"} for n in NAMES})
    results = compile_results(tmp_path / "all", state)
    v = analysis_verdict(results)
    assert v["verified"], (v, _units(results))
    # the accepted opposite direction: an approved anchor refit to a salvaged unit is a salvaged row
    state, _ = run_series(tmp_path / "flip", monkeypatch, names=NAMES,
                          anchors={"spectrum_0000": {"r2": 0.92, "approved": True, "script": "M1"}},
                          follower_r2={"spectrum_0001": 0.80, "spectrum_0002": 0.80},   # flag the followers
                          refits={"spectrum_0000": {"r2": 0.93, "approved": False, "script": "M2",
                                                    "warning": "R² = 0.9300 below threshold 0.95"},
                                  "spectrum_0001": {"r2": 0.93, "approved": False, "script": "M2",
                                                    "warning": "R² = 0.9300 below threshold 0.95"},
                                  "spectrum_0002": {"r2": 0.93, "approved": False, "script": "M2",
                                                    "warning": "below"}})
    results = compile_results(tmp_path / "flip", state)
    v = analysis_verdict(results)
    assert not v["verified"] and "salvaged" in v["reason"], (v, _units(results))


def test_the_recipe_is_the_script_the_followers_replayed(tmp_path, monkeypatch):
    """An approved M1 anchor refit to an approved M2 while the followers stay
    on M1: the series verifies; the driver's recipe record holds M1 (recorded
    when the script was locked, untouched by the refit); the agent's folder
    gets nothing new; the board copies M1 into its own folder."""
    from scilink.agents.exp_agents._verification_record import series_recipes
    from scilink.agents.exp_agents.controllers.curve_fitting_controllers import _load_prior_curve_fit_state
    from scilink.agents.meta_agent.board import Board, post_delegation
    # the anchor sits in the soft band (approved by the verifier at 0.92) so the outlier pass flags
    # it below threshold and the refit pass refits it
    state, _ = run_series(tmp_path, monkeypatch, names=NAMES, anchors={"spectrum_0000": {**OK, "r2": 0.92}},
                          follower_r2={"spectrum_0001": 0.93, "spectrum_0002": 0.97},
                          refits={"spectrum_0000": {"r2": 0.99, "approved": True, "script": "M2"},
                                  "spectrum_0001": {"r2": 0.79, "approved": False, "script": "M2"}})
    results = compile_results(tmp_path, state)
    units = _units(results)
    assert units[0][2] is True, units                              # the anchor was refit
    v = analysis_verdict(results)
    assert v["verified"], (v, units)
    recipes = series_recipes(results)
    assert [(r["regime"], r["unit"], r["verified"], r["script"]) for r in recipes] == [("default", "spectrum_0000", True, "M1")]
    assert series_anchor_unit(results) == "spectrum_0000"
    # the agent's folder: the unit scripts as always, nothing else
    assert sorted(p.name for p in (tmp_path / "scripts").glob("*.py")) == ["spectrum_0000.py", "spectrum_0001.py", "spectrum_0002.py"]
    assert (tmp_path / "scripts" / "spectrum_0000.py").read_text() == "M2"
    # a reuse of this run replays M1, the locked recipe its table rests on, and says so (#704);
    # a reuse that NAMES the refit's script file replays that (#705)
    from scilink.agents.exp_agents.controllers.curve_fitting_controllers import _first_prior_curve_fit_script
    (tmp_path / "analysis_results.json").write_text(json.dumps({"status": "success", "locked_recipes": results["locked_recipes"]}))
    assert _first_prior_curve_fit_script({"prior_analysis_paths": [str(tmp_path)]}) == (
        "M1", f"{tmp_path.name}: spectrum_0000.py (the series' locked recipe, the anchor refit since)")
    assert _first_prior_curve_fit_script({"prior_analysis_paths": [str(tmp_path / "scripts" / "spectrum_0000.py")]}) == (
        "M2", f"{tmp_path.name}: spectrum_0000.py (the script file named)")
    # the board copies M1 into a folder of its own
    board = Board(tmp_path / "meta")
    row = {"analysis_id": "series_1", "status": "success", "output_directory": str(tmp_path), "agent_name": "CurveFittingAgent",
           **analysis_verdict(results), "recipe_unit": series_anchor_unit(results), "series": True, "recipes": recipes}
    ids = post_delegation(board, {"index": 1, "label": "Raman series", "mode": "analysis", "status": "success"},
                          {"key_findings": ["[series_1] anatase"], "analyses": [row]})
    recipe = next(board.get(f) for f in ids if board.get(f)["kind"] == "recipe")
    assert recipe["status"] == "verified" and recipe["payload"]["unit"] == "spectrum_0000"
    assert recipe["payload"]["path"].endswith("meta/swarm/recipes/01_Raman_series/series_1/spectrum_0000.py")
    assert Path(recipe["payload"]["path"]).read_text() == "M1"
    # and the board's copy is itself a recipe a reuse can name (#705): the same script, the same
    # reuse path, no run folder needed
    copy = recipe["payload"]["path"]
    assert _first_prior_curve_fit_script({"prior_analysis_paths": [copy]}) == (
        "M1", f"{Path(copy).parent.name}: spectrum_0000.py (the script file named)")
    # ... but it is NOT a run: anchor_dir is None, so the realtime profile and the live loop refuse it
    anchor_dir, summary, text, label = _load_prior_curve_fit_state(copy)
    assert (anchor_dir, summary, text, label) == (None, None, "M1", "spectrum_0000.py (the script file named)")
    assert _load_prior_curve_fit_state(str(Path(copy).parent))[0] is None   # the board folder is no run: nothing
    from scilink.live.modality import CurveModality
    assert CurveModality().anchor_script(copy) == (None, None)             # the loop arms on runs only
    # a file INSIDE the run names the run for the loop, and the loop arms on what the run replays (M1),
    # not on the named refit script (M2) its frames would never see
    assert CurveModality().anchor_script(str(tmp_path / "scripts" / "spectrum_0000.py")) == ("M1", tmp_path)
    assert CurveModality().anchor_script(str(tmp_path)) == ("M1", tmp_path)


def test_a_failed_follower_refit_leaves_the_reuse_pick_unchanged(tmp_path, monkeypatch):
    """A follower that failed and was refit: the agent's folder is exactly what
    it was, and a reuse replays the series' locked recipe (#704), which here is
    also the anchor's current script."""
    from scilink.agents.exp_agents.controllers.curve_fitting_controllers import _load_prior_curve_fit_state
    names4 = [f"spectrum_{i:04d}" for i in range(4)]        # outlier detection needs three successes
    state, ex = run_series(tmp_path, monkeypatch, names=names4, anchors={"spectrum_0000": OK},
                           follower_r2={"spectrum_0001": 0.97, "spectrum_0002": None, "spectrum_0003": 0.96},   # one replay fails
                           refits={"spectrum_0002": {"r2": 0.98, "approved": True, "script": "M2"}})
    results = compile_results(tmp_path, state)
    units = _units(results)
    assert units[2][2] is True and units[2][4] == 0.98, units          # the failed follower was refit
    assert analysis_verdict(results)["verified"], (analysis_verdict(results), units)
    assert sorted(p.name for p in (tmp_path / "scripts").glob("*.py")) == [f"{n}.py" for n in names4]
    assert (tmp_path / "scripts" / "spectrum_0002.py").read_text() == "M2"
    (tmp_path / "analysis_results.json").write_text(json.dumps({"status": "success", "locked_recipes": results["locked_recipes"]}))
    anchor_dir, summary, script_text, label = _load_prior_curve_fit_state(str(tmp_path))
    assert script_text == "M1" and label == "spectrum_0000.py (the series' locked recipe)"
    # a run from before the record: the first unit script, as before
    (tmp_path / "analysis_results.json").write_text(json.dumps({"status": "success"}))
    assert _load_prior_curve_fit_state(str(tmp_path))[2:] == ("M1", "spectrum_0000.py (representative of the series)")


def test_a_failed_regime_anchor_names_its_own_regime(tmp_path, monkeypatch):
    names = [f"spectrum_{i:04d}" for i in range(6)]
    state, _ = run_series(tmp_path, monkeypatch, names=names, regimes=[[0, 1, 2], [3, 4, 5]],
                          anchors={"spectrum_0000": OK, "spectrum_0003": {"r2": 0.0, "approved": False, "script": None, "failed": True}},
                          follower_r2={n: 0.97 for n in names})
    results = compile_results(tmp_path, state)
    units = _units(results)
    assert [u[6] for u in units] == ["R1", "R1", "R1", "R2", "R2", "R2"], units   # regime on every unit
    v = analysis_verdict(results)
    assert not v["verified"] and "spectrum_0004" in v["reason"] and "without a locked recipe" in v["reason"], v


def test_a_good_reuse_series_verifies_and_a_failed_reuse_is_salvaged(tmp_path, monkeypatch):
    """Round 7: a series whose anchor is a locked-script reuse. A good
    verdict is the replay gate passing (verified); a reuse whose script
    failed, re-derived from scratch, carries the schema-drift caveat the
    controller attaches AFTER the QC loop — the stamp must come after it."""
    reuse_ok = {"r2": 0.97, "approved": True, "script": "PRIOR", "reused": "good"}
    state, _ = run_series(tmp_path / "good", monkeypatch, names=NAMES, anchors={"spectrum_0000": reuse_ok},
                          follower_r2={"spectrum_0001": 0.97, "spectrum_0002": 0.96}, cold_start="PRIOR")
    results = compile_results(tmp_path / "good", state)
    units = _units(results)
    uv = results["individual_results"][0]["unit_verdict"]
    assert "quality_history" not in results["individual_results"][0] or not results["individual_results"][0]["quality_history"]
    assert {k: uv[k] for k in ("verified", "reason", "regime", "own_gate", "decided_by")} == {
        "verified": True, "reason": "locked-script reuse passed the replay gate", "regime": "default",
        "own_gate": True, "decided_by": "replay_gate"}, uv
    assert analysis_verdict(results)["verified"], (analysis_verdict(results), units)
    # the reuse attempted, the script could not run, full QC re-derived the model (approved):
    # the controller stamps reuse_validity script_failed + quality_warning → salvaged
    state, _ = run_series(tmp_path / "failed", monkeypatch, names=NAMES, anchors={"spectrum_0000": OK},
                          follower_r2={"spectrum_0001": 0.97, "spectrum_0002": 0.96}, cold_start="PRIOR")
    results = compile_results(tmp_path / "failed", state)
    a = results["individual_results"][0]
    assert a["reuse_validity"]["verdict"] == "script_failed" and a.get("quality_warning")
    v = analysis_verdict(results)
    assert not v["verified"] and v["reason"].startswith("salvaged best-available result"), v


# ---------------------------------------------------------------- parity
#: Scenarios where the stamped verdict is ALLOWED to differ from the legacy
#: reconstruction; anything else that differs fails the parity test. The
#: first four were the review's intended changes (the legacy rule, with
#: regime carried, reaches the same answers today); the last is a real
#: difference: the legacy rule held a FOLLOWER refit only to "finished, not
#: unverified" (a round-3 relaxation), the stamp judges every refit by its
#: own gate, so a refit that stayed salvaged is a salvaged row in the table
#: — the same reasoning as for a salvaged anchor refit.
PARITY_ALLOWED = {"two_regimes_launder", "two_regimes_clean_r2_refit", "m1_m2", "laundered_anchor",
                  "follower_refit_salvaged"}


def _scenarios(tmp_path, monkeypatch):
    """Every real-path shape of this module, named."""
    S = {}
    S["clean"] = run_series(tmp_path / "s1", monkeypatch, names=NAMES, anchors={"spectrum_0000": OK},
                            follower_r2={"spectrum_0001": 0.97, "spectrum_0002": 0.96})[0]
    S["m1_m2"] = run_series(tmp_path / "s2", monkeypatch, names=NAMES, anchors={"spectrum_0000": SALVAGED},
                            follower_r2={"spectrum_0001": 0.82, "spectrum_0002": 0.84},
                            refits={"spectrum_0000": {"r2": 0.97, "approved": True, "script": "M2"},
                                    "spectrum_0001": {"r2": 0.81, "approved": False, "script": "M2"},
                                    "spectrum_0002": {"r2": 0.81, "approved": False, "script": "M2"}})[0]
    names6 = [f"spectrum_{i:04d}" for i in range(6)]
    regimes = [[0, 1, 2], [3, 4, 5]]
    S["two_regimes_launder"] = run_series(tmp_path / "s3", monkeypatch, names=names6, regimes=regimes,
                                          anchors={"spectrum_0000": SALVAGED, "spectrum_0003": {"r2": 0.93, "approved": True, "script": "M3"}},
                                          follower_r2={"spectrum_0001": 0.82, "spectrum_0002": 0.83, "spectrum_0004": 0.97, "spectrum_0005": 0.97},
                                          refits={"spectrum_0000": {"r2": 0.97, "approved": True, "script": "M2"},
                                                  "spectrum_0003": {"r2": 0.97, "approved": True, "script": "M4"},
                                                  "spectrum_0001": {"r2": 0.81, "approved": False, "script": "M2"},
                                                  "spectrum_0002": {"r2": 0.81, "approved": False, "script": "M2"}})[0]
    S["two_regimes_clean_r2_refit"] = run_series(tmp_path / "s4", monkeypatch, names=names6, regimes=regimes,
                                                 anchors={"spectrum_0000": OK, "spectrum_0003": {**SALVAGED, "script": "M3"}},
                                                 follower_r2={"spectrum_0001": 0.97, "spectrum_0002": 0.97, "spectrum_0004": 0.82, "spectrum_0005": 0.83},
                                                 refits={"spectrum_0003": {"r2": 0.97, "approved": True, "script": "M4"},
                                                         "spectrum_0004": {"r2": 0.97, "approved": True, "script": "M4"},
                                                         "spectrum_0005": {"r2": 0.97, "approved": True, "script": "M4"}})[0]
    S["cut_anchor"] = run_series(tmp_path / "s5", monkeypatch, names=NAMES,
                                 anchors={"spectrum_0000": {"r2": 0.80, "approved": False, "script": "M1", "unverified": True}},
                                 follower_r2={"spectrum_0001": 0.97, "spectrum_0002": 0.97})[0]
    S["failed_anchor"] = run_series(tmp_path / "s6", monkeypatch, names=NAMES,
                                    anchors={"spectrum_0000": {"r2": 0.0, "approved": False, "script": None, "failed": True}},
                                    follower_r2={"spectrum_0001": 0.99, "spectrum_0002": 0.99})[0]
    S["all_refit_ok"] = run_series(tmp_path / "s7", monkeypatch, names=NAMES, anchors={"spectrum_0000": SALVAGED},
                                   follower_r2={"spectrum_0001": 0.82, "spectrum_0002": 0.84},
                                   refits={n: {"r2": 0.97, "approved": True, "script": "M2"} for n in NAMES})[0]
    S["flip"] = run_series(tmp_path / "s8", monkeypatch, names=NAMES,
                           anchors={"spectrum_0000": {"r2": 0.92, "approved": True, "script": "M1"}},
                           follower_r2={"spectrum_0001": 0.80, "spectrum_0002": 0.80},
                           refits={n: {"r2": 0.93, "approved": False, "script": "M2", "warning": "below"} for n in NAMES})[0]
    S["laundered_anchor"] = run_series(tmp_path / "s9", monkeypatch, names=NAMES, anchors={"spectrum_0000": SALVAGED},
                                       follower_r2={"spectrum_0001": 0.82, "spectrum_0002": 0.84},
                                       refits={"spectrum_0000": {"r2": 0.86, "approved": False, "script": "M2", "warning": "below"}})[0]
    S["good_reuse"] = run_series(tmp_path / "s10", monkeypatch, names=NAMES,
                                 anchors={"spectrum_0000": {"r2": 0.97, "approved": True, "script": "PRIOR", "reused": "good"}},
                                 follower_r2={"spectrum_0001": 0.97, "spectrum_0002": 0.96}, cold_start="PRIOR")[0]
    S["failed_reuse_rederived"] = run_series(tmp_path / "s11", monkeypatch, names=NAMES, anchors={"spectrum_0000": OK},
                                             follower_r2={"spectrum_0001": 0.97, "spectrum_0002": 0.96}, cold_start="PRIOR")[0]
    names4 = [f"spectrum_{i:04d}" for i in range(4)]
    S["failed_follower_refit"] = run_series(tmp_path / "s12", monkeypatch, names=names4, anchors={"spectrum_0000": OK},
                                            follower_r2={"spectrum_0001": 0.97, "spectrum_0002": None, "spectrum_0003": 0.96},
                                            refits={"spectrum_0002": {"r2": 0.98, "approved": True, "script": "M2"}})[0]
    S["follower_refit_salvaged"] = run_series(tmp_path / "s14", monkeypatch, names=names4, anchors={"spectrum_0000": OK},
                                              follower_r2={"spectrum_0001": 0.97, "spectrum_0002": None, "spectrum_0003": 0.96},
                                              refits={"spectrum_0002": {"r2": 0.86, "approved": False, "script": "M2",
                                                                        "warning": "R² = 0.8600 below threshold 0.95"}})[0]
    S["failed_regime_anchor"] = run_series(tmp_path / "s13", monkeypatch, names=names6, regimes=regimes,
                                           anchors={"spectrum_0000": OK, "spectrum_0003": {"r2": 0.0, "approved": False, "script": None, "failed": True}},
                                           follower_r2={n: 0.97 for n in names6})[0]
    return {k: compile_results(tmp_path / f"s{i + 1}", st) for i, (k, st) in enumerate(S.items())}


def test_parity_of_the_stamped_and_legacy_verdicts(tmp_path, monkeypatch):
    """The stamped verdict and the legacy reconstruction agree on every
    real-path shape, except where the change was intended (PARITY_ALLOWED):
    a future divergence fails here instead of being found in review."""
    from scilink.agents.exp_agents._verification_record import legacy_series_verdict
    differ, agree = {}, []
    for name, results in _scenarios(tmp_path, monkeypatch).items():
        stamped = analysis_verdict(results)
        assert all(isinstance(u.get("unit_verdict"), dict) for u in results["individual_results"] if u.get("success")), name
        legacy = legacy_series_verdict(results)
        if stamped["verified"] == legacy["verified"]:
            agree.append(name)
        else:
            differ[name] = (stamped, legacy)
    unexpected = {k: v for k, v in differ.items() if k not in PARITY_ALLOWED}
    assert not unexpected, unexpected
    # the one difference that exists today, and the stricter answer is the stamped one
    assert set(differ) == {"follower_refit_salvaged"}, differ
    assert differ["follower_refit_salvaged"][0]["verified"] is False
    assert set(agree) >= {"clean", "cut_anchor", "failed_anchor", "all_refit_ok", "flip", "good_reuse",
                          "failed_reuse_rederived", "failed_follower_refit", "failed_regime_anchor"}, (agree, differ)


def test_the_realtime_profile_and_the_loop_refuse_a_bare_script_and_a_reuse_names_its_regime(tmp_path, monkeypatch):
    """Round 1 of #707: a script file on its own is a recipe for a reuse, and
    nothing else — the realtime profile (locked config, drift fingerprint)
    raises as it did on main; a unit whose name is not file-system safe is
    still found for the refit check; an unverified recipe says so in its
    label; a non-string script or a missing unit is not a recipe."""
    import json
    from scilink.agents.exp_agents._verification_record import prior_recipe_scripts, series_recipes, unit_script_name
    from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
        _load_prior_curve_fit_state, _prior_curve_fit_recipes, _prior_curve_fit_block)
    bare = tmp_path / "copy" / "fitting_script.py"
    bare.parent.mkdir()
    bare.write_text("F")
    assert _load_prior_curve_fit_state(str(bare)) == (None, None, "F", "fitting_script.py (the script file named)")
    assert _prior_curve_fit_recipes({"prior_analysis_paths": [str(bare)]}) == [("F", "copy: fitting_script.py (the script file named)")]
    assert "### Prior script file: fitting_script.py" in _prior_curve_fit_block({"prior_analysis_paths": [str(bare)]})
    # the realtime entry raises on a bare file exactly as on main (anchor_dir is None)
    anchor_dir, prior_summary, _ps, _pl = _load_prior_curve_fit_state(str(bare))
    assert anchor_dir is None                       # what analyze(profile="realtime") tests before raising
    # a two-regime prior run with an unsafe unit name and an unverified second regime
    run = tmp_path / "run"
    (run / "scripts").mkdir(parents=True)
    (run / "series_fit_results.json").write_text("{}")
    (run / "scripts" / "T_300K.py").write_text("M1-refit")           # the anchor was refit (saved under the safe name)
    (run / "scripts" / "T_500K.py").write_text("M2")
    (run / "analysis_results.json").write_text(json.dumps({"locked_recipes": {
        "R1": {"unit": "T=300K", "index": 0, "regime": "R1", "script": "M1", "verdict": {"verified": True}},
        "R2": {"unit": "T=500K", "index": 1, "regime": "R2", "script": "M2", "verdict": {"verified": False, "reason": "salvaged"}},
        "bad": {"unit": "x", "index": 2, "script": 123}, "nounit": {"index": 3, "script": "S"}}}))
    assert unit_script_name("T=300K") == "T_300K"
    assert [r["unit"] for r in series_recipes(json.loads((run / "analysis_results.json").read_text()))] == ["T=300K", "T=500K"]
    assert prior_recipe_scripts(run, single_name="fitting_script.py") == [
        ("M1", "T_300K.py (the series' locked recipe, regime R1, 1 of 2, the anchor refit since)"),
        ("M2", "T_500K.py (the series' locked recipe, regime R2, 2 of 2, its anchor's gate did not pass)")]
    assert [src for _, src in _prior_curve_fit_recipes({"prior_analysis_paths": [str(run)]})] == [
        "run: T_300K.py (the series' locked recipe, regime R1, 1 of 2, the anchor refit since)",
        "run: T_500K.py (the series' locked recipe, regime R2, 2 of 2, its anchor's gate did not pass)"]


def test_a_multi_regime_prior_run_is_replayed_regime_by_regime(tmp_path, monkeypatch):
    """A prior series that locked one model below a transition and another
    above it: a reuse replays the recipes in lock order and keeps the first
    the gate calls good; when none is good, the first that executed is kept,
    poor and flagged, as a single recipe would be."""
    import json
    from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
        UnifiedSeriesProcessingController, _prior_curve_fit_recipes)
    run = tmp_path / "prior"
    (run / "scripts").mkdir(parents=True)
    (run / "series_fit_results.json").write_text("{}")
    (run / "analysis_results.json").write_text(json.dumps({"locked_recipes": {
        "low": {"unit": "spectrum_0000", "index": 0, "regime": "low", "script": "LOW", "verdict": {"verified": True}},
        "high": {"unit": "spectrum_0003", "index": 3, "regime": "high", "script": "HIGH", "verdict": {"verified": True}}}}))
    assert [t for t, _ in _prior_curve_fit_recipes({"prior_analysis_paths": [str(run)]})] == ["LOW", "HIGH"]
    # the executor scores the fit by which recipe ran: LOW fits the new spectrum poorly, HIGH well
    out = tmp_path / "new"
    out.mkdir()
    ctrl = _controller(out, FakeExecutor({"spectrum_0000": 0.97}))
    ran = []

    def fit(state, curve_data, data_path, spectrum_name, spectrum_idx, base_script=None, **kw):
        ran.append(base_script)
        r2 = {"LOW": 0.80, "HIGH": 0.985}.get(base_script, 0.5)
        return {"index": spectrum_idx, "name": spectrum_name, "success": True, "script": base_script,
                "fit_quality": {"r_squared": r2}, "fitted_parameters": {}, "model_type": base_script}
    monkeypatch.setattr(ctrl, "_fit_single_spectrum", fit)
    monkeypatch.setattr(ctrl, "_run_verification_loop", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no QC loop")), raising=False)
    state = {"num_spectra": 1, "is_single_spectrum": True, "spectra_data": [np.zeros((10, 2))], "spectrum_names": ["spectrum_0000"],
             "prior_analysis_paths": [str(run)], "reuse_locked_script": True, "system_info": {}}
    from scilink.agents.exp_agents._qc_engine import QCItemContext
    recipes = _prior_curve_fit_recipes(state)
    state["_reuse_candidates"] = [{"script": t, "source": s} for t, s in recipes]
    ctx = QCItemContext(state=state, data=np.zeros((10, 2)), data_path="new.txt", item_name="spectrum_0000", item_idx=0,
                        reuse_script=recipes[0][0], reuse_source=recipes[0][1])
    res = ctrl.qc_try_reuse(ctx)
    assert ran == ["LOW", "HIGH"] and res["script"] == "HIGH"
    rv = res["reuse_validity"]
    assert rv["verdict"] == "good" and rv["recipes_tried"] == 2 and "regime high, 2 of 2" in rv["source"] and "2 regime recipes tried" in rv["message"]
    # none good: the first executed is kept, poor
    ran.clear()
    monkeypatch.setattr(ctrl, "_fit_single_spectrum", lambda *a, base_script=None, **k: {
        "index": 0, "name": "spectrum_0000", "success": True, "script": base_script, "fit_quality": {"r_squared": 0.6}, "fitted_parameters": {}})
    res = ctrl.qc_try_reuse(ctx)
    assert res["script"] == "LOW" and res["reuse_validity"]["verdict"] == "poor" and res["reuse_validity"]["recipes_tried"] == 2
    assert "regime low, 1 of 2" in res["reuse_validity"]["source"] and res.get("quality_warning")
    # a single recipe: no candidate bookkeeping on the verdict at all
    state["_reuse_candidates"] = []
    ctx = QCItemContext(state=state, data=np.zeros((10, 2)), data_path="new.txt", item_name="spectrum_0000", item_idx=0,
                        reuse_script="LOW", reuse_source="prior")
    res = ctrl.qc_try_reuse(ctx)
    assert "recipes_tried" not in res["reuse_validity"] and res["reuse_validity"]["source"] == "prior"


class ScriptKeyedExecutor:
    """R² (or a failure) by the SCRIPT that runs, and a figure whose bytes
    name that script — so what is left on disk can be checked."""
    timeout = 30

    def __init__(self, r2_by_script, centers=None, extra_params=None, extra_quality=None):
        self.r2_by_script, self.calls, self.centers = r2_by_script, [], centers or {}
        self.extra_params = extra_params or {}          # script -> extra top-level fitted parameters
        self.extra_quality = extra_quality or {}        # script -> extra fit_quality metrics (a skill's own)

    def execute_script(self, script, working_dir=None, timeout=None, **kw):
        wd = Path(working_dir)
        self.calls.append((script, wd))
        r2 = self.r2_by_script.get(script)
        if r2 is None:
            return {"status": "error", "stdout": "", "stderr": "Traceback: boom", "message": f"{script} failed"}
        (wd / "visualization.png").write_bytes(f"png {script}".encode())
        (wd / "fit.npy").write_bytes(f"fit {script}".encode())
        out = {"model_type": script, "parameters": {"peak_1": {"center": self.centers.get(script, 144.0), "amplitude": 1.0},
                                                    **self.extra_params.get(script, {})},
               "fit_quality": {"r_squared": r2, **self.extra_quality.get(script, {})}}
        return {"status": "success", "stdout": "FIT_RESULTS_JSON:" + json.dumps(out), "stderr": "", "message": ""}


def _replay(tmp_path, monkeypatch, r2_by_script, *, strict=False, repaired=None, prior=None, data=None, centers=None,
            extra_params=None, extra_quality=None, quality_gate=None, state_extra=None, controller=None):
    """The real qc_try_reuse → _fit_single_spectrum → stage_and_run path on a
    two-regime prior (LOW, HIGH); ``repaired`` is what the correction ladder
    would hand back (counted), ``None`` makes a correction an error. With
    ``prior`` (a run folder) the candidates come through the real pick
    (``_prior_curve_fit_candidates``, fingerprints included)."""
    from scilink.agents.exp_agents._qc_engine import QCItemContext
    from scilink.agents.exp_agents.controllers.curve_fitting_controllers import _prior_curve_fit_candidates
    out = tmp_path / "new"
    out.mkdir(parents=True, exist_ok=True)
    ex = ScriptKeyedExecutor(r2_by_script, centers, extra_params, extra_quality)
    ctrl = controller(out, ex) if controller is not None else _controller(out, ex)
    corrections = []

    def correct(state, script, error):
        corrections.append(error)
        if repaired is None:
            raise AssertionError("the correction ladder ran")
        return repaired, "repaired"
    monkeypatch.setattr(ctrl, "_correct_script", correct)
    monkeypatch.setattr(ctrl, "_correct_script_with_timeout_escalation", correct)
    if prior is not None:
        cands = _prior_curve_fit_candidates({"prior_analysis_paths": [str(prior)]})
        candidates = [{k: c.get(k) for k in ("script", "source", "regime", "unit", "drift_state", "x_range")} for c in cands]
        reuse_candidates = candidates if len(candidates) > 1 else []      # as the series controller sets it
        reuse_gate = cands[0].get("gate") if cands else None
    else:
        recipes = [("LOW", "prior: LOW (regime low, 1 of 2)"), ("HIGH", "prior: HIGH (regime high, 2 of 2)")]
        candidates = [{"script": t, "source": s} for t, s in recipes]
        reuse_candidates, reuse_gate = candidates, None
    state = {"num_spectra": 1, "is_single_spectrum": True, "system_info": {}, "locked_fitting_config": {},
             "_reuse_candidates": reuse_candidates, "_reuse_gate": reuse_gate,
             **({"prior_analysis_paths": [str(prior)], "reuse_locked_script": True} if prior is not None else {})}
    if strict:
        state["_strict_replay"] = True
    if quality_gate is not None:
        state["quality_gate"] = quality_gate
    state.update(state_extra or {})
    ctx = QCItemContext(state=state, data=_spectrum(1) if data is None else data, data_path=str(tmp_path / "new.txt"),
                        item_name="spectrum_0000", item_idx=0, reuse_script=candidates[0]["script"],
                        reuse_source=candidates[0]["source"])
    res = ctrl.qc_try_reuse(ctx)
    return res, ex, corrections, out / "spectrum_0000"


def test_regime_recipes_run_in_their_own_folders_and_the_kept_one_is_promoted(tmp_path, monkeypatch):
    """Round 2 of #707: several recipes never share a working dir, so the kept
    result's figure and fit.npy on disk are its own, whichever ran last; the
    recipes run verbatim first, and the correction ladder is paid once."""
    # LOW poor, HIGH good: HIGH kept and promoted; LOW's figure stays under _candidates
    res, ex, corrections, item = _replay(tmp_path / "a", monkeypatch, {"LOW": 0.80, "HIGH": 0.985})
    assert res["script"] == "HIGH" and res["reuse_validity"]["verdict"] == "good" and res["reuse_validity"]["recipes_tried"] == 2
    assert (item / "visualization.png").read_bytes() == b"png HIGH" and (item / "fit.npy").read_bytes() == b"fit HIGH"
    assert Path(res["visualization_path"]) == item / "visualization.png"
    assert (item / "_candidates" / "recipe_01" / "visualization.png").read_bytes() == b"png LOW"
    assert [wd.name for _, wd in ex.calls] == ["recipe_01", "recipe_02"] and corrections == []
    # neither fits: the FIRST executed (LOW) is kept, poor, and its own figure is on disk although HIGH ran after it
    res, ex, _, item = _replay(tmp_path / "b", monkeypatch, {"LOW": 0.70, "HIGH": 0.75})
    assert res["script"] == "LOW" and res["reuse_validity"]["verdict"] == "poor" and res.get("quality_warning")
    assert (item / "visualization.png").read_bytes() == b"png LOW" and (item / "fit.npy").read_bytes() == b"fit LOW"
    assert "regime low, 1 of 2" in res["reuse_validity"]["source"]
    # LOW raises verbatim, HIGH fits: HIGH kept with no correction spent on LOW
    res, ex, corrections, item = _replay(tmp_path / "c", monkeypatch, {"HIGH": 0.99})
    assert res["script"] == "HIGH" and corrections == [] and (item / "visualization.png").read_bytes() == b"png HIGH"
    # LOW poor, HIGH raises: LOW kept, poor, its figure on disk (HIGH left nothing to overwrite it with)
    res, ex, _, item = _replay(tmp_path / "d", monkeypatch, {"LOW": 0.80})
    assert res["script"] == "LOW" and res["reuse_validity"]["verdict"] == "poor"
    assert (item / "visualization.png").read_bytes() == b"png LOW" and Path(res["visualization_path"]).is_file()
    # both raise verbatim: the first recipe gets the ladder once, in the spectrum's own folder
    res, ex, corrections, item = _replay(tmp_path / "e", monkeypatch, {"REPAIRED": 0.97}, repaired="REPAIRED")
    assert res["script"] == "REPAIRED" and res["reuse_validity"]["verdict"] == "good" and len(corrections) == 1
    assert res["reuse_validity"]["recipes_tried"] == 2 and (item / "visualization.png").read_bytes() == b"png REPAIRED"
    verbatim = [s for s, wd in ex.calls if wd.name.startswith("recipe_")]
    assert verbatim == ["LOW", "HIGH"] and ex.calls[-1][1] == item         # verbatim first, the ladder in the item dir
    # both raise and nothing repairs: None hands the item to re-derivation
    res, ex, corrections, item = _replay(tmp_path / "f", monkeypatch, {}, repaired="STILL_BAD")
    assert res is None and len(corrections) >= 1


def test_a_strict_replay_tries_every_regime_before_failing(tmp_path, monkeypatch):
    """On the fast clock a raising recipe moves on to the next regime, as a
    poor one does; only when none executes is the frame failed."""
    res, ex, corrections, item = _replay(tmp_path / "a", monkeypatch, {"HIGH": 0.99}, strict=True)
    assert res["script"] == "HIGH" and res["reuse_validity"]["verdict"] == "good" and corrections == []
    assert [s for s, _ in ex.calls] == ["LOW", "HIGH"]
    res, ex, corrections, item = _replay(tmp_path / "b", monkeypatch, {}, strict=True)
    assert res is not None and res["reuse_validity"]["verdict"] == "failed" and corrections == []
    assert [s for s, _ in ex.calls] == ["LOW", "HIGH"]                     # both tried, nothing repaired


def test_the_loop_refuses_a_multi_regime_series_anchor(tmp_path):
    import json
    from scilink.live.modality import CurveModality
    run = tmp_path / "prior"
    (run / "scripts").mkdir(parents=True)
    (run / "series_fit_results.json").write_text("{}")
    (run / "analysis_results.json").write_text(json.dumps({"locked_recipes": {
        "low": {"unit": "spectrum_0000", "index": 0, "regime": "low", "script": "LOW", "verdict": {"verified": True}},
        "high": {"unit": "spectrum_0003", "index": 3, "regime": "high", "script": "HIGH", "verdict": {"verified": True}}}}))
    with pytest.raises(ValueError, match="locked 2 regime recipes"):
        CurveModality().anchor_script(str(run))
    (run / "analysis_results.json").write_text(json.dumps({"locked_recipes": {
        "low": {"unit": "spectrum_0000", "index": 0, "regime": "low", "script": "LOW", "verdict": {"verified": True}}}}))
    assert CurveModality().anchor_script(str(run)) == ("LOW", run)


PINNED = [{"component": "peak_2", "parameter": "fwhm", "value": 40.0, "bound": 40.0, "side": "upper"}]


def test_an_approved_anchor_with_a_pinned_parameter_is_degenerate_not_salvaged(tmp_path, monkeypatch):
    """#726: the pinned-at-bound rule writes a quality_warning on a fit the
    gate APPROVED. The series stays unverified (a parameter at its bound is
    not a measured value) but says why: a degenerate fit, not a salvaged one,
    and the followers that replayed it say the same through their recipe."""
    state, _ = run_series(tmp_path, monkeypatch, names=NAMES,
                          anchors={"spectrum_0000": {**OK, "pinned": PINNED}},
                          follower_r2={"spectrum_0001": 0.97, "spectrum_0002": 0.96})
    results = compile_results(tmp_path, state)
    v = analysis_verdict(results)
    assert not v["verified"]
    assert v["reason"].startswith("degenerate fit: peak_2.fwhm = 40 at its upper bound 40"), v
    assert "salvaged" not in v["reason"]
    by_name = {u["name"]: u for u in results["individual_results"]}
    follower = by_name["spectrum_0001"]["unit_verdict"]
    assert not follower["verified"] and "degenerate fit" in follower["reason"] and "salvaged" not in follower["reason"]
    # a salvaged anchor (not approved) that is also pinned keeps the salvage reason
    state, _ = run_series(tmp_path / "salv", monkeypatch, names=NAMES,
                          anchors={"spectrum_0000": {**SALVAGED, "pinned": PINNED}},
                          follower_r2={"spectrum_0001": 0.97, "spectrum_0002": 0.96}, max_series_refits=0)
    v = analysis_verdict(compile_results(tmp_path / "salv", state))
    assert not v["verified"] and v["reason"].startswith("salvaged best-available result"), v
