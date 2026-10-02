"""A replayed recipe is held to the RUN's gate, not to global R² (#712
follow-up 1).

``_detect_outliers`` already holds a series' followers to the effective
``QualityGate`` — a skill's own metric and threshold (``peak_region_r2`` for
XRD profiles and NMR, a figure of merit for workflow skills) — but the reuse
path judged every replay on ``fit_quality.r_squared`` against the driver's
``r2_threshold``, whatever the skill declared. Live: an XRD phase-
identification recipe whose R² is negative by construction (−1.39 on its own
anchor, verified on the figure of merit) could never replay as ``good``, and
a correct low-SNR profile fit (high peak-region R², low global R²) was
``poor`` on reuse while its own run had passed. The behaviour this commit
changes is exactly: a reuse under a non-R² gate is judged on that gate's
metric and threshold; under an R² gate nothing changes (the live
``r2_threshold``, as before).
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_series_verdict_path as curve  # noqa: E402
import test_regime_choice as rc  # noqa: E402

from scilink.agents.exp_agents import _replay  # noqa: E402
from scilink.agents.exp_agents._verification_record import final_verdict_record, unit_verdict_for  # noqa: E402
from scilink.agents.exp_agents.quality_gate import QualityGate, from_mapping, gate_record  # noqa: E402

PROFILE = QualityGate(metric="peak_region_r2", accept_threshold=0.90, hard_reject_threshold=0.55)
FOM = QualityGate(metric="figure_of_merit", accept_threshold=0.70, hard_reject_threshold=0.30, physical_review=False)
CHI2 = QualityGate(metric="reduced_chi2", accept_threshold=1.5, hard_reject_threshold=4.0, direction="lower_is_better")


def test_score_gate_rejects_a_missing_metric_whatever_the_direction():
    g = _replay.ScoreReplayGate(CHI2.is_accept, CHI2.accept_threshold, CHI2.label)
    assert g.judge(None)["verdict"] == "poor" and "not reported" in g.judge(None)["reasons"][0]
    assert g.judge(1.2)["verdict"] == "good" and g.judge(2.0)["verdict"] == "poor"
    assert _replay.ScoreReplayGate(lambda v: v >= 0.9, 0.9, "R²").judge(None)["verdict"] == "poor"


def test_a_reuse_under_a_skill_gate_is_judged_on_the_skills_metric(tmp_path, monkeypatch):
    # a correct low-SNR profile fit: peak-region R² 0.96, global R² 0.80 — poor on main, good here
    res, ex, _, _ = curve._replay(tmp_path / "a", monkeypatch, {"LOW": 0.80, "HIGH": 0.80}, quality_gate=PROFILE,
                                  extra_quality={"LOW": {"peak_region_r2": 0.96}, "HIGH": {"peak_region_r2": 0.50}})
    rv = res["reuse_validity"]
    assert res["script"] == "LOW" and rv["verdict"] == "good" and rv["recipes_tried"] == 1
    assert rv["metric"] == "peak_region_r2" and rv["score"] == 0.96 and rv["threshold"] == 0.90
    assert rv["r_squared"] == 0.80                                   # the fit's R², for portability, unchanged
    assert "peak_region_r2 = 0.9600 meets the acceptance threshold 0.900" in rv["message"]
    uv = unit_verdict_for({**res, "success": True})
    assert uv["verified"] and uv["decided_by"] == "replay_gate"
    fv = final_verdict_record({"status": "success", "reuse_validity": rv})
    assert fv["verified"] and fv["score"] == 0.96 and fv["threshold"] == 0.90          # the gate's score, not R²
    # the first regime's recipe below the skill's bar falls through to the second, as an R² reuse does
    res, ex, _, _ = curve._replay(tmp_path / "b", monkeypatch, {"LOW": 0.99, "HIGH": 0.99}, quality_gate=PROFILE,
                                  extra_quality={"LOW": {"peak_region_r2": 0.60}, "HIGH": {"peak_region_r2": 0.95}})
    assert res["script"] == "HIGH" and res["reuse_validity"]["verdict"] == "good" and res["reuse_validity"]["recipes_tried"] == 2
    # a workflow skill's figure of merit, with R² negative by construction (the phase-identification case)
    res, ex, _, _ = curve._replay(tmp_path / "c", monkeypatch, {"LOW": -1.39, "HIGH": -1.2}, quality_gate=FOM,
                                  extra_quality={"LOW": {"figure_of_merit": 0.82}})
    rv = res["reuse_validity"]
    assert rv["verdict"] == "good" and rv["metric"] == "figure_of_merit" and rv["r_squared"] == -1.39
    assert unit_verdict_for({**res, "success": True})["verified"]
    # the metric missing from the replayed script's output is a reject, not a pass
    res, ex, _, _ = curve._replay(tmp_path / "d", monkeypatch, {"LOW": 0.99, "HIGH": 0.99}, quality_gate=FOM)
    rv = res["reuse_validity"]
    assert rv["verdict"] == "poor" and rv["score"] is None and "not reported" in rv["message"]
    # a lower-is-better gate
    res, ex, _, _ = curve._replay(tmp_path / "e", monkeypatch, {"LOW": 0.99, "HIGH": 0.99}, quality_gate=CHI2,
                                  extra_quality={"LOW": {"reduced_chi2": 1.1}})
    assert res["reuse_validity"]["verdict"] == "good" and res["reuse_validity"]["threshold"] == 1.5


def test_an_r2_gate_keeps_the_drivers_live_threshold(tmp_path, monkeypatch):
    # an R² gate: the driver's r2_threshold (0.95 here), the skill's own 0.90 is not what the driver applies
    epr = QualityGate(metric="r_squared", accept_threshold=0.90, hard_reject_threshold=0.75)
    res, ex, _, _ = curve._replay(tmp_path / "a", monkeypatch, {"LOW": 0.92, "HIGH": 0.80}, quality_gate=epr)
    rv = res["reuse_validity"]
    assert rv["verdict"] == "poor" and rv["metric"] == "r_squared" and rv["threshold"] == curve.THRESHOLD and rv["score"] == 0.92
    # no gate in state (a legacy caller): R² at the driver's threshold, as always
    res, ex, _, _ = curve._replay(tmp_path / "b", monkeypatch, {"LOW": 0.99, "HIGH": 0.80})
    assert res["reuse_validity"]["verdict"] == "good" and res["reuse_validity"]["metric"] == "r_squared"


def _recorded(prior, gate, *, recipes_too=True):
    """Stamp a prior run's results with the gate it was held to: on the run
    and (``recipes_too``) on each locked recipe, as the agent and the lock do."""
    import json
    f = prior / "analysis_results.json"
    d = json.loads(f.read_text())
    d["quality_gate"] = gate_record(gate)
    if recipes_too:
        for r in d.get("locked_recipes", {}).values():
            r["gate"] = gate_record(gate)
    f.write_text(json.dumps(d))


def test_the_gate_record_round_trips():
    rec = gate_record(FOM)
    assert rec == {"metric": "figure_of_merit", "accept_threshold": 0.70, "hard_reject_threshold": 0.30,
                   "direction": "higher_is_better", "physical_review": False, "value_source": "result"}
    assert from_mapping(rec) == FOM and from_mapping(gate_record(CHI2)) == CHI2
    assert gate_record(None) is None and gate_record({"metric": "x"}) is None


def test_a_replay_is_held_to_the_gate_the_recipe_was_approved_under(tmp_path, monkeypatch):
    """Live: a series run given the phase-identification skill locked its
    recipes under figure_of_merit >= 0.70; the reuse run, given no skill,
    resolved the R² default and judged the replay on R² = -1.12. The recipe
    now carries its gate (stamped at lock time), the run records its own,
    and the replay is held to the recipe's — whatever the reuse run's."""
    prior = rc.prior_two_regime_run(tmp_path)
    _recorded(prior, FOM)
    # the reuse run's own gate is the R² default (no skill given): the recipe's FOM gate is applied
    res, ex, _, _ = curve._replay(tmp_path / "a", monkeypatch, {"LOW": -1.3, "HIGH": -1.12}, prior=prior,
                                  data=rc.spectrum(rc.RUTILE, shift=0.5, seed=11),
                                  extra_quality={"HIGH": {"figure_of_merit": 0.81}, "LOW": {"figure_of_merit": 0.75}},
                                  extra_params={"HIGH": rc.auto_detect_parameters(rc.RUTILE, shift=0.5, seed=50, n_noise=7)})
    rv = res["reuse_validity"]
    assert res["script"] == "HIGH" and rv["verdict"] == "good"
    assert rv["metric"] == "figure_of_merit" and rv["score"] == 0.81 and rv["threshold"] == 0.70 and rv["r_squared"] == -1.12
    # ...and the reuse run's own skill gate, had it one, does not override the recipe's
    res, ex, _, _ = curve._replay(tmp_path / "b", monkeypatch, {"LOW": -1.3, "HIGH": -1.12}, prior=prior, quality_gate=PROFILE,
                                  data=rc.spectrum(rc.RUTILE, shift=0.5, seed=11),
                                  extra_quality={"HIGH": {"figure_of_merit": 0.81, "peak_region_r2": 0.2}},
                                  extra_params={"HIGH": rc.auto_detect_parameters(rc.RUTILE, shift=0.5, seed=50, n_noise=7)})
    assert res["reuse_validity"]["verdict"] == "good" and res["reuse_validity"]["metric"] == "figure_of_merit"
    # a named unit script of that run: the RUN's recorded gate (the recipe files carry none)
    (prior / "scripts" / "spectrum_0003.py").write_text("HIGH")
    res, ex, _, _ = curve._replay(tmp_path / "c", monkeypatch, {"HIGH": -1.12}, prior=prior / "scripts" / "spectrum_0003.py",
                                  data=rc.spectrum(rc.RUTILE, shift=0.5, seed=11), extra_quality={"HIGH": {"figure_of_merit": 0.81}},
                                  extra_params={"HIGH": rc.auto_detect_parameters(rc.RUTILE, shift=0.5, seed=50, n_noise=7)})
    assert res["reuse_validity"]["verdict"] == "good" and res["reuse_validity"]["metric"] == "figure_of_merit"
    # a recipe approved under an R² gate at 0.90 replays at 0.90, not at the reuse run's 0.95
    epr = QualityGate(metric="r_squared", accept_threshold=0.90, hard_reject_threshold=0.75)
    prior2 = rc.prior_two_regime_run(tmp_path / "p2")
    _recorded(prior2, epr)
    res, ex, _, _ = curve._replay(tmp_path / "d", monkeypatch, {"LOW": 0.80, "HIGH": 0.92}, prior=prior2,
                                  data=rc.spectrum(rc.RUTILE, shift=0.5, seed=11),
                                  extra_params={"HIGH": rc.auto_detect_parameters(rc.RUTILE, shift=0.5, seed=50, n_noise=7)})
    assert res["reuse_validity"]["verdict"] == "good" and res["reuse_validity"]["threshold"] == 0.90 and res["reuse_validity"]["metric"] == "r_squared"
    # an older run that recorded no gate: the reuse run's own, as before
    prior3 = rc.prior_two_regime_run(tmp_path / "p3")
    res, ex, _, _ = curve._replay(tmp_path / "e", monkeypatch, {"LOW": 0.80, "HIGH": 0.92}, prior=prior3,
                                  data=rc.spectrum(rc.RUTILE, shift=0.5, seed=11),
                                  extra_params={"HIGH": rc.auto_detect_parameters(rc.RUTILE, shift=0.5, seed=50, n_noise=7)})
    assert res["reuse_validity"]["verdict"] == "poor" and res["reuse_validity"]["threshold"] == curve.THRESHOLD


def test_the_lock_and_the_run_record_their_gate(tmp_path, monkeypatch):
    names6 = [f"spectrum_{i:04d}" for i in range(6)]
    state, _ = curve.run_series(tmp_path / "lock", monkeypatch, names=names6, regimes=[[0, 1, 2], [3, 4, 5]],
                                anchors={"spectrum_0000": curve.OK, "spectrum_0003": {**curve.OK, "script": "M3"}},
                                follower_r2={n: 0.97 for n in names6})
    results = curve.compile_results(tmp_path / "lock", state)
    for r in results["locked_recipes"].values():
        assert r["gate"]["metric"] == "r_squared" and r["gate"]["accept_threshold"] == curve.THRESHOLD
    from scilink.agents.exp_agents._verification_record import series_recipes
    assert all(r["gate"]["metric"] == "r_squared" for r in series_recipes(results))


def test_an_explicit_gate_on_the_reuse_run_wins_with_a_warning(tmp_path, monkeypatch, caplog):
    """#717 review, item 1: the recipe's recorded gate is the default, but a
    caller who asked for a gate on THIS run (quality_gate= or r2_threshold=)
    gets it, as resolve_gate's priority says — a one-off 0.999 on a prior run
    must not bind every later reuse, and a user asking for 0.80 must not be
    judged at the recipe's 0.95."""
    import logging
    prior = rc.prior_two_regime_run(tmp_path)
    _recorded(prior, QualityGate(metric="r_squared", accept_threshold=0.95, hard_reject_threshold=0.90))
    data = rc.spectrum(rc.RUTILE, shift=0.5, seed=11)
    params = {"HIGH": rc.auto_detect_parameters(rc.RUTILE, shift=0.5, seed=50, n_noise=7)}
    # no ask: the recipe's 0.95 → 0.915 is poor
    res, ex, _, _ = curve._replay(tmp_path / "a", monkeypatch, {"LOW": 0.80, "HIGH": 0.915}, prior=prior, data=data, extra_params=params)
    assert res["reuse_validity"]["verdict"] == "poor" and res["reuse_validity"]["threshold"] == 0.95
    # the caller asked for 0.80 on this run: good, and the record says what the recipe was approved under
    loose = QualityGate(metric="r_squared", accept_threshold=0.80, hard_reject_threshold=0.70)
    with caplog.at_level(logging.WARNING):
        res, ex, _, _ = curve._replay(tmp_path / "b", monkeypatch, {"LOW": 0.80, "HIGH": 0.915}, prior=prior, data=data, extra_params=params,
                                      quality_gate=loose, state_extra={"quality_gate_explicit": "gate"},
                                      controller=lambda out, ex: _loose_controller(out, ex, 0.80))
    assert res["reuse_validity"]["verdict"] == "good" and res["reuse_validity"]["threshold"] == 0.80
    assert any("approved under r_squared 0.95" in r.message and "asked for r_squared 0.8" in r.message for r in caplog.records)
    # asked for 0.99: judged at 0.99, not the recipe's 0.95
    strict = QualityGate(metric="r_squared", accept_threshold=0.99, hard_reject_threshold=0.95)
    res, ex, _, _ = curve._replay(tmp_path / "c", monkeypatch, {"LOW": 0.80, "HIGH": 0.97}, prior=prior, data=data, extra_params=params,
                                  quality_gate=strict, state_extra={"quality_gate_explicit": "gate"},
                                  controller=lambda out, ex: _loose_controller(out, ex, 0.99))
    assert res["reuse_validity"]["verdict"] == "poor" and res["reuse_validity"]["threshold"] == 0.99
    # an explicit figure-of-merit gate on the reuse run is applied to a recipe approved under R²
    res, ex, _, _ = curve._replay(tmp_path / "d", monkeypatch, {"LOW": 0.80, "HIGH": 0.915}, prior=prior, data=data,
                                  extra_params=params, extra_quality={"HIGH": {"figure_of_merit": 0.9}},
                                  quality_gate=FOM, state_extra={"quality_gate_explicit": "gate"})
    assert res["reuse_validity"]["verdict"] == "good" and res["reuse_validity"]["metric"] == "figure_of_merit"
    # an r2_threshold= ask (honoured by resolve_gate: "threshold") moves an R² recipe's threshold...
    res, ex, _, _ = curve._replay(tmp_path / "e", monkeypatch, {"LOW": 0.80, "HIGH": 0.915}, prior=prior, data=data, extra_params=params,
                                  state_extra={"quality_gate_explicit": "threshold"},
                                  controller=lambda out, ex: _loose_controller(out, ex, 0.80))
    assert res["reuse_validity"]["verdict"] == "good" and res["reuse_validity"]["threshold"] == 0.80 and res["reuse_validity"]["metric"] == "r_squared"
    # ...but cannot replace a recipe's figure-of-merit gate: the recipe's stands, and the log says the override was ignored
    fom_prior = rc.prior_two_regime_run(tmp_path / "fp")
    _recorded(fom_prior, FOM)
    with caplog.at_level(logging.WARNING):
        caplog.clear()
        res, ex, _, _ = curve._replay(tmp_path / "f", monkeypatch, {"LOW": 0.05, "HIGH": 0.057}, prior=fom_prior, data=data,
                                      extra_params=params, extra_quality={"HIGH": {"figure_of_merit": 0.88}},
                                      state_extra={"quality_gate_explicit": "threshold"},
                                      controller=lambda out, ex: _loose_controller(out, ex, 0.90))
    assert res["reuse_validity"]["verdict"] == "good" and res["reuse_validity"]["metric"] == "figure_of_merit"
    assert any("R² override 0.9 ignored" in r.message for r in caplog.records)
    # the agent sets the flag from resolve_gate's outcome: a bare number under a skill's non-R² gate is no ask
    from scilink.agents.exp_agents.quality_gate import resolve_gate
    eff = resolve_gate(user_threshold=0.9, skill_meta={"quality_gate": gate_record(PROFILE)})
    assert eff.metric == "peak_region_r2"                                   # the guard dropped the number
    flag = "gate" if False else ("threshold" if 0.9 is not None and eff.metric == "r_squared" else None)
    assert flag is None


def _loose_controller(out, ex, r2):
    from scilink.agents.exp_agents.controllers.curve_fitting_controllers import UnifiedSeriesProcessingController
    import json, logging
    from unittest.mock import MagicMock
    return UnifiedSeriesProcessingController(
        model=MagicMock(), logger=logging.getLogger("series_path"), generation_config=None, safety_settings=None,
        parse_fn=lambda r: (json.loads(r.text), None), executor=ex, script_instructions="", correction_instructions="",
        quality_instructions="", output_dir=str(out), plot_fn=lambda d, i: b"plot", r2_threshold=r2, parallel_workers=1)


def test_the_boards_copy_of_a_recipe_carries_its_gate_and_model(tmp_path, monkeypatch):
    """#717 review, item 2: the board copied the script alone, so a
    figure-of-merit recipe replayed from the swarm path was judged on the R²
    default (poor at R² 0.46). The copy's sidecar carries the gate and the
    model; a replay of the copy reads them."""
    from scilink.agents.meta_agent import board as board_mod
    from scilink.agents.meta_agent.board import Board
    from scilink.agents.exp_agents._verification_record import prior_recipe_candidates, recipe_sidecar
    board = Board(tmp_path / "meta")
    entry = {"index": 2, "label": "xrd", "mode": "analysis", "status": "success"}
    row = {"analysis_id": "s1", "status": "success", "verified": True, "reason": "ok", "series": True, "agent_name": "CurveFittingAgent",
           "recipes": [{"regime": "rutile", "unit": "spectrum_0003", "index": 3, "verified": True, "reason": "ok", "script": "HIGH",
                        "gate": gate_record(FOM), "model": "rutile TiO2, P42/mnm"}]}
    board_mod.post_delegation(board, entry, {"analyses": [row]})
    rec = next(r for r in board.records() if r["kind"] == "recipe")
    copy = Path(rec["payload"]["path"])
    assert rec["payload"]["quality_gate"] == gate_record(FOM) and rec["payload"]["model"] == "rutile TiO2, P42/mnm"
    side = recipe_sidecar(copy)
    same = lambda rec, g: from_mapping(rec) == from_mapping(gate_record(g))      # noqa: E731 - the reader fills a known best value
    assert same(side["quality_gate"], FOM) and side["model"] == "rutile TiO2, P42/mnm" and side["regime"] == "rutile"
    cands = prior_recipe_candidates(copy.parent, single_name="fitting_script.py", named=copy)
    assert same(cands[0]["gate"], FOM) and cands[0]["model"] == "rutile TiO2, P42/mnm" and cands[0]["regime"] == "rutile"
    # the sidecar sits beside the script it describes, also when the copy's name was taken (u_1-2.py)
    board_mod.post_delegation(board, {**entry, "index": 2, "label": "xrd"}, {"analyses": [{**row, "analysis_id": "s1",
                              "recipes": [{**row["recipes"][0], "script": "OTHER"}]}]})
    copies = sorted(copy.parent.glob("*.py"))
    assert len(copies) == 2 and all(c.with_name(f"{c.stem}.recipe.json").is_file() for c in copies)
    # a malformed sidecar is no sidecar: the replay falls back to the run's gate and the step does not crash
    bad = tmp_path / "bad" / "recipe.py"
    bad.parent.mkdir()
    bad.write_text("HIGH")
    bad.with_name("recipe.recipe.json").write_text(json.dumps({"quality_gate": {"metric": "figure_of_merit", "accept_threshold": "abc"},
                                                               "regime": ["not", "a", "string"], "unit": 7}))
    side = recipe_sidecar(bad)
    assert "quality_gate" not in side and side["regime"] == "['not', 'a', 'string']" and side["unit"] == "7"
    res, ex, _, _ = curve._replay(tmp_path / "g", monkeypatch, {"HIGH": 0.99}, prior=bad, data=rc.spectrum(rc.RUTILE, seed=4),
                                  extra_params={"HIGH": rc.auto_detect_parameters(rc.RUTILE, seed=61)},
                                  state_extra={"quality_gate_explicit": "threshold"})
    assert res["reuse_validity"]["verdict"] == "good" and res["reuse_validity"]["metric"] == "r_squared"
    # a replay of the copy, with no skill on the reuse run: judged on the recipe's figure of merit, R² -0.5 beside it
    res, ex, _, _ = curve._replay(tmp_path / "a", monkeypatch, {"HIGH": -0.5}, prior=copy, data=rc.spectrum(rc.RUTILE, seed=3),
                                  extra_quality={"HIGH": {"figure_of_merit": 1.0}},
                                  extra_params={"HIGH": rc.auto_detect_parameters(rc.RUTILE, seed=60)})
    rv = res["reuse_validity"]
    assert rv["verdict"] == "good" and rv["metric"] == "figure_of_merit" and rv["score"] == 1.0 and rv["r_squared"] == -0.5
    # a single run's copy carries the run's recorded gate too
    run_dir = tmp_path / "single_run"
    (run_dir / "scripts").mkdir(parents=True)
    (run_dir / "scripts" / "fitting_script.py").write_text("ONE")
    (run_dir / "analysis_results.json").write_text(json.dumps({"status": "success", "quality_gate": gate_record(PROFILE)}))
    row2 = {"analysis_id": "r2", "status": "success", "verified": True, "reason": "ok", "agent_name": "CurveFittingAgent",
            "output_directory": str(run_dir)}
    board_mod.post_delegation(board, {**entry, "index": 3}, {"analyses": [row2]})
    rec2 = next(r for r in board.records() if r["kind"] == "recipe" and r["payload"]["analysis_id"] == "r2")
    assert rec2["payload"]["quality_gate"] == gate_record(PROFILE)
    assert same(recipe_sidecar(Path(rec2["payload"]["path"]))["quality_gate"], PROFILE)


def test_a_series_derived_live_reference_carries_the_frames_gate(tmp_path):
    """#717 review, item 4: the live loop laid a series' last frame out as a
    single anchor without the gate, so a strict frame against it was judged
    on the R² default (0/6 good) while a profile-gated single run gave 6/6.
    The anchor now records the frame's regime gate, else the series run's."""
    import json
    from scilink.live.measurement_loop import MeasurementLoop
    series = tmp_path / "series"
    series.mkdir()
    rows = [{"index": i, "name": f"f{i}", "success": True, "script": f"S{i}", "regime": "hot" if i else "cold",
             "parameters": {}, "fit_quality": {"r_squared": 0.3, "figure_of_merit": 0.9}, "model_type": "phase"} for i in range(2)]
    (series / "series_fit_results.json").write_text(json.dumps({"results": rows, "locked_config": {}}))
    (series / "analysis_results.json").write_text(json.dumps({"quality_gate": gate_record(PROFILE), "locked_recipes": {
        "cold": {"unit": "f0", "index": 0, "regime": "cold", "script": "S0", "verdict": {"verified": True}, "gate": gate_record(PROFILE)},
        "hot": {"unit": "f1", "index": 1, "regime": "hot", "script": "S1", "verdict": {"verified": True}, "gate": gate_record(FOM)}}}))
    dest = tmp_path / "anchor"
    dest.mkdir()
    MeasurementLoop._single_frame_anchor(series, dest, ["a.txt", "b.txt"])
    rec = json.loads((dest / "analysis_results.json").read_text())
    assert rec["quality_gate"] == gate_record(FOM)                         # the LAST frame's regime, the one locked
    # a series run with no recipe gates: the run's
    (series / "analysis_results.json").write_text(json.dumps({"quality_gate": gate_record(PROFILE), "locked_recipes": {}}))
    MeasurementLoop._single_frame_anchor(series, dest, ["a.txt", "b.txt"])
    assert json.loads((dest / "analysis_results.json").read_text())["quality_gate"] == gate_record(PROFILE)


def test_series_units_of_a_reuse_are_held_to_the_recipes_gate(tmp_path, monkeypatch):
    """#717 round 2, item 1: a profile-gated recipe reused over a series with
    no skill had its anchor good on peak_region_r2 and every unit flagged
    below_threshold on R² 0.95, with LLM refits. The units are held to the
    reuse decision's gate (_series_gate), as the replay is."""
    from unittest.mock import MagicMock
    ctrl = _loose_controller(tmp_path, MagicMock(), 0.95)
    rows = [{"index": i, "name": f"s{i}", "success": True, "fit_quality": {"r_squared": 0.80 + 0.01 * i, "peak_region_r2": 0.96 + 0.005 * i}}
            for i in range(5)]
    # the run's own gate is the R² default (no skill): every unit flagged on main's rule
    flagged = ctrl._detect_outliers(rows, gate=ctrl._series_gate({}))
    assert [f["reason"] for f in flagged] == ["below_threshold"] * 5
    # the run replays a profile-gated recipe: the units are held to peak_region_r2 ≥ 0.90 — none flagged
    state = {"_reuse_gate": gate_record(PROFILE)}
    assert ctrl._series_gate(state).metric == "peak_region_r2"
    assert ctrl._detect_outliers(rows, gate=ctrl._series_gate(state)) == []
    # a recipe approved at R² 0.90 reused on a run whose driver sits at 0.95: units at 0.92 are not flagged
    state = {"_reuse_gate": gate_record(QualityGate(metric="r_squared", accept_threshold=0.90, hard_reject_threshold=0.75))}
    rows2 = [{"index": i, "name": f"s{i}", "success": True, "fit_quality": {"r_squared": 0.92}} for i in range(5)]
    assert ctrl._detect_outliers(rows2, gate=ctrl._series_gate(state)) == []
    assert [f["reason"] for f in ctrl._detect_outliers(rows2, gate=ctrl._series_gate({}))] == ["below_threshold"] * 5
    # the caller's ask on the reuse run binds the units too
    state = {"_reuse_gate": gate_record(PROFILE), "quality_gate_explicit": "gate", "quality_gate": FOM}
    assert ctrl._series_gate(state).metric == "figure_of_merit"
