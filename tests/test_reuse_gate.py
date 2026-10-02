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
    assert rv["verdict"] == "poor" and rv["metric"] == "R²" and rv["threshold"] == curve.THRESHOLD and rv["score"] == 0.92
    # no gate in state (a legacy caller): R² at the driver's threshold, as always
    res, ex, _, _ = curve._replay(tmp_path / "b", monkeypatch, {"LOW": 0.99, "HIGH": 0.80})
    assert res["reuse_validity"]["verdict"] == "good" and res["reuse_validity"]["metric"] == "R²"


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
    assert res["reuse_validity"]["verdict"] == "good" and res["reuse_validity"]["threshold"] == 0.90 and res["reuse_validity"]["metric"] == "R²"
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
