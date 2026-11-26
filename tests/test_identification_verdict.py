"""A workflow tool that rejects its own result is not passed by the gate's band.

The XRD scorers return an authoritative ``verdict`` beside their figure of
merit, and the multi-phase figure (1 - cost) can sit just above the gate's
hard-reject threshold on a match the scorer itself rejects: live, an
identification of "O2 Si" with verdict "reject", figure 0.401 and R² -2.08 was
approved and handed on as verified while the narrative discarded it. A series follower whose scorer found no candidate reported no
figure at all and was never flagged.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
    UnifiedSeriesProcessingController,
)
from scilink.agents.exp_agents.quality_gate import QualityGate, from_mapping, gate_record, resolve_gate
from scilink.skills.loader import load_skill


def _gate(verdict_key="verdict"):
    return QualityGate(metric="figure_of_merit", accept_threshold=0.70, hard_reject_threshold=0.40,
                       direction="higher_is_better", physical_review=False, verdict_key=verdict_key)


def _ctrl():
    c = UnifiedSeriesProcessingController.__new__(UnifiedSeriesProcessingController)
    c.logger = MagicMock()
    c.r2_threshold = 0.95
    c.outlier_sigma = 3.0
    return c


def _fit(fom, verdict=None):
    fq = {"figure_of_merit": fom} if fom is not None else {}
    if verdict is not None:
        fq["verdict"] = verdict
    return {"fit_quality": fq, "visualization_bytes": b"<png>", "model_type": "t", "parameters": {}}


def test_declared_reject_reads_only_the_declared_key():
    g = _gate()
    assert g.declared_reject({"figure_of_merit": 0.401, "verdict": "reject"})
    assert g.declared_reject({"verdict": " Reject "})
    assert g.declared_reject({"verdict": "marginal"}) is None
    assert g.declared_reject({"verdict": "accept"}) is None
    assert g.declared_reject({}) is None
    assert _gate(verdict_key=None).declared_reject({"verdict": "reject"}) is None


def test_the_key_travels_with_the_gate():
    g = _gate()
    assert from_mapping(gate_record(g)) == g
    assert "verdict_key" not in gate_record(_gate(verdict_key=None))


def test_the_xrd_skill_declares_its_scorers_verdict():
    gate = resolve_gate(skill_meta=load_skill("xrd", domain="structure_matching")["meta"])
    assert gate.metric == "figure_of_merit" and gate.verdict_key == "verdict"


def test_a_rejected_match_in_the_band_is_a_hard_reject():
    ctrl = _ctrl()
    v = ctrl._verify_fit_with_llm({"quality_gate": _gate()}, _fit(0.401, "reject"))
    assert v["fit_acceptable"] is False
    assert v["recommended_action"] != "none"
    assert "reject" in v["overall_assessment"]


def test_the_band_is_unchanged_without_a_rejection():
    ctrl = _ctrl()
    for verdict in ("marginal", None):
        v = ctrl._verify_fit_with_llm({"quality_gate": _gate()}, _fit(0.55, verdict))
        assert v["fit_acceptable"] is True and "marginal" in v["overall_assessment"].lower()
    v = ctrl._verify_fit_with_llm({"quality_gate": _gate(verdict_key=None)}, _fit(0.401, "reject"))
    assert v["fit_acceptable"] is True


def _series(rows):
    # unit 0 answers to its own gate; the rest are followers, verified by the recipe they replayed
    return [{"index": i, "name": f"f{i}", "success": True, "fit_quality": fq,
             "unit_verdict": {"verified": True, "decided_by": "qc_gate" if i == 0 else "recipe"}}
            for i, fq in enumerate(rows)]


def test_a_follower_with_no_figure_is_flagged_below_its_gate():
    rows = [{"figure_of_merit": 0.80, "verdict": "accept"}] * 5 + [{"verdict": "no_candidates"}]
    flagged = {f["index"]: f for f in _ctrl()._detect_outliers(_series(rows), gate=_gate())}
    assert set(flagged) == {5}
    assert flagged[5]["reason"] == "below_threshold" and "not reported" in flagged[5]["recommendation"]
    # an anchor with no figure answers to its own gate, not to this pass
    rows = [{"verdict": "no_candidates"}] + [{"figure_of_merit": 0.80, "verdict": "accept"}] * 5
    assert _ctrl()._detect_outliers(_series(rows), gate=_gate()) == []


def test_a_follower_its_scorer_rejects_is_flagged_even_above_the_accept_threshold():
    # inside the band (0.40-0.70) a follower was already below threshold; the
    # declared reject matters where the number alone would pass
    rows = [{"figure_of_merit": 0.80, "verdict": "accept"}] * 5 + [{"figure_of_merit": 0.75, "verdict": "reject"}]
    flagged = {f["index"]: f for f in _ctrl()._detect_outliers(_series(rows), gate=_gate())}
    assert set(flagged) == {5} and flagged[5]["reason"] == "below_threshold"
    assert "scoring tool's own verdict is reject" in flagged[5]["recommendation"]


def test_followers_with_no_figure_are_flagged_however_few_units_were_scored():
    # the series statistics need three scores; the gate's own rejection does not
    rows = [{"figure_of_merit": 0.80, "verdict": "accept"}] * 2 + [{"verdict": "no_candidates"}] * 6
    flagged = {f["index"]: f["reason"] for f in _ctrl()._detect_outliers(_series(rows), gate=_gate())}
    assert flagged == {i: "below_threshold" for i in range(2, 8)}


def test_the_r2_path_is_unchanged_for_a_missing_score():
    rows = [{"r_squared": 0.99}] * 5 + [{}]
    assert _ctrl()._detect_outliers(_series(rows)) == []
    # a goodness-of-fit gate's metric left unreported keeps main's path too
    profile = QualityGate(metric="peak_region_r2", accept_threshold=0.85, hard_reject_threshold=0.5)
    rows = [{"peak_region_r2": 0.97}] * 5 + [{"r_squared": 0.8}]
    assert _ctrl()._detect_outliers(_series(rows), gate=profile) == []


def test_a_pinned_follower_with_no_figure_is_the_pin_passs_in_both_paths():
    pinned = {"verdict": "no_candidates", "pinned_at_bound": [
        {"component": "phase_1", "parameter": "lattice_scale", "value": 1.02, "side": "upper", "bound": 1.02}]}
    few = [{"figure_of_merit": 0.80, "verdict": "accept"}] * 2 + [pinned]
    many = [{"figure_of_merit": 0.80, "verdict": "accept"}] * 5 + [pinned]
    assert _ctrl()._detect_outliers(_series(few), gate=_gate()) == []
    assert {f["index"]: f["reason"] for f in _ctrl()._detect_outliers(_series(many), gate=_gate())} == {5: "pinned_at_bound"}


def test_the_scorers_reason_survives_when_the_unit_is_also_an_outlier():
    rows = [{"figure_of_merit": 0.98, "verdict": "accept"}] * 5 + [{"figure_of_merit": 0.75, "verdict": "reject"}]
    flagged = {f["index"]: f for f in _ctrl()._detect_outliers(_series(rows), gate=_gate())}
    assert flagged[5]["reason"] == "outlier_and_below_threshold"
    assert "scoring tool's own verdict is reject" in flagged[5]["recommendation"]


def test_the_report_badge_says_poor_for_a_declared_reject():
    from scilink.agents.exp_agents.controllers.curve_fitting_controllers import GenerateCurveFittingReportController
    ctrl = GenerateCurveFittingReportController.__new__(GenerateCurveFittingReportController)
    ctrl.r2_threshold = 0.95
    html = ctrl._format_fit_quality({"figure_of_merit": 0.401, "verdict": "reject", "r_squared": -2.08}, gate=_gate())
    assert "Poor" in html and "Marginal" not in html
    # no figure to show: still Poor, never the R² badge's "Excellent"
    html = ctrl._format_fit_quality({"verdict": "reject", "r_squared": 0.995}, gate=_gate())
    assert "Poor" in html and "Excellent" not in html
    # the band is unchanged without a rejection
    assert "Marginal" in ctrl._format_fit_quality({"figure_of_merit": 0.55, "verdict": "marginal"}, gate=_gate())


def test_scoring_gated_flags_trigger_no_refit():
    from scilink.agents.exp_agents.controllers.curve_fitting_controllers import AdaptiveRefitController
    ctrl = AdaptiveRefitController.__new__(AdaptiveRefitController)
    ctrl.logger = MagicMock()
    rows = [{"figure_of_merit": 0.80, "verdict": "accept"}] * 2 + [{"verdict": "no_candidates"}] * 3
    series = _series(rows)
    flagged = _ctrl()._detect_outliers(series, gate=_gate())
    state = {"is_single_spectrum": False, "quality_gate": _gate(), "series_results": series,
             "flagged_spectra": flagged}
    out = ctrl.execute(state)   # any refit would need the model, the executor and the data: none are set
    assert out is state and not out.get("refit_summary") and len(out["flagged_spectra"]) == 3


# ── a result the gate's own code rejects is never verified, on a follower as on its anchor ──
#
# A follower is verified by the recipe it replayed, and no verifier looks at
# it. The scorer's "marginal" was withheld on a follower (the provisional
# stamp) while its "reject" was only flagged: the unit kept verified=True.

def _workflow_gate():
    return QualityGate(metric="figure_of_merit", accept_threshold=0.70, hard_reject_threshold=0.40,
                       direction="higher_is_better", physical_review=False, verdict_key="verdict",
                       certifying_verdicts=("accept",))


def test_the_gates_own_code_rejects():
    g = _workflow_gate()
    assert "verdict is reject" in g.rejects({"figure_of_merit": 0.75, "verdict": "reject"})
    assert "not reported" in g.rejects({"verdict": "no_candidates"})
    assert "not reported" in g.rejects({})
    assert "past its gate's hard reject" in g.rejects({"figure_of_merit": 0.20, "verdict": "accept"})
    assert g.rejects({"figure_of_merit": 0.55, "verdict": "marginal"}) is None       # the band: provisional's
    assert g.rejects({"figure_of_merit": 0.85, "verdict": "accept"}) is None
    # a goodness-of-fit gate's band is its verifier's to judge: only a declared reject is code
    fit = QualityGate(metric="peak_region_r2", accept_threshold=0.85, hard_reject_threshold=0.5)
    assert fit.rejects({"peak_region_r2": 0.10}) is None and fit.rejects({}) is None
    assert QualityGate().rejects({"r_squared": 0.10}) is None
    declared = QualityGate(metric="peak_region_r2", accept_threshold=0.85, hard_reject_threshold=0.5,
                           verdict_key="verdict")
    assert "verdict is reject" in declared.rejects({"peak_region_r2": 0.99, "verdict": "reject"})


def test_a_null_result_is_rejected_not_provisional():
    # the skill's own sentinel for "the scorer found no candidate": no figure, a verdict that is
    # neither certifying nor "reject". It was stamped "provisional ... acceptable as a result".
    g = _workflow_gate()
    assert g.uncertified({"verdict": "no_candidates"}) is None
    assert g.uncertified({"figure_of_merit": 0.20, "verdict": "marginal"}) is None
    assert "marginal" in g.uncertified({"figure_of_merit": 0.55, "verdict": "marginal"})   # unchanged in the band


def test_a_rejected_unit_is_withheld_by_every_verdict_path():
    from scilink.agents.exp_agents._verification_record import fit_check_withheld, unit_verdict_for
    why = _workflow_gate().rejects({"figure_of_merit": 0.20, "verdict": "reject"})
    unit = {"success": True, "fit_quality": {"figure_of_merit": 0.20, "verdict": "reject", "gate_rejected": [why]}}
    assert fit_check_withheld(unit) == why
    follower = unit_verdict_for(unit, recipe={"unit": "anchor", "verdict": {"verified": True}})
    assert follower["verified"] is False and follower["decided_by"] == "recipe" and "verdict is reject" in follower["reason"]
    own = unit_verdict_for({**unit, "quality_history": {"approved": True, "final_r2": 0.2, "threshold": 0.7}})
    assert own["verified"] is False and "verdict is reject" in own["reason"]
    reuse = unit_verdict_for({**unit, "reuse_validity": {"reused": True, "verdict": "good"}})
    assert reuse["verified"] is False and "verdict is reject" in reuse["reason"]


class _ScoredExecutor:
    """As the series harness's executor, with the skill's own scores per unit."""
    timeout = 30

    def __init__(self, quality_by_name):
        self.quality_by_name, self.calls = quality_by_name, []

    def execute_script(self, script, working_dir=None, timeout=None, **kw):
        import json
        from pathlib import Path
        wd = Path(working_dir)
        name = next((n for n in self.quality_by_name if n in wd.as_posix()), None)
        self.calls.append((name, script))
        (wd / "visualization.png").write_bytes(b"png")
        out = {"model_type": f"model of {script}", "parameters": {"identified_phase": "Si", "lattice_scale": 1.001},
               "fit_quality": {"r_squared": 0.9, **self.quality_by_name.get(name, {})}}
        return {"status": "success", "stdout": "FIT_RESULTS_JSON:" + json.dumps(out), "stderr": "", "message": ""}


def _xrd_series(tmp_path, monkeypatch, follower_quality):
    from test_series_verdict_path import OK, run_series
    names = [f"spectrum_{i:04d}" for i in range(len(follower_quality) + 1)]
    quality = dict(zip(names[1:], follower_quality))
    state, _ = run_series(tmp_path, monkeypatch, names=names, anchors={names[0]: OK}, follower_r2={},
                          executor=_ScoredExecutor(quality), state_extra={"quality_gate": _workflow_gate()})
    units = {u["name"]: u for u in state["series_results"]}
    flags = {f["name"]: f for f in (state.get("flagged_spectra") or [])}
    return names, units, flags


def test_a_rejected_follower_is_not_verified_through_the_real_series_path(tmp_path, monkeypatch):
    names, units, flags = _xrd_series(tmp_path, monkeypatch, [
        {"figure_of_merit": 0.85, "verdict": "accept"},
        {"figure_of_merit": 0.20, "verdict": "reject"},          # the phase is gone on this frame
        {"verdict": "no_candidates"},                            # the scorer found nothing
        {"figure_of_merit": 0.75, "verdict": "reject"},          # rejected above the accept threshold
        {"figure_of_merit": 0.55, "verdict": "marginal"},
        {"figure_of_merit": 0.80, "verdict": "accept"},
    ])
    verdicts = {n: units[n]["unit_verdict"] for n in names[1:]}
    assert [verdicts[n]["verified"] for n in names[1:]] == [True, False, False, False, False, True]
    assert all(verdicts[n]["decided_by"] == "recipe" for n in names[1:])     # the recipe is kept: no refit
    assert all(units[n]["fitted_from"] == "locked_script" for n in names[1:])
    assert "verdict is reject" in verdicts[names[2]]["reason"]
    assert "not reported" in verdicts[names[3]]["reason"]
    assert "verdict is reject" in verdicts[names[4]]["reason"]
    assert "marginal" in verdicts[names[5]]["reason"]                        # provisional, as before
    assert "gate_rejected" not in units[names[5]]["fit_quality"] and "provisional" not in units[names[2]]["fit_quality"]
    # one flag per unit, and a rejected unit's says why
    assert set(flags) == set(names[2:6])
    assert flags[names[2]]["reason"] in ("below_threshold", "outlier_and_below_threshold")
    assert flags[names[3]]["reason"] == "below_threshold" and flags[names[4]]["reason"] == "below_threshold"
    assert flags[names[5]]["reason"] in ("below_threshold", "provisional")       # in the band: the band's flag first
    assert all("Rejected" in flags[n]["recommendation"] for n in (names[2], names[4]))
    assert "not reported" in flags[names[3]]["recommendation"]


def test_a_two_unit_series_flags_its_rejected_follower(tmp_path, monkeypatch):
    names, units, flags = _xrd_series(tmp_path, monkeypatch, [{"figure_of_merit": 0.20, "verdict": "reject"}])
    assert units[names[1]]["unit_verdict"]["verified"] is False
    assert flags[names[1]]["reason"] == "below_threshold" and "verdict is reject" in flags[names[1]]["recommendation"]


def test_a_reuse_the_recipes_gate_rejects_is_not_verified_by_its_number(tmp_path, monkeypatch):
    # no XRD gate on the run; the recipe was recorded under one, and the scorer rejects at a figure
    # the replay gate's number passes
    from scilink.agents.exp_agents._verification_record import unit_verdict_for
    from test_series_verdict_path import _replay
    res, _, _, _ = _replay(tmp_path, monkeypatch, {"LOW": 0.99},
                           state_extra={"_reuse_gate": gate_record(_workflow_gate())},
                           extra_quality={"LOW": {"figure_of_merit": 0.75, "verdict": "reject"}})
    assert "verdict is reject" in " ".join((res.get("fit_quality") or {}).get("gate_rejected") or [])
    assert "provisional" not in res["fit_quality"]
    assert unit_verdict_for(res)["verified"] is False


def test_a_scripts_own_gate_rejected_key_is_not_the_gates(tmp_path, monkeypatch):
    from test_series_verdict_path import _replay
    res, _, _, _ = _replay(tmp_path, monkeypatch, {"LOW": 0.99},
                           extra_quality={"LOW": {"gate_rejected": ["said the script"]}})
    assert "gate_rejected" not in (res.get("fit_quality") or {})


def test_the_report_badge_says_poor_for_what_the_gate_rejects():
    from scilink.agents.exp_agents.controllers.curve_fitting_controllers import GenerateCurveFittingReportController
    ctrl = GenerateCurveFittingReportController.__new__(GenerateCurveFittingReportController)
    ctrl.r2_threshold = 0.95
    g = _workflow_gate()
    html = ctrl._format_fit_quality({"verdict": "no_candidates", "r_squared": 0.99}, gate=g)
    assert "Poor" in html and "Provisional" not in html and "Excellent" not in html
    html = ctrl._format_fit_quality({"figure_of_merit": 0.0, "verdict": "marginal"}, gate=g)
    assert "Poor" in html and "Provisional" not in html
    # stamped by the recipe's gate on a replay this run scores by R²
    html = ctrl._format_fit_quality({"r_squared": 0.99, "gate_rejected": ["the scoring tool's own verdict is reject"]})
    assert "Poor" in html and "Excellent" not in html
    html = ctrl._format_fit_quality({"r_squared": 0.99, "provisional": ["marginal"]})
    assert "Provisional" in html and "Excellent" not in html


def test_a_reuse_that_failed_into_fresh_code_is_not_held_to_the_recipes_gate(tmp_path, monkeypatch):
    # the recipe was recorded under the workflow gate; its script cannot run here and the unit is
    # re-derived under the run's own R² gate. Fresh code reports no figure of merit: the recipe's gate
    # read that absence as a reject on every unit (stamped, flagged, sent to the refit).
    import json
    from test_series_verdict_path import OK, run_series
    prior = tmp_path / "prior"
    (prior / "scripts").mkdir(parents=True)
    (prior / "series_fit_results.json").write_text("{}")
    (prior / "analysis_results.json").write_text(json.dumps({"locked_recipes": {
        "default": {"unit": "spectrum_0000", "index": 0, "regime": "default", "script": "PRIOR",
                    "verdict": {"verified": True}, "gate": gate_record(_workflow_gate())}}}))
    names = [f"spectrum_{i:04d}" for i in range(5)]
    state, _ = run_series(tmp_path / "new", monkeypatch, names=names, anchors={names[0]: OK},
                          follower_r2={n: 0.99 for n in names[1:]}, max_series_refits=2,
                          state_extra={"prior_analysis_paths": [str(prior)], "reuse_locked_script": True})
    assert state.get("_reuse_gate") and not state.get("_replayed_regimes")
    assert not any("gate_rejected" in (u.get("fit_quality") or {}) for u in state["series_results"])
    assert not state.get("flagged_spectra") and not state.get("refit_summary")


def test_a_goodness_of_fit_series_is_unchanged(tmp_path, monkeypatch):
    # a skill metric's band is its verifier's to judge: a follower far below it is not stamped by
    # code (it keeps its flag and its recipe's verdict, as before)
    from test_series_verdict_path import OK, run_series
    profile = QualityGate(metric="peak_region_r2", accept_threshold=0.90, hard_reject_threshold=0.55)
    names = [f"spectrum_{i:04d}" for i in range(5)]
    quality = {n: {"peak_region_r2": 0.97} for n in names[1:]}
    quality[names[3]] = {"peak_region_r2": 0.20}
    state, _ = run_series(tmp_path, monkeypatch, names=names, anchors={names[0]: OK}, follower_r2={},
                          executor=_ScoredExecutor(quality), state_extra={"quality_gate": profile})
    units = {u["name"]: u for u in state["series_results"]}
    assert not any("gate_rejected" in (u.get("fit_quality") or {}) for u in units.values())
    assert units[names[3]]["unit_verdict"]["verified"] is True
    assert {f["name"] for f in state.get("flagged_spectra") or []} == {names[3]}


def test_a_stamped_unit_yields_to_the_pin_pass_in_both_paths():
    why = "the scoring tool's own verdict is reject (verdict = 'reject')"
    pinned = {"figure_of_merit": 0.2, "verdict": "reject", "gate_rejected": [why], "pinned_at_bound": [
        {"component": "phase_1", "parameter": "lattice_scale", "value": 1.02, "side": "upper", "bound": 1.02}]}
    few = [{"figure_of_merit": 0.80, "verdict": "accept"}] + [pinned]        # two scores: no series statistics
    many = [{"figure_of_merit": 0.80, "verdict": "accept"}] * 5 + [pinned]
    assert _ctrl()._detect_outliers(_series(few), gate=_gate()) == []
    assert {f["index"]: f["reason"] for f in _ctrl()._detect_outliers(_series(many), gate=_gate())} == {5: "pinned_at_bound"}
    # the same row without a pin is the gate's rejection
    bare = {k: v for k, v in pinned.items() if k != "pinned_at_bound"}
    assert [f["reason"] for f in _ctrl()._detect_outliers(_series(few[:1] + [bare]), gate=_gate())] == ["below_threshold"]
