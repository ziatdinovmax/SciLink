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
