"""An XRD match its scorer called "marginal" is reported as provisional, never
verified (the maintainer's decision): checked against their data, 1 of
21 marginal identifications in saved runs was right (a clathrasil, sodium
tartrate and sodium propionate reported for ibuprofen sodium; SiC4(SN)4 for
kyanite; BaGe for Si), and no figure-of-merit threshold inside the band
separates them. It is not retried (acceptable), only not certified."""
from __future__ import annotations

from scilink.agents.exp_agents._verification_record import fit_check_withheld, unit_verdict_for
from scilink.agents.exp_agents.quality_gate import QualityGate, from_mapping, gate_record, resolve_gate
from scilink.skills.loader import load_skill

from test_series_verdict_path import _replay  # the real qc_try_reuse -> _fit_single_spectrum path


def _gate(certify=("accept",)):
    return QualityGate(metric="figure_of_merit", accept_threshold=0.70, hard_reject_threshold=0.40,
                       physical_review=False, verdict_key="verdict", certifying_verdicts=certify)


def test_only_a_certifying_verdict_certifies():
    g = _gate()
    assert "marginal" in g.uncertified({"figure_of_merit": 0.55, "verdict": "marginal"})
    assert g.uncertified({"figure_of_merit": 0.85, "verdict": "accept"}) is None
    assert g.uncertified({"figure_of_merit": 0.35, "verdict": "reject"}) is None     # a reject is declared_reject's
    assert g.uncertified({"figure_of_merit": 0.55}) is None                           # nothing declared
    assert _gate(certify=None).uncertified({"verdict": "marginal"}) is None
    assert from_mapping(gate_record(g)) == g


def test_the_xrd_skill_certifies_accept_only():
    gate = resolve_gate(skill_meta=load_skill("xrd", domain="structure_matching")["meta"])
    assert gate.certifying_verdicts == ("accept",) and gate.verdict_key == "verdict"


def test_a_provisional_result_is_withheld_by_every_verdict_path():
    unit = {"success": True, "fit_quality": {"figure_of_merit": 0.55, "verdict": "marginal",
                                             "provisional": "the scoring tool's own verdict is 'marginal'"},
            "quality_history": {"approved": True, "final_r2": 0.55, "threshold": 0.70}}
    assert "marginal" in fit_check_withheld(unit)
    own = unit_verdict_for(unit)
    assert own["verified"] is False and "marginal" in own["reason"]
    follower = {**unit, "quality_history": {}}
    rec = {"unit": "anchor", "verdict": {"verified": True}}
    fv = unit_verdict_for(follower, recipe=rec)
    assert fv["verified"] is False and fv["decided_by"] == "recipe" and "marginal" in fv["reason"]


def test_the_fit_path_stamps_a_marginal_match_provisional(tmp_path, monkeypatch):
    res, _, _, _ = _replay(tmp_path, monkeypatch, {"LOW": 0.99}, quality_gate=_gate(),
                           extra_quality={"LOW": {"figure_of_merit": 0.55, "verdict": "marginal"}})
    assert "marginal" in " ".join((res.get("fit_quality") or {}).get("provisional") or [])
    # an accepted match is not
    res, _, _, _ = _replay(tmp_path / "b", monkeypatch, {"LOW": 0.99}, quality_gate=_gate(),
                           extra_quality={"LOW": {"figure_of_merit": 0.85, "verdict": "accept"}})
    assert "provisional" not in (res.get("fit_quality") or {})


def test_a_replay_is_judged_by_its_recipes_recorded_gate_too(tmp_path, monkeypatch):
    # the review's B1: no XRD gate on the run (the live fast path, a reuse not given the skill), the
    # recipe recorded under the XRD gate; a multiphase "marginal" at 0.72 passes the replay gate's number
    res, _, _, _ = _replay(tmp_path, monkeypatch, {"LOW": 0.99},
                           state_extra={"_reuse_gate": gate_record(_gate())},
                           extra_quality={"LOW": {"figure_of_merit": 0.72, "verdict": "marginal"}})
    assert "marginal" in " ".join((res.get("fit_quality") or {}).get("provisional") or [])
    assert unit_verdict_for(res)["verified"] is False


def test_the_table_carries_a_flag_not_a_column(tmp_path):
    import csv, json
    from unittest.mock import MagicMock
    from scilink.agents.exp_agents.controllers.curve_fitting_controllers import UnifiedSeriesProcessingController
    from scilink.agents.exp_agents.feature_table import describe_feature_table, write_feature_table
    rows = [{"index": i, "name": f"s{i}", "success": True, "parameters": {"identified_phase": "Si"},
             "fit_quality": {"figure_of_merit": 0.85 if i != 1 else 0.55, "verdict": "accept" if i != 1 else "marginal",
                             **({"provisional": ["the scoring tool's own verdict is 'marginal'"]} if i == 1 else {})},
             "unit_verdict": {"verified": i != 1, "decided_by": "qc_gate" if i == 0 else "recipe"}} for i in range(3)]
    c = UnifiedSeriesProcessingController.__new__(UnifiedSeriesProcessingController)
    c.logger = MagicMock(); c.r2_threshold = 0.95; c.outlier_sigma = 3.0
    flags = {f["index"]: f["reason"] for f in c._detect_outliers(rows, gate=_gate())}
    # one flag per unit: a marginal figure below accept is flagged by the band first
    assert flags.get(1) in ("provisional", "below_threshold", "outlier_and_below_threshold")
    for r in rows:
        r["flag_reason"] = flags.get(r["index"])
    (tmp_path / "series_fit_results.json").write_text(json.dumps({"results": rows}))
    path = write_feature_table(tmp_path)
    header = next(csv.reader(open(path)))
    assert not any("provisional" in h for h in header)
    assert "fit_provisional" not in str(describe_feature_table(path))


def test_a_provisional_unit_in_the_band_gets_the_provisional_flag_when_no_other():
    # a multiphase "marginal" ABOVE accept is not below threshold: the caveat flag names it
    from unittest.mock import MagicMock
    from scilink.agents.exp_agents.controllers.curve_fitting_controllers import UnifiedSeriesProcessingController
    rows = [{"index": i, "name": f"s{i}", "success": True,
             "fit_quality": {"figure_of_merit": 0.80, "verdict": "accept"},
             "unit_verdict": {"verified": True, "decided_by": "qc_gate" if i == 0 else "recipe"}} for i in range(5)]
    rows.append({"index": 5, "name": "s5", "success": True,
                 "fit_quality": {"figure_of_merit": 0.80, "verdict": "marginal",
                                 "provisional": ["the scoring tool's own verdict is 'marginal'"]},
                 "unit_verdict": {"verified": False, "decided_by": "recipe"}})
    g = QualityGate(metric="figure_of_merit", accept_threshold=0.70, hard_reject_threshold=0.40,
                    physical_review=False, verdict_key="verdict", certifying_verdicts=("accept",))
    c = UnifiedSeriesProcessingController.__new__(UnifiedSeriesProcessingController)
    c.logger = MagicMock(); c.r2_threshold = 0.95; c.outlier_sigma = 3.0
    flags = {f["index"]: f for f in c._detect_outliers(rows, gate=g)}
    assert flags[5]["reason"] == "provisional" and "Provisional" in flags[5]["recommendation"]


def test_the_report_badge_says_provisional():
    from scilink.agents.exp_agents.controllers.curve_fitting_controllers import GenerateCurveFittingReportController
    ctrl = GenerateCurveFittingReportController.__new__(GenerateCurveFittingReportController)
    ctrl.r2_threshold = 0.95
    html = ctrl._format_fit_quality({"figure_of_merit": 0.72, "verdict": "marginal"}, gate=_gate())
    assert "Provisional" in html and "Good" not in html


def test_a_scripts_own_provisional_key_is_not_the_gates(tmp_path, monkeypatch):
    # an R²-gated, non-XRD fit whose script writes "provisional": true is not withheld by it
    res, _, _, _ = _replay(tmp_path, monkeypatch, {"LOW": 0.99},
                           extra_quality={"LOW": {"provisional": True}})
    assert "provisional" not in (res.get("fit_quality") or {})


def test_a_short_series_still_names_its_provisional_unit():
    from unittest.mock import MagicMock
    from scilink.agents.exp_agents.controllers.curve_fitting_controllers import UnifiedSeriesProcessingController
    rows = [{"index": 0, "name": "s0", "success": True, "fit_quality": {"figure_of_merit": 0.85, "verdict": "accept"},
             "unit_verdict": {"verified": True, "decided_by": "qc_gate"}},
            {"index": 1, "name": "s1", "success": True,
             "fit_quality": {"figure_of_merit": 0.72, "verdict": "marginal",
                             "provisional": ["the scoring tool's own verdict is 'marginal'"]},
             "unit_verdict": {"verified": False, "decided_by": "recipe"}}]
    c = UnifiedSeriesProcessingController.__new__(UnifiedSeriesProcessingController)
    c.logger = MagicMock(); c.r2_threshold = 0.95; c.outlier_sigma = 3.0
    assert {f["index"]: f["reason"] for f in c._detect_outliers(rows, gate=_gate())} == {1: "provisional"}
