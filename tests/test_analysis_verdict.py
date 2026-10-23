"""``analysis_verdict`` over the shapes the agents return: a salvaged,
unverified or unapproved run stays provisional whatever its status says; a
curve, image or hyperspectral series is judged by the units' own records as
the drivers stamp them. The real-path series cases are in
``test_series_verdict_path.py``; these are the unit cases of the reader.
"""

from scilink.agents.exp_agents._verification_record import analysis_verdict, series_anchor_unit


def test_a_salvaged_or_unverified_analysis_stays_provisional():
    """An agent returns "success" for a salvaged best-available fit, an
    unverified (out of budget) run and a result the verifier never approved
    (review of #702). The row's verdict decides, never the status."""
    ok = {"status": "success", "quality_history": {"approved": True}}
    assert analysis_verdict(ok) == {"verified": True, "reason": "met the acceptance threshold"}
    assert not analysis_verdict({**ok, "quality_warning": "below threshold"})["verified"]
    assert "did not finish" in analysis_verdict(
        {"status": "success", "quality_history": {"approved": False, "unverified": True, "stopped_by": "time_budget"}})["reason"]
    assert "did not approve" in analysis_verdict({"status": "success", "quality_history": {"approved": False}})["reason"]
    assert "no verification record" in analysis_verdict({"status": "success"})["reason"]
    # a verifier may approve below the numeric threshold on physics grounds (seen live): verified
    physics = {"status": "success", "quality_history": {"final_r2": 0.98, "threshold": 0.99999, "approved": True,
                                                        "approved_by": "verifier",
                                                        "verification_iterations": [{"r_squared": 0.98}]}}
    assert analysis_verdict(physics) == {"verified": True, "reason": "approved by the analysis verifier"}
    # a bypassed verification with the metric below its threshold: nobody looked
    bypass = {"status": "success", "quality_history": {"final_r2": 0.98, "threshold": 0.99, "approved": True,
                                                       "approved_by": "verifier", "verification_iterations": []}}
    assert "bypassed" in analysis_verdict(bypass)["reason"]
    bypass["quality_history"]["approved_by"] = "bypass"
    assert "bypassed" in analysis_verdict(bypass)["reason"]
    bypass["quality_history"]["final_r2"] = 0.995                       # the metric met the threshold: verified
    assert analysis_verdict(bypass) == {"verified": True, "reason": "met the acceptance threshold"}
    assert "reused script" in analysis_verdict({**ok, "reuse_validity": {"reused": True, "verdict": "poor"}})["reason"]
    assert analysis_verdict({**ok, "reuse_validity": {"reused": True, "verdict": "good"}})["verified"]
    assert analysis_verdict({"status": "partial", "quality_history": {"approved": True}})["reason"] == "status 'partial'"
    # a single run the judge picked as best available
    assert "judge found no acceptable" in analysis_verdict(
        {"status": "success", "judge_warning": "Judge selected this as best available",
         "quality_history": {"approved": True, "final_r2": 0.97, "threshold": 0.95}})["reason"]
    # a good-verdict locked-script reuse has no QC-engine record: the replay gate passed it
    assert analysis_verdict({"status": "success", "reuse_validity": {"reused": True, "verdict": "good"}}) == {
        "verified": True, "reason": "locked-script reuse passed the replay gate"}
    # an image locked replay the gate rejected says so
    assert "replay gate rejected" in analysis_verdict(
        {"status": "success", "reuse_validity": {"reused": True, "verdict": "good"},
         "quality_history": {"approved": False, "approved_by": None, "final_score": 0.4, "threshold": 0.7}})["reason"]
    # a row stamped verified by an older caller with no record is not promoted on its status
    assert not analysis_verdict({"status": "success", "verified": True})["verified"]


def _curve_series(anchor_qh, followers=2, anchor_extra=None, follower_qh=None):
    """individual_results as CurveFittingAgent._compile_results writes them
    (curve_fitting_agent.py): the anchor carries the QC engine's record, a
    follower fitted by _fit_single_spectrum carries at most the profile
    stamp; quality_warning / judge_warning ride on the unit."""
    def unit(i, name, qh, extra=None):
        return {"index": i, "name": name, "success": True, "model_type": "pseudo_voigt",
                "parameters": {"peak_1.center": 144.0}, "fit_quality": {"r_squared": 0.99},
                "visualization_path": f"/out/spectrum_{i:04d}/fit.png", "error": None, "flagged": False,
                "flag_reason": None, "flag_recommendation": None, "adaptively_refitted": False,
                "original_r2": None, "locked_model_type": "pseudo_voigt", "quality_history": qh,
                "reuse_validity": None, **(extra or {})}
    items = [unit(0, "spectrum_0000", anchor_qh, anchor_extra)]
    for i in range(1, followers + 1):
        items.append(unit(i, f"spectrum_{i:04d}", follower_qh))
    return {"status": "success", "individual_results": items, "flagged_spectra": [],
            "scientific_claims": [{"claim": "anatase throughout"}]}


ANCHOR_OK = {"final_r2": 0.995, "threshold": 0.95, "approved": True,
             "verification_iterations": [{"r_squared": 0.995, "annealing_level": 0}],
             "alternative_models": [], "script_errors": [], "judge_reasoning": None}


def test_series_verdicts_follow_the_agents_shapes():
    """Review of #702, round 2: series followers have no verification record
    (they replay the locked recipe), a salvaged series anchor keeps
    approved = R² >= threshold, hyperspectral keeps its histories on the
    target records and its series rows carry `verified` only."""
    # a clean curve series: anchor approved, followers replayed
    v = analysis_verdict(_curve_series(ANCHOR_OK))
    assert v == {"verified": True, "reason": "every unit verified by the series driver"}
    # quick profile: followers carry only the profile stamp — still verified
    v = analysis_verdict(_curve_series({**ANCHOR_OK, "produced_under_profile": "quick"},
                                       follower_qh={"produced_under_profile": "quick"}))
    assert v["verified"]
    # a salvaged anchor: R² above threshold but the verifier kept rejecting, judge fallback
    salvaged = _curve_series({**ANCHOR_OK, "judge_reasoning": "none acceptable"},
                             anchor_extra={"quality_warning": "R² = 0.9950 meets the threshold 0.95 but the fit "
                                                              "was not accepted on physical grounds"})
    assert analysis_verdict(salvaged) == {"verified": False, "reason": "salvaged best-available result (unit spectrum_0000)"}
    judged = _curve_series(ANCHOR_OK, anchor_extra={"judge_warning": "Judge selected this as best available"})
    assert "judge" in analysis_verdict(judged)["reason"]
    # an anchor cut by the budget
    cut = _curve_series({**ANCHOR_OK, "approved": False, "unverified": True, "stopped_by": "time_budget"})
    assert analysis_verdict(cut)["reason"] == "verification did not finish (unit spectrum_0000): time_budget"
    # a follower whose adaptive refit was cut: unverified, and in the table
    unv = _curve_series(ANCHOR_OK, follower_qh={"approved": False, "unverified": True})
    assert "verification did not finish (unit spectrum_0001)" in analysis_verdict(unv)["reason"]
    unv2 = _curve_series(ANCHOR_OK, follower_qh={"unverified": True, "produced_under_profile": "quick"})
    assert analysis_verdict(unv2)["reason"] == "follower unverified (unit spectrum_0001)"
    # a failed unit is flagged and excluded by the agent: it does not block
    failed = _curve_series(ANCHOR_OK)
    failed["individual_results"][2].update({"success": False, "error": "fit diverged", "quality_history": None})
    assert analysis_verdict(failed) == {"verified": True, "reason": "every unit verified by the series driver "
                                                                    "(1 failed unit(s) excluded by the agent)"}
    # a refit the driver accepted by its consistency rule is held like a follower: finished, not
    # unverified — not to the anchor's salvage markers (a refit that improved a unit must not
    # unverify a series the unrefit unit would have passed)
    refit = _curve_series(ANCHOR_OK)
    refit["individual_results"][1].update({"adaptively_refitted": True, "original_r2": 0.5,
                                           "quality_warning": "R² = 0.9000 below threshold 0.95",
                                           "quality_history": {**ANCHOR_OK, "final_r2": 0.9, "approved": False}})
    assert analysis_verdict(refit)["verified"]
    refit["individual_results"][1]["quality_history"]["unverified"] = True
    assert analysis_verdict(refit)["reason"] == "refit unverified (unit spectrum_0001)"
    # round 4: a salvaged ANCHOR is not laundered by refitting it — the anchor keeps its role
    # (the controllers stamp role="anchor" on first-in-regime units and carry it through a
    # refit) and the anchor's bar; the series stays provisional, as it was before the refit
    laundered = _curve_series({**ANCHOR_OK, "final_r2": 0.86, "approved": False},
                              anchor_extra={"role": "anchor", "adaptively_refitted": True, "original_r2": 0.80,
                                            "quality_warning": "R² = 0.8600 below threshold 0.95"})
    assert analysis_verdict(laundered)["reason"] == "salvaged best-available result (unit spectrum_0000)"
    # round 5: an approved anchor refit (M2) does not vouch for followers still replaying the
    # salvaged M1 — the refit carries a summary of the unit it replaced, and followers on the
    # locked script are judged by THAT recipe
    m1 = {"name": "spectrum_0000", "quality_history": {"approved": False, "final_r2": 0.80, "threshold": 0.95},
          "quality_warning": "R² = 0.8000 below threshold 0.95", "judge_warning": None, "reuse_validity": None}
    relocked = _curve_series({**ANCHOR_OK, "final_r2": 0.97},
                             anchor_extra={"role": "anchor", "adaptively_refitted": True, "original_r2": 0.80,
                                           "replaced_unit": m1}, follower_qh={"produced_under_profile": "quick"})
    for u in relocked["individual_results"][1:]:
        u["fitted_from"] = "locked_script"
    assert analysis_verdict(relocked)["reason"] == (
        "follower replays a recipe that was not approved (unit spectrum_0001): salvaged best-available "
        "result (recipe from unit spectrum_0000, since refit)")
    # ... but a follower that was itself refit (its own record) is judged on its own
    relocked["individual_results"][1].update({"adaptively_refitted": True, "quality_history": dict(ANCHOR_OK)})
    relocked["individual_results"][2].update({"adaptively_refitted": True, "quality_history": dict(ANCHOR_OK)})
    assert analysis_verdict(relocked)["verified"]
    # a refit anchor that WAS approved after the refit, whose original was approved too: verified
    approved_refit = _curve_series(ANCHOR_OK, anchor_extra={
        "role": "anchor", "adaptively_refitted": True,
        "replaced_unit": {"name": "spectrum_0000", "quality_history": {**ANCHOR_OK, "final_r2": 0.96}}})
    assert analysis_verdict(approved_refit)["verified"]
    # the opposite direction (accepted as conservative): an approved anchor refit to a salvaged unit
    # is a salvaged row in the table, whatever its followers replayed
    flipped = _curve_series({**ANCHOR_OK, "final_r2": 0.93, "approved": False},
                            anchor_extra={"role": "anchor", "adaptively_refitted": True,
                                          "quality_warning": "R² = 0.9300 below threshold 0.95",
                                          "replaced_unit": {"name": "spectrum_0000",
                                                            "quality_history": {**ANCHOR_OK, "final_r2": 0.92}}})
    assert analysis_verdict(flipped)["reason"] == "salvaged best-available result (unit spectrum_0000)"
    # the recipe is the first ANCHOR by role, refit or not
    assert series_anchor_unit(relocked) == "spectrum_0000"
    # an anchor refit WITHOUT a role (a checkpoint from before the stamp) is not counted as an anchor
    old_shape = _curve_series({**ANCHOR_OK, "approved": False},
                              anchor_extra={"adaptively_refitted": True, "quality_warning": "below"})
    assert analysis_verdict(old_shape)["reason"] == "no unit carries a verification record"
    # a follower of a FAILED regime anchor is fresh code with no verifier (round 3)
    fresh = _curve_series(ANCHOR_OK)
    fresh["individual_results"][2]["fitted_from"] = "fresh_code"
    assert analysis_verdict(fresh)["reason"] == "follower fitted without a locked recipe (unit spectrum_0002)"
    fresh["individual_results"][2]["fitted_from"] = "locked_script"
    assert analysis_verdict(fresh)["verified"]
    # a series whose anchor is a good-verdict locked reuse (no QC-engine record) is verified by the gate
    reused = _curve_series(None)
    reused["individual_results"][0]["reuse_validity"] = {"reused": True, "verdict": "good", "r_squared": 0.99}
    assert analysis_verdict(reused)["verified"]
    reused["individual_results"][0]["reuse_validity"]["verdict"] = "poor"
    assert "reused script verdict 'poor' (unit spectrum_0000)" in analysis_verdict(reused)["reason"]
    # a cut anchor reads as cut, even when the deterministic gate also warned
    both = _curve_series({**ANCHOR_OK, "approved": False, "unverified": True, "stopped_by": "time_budget"},
                         anchor_extra={"quality_warning": "R² = 0.80 below threshold 0.95"})
    assert analysis_verdict(both)["reason"].startswith("verification did not finish")
    # image series (image_analysis_agent.py): the unit record now travels with the same names
    img = {"status": "success", "individual_results": [
        {"index": 0, "name": "img0", "success": True, "analysis_type": "particle", "verification_score": 0.9,
         "quality_history": {"final_score": 0.9, "threshold": 0.7, "approved": True,
                             "verification_iterations": [{"quality_score": 0.9}]}, "adaptively_refitted": False},
        {"index": 1, "name": "img1", "success": True, "analysis_type": "particle", "verification_score": None,
         "quality_history": None, "adaptively_refitted": False}]}
    assert analysis_verdict(img)["verified"]
    # hyperspectral single cube (hyperspectral_controllers._build_target_record)
    rec = {"target": "phase_map", "required_outputs": ["phase_map"], "task_success": True, "salvaged": False,
           "script": "import numpy", "quality_history": {"final_passed_fraction": 1.0, "threshold": 0.8,
                                                         "approved": True, "verification_iterations": [{}]}}
    cube = {"status": "success", "dynamic_analysis_records": [rec, {"target": "nm", "task_success": False,
                                                                    "salvaged": False, "script": None,
                                                                    "not_measurable": {"reason": "x"}}]}
    assert analysis_verdict(cube) == {"verified": True, "reason": "every target passed verification"}
    cube["dynamic_analysis_records"][0] = {**rec, "task_success": False, "salvaged": True,
                                           "quality_history": {**rec["quality_history"], "approved": False}}
    assert analysis_verdict(cube)["reason"] == "salvaged target (target phase_map)"
    # a target that failed before any code ran is not "unscripted and ignored"
    cube["dynamic_analysis_records"][0] = {"target": "phase_map", "task_success": False, "salvaged": False,
                                           "script": None, "required_outputs": ["phase_map"]}
    assert analysis_verdict(cube)["reason"] == "target failed before any code ran (target phase_map)"
    assert analysis_verdict({**cube, "status": "partial"})["reason"] == "status 'partial'"
    # hyperspectral series (hyperspectral_series._row): rows carry verified, no history
    def row(i, ok, verified, role="follower"):
        return {"index": i, "name": f"cube{i}", "data_path": f"/d/cube{i}.npy", "success": ok, "status": "success",
                "role": role, "confidence": "high", "output_directory": f"/o/dataset_{i:04d}", "error": None,
                "flagged": False, "flag_reason": None, "adaptively_refitted": False, "reuse_validity": None,
                "quality_metrics": {"n_targets": 2, "n_approved": 2}, "warnings": [], "regime": "r1",
                "verified": verified, "n_features": 3}
    hs = {"status": "success", "individual_results": [row(0, True, True, "anchor"), row(1, True, True)]}
    assert analysis_verdict(hs)["verified"]
    hs["individual_results"][1]["verified"] = False
    assert analysis_verdict(hs)["reason"] == "unit not verified by the series driver (unit cube1)"
    # held to the single-cube rule: a partial (salvaged / degraded) row or one that extracted nothing
    hs["individual_results"][1].update({"verified": True, "status": "partial"})
    assert not analysis_verdict(hs)["verified"]
    hs["individual_results"][1].update({"status": "success", "n_features": 0})
    assert not analysis_verdict(hs)["verified"]
    # ... and every target approved: the driver's `verified` asks for one
    hs["individual_results"][1].update({"n_features": 3, "quality_metrics": {"n_targets": 2, "n_approved": 1}})
    assert analysis_verdict(hs)["reason"] == "not every target of the unit was approved: 1 of 2 (unit cube1)"
    hs["individual_results"][1]["quality_metrics"] = {"n_targets": 0, "n_approved": 0}
    assert analysis_verdict(hs)["reason"] == "the unit has no dynamic-analysis record (unit cube1)"
