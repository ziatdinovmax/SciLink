"""#571 — the meta records the cross-delegation dependencies it can prove
from the task and context, on top of what the LLM declared in
context_from: a cited prior analysis id (same-series continuation), a
finding threaded verbatim, and a task that sits at the point a planning
delegation recommended (loop closure through a human courier)."""
import json

from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent


def _meta():
    """An orchestrator shell: the provenance methods only touch the ledger."""
    m = MetaOrchestratorAgent.__new__(MetaOrchestratorAgent)
    m._delegation_ledger = []
    m._auto_checkpoint = lambda verbose=True: None
    return m


def _entry(index, mode, **kw):
    e = {"index": index, "mode": mode, "status": "success", "task": f"task {index}",
         "key_findings": [], "files_produced": [], "analysis_ids": [],
         "recommended_parameters": []}
    e.update(kw)
    return e


def test_close_records_analysis_ids_and_recommended_points(tmp_path):
    m = _meta()
    hist = tmp_path / "bo_history.json"
    hist.write_text(json.dumps([
        {"step": 1, "recommendation_batch": [{"temperature": 15.0, "pressure": 2.5}]},
        {"step": 2, "recommendation_batch": [{"temperature": 20.0, "pressure": 3.0}]},
    ]))
    a = m._open_delegation("analysis", "fit the series", None, None, "series")
    m._close_delegation(a, {"status": "success", "summary": "ok", "key_findings": ["k"],
                            "files_produced": [], "analyses": [
                                {"analysis_id": "analysis_spectra_CurveFit_20260907_215143_001"}]})
    p = m._open_delegation("planning", "recommend", None, [1], "bo")
    m._close_delegation(p, {"status": "success", "summary": "ok", "key_findings": [],
                            "files_produced": [str(hist)]})
    assert a["analysis_ids"] == ["analysis_spectra_CurveFit_20260907_215143_001"]
    assert p["recommended_parameters"] == [{"temperature": 20.0, "pressure": 3.0}]  # last step
    assert a["recommended_parameters"] == []  # only planning delegations recommend


def test_infers_continuation_loop_closure_and_threaded_findings():
    m = _meta()
    m._delegation_ledger = [
        _entry(1, "analysis", analysis_ids=["analysis_spectra_CurveFit_20260907_215143_001"],
               key_findings=["The dominant peak near 400 shows no resolvable temperature "
                             "dependence between 5 K and 50 K within uncertainty."]),
        _entry(2, "planning", recommended_parameters=[{"temperature": 20.0, "pressure": 3.0}]),
        _entry(3, "analysis", analysis_ids=["analysis_other_000"], key_findings=["unrelated"]),
    ]
    # continuation: the task cites #1's analysis id; the LLM declared nothing
    e = m._open_delegation(
        "analysis",
        "Apply the SAME locked model as analysis_spectra_CurveFit_20260907_215143_001 so the "
        "features are comparable.", None, None, "refit")
    assert e["context_from"] == [1] and e["context_from_inferred"] == [1]
    # loop closure: the new measurement sits at the point #2 recommended
    e = m._open_delegation(
        "analysis", "Analyze the new spectrum measured at temperature = 20 K and pressure 3.0 bar.",
        None, None, "20 K point")
    assert e["context_from"] == [2]
    # threaded finding (the meta passed #1's finding through context) + declared #2
    e = m._open_delegation(
        "planning", "Plan the next step.",
        {"findings": ["The dominant peak near 400 shows no resolvable temperature dependence "
                      "between 5 K and 50 K within uncertainty."]}, [2], "next")
    assert e["context_from"] == [1, 2] and e["context_from_inferred"] == [1]
    # nothing provable → only what was declared; a bare number is not a match
    e = m._open_delegation("analysis", "Look at 3.0 things at 20 degrees.", None, ["#3"], "x")
    assert e["context_from"] == [3] and e["context_from_inferred"] == []


def test_single_parameter_point_needs_its_name():
    m = _meta()
    m._delegation_ledger = [_entry(1, "planning", recommended_parameters=[{"dose": 20.0}])]
    assert m._open_delegation("analysis", "20 images to segment", None, None, "x")["context_from"] == []
    assert m._open_delegation("analysis", "Analyze the sample irradiated at dose 20", None, None, "x")["context_from"] == [1]


def test_stated_recommendation_closes_the_loop_without_a_bo_history():
    """The live case: the BO tool declined to fit on two points, the planning
    specialist stated "Next temperature to measure: ≈ 27.5 K" in prose, and
    the later analysis of that point must still be recorded as depending on
    it — while a bare "27.5" elsewhere must not."""
    m = _meta()
    p = m._open_delegation("planning", "recommend", None, [1], "bo")
    m._close_delegation(p, {"status": "success", "key_findings": ["Optimization target: peak_1_fwhm (maximize)."],
                            "summary": "The optimizer will not run on 2 points.\n\n## Recommendation\n\n"
                                       "**Next temperature to measure: ≈ 27.5 K** (the midpoint of the 5–50 K range).\n",
                            "files_produced": []})
    assert p["recommended_parameters"] == []
    vals = {(v["value"], "temperature" in v["context"]) for v in p["recommended_values"]}
    assert (27.5, True) in vals
    assert not any(v["value"] in (5.0, 50.0) for v in p["recommended_values"])  # range endpoints excluded
    # counts are not values: "3 points" / "3-point floor" never become a recommendation
    q = m._open_delegation("planning", "recommend", None, [1], "bo")
    m._close_delegation(q, {"status": "success", "key_findings": [],
                            "summary": "The recommended step needs at least 3 points; the 3-point floor "
                                       "cannot be lowered. Recommended: set the temperature at 34.2 K next.",
                            "files_produced": []})
    assert [v["value"] for v in q["recommended_values"]] == [34.2]
    e = m._open_delegation("analysis", "A new spectrum was measured at temperature = 27.5 K: uploads/spectrum_27p5K.csv",
                           None, None, "27.5 K point")
    assert e["context_from"] == [1] and e["context_from_inferred"] == [1]
    e = m._open_delegation("planning", "The 27.5 K point you recommended has now been measured; re-run the BO "
                           "step over temperature with all three points.", None, None, "bo 3 points")
    assert 1 in e["context_from"]
    # the same number without any framing word is not a match
    e = m._open_delegation("analysis", "Segment 27.5 percent of the images.", None, None, "x")
    assert e["context_from"] == []


def test_inferred_fusion_edge_does_not_trigger_the_fusion_feedback_stamp():
    """Provenance only: an analysis that quotes a fusion's finding gets the
    fusion in context_from (inferred), but the independence stamp — which
    appends the ADDITIVE-ONLY note to the task — still follows what the
    LLM declared, so an inferred edge never changes a specialist's task."""
    m = _meta()
    finding = "Both datasets show the same 0.3 eV blue shift of the plasmon peak at the interface."
    m._delegation_ledger = [_entry(1, "fusion", key_findings=[finding], labels=["a", "b"])]
    e = m._open_delegation("analysis", "Re-check this: " + finding, None, None, "recheck")
    assert e["context_from"] == [1] and e["context_from_inferred"] == [1]
    assert e.get("informed_via") is None and "informed_by" not in e
    e = m._open_delegation("analysis", "Re-check this: " + finding, None, [1], "recheck")
    assert e["informed_via"] == "fusion_feedback" and e["informed_by"] == ["a", "b"]
