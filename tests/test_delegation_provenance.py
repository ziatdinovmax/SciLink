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
