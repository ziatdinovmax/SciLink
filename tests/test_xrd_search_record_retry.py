"""A missing figure of merit is retried with its true cause, read from the
search tool's own record.

On saved ibuprofen-sodium runs, generated scripts turned a failed search or
scoring call into an empty candidate list; the gate then read the missing
figure as a hard reject and told the retry to "widen the database query,
broaden the chemistry hypothesis", and the retries ended on spurious phases.
The search now prints what it did, the agent lifts every record, and the
retry is told the cause; no cause points it at a different chemistry.
"""

from __future__ import annotations

import json

from tests.test_identification_verdict import _ctrl, _fit, _gate
from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
    _parse_script_markers, _workflow_reject_fix,
)


def _rec(outcomes, n=0, chemistry=(("C", "H", "Na", "O"),), filters=None):
    clean = n == 0 and bool(outcomes) and all(o.get("status") == "ok" and not o.get("dropped")
                                               and not o.get("truncated") for o in outcomes)
    return {"query": {"chemistry": [list(c) for c in chemistry], "filters": filters or {}, "top_n": 10},
            "outcomes": outcomes, "n_candidates": n, "clean_empty": clean}


def _line(rec):
    return "DB_MATCHES_JSON: " + json.dumps({"candidates": [], "sources_queried": [], "warnings": [],
                                             "search_record": rec})


def test_every_search_record_is_lifted_and_a_script_copy_is_dropped():
    a = _rec([{"backend": "cod", "chemistry": ["Si"], "status": "failed", "error": "down"}])
    b = _rec([{"backend": "cod", "chemistry": ["Ge"], "status": "ok", "n": 0}])
    out = _parse_script_markers("\n".join([
        _line(a), _line(b),
        'FIT_RESULTS_JSON: {"fit_quality": {}, "search_records": [{"forged": true}]}']))
    assert out["search_records"] == [a, b]
    # the first marker still fills db_matches, as before
    assert out["db_matches"]["search_record"] == a
    assert "search_records" not in _parse_script_markers('FIT_RESULTS_JSON: {"search_records": [1]}')


def _fix(fit):
    v = _ctrl()._verify_fit_with_llm({"quality_gate": _gate()}, fit)
    assert v["fit_acceptable"] is False and v["recommended_action"] == "retry_fitting_attempt_with_changes"
    issue = v["issues_found"][0]
    assert "broaden the chemistry" not in issue["suggested_fix"]
    return issue["problem"], issue["suggested_fix"]


def test_a_failed_search_is_named_and_not_a_reason_to_change_chemistry():
    fit = _fit(None, "no_candidates")
    fit["search_records"] = [_rec([{"backend": "cod", "chemistry": ["C", "H", "Na", "O"],
                                    "status": "failed", "error": "COD REST search failed: timed out"}])]
    problem, fix = _fix(fit)
    assert "did not run cleanly" in problem and "timed out" in problem
    assert "not evidence that nothing matches" in fix
    # live, "fix the failure" was read as licence to score a reference written into the script
    assert "fix or report" not in fix and "never score against a reference" in fix


def test_candidates_lost_before_scoring_are_named():
    fit = _fit(None, "no_candidates")
    fit["search_records"] = [_rec([{"backend": "cod", "chemistry": ["C"], "status": "ok", "n": 10}], n=10)]
    problem, fix = _fix(fit)
    assert "returned 10 candidate" in problem and "let scoring errors raise" in fix


def test_a_clean_empty_search_states_the_query_and_the_principle():
    fit = _fit(None, "no_candidates")
    fit["search_records"] = [_rec([{"backend": "cod", "chemistry": ["Xe", "F"], "status": "ok", "n": 0}],
                                  chemistry=(("Xe", "F"),), filters={"space_group_hints": [139]})]
    problem, fix = _fix(fit)
    assert "ran cleanly" in problem and "Xe" in problem and "space_group_hints" in problem
    assert "must come from the sample's own record" in fix


def test_files_that_could_not_be_parsed_are_named():
    fit = _fit(None, "no_candidates")
    rec = _rec([{"backend": "local", "chemistry": ["Si"], "status": "ok", "n": 0, "unparsed": 3}])
    rec["clean_empty"] = False
    fit["search_records"] = [rec]
    problem, _ = _fix(fit)
    assert "did not run cleanly" in problem and "3 files could not be parsed" in problem


def test_a_clean_empty_search_names_the_backends_not_asked():
    fit = _fit(None, "no_candidates")
    fit["search_records"] = [_rec([{"backend": "cod", "chemistry": ["Si"], "status": "ok", "n": 0},
                                   {"backend": "mp", "chemistry": ["Si"], "status": "skipped",
                                    "reason": "no API key"}])]
    fit["search_records"][0]["clean_empty"] = True
    problem, _ = _fix(fit)
    assert "ran cleanly" in problem and "not asked: mp (no API key)" in problem


def test_no_record_asks_for_the_figure_and_for_errors_to_raise():
    problem, fix = _fix(_fit(None))
    assert "was not reported" in problem and "must raise" in fix


def test_a_scored_reject_points_at_wavelength_and_metadata():
    problem, fix = _fix(_fit(0.2, "reject"))
    assert "wavelength" in fix and "must come from the sample's own record" in fix
    problem, fix = _fix(_fit(0.2))
    assert "below acceptable range" in problem


def test_one_unclean_search_among_several_is_not_a_clean_empty():
    recs = [_rec([{"backend": "cod", "chemistry": ["Si"], "status": "ok", "n": 0}]),
            _rec([{"backend": "cod", "chemistry": ["Ge"], "status": "ok", "n": 0, "truncated": True}])]
    problem, _ = _workflow_reject_fix(_gate(), {"search_records": recs}, None)
    assert "did not run cleanly" in problem and "row cap" in problem


def test_a_declared_reject_without_a_figure_is_a_scored_reject():
    fit = _fit(None, "reject")
    fit["search_records"] = [_rec([{"backend": "cod", "chemistry": ["C"], "status": "ok", "n": 5}], n=5)]
    v = _ctrl()._verify_fit_with_llm({"quality_gate": _gate()}, fit)
    fix = v["issues_found"][0]["suggested_fix"]
    assert "wavelength" in fix and "lost between" not in fix


def test_a_malformed_record_does_not_crash_the_verifier():
    fit = _fit(None, "no_candidates")
    fit["search_records"] = [{"query": None, "outcomes": [None, {"status": "failed", "error": "x"}],
                              "n_candidates": "many"}]
    problem, fix = _fix(fit)
    assert "did not run cleanly" in problem
    # a record with no outcomes list is no record
    problem, _ = _workflow_reject_fix(_gate(), {"search_records": [{"n_candidates": 3}]}, None)
    assert "was not reported" in problem


def test_end_to_end_a_real_failed_search_reaches_the_retry_text(tmp_path, capsys):
    """The tool's own marker, through the agent's parser, into the reject text."""
    from unittest.mock import patch
    from scilink.skills.structure_matching._backends import LocalCIFBackend
    from scilink.skills.structure_matching.xrd.search_structures import search_structures

    def down(self, spec):
        raise RuntimeError("connection refused")

    (tmp_path / "cifs").mkdir()
    with patch.object(LocalCIFBackend, "query", down), \
         patch.dict("os.environ", {"SCILINK_LOCAL_CIF_DIR": str(tmp_path / "cifs")}):
        search_structures(query={"chemistry": ["Si"]}, sources=["local"], output_dir=str(tmp_path / "o"))
    printed = capsys.readouterr().out + '\nFIT_RESULTS_JSON: {"fit_quality": {"verdict": "no_candidates"}}'
    parsed = _parse_script_markers(printed)
    fit = {**_fit(None, "no_candidates"), "search_records": parsed["search_records"]}
    problem, fix = _fix(fit)
    assert "connection refused" in problem and "not evidence" in fix
