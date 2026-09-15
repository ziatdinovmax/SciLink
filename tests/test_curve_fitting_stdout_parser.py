"""Tests for the curve-fitting stdout marker parser.

The parser recognizes two markers emitted by analysis scripts:
  - ``FIT_RESULTS_JSON: {...}`` — the standard fit-results payload
  - ``DB_MATCHES_JSON: {...}`` — emitted by the structure_matching skill's
    ``search_structures`` tool; merged into ``fit_results['db_matches']``.
"""

from __future__ import annotations

import json

import pytest

from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
    _parse_script_markers,
)


def test_parser_handles_empty_stdout():
    assert _parse_script_markers("") == {}
    assert _parse_script_markers(None) == {}


def test_parser_finds_fit_results():
    stdout = 'noise\nFIT_RESULTS_JSON: {"r_squared": 0.97}\nmore noise\n'
    out = _parse_script_markers(stdout)
    assert out == {"r_squared": 0.97}


def test_parser_takes_first_fit_results():
    """First parseable FIT_RESULTS_JSON wins (preserves long-standing behavior)."""
    stdout = (
        'FIT_RESULTS_JSON: {"r_squared": 0.50}\n'
        'FIT_RESULTS_JSON: {"r_squared": 0.99}\n'
    )
    out = _parse_script_markers(stdout)
    assert out["r_squared"] == 0.50


def test_parser_ignores_later_fit_results_even_when_first_is_malformed():
    """Once the FIT marker is seen, later FIT lines are skipped — matches
    the original break-on-first behavior."""
    stdout = (
        'FIT_RESULTS_JSON: {malformed json\n'
        'FIT_RESULTS_JSON: {"r_squared": 0.99}\n'
    )
    out = _parse_script_markers(stdout)
    assert out == {}


def test_parser_recognizes_db_matches():
    matches = {"candidates": [{"id": "mp-149", "formula": "Si"}], "sources_queried": ["mp"]}
    stdout = "DB_MATCHES_JSON: " + json.dumps(matches)
    out = _parse_script_markers(stdout)
    assert out["db_matches"] == matches


def test_parser_combines_both_markers():
    """The structure_matching skill emits DB_MATCHES_JSON early then
    FIT_RESULTS_JSON at the end. Both must be captured."""
    matches = {"candidates": [{"id": "mp-149", "formula": "Si"}]}
    fit = {"r_squared": 0.96, "verdict": "accept"}
    stdout = (
        "stage 1: searching DB\n"
        "DB_MATCHES_JSON: " + json.dumps(matches) + "\n"
        "stage 2: fitting\n"
        "FIT_RESULTS_JSON: " + json.dumps(fit) + "\n"
        "done\n"
    )
    out = _parse_script_markers(stdout)
    assert out["r_squared"] == 0.96
    assert out["verdict"] == "accept"
    assert out["db_matches"] == matches


def test_parser_handles_db_marker_after_fit_marker():
    """Even when scripts emit DB_MATCHES_JSON after FIT_RESULTS_JSON (out
    of skill convention), both are captured."""
    matches = {"candidates": []}
    fit = {"r_squared": 0.5}
    stdout = (
        "FIT_RESULTS_JSON: " + json.dumps(fit) + "\n"
        "DB_MATCHES_JSON: " + json.dumps(matches) + "\n"
    )
    out = _parse_script_markers(stdout)
    assert out["r_squared"] == 0.5
    assert out["db_matches"] == matches


def test_parser_takes_first_db_matches():
    a = {"candidates": [{"id": "first"}]}
    b = {"candidates": [{"id": "second"}]}
    stdout = (
        "DB_MATCHES_JSON: " + json.dumps(a) + "\n"
        "DB_MATCHES_JSON: " + json.dumps(b) + "\n"
    )
    out = _parse_script_markers(stdout)
    assert out["db_matches"] == a


def test_parser_skips_malformed_db_matches():
    stdout = (
        "DB_MATCHES_JSON: not valid json\n"
        "FIT_RESULTS_JSON: {\"r_squared\": 0.9}\n"
    )
    out = _parse_script_markers(stdout)
    # FIT still parsed
    assert out["r_squared"] == 0.9
    # db_matches absent because the only DB line was malformed
    assert "db_matches" not in out


def test_parser_does_not_override_explicit_db_matches_in_fit_payload():
    """If a script puts db_matches inside FIT_RESULTS_JSON directly,
    the stdout marker should not overwrite it."""
    explicit = {"candidates": [{"id": "from_fit"}]}
    from_marker = {"candidates": [{"id": "from_marker"}]}
    stdout = (
        'FIT_RESULTS_JSON: ' + json.dumps({"db_matches": explicit, "r_squared": 0.95}) + '\n'
        'DB_MATCHES_JSON: ' + json.dumps(from_marker) + '\n'
    )
    out = _parse_script_markers(stdout)
    assert out["db_matches"] == explicit  # setdefault, not overwrite


def test_has_fit_parameters_requires_nonempty_top_level_dict():
    """The marker alone is not the contract: a payload without a non-empty
    top-level ``parameters`` dict (e.g. a script that invents its own
    layout) must be treated as a missing output so the correction loop
    runs — otherwise the series feature table has nothing to optimize."""
    from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
        _has_fit_parameters,
    )
    good = "FIT_RESULTS_JSON:" + json.dumps(
        {"model_type": "gauss", "parameters": {"peak_1": {"center": 532.0}},
         "fit_quality": {"r_squared": 0.99}})
    assert _has_fit_parameters(good)
    own_layout = "FIT_RESULTS_JSON:" + json.dumps(
        {"model": "pv", "spectra": [{"peak": {"amplitude": 1.0}}]})
    assert not _has_fit_parameters(own_layout)
    assert not _has_fit_parameters(
        "FIT_RESULTS_JSON:" + json.dumps({"parameters": {}}))
    assert not _has_fit_parameters(
        "FIT_RESULTS_JSON:" + json.dumps({"parameters": [1, 2]}))
    assert not _has_fit_parameters("no marker at all")
    assert not _has_fit_parameters(None)
