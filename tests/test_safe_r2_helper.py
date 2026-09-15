"""Regression test for the _safe_r2 helper.

Surfaced from the hard-reject live test: XRD-style scripts emit
FIT_RESULTS_JSON with `r_squared: null` (the natural value when the
script's scoring is FOM-based). The pre-existing pattern
`.get('r_squared', 0)` returned None in that case, which then crashed
arithmetic comparisons like `if r2 > best_r2:`.

These tests assert the helper handles every shape the LLM script can
emit and that arithmetic / comparison on its return value never
crashes.
"""
from __future__ import annotations

import math

import pytest

from scilink.agents.exp_agents.controllers.curve_fitting_controllers import _safe_r2


def test_missing_fit_quality_returns_default():
    assert _safe_r2({}) == 0.0
    assert _safe_r2({"other_key": "value"}) == 0.0


def test_missing_r_squared_key_returns_default():
    assert _safe_r2({"fit_quality": {"figure_of_merit": 0.7}}) == 0.0


def test_none_r_squared_returns_default():
    """The bug the hard-reject test surfaced — r_squared: null in the JSON."""
    assert _safe_r2({"fit_quality": {"r_squared": None}}) == 0.0


def test_valid_r_squared_returns_value():
    assert _safe_r2({"fit_quality": {"r_squared": 0.9876}}) == pytest.approx(0.9876)


def test_accepts_fit_quality_dict_directly():
    assert _safe_r2({"r_squared": 0.85}) == pytest.approx(0.85)
    assert _safe_r2({"r_squared": None}) == 0.0


def test_negative_r_squared_is_preserved():
    """R² can be mathematically negative (fit worse than the mean). Don't clamp."""
    assert _safe_r2({"fit_quality": {"r_squared": -0.5}}) == pytest.approx(-0.5)


def test_zero_r_squared_is_preserved():
    assert _safe_r2({"fit_quality": {"r_squared": 0.0}}) == 0.0


def test_string_value_returns_default():
    """If the script emits a non-numeric string, fall back to default."""
    assert _safe_r2({"fit_quality": {"r_squared": "not a number"}}) == 0.0


def test_custom_default():
    assert _safe_r2({}, default=1.0) == 1.0
    assert _safe_r2({"fit_quality": {"r_squared": None}}, default=-1.0) == -1.0


def test_non_dict_input_returns_default():
    assert _safe_r2(None) == 0.0
    assert _safe_r2("not a dict") == 0.0
    assert _safe_r2(42) == 0.0


def test_nested_non_dict_fit_quality_returns_default():
    """If fit_quality somehow ended up as a list (corrupt script output)."""
    assert _safe_r2({"fit_quality": [0.5, 0.7]}) == 0.0


def test_can_be_compared_arithmetically():
    """The whole point: every return value supports < > >= <= without raising."""
    values = [
        _safe_r2({}),
        _safe_r2({"fit_quality": {"r_squared": None}}),
        _safe_r2({"fit_quality": {"r_squared": 0.85}}),
        _safe_r2({"fit_quality": {"r_squared": "garbage"}}),
    ]
    for v in values:
        assert isinstance(v, float)
        assert v > -10.0 or v <= 1.0  # always comparable, never raises
