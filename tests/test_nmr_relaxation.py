"""Offline tests for the NMR relaxation (T1/T2) fitter."""

from __future__ import annotations

import numpy as np
import pytest

from scilink.skills.curve_fitting.nmr_relaxation.relaxation import fit_relaxation

T = np.array([0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064,
              0.128, 0.256, 0.512, 1.024, 2.048, 4.096, 8.192, 16.384])


def _noise(n, seed=0, s=0.01):
    return np.random.default_rng(seed).normal(0, s, n)


def test_inversion_recovery_roundtrip():
    y = 1.0 * (1 - 2 * np.exp(-T / 1.4)) + _noise(T.size)
    r = fit_relaxation(T, y, model="inversion_recovery")
    assert r["parameters"]["T1_s"] == pytest.approx(1.4, abs=0.15)
    assert r["fit_quality"]["r_squared"] > 0.99


def test_saturation_recovery_roundtrip():
    y = 1.0 * (1 - np.exp(-T / 0.8)) + _noise(T.size)
    r = fit_relaxation(T, y, model="saturation_recovery")
    assert r["parameters"]["T1_s"] == pytest.approx(0.8, abs=0.1)
    assert r["fit_quality"]["r_squared"] > 0.99


def test_t2_decay_roundtrip():
    y = 1.0 * np.exp(-T / 2.0) + _noise(T.size)
    r = fit_relaxation(T, y, model="t2_decay")
    assert r["parameters"]["T2_s"] == pytest.approx(2.0, abs=0.2)
    assert "T1_s" not in r["parameters"]


def test_stretched_recovers_beta():
    y = 1.0 * (1 - np.exp(-(T / 0.5) ** 0.6)) + _noise(T.size, s=0.005)
    r = fit_relaxation(T, y, model="saturation_recovery", stretched=True)
    assert r["parameters"]["beta"] == pytest.approx(0.6, abs=0.12)
    assert r["parameters"]["T1_s"] == pytest.approx(0.5, abs=0.1)


def test_mono_pins_beta_to_one():
    y = 1.0 * (1 - np.exp(-T / 0.8)) + _noise(T.size)
    r = fit_relaxation(T, y, model="saturation_recovery", stretched=False)
    assert r["parameters"]["beta"] == pytest.approx(1.0, abs=1e-3)


def test_two_component_returns_populations():
    y = 0.6 * (1 - np.exp(-T / 0.2)) + 0.4 * (1 - np.exp(-T / 5.0)) + _noise(T.size, s=0.005)
    r = fit_relaxation(T, y, model="saturation_recovery", n_components=2)
    assert "population_a" in r["parameters"] and "population_b" in r["parameters"]
    assert r["parameters"]["population_a"] + r["parameters"]["population_b"] == pytest.approx(1.0, abs=1e-6)


def test_bad_model_raises():
    with pytest.raises(ValueError):
        fit_relaxation(T, T, model="nonsense")


def test_long_t1_beyond_window_is_flagged_not_zero():
    # T1 (30 s) far longer than the sampled window (8 s): the recovery barely
    # starts. The fit must NOT collapse to T1=0; it returns a large value and
    # flags it unconstrained (general robustness, not tuned to any dataset).
    t = np.geomspace(0.01, 8.0, 14)
    y = (1 - 2 * np.exp(-t / 30.0)) + _noise(t.size)
    r = fit_relaxation(t, y, model="inversion_recovery")
    assert r["parameters"]["T1_s"] > 8.0
    assert any("unconstrained" in f for f in r["flags"])


def test_tool_registered_and_gated():
    from scilink.skills._shared._registry import get_tools_for, get_tool_function
    names = {t.name for t in get_tools_for("curve_fitting", active_skills=["nmr_relaxation"])}
    assert "fit_relaxation" in names
    assert "fit_relaxation" not in {t.name for t in get_tools_for("curve_fitting", active_skills=["nmr"])}
    assert callable(get_tool_function("fit_relaxation", active_skills=["nmr_relaxation"]))
