"""Offline tests for the residual-driven multi-Voigt fitter.

The fitter must let the DATA choose the peak count: recover the right number of
overlapping components (including a sharp+broad pair) WITHOUT over-fitting noise.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import wofz

from scilink.skills.curve_fitting.nmr.multipeak import fit_multipeak_voigt

N = 8000
X = np.linspace(-100.0, 100.0, N)


def _vh(c, s, g):
    z = ((X - c) + 1j * g) / (s * np.sqrt(2))
    z0 = (1j * g) / (s * np.sqrt(2))
    return np.real(wofz(z)) / np.real(wofz(z0))


def _noise(seed=0, sigma=0.1):
    return np.random.default_rng(seed).normal(0, sigma, N)


@pytest.mark.parametrize("signal,want", [
    (10 * _vh(0, 2, 1), 1),                                   # single
    (10 * _vh(-3, 1.5, 0.5) + 6 * _vh(2, 5, 2), 2),           # sharp + broad, offset
    (10 * _vh(-2, 1, 0.5) + 8 * _vh(3, 1, 0.5), 2),           # two close sharp
    (8 * _vh(-20, 2, 1) + 10 * _vh(0, 2, 1) + 5 * _vh(18, 2, 1), 3),  # three
])
def test_recovers_correct_peak_count(signal, want):
    y = signal + _noise()
    r = fit_multipeak_voigt(X, y)
    assert r["n_peaks"] == want
    assert r["fit_quality"]["peak_region_r2"] > 0.95


def test_does_not_overfit_noise():
    # A single peak buried in heavier noise stays one peak; pure noise stays one.
    assert fit_multipeak_voigt(X, 10 * _vh(0, 2, 1) + _noise(sigma=0.3))["n_peaks"] == 1
    assert fit_multipeak_voigt(X, _noise(seed=1, sigma=1.0))["n_peaks"] == 1


def test_improve_thresh_controls_parsimony():
    # A faint shoulder: the default may keep it parsimonious, but lowering the
    # improvement threshold (an LLM-tunable knob) must never INCREASE under a
    # stricter threshold. Monotonic in the parsimony direction.
    y = 10 * _vh(-3, 1.5, 0.5) + 6 * _vh(2, 5, 2) + _noise()
    strict = fit_multipeak_voigt(X, y, improve_thresh=0.20)["n_peaks"]
    loose = fit_multipeak_voigt(X, y, improve_thresh=0.005)["n_peaks"]
    assert loose >= strict


def test_max_peaks_caps_count():
    y = sum(_vh(c, 2, 1) for c in (-40, -20, 0, 20, 40)) + _noise()
    assert fit_multipeak_voigt(X, y, max_peaks=2)["n_peaks"] <= 2


def test_tool_registered_and_skill_gated():
    from scilink.skills._shared._registry import get_tools_for, get_tool_function
    names = {t.name for t in get_tools_for("curve_fitting", active_skills=["nmr"])}
    assert "fit_multipeak_voigt" in names
    assert "fit_multipeak_voigt" not in {
        t.name for t in get_tools_for("curve_fitting", active_skills=[])}
    assert callable(get_tool_function("fit_multipeak_voigt", active_skills=["nmr"]))
