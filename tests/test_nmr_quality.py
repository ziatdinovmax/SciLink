"""Offline tests for the NMR peak-region quality metric.

The metric must (a) RESCUE a correct fit of a narrow low-SNR peak that a
whole-window R² scores near zero, and (b) still FAIL genuinely bad fits
(missed peak, wrong shape, incomplete multi-site) — i.e. it is not a blanket
pass. The signal region is detected from the data/model at run time, so the
metric is general, not tuned to any spectrum.
"""

from __future__ import annotations

import numpy as np
import pytest

from scilink.skills.curve_fitting.nmr.quality import peak_region_r2

N = 20000
X = np.linspace(-1000.0, 1000.0, N)


def _lor(A, x0, w):
    return A * (w / 2) ** 2 / ((X - x0) ** 2 + (w / 2) ** 2)


def _noise(seed=0):
    return np.random.default_rng(seed).normal(0, 1.0, N)


def test_rescues_correct_low_snr_fit():
    # 5σ peak in a wide noisy window: global R² ≈ 0, region R² clearly positive.
    y = _lor(5, 0, 3) + _noise()
    q = peak_region_r2(X, y, _lor(5, 0, 3))
    assert q["r_squared"] < 0.2            # whole-window metric is misleading
    assert q["peak_region_r2"] > 0.3       # peak-region metric rescues it
    assert q["peak_region_r2"] > q["r_squared"]


def test_high_snr_fit_scores_high():
    y = _lor(100, 0, 3) + _noise()
    q = peak_region_r2(X, y, _lor(100, 0, 3))
    assert q["peak_region_r2"] > 0.9


@pytest.mark.parametrize("y_fit_desc,y_fit", [
    ("missed_peak", np.zeros(N)),
    ("wrong_width", _lor(2.5, 0, 30)),
])
def test_bad_fits_fail(y_fit_desc, y_fit):
    # Genuinely wrong fits must score low (negative here) — not rescued.
    y = _lor(5, 0, 3) + _noise()
    q = peak_region_r2(X, y, y_fit)
    assert q["peak_region_r2"] < 0.3, y_fit_desc


def test_incomplete_multisite_fails():
    # Two real sites, model fits only one broad peak: region R² must drop.
    y = _lor(5, -2, 3) + _lor(4, 2, 4) + _noise()
    q = peak_region_r2(X, y, _lor(5, 0, 8))
    assert q["peak_region_r2"] < 0.3
    # the correct two-site model scores far better on the same data
    q_ok = peak_region_r2(X, y, _lor(5, -2, 3) + _lor(4, 2, 4))
    assert q_ok["peak_region_r2"] > q["peak_region_r2"] + 0.3


def test_pure_noise_falls_back_to_global():
    # No real signal: too few signal points → fall back to the global R².
    y = _noise()
    q = peak_region_r2(X, y, np.zeros(N))
    assert q["fell_back_to_global"] is True
    assert q["peak_region_r2"] == pytest.approx(q["r_squared"])


def test_tool_registered_and_skill_gated():
    from scilink.skills._shared._registry import get_tools_for, get_tool_function
    names = {t.name for t in get_tools_for("curve_fitting", active_skills=["nmr"])}
    assert "peak_region_r2" in names
    assert "peak_region_r2" not in {
        t.name for t in get_tools_for("curve_fitting", active_skills=[])}
    assert callable(get_tool_function("peak_region_r2", active_skills=["nmr"]))
