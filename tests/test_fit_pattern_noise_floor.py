"""fit_pattern peak detection: noise-referenced prominence floor.

A fixed 3-sigma floor admitted hundreds of noise 'peaks' on any pattern
longer than a few hundred points (30 on a 700-point single-peak pattern,
>200 on an 8000-point one), which then propped up bad baselines and
crashed/derailed the global fit. The floor is now sample-size aware
(~2*sqrt(2 ln N) sigma) and exposed as min_prominence_sigma.
"""
import numpy as np

from scilink.skills.curve_fitting.xrd_profile.fit_pattern import (
    fit_pattern, noise_prominence_floor,
)


def _single_peak(n=700, step=0.1431, seed=5):
    rng = np.random.default_rng(seed)
    x = np.arange(n) * step
    y = 2.2 * np.exp(-0.5 * ((x - 31.0) / 1.6) ** 2) + 0.04 * rng.normal(size=n)
    return x, y


def _five_peaks(n=8000, amp=10.0, seed=0):
    x = np.linspace(10, 80, n)
    y = np.zeros(n)
    for c in (22.0, 28.0, 35.0, 48.0, 61.0):
        y += amp * 0.15 ** 2 / ((x - c) ** 2 + 0.15 ** 2)
    return x, y + np.random.default_rng(seed).normal(0, 1.0, n)


def test_floor_is_sample_size_aware_and_bounded_below():
    assert noise_prominence_floor(10) == 5.0
    assert 7.0 < noise_prominence_floor(700) < 7.5
    assert 8.0 < noise_prominence_floor(8000) < 9.0
    assert noise_prominence_floor(8000) > noise_prominence_floor(700)


def test_single_broad_peak_no_spurious_peaks():
    x, y = _single_peak()
    r = fit_pattern(x.tolist(), y.tolist(), max_fwhm_deg=8.0)
    assert r["n_peaks"] == 1 and abs(r["peak_centers"][0] - 31.0) < 0.3
    assert r["r_squared"] > 0.95
    assert r["noise_prominence_sigma"] == noise_prominence_floor(700)


def test_weak_real_reflections_all_found_with_at_most_one_extra():
    # 10-sigma reflections; matched-filter detection must find all five and
    # admit at most one noise excursion (was 30 = max_peaks with the 3-sigma floor).
    x, y = _five_peaks(amp=10.0)
    r = fit_pattern(x.tolist(), y.tolist())
    truth = (22, 28, 35, 48, 61)
    found = [t for t in truth if any(abs(c - t) < 0.3 for c in r["peak_centers"])]
    assert len(found) == 5
    assert r["n_peaks"] <= 6


def test_six_sigma_reflections_recovered_by_matched_filter():
    # Too weak for a raw-trace floor; the width-matched smoothing recovers them.
    x, y = _five_peaks(amp=6.0)
    r = fit_pattern(x.tolist(), y.tolist())
    truth = (22, 28, 35, 48, 61)
    found = [t for t in truth if any(abs(c - t) < 0.3 for c in r["peak_centers"])]
    assert len(found) >= 4
    assert r["n_peaks"] <= 8


def test_knob_lowers_floor_and_is_reported():
    x, y = _single_peak()
    r = fit_pattern(x.tolist(), y.tolist(), max_fwhm_deg=8.0, min_prominence_sigma=3.0)
    assert r["noise_prominence_sigma"] == 3.0
    assert r["n_peaks"] > 5          # the old behaviour: noise peaks admitted
