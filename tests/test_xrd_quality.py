"""Offline tests for the XRD peak-region quality metric and its wiring.

The metric is the shared 1D-spectroscopy ``peak_region_r2`` (hoisted to
``_shared``). For XRD it must (a) RESCUE a correct fit of a weak/noisy powder
pattern that a whole-pattern R² scores low, (b) still FAIL a fit that genuinely
misses reflections, (c) leave a high-SNR fit unchanged (no regression), and be
correctly skill-gated to the ``xrd_profile`` skill. ``fit_pattern`` must report
it. The NMR skill must keep re-exporting the same implementation.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
import yaml
from pathlib import Path

from scilink.skills._shared._quality_metrics import peak_region_r2

N = 8000
X = np.linspace(10.0, 80.0, N)  # 2-theta degrees


def _pv(A, x0, w):
    return A * (w / 2) ** 2 / ((X - x0) ** 2 + (w / 2) ** 2)


def _pattern(amp, seed=0):
    y = np.zeros(N)
    for c in (22.0, 28.0, 35.0, 48.0, 61.0):
        y += _pv(amp, c, 0.3)
    return y + np.random.default_rng(seed).normal(0, 1.0, N)


def _model(amp):
    y = np.zeros(N)
    for c in (22.0, 28.0, 35.0, 48.0, 61.0):
        y += _pv(amp, c, 0.3)
    return y


def test_rescues_correct_low_snr_pattern():
    # Weak peaks (5σ) on a noisy background: global R² low, region R² rescues.
    y = _pattern(amp=5)
    q = peak_region_r2(X, y, _model(5), baseline=np.zeros(N))
    assert q["r_squared"] < q["peak_region_r2"]
    assert q["peak_region_r2"] > 0.3


def test_high_snr_pattern_unchanged():
    # Strong peaks: region R² ≈ global R², both high → no regression on clean data.
    y = _pattern(amp=200)
    q = peak_region_r2(X, y, _model(200), baseline=np.zeros(N))
    assert q["peak_region_r2"] > 0.9
    assert abs(q["peak_region_r2"] - q["r_squared"]) < 0.05


def test_missed_reflections_not_rescued():
    # Model omits two of the five reflections: region R² must drop (genuine miss).
    y = _pattern(amp=40)
    partial = _pv(40, 22.0, 0.3) + _pv(40, 28.0, 0.3) + _pv(40, 35.0, 0.3)
    q = peak_region_r2(X, y, partial, baseline=np.zeros(N))
    full = peak_region_r2(X, y, _model(40), baseline=np.zeros(N))
    assert q["peak_region_r2"] < full["peak_region_r2"] - 0.2


def test_reexport_identity():
    # NMR and XRD bundles must re-export the SAME shared implementation.
    from scilink.skills.curve_fitting.nmr.quality import peak_region_r2 as nmr
    from scilink.skills.curve_fitting.xrd_profile.quality import peak_region_r2 as xrd
    assert nmr is xrd is peak_region_r2


def test_tool_registered_and_skill_gated():
    from scilink.skills._shared._registry import get_tools_for, get_tool_function
    names = {t.name for t in get_tools_for("curve_fitting", active_skills=["xrd_profile"])}
    assert "peak_region_r2" in names
    assert "peak_region_r2" not in {
        t.name for t in get_tools_for("curve_fitting", active_skills=[])}
    assert callable(get_tool_function("peak_region_r2", active_skills=["xrd_profile"]))


def test_fit_pattern_reports_peak_region_r2():
    from scilink.skills.curve_fitting.xrd_profile.fit_pattern import fit_pattern
    y = _pattern(amp=200)
    res = fit_pattern(X.tolist(), y.tolist())
    assert "peak_region_r2" in res and "n_signal_points" in res
    assert 0.0 <= res["peak_region_r2"] <= 1.0


def test_max_abs_residual_flags_localized_misfit():
    """The dynamic-range diagnostic: max_abs_residual_over_noise is the worst
    LOCAL misfit, the complement of the averaged R²/peak_region_r2/RMS. A clean,
    modest-dynamic-range fit is a few sigma; a localized misfit makes it spike far
    above the (averaged) residual_rms_over_noise — which is exactly why the
    averaged metrics hide it. It is read AGAINST the RMS (localized-ness), not as
    an absolute threshold: its scale rises with dynamic range (a tiny fractional
    residual on a giant peak is many sigma), so it directs attention rather than
    issuing a verdict."""
    from scilink.skills.curve_fitting.xrd_profile.fit_pattern import fit_pattern

    def make(amp, noise_frac, seed, apex_width=0.15):
        x = np.linspace(10.0, 70.0, 2400)
        y = np.full_like(x, 200.0)
        # last peak's width is tunable: a sub-resolution apex_width makes a tall
        # peak that IS detected (so peak_region_r2 stays high) but is too sharp
        # for the model floor to fit -> a localized apex misfit.
        # the apex peak is DOMINANT (high dynamic range, like a real super-peak):
        # fittable at the default width (clean -> low max), sub-resolution sharp
        # at apex_width=0.012 (misfit the smooth model cannot capture -> high max).
        for c, f, w in [(20, 1.0, 0.15), (30, 0.6, 0.15), (45, 0.4, 0.15), (58, 4.0, apex_width)]:
            s = w / 2.355
            g = np.exp(-((x - c) ** 2) / (2 * s ** 2))
            l = (w / 2) ** 2 / ((x - c) ** 2 + (w / 2) ** 2)
            y += amp * f * (0.5 * l + 0.5 * g)
        y += np.random.default_rng(seed).normal(0, noise_frac * amp, y.shape)
        return x, y

    # clean fits: low max-residual at BOTH high and low SNR (not just tracking noise)
    clean_vals = []
    for amp in (2e5, 5e3):
        x, y = make(amp, 0.01, 0)
        r = fit_pattern(x.tolist(), y.tolist())
        assert "max_abs_residual_over_noise" in r
        assert r["max_abs_residual_over_noise"] < 6.0, amp        # clean ~ few sigma
        assert r["peak_region_r2"] > 0.95
        clean_vals.append(r["max_abs_residual_over_noise"])

    # A tall, sub-resolution-sharp apex the smooth model cannot capture -> a
    # localized misfit. The whole point: it is HUGE in max_abs_residual_over_noise
    # but small in the AVERAGED residual_rms_over_noise -- which is exactly why the
    # averaged metrics (R²/peak_region_r2/RMS) hide it. (On a many-peak pattern the
    # few misfit points also leave peak_region_r2 high; validated on real IBD data.)
    x, y = make(2e5, 0.005, 0, apex_width=0.012)                  # << model FWHM floor
    r = fit_pattern(x.tolist(), y.tolist())
    assert r["max_abs_residual_over_noise"] > 8.0 * max(clean_vals)   # loudly flagged
    assert r["max_abs_residual_over_noise"] > 5.0 * r["residual_rms_over_noise"]  # averaging hides it


def test_fit_curve_raw_overlays_raw_data():
    """fit_pattern returns fit_curve_raw (= fit_curve + background) — the model on
    the RAW-intensity scale. Saving THIS (not fit_curve, which is background-
    subtracted) is what makes a good fit overlay the raw data instead of looking
    collapsed when the background is large."""
    from scilink.skills.curve_fitting.xrd_profile.fit_pattern import fit_pattern
    x = np.linspace(10.0, 60.0, 3000)
    bg = 50000.0 + 1e5 * np.exp(-(x - 10.0) * 0.3)        # large, sloping background
    y = bg.copy()
    for c, a in [(20.0, 8e5), (30.0, 4e5), (45.0, 2e5)]:
        y += a * (0.1 ** 2) / ((x - c) ** 2 + 0.1 ** 2)
    r = fit_pattern(x.tolist(), y.tolist())
    assert "fit_curve_raw" in r
    fcr = np.asarray(r["fit_curve_raw"]); fc = np.asarray(r["fit_curve"])
    assert fcr.shape == x.shape
    # fit_curve_raw overlays the raw data (peaks at full raw height); fit_curve does NOT
    i = int(np.argmin(np.abs(x - 20.0)))
    assert 0.9 < fcr[i] / y[i] < 1.1                      # raw-scale overlay matches data
    assert fc[i] / y[i] < 0.7                             # bg-subtracted is well below raw
    # in the gaps the raw-scale model sits at the background, not at ~0
    g = int(np.argmin(np.abs(x - 15.0)))
    assert fcr[g] > 0.5 * bg[g]


def test_fit_range_contract():
    """fit_range restricts detection/background/fit to a window, finds the in-window
    peaks, and returns FULL-length arrays with the excluded region at zero residual
    (so the verifier's length-N residual diagnostics still run). The real-data
    recovery (low-angle upturn: 1 -> 20 peaks on the RRUFF/IBD series) is validated
    in the benchmark; SNIP's exact overshoot is not reliably reproducible in a unit
    synthetic, so here we assert the mechanical contract."""
    from scilink.skills.curve_fitting.xrd_profile.fit_pattern import fit_pattern
    x = np.linspace(2.0, 45.0, 4300)
    y = np.full_like(x, 500.0)
    for c, a in [(11.0, 8000), (17.0, 3000), (22.0, 9000), (29.0, 2000), (36.0, 1500)]:
        y += a * (0.05 ** 2) / ((x - c) ** 2 + 0.05 ** 2)
    y += 5e5 * np.exp(-(x - 2.0) * 3.0)   # non-Bragg low-angle upturn below ~4 deg

    ranged = fit_pattern(x.tolist(), y.tolist(), fit_range=(5.0, 45.0))
    assert ranged["n_peaks"] >= 5                       # finds the 5 in-window peaks
    # full-length arrays preserved (residual-diagnostic length contract)
    assert len(ranged["fit_curve"]) == x.size
    assert len(ranged["intensity_corrected"]) == x.size
    # excluded region is matched -> zero residual on raw-scale reconstruction
    fc = np.asarray(ranged["fit_curve"]); ic = np.asarray(ranged["intensity_corrected"])
    fit_raw = fc + (y - ic)
    excl = x < 5.0
    assert np.max(np.abs(y[excl] - fit_raw[excl])) < 1e-6
    assert ranged["fit_range"] == [5.0, 45.0]
    # scores reflect the window only and the in-window fit is good
    assert ranged["peak_region_r2"] > 0.9


def test_fit_range_none_is_default_behavior():
    """fit_range=None must be byte-identical to omitting it (no regression)."""
    from scilink.skills.curve_fitting.xrd_profile.fit_pattern import fit_pattern
    x = np.linspace(10.0, 60.0, 2500)
    y = np.full_like(x, 100.0)
    for c in (20.0, 30.0, 45.0):
        y += 2000 * (0.1 ** 2) / ((x - c) ** 2 + 0.1 ** 2)
    a = fit_pattern(x.tolist(), y.tolist())
    b = fit_pattern(x.tolist(), y.tolist(), fit_range=None)
    assert a["n_peaks"] == b["n_peaks"]
    assert a["r_squared"] == pytest.approx(b["r_squared"])
    assert "fit_range" not in a  # only present when a window was applied


def test_frontmatter_gate_is_peak_region_r2():
    md = Path("scilink/skills/curve_fitting/xrd_profile/xrd_profile.md").read_text()
    fm = yaml.safe_load(md.split("---")[1])
    gate = fm["quality_gate"]
    assert gate["metric"] == "peak_region_r2"
    from scilink.agents.exp_agents.quality_gate import _coerce
    g = _coerce(gate)
    assert g.metric == "peak_region_r2"
    assert g.hard_reject_threshold <= g.accept_threshold
    # the gate reads the metric out of a fit_quality dict
    assert g.extract({"peak_region_r2": 0.89, "r_squared": 0.55}) == pytest.approx(0.89)
