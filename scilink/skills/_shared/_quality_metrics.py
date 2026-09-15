"""Peak-region quality metric for 1D-spectroscopy fits (NMR, XRD, …).

A 1D spectrum/diffractogram often has an enormous digitized width relative to
the features that carry the science, so a *correct* fit can be dominated, in a
whole-window R², by the noise-filled regions between peaks — a good fit then
scores a misleadingly low R² (see the low-SNR ²³Na/⁶⁷Zn NMR references, or a
weak, noisy powder-XRD pattern whose Bragg peaks are nonetheless well fit). The
gate metric should instead measure how well the model reproduces the spectrum
**where there is signal**.

``peak_region_r2`` computes R² over the signal region only — the union of the
points where the *data* rises above the noise (so a missed real peak is still
penalized) and the points where the *fitted model* places intensity (so the
metric tracks the claimed peaks). It falls back to the global R² when too few
signal points are found, and is reported alongside the global R² rather than
replacing it. This is a general 1D metric, not tuned to any sample or modality:
the signal region is detected from the data/fit at run time.

This is the shared implementation. Per-skill bundles re-export ``peak_region_r2``
from here and declare their own (modality-flavoured) ``TOOL_SPEC`` so the tool is
skill-gated — visible to the LLM only when that skill is active.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


def _robust_noise(resid: np.ndarray) -> float:
    """MAD-based noise estimate; robust because signal occupies a small fraction
    of the wide window."""
    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    return float(1.4826 * mad) or float(np.std(resid)) or 1.0


def _contiguous_runs(mask: np.ndarray, min_run: int) -> np.ndarray:
    """Keep only contiguous True runs of length >= ``min_run`` (drops isolated
    noise spikes; real peaks are contiguous)."""
    out = np.zeros_like(mask)
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_run:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def peak_region_r2(
    x: Sequence[float],
    y: Sequence[float],
    y_fit: Sequence[float],
    baseline: Optional[Sequence[float]] = None,
    k_sigma: float = 3.0,
    min_run: int = 3,
    dilate: int = 5,
    model_frac: float = 0.02,
    min_points: int = 15,
    struct_apex_frac: float = 0.15,
    struct_autocorr: float = 0.9,
) -> dict[str, Any]:
    """R² over the signal region only (see module docstring).

    The signal region is ``(|y - baseline| > k_sigma·noise``, kept as contiguous
    runs and dilated) ``OR (|y_fit - baseline| > model_frac·max)``. Returns a
    dict with ``peak_region_r2``, the global ``r_squared``, ``n_signal_points``,
    and ``fell_back_to_global`` (True when the signal region was too small to be
    meaningful). ``baseline`` defaults to the median of ``y`` (pass an array of
    zeros when ``y`` is already background-subtracted, as for a corrected XRD
    pattern).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    y_fit = np.asarray(y_fit, float)
    base = (np.asarray(baseline, float) if baseline is not None
            else np.full_like(y, np.median(y)))

    resid_base = y - base
    noise = _robust_noise(resid_base)
    data_sig = _contiguous_runs(np.abs(resid_base) > k_sigma * noise, min_run)
    if dilate > 0 and data_sig.any():
        data_sig = np.convolve(data_sig, np.ones(2 * dilate + 1), mode="same") > 0
    # Where the model places real intensity — thresholded at the SAME noise
    # floor, so a (near-)flat model contributes nothing (no spurious all-True
    # mask) while a hallucinated peak where the data is empty is still included
    # and therefore penalized. ``model_frac`` only raises the floor for tall
    # models so sub-percent model wings don't dominate the region.
    model_dev = np.abs(y_fit - base)
    model_floor = max(k_sigma * noise, model_frac * (model_dev.max() or 0.0))
    model_sig = model_dev > model_floor
    sig = data_sig | model_sig

    def _r2(mask):
        yr, yfr = y[mask], y_fit[mask]
        ss_res = float(np.sum((yr - yfr) ** 2))
        ss_tot = float(np.sum((yr - np.mean(yr)) ** 2)) or 1e-30
        return 1.0 - ss_res / ss_tot

    global_r2 = _r2(np.ones_like(y, dtype=bool))
    fell_back = int(sig.sum()) < min_points
    region_r2 = global_r2 if fell_back else _r2(sig)

    # --- residual-structure diagnostics (over the signal region) -------------
    # A high R² can still hide a fit that misses where the signal actually is:
    # on a tall, finely-sampled peak the variance is dominated by the apex, so a
    # systematic apex/shoulder miss barely moves R² yet leaves large, correlated
    # residuals. These quantities expose that — judge a fit by them, not R² alone.
    rmask = sig if not fell_back else np.ones_like(y, dtype=bool)
    resid = (y - y_fit)[rmask]
    peak_height = float(np.max(np.abs(y[rmask] - base[rmask]))) or 1.0
    if resid.size >= 4 and np.std(resid) > 0:
        apex_over_noise = float(np.max(np.abs(resid)) / noise)
        apex_frac = float(np.max(np.abs(resid)) / peak_height)
        autocorr = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
        frac_gt_3 = float(np.mean(np.abs(resid) > 3.0 * noise))
        # SNR-ROBUST overall-misfit size: RMS residual as a fraction of the peak
        # height. Unlike apex_resid_over_noise / frac_resid_gt_3sigma (both
        # noise-relative, so they are large even for a PERFECT fit of a very
        # high-SNR peak), this is normalized to the signal, so it measures how
        # far the fit is from the data in physically meaningful (fraction-of-
        # peak) terms.
        resid_rms_frac = float(np.sqrt(np.mean(resid ** 2)) / peak_height)
    else:
        apex_over_noise = apex_frac = autocorr = frac_gt_3 = resid_rms_frac = 0.0
    # "Structured" = the residual is SYSTEMATIC (high lag-1 autocorrelation, i.e.
    # not random noise) AND SEVERE at the apex (a large miss relative to the peak
    # height — judging on the fraction, not σ, so ultra-high-SNR peaks aren't
    # false-flagged and a localized cusp miss is still caught). frac_resid_gt_3sigma
    # is returned for context but deliberately NOT an AND-gate. Thresholds are
    # tunable knobs; raw values always returned.
    structured = bool(autocorr > struct_autocorr and apex_frac > struct_apex_frac)

    return {
        "peak_region_r2": float(region_r2),
        "r_squared": float(global_r2),
        "n_signal_points": int(sig.sum()),
        "fell_back_to_global": bool(fell_back),
        # Residual-structure diagnostics (added; do not replace peak_region_r2):
        "apex_resid_over_noise": apex_over_noise,
        "apex_resid_frac": apex_frac,
        "resid_autocorr_lag1": autocorr,
        "frac_resid_gt_3sigma": frac_gt_3,
        "resid_rms_frac": resid_rms_frac,
        "residual_structured": structured,
    }
