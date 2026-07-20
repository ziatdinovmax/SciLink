"""Detection-significance guard for low-SNR 1D NMR fits.

The skill rightly tells the agent to *recover* weak, broad resonances (common
for low-γ quadrupolar nuclei like ⁶⁷Zn, ²⁵Mg, ³³S near their detection limit)
rather than declaring "no signal". But that guidance has no counterweight: on a
spectrum that is essentially noise, a flexible Voigt+baseline model will still
report a confident-looking fit (a high peak-region R²) by fitting the noise —
overstating a measurement the data cannot support.

``assess_detection`` supplies the missing test. It estimates the noise robustly
from the signal-free wings, measures the peak SNR, and — when given the fitted
model — runs an F-test of "peak present" vs. "baseline only" (extra-sum-of-
squares). It returns a graded verdict:

  * ``"detected"``  — SNR above the Rose criterion AND the peak model is
                      statistically justified → fit and report normally.
  * ``"marginal"``  — some excess but not significant → report the shift/width
                      with a wide uncertainty or as an upper bound; do not claim
                      a precise lineshape.
  * ``"absent"``    — indistinguishable from noise → report a non-detection
                      (intensity upper bound), do NOT report a fitted lineshape.

The thresholds are the standard ones (Rose-criterion SNR, an F-test α), not
tuned to any spectrum; the noise and SNR are measured from the data at run time.
This is a *diagnostic* — it never edits the fit, it tells the agent and the
verifier whether the fit is supportable.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from scipy import stats


def _robust_noise(resid: np.ndarray) -> float:
    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    return float(1.4826 * mad) or float(np.std(resid)) or 1.0


def _has_contiguous_feature(yb: np.ndarray, noise: float, k: float, min_run: int) -> tuple[bool, float]:
    """Is there a *resolved* feature — ``min_run`` contiguous points above
    ``k·noise`` — and if so what is its peak |amplitude|?

    A single-point max/σ is the wrong detection test on a long spectrum: the
    maximum of N noise samples is ~√(2·lnN)·σ (≈4σ for a few thousand points),
    so a per-point threshold flags pure noise. A genuine resonance instead spans
    several contiguous points; requiring a contiguous run removes that
    extreme-value false alarm while still catching a real broad line.
    """
    mask = np.abs(yb) > k * noise
    best = 0.0
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_run:
                best = max(best, float(np.max(np.abs(yb[i:j]))))
            i = j
        else:
            i += 1
    return best > 0.0, best


def assess_detection(
    x: Sequence[float],
    y: Sequence[float],
    y_fit: Optional[Sequence[float]] = None,
    baseline: Optional[Sequence[float]] = None,
    n_model_params: Optional[int] = None,
    snr_detect: float = 3.0,
    snr_quantify: float = 10.0,
    alpha: float = 0.01,
) -> dict[str, Any]:
    """Grade whether an NMR resonance is real enough to fit/quantify.

    Estimates noise from the signal-free region, computes the peak SNR, and (if
    ``y_fit`` and ``n_model_params`` are given) F-tests the peak model against a
    baseline-only model. Returns ``verdict`` ∈ {detected, marginal, absent}, the
    measured ``snr``, the F-test ``p_value``, an intensity ``upper_bound`` for a
    non-detection, and a ``recommendation`` string.

    ``snr_detect`` (Rose criterion, default 3) is the floor for a real feature;
    ``snr_quantify`` (default 10, the textbook quantitative-NMR SNR floor) the
    floor for a quantitative lineshape; ``alpha``
    the F-test significance. Defaults are standard, not sample-specific.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    base = (np.asarray(baseline, float) if baseline is not None
            else np.full_like(y, np.median(y)))
    yb = y - base

    # Noise from points the data itself calls signal-free (robust to the peak).
    coarse = np.abs(yb) <= 3.0 * _robust_noise(yb)
    noise = _robust_noise(yb[coarse]) if coarse.sum() > 50 else _robust_noise(yb)
    noise = noise or 1.0
    point_amp = float(np.max(np.abs(yb)))           # single-point max (reported)
    has_feature, feat_amp = _has_contiguous_feature(yb, noise, snr_detect, min_run=3)
    # SNR of the resolved feature drives the verdict; fall back to the point max
    # only for reporting when nothing is resolved.
    peak_amp = feat_amp if has_feature else point_amp
    snr = peak_amp / noise

    # F-test: does the peak model explain variance beyond a flat baseline?
    p_value = None
    f_stat = None
    if y_fit is not None and n_model_params:
        y_fit = np.asarray(y_fit, float)
        n = y.size
        ss_full = float(np.sum((y - y_fit) ** 2))
        ss_base = float(np.sum((y - base) ** 2))
        p_full = int(n_model_params)
        p_base = 1  # baseline-only (constant/known background)
        df1 = max(p_full - p_base, 1)
        df2 = max(n - p_full, 1)
        if ss_full > 0 and ss_base > ss_full:
            f_stat = ((ss_base - ss_full) / df1) / (ss_full / df2)
            p_value = float(stats.f.sf(f_stat, df1, df2))
        else:
            p_value = 1.0

    significant = (p_value is not None) and (p_value < alpha)
    # A resonance is real if it is RESOLVED (a contiguous feature) or if the
    # peak model is statistically justified over a flat baseline (F-test) — the
    # latter rescues a broad, shallow line that never clears the per-point
    # threshold. Pure noise satisfies neither. The F-test, when run, can also
    # downgrade an apparent feature the model does not actually justify.
    real = has_feature or significant
    if not real:
        verdict = "absent"
        rec = ("No resolved feature (no run of points above the noise) and the "
               "fit is not justified over a flat baseline. Report a "
               "NON-DETECTION with an intensity upper bound; do NOT report a "
               "fitted peak — a high peak-region R² here is overfitting the "
               "noise. Recommend more scans / longer acquisition.")
    elif has_feature and (p_value is not None) and not significant:
        verdict = "marginal"
        rec = ("Apparent peak is NOT statistically justified over a flat "
               "baseline (F-test). Report the shift as tentative with a wide "
               "uncertainty / as an upper bound; do not claim a precise "
               "lineshape, linewidth, or quantitative integral.")
    elif snr >= snr_quantify:
        verdict = "detected"
        rec = ("Signal well above noise — fit and report the lineshape "
               "(shift, width) normally.")
    else:
        verdict = "detected"
        rec = ("Weak but real resonance (above the Rose criterion / F-test "
               "justified, but below the quantitative-SNR floor). Recover it; "
               "report shift and width WITH explicit uncertainty, and flag that "
               "quantification is SNR-limited.")

    return {
        "verdict": verdict,
        "snr": float(snr),
        "has_resolved_feature": bool(has_feature),
        "noise": float(noise),
        "peak_amplitude": peak_amp,
        "f_stat": (float(f_stat) if f_stat is not None else None),
        "p_value": p_value,
        "significant": bool(significant) if p_value is not None else None,
        "upper_bound": float(snr_detect * noise),
        "snr_detect": float(snr_detect),
        "snr_quantify": float(snr_quantify),
        "recommendation": rec,
    }


from ..._shared._spec import ToolSpec  # noqa: E402  (kept next to the spec below)

TOOL_SPEC = ToolSpec(
    name="assess_detection",
    description=(
        "Grade whether an NMR resonance is real enough to fit and quantify, so a "
        "near-noise spectrum is not 'fit' into a confident-looking lineshape. "
        "Estimates noise from the signal-free region, computes the peak SNR, and "
        "(given the fit) F-tests the peak model against a baseline-only model. "
        "Returns a verdict — detected / marginal / absent — with an SNR, p-value, "
        "intensity upper bound, and a recommended reporting action. The "
        "counterweight to 'recover weak lines': recover a real broad resonance, "
        "but abstain (report a non-detection / upper bound) when the data cannot "
        "support a fit."
    ),
    import_line="from scilink.skills.curve_fitting.nmr.detection import assess_detection",
    signature=(
        "assess_detection(x, y, y_fit=None, baseline=None, n_model_params=None, "
        "snr_detect=3.0, snr_quantify=10.0, alpha=0.01) -> dict"
    ),
    parameters={
        "x": {"type": "list[float]", "description": "Chemical-shift axis (ppm)."},
        "y": {"type": "list[float]", "description": "Spectrum intensity."},
        "y_fit": {"type": "list[float]", "description": "Fitted model on x (optional). Provide it (with n_model_params) to enable the F-test of peak-vs-baseline; omit for an SNR-only screen before fitting."},
        "baseline": {"type": "list[float]", "description": "Fitted baseline on x (default: median of y)."},
        "n_model_params": {"type": "int", "description": "Number of free parameters in the peak model (e.g. 4 per Voigt + 1 offset). Needed for the F-test; the test is skipped if omitted."},
        "snr_detect": {"type": "float", "description": "Rose-criterion SNR floor for a real feature (default 3). Below this → 'absent'."},
        "snr_quantify": {"type": "float", "description": "SNR floor for a quantitative lineshape (default 10, the textbook quantitative-NMR floor). Between snr_detect and this → recover the line but report SNR-limited, with explicit uncertainty."},
        "alpha": {"type": "float", "description": "F-test significance level (default 0.01)."},
    },
    required=["x", "y"],
    returns=(
        "dict with 'verdict' (detected/marginal/absent), 'snr', 'p_value', "
        "'significant', 'upper_bound', and a 'recommendation'. Use it to decide "
        "whether to report a fitted lineshape or a non-detection / upper bound."
    ),
    when_to_use=(
        "Any low-SNR or low-γ spectrum (⁶⁷Zn, ²⁵Mg, ³³S, dilute species) where it "
        "is unclear whether there is a quantifiable resonance — run it before "
        "trusting a fit, and again with the fit to F-test it. Skip it for an "
        "obviously strong, high-SNR peak."
    ),
)

TOOL_SPECS = [TOOL_SPEC]
