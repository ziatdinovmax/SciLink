"""``score_xrd_match_fast`` tool — fast cross-correlation scoring.

Joint (shift, scale) optimization via 1D cross-correlation at each scale.
Returns best zero-shift, lattice scale, and Pearson correlation as the
similarity score. Designed for on-the-fly ranking — milliseconds per
candidate — not for confident identification on real-lab data.

The robust tier (``score_xrd_match_robust``) accepts the fast tier's
shift/scale as a warm start and refines via Hanawalt or MIP.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from scipy.signal import correlate, find_peaks, peak_widths

from ..._shared._spec import ToolSpec

_logger = logging.getLogger(__name__)


_CORR_ACCEPT_MIN = 0.85
_CORR_MARGINAL_MIN = 0.60


TOOL_SPEC = ToolSpec(
    name="score_xrd_match_fast",
    description=(
        "Cross-correlation scoring of a simulated XRD pattern against an "
        "experimental one. Jointly fits zero-shift (over a search window) "
        "and lattice scale (over a coarse grid); returns Pearson "
        "correlation as the similarity score. Fast tier — use for "
        "on-the-fly ranking. For confident identification on real-lab "
        "data, escalate to score_xrd_match_robust."
    ),
    import_line="from scilink.skills.structure_matching.xrd.score_match_fast import score_xrd_match_fast",
    signature=(
        "score_xrd_match_fast(exp_two_theta, exp_intensity, sim_two_theta, "
        "sim_intensity, fwhm='auto', shift_search=(-0.5, 0.5), "
        "scale_search=(0.99, 1.01, 0.0025), background='subtract_min') -> dict"
    ),
    parameters={
        "exp_two_theta": {
            "type": "list[float]",
            "description": "Experimental 2-theta grid (degrees), monotonically increasing.",
        },
        "exp_intensity": {
            "type": "list[float]",
            "description": "Experimental intensity at each 2-theta. Same length as exp_two_theta.",
        },
        "sim_two_theta": {
            "type": "list[float]",
            "description": "Simulated peak positions (degrees) from simulate_xrd_pattern.",
        },
        "sim_intensity": {
            "type": "list[float]",
            "description": "Simulated peak intensities (relative) from simulate_xrd_pattern.",
        },
        "fwhm": {
            "type": "float | str",
            "description": (
                "Lorentzian FWHM (degrees) for broadening simulated peaks. "
                "Default 'auto' estimates it from the experimental peak widths "
                "(floored at 0.15°), so broad nanocrystalline patterns match "
                "without manual tuning; pass a number to force an exact width."
            ),
        },
        "shift_search": {
            "type": "tuple",
            "description": "(min, max) zero-shift search window in degrees. Default (-0.5, 0.5).",
        },
        "scale_search": {
            "type": "tuple",
            "description": "(min, max, step) lattice-scale grid. Default (0.98, 1.02, 0.002) — covers ±2% lattice-parameter mismatch. Pass None to disable scale search (shift only).",
        },
        "background": {
            "type": "str",
            "description": "Background handling: 'subtract_min' (default) or 'none'.",
        },
    },
    required=["exp_two_theta", "exp_intensity", "sim_two_theta", "sim_intensity"],
    returns=(
        "dict with 'correlation' (float in [-1, 1], higher is better), "
        "'fitted_shift' (degrees), 'fitted_scale' (dimensionless), "
        "'verdict' ('accept' | 'marginal' | 'reject'), 'fwhm_used' (echo)."
    ),
    when_to_use=(
        "Fast triage of candidates from search_structures — on-the-fly "
        "during an experiment, or scout pass over many candidates. For "
        "confident identification on real-lab patterns, escalate to "
        "score_xrd_match_robust."
    ),
)


def score_xrd_match_fast(
    exp_two_theta: Sequence[float],
    exp_intensity: Sequence[float],
    sim_two_theta: Sequence[float],
    sim_intensity: Sequence[float],
    fwhm: float | str = "auto",
    shift_search: tuple = (-0.5, 0.5),
    scale_search: tuple | None = (0.98, 1.02, 0.002),
    background: str = "subtract_min",
) -> dict[str, Any]:
    """Cross-correlation fast scoring. See ``TOOL_SPEC`` for full contract."""
    exp_x = np.asarray(exp_two_theta, dtype=float)
    exp_y = np.asarray(exp_intensity, dtype=float)
    sim_x = np.asarray(sim_two_theta, dtype=float)
    sim_y = np.asarray(sim_intensity, dtype=float)

    if exp_x.shape != exp_y.shape:
        raise ValueError("exp_two_theta and exp_intensity must have the same length")
    if sim_x.shape != sim_y.shape:
        raise ValueError("sim_two_theta and sim_intensity must have the same length")
    if exp_x.size < 16:
        raise ValueError("exp_two_theta must contain at least 16 points for cross-correlation")

    if background == "subtract_min":
        exp_y = exp_y - float(np.min(exp_y))
    elif background != "none":
        raise ValueError(f"Unknown background option: {background!r}")

    grid_step = float(np.mean(np.diff(exp_x)))
    if grid_step <= 0:
        raise ValueError("exp_two_theta must be monotonically increasing")

    # Resolve the simulated-pattern broadening. 'auto' (default) matches it to
    # the experimental peak width, so a nanocrystalline (broad) pattern is not
    # penalised by an over-sharp simulated profile — the failure mode where a
    # broadened correct phase scores worse than a wrong sharp one. Floored at
    # 0.15° so sharp patterns are byte-for-byte identical to the historical
    # fixed default; a numeric value still forces an exact width.
    if isinstance(fwhm, str):
        if fwhm != "auto":
            raise ValueError(f"fwhm must be a positive number or 'auto', got {fwhm!r}")
        fwhm = _estimate_exp_fwhm(exp_y, grid_step)
    fwhm = float(fwhm)
    if fwhm <= 0:
        raise ValueError("fwhm must be positive")

    if sim_x.size == 0:
        return {
            "correlation": 0.0,
            "fitted_shift": 0.0,
            "fitted_scale": 1.0,
            "verdict": "reject",
            "fwhm_used": float(fwhm),
        }

    scales = _build_scale_grid(scale_search)

    exp_z = _zscore(exp_y)
    best = {
        "correlation": -1.0,
        "fitted_shift": 0.0,
        "fitted_scale": 1.0,
    }
    shift_lo, shift_hi = float(shift_search[0]), float(shift_search[1])

    for scale in scales:
        sim_b = _broaden_peaks(exp_x, scale * sim_x, sim_y, fwhm)
        sim_z = _zscore(sim_b)
        if np.allclose(sim_z, 0):
            continue
        # 'same'-mode correlation: result length matches exp_z; index N//2 corresponds to lag 0.
        corr_curve = correlate(exp_z, sim_z, mode="same")
        # Normalize by sample count so the result is Pearson-like in [-1, 1].
        corr_curve = corr_curve / exp_z.size

        # Lag in samples → shift in degrees. lag k means sim is shifted by k*grid_step relative to exp.
        lags = (np.arange(corr_curve.size) - corr_curve.size // 2) * grid_step
        window = (lags >= shift_lo) & (lags <= shift_hi)
        if not np.any(window):
            continue
        windowed = np.where(window, corr_curve, -np.inf)
        peak_idx = int(np.argmax(windowed))
        peak_corr = float(corr_curve[peak_idx])
        if peak_corr > best["correlation"]:
            best = {
                "correlation": peak_corr,
                "fitted_shift": float(lags[peak_idx]),
                "fitted_scale": float(scale),
            }

    corr = best["correlation"]
    if corr >= _CORR_ACCEPT_MIN:
        verdict = "accept"
    elif corr >= _CORR_MARGINAL_MIN:
        verdict = "marginal"
    else:
        verdict = "reject"

    return {
        "correlation": corr,
        "fitted_shift": best["fitted_shift"],
        "fitted_scale": best["fitted_scale"],
        "verdict": verdict,
        "fwhm_used": float(fwhm),
    }


# --- numerical helpers --------------------------------------------------------

def _estimate_exp_fwhm(exp_y, grid_step: float, floor: float = 0.15, ceil: float = 1.0) -> float:
    """Median FWHM (degrees) of the strongest experimental peaks, for adaptive
    simulated-pattern broadening. Floored so sharp patterns keep the historical
    0.15° default and only genuinely broad (nanocrystalline) patterns widen;
    falls back to the floor when too few peaks are resolvable."""
    try:
        y = np.asarray(exp_y, dtype=float)
        ymax = float(y.max())
        if ymax <= 0:
            return floor
        min_sep = max(1, int(round(0.1 / grid_step)))
        idx, _ = find_peaks(y, prominence=0.05 * ymax, distance=min_sep)
        if idx.size == 0:
            return floor
        widths_samples, _, _, _ = peak_widths(y, idx, rel_height=0.5)
        fwhms = np.asarray(widths_samples, dtype=float) * grid_step
        fwhms = fwhms[fwhms > 0]
        if fwhms.size == 0:
            return floor
        return float(min(max(float(np.median(fwhms)), floor), ceil))
    except Exception:
        return floor



def _build_scale_grid(scale_search: tuple | None) -> np.ndarray:
    if scale_search is None:
        return np.array([1.0])
    lo, hi, step = float(scale_search[0]), float(scale_search[1]), float(scale_search[2])
    if step <= 0 or hi < lo:
        raise ValueError(f"Invalid scale_search: {scale_search!r}")
    return np.arange(lo, hi + step * 0.5, step)


def _broaden_peaks(
    grid: np.ndarray,
    peak_x: np.ndarray,
    peak_y: np.ndarray,
    fwhm: float,
) -> np.ndarray:
    """Place Lorentzian peaks of given FWHM on the experimental grid."""
    if peak_x.size == 0:
        return np.zeros_like(grid)
    gamma = fwhm / 2.0
    out = np.zeros_like(grid)
    for x0, amp in zip(peak_x, peak_y):
        if amp <= 0:
            continue
        out += amp * (gamma ** 2) / ((grid - x0) ** 2 + gamma ** 2)
    return out


def _zscore(y: np.ndarray) -> np.ndarray:
    """Subtract the mean and divide by the standard deviation; safe on zeros."""
    mean = float(np.mean(y))
    std = float(np.std(y))
    if std <= 0:
        return np.zeros_like(y)
    return (y - mean) / std
