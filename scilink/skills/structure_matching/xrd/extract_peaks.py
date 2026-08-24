"""``extract_peaks`` tool — pull a peak list out of a continuous XRD pattern.

scipy.signal.find_peaks for detection, optional 3-point parabolic refinement
for sub-pixel positions, optional Lorentzian fit for FWHM. Pure numpy/scipy.

Used by the robust tier (Hanawalt + MIP scorers) which work on peak lists
rather than continuous patterns. The fast tier (cross-correlation) does not
need this helper, but the LLM can call ``extract_peaks`` directly when it
wants to inspect or report the experimental peak list itself.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths

from ..._shared._spec import ToolSpec

_logger = logging.getLogger(__name__)


TOOL_SPEC = ToolSpec(
    name="extract_peaks",
    description=(
        "Detect peaks in a continuous XRD pattern. Uses scipy.signal.find_peaks "
        "for detection, then optionally refines each peak with a 3-point "
        "parabolic fit (sub-pixel position) and a Lorentzian fit (FWHM)."
    ),
    import_line="from scilink.skills.structure_matching.xrd.extract_peaks import extract_peaks",
    signature=(
        "extract_peaks(two_theta, intensity, prominence_frac=0.02, "
        "min_distance_deg=0.2, max_peaks=30, refine=True, "
        "background='subtract_min') -> dict"
    ),
    parameters={
        "two_theta": {
            "type": "list[float]",
            "description": "Two-theta grid (degrees), monotonically increasing.",
        },
        "intensity": {
            "type": "list[float]",
            "description": "Intensity at each 2-theta. Same length as two_theta.",
        },
        "prominence_frac": {
            "type": "float",
            "description": "Minimum prominence as a fraction of (max - min) intensity. Default 0.02 (~2 percent of pattern range).",
        },
        "min_distance_deg": {
            "type": "float",
            "description": "Minimum spacing between detected peaks (degrees). Default 0.2.",
        },
        "max_peaks": {
            "type": "int",
            "description": "Cap on the number of returned peaks (strongest kept). Default 30.",
        },
        "refine": {
            "type": "bool",
            "description": "If True, refine peak position via parabolic interpolation and fit a Lorentzian for FWHM. Default True.",
        },
        "background": {
            "type": "str",
            "description": "Background handling before detection: 'subtract_min' (default) or 'none'.",
        },
    },
    required=["two_theta", "intensity"],
    returns=(
        "dict with 'positions' (list[float], degrees), 'intensities' "
        "(list[float], absolute), 'fwhms' (list[float], degrees; 0.0 if "
        "refine=False), 'prominences' (list[float]). All lists are sorted "
        "by descending intensity and trimmed to max_peaks."
    ),
    when_to_use=(
        "Before calling score_xrd_match_robust to get an experimental peak "
        "list; or any time the LLM wants to inspect/report which peaks "
        "were found in a pattern."
    ),
)


def extract_peaks(
    two_theta: Sequence[float],
    intensity: Sequence[float],
    prominence_frac: float = 0.02,
    min_distance_deg: float = 0.2,
    max_peaks: int = 30,
    refine: bool = True,
    background: str = "subtract_min",
) -> dict[str, Any]:
    """Detect (and optionally refine) peaks in an XRD pattern."""
    x = np.asarray(two_theta, dtype=float)
    y = np.asarray(intensity, dtype=float)

    if x.shape != y.shape:
        raise ValueError("two_theta and intensity must have the same length")
    if x.size < 8:
        raise ValueError("two_theta must contain at least 8 points")
    if not (0 < prominence_frac < 1):
        raise ValueError("prominence_frac must be in (0, 1)")
    if min_distance_deg <= 0:
        raise ValueError("min_distance_deg must be positive")
    if max_peaks <= 0:
        raise ValueError("max_peaks must be positive")

    if background == "subtract_min":
        y = y - float(np.min(y))
    elif background != "none":
        raise ValueError(f"Unknown background option: {background!r}")

    grid_step = float(np.mean(np.diff(x)))
    if grid_step <= 0:
        raise ValueError("two_theta must be monotonically increasing")

    prominence = max(prominence_frac * float(np.ptp(y)), 1e-12)
    distance_samples = max(int(round(min_distance_deg / grid_step)), 1)

    indices, props = find_peaks(y, prominence=prominence, distance=distance_samples)
    if indices.size == 0:
        return _empty_result()

    # Sort by descending intensity, keep top max_peaks
    order = np.argsort(y[indices])[::-1]
    keep = order[: max_peaks]
    indices = indices[keep]
    prominences = props["prominences"][keep]

    positions = []
    intensities = []
    fwhms = []
    refined_proms = []

    if refine:
        widths_samples, _, _, _ = peak_widths(y, indices, rel_height=0.5)
    else:
        widths_samples = np.zeros_like(indices, dtype=float)

    for idx, prom, w_samp in zip(indices, prominences, widths_samples):
        if refine:
            pos = _refine_position(x, y, int(idx))
            fwhm_guess = max(float(w_samp) * grid_step, 2 * grid_step)
            fwhm = _refine_fwhm_lorentzian(x, y, pos, fwhm_guess)
            amp = float(np.interp(pos, x, y))
        else:
            pos = float(x[idx])
            amp = float(y[idx])
            fwhm = 0.0
        positions.append(pos)
        intensities.append(amp)
        fwhms.append(fwhm)
        refined_proms.append(float(prom))

    # Sort by intensity descending (in case refinement perturbed order)
    order = np.argsort(intensities)[::-1]
    return {
        "positions": [positions[i] for i in order],
        "intensities": [intensities[i] for i in order],
        "fwhms": [fwhms[i] for i in order],
        "prominences": [refined_proms[i] for i in order],
    }


def _empty_result() -> dict[str, Any]:
    return {"positions": [], "intensities": [], "fwhms": [], "prominences": []}


def _refine_position(x: np.ndarray, y: np.ndarray, idx: int) -> float:
    """Parabolic interpolation around an integer-index peak for sub-pixel precision."""
    if idx <= 0 or idx >= len(x) - 1:
        return float(x[idx])
    x0, x1, x2 = x[idx - 1], x[idx], x[idx + 1]
    y0, y1, y2 = y[idx - 1], y[idx], y[idx + 1]
    denom = (y0 - 2 * y1 + y2)
    if abs(denom) < 1e-12:
        return float(x1)
    shift = 0.5 * (y0 - y2) / denom
    return float(x1 + shift * (x2 - x1))


def _lorentzian(x, amp, x0, gamma):
    return amp * (gamma ** 2) / ((x - x0) ** 2 + gamma ** 2)


def _refine_fwhm_lorentzian(
    x: np.ndarray,
    y: np.ndarray,
    pos: float,
    fwhm_guess: float,
) -> float:
    """Fit a Lorentzian in a window around ``pos`` and return its FWHM."""
    window = max(fwhm_guess * 3, 0.5)
    mask = (x >= pos - window) & (x <= pos + window)
    if np.count_nonzero(mask) < 5:
        return float(fwhm_guess)
    try:
        amp_guess = float(np.interp(pos, x, y))
        popt, _ = curve_fit(
            _lorentzian,
            x[mask],
            y[mask],
            p0=[amp_guess, pos, fwhm_guess / 2],
            maxfev=200,
        )
        gamma = abs(popt[2])
        # FWHM = 2 * gamma for Lorentzian
        return float(min(max(2 * gamma, 0.0), 5.0))
    except Exception as e:
        _logger.debug("Lorentzian fit failed at %.3f: %s", pos, e)
        return float(fwhm_guess)
