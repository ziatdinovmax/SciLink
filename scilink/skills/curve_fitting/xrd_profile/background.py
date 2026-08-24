"""``fit_background`` tool — XRD background estimation.

Two methods:

- ``'snip'`` — Statistics-sensitive Non-linear Iterative Peak-clipping.
  Standard p-XRD choice for smooth amorphous backgrounds and
  fluorescence floors. No polynomial-shape assumption.

- ``'polynomial'`` — least-squares Chebyshev fit to the lower envelope
  of the pattern. Use when the background is genuinely polynomial
  (capillary scattering on a flat baseline).
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from ..._shared._spec import ToolSpec

_logger = logging.getLogger(__name__)


TOOL_SPEC = ToolSpec(
    name="fit_background",
    description=(
        "Estimate and subtract the continuous background of an XRD "
        "pattern. SNIP (default) handles amorphous halos and "
        "fluorescence floors; 'polynomial' uses a Chebyshev fit to the "
        "lower envelope. Returns the background curve and the "
        "background-subtracted intensity."
    ),
    import_line="from scilink.skills.curve_fitting.xrd_profile.background import fit_background",
    signature=(
        "fit_background(two_theta, intensity, method='snip', "
        "iterations=24, poly_order=3) -> dict"
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
        "method": {
            "type": "str",
            "description": "'snip' (default, recommended for p-XRD) or 'polynomial'.",
        },
        "iterations": {
            "type": "int",
            "description": "SNIP iterations (controls how broad features the algorithm treats as background). Default 24. Ignored by 'polynomial'.",
        },
        "poly_order": {
            "type": "int",
            "description": "Chebyshev polynomial order for 'polynomial' method. Default 3. Ignored by 'snip'.",
        },
    },
    required=["two_theta", "intensity"],
    returns=(
        "dict with 'background' (list[float]; the estimated background), "
        "'intensity_corrected' (list[float]; intensity minus background, "
        "clipped at 0), 'method' (echoed)."
    ),
    when_to_use=(
        "Before per-peak fitting (fit_profile) or peak detection "
        "(extract_peaks) — both work much better on background-"
        "subtracted patterns."
    ),
)


def fit_background(
    two_theta: Sequence[float],
    intensity: Sequence[float],
    method: str = "snip",
    iterations: int = 24,
    poly_order: int = 3,
) -> dict[str, Any]:
    """Estimate continuous background. See ``TOOL_SPEC`` for full contract."""
    x = np.asarray(two_theta, dtype=float)
    y = np.asarray(intensity, dtype=float)

    if x.shape != y.shape:
        raise ValueError("two_theta and intensity must have the same length")
    if x.size < 8:
        raise ValueError("two_theta must contain at least 8 points")
    if method not in {"snip", "polynomial"}:
        raise ValueError(f"Unknown method: {method!r}. Use 'snip' or 'polynomial'.")

    if method == "snip":
        bg = _snip(y, iterations=iterations)
    else:
        bg = _polynomial_envelope(x, y, order=poly_order)

    corrected = np.clip(y - bg, 0.0, None)
    return {
        "background": [float(v) for v in bg],
        "intensity_corrected": [float(v) for v in corrected],
        "method": method,
    }


def _snip(y: np.ndarray, iterations: int = 24) -> np.ndarray:
    """SNIP algorithm on log-log-transformed counts."""
    if iterations < 1:
        raise ValueError(f"iterations must be ≥ 1; got {iterations}")
    y_shifted = y - float(np.min(y)) + 1.0
    v = np.log(np.log(np.sqrt(y_shifted) + 1.0) + 1.0)
    n = v.size

    for p in range(1, iterations + 1):
        v_new = v.copy()
        for i in range(p, n - p):
            v_new[i] = min(v[i], 0.5 * (v[i - p] + v[i + p]))
        v = v_new

    bg_shifted = (np.exp(np.exp(v) - 1.0) - 1.0) ** 2
    bg = bg_shifted + float(np.min(y)) - 1.0
    return bg


def _polynomial_envelope(x: np.ndarray, y: np.ndarray, order: int = 3) -> np.ndarray:
    """Iteratively fit a Chebyshev polynomial to the lower envelope of (x, y)."""
    if order < 0:
        raise ValueError(f"poly_order must be ≥ 0; got {order}")
    mask = np.ones_like(y, dtype=bool)
    bg = np.zeros_like(y)
    for _ in range(8):
        coeffs = np.polynomial.chebyshev.chebfit(x[mask], y[mask], order)
        bg = np.polynomial.chebyshev.chebval(x, coeffs)
        residual = y - bg
        mask_new = residual < np.percentile(residual[mask], 60)
        if np.array_equal(mask_new, mask):
            break
        mask = mask_new
    return bg
