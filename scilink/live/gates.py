"""Gates calibrated on the reference, in units of its own noise.

The curve agent's replay gates were tuned on clean data: a fit is "poor" below
R² 0.95, and the data has "drifted" when its fingerprint is less than 0.92
similar to the anchor's. On a real STEM-EELS line scan both failed by
construction. R² there is a measure of signal strength — the thorough reference
run was approved at R² 0.91, and frames with the same fit quality in noise units
swung between 0.89 and 0.98 with the plasmon's height — and the fingerprint,
which counts peaks, wobbled between 0.82 and 0.93 on noise alone. Three frames
in four were flagged and the loop re-anchored twice for nothing.

So the loop asks the reference run what "as good as approved" and "the same
data" look like under THIS measurement's noise:

- :func:`residual_excess` — residual scatter over the data's own point-to-point
  noise. About 1 for white noise, higher (and stable) with correlated detector
  noise; it does not move with signal strength, and it jumps when the model no
  longer describes the data.
- :func:`similarity_under_noise` — the fingerprint similarity of the reference
  to copies of itself carrying resampled residuals: the floor below which a
  difference is more than noise.

Both are deterministic, read only arrays the fit already wrote
(``spectrum_0000/data.npy`` and ``fit.npy``), and cost milliseconds.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

#: The constant bars these calibrations relax — never tighten.
DEFAULT_DRIFT_BAR = 0.92


def _arrays(run_dir: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(x, y, fit) of a single-spectrum run, or None when they are not there."""
    d = Path(run_dir) / "spectrum_0000"
    try:
        data = np.asarray(np.load(d / "data.npy"), dtype=float)
        fit = np.asarray(np.load(d / "fit.npy"), dtype=float)
    except (OSError, ValueError):
        return None
    if data.ndim != 2:
        return None
    if data.shape[0] == 2 and data.shape[1] != 2:
        data = data.T
    if fit.ndim == 2:
        fit = fit[1] if (fit.shape[0] == 2 and fit.shape[1] != 2) else fit[:, -1]
    x, y, fit = data[:, 0], data[:, 1], fit.ravel()
    if fit.size != y.size:                       # a fit over a sub-window: not comparable
        return None
    ok = np.isfinite(y) & np.isfinite(fit)
    if ok.sum() < 20:
        return None
    return x[ok], y[ok], fit[ok]


def _point_noise(y: np.ndarray) -> float:
    """Robust point-to-point noise: MAD of SECOND differences. A smooth signal
    cancels to second order (first differences still carry a strong peak's
    slope and inflate the estimate at high signal-to-noise); outliers do not
    move a median. For white noise var(y[i-1] - 2 y[i] + y[i+1]) = 6 σ²."""
    d = np.diff(y, n=2)
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(6.0))


def residual_excess(run_dir: str) -> Optional[float]:
    """std(data − fit) over the data's own point-to-point noise, or None."""
    arrays = _arrays(run_dir)
    if arrays is None:
        return None
    _, y, fit = arrays
    noise = _point_noise(y)
    if not np.isfinite(noise) or noise <= 0:
        return None
    return float(np.std(y - fit) / noise)


def similarity_under_noise(run_dir: str, n_draws: int = 12, block: int = 8,
                           seed: int = 0) -> Optional[List[float]]:
    """Fingerprint similarity between the reference and ``n_draws`` copies of it
    — its own fit plus its own residuals, reshuffled in blocks so detector
    correlation survives. What the drift signal reads from noise alone."""
    arrays = _arrays(run_dir)
    if arrays is None:
        return None
    from ..skills._shared import _script_bank
    x, y, fit = arrays
    resid = y - fit
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(resid.size / block))
    anchor = _script_bank.curve_fingerprint(x, y)
    sims = []
    for _ in range(n_draws):
        starts = rng.integers(0, max(1, resid.size - block), n_blocks)
        shuffled = np.concatenate([resid[s:s + block] for s in starts])[:resid.size]
        if shuffled.size < resid.size:
            shuffled = np.resize(shuffled, resid.size)
        draw = _script_bank.curve_fingerprint(x, fit + shuffled)
        sims.append(float(_script_bank._curve_similarity(draw, anchor)))
    return sims


def calibrate(run_dir: str, drift_margin: float = 0.05) -> dict:
    """What the reference run says the gates should expect. Empty when the run
    left no arrays — the loop then keeps the agent's constant bars."""
    out: dict = {}
    excess = residual_excess(run_dir)
    if excess is not None:
        out["residual_excess"] = round(excess, 3)
    sims = similarity_under_noise(run_dir)
    if sims:
        out["similarity_under_noise"] = [round(min(sims), 3), round(float(np.median(sims)), 3)]
        out["drift_floor"] = round(min(DEFAULT_DRIFT_BAR, min(sims) - drift_margin), 3)
    return out
