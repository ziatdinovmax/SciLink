"""The fit gate calibrated on the reference, in units of its own noise.

The curve agent's replay gate was tuned on clean data: a fit is "poor" below
R² 0.95. On a real STEM-EELS line scan that fails by construction: R² there is a
measure of signal strength — the thorough reference run was approved at R² 0.91,
and frames with the same fit quality in noise units swung between 0.89 and 0.98
with the plasmon's height.

So the loop asks the reference run what "as good as approved" looks like under
THIS measurement's noise. :func:`residual_excess` is the residual scatter over
the data's own point-to-point noise: about 1 for white noise, higher (and
stable) with correlated detector noise; it does not move with signal strength,
and it jumps when the model no longer describes the data. Deterministic, read
from arrays the fit already wrote (``spectrum_0000/data.npy`` and ``fit.npy``).

(The CHANGE signal used to be calibrated here too, from the fingerprint's
behaviour under resampled residuals. It now lives in :mod:`scilink.live.drift`
and reads the data alone, so its verdict no longer depends on the recipe.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np



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


def calibrate(run_dir: str) -> dict:
    """What the reference run says the fit gate should expect. Empty when the
    run left no arrays — the loop then keeps the agent's constant bar."""
    out: dict = {}
    excess = residual_excess(run_dir)
    if excess is not None:
        out["residual_excess"] = round(excess, 3)
    return out
