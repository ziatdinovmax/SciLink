"""Decide what kind of data an array is from its axes, and put it in canonical order.

Readers hand over the array in stored order with whatever they know about
each axis. A format that marks signal and navigation axes (HyperSpy's
``navigate``) decides by those flags; otherwise an axis's name and units
decide. A wavelength axis in nm is told from a spatial one in nm by its
name, never by its units alone.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np

from ._model import Axis

_SPECTRAL_UNITS = {
    "ev", "kev", "mev", "cm-1", "cm^-1", "1/cm", "cm⁻¹", "/cm", "thz", "ghz",
    "mhz", "hz", "ppm", "deg", "degree", "degrees", "°", "2θ", "mt", "gauss",
    "1/nm", "nm-1", "1/a", "1/å", "a-1", "å-1",
}
_SPATIAL_UNITS = {
    "nm", "µm", "μm", "um", "micron", "microns", "mm", "m", "å", "a",
    "angstrom", "angstroms", "pm", "px", "pixel", "pixels",
}
_SPECTRAL_WORDS = re.compile(
    r"energy|wavelength|wave ?number|raman|shift|frequency|binding|kinetic|"
    r"two ?theta|2 ?theta|2θ|field|spectr|loss|q[- ]?space|scattering vector",
    re.IGNORECASE,
)
_SPATIAL_NAMES = {"x", "y", "z", "width", "height", "row", "column", "col",
                  "scan x", "scan y", "position"}


def _norm_units(units: str) -> str:
    return (units or "").strip().lower().replace(" ", "")


def is_spectral(ax: Axis) -> bool:
    if _SPECTRAL_WORDS.search(ax.name or ""):
        return True
    u = _norm_units(ax.units)
    return u in _SPECTRAL_UNITS


def is_spatial(ax: Axis) -> bool:
    if is_spectral(ax):
        return False
    if (ax.name or "").strip().lower() in _SPATIAL_NAMES:
        return True
    return _norm_units(ax.units) in _SPATIAL_UNITS


def _move(data: np.ndarray, axes: List[Axis], order: List[int]):
    return np.transpose(data, order), [axes[i] for i in order]


def canonicalize(data: np.ndarray, axes: List[Axis]) -> Tuple[np.ndarray, List[Axis], str]:
    """Return ``(data, axes, kind)`` in the canonical order of ``_model``."""
    nd = data.ndim
    if len(axes) != nd:
        raise ValueError(f"{len(axes)} axes for a {nd}-D array")
    for ax, n in zip(axes, data.shape):
        ax.size = int(n)

    flags = [a.navigate for a in axes]
    known = all(f is not None for f in flags)
    if known:
        signal = [i for i, f in enumerate(flags) if not f]
        nav = [i for i, f in enumerate(flags) if f]
    else:
        signal = [i for i, a in enumerate(axes) if is_spectral(a)]
        nav = [i for i in range(nd) if i not in signal]

    if nd == 1:
        axes[0].navigate = False
        return data, axes, "spectrum"

    if nd == 2:
        if len(signal) == 1 and len(nav) == 1 and (known or not is_spatial(axes[signal[0]])):
            d, a = _move(data, axes, [nav[0], signal[0]])
            a[0].navigate, a[1].navigate = True, False
            return d, a, "spectrum_series"
        for ax in axes:
            ax.navigate = False
        return data, axes, "image"

    if nd == 3:
        if len(signal) == 1:
            s = signal[0]
            rest = [i for i in range(3) if i != s]
            d, a = _move(data, axes, rest + [s])
            a[0].navigate = a[1].navigate = True
            a[2].navigate = False
            return d, a, "spectrum_image"
        if known and len(signal) == 2:
            n = nav[0]
            d, a = _move(data, axes, [n] + signal)
            a[0].navigate = True
            a[1].navigate = a[2].navigate = False
            return d, a, "image_stack"
        if not known and not signal:
            spatial = [i for i, ax in enumerate(axes) if is_spatial(ax)]
            if len(spatial) == 2:
                other = [i for i in range(3) if i not in spatial][0]
                d, a = _move(data, axes, [other] + spatial)
                a[0].navigate = True
                a[1].navigate = a[2].navigate = False
                return d, a, "image_stack"
        # Nothing to decide by: keep the stored order and follow SciLink's
        # long-standing convention for a bare 3-D array (channels last).
        if data.shape[-1] > 4:
            return data, axes, "spectrum_image"
        return data, axes, "nd"

    return data, axes, "nd"


def uniform_or_values(values: np.ndarray, *, rtol: float = 1e-6) -> Tuple[float, float, Optional[np.ndarray]]:
    """``(offset, scale, None)`` for evenly spaced coordinates, else ``(first, 1, values)``."""
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        return 0.0, 1.0, None
    if v.size == 1:
        return float(v[0]), 1.0, None
    step = np.diff(v)
    if np.allclose(step, step[0], rtol=rtol, atol=abs(step[0]) * rtol):
        return float(v[0]), float(step[0]), None
    return float(v[0]), 1.0, v
