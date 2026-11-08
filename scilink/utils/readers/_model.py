"""What every reader returns: an array, its calibrated axes, and what kind of data it is.

Array order is canonical so callers never reorder by format:

==================  =========================  =================================
kind                shape                      example
==================  =========================  =================================
``spectrum``        ``(channels,)``            one EELS / EDS / Raman spectrum
``image``           ``(y, x)``                 a HAADF image
``spectrum_series`` ``(n, channels)``          a line scan, a time series of spectra
``spectrum_image``  ``(y, x, channels)``       an EELS / EDS / CL / Raman map
``image_stack``     ``(n, y, x)``              a time / focal / tilt series
``nd``              as stored                  anything else (4D-STEM, ...)
==================  =========================  =================================

``axes`` follows the array dimensions one to one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Axis:
    name: str = ""
    size: int = 0
    offset: float = 0.0
    scale: float = 1.0
    units: str = ""
    # True for a navigation (scan / frame) axis, False for a signal axis
    # (energy, wavelength, or the two axes of an image), None when unknown.
    navigate: Optional[bool] = None
    # Coordinates of a non-uniform axis; ``offset`` / ``scale`` are then unused.
    values: Optional[np.ndarray] = None

    def coordinates(self) -> np.ndarray:
        if self.values is not None:
            return np.asarray(self.values, dtype=float)
        return self.offset + self.scale * np.arange(self.size, dtype=float)

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"name": self.name, "size": int(self.size),
                             "units": self.units}
        if self.values is not None:
            c = self.coordinates()
            d.update(uniform=False, start=float(c[0]) if c.size else None,
                     end=float(c[-1]) if c.size else None)
        else:
            d.update(offset=float(self.offset), scale=float(self.scale))
        if self.navigate is not None:
            d["navigate"] = bool(self.navigate)
        return d


@dataclass
class Dataset:
    data: np.ndarray
    axes: List[Axis]
    kind: str
    format: str
    source: str
    title: str = ""
    # Small, JSON-friendly facts lifted from the file (signal type, beam
    # energy, instrument, ...). Never the full original tag tree.
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def signal_axis(self) -> Optional[Axis]:
        """The spectral axis of a spectrum / series / spectrum image."""
        if self.kind in ("spectrum", "spectrum_series", "spectrum_image"):
            return self.axes[-1]
        return None

    @property
    def spatial_axes(self) -> List[Axis]:
        if self.kind == "image":
            return list(self.axes)
        if self.kind == "spectrum_image":
            return list(self.axes[:2])
        if self.kind == "image_stack":
            return list(self.axes[1:])
        return []

    def describe(self) -> Dict[str, Any]:
        """A JSON-friendly summary: shape, kind, axes and lifted metadata."""
        return {
            "format": self.format,
            "kind": self.kind,
            "title": self.title,
            "shape": list(self.data.shape),
            "dtype": str(self.data.dtype),
            "axes": [a.as_dict() for a in self.axes],
            "metadata": dict(self.metadata),
        }


def stand_in(shape, dtype) -> np.ndarray:
    """A zero-memory array of ``shape`` and ``dtype``, for a header-only read:
    it answers every question about the dataset but its values."""
    return np.broadcast_to(np.zeros((), dtype=dtype), tuple(int(n) for n in shape))


def load(dset, header_only: bool) -> np.ndarray:
    """An HDF5 dataset's array, or its stand-in when only the header is wanted."""
    return stand_in(dset.shape, dset.dtype) if header_only else np.asarray(dset[()])


class UnsupportedContent(ValueError):
    """The file is in a known format but holds something no reader handles yet."""
