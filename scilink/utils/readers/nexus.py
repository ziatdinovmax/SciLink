"""NeXus and generic HDF5: the loader SciLink always used, now also returning axes.

The array comes from ``utils.hdf5_utils.load_hdf5_signal`` unchanged (the
NeXus default signal, else the largest dataset), so every caller that
read an ``.h5`` before reads the same array now. Axes are the NeXus
coordinate arrays where the file declares them, else bare.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ._layout import canonicalize, uniform_or_values
from ._model import Axis, Dataset


def read(path: str) -> List[Dataset]:
    from ..hdf5_utils import load_hdf5_signal

    data, coords = load_hdf5_signal(path, return_axes=True)
    data = np.asarray(data)
    axes: List[Axis] = []
    for i, n in enumerate(data.shape):
        c = coords[i] if i < len(coords) else None
        ax = Axis(size=n)
        if c is not None:
            c = np.asarray(c).ravel()
            if c.size == n and np.issubdtype(c.dtype, np.number):
                ax.offset, ax.scale, ax.values = uniform_or_values(c)
        axes.append(ax)
    # No names or units to decide by: canonicalize keeps the stored order.
    data, axes, kind = canonicalize(data, axes)
    return [Dataset(data=data, axes=axes, kind=kind, format="hdf5",
                    source=path)]
