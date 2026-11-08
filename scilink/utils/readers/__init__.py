"""Readers for common microscopy and spectroscopy files, with calibrated axes.

``read_dataset(path)`` returns a :class:`Dataset`: the array in canonical
order (``(y, x, channels)`` for a spectrum image, ``(y, x)`` for an
image, ...), one :class:`Axis` per dimension with name, units, offset and
scale, the kind of data, and a few facts lifted from the file (signal
type, beam energy, instrument). Only numpy, h5py and the standard library
are used.

==============================  =========================================
suffix                          reader
==============================  =========================================
``.dm3``, ``.dm4``              Gatan DigitalMicrograph (``dm``)
``.hspy``                       HyperSpy (``hspy``)
``.emd``                        Berkeley/NCEM EMD and Velox (``emd``)
``.h5``, ``.hdf5``, ``.nxs``    HyperSpy, EMD or Velox when the file says
                                so, else NeXus / largest dataset
``.npy``                        the array as stored, no calibration
==============================  =========================================

Scope: one file is one dataset. A folder of these files is not yet read as
a series: the orchestrator's folder detection counts only CSV, ``.npy`` and
image files, the file's own metadata is used for a single file only, and a
series' control variable (temperature, time, dose) seldom sits in the file
and must come from the person, per-file sidecars or the file names.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ._model import Axis, Dataset, UnsupportedContent

__all__ = ["Axis", "Dataset", "UnsupportedContent", "CALIBRATED_SUFFIXES",
           "READABLE_SUFFIXES", "can_read", "pick", "read_dataset", "read_datasets",
           "is_native", "native_array", "calibrated_dataset", "file_metadata",
           "merge_file_metadata", "records_calibration", "PREFER_CURVE",
           "PREFER_IMAGE", "PREFER_CUBE"]

# Formats SciLink could not open before these readers.
CALIBRATED_SUFFIXES = frozenset({".dm3", ".dm4", ".hspy", ".emd"})
_HDF5_SUFFIXES = frozenset({".h5", ".hdf5", ".nxs"})
READABLE_SUFFIXES = CALIBRATED_SUFFIXES | _HDF5_SUFFIXES | {".npy"}

# Which dataset of a file each kind of analysis takes, in order of
# preference; a loader and the metadata read for its run use the same one,
# so the calibration describes the array that is analysed.
PREFER_CURVE = ("spectrum", "spectrum_series")
PREFER_IMAGE = ("image", "image_stack")
PREFER_CUBE = ("spectrum_image",)


def can_read(path) -> bool:
    return Path(str(path)).suffix.lower() in READABLE_SUFFIXES


def _read_hdf5_family(path: str, suffix: str, decode_streams: bool = True) -> List[Dataset]:
    import h5py

    from . import emd, hspy, nexus

    with h5py.File(path, "r") as h5:
        if hspy.is_hyperspy(h5):
            return hspy.read_open(h5, path)
        if emd.is_velox(h5) or emd._berkeley_groups(h5):
            return emd.read_open(h5, path, decode_streams=decode_streams)
    if suffix == ".emd":
        raise UnsupportedContent(f"{path} is HDF5 but not a recognised EMD layout")
    return nexus.read(path)


def read_datasets(path, decode_streams: bool = True) -> List[Dataset]:
    """Every dataset the file holds, primary first where the format says which.

    ``decode_streams=False`` is for a caller that needs only kinds, shapes,
    axes and metadata (a probe, a metadata read): an EDS event stream is then
    not decoded, and its dataset's ``data`` is a zero-memory stand-in of the
    right shape, marked ``metadata["not_decoded"]``.
    """
    path = str(path)
    suffix = Path(path).suffix.lower()
    if suffix in (".dm3", ".dm4"):
        from . import dm
        return dm.read(path)
    if suffix == ".hspy":
        from . import hspy
        return hspy.read(path)
    if suffix == ".emd" or suffix in _HDF5_SUFFIXES:
        return _read_hdf5_family(path, suffix, decode_streams)
    if suffix == ".npy":
        from ._layout import canonicalize
        data = np.load(path)
        axes = [Axis(size=n) for n in data.shape]
        data, axes, kind = canonicalize(data, axes)
        return [Dataset(data=data, axes=axes, kind=kind, format="npy", source=path)]
    raise UnsupportedContent(f"no reader for '{suffix}' files")


def pick(found: Sequence[Dataset], prefer: Optional[Sequence[str]] = None) -> Dataset:
    """The first dataset of the first preferred kind, else the largest."""
    for kind in prefer or ():
        for ds in found:
            if ds.kind == kind:
                return ds
    return max(found, key=lambda d: d.data.size)


def read_dataset(path, prefer: Optional[Sequence[str]] = None,
                 decode_streams: bool = True) -> Dataset:
    """The file's primary dataset.

    ``prefer`` lists kinds in order of preference (e.g. ``("spectrum_image",)``
    for a hyperspectral caller); the first dataset of the first listed kind
    wins. Otherwise the largest dataset wins, so a survey image or a
    thumbnail stored beside a spectrum image never replaces it.
    """
    return pick(read_datasets(path, decode_streams=decode_streams), prefer)


from ._scilink import (calibrated_dataset, file_metadata,  # noqa: E402
                       is_native, merge_file_metadata, native_array,
                       records_calibration)
