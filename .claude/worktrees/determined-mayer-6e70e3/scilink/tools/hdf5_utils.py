"""HDF5 / NeXus loaders shared across analysis agents.

NeXus parsing uses the official ``nexusformat`` library; no hand-rolled
NeXus walking lives here.  For non-NeXus HDF5, we pick the largest
dataset as a best-effort signal so agents get *something* to work with.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def load_hdf5_signal(
    path: str,
    *,
    return_axes: bool = False,
) -> "np.ndarray | Tuple[np.ndarray, list[np.ndarray]]":
    """Load the primary signal array from an HDF5 file.

    Tries ``nexusformat`` first to honour the NeXus standard
    (default NXentry → default NXdata → signal). Falls back to walking
    the file with ``h5py`` and picking the largest dataset for non-NeXus
    files.

    Parameters
    ----------
    path
        Path to ``.h5`` / ``.hdf5`` file.
    return_axes
        If True, also return a list of 1D numpy axis arrays in the same
        order as the signal's dimensions. Axes that aren't defined are
        returned as ``None`` placeholders.

    Returns
    -------
    np.ndarray or (np.ndarray, list[np.ndarray | None])
    """
    try:
        import nexusformat.nexus as nx
    except ImportError as exc:
        raise ImportError(
            "Reading HDF5/NeXus files requires nexusformat. "
            "Install with: pip install nexusformat"
        ) from exc

    signal_arr: Optional[np.ndarray] = None
    axes_arr: list = []

    try:
        root = nx.nxload(path, mode="r")
        entries = list(root.NXentry)
        if entries:
            default_entry_name = root.attrs.get("default")
            entry = next(
                (e for e in entries if e.nxname == default_entry_name),
                entries[0],
            )
            nxdatas = list(entry.NXdata)
            if nxdatas:
                default_nxdata = entry.attrs.get("default")
                nxdata = next(
                    (d for d in nxdatas if d.nxname == default_nxdata),
                    nxdatas[0],
                )
                signal = nxdata.nxsignal
                if signal is not None:
                    signal_arr = np.asarray(signal.nxdata)
                    if return_axes:
                        try:
                            for ax in nxdata.nxaxes or []:
                                try:
                                    axes_arr.append(np.asarray(ax.nxdata))
                                except Exception:
                                    axes_arr.append(None)
                        except Exception:
                            axes_arr = []
    except Exception:
        signal_arr = None  # fall through to h5py

    if signal_arr is None:
        # Non-NeXus or unparseable — pick the largest dataset.
        import h5py

        largest: tuple[int, "h5py.Dataset | None"] = (0, None)

        def visit(name, obj):
            nonlocal largest
            if isinstance(obj, h5py.Dataset):
                size = int(np.prod(obj.shape)) if obj.shape else 0
                if size > largest[0]:
                    largest = (size, obj)

        with h5py.File(path, "r") as f:
            f.visititems(visit)
            if largest[1] is None:
                raise ValueError(f"No datasets found in HDF5 file: {path}")
            signal_arr = np.asarray(largest[1][()])

    if return_axes:
        # Pad axes list to match signal rank.
        while len(axes_arr) < signal_arr.ndim:
            axes_arr.append(None)
        return signal_arr, axes_arr
    return signal_arr
