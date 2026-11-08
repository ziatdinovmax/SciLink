"""HyperSpy files (``.hspy``, and HyperSpy's older ``.hdf5``), read with h5py.

Layout: ``Experiments/<title>/data`` with one ``axis-<i>`` group per array
dimension whose attributes carry ``name``, ``units``, ``offset``, ``scale``,
``size`` and ``navigate``; a non-uniform axis stores its coordinates as an
``axis`` dataset (or attribute) instead. Signal metadata sits in the
signal's ``metadata`` group.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ._layout import canonicalize, uniform_or_values
from ._model import Axis, Dataset, UnsupportedContent


def _s(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.ndarray) and v.shape == ():
        return _s(v.item())
    return "" if v is None else str(v)


def is_hyperspy(h5) -> bool:
    fmt = _s(h5.attrs.get("file_format", "")).lower()
    return fmt == "hyperspy" or ("Experiments" in h5 and fmt in ("", "hyperspy"))


def _metadata(group) -> Dict[str, Any]:
    md: Dict[str, Any] = {}
    meta = group.get("metadata")
    if meta is None:
        return md

    def attr(path, key):
        g = meta.get(path) if path else meta
        return None if g is None else g.attrs.get(key)

    st = _s(attr("Signal", "signal_type"))
    if st:
        md["signal_type"] = st
    q = _s(attr("Signal", "quantity"))
    if q:
        md["intensity_quantity"] = q
    for path, key, out in (
        ("Acquisition_instrument/TEM", "beam_energy", "beam_energy_kV"),
        ("Acquisition_instrument/SEM", "beam_energy", "beam_energy_kV"),
        ("Acquisition_instrument/TEM", "magnification", "magnification"),
        ("Acquisition_instrument/TEM/Detector/EELS", "collection_angle",
         "collection_semi_angle_mrad"),
        ("Acquisition_instrument/TEM", "convergence_angle",
         "convergence_semi_angle_mrad"),
        ("Acquisition_instrument/TEM", "microscope", "microscope"),
    ):
        v = attr(path, key)
        if v is None:
            continue
        if isinstance(v, (bytes, str)):
            if _s(v):
                md[out] = _s(v)
        else:
            try:
                md[out] = float(v)
            except (TypeError, ValueError):
                pass
    title = _s(attr("General", "title"))
    if title:
        md["_title"] = title
    return md


def _axis(g) -> Axis:
    a = g.attrs
    nav = a.get("navigate")
    ax = Axis(name=_s(a.get("name", "")), size=int(a.get("size", 0) or 0),
              units=_s(a.get("units", "")),
              navigate=None if nav is None else bool(nav))
    values = None
    if "axis" in g:
        values = np.asarray(g["axis"][()])
    elif "axis" in a:
        values = np.asarray(a["axis"])
    if values is not None:
        ax.offset, ax.scale, ax.values = uniform_or_values(values)
    else:
        ax.offset = float(a.get("offset", 0.0) or 0.0)
        ax.scale = float(a.get("scale", 1.0) or 1.0)
    return ax


def read_open(h5, path: str) -> List[Dataset]:
    exps = h5.get("Experiments")
    if exps is None:
        raise UnsupportedContent(f"no Experiments group in HyperSpy file {path}")
    out: List[Dataset] = []
    for name in exps:
        g = exps[name]
        if "data" not in g:
            continue
        data = np.asarray(g["data"][()])
        axes = []
        for i in range(data.ndim):
            key = f"axis-{i}"
            axes.append(_axis(g[key]) if key in g else Axis(size=data.shape[i]))
        md = _metadata(g)
        title = md.pop("_title", "") or name
        data, axes, kind = canonicalize(data, axes)
        out.append(Dataset(data=data, axes=axes, kind=kind, format="hspy",
                           source=path, title=title, metadata=md))
    if not out:
        raise UnsupportedContent(f"no signal with data in HyperSpy file {path}")
    return out


def read(path: str) -> List[Dataset]:
    import h5py

    with h5py.File(path, "r") as h5:
        return read_open(h5, path)
