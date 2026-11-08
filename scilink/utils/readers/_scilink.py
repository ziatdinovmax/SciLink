"""The bridge to SciLink: arrays for the agents' loaders, metadata in SciLink's schema.

Loaders call :func:`native_array` only for the formats SciLink could not open
before (``CALIBRATED_SUFFIXES``); every other format keeps its own code path.
:func:`file_metadata` writes what a file records about itself in the schema
the agents already read (``energy_range``, ``spatial_info``, ``axis_spec``,
``experiment``), so it needs no normalization and no model call.
:func:`merge_file_metadata` fills what the person did not give from the file
and reports any value where the two disagree; the person's value stands.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._model import Dataset

_TECHNIQUE = {
    "eels": "EELS (electron energy-loss spectroscopy)",
    "eds": "EDS (energy-dispersive X-ray spectroscopy)",
    "eds_tem": "EDS (energy-dispersive X-ray spectroscopy)",
    "eds_sem": "EDS (energy-dispersive X-ray spectroscopy)",
    "x-ray": "EDS (energy-dispersive X-ray spectroscopy)",
    "cl": "Cathodoluminescence",
    "cl_sem": "Cathodoluminescence",
    "cl_stem": "Cathodoluminescence",
    "raman": "Raman spectroscopy",
}


# What the file metadata says where the file records nothing. It keeps
# metadata read from a file alone complete for the schema; it is never
# written over, or beside, anything the person gave.
NOT_RECORDED = "not recorded in the file"
TECHNIQUE_NOT_NAMED = "(technique not named in the file)"


def _is_placeholder(v) -> bool:
    return isinstance(v, str) and (v == NOT_RECORDED or v.endswith(TECHNIQUE_NOT_NAMED))


def is_native(path) -> bool:
    """A format only the readers open (``.dm3``, ``.dm4``, ``.hspy``, ``.emd``)."""
    from . import CALIBRATED_SUFFIXES
    return Path(str(path)).suffix.lower() in CALIBRATED_SUFFIXES


def native_array(path, prefer: Sequence[str] = (),
                 accept: Optional[Sequence[str]] = None) -> np.ndarray:
    """The array of the file's dataset of the first ``prefer`` kind.

    With ``accept``, a file holding no dataset of an accepted kind is refused
    by name rather than handing the caller another kind of data (a stack read
    as an image, an image read as a cube).
    """
    from . import read_dataset, read_datasets, pick
    if accept is None:
        return read_dataset(path, prefer=prefer).data
    found = read_datasets(path)
    ds = pick(found, prefer)
    if ds.kind not in accept:
        from ._model import UnsupportedContent
        held = ", ".join(sorted({f"{d.kind} {tuple(d.data.shape)}" for d in found}))
        raise UnsupportedContent(
            f"{Path(str(path)).name} holds {held}; this loader takes "
            f"{' or '.join(accept)} data")
    return ds.data


def records_calibration(path) -> bool:
    """True when the file carries its own calibration (no array is read)."""
    if is_native(path):
        return True
    if Path(str(path)).suffix.lower() in (".h5", ".hdf5", ".nxs"):
        try:
            import h5py
            from . import emd, hspy
            with h5py.File(str(path), "r") as h5:
                return bool(hspy.is_hyperspy(h5) or emd.is_velox(h5)
                            or emd._berkeley_groups(h5))
        except Exception:
            return False
    return False


def calibrated_dataset(path, prefer: Sequence[str] = (),
                       header_only: bool = False) -> Optional[Dataset]:
    """The dataset when the file carries its own calibration, else None.

    True for the native formats and for an HDF5 file that is HyperSpy, EMD or
    Velox inside; a plain array (``.npy``, generic HDF5, an image) is None.
    ``header_only=True`` when only the metadata is wanted (see
    :func:`read_datasets`).
    """
    from . import read_dataset
    if records_calibration(path):
        return read_dataset(path, prefer=prefer, header_only=header_only)
    return None


def _num(v: float) -> float:
    return float(f"{float(v):.10g}")


def file_metadata(ds: Dataset) -> Dict[str, Any]:
    """SciLink metadata from what the file itself records. Nothing is inferred."""
    md: Dict[str, Any] = {}
    spectral = ds.kind in ("spectrum", "spectrum_series", "spectrum_image")
    md["experiment_type"] = "Spectroscopy" if spectral else "Microscopy"
    sig = str(ds.metadata.get("signal_type") or "")
    tech = _TECHNIQUE.get(sig.lower())
    # The schema requires a technique and a sample. A file seldom names the
    # sample and not every file names its technique; say so plainly rather
    # than guess, so the metadata is complete as it stands and never goes
    # through a model-based normalization that could touch its numbers.
    kind_words = {"spectrum": "spectrum", "spectrum_series": "spectra",
                  "spectrum_image": "spectrum image", "image": "image",
                  "image_stack": "image series"}.get(ds.kind, "data")
    exp: Dict[str, Any] = {
        "technique": tech or (f"{sig} {kind_words} {TECHNIQUE_NOT_NAMED}"
                              if sig else f"{kind_words} {TECHNIQUE_NOT_NAMED}")}
    if ds.metadata.get("microscope"):
        exp["instrument"] = ds.metadata["microscope"]
    md["experiment"] = exp
    md["sample"] = {"material": NOT_RECORDED}

    ax = ds.signal_axis
    if ax is not None and ax.size >= 2:
        c = ax.coordinates()
        md["energy_range"] = {"start": _num(c[0]), "end": _num(c[-1]), "units": ax.units or ""}
        if ax.values is not None:
            md["axis_spec"] = {"axis_2": {"name": ax.name or "signal", "units": ax.units or "",
                                          "samples": [float(v) for v in c]}}
        if ds.kind == "spectrum":
            md["xlabel"] = f"{ax.name or 'Signal'} ({ax.units})" if ax.units else (ax.name or "Signal")
            if ds.metadata.get("intensity_units"):
                md["ylabel"] = str(ds.metadata["intensity_units"])

    sp = ds.spatial_axes
    if ds.metadata.get("rgb"):
        sp = sp[:2]
    if len(sp) == 2 and sp[0].units and sp[0].units == sp[1].units:
        y, x = sp
        md["spatial_info"] = {
            "field_of_view_x": _num(x.size * x.scale),
            "field_of_view_y": _num(y.size * y.scale),
            "field_of_view_units": x.units,
            "pixel_size": _num(x.scale),
        }

    acq = {k: v for k, v in ds.metadata.items()
           if k in ("beam_energy_kV", "magnification", "collection_semi_angle_mrad",
                    "convergence_semi_angle_mrad", "exposure_s", "operation_mode",
                    "illumination_mode", "detector", "summed_over", "intensity_units",
                    "intensity_quantity")}
    if sig:
        acq["signal_type"] = sig
    if acq:
        md["acquisition"] = acq
    md["metadata_source"] = (f"Read from the {ds.format} file: axis calibration and "
                             "acquisition values as the instrument recorded them.")
    return md


_CHECKED = (("energy_range", "start"), ("energy_range", "end"), ("energy_range", "units"),
            ("spatial_info", "pixel_size"), ("spatial_info", "field_of_view_units"))


def _close(a, b) -> bool:
    try:
        fa, fb = float(a), float(b)
        return math.isclose(fa, fb, rel_tol=1e-3, abs_tol=1e-9)
    except (TypeError, ValueError):
        return str(a).strip().lower() == str(b).strip().lower()


def merge_file_metadata(given: Optional[Dict[str, Any]], from_file: Dict[str, Any]
                        ) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """``(merged, filled, conflicts)``.

    The person's metadata stands on every field it sets; the file fills the
    rest (nested dicts merged key by key). ``conflicts`` names each
    calibration value where the two disagree, so it is said, never resolved
    silently. ``given`` should be in the schema's shape already (the
    caller's deterministic normalizer); a technique or material it gives at
    the top level is still never covered by the file's "not recorded"
    placeholders, which are filled only into metadata that names neither.
    """
    merged: Dict[str, Any] = dict(given or {})
    filled: List[str] = []
    flat = {k.lower() for k in merged if merged.get(k) not in (None, "")}
    for key, val in from_file.items():
        if key == "metadata_source" or (key == "sample" and key in merged):
            continue
        if isinstance(val, dict):
            val = {k2: v2 for k2, v2 in val.items()
                   if not (_is_placeholder(v2) and k2.lower() in flat)}
            if not val:
                continue
        if key not in merged or merged[key] in (None, "", {}):
            merged[key] = val
            filled.append(key)
        elif isinstance(merged[key], dict) and isinstance(val, dict):
            inner = dict(merged[key])
            for k2, v2 in val.items():
                if k2 not in inner or inner[k2] in (None, ""):
                    inner[k2] = v2
                    filled.append(f"{key}.{k2}")
            merged[key] = inner
    if filled:
        merged["metadata_filled_from_file"] = filled
    conflicts: List[str] = []
    for a, b in _CHECKED:
        g = (given or {}).get(a)
        f = from_file.get(a)
        if isinstance(g, dict) and isinstance(f, dict) and g.get(b) not in (None, "") \
                and f.get(b) not in (None, "") and not _close(g[b], f[b]):
            conflicts.append(f"{a}.{b}: given {g[b]!r}, file records {f[b]!r}")
    return merged, filled, conflicts
