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
    """A format only the readers open (``.dm3``, ``.dm4``, ``.hspy``, ``.emd``, and
    the diffraction instrument files when their content is one)."""
    from . import CALIBRATED_SUFFIXES, DIFFRACTION_SUFFIXES
    suffix = Path(str(path)).suffix.lower()
    if suffix in DIFFRACTION_SUFFIXES:
        from . import diffraction
        return diffraction.sniff(path) is not None
    return suffix in CALIBRATED_SUFFIXES


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


def _diffraction_metadata(ds: Dataset) -> Dict[str, Any]:
    """An X-ray diffraction file's record in the schema the XRD tools read:
    ``experiment.wavelength`` (the line dictionary), ``experiment.radiation``,
    ``experiment.geometry`` (only when the file records the sample mode)."""
    m = ds.metadata
    exp: Dict[str, Any] = {"technique": "X-ray diffraction (XRD)"}
    if m.get("instrument"):
        exp["instrument"] = m["instrument"]
    anode = m.get("anode")
    w = m.get("wavelength_angstrom")
    if w:
        exp["wavelength"] = dict(w)
    elif anode:
        exp["wavelength"] = f"{anode}Ka"
    if anode or w:
        src = f"{anode} " if anode else ""
        ratio = (w or {}).get("K-alpha2/K-alpha1")
        if m.get("incident_monochromator"):
            rad = f"{src}K-alpha1 (incident-beam monochromator: {m['incident_monochromator']})"
        elif w and ratio == 0:
            rad = f"{src}K-alpha1 (K-alpha1 only, as the file records)"
        elif w and "K-alpha2" in w:
            rad = f"{src}K-alpha1 + K-alpha2 (the tube's K-alpha doublet; no monochromator in the recorded beam path)"
        else:
            rad = f"{src}K-alpha".strip()
        exp["radiation"] = rad
    if m.get("sample_mode"):
        exp["geometry"] = f"{str(m['sample_mode']).lower()} (sample mode recorded by the instrument)"
    for k in ("goniometer", "sample_stage", "detector", "filters", "optics", "incident_monochromator"):
        if m.get(k):
            exp[k] = m[k]
    md: Dict[str, Any] = {"experiment_type": "Diffraction", "experiment": exp,
                          "sample": {"material": NOT_RECORDED}}
    if m.get("sample_id"):
        md["sample"]["id_in_file"] = m["sample_id"]
    if m.get("file_memo"):
        md["sample"]["memo_in_file"] = m["file_memo"]
    ax = ds.axes[0] if ds.axes else None
    if ax is not None:
        md["xlabel"] = {"2theta": "2θ (deg)", "omega": "ω (deg)"}.get(ax.name, f"{ax.name} (deg)")
    md["ylabel"] = f"Intensity ({m.get('intensity_units') or 'counts'})"
    if m.get("scan"):
        md["scan"] = dict(m["scan"])
    acq = {k: m[k] for k in ("tube_kV", "tube_mA", "goniometer_radius_mm", "temperature_C",
                              "measured_at", "scans_in_file") if m.get(k) is not None}
    if acq:
        md["acquisition"] = acq
    if m.get("notes"):
        md["file_notes"] = list(m["notes"])
    md["metadata_source"] = (f"Read from the {ds.format} file: wavelength, optics and scan as the "
                             "instrument recorded them.")
    return md


def file_metadata(ds: Dataset) -> Dict[str, Any]:
    """SciLink metadata from what the file itself records. Nothing is inferred."""
    if ds.metadata.get("signal_type") == "xrd":
        return _diffraction_metadata(ds)
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


import re as _re

# The person states the radiation only when a value NAMES it: a number under a
# wavelength-type key, an anode line ('Cu Kα', 'MoKa1'), an anode under an
# anode / tube key, 'K-alpha', a value in Å, or a synchrotron. A key merely called
# 'source' or 'target' (a supplier, an objective) says nothing about it.
_WAVE_KEY = _re.compile(r"wavelength|lambda|radiation|anode|x_?ray_source|tube", _re.I)
_ANODE_LINE = _re.compile(r"\b(Cu|Mo|Co|Fe|Cr|Ag)\s*-?\s*K\s*-?\s*(?:alpha|α|a)\s*-?\s*[12]?(?![a-z])", _re.I)
_KALPHA = _re.compile(r"(?<![a-z])k\s*-?\s*(?:alpha|α)(?![a-z])|kα", _re.I)
_ANGSTROM = _re.compile(r"(\d*\.\d+|\d+)\s*(?:å|angstrom)", _re.I)
_SYNCHROTRON = _re.compile(r"synchrotron|beamline", _re.I)
_ANODE_WORDS = {"Cu": "copper", "Mo": "molybdenum", "Co": "cobalt", "Fe": "iron", "Cr": "chromium", "Ag": "silver"}
_GEOM_TEXT = _re.compile(r"bragg|brentano|debye|scherrer|capillary|reflection|transmission|"
                         r"flat[- ]?plate", _re.I)
_LINES = {"Cu": 1.5406, "Mo": 0.7093, "Co": 1.7890, "Fe": 1.9360, "Cr": 2.2897, "Ag": 0.5594}


def _walk(d, path=()):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _walk(v, path + (str(k),))
    elif isinstance(d, (list, tuple)):
        for v in d:
            yield from _walk(v, path)
    else:
        yield path, d


def _given_wavelength(given) -> Tuple[bool, List[float], List[str]]:
    """Whether the person's metadata states the radiation, and what it names."""
    said, nums, anodes = False, [], []
    for path, v in _walk(given or {}):
        wave_key = bool(_WAVE_KEY.search(path[-1] if path else ""))
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)):
            if wave_key and 0.1 < v < 5:
                said = True
                nums.append(float(v))
            continue
        text = str(v)
        found = [m.group(1).capitalize() for m in _ANODE_LINE.finditer(text)]
        ang = [float(m.group(1)) for m in _ANGSTROM.finditer(text) if 0.1 < float(m.group(1)) < 5]
        if wave_key:
            found += [a for a, w in _ANODE_WORDS.items() if _re.search(rf"\b(?:{a}|{w})\b", text, _re.I)]
            try:
                if 0.1 < float(text.strip()) < 5:
                    ang.append(float(text.strip()))
            except ValueError:
                pass
        if found or ang or _KALPHA.search(text) or _SYNCHROTRON.search(text):
            said = True
            anodes += found
            nums += ang
    return said, nums, sorted(set(anodes))


def _diffraction_precedence(given, from_file) -> Tuple[Dict[str, Any], List[str]]:
    """The person's radiation / geometry stand wherever in their metadata they are
    given (top level or ``experiment``, any key name): the file's are then not filled
    in beside them, and a disagreement is reported."""
    exp = dict(from_file.get("experiment") or {})
    conflicts: List[str] = []
    said, nums, anodes = _given_wavelength(given)
    w = exp.get("wavelength")
    if said and w is not None:
        f_ka1 = w.get("K-alpha1") if isinstance(w, dict) else None
        f_anode = None
        rad = str(exp.get("radiation") or w)
        for a in _LINES:
            if rad.startswith(a + " ") or rad.startswith(a + "K") or (
                    f_ka1 is not None and abs(f_ka1 - _LINES[a]) < 0.003):
                f_anode = a
                break
        disagree = (nums and f_ka1 is not None and not any(abs(n - f_ka1) < 0.01 or
                                                         (f_anode and abs(n - _LINES[f_anode]) < 0.01)
                                                         for n in nums)) \
            or (anodes and f_anode and f_anode not in anodes)
        if disagree:
            conflicts.append(f"wavelength: given {', '.join(map(str, nums + anodes))}; "
                             f"file records {exp.get('radiation') or w}")
        for k in ("wavelength", "radiation"):
            exp.pop(k, None)
    g_text = " ".join(str(v) for _, v in _walk(given or {}) if isinstance(v, str))
    if exp.get("geometry") and _GEOM_TEXT.search(g_text):
        f_refl = exp["geometry"].startswith("reflection")
        g_refl = bool(_re.search(r"bragg|brentano|reflection|flat[- ]?plate", g_text, _re.I))
        g_trans = bool(_re.search(r"debye|scherrer|capillary|transmission", g_text, _re.I))
        if (f_refl and g_trans and not g_refl) or (not f_refl and g_refl and not g_trans):
            conflicts.append(f"geometry: given text says {'transmission' if g_trans else 'reflection'}; "
                             f"file records {exp['geometry']}")
        exp.pop("geometry", None)
    out = dict(from_file)
    out["experiment"] = exp
    return out, conflicts


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
    pre_conflicts: List[str] = []
    if from_file.get("experiment_type") == "Diffraction":
        from_file, pre_conflicts = _diffraction_precedence(given, from_file)
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
    conflicts: List[str] = list(pre_conflicts)
    for a, b in _CHECKED:
        g = (given or {}).get(a)
        f = from_file.get(a)
        if isinstance(g, dict) and isinstance(f, dict) and g.get(b) not in (None, "") \
                and f.get(b) not in (None, "") and not _close(g[b], f[b]):
            conflicts.append(f"{a}.{b}: given {g[b]!r}, file records {f[b]!r}")
    return merged, filled, conflicts
