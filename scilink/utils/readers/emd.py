"""EMD files, read with h5py: the open Berkeley/NCEM layout and Thermo Fisher Velox.

Berkeley EMD: any group whose ``emd_group_type`` attribute is 1 holds a
``data`` array and ``dim1`` ... ``dimN`` coordinate vectors (``name`` and
``units`` attributes). A two-element vector on a longer axis is
``[start, start + step]``; units may be bracketed (``[nm]``) and NCEM
writes them with underscores between characters (``[n_m]``).

Velox: a ``Version`` dataset says ``"format": "Velox"``; each acquisition
is ``Data/<Kind>/<id>/{Data, Metadata}`` with the metadata as JSON text
in a uint8 array. Images are stored ``(y, x, frames)``, spectra
``(channels, 1)``. An EDS spectrum image is a photon event stream
(``Data/SpectrumStream``): uint16 channel numbers, a pixel's events
followed by 65535, which moves to the next pixel of the raster, frame
after frame. It is decoded here into
a ``(y, x, channels)`` cube summed over frames and over detectors.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np

from ._layout import canonicalize, uniform_or_values
from ._model import Axis, Dataset, UnsupportedContent, load, stand_in

_PIXEL_MARK = 65535


def _s(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.ndarray):
        if v.shape == ():
            return _s(v.item())
        if v.size == 1:
            return _s(v.ravel()[0])
    return "" if v is None else str(v)


# --------------------------------------------------------------------------
# Berkeley / NCEM
# --------------------------------------------------------------------------

def _emd_units(u: str) -> str:
    u = _s(u).strip()
    if u.startswith("[") and u.endswith("]"):
        u = u[1:-1]
    if "_" in u and all(len(p) <= 1 for p in u.split("_")):
        u = u.replace("_", "")
    return u


def _berkeley_groups(h5) -> list:
    found = []

    def visit(name, obj):
        if getattr(obj, "attrs", None) is not None and "data" in getattr(obj, "keys", lambda: [])():
            if int(np.asarray(obj.attrs.get("emd_group_type", 0)).ravel()[0] or 0) == 1:
                found.append(name)

    h5.visititems(visit)
    return found


def _read_berkeley(h5, path: str, header_only: bool = False) -> List[Dataset]:
    out: List[Dataset] = []
    root_md: Dict[str, Any] = {}
    mic = h5.get("microscope")
    if mic is not None:
        name = _s(mic.attrs.get("name", ""))
        if name:
            root_md["microscope"] = name
        volt = _s(mic.attrs.get("voltage", "")).lower().replace(" ", "")
        if volt.endswith("kv"):
            try:
                root_md["beam_energy_kV"] = float(volt[:-2])
            except ValueError:
                pass
    for gname in _berkeley_groups(h5):
        g = h5[gname]
        data = load(g["data"], header_only)
        axes: List[Axis] = []
        for i in range(data.ndim):
            d = g.get(f"dim{i + 1}")
            n = data.shape[i]
            if d is None:
                axes.append(Axis(size=n))
                continue
            v = np.asarray(d[()], dtype=float).ravel()
            ax = Axis(name=_s(d.attrs.get("name", "")), size=n,
                      units=_emd_units(d.attrs.get("units", "")))
            if v.size == n:
                ax.offset, ax.scale, ax.values = uniform_or_values(v)
            elif v.size >= 2:
                ax.offset, ax.scale = float(v[0]), float(v[1] - v[0])
            elif v.size == 1:
                ax.offset = float(v[0])
            axes.append(ax)
        md = dict(root_md)
        st = _s(g.attrs.get("signal_type", ""))
        if st:
            md["signal_type"] = st
        record_by = _s(g.attrs.get("record_by", "")).lower()
        # HyperSpy's EMD writer marks how the signal was recorded.
        if record_by == "spectrum" and data.ndim >= 1:
            for j, ax in enumerate(axes):
                ax.navigate = j != data.ndim - 1
        elif record_by == "image" and data.ndim >= 2:
            for j, ax in enumerate(axes):
                ax.navigate = j < data.ndim - 2
        data, axes, kind = canonicalize(data, axes)
        title = gname.rsplit("/", 1)[-1]
        if title == "__unnamed__":
            title = ""
        out.append(Dataset(data=data, axes=axes, kind=kind, format="emd",
                           source=path, title=title, metadata=md))
    return out


# --------------------------------------------------------------------------
# Velox
# --------------------------------------------------------------------------

def is_velox(h5) -> bool:
    if "Version" not in h5:
        return False
    try:
        return "velox" in _s(h5["Version"][()]).lower()
    except Exception:
        return False


def _velox_meta(group, frame: int = 0) -> Dict[str, Any]:
    if "Metadata" not in group:
        return {}
    raw = np.asarray(group["Metadata"][:, frame], dtype=np.uint8).tobytes()
    try:
        return json.loads(raw.rstrip(b"\0").decode("utf-8", "replace"))
    except ValueError:
        return {}


def _velox_common(md: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        out["beam_energy_kV"] = float(md["Optics"]["AccelerationVoltage"]) / 1000.0
    except (KeyError, TypeError, ValueError):
        pass
    model = (md.get("Instrument") or {}).get("InstrumentModel")
    if model:
        out["microscope"] = model
    det = (md.get("BinaryResult") or {}).get("Detector")
    if det:
        out["detector"] = det
    return out


_SI_PREFIX = [(1e-9, "nm"), (1e-6, "µm"), (1e-3, "mm")]


def _spatial_axes(md: Dict[str, Any], ny: int, nx: int) -> List[Axis]:
    br = md.get("BinaryResult") or {}
    try:
        sx = float(br["PixelSize"]["width"])
        sy = float(br["PixelSize"]["height"])
        ox = float((br.get("Offset") or {}).get("x", 0.0))
        oy = float((br.get("Offset") or {}).get("y", 0.0))
        unit = br.get("PixelUnitX", "m")
    except (KeyError, TypeError, ValueError):
        return [Axis(name="y", size=ny), Axis(name="x", size=nx)]
    if unit == "m":
        factor, unit = next(((f, u) for f, u in _SI_PREFIX if sx < f * 1000), (1.0, "m"))
        sx, sy, ox, oy = sx / factor, sy / factor, ox / factor, oy / factor
    return [Axis(name="y", size=ny, offset=oy, scale=sy, units=unit),
            Axis(name="x", size=nx, offset=ox, scale=sx, units=unit)]


def _energy_axis(md: Dict[str, Any], n: int) -> Axis:
    dets = md.get("Detectors") or {}
    for d in dets.values():
        if isinstance(d, dict) and "Dispersion" in d and d.get("Enabled", "true") != "false":
            try:
                disp = float(d["Dispersion"])
                off = float(d.get("OffsetEnergy", 0.0))
                return Axis(name="Energy", size=n, offset=off / 1000.0,
                            scale=disp / 1000.0, units="keV")
            except (TypeError, ValueError):
                continue
    return Axis(name="Energy", size=n)


_STREAM_CHUNK = 1 << 22          # stream entries per pass: bounds the working memory


def _chunks(dset):
    """The stream in chunks of entries, each as a flat array (an h5py
    dataset is read chunk by chunk, never whole)."""
    n = int(dset.shape[0])
    for i in range(0, n, _STREAM_CHUNK):
        yield np.asarray(dset[i:i + _STREAM_CHUNK]).ravel()


def _decode_stream(dset, npix: int, nch: int, out: Optional[np.ndarray] = None) -> np.ndarray:
    """``(npix * nch,)`` counts summed over frames, added into ``out``.

    An event belongs to the pixel numbered by the marks before it (a mark
    closes a pixel). The scan covers ``ceil(marks / npix)`` frames; events
    after the mark that closes the last of them belong to no pixel and are
    dropped. One pass over the stream, a chunk at a time: each chunk is
    counted up to its last mark (its bins sorted, which is cheap: the stream
    runs pixel by pixel, and counted as runs), and the events after that
    mark wait for the next chunk; only the events after the stream's final
    mark are decided at the end. Working memory beside the counts is a few
    chunk-sized arrays.
    """
    if out is not None and out.size != npix * nch:
        raise ValueError(f"detector streams disagree on the raster: {out.size} "
                         f"counts against {npix} x {nch}")
    if getattr(dset, "ndim", 1) == 0:
        dset = np.asarray(dset).reshape(-1)
    entries = int(np.prod(dset.shape))
    if out is None:
        # No bin can hold more counts than the stream has entries.
        out = np.zeros(npix * nch, dtype=np.uint32 if entries < 2**32 else np.uint64)
    elif out.dtype == np.uint32 and int(out.max(initial=0)) + entries >= 2**32:
        out = out.astype(np.uint64)

    def add(pix, ch):
        keep = ch < nch
        if not keep.any():
            return
        idx = np.sort((pix[keep] % npix) * nch + ch[keep], kind="stable")
        new = np.empty(idx.size, dtype=bool)
        new[0] = True
        np.not_equal(idx[1:], idx[:-1], out=new[1:])
        starts = np.flatnonzero(new)
        out[idx[starts]] += np.diff(np.append(starts, idx.size)).astype(out.dtype)

    seen = 0                                     # marks so far
    carry = None                                 # events after the last mark seen
    for s in _chunks(dset):
        if carry is not None and carry.size:
            s = np.concatenate([carry, s])
        mark = s == _PIXEL_MARK
        if not mark.any():
            carry = s                            # one pixel still open
            continue
        end = int(np.flatnonzero(mark)[-1]) + 1
        body, carry, mark = s[:end], s[end:], mark[:end]
        pix = (np.cumsum(mark, dtype=np.int64) + seen)[~mark]
        seen += int(np.count_nonzero(mark))
        add(pix, body[~mark].astype(np.int64))
    # Events after the final mark open pixel `seen`: kept only if that pixel
    # lies inside the last frame the marks begin.
    if carry is not None and carry.size:
        last = max(1, -(-seen // npix)) * npix
        if seen < last:
            add(np.full(carry.size, seen, dtype=np.int64), carry.astype(np.int64))
    return out


def _read_velox(h5, path: str, header_only: bool = False) -> List[Dataset]:
    data_root = h5.get("Data")
    if data_root is None:
        raise UnsupportedContent(f"no Data group in Velox file {path}")
    out: List[Dataset] = []
    image_shapes: List[tuple] = []

    for gid, g in (data_root.get("Image") or {}).items():
        arr = load(g["Data"], header_only)       # (y, x, frames)
        md = _velox_meta(g)
        ny, nx = arr.shape[:2]
        image_shapes.append((ny, nx))
        sp = _spatial_axes(md, ny, nx)
        meta = _velox_common(md)
        title = meta.get("detector", "") or gid
        if arr.ndim == 3 and arr.shape[2] > 1:
            stack = np.moveaxis(arr, 2, 0)
            for a in sp:
                a.navigate = False
            frame = Axis(name="frame", size=stack.shape[0], navigate=True)
            # A scanned stack has a frame time; a camera stack, its detector's
            # exposure time.
            ft = (md.get("Scan") or {}).get("FrameTime")
            if ft is None:
                det = (md.get("BinaryResult") or {}).get("DetectorIndex")
                ft = ((md.get("Detectors") or {}).get(f"Detector-{det}") or {}).get("ExposureTime")
            try:
                frame.scale, frame.units, frame.name = float(ft), "s", "time"
            except (TypeError, ValueError):
                pass
            axes = [frame] + sp
            d, axes, kind = canonicalize(stack, axes)
        else:
            img = arr[..., 0] if arr.ndim == 3 else arr
            for a in sp:
                a.navigate = False
            d, axes, kind = canonicalize(img, sp)
        out.append(Dataset(data=d, axes=axes, kind=kind, format="velox",
                           source=path, title=title, metadata=meta))

    for gid, g in (data_root.get("Spectrum") or {}).items():
        arr = np.asarray(g["Data"][()]).ravel()
        md = _velox_meta(g)
        meta = _velox_common(md)
        meta["signal_type"] = "EDS"
        ax = _energy_axis(md, arr.size)
        d, axes, kind = canonicalize(arr, [ax])
        out.append(Dataset(data=d, axes=axes, kind=kind, format="velox",
                           source=path, title=meta.get("detector", "") or gid,
                           metadata=meta))

    streams = data_root.get("SpectrumStream") or {}
    if streams:
        cube = None
        md0: Dict[str, Any] = {}
        ny = nx = None
        for gid, g in streams.items():
            try:
                acq = json.loads(_s(g["AcquisitionSettings"][()]))
            except (KeyError, ValueError):
                acq = {}
            nch = int(acq.get("bincount", 4096))
            md = _velox_meta(g)
            rs = acq.get("RasterScanDefinition") or {}
            if not rs:
                continue                          # a point spectrum's stream
            raster = (int(rs["Height"]), int(rs["Width"]))
            # The raster is the scan field; a spectrum image of a sub-area is
            # sized by the image acquired with it.
            if raster in image_shapes or not image_shapes:
                ny, nx = raster
            else:
                ny, nx = image_shapes[0]
            if not header_only:
                cube = _decode_stream(g["Data"], ny * nx, nch, out=cube)
            elif cube is None:
                # Shape only, for a probe or a metadata read: a zero-memory
                # stand-in, so the event stream is never decoded for them.
                cube = stand_in((ny * nx * nch,), np.uint32)
            md0 = md0 or md
        if cube is not None:
            meta = _velox_common(md0)
            meta["signal_type"] = "EDS"
            meta["summed_over"] = "frames and detectors"
            nch = cube.size // (ny * nx)
            sp = _spatial_axes(md0, ny, nx)
            axes = sp + [_energy_axis(md0, nch)]
            axes[0].navigate = axes[1].navigate = True
            axes[2].navigate = False
            d, axes, kind = canonicalize(cube.reshape(ny, nx, nch), axes)
            out.insert(0, Dataset(data=d, axes=axes, kind=kind, format="velox",
                                  source=path, title="EDS spectrum image",
                                  metadata=meta))

    eels = data_root.get("EelsSpectrumImage") or {}
    if eels:
        si = data_root.get("SpectrumImage") or {}
        md_si = _velox_meta(si[list(si)[0]]) if si else {}
        for gid, g in eels.items():
            arr = load(g["Data"], header_only)
            if arr.ndim != 3:
                continue
            cube = np.transpose(arr, (2, 0, 1))  # stored (x, energy, y)
            ny, nx, ne = cube.shape
            md = md_si or _velox_meta(g)
            try:
                acq = json.loads(bytes(np.asarray(g["AcquisitionMetadata"][:, 0], dtype=np.uint8))
                                 .rstrip(b"\0").decode("utf-8", "replace"))["Data"]
                energy = Axis(name="Energy loss", size=ne, offset=float(acq["offset"]),
                              scale=float(acq["dispersion"]), units="eV")
            except (KeyError, ValueError, TypeError):
                energy = Axis(name="Energy loss", size=ne)
            axes = _spatial_axes(md, ny, nx) + [energy]
            axes[0].navigate = axes[1].navigate = True
            axes[2].navigate = False
            d, axes, kind = canonicalize(cube, axes)
            meta = _velox_common(md)
            meta["signal_type"] = "EELS"
            out.insert(0, Dataset(data=d, axes=axes, kind=kind, format="velox",
                                  source=path,
                                  title=f"EELS spectrum image from {energy.offset:g} eV",
                                  metadata=meta))

    unread = sorted(k for k in data_root.keys()
                    if k not in ("Image", "Spectrum", "SpectrumStream",
                                 "SpectrumImage", "EelsSpectrumImage", "Text"))
    if unread and out:
        out[0].metadata["not_read"] = unread
    if not out:
        kinds = sorted(data_root.keys())
        raise UnsupportedContent(
            f"Velox file {path} holds {kinds}; only images, spectra, EDS "
            "spectrum streams and EELS spectrum images are read")
    return out


def read_open(h5, path: str, header_only: bool = False) -> List[Dataset]:
    if is_velox(h5):
        return _read_velox(h5, path, header_only=header_only)
    out = _read_berkeley(h5, path, header_only=header_only)
    if not out:
        raise UnsupportedContent(f"no EMD signal group (emd_group_type=1) in {path}")
    return out


def read(path: str, header_only: bool = False) -> List[Dataset]:
    import h5py

    with h5py.File(path, "r") as h5:
        return read_open(h5, path, header_only=header_only)
