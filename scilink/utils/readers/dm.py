"""Gatan DigitalMicrograph ``.dm3`` / ``.dm4`` files, read with numpy and struct only.

A DM file is a tree of tags. The header and the tag structure are
big-endian; tag *values* use the byte order the header declares. DM3 and
DM4 differ only in field widths: tag counts, info counts and the root
length are 4 bytes in DM3 and 8 in DM4, and DM4 stores each tag's byte
length after its label. Images live under ``ImageList``, each with
``ImageData/{Data, DataType, Dimensions, Calibrations}`` (dimensions and
calibrations listed fastest-varying first) and ``ImageTags`` (the
instrument record). Large arrays are not read while walking the tree;
their position is kept and only the image data is read.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ._layout import canonicalize
from ._model import Axis, Dataset, UnsupportedContent, stand_in

# Tag value type codes.
_SIMPLE = {
    2: np.dtype("i2"), 3: np.dtype("i4"), 4: np.dtype("u2"), 5: np.dtype("u4"),
    6: np.dtype("f4"), 7: np.dtype("f8"), 8: np.dtype("u1"), 9: np.dtype("i1"),
    10: np.dtype("u1"), 11: np.dtype("i8"), 12: np.dtype("u8"),
}
_STRUCT, _STRING, _ARRAY = 15, 18, 20
_GROUP, _DATA = 20, 21

# Image ``DataType`` codes -> numpy dtype of one pixel.
_IMAGE_DTYPE = {
    1: np.dtype("i2"), 2: np.dtype("f4"), 3: np.dtype("c8"), 6: np.dtype("u1"),
    7: np.dtype("i4"), 9: np.dtype("i1"), 10: np.dtype("u2"), 11: np.dtype("u4"),
    12: np.dtype("f8"), 13: np.dtype("c16"), 14: np.dtype("u1"),
    39: np.dtype("i8"), 40: np.dtype("u8"),
}
_RGB_TYPES = {8, 23}            # four uint8 per pixel (B, G, R, A)
_PACKED_COMPLEX = {5, 27, 28}   # half-plane FFT storage; not unpacked here

# Arrays at least this large are located, not read, while walking the tree.
_LAZY_BYTES = 1 << 16


@dataclass
class _Located:
    """An array kept on disk: read with ``read(fh)``."""
    offset: int
    dtype: np.dtype
    count: int

    def read(self, fh) -> np.ndarray:
        fh.seek(self.offset)
        return np.fromfile(fh, dtype=self.dtype, count=self.count)


class _TagReader:
    def __init__(self, fh):
        self.fh = fh
        self.version = struct.unpack(">i", fh.read(4))[0]
        if self.version not in (3, 4):
            raise ValueError(f"not a DigitalMicrograph 3/4 file (version {self.version})")
        self.wide = self.version == 4
        self._read_len()                          # root length
        little = struct.unpack(">i", fh.read(4))[0] == 1
        self.bo = "<" if little else ">"

    def _read_len(self) -> int:
        n = 8 if self.wide else 4
        return int.from_bytes(self.fh.read(n), "big")

    def read_root(self):
        return self._group()

    def _group(self):
        self.fh.read(2)                           # sorted, open
        ntags = self._read_len()
        items: List[tuple] = []
        for _ in range(ntags):
            ttype = self.fh.read(1)[0]
            lab_len = struct.unpack(">H", self.fh.read(2))[0]
            label = self.fh.read(lab_len).decode("latin-1")
            if self.wide:
                self.fh.read(8)                   # tag byte length
            if ttype == _GROUP:
                items.append((label, self._group()))
            elif ttype == _DATA:
                items.append((label, self._data()))
            else:
                raise ValueError(f"unexpected tag type {ttype} at {self.fh.tell()}")
        if items and all(lab == "" for lab, _ in items):
            return [v for _, v in items]
        out: Dict[str, Any] = {}
        for i, (lab, v) in enumerate(items):
            out[lab or str(i)] = v
        return out

    def _data(self):
        if self.fh.read(4) != b"%%%%":
            raise ValueError(f"bad tag delimiter at {self.fh.tell() - 4}")
        ninfo = self._read_len()
        info = [self._read_len() for _ in range(ninfo)]
        return self._value(info)

    def _scalar(self, code: int):
        dt = _SIMPLE[code].newbyteorder(self.bo)
        v = np.frombuffer(self.fh.read(dt.itemsize), dtype=dt)[0]
        return v.item()

    def _value(self, info: List[int]):
        code = info[0]
        if code in _SIMPLE:
            return self._scalar(code)
        if code == _STRING:
            return self.fh.read(info[1]).decode("latin-1", "replace")
        if code == _STRUCT:
            nfields = info[2]
            types = info[4:4 + 2 * nfields:2]
            return tuple(self._scalar(t) for t in types)
        if code == _ARRAY:
            elem = info[1]
            if elem == _STRUCT:
                nfields = info[3]
                types = info[5:5 + 2 * nfields:2]
                length = info[-1]
                dt = np.dtype([(f"f{i}", _SIMPLE[t].newbyteorder(self.bo))
                               for i, t in enumerate(types)])
                return self._array(dt, length)
            if elem in _SIMPLE:
                dt = _SIMPLE[elem].newbyteorder(self.bo)
                return self._array(dt, info[2])
            raise UnsupportedContent(f"DM array of element type {elem}")
        raise UnsupportedContent(f"DM tag value type {code}")

    def _array(self, dt: np.dtype, length: int):
        nbytes = dt.itemsize * length
        if nbytes >= _LAZY_BYTES:
            loc = _Located(self.fh.tell(), dt, length)
            self.fh.seek(nbytes, 1)
            return loc
        # Struct arrays stay structured: image data stored as an array of
        # (real, imag) structs must keep its bytes for the pixel decoding.
        return np.frombuffer(self.fh.read(nbytes), dtype=dt)


def _text(v) -> str:
    """DM stores short strings as uint16 arrays (UTF-16)."""
    if isinstance(v, str):
        return v
    if isinstance(v, np.ndarray) and v.dtype.kind == "u" and v.dtype.itemsize == 2:
        return v.astype("<u2").tobytes().decode("utf-16-le", "replace").rstrip("\x00")
    if isinstance(v, np.ndarray) and v.dtype.itemsize == 1:
        return v.tobytes().decode("latin-1", "replace").rstrip("\x00")
    return "" if v is None else str(v)


def _get(tree, *path, default=None):
    cur = tree
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        elif isinstance(cur, list) and isinstance(p, int) and p < len(cur):
            cur = cur[p]
        else:
            return default
    return cur


def _metadata(tags: Dict[str, Any]) -> Dict[str, Any]:
    md: Dict[str, Any] = {}
    sig = _text(_get(tags, "Meta Data", "Signal"))
    if sig:
        md["signal_type"] = sig
    fmt = _text(_get(tags, "Meta Data", "Format"))
    if fmt:
        md["dm_format"] = fmt
    volt = _get(tags, "Microscope Info", "Voltage")
    if isinstance(volt, (int, float)) and volt > 0:
        md["beam_energy_kV"] = round(float(volt) / 1000.0, 3)
    for key, out in (("Name", "microscope"), ("Operation Mode", "operation_mode"),
                     ("Illumination Mode", "illumination_mode")):
        v = _text(_get(tags, "Microscope Info", key))
        if v:
            md[out] = v
    mag = _get(tags, "Microscope Info", "Indicated Magnification")
    if isinstance(mag, (int, float)) and mag > 0:
        md["magnification"] = float(mag)
    for key, out in (("Collection semi-angle (mrad)", "collection_semi_angle_mrad"),
                     ("Convergence semi-angle (mrad)", "convergence_semi_angle_mrad")):
        v = _get(tags, "EELS", "Experimental Conditions", key)
        if isinstance(v, (int, float)):
            md[out] = float(v)
    exp = _get(tags, "DataBar", "Exposure Time (s)")
    if isinstance(exp, (int, float)):
        md["exposure_s"] = float(exp)
    return md


def _image(fh, entry: Dict[str, Any], source: str, bo: str,
           header_only: bool = False) -> Optional[Dataset]:
    idata = entry.get("ImageData") if isinstance(entry, dict) else None
    if not isinstance(idata, dict) or "Data" not in idata:
        return None
    dims = idata.get("Dimensions") or []
    if isinstance(dims, dict):
        dims = list(dims.values())
    dims = [int(d) for d in dims]
    code = int(idata.get("DataType", 0))
    raw = idata["Data"]
    if code in _PACKED_COMPLEX:
        raise UnsupportedContent(
            "DM packed-complex FFT image; export the FFT as complex or the "
            "real-space image instead")
    if header_only and isinstance(raw, _Located):
        # Shape and type from the tags; the array stays on disk.
        nbytes = raw.count * raw.dtype.itemsize
        if code in _RGB_TYPES:
            data = stand_in(tuple(reversed(dims)) + (3,), np.uint8)
            rgb = True
        elif code in _IMAGE_DTYPE:
            dt = np.dtype(bool) if code == 14 else _IMAGE_DTYPE[code].newbyteorder("=")
            n = int(np.prod(dims)) if dims else nbytes // _IMAGE_DTYPE[code].itemsize
            if nbytes // _IMAGE_DTYPE[code].itemsize != n:
                raise UnsupportedContent(
                    f"DM image data has {nbytes // _IMAGE_DTYPE[code].itemsize} values "
                    f"for dimensions {dims}")
            data = stand_in(tuple(reversed(dims)) if dims else (n,), dt)
            rgb = False
        else:
            raise UnsupportedContent(f"DM image data type {code}")
        return _finish_image(entry, idata, dims, data, rgb, source)
    raw = raw.read(fh) if isinstance(raw, _Located) else np.asarray(raw)
    buf = raw.tobytes()
    if code in _RGB_TYPES:
        px = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 4)[:, [2, 1, 0]]
        data = px.reshape(tuple(reversed(dims)) + (3,))
        rgb = True
    elif code in _IMAGE_DTYPE:
        dt = _IMAGE_DTYPE[code].newbyteorder(bo)
        data = np.frombuffer(buf, dtype=dt)
        if code == 14:
            data = data.astype(bool)
        n = int(np.prod(dims)) if dims else data.size
        if data.size != n:
            raise UnsupportedContent(
                f"DM image data has {data.size} values for dimensions {dims}")
        data = data.reshape(tuple(reversed(dims))).astype(data.dtype.newbyteorder("="))
        rgb = False
    else:
        raise UnsupportedContent(f"DM image data type {code}")
    return _finish_image(entry, idata, dims, data, rgb, source)


def _finish_image(entry, idata, dims, data, rgb, source) -> Dataset:
    """Axes, metadata and canonical order for an image's array (or stand-in)."""
    cal = _get(idata, "Calibrations", "Dimension") or []
    if isinstance(cal, dict):
        cal = list(cal.values())
    axes: List[Axis] = []
    for i, n in enumerate(dims):
        c = cal[i] if i < len(cal) and isinstance(cal[i], dict) else {}
        scale = float(c.get("Scale", 1.0) or 1.0)
        origin = float(c.get("Origin", 0.0) or 0.0)
        axes.append(Axis(name="", size=n, offset=-origin * scale, scale=scale,
                         units=_text(c.get("Units"))))
    axes = list(reversed(axes))                 # to C order, matching ``data``

    tags = entry.get("ImageTags") or {}
    md = _metadata(tags)
    bright = _get(idata, "Calibrations", "Brightness") or {}
    if isinstance(bright, dict) and _text(bright.get("Units")):
        md["intensity_units"] = _text(bright.get("Units"))
    title = _text(entry.get("Name"))

    if rgb:
        for a, nm in zip(axes, ("y", "x")):
            a.name, a.navigate = nm, False
        axes.append(Axis(name="channel", size=3, units="", navigate=False))
        md["rgb"] = True
        return Dataset(data=data, axes=axes, kind="image", format="dm",
                       source=source, title=title, metadata=md)

    _name_axes(axes, md)
    data, axes, kind = canonicalize(data, axes)
    return Dataset(data=data, axes=axes, kind=kind, format="dm",
                   source=source, title=title, metadata=md)


def _name_axes(axes: List[Axis], md: Dict[str, Any]) -> None:
    """DM axes carry units but no names; name them from the units and signal type."""
    from ._layout import is_spatial, is_spectral

    spatial = [a for a in axes if is_spatial(a)]
    for a, nm in zip(reversed(spatial), ("x", "y", "z")):
        a.name = nm
    sig = (md.get("signal_type") or "").lower()
    for a in axes:
        if a.name:
            continue
        u = (a.units or "").lower()
        if u in ("ev", "kev", "mev"):
            a.name = "Energy loss" if "eels" in sig else "Energy"
        elif u in ("nm", "µm", "um") and ("cl" in sig or "cathodo" in sig):
            a.name = "Wavelength"
        elif is_spectral(a):
            a.name = "Signal"


def read(path: str, header_only: bool = False) -> List[Dataset]:
    with open(path, "rb") as fh:
        reader = _TagReader(fh)
        root = reader.read_root()
        images = root.get("ImageList") if isinstance(root, dict) else None
        if isinstance(images, dict):
            images = list(images.values())
        out: List[Dataset] = []
        for entry in images or []:
            ds = _image(fh, entry, path, reader.bo, header_only)
            if ds is not None:
                out.append(ds)
    if not out:
        raise UnsupportedContent(f"no image data in {path}")
    # DM puts a small RGB thumbnail first; drop it when real data follows.
    real = [d for d in out if not d.metadata.get("rgb")]
    if real and len(real) < len(out):
        out = real
    return out
