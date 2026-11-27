"""X-ray diffraction instrument files: the pattern and what the instrument recorded about it.

==================  ==============================================================
suffix              format
==================  ==============================================================
``.xrdml``          Malvern Panalytical XRDML (schema 1.x / 2.x)
``.raw``            Bruker DIFFRAC RAW, versions 3 (``RAW1.01``) and 4 (``RAW4.00``)
``.brml``           Bruker DIFFRAC.SUITE (zip of XML; ``RawData*.xml``)
``.ras``            Rigaku RAS text (SmartLab Guidance / SmartLab Studio)
``.rasx``           Rigaku SmartLab Studio II (zip; ``Profile*.txt`` + conditions XML)
==================  ==============================================================

``.raw`` and ``.ras`` are claimed only when the content says so (Bruker
``RAW`` magic, Rigaku ``*RAS_DATA_START``): other instruments and image
formats use the same suffixes. Bruker RAW versions 1 and 2 are recognised
and refused by name.

Every scan (XRDML ``scan``, RAW range, BRML ``RawData`` file, RAS block, RASX
profile) becomes one ``spectrum`` :class:`Dataset` on its scan axis, in file
order. Intensities are the instrument's counts with any recorded attenuator
factor applied; nothing is smoothed, stripped or rescaled.

``metadata`` holds only what the file records, under these keys (absent when
the file is silent): ``signal_type`` ('xrd'), ``instrument``, ``anode``,
``wavelength_angstrom`` ({'K-alpha1', 'K-alpha2', 'K-alpha2/K-alpha1'}),
``incident_monochromator``, ``filters``, ``sample_mode`` (XRDML's
Reflection / Transmission), ``goniometer``, ``sample_stage``, ``optics``,
``detector``, ``tube_kV``, ``tube_mA``, ``goniometer_radius_mm``,
``scan`` (axis, mode, start, end, step, points, time per step),
``temperature_C``, ``sample_id``, ``measured_at``, ``intensity_units``,
``scans_in_file``, ``notes``. The K-alpha2/K-alpha1 ratio is the one the
file records, except that an incident-beam monochromator recorded in the
beam path sets it to 0 (the note says so), because XRDML and BRML keep the
tube's ratio in the header whatever the optics.
"""

from __future__ import annotations

import re
import struct
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ._model import Axis, Dataset, UnsupportedContent

SUFFIXES = frozenset({".xrdml", ".raw", ".brml", ".ras", ".rasx"})

_ANODE_BY_Z = {24: "Cr", 26: "Fe", 27: "Co", 29: "Cu", 42: "Mo", 47: "Ag", 74: "W"}


# ------------------------------------------------------------------ sniffing
def sniff(path) -> Optional[str]:
    """'xrdml' | 'bruker_raw' | 'brml' | 'ras' | 'rasx' when the file is one, else None.

    Decided by content, never by suffix alone; no array is read."""
    p = Path(str(path))
    suf = p.suffix.lower()
    if suf not in SUFFIXES or not p.is_file():
        return None
    try:
        with open(p, "rb") as f:
            head = f.read(4096)
    except OSError:
        return None
    if suf == ".raw":
        return "bruker_raw" if head[:4] in (b"RAW1", b"RAW4", b"RAW2", b"RAW ") else None
    if suf == ".ras":
        return "ras" if head.lstrip(b"\xef\xbb\xbf \r\n\t").startswith(b"*RAS_DATA_START") else None
    if suf == ".xrdml":
        return "xrdml" if b"xrdMeasurement" in head else None
    if head[:2] != b"PK":
        return None
    try:
        with zipfile.ZipFile(p) as z:
            names = z.namelist()
    except (zipfile.BadZipFile, OSError):
        return None
    if suf == ".rasx" and any(re.search(r"(^|/)Profile\d*\.txt$", n) for n in names):
        return "rasx"
    if suf == ".brml" and any(re.search(r"(^|/)RawData\d*\.xml$", n) for n in names):
        return "brml"
    return None


def read(path, header_only: bool = False) -> List[Dataset]:
    kind = sniff(path)
    if kind is None:
        raise UnsupportedContent(f"{Path(str(path)).name} is not a recognised diffraction file")
    reader = {"xrdml": _read_xrdml, "bruker_raw": _read_bruker_raw, "brml": _read_brml,
              "ras": _read_ras, "rasx": _read_rasx}[kind]
    found = reader(str(path))
    if not found:
        raise UnsupportedContent(f"{Path(str(path)).name} holds no scan")
    n = len(found)
    for i, ds in enumerate(found):
        ds.metadata["scans_in_file"] = n
        if n > 1:
            ds.metadata.setdefault("notes", []).append(
                f"the file holds {n} scans; this is scan {i + 1}. Scans of one "
                "sample over different angle ranges are parts of one pattern; scans at "
                "different temperatures / times are a series")
    if header_only:
        from ._model import stand_in
        for ds in found:
            ds.data = stand_in(ds.data.shape, ds.data.dtype)
    return found


# ------------------------------------------------------------------ helpers
def _f(v) -> Optional[float]:
    try:
        x = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _clean(s) -> str:
    return re.sub(r"\s+", " ", str(s or "").replace("\x00", "")).strip()


def _wavelength(ka1, ka2, ratio) -> Optional[Dict[str, float]]:
    if ka1 is None or not 0.1 < ka1 < 5.0:
        return None
    w = {"K-alpha1": round(ka1, 6)}
    if ka2 is not None and 0.1 < ka2 < 5.0 and abs(ka2 - ka1) > 1e-6:
        w["K-alpha2"] = round(ka2, 6)
        w["K-alpha2/K-alpha1"] = round(ratio, 4) if ratio is not None else None
        if w["K-alpha2/K-alpha1"] is None:
            del w["K-alpha2/K-alpha1"]
    return w


def _monochromated(md: Dict[str, Any]) -> None:
    """An incident-beam monochromator passes K-alpha1 only: ratio 0, said in a note."""
    w = md.get("wavelength_angstrom")
    if md.get("incident_monochromator") and isinstance(w, dict) and w.get("K-alpha2/K-alpha1"):
        md.setdefault("notes", []).append(
            f"the header gives K-alpha2/K-alpha1 = {w['K-alpha2/K-alpha1']}, but the beam path "
            "records an incident-beam monochromator, which passes K-alpha1 only: ratio set to 0")
        w["K-alpha2/K-alpha1"] = 0.0


def _axis(x: np.ndarray, name: str) -> Axis:
    x = np.asarray(x, float)
    if x.size >= 2:
        d = np.diff(x)
        if np.allclose(d, d[0], rtol=1e-6, atol=1e-9):
            return Axis(name=name, size=x.size, offset=float(x[0]), scale=float(d[0]),
                        units="deg", navigate=False)
    return Axis(name=name, size=x.size, units="deg", navigate=False, values=x)


_AXIS_NAMES = {"2theta": "2theta", "twotheta": "2theta", "gonio": "2theta",
               "2theta-omega": "2theta", "omega-2theta": "2theta", "twothetaomega": "2theta",
               "theta/2-theta": "2theta", "2-theta": "2theta", "2theta/omega": "2theta",
               "2-theta/omega": "2theta", "locked coupled": "2theta", "unlocked coupled": "2theta",
               "detector scan": "2theta", "omega": "omega", "theta": "omega", "phi": "phi",
               "chi": "chi", "psi": "psi"}


def _axis_name(scan_axis: str) -> str:
    k = re.sub(r"\s+", " ", (scan_axis or "").strip().lower())
    return _AXIS_NAMES.get(k, _AXIS_NAMES.get(k.replace(" ", ""), k or "2theta"))


def _dataset(x, y, fmt, path, md, axis_label, title="") -> Dataset:
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    n = min(x.size, y.size)
    if x.size != y.size:
        md.setdefault("notes", []).append(
            f"the file gives {x.size} angles and {y.size} intensities; the first {n} are used")
    return Dataset(data=y[:n], axes=[_axis(x[:n], _axis_name(axis_label))], kind="spectrum",
                   format=fmt, source=path, title=title, metadata=md)


# ------------------------------------------------------------------ XRDML
def _read_xrdml(path: str) -> List[Dataset]:
    root = ET.parse(path).getroot()
    ns = root.tag[1:].split("}")[0] if root.tag.startswith("{") else ""

    def q(tag):
        return f"{{{ns}}}{tag}" if ns else tag

    def text(el, *tags):
        for t in tags:
            el = el.find(q(t)) if el is not None else None
        return el.text if el is not None and el.text is not None else None

    comments = [_clean(e.text) for e in root.iter(q("entry")) if e.text and e.text.strip()]
    cdict = {}
    for c in comments:
        for part in re.split(r";\s*", c):
            if "=" in part:
                k, v = part.split("=", 1)
                cdict.setdefault(k.strip(), v.strip())
    sample = root.find(q("sample"))
    out: List[Dataset] = []
    for meas in root.findall(q("xrdMeasurement")):
        md: Dict[str, Any] = {"signal_type": "xrd"}
        wl = meas.find(q("usedWavelength"))
        if wl is not None:
            w = _wavelength(_f(text(wl, "kAlpha1")), _f(text(wl, "kAlpha2")),
                            _f(text(wl, "ratioKAlpha2KAlpha1")))
            if w:
                md["wavelength_angstrom"] = w
        inc = meas.find(q("incidentBeamPath"))
        dif = meas.find(q("diffractedBeamPath"))
        tube = inc.find(q("xRayTube")) if inc is not None else None
        if tube is not None:
            if text(tube, "anodeMaterial"):
                md["anode"] = _clean(text(tube, "anodeMaterial"))
            if _f(text(tube, "tension")) is not None:
                md["tube_kV"] = _f(text(tube, "tension"))
            if _f(text(tube, "current")) is not None:
                md["tube_mA"] = _f(text(tube, "current"))
        if _f(text(inc, "radius")) is not None:
            md["goniometer_radius_mm"] = _f(text(inc, "radius"))
        mono = inc.find(q("monochromator")) if inc is not None else None
        if mono is not None:
            md["incident_monochromator"] = _clean(mono.get("name") or "monochromator") or "monochromator"
        optics, filters = [], []
        # 'source side' / 'detector side', not 'incident' / 'diffracted': free-text
        # readers treat "incident(-beam) ... monochromator" as K-alpha1-only optics
        for side, path_el in (("source side", inc), ("detector side", dif)):
            if path_el is None:
                continue
            for child in path_el:
                tag = child.tag.split("}")[-1]
                if tag in ("radius", "xRayTube"):
                    continue
                name = _clean(child.get("name") or "")
                if tag == "filter":
                    filters.append(f"{side}: {name or text(child, 'material') or 'filter'}")
                elif tag == "detector":
                    md["detector"] = name or "detector"
                elif tag == "monochromator" and side == "detector side":
                    optics.append(f"detector side monochromator (K-beta / fluorescence): {name}")
                elif tag != "monochromator" and name:
                    optics.append(f"{side} {tag}: {name}")
        if filters:
            md["filters"] = filters
        if optics:
            md["optics"] = optics
        if meas.get("sampleMode"):
            md["sample_mode"] = meas.get("sampleMode")
        mv = meas.find(q("sampleMovement"))
        if mv is not None:
            t = (mv.get("{http://www.w3.org/2001/XMLSchema-instance}type") or "").replace("Type", "")
            if t:
                md["sample_stage"] = re.sub(r"(?<!^)(?=[A-Z])", " ", t).lower()
        if cdict.get("Sample stage"):
            md["sample_stage"] = cdict["Sample stage"] + (f" ({md['sample_stage']})" if md.get("sample_stage") else "")
        if cdict.get("Goniometer"):
            md["goniometer"] = cdict["Goniometer"]
        if sample is not None:
            sid = _clean(text(sample, "id")) or _clean(text(sample, "name"))
            if sid:
                md["sample_id"] = sid
        _monochromated(md)
        for scan in meas.findall(q("scan")):
            smd = {k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v)
                   for k, v in md.items()}
            hdr = scan.find(q("header"))
            src = hdr.find(q("source")) if hdr is not None else None
            inst = [cdict.get("Diffractometer system")]
            if src is not None:
                inst += [text(src, "instrumentControlSoftware"), text(src, "applicationSoftware")]
            inst = [i for i in dict.fromkeys(_clean(i) for i in inst if i) if i]
            if inst:
                smd["instrument"] = "Malvern Panalytical " + " / ".join(inst)
            if hdr is not None and text(hdr, "startTimeStamp"):
                smd["measured_at"] = _clean(text(hdr, "startTimeStamp"))
            dp = scan.find(q("dataPoints"))
            if dp is None:
                continue
            yel = dp.find(q("intensities"))
            raw_counts = yel is None
            if yel is None:
                yel = dp.find(q("counts"))
            if yel is None or not (yel.text or "").strip():
                continue
            y = np.array(yel.text.split(), float)
            smd["intensity_units"] = yel.get("unit") or "counts"
            att = dp.find(q("beamAttenuationFactors"))
            if att is not None and (att.text or "").strip():
                a = np.array(att.text.split(), float)
                if a.size == y.size and not np.allclose(a, 1.0):
                    if raw_counts:   # <counts> are as detected; <intensities> are already corrected
                        y = y * a
                        smd.setdefault("notes", []).append(
                            "counts multiplied by the recorded beam-attenuation factors")
                    else:
                        smd.setdefault("notes", []).append(
                            "the file records beam-attenuation factors; its <intensities> are taken "
                            "as already corrected for them")
            positions = {}
            for pos in dp.findall(q("positions")):
                ax = pos.get("axis") or ""
                lst = pos.find(q("listPositions"))
                if lst is not None and (lst.text or "").strip():
                    positions[ax] = np.array(lst.text.split(), float)
                else:
                    a0, a1 = _f(text(pos, "startPosition")), _f(text(pos, "endPosition"))
                    if a0 is not None and a1 is not None:
                        positions[ax] = np.linspace(a0, a1, y.size)
                    elif _f(text(pos, "commonPosition")) is not None:
                        positions[ax] = np.full(y.size, _f(text(pos, "commonPosition")))
            scan_axis = scan.get("scanAxis") or ""
            name = _axis_name(scan_axis)
            key = {"2theta": "2Theta", "omega": "Omega"}.get(name, scan_axis)
            x = positions.get(key)
            if x is None or np.ptp(x) == 0:
                moving = [k for k, v in positions.items() if np.ptp(v) > 0]
                if not moving:
                    continue
                key = "2Theta" if "2Theta" in moving else moving[0]
                x = positions[key]
                name = _axis_name(key)
            smd["scan"] = {"axis": scan_axis or key, "mode": scan.get("mode"),
                           "start": round(float(x[0]), 6), "end": round(float(x[-1]), 6),
                           "points": int(y.size)}
            if x.size > 1:
                smd["scan"]["step"] = round(float((x[-1] - x[0]) / (x.size - 1)), 6)
            ct = _f(text(dp, "commonCountingTime"))
            if ct is not None:
                smd["scan"]["time_per_step_s"] = ct
            smd["scan"] = {k: v for k, v in smd["scan"].items() if v is not None}
            out.append(_dataset(x, y, "xrdml", path, smd, key,
                                title=f"scan {scan.get('appendNumber', len(out))}"))
    return out


# ------------------------------------------------------------------ Bruker RAW
def _read_bruker_raw(path: str) -> List[Dataset]:
    b = Path(path).read_bytes()
    if b[:7] == b"RAW1.01":
        return _raw_v3(b, path)
    if b[:7] == b"RAW4.00":
        return _raw_v4(b, path)
    ver = "1" if b[:4] == b"RAW " else "2"
    raise UnsupportedContent(
        f"{Path(path).name} is a Bruker RAW version {ver} file, which SciLink does not read; "
        "export it from DIFFRAC.EVA as RAW version 4, .brml, or a two-column .xy")


def _cstr(b: bytes, o: int, n: int) -> str:
    return _clean(b[o:o + n].split(b"\x00")[0].decode("latin-1", "replace"))


def _raw_v3(b: bytes, path: str) -> List[Dataset]:
    """RAW1.01: a 712-byte file header, then per range a 304-byte header, a
    supplementary block whose length the range header gives, and float32 counts."""
    n_ranges = struct.unpack_from("<I", b, 12)[0]
    head: Dict[str, Any] = {"signal_type": "xrd", "instrument": "Bruker (DIFFRAC RAW v3)"}
    anode = _cstr(b, 608, 4)
    if anode:
        head["anode"] = anode
    ka1, ka2, _kb, ratio = struct.unpack_from("<4d", b, 624)
    w = _wavelength(ka1, ka2, ratio)
    if w:
        head["wavelength_angstrom"] = w
    r = struct.unpack_from("<f", b, 564)[0]
    if 50 < r < 1000:
        head["goniometer_radius_mm"] = round(float(r), 2)
    sid = _cstr(b, 326, 60)
    if sid:
        head["sample_id"] = sid
    date, time = _cstr(b, 16, 10), _cstr(b, 26, 10)
    if date:
        head["measured_at"] = f"{date} {time}".strip()
    out, o = [], 712
    for i in range(n_ranges):
        if o + 304 > len(b):
            break
        hl, steps = struct.unpack_from("<II", b, o)
        if not steps:
            break
        start = struct.unpack_from("<d", b, o + 16)[0]
        step = struct.unpack_from("<d", b, o + 176)[0]
        t_k = struct.unpack_from("<f", b, o + 212)[0]
        kv, ma = struct.unpack_from("<II", b, o + 224)
        sup = struct.unpack_from("<I", b, o + 256)[0]
        if not 0 <= sup < 1 << 16 or o + hl + sup + 4 * steps > len(b):
            raise UnsupportedContent(f"{Path(path).name}: range {i + 1} header is inconsistent")
        y = np.frombuffer(b, "<f4", steps, o + hl + sup).astype(float)
        md = {k: (dict(v) if isinstance(v, dict) else v) for k, v in head.items()}
        md["intensity_units"] = "counts"
        md["scan"] = {"axis": "2theta", "start": round(start, 6),
                      "end": round(start + step * (steps - 1), 6), "step": round(step, 6),
                      "points": int(steps)}
        if 1 < kv < 200 and 0 < ma < 2000:
            md["tube_kV"], md["tube_mA"] = float(kv), float(ma)
        if 3 < t_k < 3000:
            md["temperature_C"] = round(float(t_k) - 273.15, 2)
        out.append(_dataset(start + step * np.arange(steps), y, "bruker_raw", path, md,
                            "2theta", title=f"range {i + 1}"))
        o += hl + sup + 4 * steps
    if o != len(b) and out:
        out[0].metadata.setdefault("notes", []).append(
            f"{len(b) - o} bytes after the last range were not read")
    return out


def _raw_v4(b: bytes, path: str) -> List[Dataset]:
    """RAW4.00: file header, tagged segments (10 = name/value text, 30 = X-ray
    source, 60 = drive), then per range a 160-byte header, its own segments
    and the counts."""
    head: Dict[str, Any] = {"signal_type": "xrd", "instrument": "Bruker (DIFFRAC RAW v4)"}
    date, time = _cstr(b, 12, 12), _cstr(b, 24, 10)
    if date:
        head["measured_at"] = f"{date} {time}".strip()
    texts: Dict[str, str] = {}
    o = 61
    while o + 8 <= len(b):
        seg = struct.unpack_from("<I", b, o)[0]
        if seg in (0, 160):
            break
        slen = max(struct.unpack_from("<I", b, o + 4)[0], 8)
        if seg == 10 and slen > 36:
            texts[_cstr(b, o + 12, 24)] = _cstr(b, o + 36, slen - 36)
        elif seg == 30 and slen >= 120:
            _ka, ka1, ka2, _kb, ratio = struct.unpack_from("<5d", b, o + 72)
            w = _wavelength(ka1, ka2, ratio)
            if w:
                head["wavelength_angstrom"] = w
            anode = _cstr(b, o + 116, 4)
            if anode:
                head["anode"] = anode
        o += slen
    for k in ("SampleID", "SAMPLEID", "Sample"):
        if texts.get(k):
            head["sample_id"] = texts[k]
            break
    out = []
    while o + 160 <= len(b):
        seg = struct.unpack_from("<I", b, o)[0]
        if seg not in (0, 160):
            break
        scan_type = _cstr(b, o + 32, 24)
        start, step = struct.unpack_from("<dd", b, o + 72)
        steps = struct.unpack_from("<I", b, o + 88)[0]
        step_time = struct.unpack_from("<f", b, o + 92)[0]
        kv, ma = struct.unpack_from("<ff", b, o + 100)
        datum, hdr = struct.unpack_from("<II", b, o + 136)
        d0 = o + 160 + hdr
        if datum < 4 or d0 + datum * steps > len(b) or not steps:
            if out:     # padding or trailing bytes after the last good range
                break
            raise UnsupportedContent(f"{Path(path).name}: range {len(out) + 1} header is inconsistent")
        t = struct.unpack_from("<f", b, d0 - 16)[0] if hdr >= 16 else 0.0
        if datum == 4:
            y = np.frombuffer(b, "<f4", steps, d0).astype(float)
        else:
            y = np.frombuffer(b, np.dtype([("y", "<f4"), ("rest", f"V{datum - 4}")]), steps, d0)["y"].astype(float)
        md = {k: (dict(v) if isinstance(v, dict) else v) for k, v in head.items()}
        if datum != 4:
            md.setdefault("notes", []).append(
                f"each step stores {datum} bytes; the first 4 (float32) are read as the counts")
        md["intensity_units"] = "counts"
        md["scan"] = {"axis": scan_type or "2theta", "start": round(start, 6),
                      "end": round(start + step * (steps - 1), 6), "step": round(step, 6),
                      "points": int(steps)}
        if 0 < step_time < 1e8:
            md["scan"]["time_per_step_s"] = round(float(step_time) / 1000.0, 4)
        if 1 < kv < 200:
            md["tube_kV"] = round(float(kv), 2)
        if 0 < ma < 2000:
            md["tube_mA"] = round(float(ma), 2)
        if 7 < t < 3000:   # Kelvin; one RAW4 writer leaves 9999 or 0 here
            if t != 9999:
                md["temperature_C"] = round(float(t) - 273.15, 2)
        name = "2theta" if scan_type in ("", "Locked Coupled", "Unlocked Coupled", "Detector Scan") \
            else scan_type
        if name != "2theta":
            md.setdefault("notes", []).append(
                f"scan type '{scan_type}': the axis is the scanned drive, not necessarily 2-theta")
        out.append(_dataset(start + step * np.arange(steps), y, "bruker_raw", path, md, name,
                            title=f"range {len(out) + 1}"))
        o = d0 + datum * steps
    if o != len(b) and out:
        out[0].metadata.setdefault("notes", []).append(
            f"{len(b) - o} bytes after the last range were not read")
    return out


# ------------------------------------------------------------------ Bruker BRML
def _local(tag: str) -> str:
    return tag.split("}")[-1]


def _read_brml(path: str) -> List[Dataset]:
    out = []
    with zipfile.ZipFile(path) as z:
        names = sorted((n for n in z.namelist() if re.search(r"(^|/)RawData\d*\.xml$", n)),
                       key=lambda n: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", n)])
        for n in names:
            root = ET.fromstring(z.read(n))
            ds = _brml_rawdata(root, path, n)
            if ds is not None:
                out.append(ds)
    return out


def _brml_rawdata(root, path: str, member: str) -> Optional[Dataset]:
    def first(el, name):
        for e in el.iter():
            if _local(e.tag) == name:
                return e
        return None

    def val(el):
        if el is None:
            return None
        return el.get("Value") if el.get("Value") is not None else el.text

    route = None
    for r in root.iter():
        if _local(r.tag) == "DataRoute":
            if route is None or (r.get("RouteFlag") or "").lower() == "measured":
                route = r
                if (r.get("RouteFlag") or "").lower() == "measured":
                    break
    if route is None:
        return None
    rows = [e.text for e in route if _local(e.tag) == "Datum" and e.text]
    if not rows:
        return None
    table = np.array([[_f(v) if _f(v) is not None else np.nan for v in r.split(",")] for r in rows],
                     float)
    views: Dict[str, int] = {}
    count_col = None
    for v in route.iter():
        if _local(v.tag) != "RawDataView":
            continue
        start = int(v.get("Start", -1))
        logic = v.get("LogicName")
        if logic:
            views[logic] = start
        varying = first(v, "Varying")
        if varying is not None and varying.get("LogicName"):
            views[varying.get("LogicName")] = start
        rec = first(v, "Recording")
        if rec is not None and count_col is None and (rec.get("Category") or "") == "Count":
            count_col = start
            if int(v.get("Length", 1)) > 1:
                raise UnsupportedContent(
                    f"{Path(path).name} ({member}): a stationary-detector frame "
                    f"({v.get('Length')} detector channels per point), not a scan; its "
                    "channel-to-angle mapping is not read — export the pattern from DIFFRAC.EVA "
                    "as .xy or RAW version 4")
    info = first(route, "ScanInformation")
    axes_info = [a for a in (info.iter() if info is not None else []) if _local(a.tag) == "ScanAxisInfo"]
    md: Dict[str, Any] = {"signal_type": "xrd", "instrument": "Bruker (DIFFRAC.SUITE .brml)",
                          "intensity_units": "counts"}
    if count_col is None or "ScanAxes" not in views:
        raise UnsupportedContent(
            f"{Path(path).name} ({member}): no counts / scan-axis columns declared; this .brml "
            "layout is not read — export the scan from DIFFRAC.EVA as .xy or RAW version 4")
    x = table[:, views["ScanAxes"]]
    axis_label = axes_info[0].get("AxisId") if axes_info else "TwoTheta"
    y = table[:, count_col]
    if "AbsorptionFactor" in views:
        a = table[:, views["AbsorptionFactor"]]
        if np.all(np.isfinite(a)) and not np.allclose(a, 1.0):
            y = y * a
            md.setdefault("notes", []).append("counts multiplied by the recorded absorption (attenuator) factors")
    if info is not None:
        scan = {"axis": info.get("VisibleName") or info.get("ScanName") or axis_label,
                "start": round(float(x[0]), 6), "end": round(float(x[-1]), 6), "points": int(len(x))}
        inc = first(axes_info[0], "Increment") if axes_info else None
        if _f(val(inc)) is not None:
            scan["step"] = _f(val(inc))
        tps = _f(val(first(info, "TimePerStep")))
        if tps is not None:
            scan["time_per_step_s"] = tps
        mode = val(first(info, "ScanModeVisibleName")) or val(first(info, "ScanMode"))
        if mode:
            scan["mode"] = _clean(mode)
        md["scan"] = scan
    fixed = first(root, "FixedInformation")
    if fixed is not None:
        w = _wavelength(_f(val(first(fixed, "WaveLengthAlpha1"))), _f(val(first(fixed, "WaveLengthAlpha2"))),
                        _f(val(first(fixed, "WaveLengthRatio"))))
        if w is None:
            w = _wavelength(_f(val(first(fixed, "WaveLength"))), None, None)
        if w:
            md["wavelength_angstrom"] = w
        tube = val(first(fixed, "TubeMaterial"))
        if tube:
            md["anode"] = _clean(tube)
        for k, key in (("Voltage", "tube_kV"), ("Current", "tube_mA")):
            v = _f(val(first(fixed, k)))
            if v is not None:
                md[key] = v
        gon = val(first(fixed, "GoniometerType"))
        if gon:
            md["goniometer"] = _clean(gon)
        radii = [_f(e.get("Value")) for e in fixed.iter() if _local(e.tag) == "Radius"]
        radii = [r for r in radii if r]
        if radii:
            md["goniometer_radius_mm"] = radii[-1]
        monos = sorted({_clean(e.get("VisibleName") or e.get("LogicName") or "")
                        for e in fixed.iter()
                        if "monochromator" in (_local(e.tag) + (e.get("LogicName") or "")
                                               + (e.get("VisibleName") or "")).lower()} - {""})
        if monos:
            md["optics"] = [f"monochromator: {m}" for m in monos]
            md.setdefault("notes", []).append(
                "a monochromator is named in the instrument record but its position in the beam "
                "path is not; the K-alpha2 ratio is as recorded")
        for e in fixed.iter():
            if _local(e.tag) == "InfoItem" and e.get("Name") == "SampleName" and e.get("Value"):
                md["sample_id"] = _clean(e.get("Value"))
    ts = val(first(root, "TimeStampStarted"))
    if ts:
        md["measured_at"] = _clean(ts)
    temps = [c for c in ("Temperature", "SampleTemperature") if c in views]
    if temps:
        t = table[:, views[temps[0]]]
        if np.all(np.isfinite(t)):
            md["temperature_C"] = round(float(np.mean(t)), 2)
    ok = np.isfinite(x) & np.isfinite(y)
    return _dataset(x[ok], y[ok], "brml", path, md, axis_label, title=member)


# ------------------------------------------------------------------ Rigaku RAS / RASX
def _ras_text(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp932", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", "replace")


def _english(v: str) -> str:
    """RAS values are often 'Japanese|English' pairs."""
    v = v.strip().strip('"')
    return v.split("|")[-1].strip() if "|" in v else v


def _ras_metadata(h: Dict[str, str]) -> Dict[str, Any]:
    md: Dict[str, Any] = {"signal_type": "xrd"}
    anode = _english(h.get("HW_XG_TARGET_NAME", ""))
    if not anode:
        z = _f(h.get("HW_XG_TARGET_ATOMIC_NUMBER"))
        anode = _ANODE_BY_Z.get(int(z)) if z else ""
    if anode:
        md["anode"] = anode
    wave_type = _english(h.get("MEAS_COND_XG_WAVE_TYPE", ""))
    ka1, ka2 = _f(h.get("HW_XG_WAVE_LENGTH_ALPHA1")), _f(h.get("HW_XG_WAVE_LENGTH_ALPHA2"))
    ratio = 0.0 if wave_type.lower() == "ka1" else (0.5 if wave_type.lower() == "ka" else None)
    w = _wavelength(ka1, ka2, ratio)
    if w:
        md["wavelength_angstrom"] = w
        if wave_type:
            md.setdefault("notes", []).append(
                f"the file records wave type '{wave_type}' and no K-alpha2/K-alpha1 ratio"
                + ("; the natural tube ratio 0.5 is given" if ratio == 0.5 else
                   "; K-alpha1 only (ratio 0)" if ratio == 0.0 else ""))
    gon = _english(h.get("HW_GONIOMETER_NAME", ""))
    md["instrument"] = f"Rigaku {gon}".strip() if gon else "Rigaku"
    for k, key in (("MEAS_COND_XG_VOLTAGE", "tube_kV"), ("MEAS_COND_XG_CURRENT", "tube_mA")):
        if _f(h.get(k)) is not None:
            md[key] = _f(h.get(k))
    stage = [_english(h.get(k, "")) for k in ("HW_ATTACHMENT_NAME", "HW_SAMPLE_HOLDER_NAME")]
    stage = [s for s in stage if s and s.lower() not in ("none", "standard")]
    if stage:
        md["sample_stage"] = ", ".join(stage)
    opt = _english(h.get("MEAS_COND_OPT_ATTR", "")) or _english(h.get("MEAS_COND_OPT_NAME", ""))
    if opt:
        md["optics"] = [opt]
    names = {k.rsplit("-", 1)[-1]: _english(v) for k, v in h.items() if k.startswith("MEAS_COND_AXIS_NAME-")}
    for i, nm in names.items():
        pos = _english(h.get(f"MEAS_COND_AXIS_POSITION-{i}", ""))
        if "monochromator" in nm.lower() and pos and pos.lower() not in ("none", "") and _f(pos) is None:
            md["incident_monochromator" if "incident" in nm.lower() else "optics_monochromator"] = pos
    det = _english(h.get("HW_COUNTER_SELECT_NAME", ""))
    if det:
        md["detector"] = det
    sid = _english(h.get("FILE_SAMPLE", ""))
    if sid:
        md["sample_id"] = sid
    memo = _english(h.get("FILE_MEMO", ""))
    if memo:
        md["file_memo"] = memo
    if h.get("MEAS_SCAN_START_TIME"):
        md["measured_at"] = _english(h["MEAS_SCAN_START_TIME"])
    for k in ("MEAS_COND_TEMPERATURE", "MEAS_TEMPERATURE"):
        if _f(_english(h.get(k, ""))) is not None:
            md["temperature_C"] = _f(_english(h[k]))
    unit = _english(h.get("MEAS_SCAN_UNIT_Y", "")) or _english(h.get("DISP_UNIT_Y", ""))
    md["intensity_units"] = unit or "counts"
    _monochromated(md)
    return md


def _ras_columns(lines: List[str]):
    rows = []
    for ln in lines:
        p = ln.replace(",", " ").split()
        if len(p) >= 2 and _f(p[0]) is not None and _f(p[1]) is not None:
            rows.append([float(v) for v in p[:3]] + ([1.0] if len(p) < 3 else []))
    a = np.array(rows, float) if rows else np.zeros((0, 3))
    if a.size == 0:
        return a[:, 0], a[:, 1], False
    att = a[:, 2]
    scaled = not np.allclose(att, 1.0)
    return a[:, 0], a[:, 1] * (att if scaled else 1.0), scaled


def _read_ras(path: str) -> List[Dataset]:
    txt = _ras_text(Path(path).read_bytes())
    out = []
    for block in re.split(r"\*RAS_DATA_START", txt)[1:]:
        h: Dict[str, str] = {}
        hm = re.search(r"\*RAS_HEADER_START(.*?)\*RAS_HEADER_END", block, re.S)
        for ln in (hm.group(1) if hm else "").splitlines():
            m = re.match(r"\*(\S+)\s+(.*)$", ln.strip())
            if m:
                h[m.group(1)] = m.group(2).strip().strip('"').strip()
        im = re.search(r"\*RAS_INT_START(.*?)\*RAS_INT_END", block, re.S)
        if not im:
            continue
        x, y, scaled = _ras_columns(im.group(1).splitlines())
        if x.size == 0:
            continue
        md = _ras_metadata(h)
        if scaled:
            md.setdefault("notes", []).append("intensities multiplied by the attenuator factors in column 3")
        axis = _english(h.get("MEAS_SCAN_AXIS_X", "")) or "2-Theta"
        md["scan"] = {k: v for k, v in {
            "axis": axis, "mode": _english(h.get("MEAS_SCAN_MODE", "")) or None,
            "start": round(float(x[0]), 6), "end": round(float(x[-1]), 6),
            "step": _f(h.get("MEAS_SCAN_STEP")), "points": int(x.size),
            "speed": (f"{_f(h.get('MEAS_SCAN_SPEED'))} {_english(h.get('MEAS_SCAN_SPEED_UNIT', ''))}".strip()
                      if _f(h.get("MEAS_SCAN_SPEED")) is not None else None)}.items() if v is not None}
        out.append(_dataset(x, y, "ras", path, md, axis, title=f"block {len(out) + 1}"))
    return out


def _read_rasx(path: str) -> List[Dataset]:
    out = []
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        profiles = sorted((n for n in names if re.search(r"(^|/)Profile\d*\.txt$", n)),
                          key=lambda n: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", n)])
        for prof in profiles:
            folder = prof.rsplit("/", 1)[0] + "/" if "/" in prof else ""
            idx = re.search(r"Profile(\d*)\.txt$", prof).group(1)
            cond = next((n for n in names if n.startswith(folder) and
                         re.search(rf"Mea?surementConditions{idx}\.xml$", n)), None)
            h: Dict[str, str] = {}
            if cond:
                croot = ET.fromstring(z.read(cond))
                for e in croot.iter():
                    tag = _local(e.tag)
                    if e.text and e.text.strip() and len(list(e)) == 0:
                        h.setdefault(tag, e.text.strip())
            md: Dict[str, Any] = {"signal_type": "xrd"}
            anode = h.get("TargetName") or _ANODE_BY_Z.get(int(_f(h.get("TargetAtomicNumber")) or 0), "")
            if anode:
                md["anode"] = anode
            wave_type = h.get("WaveType", "")
            ratio = 0.0 if wave_type.lower() == "ka1" else (0.5 if wave_type.lower() == "ka" else None)
            w = _wavelength(_f(h.get("WavelengthKalpha1")), _f(h.get("WavelengthKalpha2")), ratio)
            if w:
                md["wavelength_angstrom"] = w
                if wave_type:
                    md["notes"] = [f"the file records wave type '{wave_type}' and no K-alpha2/K-alpha1 ratio"
                                   + ("; the natural tube ratio 0.5 is given" if ratio == 0.5 else
                                      "; K-alpha1 only (ratio 0)" if ratio == 0.0 else "")]
            md["instrument"] = f"Rigaku {h.get('SystemName', '')}".strip()
            for k, key in (("Voltage", "tube_kV"), ("Current", "tube_mA")):
                if _f(h.get(k)) is not None:
                    md[key] = _f(h.get(k))
            if h.get("StartTime"):
                md["measured_at"] = h["StartTime"]
            sname = h.get("SampleName") or h.get("Comment")
            if sname:
                md["sample_id"] = sname
            md["intensity_units"] = h.get("IntensityUnit") or "counts"
            x, y, scaled = _ras_columns(_ras_text(z.read(prof)).splitlines())
            if x.size == 0:
                continue
            if scaled:
                md.setdefault("notes", []).append("intensities multiplied by the attenuator factors in column 3")
            axis = h.get("AxisName") or "TwoTheta"
            md["scan"] = {k: v for k, v in {
                "axis": axis, "mode": h.get("Mode"), "start": round(float(x[0]), 6),
                "end": round(float(x[-1]), 6), "step": _f(h.get("Step")), "points": int(x.size),
                "speed": (f"{h.get('Speed')} {h.get('SpeedUnit', '')}".strip() if h.get("Speed") else None)
            }.items() if v is not None}
            out.append(_dataset(x, y, "rasx", path, md, axis, title=prof))
    return out
