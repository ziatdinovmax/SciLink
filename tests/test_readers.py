"""scilink.utils.readers: DM3/DM4, HyperSpy, EMD (Berkeley, Velox) and their wiring.

Every file here is built in the test with known contents, so each check is
against values the test chose: the array in canonical order, every axis's
calibration, the metadata in SciLink's schema, the loaders, examine_data,
the reader split behind prepare_inputs, and the merge with given metadata.
"""

import json
import struct
import types
from pathlib import Path

import h5py
import numpy as np
import pytest

from scilink.utils.readers import (
    file_metadata, merge_file_metadata, read_dataset, read_datasets,
    records_calibration,
)
from scilink.utils.readers._layout import canonicalize
from scilink.utils.readers._model import Axis, UnsupportedContent


# ---------------------------------------------------------------- DM writer --

class _DM:
    """A minimal DigitalMicrograph tag-file writer (DM3 or DM4, little-endian data)."""

    SIMPLE = {np.dtype("i2"): 2, np.dtype("i4"): 3, np.dtype("u2"): 4, np.dtype("u4"): 5,
              np.dtype("f4"): 6, np.dtype("f8"): 7, np.dtype("i1"): 9, np.dtype("u1"): 10}

    def __init__(self, version):
        self.v = version
        self.n = 8 if version == 4 else 4

    def _len(self, x):
        return int(x).to_bytes(self.n, "big")

    def _data(self, info, payload):
        return b"%%%%" + self._len(len(info)) + b"".join(self._len(i) for i in info) + payload

    def value(self, v):
        if isinstance(v, str):
            a = np.frombuffer(v.encode("utf-16-le"), dtype="<u2")
            return self._data([20, 4, a.size], a.tobytes())
        if isinstance(v, np.ndarray):
            code = self.SIMPLE[v.dtype.newbyteorder("=")]
            return self._data([20, code, v.size], v.astype(v.dtype.newbyteorder("<")).tobytes())
        if isinstance(v, float):
            return self._data([7], struct.pack("<d", v))
        return self._data([3], struct.pack("<i", int(v)))

    def tag(self, label, v):
        lab = label.encode("latin-1")
        if isinstance(v, (dict, list)):
            body, ttype = self.group(v), 20
        else:
            body, ttype = self.value(v), 21
        head = bytes([ttype]) + struct.pack(">H", len(lab)) + lab
        if self.v == 4:
            head += struct.pack(">Q", len(body))
        return head + body

    def group(self, g):
        items = list(g.items()) if isinstance(g, dict) else [("", x) for x in g]
        return b"\x00\x01" + self._len(len(items)) + b"".join(self.tag(k, v) for k, v in items)

    def file(self, root):
        body = self.group(root)
        return struct.pack(">i", self.v) + self._len(len(body)) + struct.pack(">i", 1) + body + b"\0" * 8


def _dm_image(data, cals, *, dtype_code, signal="", fmt="", volt=None, name="test"):
    """One ImageList entry; ``data`` in C order, ``cals`` fastest-first."""
    dims = list(reversed(data.shape))
    tags = {"Meta Data": {"Signal": signal, "Format": fmt}} if signal or fmt else {}
    if volt:
        tags["Microscope Info"] = {"Voltage": float(volt)}
    return {
        "ImageData": {
            "Calibrations": {
                "Brightness": {"Origin": 0.0, "Scale": 1.0, "Units": "Counts"},
                "Dimension": [{"Origin": float(o), "Scale": float(s), "Units": u} for o, s, u in cals],
            },
            "Data": data.ravel(),
            "DataType": dtype_code,
            "Dimensions": [int(d) for d in dims],
        },
        "ImageTags": tags,
        "Name": name,
    }


def _write_dm(path, images, version):
    thumb = _dm_image(np.zeros((4, 4), np.uint32), [(0, 1, ""), (0, 1, "")], dtype_code=23)
    path.write_bytes(_DM(version).file({"ImageList": [thumb] + images}))


@pytest.fixture(params=[3, 4], ids=["dm3", "dm4"])
def dm_files(request, tmp_path):
    v = request.param
    rng = np.random.default_rng(0)
    # spectrum image stored DM-style: dims fastest-first (x, y, E) -> C order (E, y, x)
    cube_yxe = rng.random((5, 4, 30)).astype(np.float32)
    stored = np.ascontiguousarray(np.transpose(cube_yxe, (2, 0, 1)))
    si = _dm_image(stored, [(0, 0.03, "nm"), (0, 0.03, "nm"), (-400.0, 1.0, "eV")],
                   dtype_code=2, signal="EELS", fmt="Spectrum image", volt=200000.0)
    _write_dm(tmp_path / f"si.dm{v}", [si], v)
    img = (rng.random((6, 7)) * 1000).astype(np.uint16)
    _write_dm(tmp_path / f"img.dm{v}", [_dm_image(img, [(0, 0.05, "nm"), (0, 0.05, "nm")], dtype_code=10)], v)
    spec = rng.random(50).astype(np.float32)
    _write_dm(tmp_path / f"spec.dm{v}", [_dm_image(spec, [(-1400.0, 0.25, "eV")], dtype_code=2, signal="EELS")], v)
    return tmp_path, v, cube_yxe, img, spec


def test_dm_spectrum_image_canonical_order_and_calibration(dm_files):
    d, v, cube, _, _ = dm_files
    ds = read_dataset(d / f"si.dm{v}")
    assert ds.kind == "spectrum_image" and ds.format == "dm"
    np.testing.assert_array_equal(ds.data, cube)
    y, x, e = ds.axes
    assert (y.name, x.name, e.name) == ("y", "x", "Energy loss")
    assert e.units == "eV" and e.offset == pytest.approx(400.0) and e.scale == 1.0
    assert x.units == "nm" and x.scale == pytest.approx(0.03)
    assert ds.metadata["signal_type"] == "EELS" and ds.metadata["beam_energy_kV"] == 200.0


def test_dm_image_spectrum_and_thumbnail_dropped(dm_files):
    d, v, _, img, spec = dm_files
    found = read_datasets(d / f"img.dm{v}")
    assert len(found) == 1                        # the RGB thumbnail is not data
    np.testing.assert_array_equal(found[0].data, img)
    assert found[0].kind == "image" and found[0].axes[1].scale == pytest.approx(0.05)
    sp = read_dataset(d / f"spec.dm{v}")
    assert sp.kind == "spectrum"
    np.testing.assert_allclose(sp.axes[0].coordinates()[:2], [350.0, 350.25])


def test_dm_rejects_non_dm(tmp_path):
    p = tmp_path / "bad.dm3"
    p.write_bytes(struct.pack(">i", 7) + b"\0" * 32)
    with pytest.raises(ValueError):
        read_dataset(p)


# ---------------------------------------------------------- HyperSpy / EMD --

def _hspy(path, data, axes, signal_type="", beam=None):
    with h5py.File(path, "w") as h:
        h.attrs["file_format"] = "HyperSpy"
        g = h.create_group("Experiments/sig")
        g["data"] = data
        for i, (name, units, off, sc, nav) in enumerate(axes):
            a = g.create_group(f"axis-{i}")
            a.attrs.update(name=name, units=units, offset=off, scale=sc,
                           size=data.shape[i], navigate=nav)
        s = g.create_group("metadata/Signal")
        s.attrs["signal_type"] = signal_type
        if beam:
            g.create_group("metadata/Acquisition_instrument/TEM").attrs["beam_energy"] = beam


def test_hspy_wavelength_axis_is_spectral_even_in_nm(tmp_path):
    cube = np.random.default_rng(1).random((3, 4, 20)).astype(np.float32)
    _hspy(tmp_path / "cl.hspy", cube, [("Height", "µm", 0, 2.0, True), ("Width", "µm", 0, 2.0, True),
                                        ("Wavelength", "nm", 200.0, 0.5, False)], "CL")
    ds = read_dataset(tmp_path / "cl.hspy")
    assert ds.kind == "spectrum_image" and ds.signal_axis.name == "Wavelength"
    np.testing.assert_array_equal(ds.data, cube)
    md = file_metadata(ds)
    assert md["energy_range"] == {"start": 200.0, "end": 209.5, "units": "nm"}
    assert md["experiment"]["technique"] == "Cathodoluminescence"


def test_hyperspy_inside_hdf5_suffix_is_recognised(tmp_path):
    cube = np.zeros((2, 3, 10), np.float32)
    _hspy(tmp_path / "x.hdf5", cube, [("y", "nm", 0, 1, True), ("x", "nm", 0, 1, True),
                                       ("Energy loss", "eV", 100, 1, False)], "EELS", beam=300.0)
    assert records_calibration(tmp_path / "x.hdf5")
    ds = read_dataset(tmp_path / "x.hdf5")
    assert ds.format == "hspy" and ds.metadata["beam_energy_kV"] == 300.0


def test_berkeley_emd_with_two_element_dims_and_ncem_units(tmp_path):
    img = np.arange(12.0).reshape(3, 4)
    with h5py.File(tmp_path / "b.emd", "w") as h:
        h.attrs.update(version_major=0, version_minor=2)
        g = h.create_group("data/img")
        g.attrs["emd_group_type"] = 1
        g["data"] = img
        g["dim1"] = np.array([0.0, 0.5]); g["dim1"].attrs.update(name="y", units="[n_m]")
        g["dim2"] = np.arange(4) * 0.5;   g["dim2"].attrs.update(name="x", units="[n_m]")
    ds = read_dataset(tmp_path / "b.emd")
    assert ds.kind == "image" and [a.units for a in ds.axes] == ["nm", "nm"]
    assert [a.scale for a in ds.axes] == [0.5, 0.5]


def _velox_md(**extra):
    md = {"Optics": {"AccelerationVoltage": "300000"},
          "BinaryResult": {"PixelSize": {"width": "2e-10", "height": "2e-10"},
                           "PixelUnitX": "m", "PixelUnitY": "m",
                           "Offset": {"x": "0", "y": "0"}, "Detector": "HAADF"},
          "Detectors": {"Detector-0": {"DetectorType": "AnalyticalDetector", "Dispersion": "10",
                                       "OffsetEnergy": "-1000"}}}
    md.update(extra)
    raw = np.frombuffer(json.dumps(md).encode(), dtype=np.uint8)
    out = np.zeros(6000, np.uint8)
    out[:raw.size] = raw
    return out


def test_velox_eds_stream_decodes_to_the_summed_cube(tmp_path):
    rng = np.random.default_rng(2)
    ny, nx, nch, frames = 3, 4, 64, 2
    counts = rng.integers(0, 3, size=(frames, ny * nx, nch))
    stream = []
    for f in range(frames):
        for p in range(ny * nx):
            for ch in range(nch):
                stream += [ch] * int(counts[f, p, ch])
            stream.append(65535)                  # a mark closes every pixel
    stream += [5, 6]                              # events after the last frame: dropped
    with h5py.File(tmp_path / "v.emd", "w") as h:
        h["Version"] = np.array([json.dumps({"format": "Velox"}).encode()], dtype=object)
        img = h.create_group("Data/Image/a")
        img["Data"] = rng.random((ny, nx, frames)).astype(np.float32)
        img["Metadata"] = np.stack([_velox_md()] * frames, axis=1)
        st = h.create_group("Data/SpectrumStream/b")
        st["Data"] = np.asarray(stream, np.uint16)[:, None]
        st["AcquisitionSettings"] = np.array([json.dumps(
            {"bincount": str(nch), "RasterScanDefinition": {"Width": "512", "Height": "512"}}).encode()],
            dtype=object)
        st["Metadata"] = _velox_md()[:, None]
    found = read_datasets(tmp_path / "v.emd")
    si = found[0]
    assert si.kind == "spectrum_image" and si.data.shape == (ny, nx, nch)
    np.testing.assert_array_equal(si.data, counts.sum(0).reshape(ny, nx, nch))
    e = si.signal_axis
    assert (e.units, e.offset, e.scale) == ("keV", -1.0, 0.01)
    assert si.axes[0].units == "nm" and si.axes[0].scale == pytest.approx(0.2)
    assert any(d.kind == "image_stack" and d.data.shape == (frames, ny, nx) for d in found)


# -------------------------------------------------------------- layout -------

def test_canonicalize_by_names_and_units():
    d = np.zeros((7, 3, 2))
    out, axes, kind = canonicalize(d, [Axis(name="Energy", units="eV"), Axis(units="nm"), Axis(units="nm")])
    assert kind == "spectrum_image" and out.shape == (3, 2, 7)
    out, axes, kind = canonicalize(np.zeros((3, 4, 5)), [Axis(units="nm"), Axis(units="nm"), Axis(name="frame")])
    assert kind == "image_stack" and out.shape == (5, 3, 4)
    out, axes, kind = canonicalize(np.zeros((6, 40)), [Axis(name="x", units="nm"), Axis(name="Raman shift", units="cm-1")])
    assert kind == "spectrum_series"


# ------------------------------------------------------ SciLink metadata -----

def test_file_metadata_is_canonical_and_never_needs_normalizing(dm_files):
    from scilink.agents.exp_agents.metadata_converter import (
        check_schema_conformance, normalize_metadata_dict, resolve_axis_spec)
    d, v, cube, _, _ = dm_files
    md = file_metadata(read_dataset(d / f"si.dm{v}"))
    assert check_schema_conformance(md) == (True, [])
    assert normalize_metadata_dict(md)[1] is False
    ax = resolve_axis_spec(md)["axis_2"]
    assert (ax["units"], ax["start"], ax["end"]) == ("eV", 400.0, 400.0 + cube.shape[2] - 1)
    assert md["spatial_info"]["pixel_size"] == pytest.approx(0.03)
    assert md["sample"] == {"material": "not recorded in the file"}


def test_non_uniform_signal_axis_travels_as_samples(tmp_path):
    w = np.array([1.0, 2.0, 4.0, 8.0])
    with h5py.File(tmp_path / "n.hspy", "w") as h:
        h.attrs["file_format"] = "HyperSpy"
        g = h.create_group("Experiments/s")
        g["data"] = np.ones(4)
        a = g.create_group("axis-0")
        a.attrs.update(name="Raman shift", units="cm-1", size=4, navigate=False)
        a["axis"] = w
    md = file_metadata(read_dataset(tmp_path / "n.hspy"))
    assert md["axis_spec"]["axis_2"]["samples"] == w.tolist()


def test_merge_keeps_given_values_fills_the_rest_and_names_conflicts():
    from_file = {"energy_range": {"start": 401.96, "end": 2447.0, "units": "eV"},
                 "spatial_info": {"pixel_size": 0.03, "field_of_view_units": "nm"},
                 "sample": {"material": "not recorded in the file"},
                 "metadata_source": "Read from the dm file"}
    given = {"sample": {"material": "the sample"}, "energy_range": {"start": 400.0, "units": "eV"}}
    merged, filled, conflicts = merge_file_metadata(given, from_file)
    assert merged["sample"] == {"material": "the sample"}
    assert merged["energy_range"] == {"start": 400.0, "end": 2447.0, "units": "eV"}
    assert "spatial_info" in filled and "energy_range.end" in filled
    assert conflicts == ["energy_range.start: given 400.0, file records 401.96"]
    assert "metadata_source" not in merged


# ----------------------------------------------------------- wiring ----------

def test_agent_loaders_read_native_files_and_keep_npy_untouched(dm_files, tmp_path):
    import logging
    from scilink.agents.exp_agents.hyperspectral_analysis_agent import HyperspectralAnalysisAgent
    from scilink.skills._shared.curve_fitting_tools import load_curve_data
    from scilink.skills._shared.image_processor import load_image
    d, v, cube, img, spec = dm_files
    stub = types.SimpleNamespace(logger=logging.getLogger("t"))
    np.testing.assert_array_equal(
        HyperspectralAnalysisAgent._load_hyperspectral_data(stub, str(d / f"si.dm{v}")), cube)
    np.testing.assert_array_equal(load_image(str(d / f"img.dm{v}"), raw=True), img)
    xy = load_curve_data(str(d / f"spec.dm{v}"))
    assert xy.shape == (50, 2) and xy[0, 0] == pytest.approx(350.0)
    np.testing.assert_array_equal(xy[:, 1], spec)
    np.save(tmp_path / "c.npy", cube)             # an existing format keeps its own path
    np.testing.assert_array_equal(
        HyperspectralAnalysisAgent._load_hyperspectral_data(stub, str(tmp_path / "c.npy")), cube)


def test_examine_data_routes_a_native_file_with_its_calibration(dm_files):
    from scilink.agents.exp_agents.analysis_orchestrator_tools import AnalysisOrchestratorTools
    d, v, *_ = dm_files
    orch = types.SimpleNamespace(base_dir=str(d), _agent_registry={}, current_metadata=None)
    tools = AnalysisOrchestratorTools(orch)
    r = json.loads(tools.functions_map["examine_data"](str(d / f"si.dm{v}")))
    assert r["data_type"] == "hyperspectral" and r["suggested_agents"] == [2]
    assert r["file_metadata"]["energy_range"]["start"] == pytest.approx(400.0)
    r = json.loads(tools.functions_map["examine_data"](str(d / f"img.dm{v}")))
    assert r["data_type"] == "image" and r["suggested_agents"] == [1]


def test_split_with_reader_needs_no_generated_code(dm_files, tmp_path):
    from scilink.utils.file_prep import prepare_inputs, prepare_inputs_batch

    class NoModel:
        def __getattr__(self, name):
            raise AssertionError(f"the codegen path was used ({name})")

    d, v, cube, img, _ = dm_files
    r = prepare_inputs(d / f"si.dm{v}", model=NoModel(), executor=NoModel(),
                       output_dir=tmp_path / "prep", probe=None)
    assert r["status"] == "success" and r["deterministic"]
    np.testing.assert_array_equal(np.load(r["data_path"]), cube)
    assert json.load(open(r["metadata_path"]))["energy_range"]["units"] == "eV"
    b = prepare_inputs_batch([d / f"si.dm{v}", d / f"img.dm{v}"], model=NoModel(),
                             executor=NoModel(), output_dir=tmp_path / "prep", probes=[None, None])
    assert b["status"] == "success" and b["summary"]["n_codegens"] == 0


def test_split_writes_a_spectrum_as_xy_columns(dm_files, tmp_path):
    from scilink.utils.file_prep import split_with_reader, _file_paths
    from scilink.skills._shared.curve_fitting_tools import load_curve_data
    d, v, _, _, spec = dm_files
    src = d / f"spec.dm{v}"
    r = split_with_reader(src, _file_paths(src, tmp_path / "o"))
    xy = np.load(r["data_path"])
    np.testing.assert_array_equal(xy, load_curve_data(str(src)))
    assert xy.shape == (50, 2) and xy[0, 0] == pytest.approx(350.0)


def test_split_leaves_other_files_to_the_codegen_path(tmp_path):
    from scilink.utils.file_prep import split_with_reader, _file_paths
    np.save(tmp_path / "a.npy", np.ones(3))
    assert split_with_reader(tmp_path / "a.npy", _file_paths(tmp_path / "a.npy", tmp_path / "o")) is None


def test_meta_probe_marks_native_files(dm_files):
    from scilink.agents.meta_agent.meta_orchestrator_tools import _probe_file
    d, v, *_ = dm_files
    info = _probe_file(d / f"si.dm{v}")
    assert info["read_natively"] and info["data_kind"] == "spectrum_image"
    assert info["signal_axis"]["units"] == "eV"


def test_unsupported_content_is_named(tmp_path):
    with h5py.File(tmp_path / "x.emd", "w") as h:
        h["junk"] = np.ones(3)
    with pytest.raises(UnsupportedContent):
        read_dataset(tmp_path / "x.emd")


def test_hdf5_routing_follows_the_files_own_axes(tmp_path):
    """A HyperSpy image stack saved as .hdf5 goes to the image agent; a plain
    HDF5 array of the same shape keeps the shape rule (3-D is a datacube)."""
    from scilink.agents.exp_agents.analysis_orchestrator_tools import AnalysisOrchestratorTools
    stack = np.random.default_rng(3).random((8, 32, 32)).astype(np.float32)
    _hspy(tmp_path / "stack.hdf5", stack, [("time", "s", 0, 0.5, True), ("y", "nm", 0, 0.1, False),
                                            ("x", "nm", 0, 0.1, False)])
    with h5py.File(tmp_path / "plain.h5", "w") as h:
        h["data"] = stack
    orch = types.SimpleNamespace(base_dir=str(tmp_path), _agent_registry={}, current_metadata=None)
    ex = AnalysisOrchestratorTools(orch).functions_map["examine_data"]
    r = json.loads(ex(str(tmp_path / "stack.hdf5")))
    assert (r["file_kind"], r["data_type"], r["suggested_agents"]) == ("image_stack", "image_series", [1])
    r = json.loads(ex(str(tmp_path / "plain.h5")))
    assert r["data_type"] == "hyperspectral" and "file_kind" not in r


# ------------------------------------------------------- review fixes -------

def _velox_stream_file(path, seed=2, ny=3, nx=4, nch=64, frames=2):
    """A Velox file with a HAADF frame stack and an EDS event stream; returns
    the per-frame counts the stream encodes."""
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 3, size=(frames, ny * nx, nch))
    stream = []
    for f in range(frames):
        for p in range(ny * nx):
            for ch in range(nch):
                stream += [ch] * int(counts[f, p, ch])
            stream.append(65535)
    stream += [5, 6]
    with h5py.File(path, "w") as h:
        h["Version"] = np.array([json.dumps({"format": "Velox"}).encode()], dtype=object)
        img = h.create_group("Data/Image/a")
        img["Data"] = rng.random((ny, nx, frames)).astype(np.float32)
        img["Metadata"] = np.stack([_velox_md()] * frames, axis=1)
        st = h.create_group("Data/SpectrumStream/b")
        st["Data"] = np.asarray(stream, np.uint16)[:, None]
        st["AcquisitionSettings"] = np.array([json.dumps(
            {"bincount": str(nch), "RasterScanDefinition": {"Width": "512", "Height": "512"}}).encode()],
            dtype=object)
        st["Metadata"] = _velox_md()[:, None]
    return counts


def test_stream_decode_is_the_same_at_every_chunk_size(tmp_path, monkeypatch):
    from scilink.utils.readers import emd
    counts = _velox_stream_file(tmp_path / "v.emd", ny=5, nx=7, nch=32, frames=3)
    want = counts.sum(0).reshape(5, 7, 32)
    for chunk in (3, 50, 1000, 1 << 22):         # chunks that cut pixels and frames apart
        monkeypatch.setattr(emd, "_STREAM_CHUNK", chunk)
        got = read_dataset(tmp_path / "v.emd", prefer=("spectrum_image",)).data
        np.testing.assert_array_equal(got, want)
        assert got.dtype == np.uint32


def test_probe_examine_and_metadata_never_decode_the_stream(tmp_path, monkeypatch):
    """Only the loaders and the split need the counts; a probe, examine_data
    and the metadata read get the shape without decoding the event stream."""
    from scilink.agents.exp_agents.analysis_orchestrator_tools import AnalysisOrchestratorTools
    from scilink.agents.meta_agent.meta_orchestrator_tools import _native_summary
    from scilink.utils.readers import emd
    _velox_stream_file(tmp_path / "v.emd")

    def boom(*a, **k):
        raise AssertionError("the event stream was decoded")
    monkeypatch.setattr(emd, "_decode_stream", boom)
    info = _native_summary(tmp_path / "v.emd")
    assert info["data_kind"] == "spectrum_image" and info["shape"] == [3, 4, 64]
    orch = types.SimpleNamespace(base_dir=str(tmp_path), _agent_registry={}, current_metadata=None)
    tools = AnalysisOrchestratorTools(orch)
    r = json.loads(tools.functions_map["examine_data"](str(tmp_path / "v.emd")))
    assert r["data_type"] == "hyperspectral" and r["shape"] == [3, 4, 64]
    md = tools._file_metadata_for(tmp_path / "v.emd", prefer=("spectrum_image",))
    assert md["energy_range"]["units"] == "keV"


def test_split_decodes_a_native_file_once(tmp_path, monkeypatch):
    from scilink.utils.file_prep import split_with_reader, _file_paths
    from scilink.utils.readers import emd
    counts = _velox_stream_file(tmp_path / "v.emd")
    calls = []
    real = emd._decode_stream
    monkeypatch.setattr(emd, "_decode_stream", lambda *a, **k: calls.append(1) or real(*a, **k))
    r = split_with_reader(tmp_path / "v.emd", _file_paths(tmp_path / "v.emd", tmp_path / "o"))
    assert r["status"] == "success" and len(calls) == 1
    np.testing.assert_array_equal(np.load(r["data_path"]), counts.sum(0).reshape(3, 4, 64))
    md = json.load(open(r["metadata_path"]))
    assert [d["kind"] for d in md["other_datasets_in_file"]] == ["image_stack"]


def test_a_failed_native_split_asks_consent_before_generated_code(dm_files, tmp_path, monkeypatch):
    """Native files skip the consent prompt because their split runs no
    generated code; when a reader fails, consent is asked before the
    generated-code fallback, and a refusal stops it."""
    import scilink.executors as ex
    from scilink.agents.meta_agent.meta_orchestrator_tools import MetaOrchestratorTools
    asked = []
    monkeypatch.setattr(ex, "require_sandbox_approval", lambda **k: asked.append(k) or False)
    d, v, *_ = dm_files
    with h5py.File(tmp_path / "broken.emd", "w") as h:       # HDF5, but no EMD layout
        h["junk"] = np.ones(3)
    tools = MetaOrchestratorTools(types.SimpleNamespace(base_dir=tmp_path, model=None))
    prep = tools.functions_map["prepare_inputs"]
    r = json.loads(prep(str(d / f"si.dm{v}")))
    assert r["status"] == "success" and r["deterministic"] and asked == []
    r = json.loads(prep(str(tmp_path / "broken.emd")))
    assert r["status"] == "error" and "declined" in r["message"] and len(asked) == 1
    r = json.loads(prep([str(d / f"si.dm{v}"), str(tmp_path / "broken.emd")]))
    by_file = {Path(x["file"]).name: x for x in r["results"]}
    assert by_file[f"si.dm{v}"]["status"] == "success"
    assert "declined" in by_file["broken.emd"]["message"] and len(asked) == 2


def test_image_agent_loader_reads_a_hyperspy_hdf5_image(tmp_path):
    from scilink.skills._shared.image_analysis_tools import load_image_data
    img = np.random.default_rng(0).random((40, 48)).astype(np.float32)
    _hspy(tmp_path / "img.hdf5", img, [("y", "nm", 0, .1, False), ("x", "nm", 0, .1, False)])
    np.testing.assert_array_equal(load_image_data(str(tmp_path / "img.hdf5")), img)
    with h5py.File(tmp_path / "plain.h5", "w") as h:          # a plain HDF5 keeps its old path
        h["data"] = img
    with pytest.raises(ValueError, match="Could not load image"):
        load_image_data(str(tmp_path / "plain.h5"))


def test_loaders_refuse_a_native_file_without_their_kind_of_data(tmp_path):
    """A stack or a cube is never handed over as one frame, nor an image as a
    cube: a native file holding none of the loader's kind is refused by name."""
    import logging
    from scilink.agents.exp_agents.hyperspectral_analysis_agent import HyperspectralAnalysisAgent
    from scilink.live.modality import load_cube, load_image as live_image
    from scilink.skills._shared.curve_fitting_tools import load_curve_data
    stack = np.random.default_rng(1).random((5, 64, 64)).astype(np.float32)
    _hspy(tmp_path / "stack.hspy", stack, [("time", "s", 0, .5, True), ("y", "nm", 0, .1, False),
                                           ("x", "nm", 0, .1, False)])
    img = stack[0]
    _hspy(tmp_path / "img.hspy", img, [("y", "nm", 0, .1, False), ("x", "nm", 0, .1, False)])
    stub = types.SimpleNamespace(logger=logging.getLogger("t"))
    with pytest.raises(UnsupportedContent, match="image_stack"):
        live_image(str(tmp_path / "stack.hspy"))
    assert live_image(str(tmp_path / "img.hspy")).shape == (64, 64)
    for load in (load_cube, load_curve_data,
                 lambda p: HyperspectralAnalysisAgent._load_hyperspectral_data(stub, p)):
        with pytest.raises((UnsupportedContent, ValueError)):
            load(str(tmp_path / "img.hspy"))


def _run_analysis_with(tmp_path, data_path, agent_id, **kw):
    import contextlib, io, os
    os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
    os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisMode, AnalysisOrchestratorAgent)
    seen = []

    class Agent:
        def analyze(self, data=None, system_info=None, **k):
            seen.append({"data": data, "system_info": system_info})
            return {"status": "success", "detailed_analysis": "ok", "scientific_claims": []}
    with contextlib.redirect_stdout(io.StringIO()):
        orch = AnalysisOrchestratorAgent(base_dir=str(tmp_path / "s"), api_key="sk-dummy",
                                         model_name="claude-opus-4-6",
                                         analysis_mode=AnalysisMode.AUTONOMOUS)
        orch.create_agent_for_analysis = lambda *a, **k: Agent()
        out = json.loads(orch.tools.execute_tool("run_analysis", data_path=str(data_path),
                                                 agent_id=agent_id, **kw))
    return out, seen


def test_a_single_file_frame_stack_reaches_the_image_agent_as_a_stack(tmp_path):
    stack = np.random.default_rng(1).random((5, 64, 64)).astype(np.float32)
    _hspy(tmp_path / "stack.hspy", stack, [("time", "s", 0, .5, True), ("y", "nm", 0, .1, False),
                                           ("x", "nm", 0, .1, False)])
    out, seen = _run_analysis_with(tmp_path, tmp_path / "stack.hspy", 1)
    assert out.get("status") == "success", out
    assert isinstance(seen[-1]["data"], np.ndarray)
    np.testing.assert_array_equal(seen[-1]["data"], stack)
    img = stack[0]
    _hspy(tmp_path / "img.hspy", img, [("y", "nm", 0, .1, False), ("x", "nm", 0, .1, False)])
    out, seen = _run_analysis_with(tmp_path, tmp_path / "img.hspy", 1)
    assert seen[-1]["data"] == str(tmp_path / "img.hspy")      # a single image stays a path


def test_a_flat_sidecar_technique_and_material_stand_over_the_file(tmp_path):
    spec = np.random.default_rng(2).random(200).astype(np.float32)
    _hspy(tmp_path / "sp.hspy", spec, [("Energy", "eV", 280.0, 0.5, False)])
    (tmp_path / "sp.json").write_text(json.dumps({"technique": "XPS", "material": "MoS2"}))
    out, seen = _run_analysis_with(tmp_path, tmp_path / "sp.hspy", 0)
    md = seen[-1]["system_info"]
    assert md["experiment"]["technique"] == "XPS" and md["sample"]["material"] == "MoS2"
    assert md["energy_range"]["start"] == pytest.approx(280.0)          # the file filled the axis
    assert "not recorded" not in json.dumps(md) and "not named" not in json.dumps(md)


def test_metadata_describes_the_dataset_the_agent_loads(tmp_path):
    """A file whose largest dataset is not the one the agent takes: the
    metadata for the run comes from the agent's dataset."""
    rng = np.random.default_rng(4)
    stack = rng.random((40, 32, 32)).astype(np.float32)          # 40960 values, the largest
    cube = rng.random((8, 8, 100)).astype(np.float32)            # 6400 values
    with h5py.File(tmp_path / "two.hspy", "w") as h:
        h.attrs["file_format"] = "HyperSpy"
        for name, data, axes in (
                ("frames", stack, [("time", "s", 0, 1, True), ("y", "nm", 0, .1, False),
                                   ("x", "nm", 0, .1, False)]),
                ("si", cube, [("y", "nm", 0, .2, True), ("x", "nm", 0, .2, True),
                              ("Energy", "eV", 500.0, 2.0, False)])):
            g = h.create_group(f"Experiments/{name}")
            g["data"] = data
            for i, (n, u, off, sc, nav) in enumerate(axes):
                a = g.create_group(f"axis-{i}")
                a.attrs.update(name=n, units=u, offset=off, scale=sc, size=data.shape[i], navigate=nav)
            g.create_group("metadata/Signal").attrs["signal_type"] = "EELS" if name == "si" else ""
    out, seen = _run_analysis_with(tmp_path, tmp_path / "two.hspy", 2)
    md = seen[-1]["system_info"]
    assert md["energy_range"] == {"start": 500.0, "end": 698.0, "units": "eV"}
    assert md["spatial_info"]["pixel_size"] == pytest.approx(0.2)


def test_hyperspy_unset_values_are_not_values(tmp_path):
    img = np.zeros((16, 16), np.float32)
    _hspy(tmp_path / "n.hspy", img, [("y", "nm", 0, .1, False), ("x", "nm", 0, .1, False)])
    with h5py.File(tmp_path / "n.hspy", "a") as h:
        tem = h["Experiments/sig/metadata"].create_group("Acquisition_instrument/TEM")
        tem.attrs.update(beam_energy="_None_", microscope="<undefined>")
    md = file_metadata(read_dataset(tmp_path / "n.hspy"))
    assert "_None_" not in json.dumps(md) and "<undefined>" not in json.dumps(md)
    assert "instrument" not in md["experiment"] and "beam_energy_kV" not in md.get("acquisition", {})
