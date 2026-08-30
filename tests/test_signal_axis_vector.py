"""Offline regression tests: exact per-channel signal-axis vectors.

Hyperspectral axes used to be reconstructed as ``linspace(start, end, n)``
even when the metadata preserved the instrument's exact (nonuniform) sample
positions — on a real spectrometer wavelength table that is up to ~6 nm off
mid-spectrum, which shifted every reported band center. These tests pin the
new behavior: ``axis_spec.axis_2.samples`` (or a recognized top-level vector
such as ``wavelengths_nm``) is passed through ``resolve_axis_spec`` and used
verbatim by ``create_axis`` / ``signal_axis_values``, with the historical
linspace path untouched as the fallback.

  conda run -n scilink python -m pytest tests/test_signal_axis_vector.py -q
"""
import numpy as np

from scilink.agents.exp_agents.metadata_converter import (
    check_schema_conformance,
    resolve_axis_spec,
    signal_axis_values,
)
from scilink.skills.hyperspectral.eels.eels import create_axis

# A nonuniformly spaced axis (growing step), like a real wavelength table.
WL = [400.0 + 2.0 * i + 0.02 * i * i for i in range(10)]


def _spectral_meta(**extra):
    meta = {
        "experiment_type": "Spectroscopy",
        "experiment": {"technique": "Hyperspectral imaging"},
        "sample": {"material": "test"},
    }
    meta.update(extra)
    return meta


def test_axis2_samples_passed_through_and_endpoints_derived():
    spec = resolve_axis_spec({"axis_spec": {"axis_2": {
        "name": "wavelength", "units": "nm", "kind": "signal", "samples": WL}}})
    a2 = spec["axis_2"]
    assert a2["samples"] == WL
    assert a2["start"] == WL[0] and a2["end"] == WL[-1]


def test_toplevel_wavelengths_nm_adopted():
    spec = resolve_axis_spec({
        "axis_spec": {"axis_2": {"name": "wavelength", "units": "nm",
                                 "kind": "signal",
                                 "start": WL[0], "end": WL[-1]}},
        "wavelengths_nm": WL,
    })
    assert spec["axis_2"]["samples"] == WL
    # explicit start/end are NOT overridden by the vector
    assert spec["axis_2"]["start"] == WL[0]


def test_explicit_axis2_samples_beat_toplevel_vector():
    other = [float(v) for v in range(10)]
    spec = resolve_axis_spec({
        "axis_spec": {"axis_2": {"kind": "signal", "samples": WL}},
        "wavelengths_nm": other,
    })
    assert spec["axis_2"]["samples"] == WL


def test_signal_axis_values_exact_and_mismatch():
    a2 = {"samples": WL}
    v = signal_axis_values(a2, len(WL))
    assert isinstance(v, np.ndarray) and np.allclose(v, WL)
    assert signal_axis_values(a2, len(WL) + 1) is None          # length mismatch
    assert signal_axis_values({"start": 1, "end": 2}, 5) is None  # no samples
    assert signal_axis_values({"samples": "not-a-vector"}, 5) is None
    assert signal_axis_values(None, 5) is None


def test_create_axis_uses_exact_vector_not_linspace():
    axis, xlabel, ok = create_axis(len(WL), {
        "axis_spec": {"axis_2": {"name": "wavelength", "units": "nm",
                                 "kind": "signal", "samples": WL}}})
    assert ok and np.allclose(axis, WL)
    assert xlabel == "Wavelength (nm)"
    # The nonuniform vector genuinely differs from the endpoint linspace.
    assert not np.allclose(axis, np.linspace(WL[0], WL[-1], len(WL)))


def test_create_axis_length_mismatch_falls_back_to_linspace():
    axis, _, ok = create_axis(len(WL) + 3, {
        "axis_spec": {"axis_2": {"name": "wavelength", "units": "nm",
                                 "kind": "signal", "samples": WL}}})
    # start/end were derived from the samples, so the fallback still works.
    assert ok and np.allclose(axis, np.linspace(WL[0], WL[-1], len(WL) + 3))


def test_legacy_linspace_path_byte_identical_without_samples():
    axis, xlabel, ok = create_axis(7, {
        "axis_spec": {"axis_2": {"name": "energy", "units": "eV",
                                 "kind": "signal", "start": 450, "end": 550}}})
    assert ok and np.allclose(axis, np.linspace(450, 550, 7))
    assert xlabel == "Energy (eV)"


def test_conformance_accepts_sample_vectors():
    # samples inside axis_spec, no start/end anywhere
    ok, issues = check_schema_conformance(_spectral_meta(
        axis_spec={"axis_2": {"kind": "signal", "samples": WL}}))
    assert ok, issues
    # recognized top-level vector, no axis_spec/energy_range
    ok, issues = check_schema_conformance(_spectral_meta(wavelengths_nm=WL))
    assert ok, issues
    # still non-conformant with no axis information at all
    ok, _ = check_schema_conformance(_spectral_meta())
    assert ok is False


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
