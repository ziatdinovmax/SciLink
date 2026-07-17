"""Offline regression test for the spectral-imaging energy-axis conformance gap.

A hyperspectral / spectral-imaging metadata dict that lacks a resolvable energy
axis used to pass ``check_schema_conformance`` (it had experiment/sample/
technique), which SKIPPED the LLM normalizer that would extract ``energy_range``
from prose — so the agent silently fell back to channel indices and the K-edge
windows landed at the wrong energies. The fix flags such a dict non-conformant
so Tier-2 LLM extraction fires. These assertions are deterministic (no LLM).

  conda run -n scilink python -m pytest tests/test_metadata_conformance_energy_axis.py -q
"""
from scilink.agents.exp_agents.metadata_converter import check_schema_conformance


def _ok(m):
    return check_schema_conformance(m)[0]


def test_spectral_imaging_without_energy_axis_is_nonconformant():
    # The Hexitec X-ray case: technique names spectral imaging, no energy_range.
    meta = {
        "experiment_type": "Spectroscopy",
        "experiment": {"technique": "Spectral X-ray Imaging"},
        "sample": {"material": "Au"},
    }
    conformant, issues = check_schema_conformance(meta)
    assert conformant is False
    assert any("energy axis" in i for i in issues)


def test_energy_axis_via_energy_range_or_axis_spec_is_conformant():
    assert _ok({
        "experiment_type": "Spectroscopy",
        "experiment": {"technique": "Spectral X-ray Imaging"},
        "sample": {"material": "Au"},
        "energy_range": {"start": 0.1, "end": 204.8, "units": "keV"},
    })
    assert _ok({
        "experiment_type": "Spectroscopy",
        "experiment": {"technique": "STEM-EELS spectrum image"},
        "sample": {"material": "TiOx"},
        "axis_spec": {"axis_2": {"start": 450, "end": 550, "units": "eV"}},
    })


def test_partial_energy_range_units_only_still_nonconformant():
    # units without start/end is not resolvable -> still flagged.
    assert _ok({
        "experiment_type": "hyperspectral imaging",
        "experiment": {"technique": "Raman imaging"},
        "sample": {"material": "polymer"},
        "energy_range": {"units": "cm^-1"},
    }) is False


def test_no_false_positive_on_1d_curve_or_microscopy():
    # 1D XRD curve: x-axis is 2theta, energy_range legitimately absent.
    assert _ok({
        "experiment_type": "Spectroscopy",
        "experiment": {"technique": "XRD powder diffraction"},
        "sample": {"material": "Anorthite"},
        "data_columns": [{"name": "2theta", "units": "deg"},
                         {"name": "Counts", "units": "counts"}],
    })
    # Plain microscopy image: no spectral axis at all.
    assert _ok({
        "experiment_type": "Microscopy",
        "experiment": {"technique": "STEM-HAADF imaging"},
        "sample": {"material": "SrTiO3"},
        "spatial_info": {"field_of_view_x": 8, "field_of_view_units": "nm"},
    })


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
