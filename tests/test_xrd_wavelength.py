"""resolve_wavelength: which source a metadata blob describes, including
whether a lab anode's K-alpha2 line is in the beam (the refinement tools model
'XKa' as the K-alpha1/K-alpha2 doublet and 'XKa1' or a number as one line)."""
import pytest

from scilink.skills.structure_matching.xrd.wavelength import resolve_wavelength as R


@pytest.mark.parametrize("meta, expected", [
    # tube K-alpha, unmonochromated or with a secondary (graphite) monochromator: doublet
    ("Cu K-alpha", "CuKa"),
    ("Cu Kα, Ni K-beta filter", "CuKa"),
    ("Cu K-alpha, graphite monochromator", "CuKa"),
    ("Co K-alpha radiation, Fe filter", "CoKa"),
    ("silver K-alpha", "AgKa"),
    ("Cu Kalpha1+2", "CuKa"),
    ("Cu Kα1/Kα2", "CuKa"),
    # K-alpha1 only
    ("strictly monochromatic Cu K-alpha1", "CuKa1"),
    ("Cu Ka1 (Ge111 monochromator)", "CuKa1"),
    ("Cu K-alpha1 only", "CuKa1"),
    ("Cu Ka, Johansson primary monochromator", "CuKa1"),
    ("Mo Kα1", "MoKa1"),
    # synchrotron / plain numbers
    ("synchrotron 0.7749 A", 0.7749),
])
def test_free_text(meta, expected):
    assert R(meta) == expected


def test_structured_fields():
    assert R({"wavelength": 1.5406}) == 1.5406                      # bare number: unchanged
    assert R({"wavelength": 1.5406, "radiation": "Cu K-alpha"}) == "CuKa"
    assert R({"wavelength": 1.54059, "radiation": "strictly monochromatic Cu K-alpha1"}) == "CuKa1"
    assert R({"experiment": {"source": "Cu Kα"}}) == "CuKa"
    assert R({"wavelength": "MoKa"}) == "MoKa"


def test_any_wavelength_key_is_read():
    # a '*wavelength*' key other than the canonical names must not fall through to the CuKa default
    assert R({"wavelength_angstrom": 0.49591, "instrument": "synchrotron, capillary"}) == 0.49591
    assert R({"x_ray_wavelength_A": 0.7749}) == 0.7749


def test_doublet_dict():
    lines = {"K-alpha1": 1.540598, "K-alpha2": 1.544426, "K-alpha2/K-alpha1": 0.5}
    assert R({"wavelength_angstrom": lines}) == "CuKa"
    assert R({"wavelength_angstrom": {**lines, "K-alpha2/K-alpha1": 0.0}}) == "CuKa1"


def test_defaults():
    assert R(None) == "CuKa"
    assert R({"x": "no source information"}) == "CuKa"
    assert R(None, default=1.0) == 1.0
