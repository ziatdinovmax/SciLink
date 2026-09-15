"""Loader-level tests for the raman and ir curve-fitting skill bundles.

Content-behavior validation (fluorescence protocol, band tables, blind-ID
lift) was done live against RRUFF and in-house benchmarks; see the PR. These
tests pin the packaging contract: bundles resolve by name, carry the full
five-section vocabulary, and expose selector metadata.
"""

import pytest

from scilink.skills.loader import load_skill

CORE_SECTIONS = ("overview", "planning", "implementation",
                 "interpretation", "validation")


@pytest.mark.parametrize("name", ["raman", "ir"])
class TestBundleContract:
    def test_resolves_by_bare_name(self, name):
        skill = load_skill(name)
        assert skill is not None

    def test_all_core_sections_populated(self, name):
        skill = load_skill(name)
        for section in CORE_SECTIONS:
            assert skill[section].strip(), f"empty section: {section}"

    def test_selector_metadata(self, name):
        skill = load_skill(name)
        meta = skill["meta"]
        assert meta.get("description")
        techs = meta.get("technique")
        assert isinstance(techs, list) and len(techs) >= 2

    def test_no_extras_lost(self, name):
        skill = load_skill(name)
        assert not skill.get("extras"), (
            "off-vocabulary sections present — move content under the "
            f"canonical headings: {sorted(skill['extras'])}")


def test_raman_carries_fluorescence_protocol():
    skill = load_skill("raman")
    assert "als_baseline" in skill["implementation"]
    assert "asymmetric" in skill["implementation"].lower()
    assert "background" in skill["planning"].lower()


def test_ir_carries_orientation_and_range_rules():
    skill = load_skill("ir")
    assert "transmittance" in skill["overview"].lower()
    assert "full measured range" in skill["planning"].lower()


def test_both_skills_pin_the_output_space_contract():
    # A peaks-only fit saved against raw data after baseline subtraction
    # breaks verification (fit offset by the whole continuum) and makes
    # self-reported R² disagree with R² recomputed from saved arrays.
    for name in ("raman", "ir"):
        impl = load_skill(name)["implementation"].lower()
        assert "one consistent space" in impl, name
