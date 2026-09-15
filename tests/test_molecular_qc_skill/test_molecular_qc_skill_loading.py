# tests/test_molecular_qc_skill/test_molecular_qc_skill_loading.py
"""
Tests for the NWChem skill bundle (molecular_qc scale): correct loading,
section parsing, and content requirements.

Unique basename (not the bare ``test_skill_loading.py`` used by the LAMMPS
suite) so the whole tree can be collected with ``-k`` without a basename
collision.
"""

import pytest
from scilink.skills.loader import load_skill, list_skills, list_all_skills


class TestSkillDiscovery:

    def test_nwchem_in_molecular_qc_domain(self):
        skills = list_skills(domain="molecular_qc")
        assert "nwchem" in skills

    def test_molecular_qc_in_all_skills(self):
        all_skills = list_all_skills()
        assert "molecular_qc" in all_skills
        assert "nwchem" in all_skills["molecular_qc"]


class TestSkillLoading:

    @pytest.fixture
    def skill(self):
        return load_skill("nwchem", domain="molecular_qc")

    def test_loads_without_error(self, skill):
        assert skill is not None

    def test_has_name(self, skill):
        assert skill["name"] == "nwchem"

    @pytest.mark.parametrize("section", [
        "overview", "planning", "analysis",
        "interpretation", "validation", "implementation",
    ])
    def test_has_required_section(self, skill, section):
        # ``analysis`` is populated from ``implementation`` by the loader's
        # synonym fold, so all six canonical sections are non-empty.
        assert section in skill
        assert len(skill[section]) > 0, f"Section '{section}' is empty"


class TestSkillContent:
    """Verify the bundle carries the critical decision-making information."""

    @pytest.fixture
    def skill(self):
        return load_skill("nwchem", domain="molecular_qc")

    def test_overview_is_finite_molecular(self, skill):
        o = skill["overview"].lower()
        assert "molecul" in o
        assert "not periodic" in o or "finite" in o

    def test_planning_covers_methods_beyond_dft(self, skill):
        p = skill["planning"].lower()
        assert "dft" in p
        assert "mp2" in p or "ccsd" in p

    def test_planning_has_basis_and_charge_multiplicity(self, skill):
        p = skill["planning"].lower()
        assert "basis" in p
        assert "charge" in p
        assert "multiplicity" in p

    def test_implementation_has_solvation(self, skill):
        impl = skill["implementation"].lower()
        assert "cosmo" in impl          # implicit solvation block present
        assert "task" in impl           # NWChem task lines present

    def test_implementation_has_deck_blocks(self, skill):
        impl = skill["implementation"].lower()
        assert "geometry" in impl
        assert "basis" in impl

    def test_validation_has_basis_check(self, skill):
        v = skill["validation"].lower()
        assert "basis" in v

    def test_validation_has_charge_multiplicity_check(self, skill):
        v = skill["validation"].lower()
        assert "charge" in v
        assert "mult" in v

    def test_interpretation_has_convergence_and_freqs(self, skill):
        interp = skill["interpretation"].lower()
        assert "converg" in interp
        assert "frequenc" in interp or "imaginary" in interp


class TestCustomSkillPath:
    """Loading a molecular_qc skill from an arbitrary name."""

    def test_nonexistent_name_raises(self):
        with pytest.raises(FileNotFoundError):
            load_skill("nonexistent_engine", domain="molecular_qc")
