# tests/test_lammps_skill/test_lammps_tools.py
"""
Unit tests for the LAMMPS skill's tools module.

Tests every public function against the fixture data files and scripts.
No LLM calls — these are fast, deterministic, and run in CI.
"""

import pytest
from scilink.skills.molecular_dynamics.lammps import lammps as lammps_tools


# =====================================================================
# element_from_mass
# =====================================================================

class TestElementFromMass:
    """Test ASE-backed mass-to-element lookup."""

    @pytest.mark.parametrize("mass, expected", [
        (1.008,  "H"),
        (12.011, "C"),
        (14.007, "N"),
        (15.999, "O"),
        (22.990, "Na"),
        (28.086, "Si"),
        (35.453, "Cl"),
        (55.845, "Fe"),
        (63.546, "Cu"),
        (196.97, "Au"),
        (183.84, "W"),
        (107.87, "Ag"),
        (47.867, "Ti"),
    ])
    def test_exact_masses(self, mass, expected):
        assert lammps_tools.element_from_mass(mass) == expected

    @pytest.mark.parametrize("mass, expected", [
        (63.0, "Cu"),     # Slightly off
        (12.5, "C"),      # Within tolerance
        (56.0, "Fe"),
    ])
    def test_approximate_masses(self, mass, expected):
        assert lammps_tools.element_from_mass(mass) == expected

    def test_no_match_outside_tolerance(self):
        result = lammps_tools.element_from_mass(999.0, tolerance=1.0)
        assert result is None

    def test_custom_tolerance(self):
        # Very tight tolerance should reject slightly off masses
        assert lammps_tools.element_from_mass(64.0, tolerance=0.1) is None
        assert lammps_tools.element_from_mass(63.546, tolerance=0.1) == "Cu"


# =====================================================================
# parse_data_file — system classification
# =====================================================================

class TestParseDataFileMetal:
    """Bulk Cu: atom_style atomic, no bonds, single metal element."""

    def test_atom_count(self, data_dir):
        info = lammps_tools.parse_data_file(str(data_dir / "cu_bulk.data"))
        assert info["atom_count"] == 4

    def test_atom_style_detected(self, data_dir):
        info = lammps_tools.parse_data_file(str(data_dir / "cu_bulk.data"))
        assert info["atom_style"] == "atomic"

    def test_no_bonds(self, data_dir):
        info = lammps_tools.parse_data_file(str(data_dir / "cu_bulk.data"))
        assert info["has_bonds"] is False
        assert info["bond_count"] == 0

    def test_element_detected(self, data_dir):
        info = lammps_tools.parse_data_file(str(data_dir / "cu_bulk.data"))
        assert "Cu" in info["elements"]
        assert info["element_counts"]["Cu"] == 4

    def test_system_category(self, data_dir):
        info = lammps_tools.parse_data_file(str(data_dir / "cu_bulk.data"))
        assert info["system_category"] == "metal"
        assert info["has_metal"] is True

    def test_no_false_positives(self, data_dir):
        info = lammps_tools.parse_data_file(str(data_dir / "cu_bulk.data"))
        assert info["has_water"] is False
        assert info["has_ions"] is False
        assert info["has_organic"] is False
        assert info["has_semiconductor"] is False
        assert info["has_pair_coeffs"] is False

    def test_box_dimensions(self, data_dir):
        info = lammps_tools.parse_data_file(str(data_dir / "cu_bulk.data"))
        assert pytest.approx(info["box_dimensions"][0], abs=0.01) == 3.615


class TestParseDataFileSemiconductor:
    """Bulk Si: atom_style atomic, no bonds, semiconductor."""

    def test_system_category(self, data_dir):
        info = lammps_tools.parse_data_file(str(data_dir / "si_bulk.data"))
        assert info["system_category"] == "semiconductor"
        assert info["has_semiconductor"] is True

    def test_element(self, data_dir):
        info = lammps_tools.parse_data_file(str(data_dir / "si_bulk.data"))
        assert info["elements"] == ["Si"]
        assert info["element_counts"]["Si"] == 8

    def test_atom_style(self, data_dir):
        info = lammps_tools.parse_data_file(str(data_dir / "si_bulk.data"))
        assert info["atom_style"] == "atomic"


class TestParseDataFileIonic:
    """NaCl: atom_style charge, no bonds, ionic crystal."""

    def test_atom_style(self, data_dir):
        info = lammps_tools.parse_data_file(str(data_dir / "nacl.data"))
        assert info["atom_style"] == "charge"

    def test_system_category(self, data_dir):
        info = lammps_tools.parse_data_file(str(data_dir / "nacl.data"))
        assert info["system_category"] == "ionic"
        assert info["has_ions"] is True

    def test_elements(self, data_dir):
        info = lammps_tools.parse_data_file(str(data_dir / "nacl.data"))
        assert "Na" in info["elements"]
        assert "Cl" in info["elements"]
        assert info["element_counts"]["Na"] == 4
        assert info["element_counts"]["Cl"] == 4


class TestParseDataFileOxide:
    """MgO: atom_style charge, no bonds, oxide."""

    def test_system_category(self, data_dir):
        info = lammps_tools.parse_data_file(str(data_dir / "mgo.data"))
        assert info["system_category"] == "oxide"

    def test_elements(self, data_dir):
        info = lammps_tools.parse_data_file(str(data_dir / "mgo.data"))
        assert "Mg" in info["elements"]
        assert "O" in info["elements"]


class TestParseDataFileWater:
    """SPC/E water: atom_style full, with bonds, liquid."""

    def test_atom_style(self, data_dir):
        info = lammps_tools.parse_data_file(str(data_dir / "water_spc.data"))
        assert info["atom_style"] == "full"

    def test_has_bonds(self, data_dir):
        info = lammps_tools.parse_data_file(str(data_dir / "water_spc.data"))
        assert info["has_bonds"] is True
        assert info["bond_count"] == 2

    def test_has_water(self, data_dir):
        info = lammps_tools.parse_data_file(str(data_dir / "water_spc.data"))
        assert info["has_water"] is True

    def test_has_pair_coeffs(self, data_dir):
        info = lammps_tools.parse_data_file(str(data_dir / "water_spc.data"))
        assert info["has_pair_coeffs"] is True

    def test_elements(self, data_dir):
        info = lammps_tools.parse_data_file(str(data_dir / "water_spc.data"))
        assert "O" in info["elements"]
        assert "H" in info["elements"]


class TestParseDataFileBiomolecular:
    """Protein-like: atom_style full, with bonds, C+N+H, has Pair Coeffs."""

    def test_system_category(self, data_dir):
        info = lammps_tools.parse_data_file(str(data_dir / "protein.data"))
        assert info["system_category"] == "biomolecular"

    def test_has_organic(self, data_dir):
        info = lammps_tools.parse_data_file(str(data_dir / "protein.data"))
        assert info["has_organic"] is True

    def test_has_pair_coeffs(self, data_dir):
        info = lammps_tools.parse_data_file(str(data_dir / "protein.data"))
        assert info["has_pair_coeffs"] is True


class TestParseDataFileSlab:
    """Cu slab with vacuum gap."""

    def test_vacuum_detected(self, data_dir):
        info = lammps_tools.parse_data_file(str(data_dir / "cu_slab.data"))
        assert info["has_vacuum"] is True
        assert info["vacuum_axis"] == "z"

    def test_still_classified_as_metal(self, data_dir):
        info = lammps_tools.parse_data_file(str(data_dir / "cu_slab.data"))
        assert info["system_category"] == "metal"


class TestParseDataFileErrors:
    """Edge cases and error handling."""

    def test_nonexistent_file(self):
        info = lammps_tools.parse_data_file("/nonexistent/path.data")
        assert info["atom_count"] == 0
        assert info["elements"] == []

    def test_empty_file(self, tmp_path):
        empty = tmp_path / "empty.data"
        empty.write_text("")
        info = lammps_tools.parse_data_file(str(empty))
        assert info["atom_count"] == 0


# =====================================================================
# format_type_info
# =====================================================================

class TestFormatTypeInfo:

    def test_contains_key_sections(self, data_dir):
        text = lammps_tools.format_type_info(str(data_dir / "cu_bulk.data"))
        assert "DATA FILE ANALYSIS:" in text
        assert "MASS-ELEMENT MAPPING:" in text
        assert "ELEMENT COUNTS:" in text
        assert "Cu" in text

    def test_reports_category(self, data_dir):
        text = lammps_tools.format_type_info(str(data_dir / "cu_bulk.data"))
        assert "metal" in text

    def test_reports_vacuum(self, data_dir):
        text = lammps_tools.format_type_info(str(data_dir / "cu_slab.data"))
        assert "Vacuum gap" in text or "vacuum" in text.lower()


# =====================================================================
# validate_script — valid scripts pass
# =====================================================================

class TestValidateScriptValid:

    def test_valid_metal(self, script_dir):
        r = lammps_tools.validate_script(str(script_dir / "valid_metal.lammps"))
        assert r["valid"] is True, f"Unexpected errors: {r['errors']}"
        assert r["units"] == "metal"
        assert r["atom_style"] == "atomic"
        assert r["pair_style"] == "eam/alloy"
        assert r["has_minimize"] is True
        assert r["has_run"] is True

    def test_valid_biomolecular(self, script_dir):
        r = lammps_tools.validate_script(str(script_dir / "valid_bio.lammps"))
        assert r["valid"] is True, f"Unexpected errors: {r['errors']}"
        assert r["units"] == "real"
        assert r["has_shake"] is True

    def test_valid_ionic(self, script_dir):
        r = lammps_tools.validate_script(str(script_dir / "valid_ionic.lammps"))
        assert r["valid"] is True, f"Unexpected errors: {r['errors']}"
        assert r["atom_style"] == "charge"

    def test_valid_slab(self, script_dir):
        r = lammps_tools.validate_script(str(script_dir / "valid_slab.lammps"))
        assert r["valid"] is True, f"Unexpected errors: {r['errors']}"
        assert r["boundary"] == ["p", "p", "s"]

    def test_valid_create_atoms(self, script_dir):
        # A self-contained deck builds its system in-place with create_atoms
        # (no read_data/read_restart) — a valid, common pattern that must pass.
        r = lammps_tools.validate_script(str(script_dir / "valid_create_atoms.lammps"))
        assert r["valid"] is True, f"Unexpected errors: {r['errors']}"
        assert not any("read_data" in e or "system definition" in e
                       for e in r["errors"])


# =====================================================================
# validate_script — error scripts caught
# =====================================================================

class TestValidateScriptErrors:

    def test_kspace_with_eam(self, script_dir):
        r = lammps_tools.validate_script(str(script_dir / "err_kspace_eam.lammps"))
        assert r["valid"] is False
        assert any("kspace" in e.lower() for e in r["errors"])

    def test_missing_units(self, script_dir):
        r = lammps_tools.validate_script(str(script_dir / "err_no_units.lammps"))
        assert r["valid"] is False
        assert any("units" in e.lower() for e in r["errors"])

    def test_coeff_before_style(self, script_dir):
        r = lammps_tools.validate_script(str(script_dir / "err_coeff_before_style.lammps"))
        assert r["valid"] is False
        assert any("pair_coeff" in e.lower() and "before" in e.lower() for e in r["errors"])

    def test_reaxff_no_qeq(self, script_dir):
        r = lammps_tools.validate_script(str(script_dir / "err_reaxff_no_qeq.lammps"))
        assert r["valid"] is False
        assert any("qeq" in e.lower() for e in r["errors"])

    def test_coul_long_no_kspace(self, script_dir):
        r = lammps_tools.validate_script(str(script_dir / "err_coul_no_kspace.lammps"))
        assert r["valid"] is False
        assert any("kspace" in e.lower() for e in r["errors"])

    def test_bond_style_with_atomic(self, script_dir):
        r = lammps_tools.validate_script(str(script_dir / "err_bond_atomic.lammps"))
        assert r["valid"] is False
        assert any("bond" in e.lower() and "atomic" in e.lower() for e in r["errors"])

    def test_nvt_npt_same_group(self, script_dir):
        r = lammps_tools.validate_script(str(script_dir / "err_nvt_npt_same.lammps"))
        assert r["valid"] is False
        assert any("nvt" in e.lower() and "npt" in e.lower() for e in r["errors"])

    def test_no_run(self, script_dir):
        r = lammps_tools.validate_script(str(script_dir / "err_no_run.lammps"))
        assert r["valid"] is False
        assert any("run" in e.lower() or "nothing" in e.lower() for e in r["errors"])

    def test_unresolved_variables(self, script_dir):
        r = lammps_tools.validate_script(str(script_dir / "err_unresolved.lammps"))
        assert r["valid"] is False
        assert any("template" in e.lower() or "unresolved" in e.lower() for e in r["errors"])

    def test_wrong_timestep_for_units(self, script_dir):
        r = lammps_tools.validate_script(str(script_dir / "err_metal_timestep.lammps"))
        assert r["valid"] is False
        assert any("timestep" in e.lower() for e in r["errors"])

    def test_nonexistent_file(self):
        r = lammps_tools.validate_script("/no/such/file.lammps")
        assert r["valid"] is False


# =====================================================================
# validate_script — warnings (non-fatal)
# =====================================================================

class TestValidateScriptWarnings:

    def test_slab_npt_iso_warns(self, script_dir):
        r = lammps_tools.validate_script(str(script_dir / "warn_slab_npt.lammps"))
        # Valid but should warn about NPT on non-periodic dim
        assert len(r["warnings"]) > 0
        assert any("periodic" in w.lower() or "barostat" in w.lower() for w in r["warnings"])

    def test_no_shake_with_2fs_warns(self, script_dir):
        r = lammps_tools.validate_script(str(script_dir / "warn_no_shake.lammps"))
        assert len(r["warnings"]) > 0
        assert any("shake" in w.lower() for w in r["warnings"])

    def test_metal_tdamp_real_units_warns(self, script_dir):
        r = lammps_tools.validate_script(str(script_dir / "warn_metal_tdamp.lammps"))
        assert len(r["warnings"]) > 0
        assert any("tdamp" in w.lower() or "damp" in w.lower() for w in r["warnings"])

    def test_missing_potential_file_warns(self, script_dir):
        """Valid metal script but potential file doesn't exist."""
        # Write a script pointing to a nonexistent potential
        bad_script = script_dir / "warn_no_potential.lammps"
        bad_script.write_text("""\
units metal
atom_style atomic
boundary p p p
read_data cu_bulk.data
pair_style eam/alloy
pair_coeff * * NONEXISTENT.eam.alloy Cu
timestep 0.001
run 1000
""")
        r = lammps_tools.validate_script(str(bad_script))
        assert any("not found" in w.lower() or "NONEXISTENT" in w for w in r["warnings"])


# =====================================================================
# validate_script with system_info context
# =====================================================================

class TestValidateWithSystemInfo:

    def test_warns_no_pair_coeff_and_no_data_coeffs(self, script_dir):
        """Script has no pair_coeff, data file has no Pair Coeffs section."""
        minimal = script_dir / "bare_script.lammps"
        minimal.write_text("""\
units real
atom_style full
boundary p p p
read_data system.data
pair_style lj/cut/coul/long 12.0
kspace_style pppm 1.0e-5
timestep 1.0
run 1000
""")
        system_info = {"has_pair_coeffs": False}
        r = lammps_tools.validate_script(str(minimal), system_info=system_info)
        assert any("pair" in w.lower() and "coeff" in w.lower() for w in r["warnings"])


# =====================================================================
# clean_script
# =====================================================================

class TestCleanScript:

    def test_removes_markdown_fences(self):
        raw = "```lammps\nunits metal\nrun 1000\n```"
        cleaned = lammps_tools.clean_script(raw)
        assert "```" not in cleaned
        assert "units metal" in cleaned

    def test_removes_bare_fences(self):
        raw = "```\nunits metal\n```"
        cleaned = lammps_tools.clean_script(raw)
        assert "```" not in cleaned

    def test_strips_whitespace(self):
        raw = "  \n\n units metal \n\n  "
        cleaned = lammps_tools.clean_script(raw)
        assert cleaned.startswith("units")

    def test_preserves_content(self):
        raw = "units metal\natom_style atomic\nrun 1000"
        assert lammps_tools.clean_script(raw) == raw


# =====================================================================
# substitute_variables
# =====================================================================

class TestSubstituteVariables:

    def test_temperature(self):
        script = "fix 1 all nvt temp ${temperature} ${temperature} 100.0"
        result = lammps_tools.substitute_variables(script, temperature=500.0)
        assert "500.0" in result
        assert "${temperature}" not in result

    def test_pressure(self):
        script = "fix 1 all npt temp 300 300 100 iso {pressure} {pressure} 1000"
        result = lammps_tools.substitute_variables(script, pressure=10.0)
        assert "10.0" in result
        assert "{pressure}" not in result

    def test_data_filename(self):
        script = "read_data {data_filename}"
        result = lammps_tools.substitute_variables(script, data_filename="my_system.data")
        assert "my_system.data" in result

    def test_multiple_replacements(self):
        script = "temp ${T} ${T} dt ${dt}"
        result = lammps_tools.substitute_variables(
            script, temperature=400.0, timestep=1.0
        )
        assert "400.0" in result
        assert "1.0" in result
        assert "${" not in result

    def test_no_replacement_when_absent(self):
        script = "units metal\nrun 1000"
        result = lammps_tools.substitute_variables(script)
        assert result == script


# =====================================================================
# integrate_force_field_files
# =====================================================================

class TestIntegrateForceFieldFiles:

    def test_coeff_file_after_read_data(self, tmp_path):
        """Coefficient file contents inserted after read_data."""
        ff_file = tmp_path / "params.lammps"
        ff_file.write_text("pair_coeff 1 1 0.1553 3.166\npair_coeff 2 2 0.0 0.0\n")

        script = "units real\natom_style full\nread_data system.data\nrun 1000"
        result = lammps_tools.integrate_force_field_files(
            script, {"ff": str(ff_file)}, str(tmp_path)
        )
        lines = result.split("\n")
        read_pos = next(i for i, l in enumerate(lines) if "read_data" in l)
        coeff_pos = next(i for i, l in enumerate(lines) if "pair_coeff 1 1" in l)
        assert coeff_pos > read_pos

    def test_style_file_before_read_data(self, tmp_path):
        """Style-only file included before read_data, duplicate styles removed."""
        ff_file = tmp_path / "styles.lammps"
        ff_file.write_text("pair_style lj/cut 12.0\nbond_style harmonic\n")

        script = (
            "units real\natom_style full\npair_style lj/cut 10.0\n"
            "bond_style harmonic\nread_data system.data\nrun 1000"
        )
        result = lammps_tools.integrate_force_field_files(
            script, {"styles": str(ff_file)}, str(tmp_path)
        )
        # Original pair_style should be removed
        assert result.count("pair_style") == 0 or "include" in result
        assert "include styles.lammps" in result

    def test_missing_file_skipped(self, tmp_path):
        script = "units real\nread_data system.data\nrun 1000"
        result = lammps_tools.integrate_force_field_files(
            script, {"missing": "/no/such/file.lammps"}, str(tmp_path)
        )
        assert result == script  # Unchanged

    def test_no_read_data_unchanged(self, tmp_path):
        ff_file = tmp_path / "params.lammps"
        ff_file.write_text("pair_coeff 1 1 0.1 3.0\n")

        script = "units real\natom_style full\nrun 1000"
        result = lammps_tools.integrate_force_field_files(
            script, {"ff": str(ff_file)}, str(tmp_path)
        )
        assert result == script  # No read_data → no changes

    def test_empty_dict_unchanged(self, tmp_path):
        script = "units real\nread_data system.data\nrun 1000"
        assert lammps_tools.integrate_force_field_files(script, {}, str(tmp_path)) == script


# =====================================================================
# check_lammps (smoke test — no LAMMPS required in CI)
# =====================================================================

class TestCheckLammps:

    def test_returns_dict_structure(self):
        result = lammps_tools.check_lammps()
        assert "available" in result
        assert "path" in result
        assert "packages" in result
        assert isinstance(result["available"], bool)
        assert isinstance(result["packages"], list)


# =====================================================================
# validate_script — fully-typed FF data file ordering (OpenFF Interchange)
# =====================================================================

class TestValidateTypedDataFileOrdering:
    """A data file with inline coefficients needs *_style BEFORE read_data."""

    _TYPED = {"has_pair_coeffs": True, "bond_types": 4,
              "angle_types": 4, "dihedral_types": 1}

    def _write(self, tmp_path, body):
        p = tmp_path / "in.lammps"
        p.write_text(body)
        return str(p)

    def test_read_data_before_styles_flagged(self, tmp_path):
        bad = (
            "units real\natom_style full\nboundary p p p\n"
            "read_data system.data\n"
            "pair_style lj/cut/coul/long 9.0\nbond_style harmonic\n"
            "angle_style harmonic\ndihedral_style fourier\n"
            "kspace_style pppm 1e-4\nrun 0\n"
        )
        r = lammps_tools.validate_script(self._write(tmp_path, bad), self._TYPED)
        assert r["valid"] is False
        assert any("before 'read_data'" in e.lower() for e in r["errors"])

    def test_styles_before_read_data_passes(self, tmp_path):
        good = (
            "units real\natom_style full\nboundary p p p\n"
            "pair_style lj/cut/coul/long 9.0\nbond_style harmonic\n"
            "angle_style harmonic\ndihedral_style fourier\n"
            "kspace_style pppm 1e-4\nread_data system.data\nrun 0\n"
        )
        r = lammps_tools.validate_script(self._write(tmp_path, good), self._TYPED)
        assert not any("before 'read_data'" in e.lower() for e in r["errors"]), r["errors"]

    def test_bare_data_file_unaffected(self, tmp_path):
        # No inline coeffs -> the classic read_data-then-styles ordering is fine.
        bare = (
            "units metal\natom_style atomic\nboundary p p p\n"
            "read_data system.data\npair_style eam/alloy\n"
            "pair_coeff * * Cu.eam.alloy Cu\nrun 0\n"
        )
        r = lammps_tools.validate_script(self._write(tmp_path, bare),
                                         {"has_pair_coeffs": False})
        assert not any("before 'read_data'" in e.lower() for e in r["errors"]), r["errors"]
