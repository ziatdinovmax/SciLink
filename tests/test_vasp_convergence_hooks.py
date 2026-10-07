"""Tests for the VASP convergence hooks + the `convergence:` frontmatter.

The param-setter is pure string manipulation (no VASP needed). The frontmatter
and registry tests confirm the engine-neutral driver can discover the spec and
the hooks for the active engine. The observable extractor needs real vasprun.xml
output and is validated on the cluster, not here.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import types  # noqa: E402

from scilink.skills.periodic_dft.vasp.vasp_convergence import (  # noqa: E402
    set_convergence_param, get_convergence_param,
    kspacing_to_mesh, read_convergence_observable,
)


class _StructList(list):
    """A final_structure stand-in: len() = atom count, plus a .lattice."""


class TestObservableConvergenceGate:
    """An unconverged rung must read as None, not a bogus number that could
    fake or block a plateau."""

    def _patch_vasprun(self, monkeypatch, tmp_path, **attrs):
        (tmp_path / "vasprun.xml").write_text("<modeling/>")  # just needs to exist
        struct = _StructList(range(attrs.get("natoms", 2)))
        struct.lattice = types.SimpleNamespace(abc=(attrs.get("a", 3.6),) * 3)
        fake = types.SimpleNamespace(
            converged_electronic=attrs.get("converged_electronic", True),
            converged=attrs.get("converged", True),
            final_energy=attrs.get("energy", -10.0),
            final_structure=struct,
            eigenvalue_band_properties=(attrs.get("gap", 1.2),),
        )
        monkeypatch.setattr("pymatgen.io.vasp.outputs.Vasprun",
                            lambda *a, **k: fake)
        return str(tmp_path)

    def test_energy_none_when_scf_unconverged(self, monkeypatch, tmp_path):
        d = self._patch_vasprun(monkeypatch, tmp_path, converged_electronic=False)
        assert read_convergence_observable(d, "energy_per_atom") is None

    def test_energy_value_when_converged(self, monkeypatch, tmp_path):
        d = self._patch_vasprun(monkeypatch, tmp_path,
                                converged_electronic=True, energy=-10.0, natoms=2)
        assert read_convergence_observable(d, "energy_per_atom") == -5.0

    def test_band_gap_none_when_scf_unconverged(self, monkeypatch, tmp_path):
        d = self._patch_vasprun(monkeypatch, tmp_path, converged_electronic=False)
        assert read_convergence_observable(d, "band_gap") is None

    def test_lattice_none_when_relaxation_unconverged(self, monkeypatch, tmp_path):
        # Electronically converged but ionic relaxation did not finish.
        d = self._patch_vasprun(monkeypatch, tmp_path,
                                converged_electronic=True, converged=False)
        assert read_convergence_observable(d, "lattice_constant_a") is None

    def test_lattice_value_when_fully_converged(self, monkeypatch, tmp_path):
        d = self._patch_vasprun(monkeypatch, tmp_path, converged=True, a=3.61)
        assert read_convergence_observable(d, "lattice_constant_a") == 3.61


class TestSetConvergenceParam:
    def test_replaces_existing_encut(self):
        deck = {"INCAR": "PREC = Accurate\nENCUT = 400\nISMEAR = 0\n",
                "POSCAR": "..."}
        out = set_convergence_param(deck, "ENCUT", 600)
        assert "ENCUT = 600" in out["INCAR"]
        assert "ENCUT = 400" not in out["INCAR"]
        assert out["INCAR"].count("ENCUT") == 1
        assert deck["INCAR"].count("400") == 1   # original not mutated

    def test_appends_encut_when_absent(self):
        deck = {"INCAR": "PREC = Accurate\n", "POSCAR": "..."}
        out = set_convergence_param(deck, "ENCUT", 500)
        assert "ENCUT = 500" in out["INCAR"]

    def test_kspacing_sets_incar_and_drops_kpoints(self):
        deck = {"INCAR": "PREC = Accurate\n", "KPOINTS": "mesh\n0\nG\n4 4 4\n",
                "POSCAR": "..."}
        out = set_convergence_param(deck, "k-points", 0.2)
        assert "KSPACING = 0.2" in out["INCAR"]
        assert "KPOINTS" not in out          # dropped so KSPACING governs
        assert "KPOINTS" in deck             # original untouched

    def test_replaces_existing_kspacing(self):
        deck = {"INCAR": "KSPACING = 0.5\nPREC = Accurate\n"}
        out = set_convergence_param(deck, "k-points", 0.15)
        assert "KSPACING = 0.15" in out["INCAR"]
        assert "0.5" not in out["INCAR"]

    def test_case_insensitive_existing_key(self):
        deck = {"INCAR": "encut = 400\n"}
        out = set_convergence_param(deck, "ENCUT", 600)
        assert "ENCUT = 600" in out["INCAR"]
        assert "400" not in out["INCAR"]

    def test_preserves_other_tags_on_the_same_line(self):
        # VASP allows `A = 1; B = 2`; setting ENCUT must not drop PREC.
        deck = {"INCAR": "ENCUT = 500; PREC = Accurate\nISMEAR = 0\n"}
        out = set_convergence_param(deck, "ENCUT", 400)["INCAR"]
        assert "ENCUT = 400" in out
        assert "PREC = Accurate" in out          # survived the substitution
        assert "ENCUT = 500" not in out

    def test_preserves_trailing_comment(self):
        deck = {"INCAR": "ENCUT = 500  # plane-wave cutoff\n"}
        out = set_convergence_param(deck, "ENCUT", 600)["INCAR"]
        assert "ENCUT = 600" in out
        assert "plane-wave cutoff" in out        # comment survived

    def test_edits_in_place_when_tag_is_not_first_on_its_line(self):
        # The bug: ^-anchored match missed a trailing-tag ENCUT and APPENDED a
        # second one; VASP reads the first, so every rung ran at the old value.
        deck = {"INCAR": "PREC = Accurate; ENCUT = 500\nISMEAR = 0\n"}
        out = set_convergence_param(deck, "ENCUT", 400)["INCAR"]
        assert out.count("ENCUT") == 1           # edited in place, not duplicated
        assert "ENCUT = 400" in out and "ENCUT = 500" not in out
        assert "PREC = Accurate" in out

    def test_unknown_param_raises(self):
        with pytest.raises(ValueError):
            set_convergence_param({"INCAR": ""}, "SIGMA", 0.1)

    def test_missing_incar_raises(self):
        with pytest.raises(KeyError):
            set_convergence_param({"POSCAR": "..."}, "ENCUT", 500)


class TestGetConvergenceParam:
    def test_reads_encut(self):
        deck = {"INCAR": "PREC = Accurate\nENCUT = 520\nISMEAR = 0\n"}
        assert get_convergence_param(deck, "ENCUT") == 520.0

    def test_reads_encut_case_insensitive(self):
        assert get_convergence_param({"INCAR": "encut = 400\n"}, "ENCUT") == 400.0

    def test_encut_absent_returns_none(self):
        assert get_convergence_param({"INCAR": "PREC = Accurate\n"}, "ENCUT") is None

    def test_reads_kspacing_from_incar(self):
        assert get_convergence_param({"INCAR": "KSPACING = 0.25\n"}, "k-points") == 0.25

    def test_missing_incar_returns_none(self):
        assert get_convergence_param({"POSCAR": "..."}, "ENCUT") is None

    def test_unknown_param_raises(self):
        with pytest.raises(ValueError):
            get_convergence_param({"INCAR": "ENCUT = 400\n"}, "SIGMA")

    def test_reads_encut_with_trailing_tag_on_same_line(self):
        deck = {"INCAR": "ENCUT = 500 ; PREC = Accurate\n"}
        assert get_convergence_param(deck, "ENCUT") == 500.0

    def test_reads_encut_when_it_is_the_SECOND_tag_on_a_line(self):
        # The bug: a regex anchored at ^ missed ENCUT after a ';'. pymatgen reads it.
        deck = {"INCAR": "PREC = Accurate; ENCUT = 500\nISMEAR = 0\n"}
        assert get_convergence_param(deck, "ENCUT") == 500.0

    # --- k-point density from an explicit KPOINTS mesh (pymatgen/atomate2 decks) ---
    _CUBIC_POSCAR = (
        "Cu\n1.0\n3.5 0 0\n0 3.5 0\n0 0 3.5\nCu\n1\nDirect\n0 0 0\n")

    def test_kspacing_derived_from_kpoints_mesh(self):
        # |b| = 2*pi/3.5 = 1.7952; Gamma 8x8x8 -> effective KSPACING = 1.7952/8.
        deck = {"INCAR": "PREC = Accurate\n",
                "KPOINTS": "Auto\n0\nGamma\n8 8 8\n0 0 0\n",
                "POSCAR": self._CUBIC_POSCAR}
        v = get_convergence_param(deck, "k-points")
        assert v == pytest.approx(1.7952 / 8, abs=1e-3)

    def test_kpoints_mesh_overrides_incar_kspacing(self):
        # VASP lets a KPOINTS file govern over INCAR KSPACING, so the mesh wins.
        deck = {"INCAR": "KSPACING = 0.5\n",
                "KPOINTS": "Auto\n0\nGamma\n8 8 8\n0 0 0\n",
                "POSCAR": self._CUBIC_POSCAR}
        assert get_convergence_param(deck, "k-points") == pytest.approx(1.7952 / 8, abs=1e-3)

    def test_kspacing_none_when_kpoints_mesh_but_no_poscar(self):
        deck = {"INCAR": "PREC = Accurate\n", "KPOINTS": "Auto\n0\nGamma\n4 4 4\n0 0 0\n"}
        assert get_convergence_param(deck, "k-points") is None


class TestKspacingToMesh:
    def _write_run(self, tmp_path, kspacing):
        # Conventional cubic FCC Cu (a=3.61) so the mesh is isotropic and
        # predictable: |b| = 2*pi/3.61 = 1.740 Å^-1.
        from ase.build import bulk
        from ase.io import write
        atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
        write(str(tmp_path / "POSCAR"), atoms, format="vasp", sort=True)
        (tmp_path / "INCAR").write_text(f"PREC = Accurate\nKSPACING = {kspacing}\n")
        return str(tmp_path)

    def test_coarse_kspacing_gives_small_mesh(self, tmp_path):
        # ceil(1.740 / 0.5) = 4 -> 4x4x4
        assert kspacing_to_mesh(self._write_run(tmp_path, 0.5)) == "4×4×4"

    def test_denser_kspacing_gives_larger_mesh(self, tmp_path):
        # ceil(1.740 / 0.2) = 9 -> 9x9x9
        assert kspacing_to_mesh(self._write_run(tmp_path, 0.2)) == "9×9×9"

    def test_none_without_kspacing(self, tmp_path):
        (tmp_path / "INCAR").write_text("PREC = Accurate\nENCUT = 400\n")
        (tmp_path / "POSCAR").write_text("dummy\n")
        assert kspacing_to_mesh(str(tmp_path)) is None

    def test_none_when_files_absent(self, tmp_path):
        assert kspacing_to_mesh(str(tmp_path)) is None


class TestConvergenceFrontmatter:
    def test_vasp_skill_declares_convergence_block(self):
        from scilink.skills.loader import load_skill
        meta = load_skill("vasp", domain="periodic_dft")["meta"]
        conv = meta.get("convergence")
        assert isinstance(conv, list) and conv, "no convergence block in vasp.md"
        params = {c["parameter"] for c in conv}
        assert "ENCUT" in params and "k-points" in params
        for c in conv:
            assert isinstance(c["ladder"], list) and len(c["ladder"]) >= 2
            assert c["observable"] and c["tolerance"] > 0


class TestRegistryResolution:
    def test_hooks_resolve_when_vasp_active(self):
        from scilink.skills._shared._registry import get_tool_function
        setter = get_tool_function("set_convergence_param", active_skills=["vasp"])
        reader = get_tool_function("read_convergence_observable",
                                   active_skills=["vasp"])
        # setter is callable end-to-end
        out = setter(input_files={"INCAR": "PREC = Accurate\n"},
                     param="ENCUT", value=520)
        assert "ENCUT = 520" in out["INCAR"]
        # reader returns None for a missing run dir rather than raising
        assert reader(output_dir="/nonexistent", observable="energy_per_atom") is None
        # getter resolves and reads the deck's current value (the ladder floor)
        getter = get_tool_function("get_convergence_param", active_skills=["vasp"])
        assert getter(input_files={"INCAR": "ENCUT = 520\n"}, param="ENCUT") == 520.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
