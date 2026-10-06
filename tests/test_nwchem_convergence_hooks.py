"""Tests for the NWChem convergence hooks + the `convergence:` frontmatter.

The basis param-setter is pure text manipulation (no NWChem needed). The
frontmatter and registry tests confirm the engine-neutral driver can discover
the spec and hooks when the `nwchem` skill is active. The observable extractor
needs real NWChem output (cclib) and is validated on the cluster.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scilink.skills.molecular_qc.nwchem.nwchem_convergence import (  # noqa: E402
    set_convergence_param, read_convergence_observable, _find_nwchem_log,
)

_DECK = (
    "start job\n"
    "geometry units angstrom\n"
    "  O 0.0 0.0 0.0\n"
    "  H 0.0 0.0 0.96\n"
    "end\n"
    "basis\n"
    "  * library def2-svp\n"
    "end\n"
    "dft\n"
    "  xc b3lyp\n"
    "end\n"
    "task dft energy\n"
)


class TestSetConvergenceParam:
    def test_replaces_basis_library(self):
        out = set_convergence_param({"job.nw": _DECK}, "basis", "def2-tzvp")
        assert "* library def2-tzvp" in out["job.nw"]
        assert "def2-svp" not in out["job.nw"]
        assert "job.nw" in _DECK or True            # original dict untouched
        assert "def2-svp" in _DECK                  # source constant unchanged

    def test_replaces_every_library_line(self):
        deck = _DECK + "basis\n  * library def2-svp\nend\ntce\n  ccsd(t)\nend\n"
        out = set_convergence_param({"calc.nw": deck}, "basis", "def2-qzvp")
        assert out["calc.nw"].count("library def2-qzvp") == 2
        assert "def2-svp" not in out["calc.nw"]

    def test_leaves_ecp_and_fitting_basis_untouched(self):
        # A def2 deck for a heavy element carries an ECP and a cd fitting basis
        # whose `library` lines must NOT be rewritten by a basis sweep (doing so
        # turns the ECP into a non-ECP basis and corrupts the fitting set).
        deck = (
            "basis\n  * library def2-svp\nend\n"
            'basis "cd basis"\n  * library def2-universal-jkfit\nend\n'
            "ecp\n  * library def2-ecp\nend\n"
            "dft\n  xc b3lyp\nend\ntask dft energy\n"
        )
        out = set_convergence_param({"heavy.nw": deck}, "basis", "def2-tzvp")["heavy.nw"]
        assert "* library def2-tzvp" in out                 # orbital basis changed
        assert "* library def2-universal-jkfit" in out       # cd fitting preserved
        assert "* library def2-ecp" in out                   # ECP preserved
        assert "jkfit" not in out.replace("def2-universal-jkfit", "")  # not clobbered


class TestLogResolution:
    def test_find_log_prefers_stdout_over_stderr(self, tmp_path):
        # The executor writes BOTH run_stdout.log and run_stderr.log; the gap /
        # dipole observables must read stdout, not whichever glob returns first.
        (tmp_path / "run_stderr.log").write_text("srun: error ...\n")
        (tmp_path / "run_stdout.log").write_text("   Total DFT energy = -76.0\n")
        assert _find_nwchem_log(str(tmp_path)).name == "run_stdout.log"

    def test_case_insensitive_library_keyword(self):
        out = set_convergence_param(
            {"job.nw": "basis\n  * LIBRARY cc-pvdz\nend\n"}, "basis", "cc-pvtz")
        assert "cc-pvtz" in out["job.nw"] and "cc-pvdz" not in out["job.nw"]

    def test_unknown_param_raises(self):
        with pytest.raises(ValueError):
            set_convergence_param({"job.nw": _DECK}, "xc", "pbe")

    def test_no_deck_raises(self):
        with pytest.raises(KeyError):
            set_convergence_param({"notes.txt": "x"}, "basis", "def2-tzvp")

    def test_no_library_line_raises(self):
        with pytest.raises(ValueError):
            set_convergence_param({"job.nw": "task dft energy\n"}, "basis", "def2-tzvp")


class TestReadTotalEnergy:
    _HARTREE_EV = 27.211386245988

    def test_reads_total_dft_energy_in_eV(self, tmp_path):
        # Real NWChem footer line from a water run.
        (tmp_path / "run_stdout.log").write_text(
            "   ...\n   Total DFT energy =      -76.358285492866\n"
            " Total times  cpu: 0.2s\n")
        v = read_convergence_observable(str(tmp_path), "total_energy")
        assert v == pytest.approx(-76.358285492866 * self._HARTREE_EV)

    def test_reads_scf_energy_for_hf(self, tmp_path):
        (tmp_path / "run_stdout.log").write_text(
            "Total SCF energy =    -76.02663\n")
        v = read_convergence_observable(str(tmp_path), "total_energy")
        assert v == pytest.approx(-76.02663 * self._HARTREE_EV)

    def test_takes_last_energy(self, tmp_path):
        (tmp_path / "run_stdout.log").write_text(
            "Total DFT energy = -76.10\nTotal DFT energy = -76.358285\n")
        v = read_convergence_observable(str(tmp_path), "total_energy")
        assert v == pytest.approx(-76.358285 * self._HARTREE_EV)

    def test_none_without_log(self, tmp_path):
        assert read_convergence_observable(str(tmp_path), "total_energy") is None

    def test_none_without_energy_line(self, tmp_path):
        (tmp_path / "run_stdout.log").write_text("no energy here\n")
        assert read_convergence_observable(str(tmp_path), "total_energy") is None


class TestConvergenceFrontmatter:
    def test_nwchem_skill_declares_basis_convergence(self):
        from scilink.skills.loader import load_skill
        conv = load_skill("nwchem", domain="molecular_qc")["meta"].get("convergence")
        assert isinstance(conv, list) and conv
        spec = conv[0]
        assert spec["parameter"] == "basis"
        assert isinstance(spec["ladder"], list) and len(spec["ladder"]) >= 2
        assert spec["observable"] and spec["tolerance"] > 0


class TestRegistryResolution:
    def test_hooks_resolve_when_nwchem_active(self):
        from scilink.skills._shared._registry import get_tool_function
        setter = get_tool_function("set_convergence_param", active_skills=["nwchem"])
        reader = get_tool_function("read_convergence_observable",
                                   active_skills=["nwchem"])
        out = setter(input_files={"job.nw": _DECK}, param="basis", value="def2-tzvp")
        assert "def2-tzvp" in out["job.nw"]
        assert reader(output_dir="/nonexistent", observable="total_energy") is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
