"""Tests for the GROMACS molecular_dynamics engine skill.

Covers the deterministic `.mdp` validator, the parser's normalization, and the
agent integration (discovery, tool-registry registration, engine-parameterized
entry filename) that makes GROMACS a first-class MD engine alongside LAMMPS.
"""
import tempfile

import pytest

from scilink.skills.molecular_dynamics.gromacs import gromacs as g


def _mdp(tmp_path, text):
    p = tmp_path / "grompp.mdp"
    p.write_text(text)
    return str(p)


_GOOD_NPT = """\
integrator = md
dt = 0.002
nsteps = 500000
constraints = h-bonds
cutoff-scheme = Verlet
coulombtype = PME
rvdw = 1.0
rcoulomb = 1.0
tcoupl = v-rescale
tc-grps = Protein Water
ref-t = 300 300
tau-t = 0.1 0.1
pcoupl = c-rescale
ref-p = 1.0
tau-p = 2.0
compressibility = 4.5e-5
"""

_GOOD_NVT = """\
integrator = md
dt = 0.001
cutoff-scheme = Verlet
coulombtype = PME
rvdw = 1.0
rcoulomb = 1.0
tcoupl = nose-hoover
tc-grps = System
ref-t = 310
tau-t = 0.5
nsteps = 100000
"""

_MINIMIZE = """\
integrator = steep
emtol = 1000.0
nsteps = 50000
cutoff-scheme = Verlet
coulombtype = PME
rvdw = 1.0
rcoulomb = 1.0
"""


# ── parser ─────────────────────────────────────────────────────────

def test_parse_mdp_normalizes_dashes_and_comments():
    txt = "nstxout_compressed = 5000  ; trajectory\nDT = 0.002\n"
    mdp = g.parse_mdp(txt)
    assert mdp["nstxout-compressed"] == "5000"   # '_' -> '-'
    assert mdp["dt"] == "0.002"                   # lowercased key
    assert ";" not in "".join(mdp.values())       # comment stripped


def test_parse_mdp_last_assignment_wins():
    mdp = g.parse_mdp("ref-t = 300\nref-t = 310\n")
    assert mdp["ref-t"] == "310"


# ── validator ──────────────────────────────────────────────────────

def test_good_npt_is_valid(tmp_path):
    r = g.validate_script(_mdp(tmp_path, _GOOD_NPT))
    assert r["valid"] is True
    assert r["errors"] == []
    assert r["ensemble"] == "NPT"
    assert r["thermostat"] == "v-rescale"
    assert r["barostat"] == "c-rescale"


def test_good_nvt_has_no_barostat(tmp_path):
    r = g.validate_script(_mdp(tmp_path, _GOOD_NVT))
    assert r["valid"] is True
    assert r["ensemble"] == "NVT"
    assert r["barostat"] is None


def test_minimizer_valid_without_thermostat(tmp_path):
    """A steepest-descent minimization needs no thermostat/timestep-constraint."""
    r = g.validate_script(_mdp(tmp_path, _MINIMIZE))
    assert r["valid"] is True
    assert r["integrator"] == "steep"


def test_ref_t_group_mismatch_is_an_error(tmp_path):
    txt = _GOOD_NVT.replace("tc-grps = System", "tc-grps = Protein Water")
    r = g.validate_script(_mdp(tmp_path, txt))
    assert r["valid"] is False
    assert any("tc-grps" in e for e in r["errors"])


def test_npt_missing_barostat_params_is_an_error(tmp_path):
    txt = ("integrator = md\ndt = 0.002\nconstraints = h-bonds\n"
           "tcoupl = v-rescale\ntc-grps = System\nref-t = 300\n"
           "pcoupl = parrinello-rahman\n")   # no ref-p / tau-p / compressibility
    r = g.validate_script(_mdp(tmp_path, txt))
    assert r["valid"] is False
    assert sum("missing" in e for e in r["errors"]) >= 3


def test_large_timestep_without_constraints_warns(tmp_path):
    txt = ("integrator = md\ndt = 0.004\ntcoupl = v-rescale\n"
           "tc-grps = System\nref-t = 300\ncutoff-scheme = Verlet\n")
    r = g.validate_script(_mdp(tmp_path, txt))
    assert any("dt" in w and "constraint" in w for w in r["warnings"])


def test_unreadable_file_is_invalid():
    r = g.validate_script("/no/such/file.mdp")
    assert r["valid"] is False
    assert r["errors"]


# ── agent integration ──────────────────────────────────────────────

def test_gromacs_is_a_discovered_md_skill():
    from scilink.skills.loader import list_skills
    assert "gromacs" in list_skills(domain="molecular_dynamics")


def test_gromacs_is_registered_in_the_tool_registry():
    from scilink.agents.sim_agents.md_simulation_agent import _TOOL_REGISTRY
    assert "gromacs" in _TOOL_REGISTRY
    assert hasattr(_TOOL_REGISTRY["gromacs"], "validate_script")


def test_entry_filename_is_engine_parameterized():
    from scilink.agents.sim_agents.md_simulation_agent import MDSimulationAgent
    a = MDSimulationAgent(working_dir=tempfile.mkdtemp(),
                          api_key="dummy", base_url="http://localhost:0")
    a.skill_name = "gromacs"
    assert a._entry_name() == "grompp.mdp"
    assert a._entry_name("equilibration") == "grompp_equilibration.mdp"
    a.skill_name = "lammps"                      # unchanged for LAMMPS
    assert a._entry_name() == "run.lammps"
    assert a._entry_name("production") == "run_production.lammps"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
