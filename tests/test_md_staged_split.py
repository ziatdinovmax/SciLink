"""Unit tests for ``MDSimulationAgent.generate_staged_simulation``'s per-stage
validate-and-retry (no LLM, no engine).

Covers the branches added by the staged-split fix:
  1. a clean split validates in one attempt;
  2. a split that keeps two integrators self-corrects on re-split;
  3. a persistently bad split is repaired by the per-deck ``_attempt_fix``
     fallback (no hard fail);
  4. a persistently bad split that ``_attempt_fix`` cannot repair fails
     generation (``status == "error"``).

Validation uses the REAL ``lammps.validate_script``; only the LLM-facing methods
and the monolithic ``generate_simulation`` are stubbed, so the test exercises the
actual splitting / validation / retry control flow.
"""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import scilink.agents.sim_agents.md_simulation_agent as M
from scilink.agents.sim_agents.base_agent import SimulationAgent
import scilink.skills.molecular_dynamics.lammps.lammps as lammps_tools

# A clean single-integrator equilibration stage (NPT, unfixed) ...
_EQ = ("units real\natom_style full\nread_data system.data\n"
       "minimize 1.0e-4 1.0e-6 5000 50000\n"
       "fix eq all npt temp 298 298 100 iso 1 1 1000\nrun 1000\nunfix eq\n"
       "write_restart eq.restart\n")
# ... a clean single-integrator production stage (NVT) ...
_PROD = ("units real\natom_style full\nread_restart eq.restart\n"
         "fix prod all nvt temp 298 298 100\nrun 1000\nwrite_restart prod.restart\n")
# ... and a BAD stage that kept both integrators on the same group (no unfix).
_PROD_BAD = ("units real\natom_style full\nread_data system.data\n"
             "fix eq all npt temp 298 298 100 iso 1 1 1000\nrun 1000\n"
             "fix prod all nvt temp 298 298 100\nrun 1000\n")


def _agent(tmp, split_seq, attempt_fix=lambda s, e, p: s):
    """A minimally-wired MDSimulationAgent: real _validate/_entry_name against
    the lammps tools, stubbed LLM-facing methods and monolithic generation."""
    a = object.__new__(M.MDSimulationAgent)
    a.working_dir = Path(tmp)
    a.skill_name = "lammps"
    a.tools_module = lammps_tools
    a.logger = MagicMock()
    a._validate = SimulationAgent._validate.__get__(a)
    a._entry_name = M.MDSimulationAgent._entry_name.__get__(a)
    a._get_skill_context = lambda section=None: ""
    a._campaign_shared_files = lambda *x, **k: {}
    a._clean_and_fix = lambda s, plan: s
    a._attempt_fix = attempt_fix
    a._generate_json = MagicMock(side_effect=list(split_seq))
    a.generate_simulation = lambda **k: {
        "script_path": str(Path(tmp) / "run.lammps"),
        "simulation_parameters": {}, "system_info": {},
        "validation": {"valid": False, "errors": ["monolithic conflict"],
                       "warnings": []},
    }
    Path(tmp, "run.lammps").write_text(_PROD_BAD)   # the transient monolithic deck
    return a


def test_clean_split_validates_in_one_attempt():
    a = _agent(tempfile.mkdtemp(), [{"equilibration": _EQ, "production": _PROD}])
    r = a.generate_staged_simulation("system.data", "goal")
    assert r["validation"]["valid"] is True
    assert r.get("status") != "error"
    assert a._generate_json.call_count == 1
    assert "input_files" in r


def test_conflicted_split_self_corrects_on_resplit():
    a = _agent(tempfile.mkdtemp(),
               [{"production": _PROD_BAD}, {"equilibration": _EQ, "production": _PROD}])
    r = a.generate_staged_simulation("system.data", "goal")
    assert r["validation"]["valid"] is True
    assert r.get("status") != "error"
    assert a._generate_json.call_count == 2


def test_persistent_bad_split_repaired_by_attempt_fix():
    a = _agent(tempfile.mkdtemp(), [{"production": _PROD_BAD}] * 3,
               attempt_fix=lambda s, e, p: _PROD)     # fixer returns a clean deck
    r = a.generate_staged_simulation("system.data", "goal")
    assert r["validation"]["valid"] is True
    assert r.get("status") != "error"


def test_persistent_bad_split_unrepairable_fails_generation():
    a = _agent(tempfile.mkdtemp(), [{"production": _PROD_BAD}] * 3,
               attempt_fix=lambda s, e, p: _PROD_BAD)  # fixer cannot repair it
    r = a.generate_staged_simulation("system.data", "goal")
    assert r["validation"]["valid"] is False
    assert r["status"] == "error"
    assert any("both" in e.lower() for e in r["validation"]["errors"])
