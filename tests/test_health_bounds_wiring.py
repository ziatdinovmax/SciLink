"""Wiring tests for the system-aware health-bands opt-in (issue #783).

No LLM, no API key: the deriver and the workflow are monkeypatched. Covers the
two seams #785's review flagged:

* **the flag is reachable** — ``derive_health_bounds`` is a real parameter on
  ``_run_workflow_once`` and the ``run_simulation`` orchestrator tool forwards
  it to ``run_complete_workflow`` (the gap the review found: the flag existed
  only on the private entry point).
* **placement + the #784 seam** — derivation runs AFTER input generation and
  grounds on the EFFECTIVE engine (``run_skill``/``run_domain``), so an MLIP run
  re-pointed to the ``ase`` bundle derives on ``ase``, not the routed classical
  engine; the generated deck is handed to the deriver.
"""

import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import scilink.agents.sim_agents.simulation_pipeline as sp  # noqa: E402


# ── the flag is a real pipeline parameter and rides **kwargs ─────────────────

def test_workflow_signature_has_flags():
    params = inspect.signature(sp._run_workflow_once).parameters
    assert "derive_health_bounds" in params
    assert "health_slack_factor" in params


def test_run_complete_workflow_forwards_kwargs():
    # run_complete_workflow(**kwargs) -> _run_workflow_once, so the opt-in rides
    # through for programmatic callers.
    params = inspect.signature(sp.run_complete_workflow).parameters
    assert any(p.kind == p.VAR_KEYWORD for p in params.values())


# ── placement + the #784 seam ────────────────────────────────────────────────

def _fake_ase_gen(**kw):
    (Path(kw["output_dir"]) / "run_md.py").write_text("print('md')\n")
    return {"status": "success", "runner": "ase",
            "input_files": {"run_md.py": "print('md')\n"},
            "entry_file": "run_md.py"}


def test_derivation_runs_after_gen_on_effective_engine(tmp_path, monkeypatch):
    import scilink.agents.sim_agents.health_bounds as hb_mod

    captured = {}

    class FakeDeriver:
        def __init__(self, **kw):
            pass

        def derive(self, goal, **kw):
            captured.update(kw)
            return [{"observable": "density", "min": 0.2, "max": 5.0,
                     "source": "derived"}]

    monkeypatch.setattr(hb_mod, "HealthBoundsDeriver", FakeDeriver)
    monkeypatch.setattr(sp, "_generate_inputs", _fake_ase_gen)

    structure = tmp_path / "system.data"
    structure.write_text("dummy")
    sp.run_complete_workflow(
        "equilibrate liquid water with an MLIP",
        scale="molecular_dynamics", software="lammps",
        structure_file=str(structure), output_dir=str(tmp_path / "out"),
        api_key="fake-do-not-bill", validate=False,
        derive_health_bounds=True,
    )
    # Grounds on the EFFECTIVE engine (ase), not the routed classical lammps.
    assert captured.get("skill") == "ase"
    assert captured.get("domain") == "molecular_dynamics"
    # The generated deck was handed to the deriver (goal + deck, per #783).
    assert captured.get("input_files", {}).get("run_md.py")


def test_default_off_never_calls_deriver(tmp_path, monkeypatch):
    import scilink.agents.sim_agents.health_bounds as hb_mod

    class Boom:
        def __init__(self, **kw):
            raise AssertionError("deriver constructed with the flag OFF")

    monkeypatch.setattr(hb_mod, "HealthBoundsDeriver", Boom)
    monkeypatch.setattr(sp, "_generate_inputs", _fake_ase_gen)

    structure = tmp_path / "system.data"
    structure.write_text("dummy")
    # Default OFF: no deriver is ever constructed (byte-identical to pre-#783).
    sp.run_complete_workflow(
        "equilibrate liquid water",
        scale="molecular_dynamics", software="lammps",
        structure_file=str(structure), output_dir=str(tmp_path / "out"),
        api_key="fake-do-not-bill", validate=False,
    )


# ── the run_simulation tool forwards the flag (the reachability gap) ─────────

def test_run_simulation_tool_forwards_flag(tmp_path, monkeypatch):
    from scilink.agents.sim_agents import (
        SimulationOrchestratorAgent, SimulationMode,
    )

    orch = SimulationOrchestratorAgent(
        base_dir=str(tmp_path), api_key="fake-do-not-bill",
        model_name="claude-opus-4-6",
        simulation_mode=SimulationMode.AUTONOMOUS,
    )

    captured = {}

    def fake_workflow(description, **kw):
        captured.update(kw)
        return {"final_status": "success", "refinement": {"status": "success"}}

    monkeypatch.setattr(sp, "run_complete_workflow", fake_workflow)

    tool = orch.tools.functions_map["run_simulation"]
    tool("equilibrate liquid water", scale="molecular_dynamics",
         software="lammps", run_command="echo x",
         derive_health_bounds=True, health_slack_factor=7.0)

    assert captured.get("derive_health_bounds") is True
    assert captured.get("health_slack_factor") == 7.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
