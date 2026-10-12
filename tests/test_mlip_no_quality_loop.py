"""The execution path must NOT invoke MLIP's bespoke active-learning loop.

NON-GOAL of MLIP-execution-in-loop: the execution path reaches MLIP only via
MLIPAgent.deploy_pretrained. evaluate_simulation_quality / refine_potential
(the uncertainty + DFT fine-tune loop) stay out of it entirely.

Enforced two ways:
  1. a source scan: neither simulation_pipeline.py nor refinement.py may even
     name those methods;
  2. a runtime drive of the execution path with a fake MLIPAgent whose
     quality-loop methods blow up if called — they must never fire.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scilink.agents.sim_agents import (  # noqa: E402
    critics,
    mlip_agent,
    potential_selection,
    refinement,
    simulation_pipeline,
)
from scilink.agents.sim_agents.refinement import Executor  # noqa: E402

_PIPE = Path(simulation_pipeline.__file__)
_REFINE = Path(refinement.__file__)


def test_source_never_references_the_quality_loop():
    for src in (_PIPE.read_text(), _REFINE.read_text()):
        assert "evaluate_simulation_quality" not in src
        assert "refine_potential" not in src


class FakeExecutor(Executor):
    def run(self, input_files, run_command, run_dir):
        return {"status": "completed", "output_dir": run_dir, "returncode": 0}


class _QualityLoopTripwire:
    """Stand-in MLIPAgent: deploy works; the quality loop is a tripwire."""

    def __init__(self, working_dir, **kw):
        self.working_dir = working_dir

    def deploy_pretrained(self, **kw):
        p = Path(self.working_dir) / "run_md.py"
        p.write_text("import ase\n\ndef main():\n    pass\n")
        return {"run_path": str(p), "runner": "ase", "task": "md"}

    def evaluate_simulation_quality(self, *a, **k):  # pragma: no cover
        raise AssertionError("execution path called evaluate_simulation_quality")

    def refine_potential(self, *a, **k):  # pragma: no cover
        raise AssertionError("execution path called refine_potential")


def test_execution_path_never_calls_quality_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(mlip_agent, "MLIPAgent", _QualityLoopTripwire)
    # force the MLIP family without an LLM call in potential selection
    monkeypatch.setattr(
        potential_selection, "select_potential_family",
        lambda **kw: {"family": "mlip", "source": "pretrained", "reasoning": ""},
    )
    monkeypatch.setattr(critics, "RunCritic", lambda **kw: object())
    monkeypatch.setattr(
        refinement, "run_campaign",
        lambda stages, ex, rc, pol, ctx, pre_run_verdict=None: {"status": "success"},
    )

    result = simulation_pipeline._run_workflow_once(
        "MD of a Pt slab at 300 K",
        scale="molecular_dynamics", software="lammps",
        structure_file=str(tmp_path / "structure.cif"),
        force_field_files={"noop": "x"},     # skip the step-1.5 FF block
        output_dir=str(tmp_path), validate=False,
        executor=FakeExecutor(), run_command="lmp -in {script}",
    )

    # it ran the ASE path to completion WITHOUT touching the quality loop
    assert result["final_status"] == "success"
    assert result["input_generation"]["runner"] == "ase"
