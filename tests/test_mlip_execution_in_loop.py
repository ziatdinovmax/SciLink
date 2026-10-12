"""The MLIP/ASE path executes in-loop like the classical path.

Drives _run_workflow_once with a gen_result that carries runner="ase" and
path-free input_files, a FakeExecutor, and a run_command inherited from the
routed classical engine ("lmp -in {script}"). The execution seam must:
  (a) re-resolve run_command from the ase bundle, so the Phase runs
      `<python> run_md.py`, NOT `lmp -in ...`;
  (b) re-point the RefinementContext at skill="ase" / domain="molecular_dynamics";
  (c) actually invoke the executor (reach the refinement campaign, not the
      "executor provided without a run_command" skip).
(MLIP-execution-in-loop, Blocker 2.)
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scilink.agents.sim_agents import critics, refinement  # noqa: E402
from scilink.agents.sim_agents import simulation_pipeline  # noqa: E402
from scilink.agents.sim_agents.refinement import Executor  # noqa: E402


class FakeExecutor(Executor):
    def __init__(self):
        self.calls = []

    def run(self, input_files, run_command, run_dir):
        self.calls.append((dict(input_files or {}), run_command, run_dir))
        return {"status": "completed", "output_dir": run_dir, "returncode": 0}


def _gen_result():
    return {
        "status": "success",
        "runner": "ase",
        "entry_file": "run_md.py",
        "input_files": {"run_md.py": "import ase\n\ndef main():\n    pass\n"},
        "potential_selection": {"family": "mlip", "source": "pretrained"},
    }


def _setup(monkeypatch, run_command):
    """Patch generation, the critic and the campaign; return a capture dict."""
    monkeypatch.setattr(simulation_pipeline, "_generate_inputs",
                        lambda **kw: _gen_result())
    # RunCritic is constructed before run_campaign; avoid credentials.
    monkeypatch.setattr(critics, "RunCritic", lambda **kw: object())

    captured = {}

    def fake_run_campaign(stages, executor, run_critic, policy, ctx,
                          pre_run_verdict=None):
        captured["stages"] = stages
        captured["ctx"] = ctx
        # prove the executor is wired through (not the skipped early return)
        ph = stages[0].phases[0]
        executor.run(ph.input_files, ph.run_command, ph.run_dir)
        return {"status": "success"}

    monkeypatch.setattr(refinement, "run_campaign", fake_run_campaign)
    return captured


def _run(executor, run_command, tmp_path):
    return simulation_pipeline._run_workflow_once(
        "MD of Pt bulk at 300 K",
        scale="molecular_dynamics", software="lammps",
        structure_file=str(tmp_path / "structure.cif"),  # skip structure gen
        force_field_files={"noop": "x"},  # skip FF/potential-selection block
        output_dir=str(tmp_path), validate=False,
        executor=executor, run_command=run_command,
    )


def test_classical_run_command_is_overridden_to_python(tmp_path, monkeypatch):
    captured = _setup(monkeypatch, "lmp -in {script}")
    ex = FakeExecutor()
    result = _run(ex, "lmp -in {script}", tmp_path)

    # (a) the run command the Phase carries is a python launch, not lammps
    phase = captured["stages"][0].phases[0]
    assert phase.run_command == f"{sys.executable} run_md.py"
    assert "python" in phase.run_command.lower()
    assert "lmp" not in phase.run_command

    # (b) refinement is grounded on the ase bundle
    assert captured["ctx"].skill == "ase"
    assert captured["ctx"].domain == "molecular_dynamics"

    # (c) the executor actually ran — not the skipped early return
    assert len(ex.calls) == 1
    ran_inputs, ran_cmd, _ = ex.calls[0]
    assert "import ase" in ran_inputs["run_md.py"]
    assert ran_cmd == f"{sys.executable} run_md.py"
    assert result["final_status"] == "success"
    assert result.get("refinement", {}).get("status") == "success"


def test_user_supplied_python_run_command_is_preserved(tmp_path, monkeypatch):
    # A run_command that already launches python is a deliberate override and
    # must survive the seam untouched (only skill/domain re-point).
    captured = _setup(monkeypatch, "python -O {script}")
    ex = FakeExecutor()
    _run(ex, "python -O {script}", tmp_path)

    phase = captured["stages"][0].phases[0]
    assert phase.run_command == "python -O run_md.py"
    assert captured["ctx"].skill == "ase"
