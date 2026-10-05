"""Wiring test for the opt-in parameter-convergence pipeline step.

Exercises the Step 3.5 integration in `_run_workflow_once` with a fake
executor and monkeypatched skill hooks (no LLM, no VASP), mirroring
tests/test_pipeline_refinement_integration.py's approach. The real VASP
end-to-end is validated on the cluster.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import scilink.agents.sim_agents.simulation_pipeline as sp  # noqa: E402
import scilink.skills._shared._registry as reg  # noqa: E402
import scilink.agents.sim_agents.refinement as rf  # noqa: E402


class _FakeExecutor:
    def run(self, input_files, run_command, run_dir):
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        for name, contents in (input_files or {}).items():
            (Path(run_dir) / name).write_text(contents)
        return {"status": "completed", "output_dir": run_dir, "returncode": 0}


def _install_fakes(monkeypatch, energies, spec):
    # Base deck from generation (no LLM).
    monkeypatch.setattr(sp, "_generate_inputs", lambda **kw: {
        "status": "success",
        "input_files": {"INCAR": "PREC = Accurate\n", "POSCAR": "Si\n"},
        "entry_file": None,
    })
    monkeypatch.setattr(sp, "_convergence_specs", lambda software, scale: [spec])

    def fake_get_tool_function(name, active_skills=None):
        if name == "set_convergence_param":
            return lambda input_files, param, value: {
                **input_files, "INCAR": f"{param} = {value}\n"}
        if name == "read_convergence_observable":
            # run_dir basename is the ladder setting.
            return lambda output_dir, observable: energies.get(Path(output_dir).name)
        raise LookupError(name)

    monkeypatch.setattr(reg, "get_tool_function", fake_get_tool_function)
    # Skip the real production run in Step 4.
    monkeypatch.setattr(sp, "_collect_stages", lambda *a, **k: [])
    monkeypatch.setattr(rf, "run_campaign", lambda *a, **k: {"status": "success"})


def _run(tmp_path, **overrides):
    structure = tmp_path / "POSCAR"
    structure.write_text("Si\n")
    kwargs = dict(
        scale="periodic_dft", software="vasp",
        structure_file=str(structure), output_dir=str(tmp_path / "out"),
        validate=False, executor=_FakeExecutor(), run_command="vasp_std",
        converge_parameters=True, api_key="k",
    )
    kwargs.update(overrides)
    return sp.run_complete_workflow("relax Si and report a", **kwargs)


def test_step_runs_sweep_and_adopts_converged_setting(tmp_path, monkeypatch):
    _install_fakes(
        monkeypatch,
        energies={"300": -5.0, "400": -5.401, "500": -5.4012},
        spec={"parameter": "ENCUT", "ladder": [300, 400, 500],
              "observable": "energy_per_atom", "tolerance": 0.001},
    )
    result = _run(tmp_path)

    pc = result["parameter_convergence"]
    assert pc["all_converged"] is True
    assert pc["parameters"]["ENCUT"]["converged"] is True
    assert pc["parameters"]["ENCUT"]["setting"] == 400
    assert "parameter_convergence" in result["steps_completed"]
    # The converged value was written into the deck the production run uses.
    assert result["input_generation"]["input_files"]["INCAR"].strip() == "ENCUT = 400"
    # And onto disk.
    assert (tmp_path / "out" / "INCAR").read_text().strip() == "ENCUT = 400"


def test_step_reports_non_convergence_without_adopting(tmp_path, monkeypatch):
    _install_fakes(
        monkeypatch,
        energies={"300": -5.0, "400": -5.2, "500": -5.4},   # still drifting
        spec={"parameter": "ENCUT", "ladder": [300, 400, 500],
              "observable": "energy_per_atom", "tolerance": 0.001},
    )
    result = _run(tmp_path)
    pc = result["parameter_convergence"]
    assert pc["all_converged"] is False
    assert pc["parameters"]["ENCUT"]["converged"] is False


def test_step_skipped_when_flag_off(tmp_path, monkeypatch):
    _install_fakes(
        monkeypatch,
        energies={"300": -5.0, "400": -5.401, "500": -5.4012},
        spec={"parameter": "ENCUT", "ladder": [300, 400, 500],
              "observable": "energy_per_atom", "tolerance": 0.001},
    )
    result = _run(tmp_path, converge_parameters=False)
    assert "parameter_convergence" not in result
    assert "parameter_convergence" not in result["steps_completed"]


def test_step_skipped_for_md_scale(tmp_path, monkeypatch):
    # MD is not a static scale — the DFT/QC convergence step must not fire.
    _install_fakes(
        monkeypatch,
        energies={"300": -5.0, "400": -5.401, "500": -5.4012},
        spec={"parameter": "ENCUT", "ladder": [300, 400, 500],
              "observable": "energy_per_atom", "tolerance": 0.001},
    )
    result = _run(tmp_path, scale="molecular_dynamics", software="lammps")
    assert "parameter_convergence" not in result


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
