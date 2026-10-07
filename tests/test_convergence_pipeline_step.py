"""Wiring test for the opt-in parameter-convergence pipeline step.

Exercises the Step 3.5 integration in `_run_workflow_once` with a fake
executor and monkeypatched skill hooks (no LLM, no VASP), mirroring
tests/test_pipeline_refinement_integration.py's approach. The real VASP
end-to-end is validated on the cluster.
"""

import os
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


def test_adoption_removes_base_file_the_converged_setting_dropped(tmp_path, monkeypatch):
    # k-point convergence drops the KPOINTS file (INCAR KSPACING supersedes it).
    # The stale KPOINTS that generation wrote into output_dir must be removed,
    # or the production run (which executes in output_dir) reads it and silently
    # ignores the converged spacing.
    monkeypatch.setattr(sp, "_generate_inputs", lambda **kw: {
        "status": "success",
        "input_files": {"INCAR": "PREC = Accurate\n", "POSCAR": "Cu\n",
                        "KPOINTS": "Gamma\n0\nAuto\n20\n"},
        "entry_file": None,
    })
    spec = {"parameter": "k-points", "ladder": [0.5, 0.3, 0.2],
            "observable": "energy_per_atom", "tolerance": 0.005}
    monkeypatch.setattr(sp, "_convergence_specs", lambda software, scale: [spec])
    energies = {"0.5": -5.0, "0.3": -5.41, "0.2": -5.412}

    def fake_get_tool_function(name, active_skills=None):
        if name == "set_convergence_param":
            def _set(input_files, param, value):      # k-points: drop KPOINTS
                out = {k: v for k, v in input_files.items() if k != "KPOINTS"}
                out["INCAR"] = f"KSPACING = {value}\n"
                return out
            return _set
        if name == "read_convergence_observable":
            return lambda output_dir, observable: energies.get(Path(output_dir).name)
        raise LookupError(name)

    monkeypatch.setattr(reg, "get_tool_function", fake_get_tool_function)
    monkeypatch.setattr(sp, "_collect_stages", lambda *a, **k: [])
    monkeypatch.setattr(rf, "run_campaign", lambda *a, **k: {"status": "success"})

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "KPOINTS").write_text("Gamma\n0\nAuto\n20\n")   # as generation wrote

    structure = tmp_path / "POSCAR"
    structure.write_text("Cu\n")
    result = sp.run_complete_workflow(
        "converge Cu k-points", scale="periodic_dft", software="vasp",
        structure_file=str(structure), output_dir=str(out_dir),
        validate=False, executor=_FakeExecutor(), run_command="vasp_std",
        converge_parameters=True, api_key="k")

    assert "parameter_convergence" in result["steps_completed"]
    # Converged spacing adopted into the deck, and the stale KPOINTS removed.
    assert result["input_generation"]["input_files"]["INCAR"].startswith("KSPACING")
    assert "KPOINTS" not in result["input_generation"]["input_files"]
    assert not (out_dir / "KPOINTS").exists()


def test_step_folder_cleared_before_running_a_rung(tmp_path, monkeypatch):
    # A step folder is reused across a structure retry; a stale vasprun.xml left
    # from a previous attempt (a different structure) must not survive into the
    # rung's run, or the reader accepts it as a valid result.
    seen_at_entry = {}

    class _CapturingExecutor:
        def run(self, input_files, run_command, run_dir):
            seen_at_entry[Path(run_dir).name] = set(os.listdir(run_dir))
            Path(run_dir).mkdir(parents=True, exist_ok=True)
            for name, contents in (input_files or {}).items():
                (Path(run_dir) / name).write_text(contents)
            return {"status": "completed", "output_dir": run_dir, "returncode": 0}

    _install_fakes(
        monkeypatch,
        energies={"300": -5.0, "400": -5.401, "500": -5.4012},
        spec={"parameter": "ENCUT", "ladder": [300, 400, 500],
              "observable": "energy_per_atom", "tolerance": 0.001},
    )
    out_dir = tmp_path / "out"
    stale = out_dir / "convergence" / "ENCUT" / "400"
    stale.mkdir(parents=True)
    (stale / "vasprun.xml").write_text("<previous attempt, different structure/>")

    _run(tmp_path, executor=_CapturingExecutor(), output_dir=str(out_dir))

    # The 400 rung ran against a cleared (empty) folder — the stale file is gone.
    assert "vasprun.xml" not in seen_at_entry["400"]
    assert seen_at_entry["400"] == set()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
