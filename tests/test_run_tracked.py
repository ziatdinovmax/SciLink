"""#685: model-written code and external engines that ran outside the script
executor now run through it.

- ``run_generated_script`` (knowledge queries, database screening, the
  scalarizer): SciLink's own interpreter, no provider keys in the script's
  environment, the sandbox limits, a Stop or a timeout ends the whole tree.
- ``run_engine`` (LAMMPS, AMBER tools, packmol, training runs): a drop-in for
  ``subprocess.run`` that the user's Stop reaches and whose timeout ends the
  engine's helpers too; the parent environment is kept.
- ``query_knowledge_data`` asks for the sandbox consent every other
  code-generation path asks for."""

import json
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scilink import executors as ex
import test_script_process_groups as pg                         # noqa: E402  (its process helpers)

posix = pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX")


def _script(tmp_path, body):
    p = tmp_path / "s.py"
    p.write_text(textwrap.dedent(body))
    return p


def test_generated_code_runs_on_our_interpreter_without_provider_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "planted-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "planted-secret")
    p = _script(tmp_path, """
        import json, os, sys
        print(json.dumps({"exe": sys.executable, "keys": [k for k in os.environ if k.endswith("_API_KEY")],
                          "argv": sys.argv[1:], "cwd": os.getcwd()}))
    """)
    r = ex.run_generated_script(p, timeout=30, args=["data.csv", 3])
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert r.returncode == 0 and out["exe"] == sys.executable and out["argv"] == ["data.csv", "3"]
    assert out["keys"] == []                                       # the planted keys never reach the script
    assert out["cwd"] == os.getcwd()                               # relative paths it was given still resolve


@posix
def test_a_generated_scripts_timeout_and_a_stop_end_its_whole_tree(tmp_path):
    p = _script(tmp_path, pg.SCRIPT)
    with pytest.raises(subprocess.TimeoutExpired):
        ex.run_generated_script(p, timeout=3, cwd=str(tmp_path))
    assert pg._wait_gone(pg._grandchild(tmp_path))
    (tmp_path / "grandchild.pid").unlink()
    errors = []
    t = threading.Thread(target=lambda: errors.append(
        ex.run_generated_script(p, timeout=120, cwd=str(tmp_path)).returncode))
    t.start()
    gc = pg._grandchild(tmp_path)
    ex.kill_subprocesses_for_thread(t.ident)                       # what a user Stop calls
    t.join(timeout=20)
    assert not t.is_alive() and errors and errors[0] != 0
    assert pg._wait_gone(gc)


@posix
def test_an_engines_timeout_ends_its_helpers_and_the_call_reads_as_subprocess_run(tmp_path):
    p = _script(tmp_path, pg.SCRIPT)
    with pytest.raises(subprocess.TimeoutExpired):
        ex.run_engine([sys.executable, str(p)], cwd=str(tmp_path), timeout=3, capture_output=True, text=True)
    assert pg._wait_gone(pg._grandchild(tmp_path))
    r = ex.run_engine("echo out; echo err 1>&2", shell=True, capture_output=True, text=True, timeout=10)
    assert (r.returncode, r.stdout.strip(), r.stderr.strip()) == (0, "out", "err")
    with pytest.raises(subprocess.CalledProcessError):
        ex.run_engine([sys.executable, "-c", "raise SystemExit(3)"], check=True, timeout=10)
    inp = tmp_path / "in.txt"
    inp.write_text("hello\n")
    with open(inp) as fh:                                           # packmol reads its input from stdin
        r = ex.run_engine([sys.executable, "-c", "import sys; print(sys.stdin.read().upper())"],
                          stdin=fh, capture_output=True, text=True, timeout=10)
    assert r.stdout.strip() == "HELLO"


def test_the_engine_sites_and_the_generated_script_sites_use_the_helpers():
    """Every site #685 named, plus the LAMMPS skill's probe: none calls
    subprocess.run any more, and no generated script is run with a bare
    'python' from PATH."""
    root = Path(ex.__file__).resolve().parent
    for rel in ("agents/sim_agents/refinement.py", "skills/_shared/mlip_tools.py",
                "agents/sim_agents/packmol_agent.py", "agents/sim_agents/reference_measurement.py",
                "agents/sim_agents/force_field_agent.py", "skills/force_field/amber/amber.py",
                "skills/molecular_dynamics/lammps/lammps.py"):
        src = (root / rel).read_text()
        assert "subprocess.run(" not in src and "run_engine(" in src, rel
    for rel in ("agents/planning_agents/orchestrator_tools.py", "agents/planning_agents/scalarizer_agent.py"):
        src = (root / rel).read_text()
        assert '["python", str(' not in src and "run_generated_script(" in src, rel


def test_the_scalarizer_runs_its_script_the_executors_way(tmp_path, monkeypatch):
    from scilink.agents.planning_agents.scalarizer_agent import ScalarizerAgent
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "planted-secret")
    p = _script(tmp_path, """
        import json, os, sys
        print(json.dumps({"metrics": {"n_keys": len([k for k in os.environ if k.endswith("_API_KEY")]),
                                      "arg": sys.argv[1]}}))
    """)
    agent = ScalarizerAgent.__new__(ScalarizerAgent)
    out = agent._execute_script(p, args=["x.csv"])
    assert out["status"] == "success" and out["metrics"] == {"n_keys": 0, "arg": "x.csv"}


def test_the_knowledge_query_asks_for_consent(tmp_path, monkeypatch):
    """It ran model-written code with no consent check at all."""
    from scilink.agents.planning_agents.planning_orchestrator import PlanningOrchestratorAgent, AutonomyLevel
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "false")
    monkeypatch.setattr(ex, "_GLOBAL_SANDBOX_APPROVED", False, raising=False)
    monkeypatch.setattr(ex, "check_security_sandbox_indicators", lambda *a, **k: (0, []))
    ran = []
    import scilink.agents.planning_agents.orchestrator_tools as ot
    monkeypatch.setattr(ot, "run_generated_script", lambda *a, **k: ran.append(a))
    (tmp_path / "data").mkdir()
    orch = PlanningOrchestratorAgent(base_dir=str(tmp_path / "s"), api_key="sk-dummy",
                                     model_name="anthropic/claude-sonnet-4-5",
                                     autonomy_level=AutonomyLevel.AUTONOMOUS, data_dir=str(tmp_path / "data"))
    out = json.loads(orch.tools.functions_map["query_knowledge_data"]("how many rows?", file_name="x.csv"))
    assert out["status"] == "error" and "Sandbox approval declined" in out["message"] and ran == []


@posix
def test_an_engine_run_through_the_simulation_loop_takes_its_helpers_with_it(tmp_path):
    """Through a real engine site: the simulation refinement loop's
    LocalExecutor runs a shell command; on a timeout its grandchild (an MPI
    rank, a solver's helper) used to outlive it."""
    from scilink.agents.sim_agents.refinement import LocalExecutor
    (tmp_path / "engine.py").write_text(pg.SCRIPT)
    out = LocalExecutor(timeout=3).run({}, f"{sys.executable} engine.py", str(tmp_path))
    assert (tmp_path / LocalExecutor.RETURNCODE_FILE).read_text() == "timeout"
    assert pg._wait_gone(pg._grandchild(tmp_path))
    ok = LocalExecutor(timeout=30).run({"in.txt": "x"}, "cat in.txt", str(tmp_path / "ok"))
    assert (tmp_path / "ok" / LocalExecutor.STDOUT_FILE).read_text().strip() == "x"
