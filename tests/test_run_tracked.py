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
import time
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
    """A LINT, not a behaviour test (the behaviour is pinned by the timeout,
    Stop and cancel tests through real runs): every site #685 named, plus the
    LAMMPS skill's probe, calls the helpers — so a later edit that reverts one
    to ``subprocess.run`` or a bare 'python' from PATH is caught."""
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
    csv = tmp_path / "data" / "x.csv"
    csv.write_text("a,b\n1,2\n")                                # consent is asked once the data is found
    orch = PlanningOrchestratorAgent(base_dir=str(tmp_path / "s"), api_key="sk-dummy",
                                     model_name="anthropic/claude-sonnet-4-5",
                                     autonomy_level=AutonomyLevel.AUTONOMOUS, data_dir=str(tmp_path / "data"))
    out = json.loads(orch.tools.functions_map["query_knowledge_data"]("how many rows?", file_name=str(csv)))
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


@posix
def test_a_cancelled_worker_starts_no_further_engine_run(tmp_path):
    """A budget / memory cancel sets the worker's event and kills its process
    once. The killed engine came back as an ordinary result, so the worker
    started its next run, which then ran to the end (#760 review). Now the
    cancel ends the worker at once and the second run never starts."""
    from scilink.agents.sim_agents.refinement import LocalExecutor
    from scilink.utils import log_context
    cancel, errors, started = threading.Event(), [], []

    def worker():
        log_context.register_cancel(cancel)
        try:
            for k in (1, 2):
                started.append(k)
                LocalExecutor(timeout=60).run({}, f"touch run{k}.txt; sleep 8", str(tmp_path / f"r{k}"))
        except BaseException as e:  # noqa: BLE001 - the cancel arrives as AgentStoppedError
            errors.append(type(e).__name__)
        finally:
            log_context.unregister_cancel()

    t = threading.Thread(target=worker)
    t0 = time.monotonic()
    t.start()
    while not (tmp_path / "r1" / "run1.txt").exists() and time.monotonic() - t0 < 10:
        time.sleep(0.05)
    cancel.set()
    ex.kill_subprocesses_for_thread(t.ident)                        # what the swarm's cancel does
    t.join(timeout=20)
    assert not t.is_alive() and errors == ["AgentStoppedError"] and started == [1]
    assert time.monotonic() - t0 < 6 and not (tmp_path / "r2").exists()


def test_run_engine_passes_encoding_and_refuses_unknown_keywords():
    r = ex.run_engine([sys.executable, "-c", "print('é')"], capture_output=True, encoding="utf-8", timeout=10)
    assert r.stdout.strip() == "é"
    with pytest.raises(TypeError):
        ex.run_engine([sys.executable, "-c", "pass"], bufsize=0)


def test_without_a_cancel_nothing_changes(tmp_path):
    from scilink.agents.sim_agents.refinement import LocalExecutor
    LocalExecutor(timeout=30).run({}, "echo ok", str(tmp_path / "a"))
    assert (tmp_path / "a" / LocalExecutor.STDOUT_FILE).read_text().strip() == "ok"


@posix
@pytest.mark.parametrize("kind", ["routed", "output"])
def test_the_turns_own_stop_ends_the_engine_and_starts_no_next_run(tmp_path, kind):
    """A Stop from the web UI or the shell (#760 re-review): the capture's
    stop event is the agent thread's cancel for the turn, so the stopped
    engine ends the turn and the next engine run never starts. What the
    stopped run wrote is kept beside it."""
    from scilink.agents.sim_agents.refinement import LocalExecutor
    from scilink.server.stdout_router import RoutedCapture
    from scilink.ui.output_capture import OutputCapture
    from scilink.utils import log_context
    cap = RoutedCapture(echo_console=False) if kind == "routed" else OutputCapture()
    errors, started = [], []

    def turn():
        try:
            with cap:
                for k in (1, 2):
                    started.append(k)
                    LocalExecutor(timeout=60).run({}, f"echo partial{k}; touch run{k}.txt; sleep 8",
                                                  str(tmp_path / f"r{k}"))
        except BaseException as e:  # noqa: BLE001 - the Stop arrives as AgentStoppedError
            errors.append(type(e).__name__)
        finally:
            errors.append("cancel left registered" if log_context.current_cancel() is not None else "clean")

    t = threading.Thread(target=turn)
    t0 = time.monotonic()
    t.start()
    while not (tmp_path / "r1" / "run1.txt").exists() and time.monotonic() - t0 < 10:
        time.sleep(0.05)
    cap.request_stop()                                               # what the Stop button calls
    t.join(timeout=20)
    assert not t.is_alive() and errors == ["AgentStoppedError", "clean"] and started == [1]
    assert time.monotonic() - t0 < 6 and not (tmp_path / "r2").exists()
    assert (tmp_path / "r1" / LocalExecutor.STDOUT_FILE).read_text().strip() == "partial1"
    assert (tmp_path / "r1" / LocalExecutor.RETURNCODE_FILE).read_text() == "stopped"


def test_a_capture_restores_the_cancel_it_found():
    from scilink.ui.output_capture import OutputCapture
    from scilink.utils import log_context
    outer = threading.Event()
    log_context.register_cancel(outer)
    try:
        with OutputCapture() as cap:
            assert not log_context.cancel_requested()
            cap._stop_event.set()                           # the turn's Stop
            assert log_context.cancel_requested()
        assert log_context.current_cancel() is outer and not outer.is_set()
    finally:
        log_context.unregister_cancel()


def test_a_capture_adds_the_turns_stop_to_the_cancel_it_found():
    """A capture on a thread that already carries a cancel (a worker's budget
    or memory cancel) keeps that cancel live for the turn: either ends a wait
    (#760 review)."""
    from scilink.ui.output_capture import AgentStoppedError, OutputCapture
    from scilink.utils import log_context
    outer = threading.Event()
    log_context.register_cancel(outer)
    try:
        with OutputCapture():
            log_context.raise_if_cancelled()                # nothing set yet
            outer.set()                                     # the worker's own cancel, mid-turn
            with pytest.raises(AgentStoppedError):
                log_context.raise_if_cancelled()
            assert log_context.current_cancel().wait(1.0)   # a wait on the cancel ends too
        assert log_context.current_cancel() is outer
    finally:
        log_context.unregister_cancel()
