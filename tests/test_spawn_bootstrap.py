"""SciLink's worker processes never re-run the caller's script (#721).

A ``multiprocessing`` spawn worker re-imports the main script before doing
anything else, so a driver script without an ``if __name__ == "__main__":``
guard ran a copy of itself in every hyperspectral replay worker: a new agent,
model calls, and writes into the same session folder. Three things are held
here, each through real processes:

- the replay pool launches fresh interpreters that import SciLink and nothing
  of the caller's, so an UNGUARDED driver runs once and its replays still run
  on the pool, with the same records as a serial run;
- a replay whose worker comes back with no result (killed) is re-run in the
  parent, the serial path's way — a pool failure is not a recipe failure, so
  it is never handed to a refit;
- every SciLink entry point refuses to start inside a spawn bootstrap (a pool
  SciLink does not own), and a pool's task, once bootstrapped, may run it.

  conda run -n scilink python -m pytest tests/test_spawn_bootstrap.py -q
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from scilink.utils.child_process import ChildLost, run_in_child

REPO = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent

# The single-cube pipeline is stubbed at the same seam as tests/test_hs_series.py.
# The stub reaches the replay workers through ``sitecustomize`` on the path the
# driver hands them — the workers import nothing of the driver itself.
STUB = '''
import json, os, signal
from pathlib import Path
import numpy as np

SCRIPT = "def analyze_feature(data, axis):\\n    return {'maps': {'Mean_Map': data.mean(2)}}\\n"


def _log(**rec):
    with open(os.environ["HS_STUB_LOG"], "a") as f:
        f.write(json.dumps({"pid": os.getpid(), **rec}) + "\\n")


def install():
    from scilink.agents.exp_agents.hyperspectral_analysis_agent import HyperspectralAnalysisAgent as A
    from scilink.wrappers.litellm_wrapper import LiteLLMGenerativeModel as M

    def fake_pipeline(self, data_path, system_info, instruction_prompt, reuse_records=None, **kw):
        name = Path(data_path).stem
        _log(role=self._series_role, name=name, reuse=bool(reuse_records))
        if (name == os.environ.get("HS_STUB_KILL")
                and os.getpid() != int(os.environ["HS_STUB_PARENT"])):
            os.kill(os.getpid(), signal.SIGKILL)
        value = float(np.load(data_path).mean())
        rec = {"target": "mean map", "task_success": True, "required_outputs": ["Mean_Map"],
               "script": SCRIPT, "quality_history": {"approved": True},
               "locked_replay": bool(reuse_records), "replay_verbatim": True}
        (self.output_dir / "dynamic_analysis_records.json").write_text(json.dumps([rec]))
        return {"detailed_analysis": f"analysis of {name}", "scientific_claims": [],
                "extracted_features": [{"name": "Mean_Map", "units": "a.u.",
                                        "stats": {"min": value - 1, "max": value + 1, "mean": value}}],
                "dynamic_analysis_records": [rec]}, None

    def no_model(self, *a, **k):
        _log(model_call=True)
        raise RuntimeError("no model calls in this test")

    A._run_analysis_pipeline = fake_pipeline
    A._maybe_bank_scripts = lambda self, *a, **k: []
    A._maybe_stage_t2_solutions = lambda self, *a, **k: []
    A._auto_select_skills = lambda self, *a, **k: []
    M.generate_content = no_model
'''

SITECUSTOMIZE = '''
import os
if os.environ.get("HS_STUB_LOG"):
    import _hs_spawn_stub
    _hs_spawn_stub.install()
'''

# NO __main__ guard, on purpose: this is the script #721 is about.
DRIVER = '''
import json, os, sys
from pathlib import Path
sys.path.insert(0, {stub!r})
sys.path.insert(0, {tests!r})
with open({mark!r}, "a") as f:
    f.write(f"{{os.getpid()}}\\n")
os.environ["HS_STUB_PARENT"] = str(os.getpid())
import numpy as np
import _hs_spawn_stub
_hs_spawn_stub.install()
from test_hs_series import _Calls, _FakeModel, AXIS, MEANS
from scilink.agents.exp_agents.hyperspectral_analysis_agent import HyperspectralAnalysisAgent

data = Path({data!r}); data.mkdir(exist_ok=True)
paths = []
for i, m in enumerate(MEANS):
    p = data / f"cube_T{{300 + 50 * i}}.npy"
    np.save(p, np.full((4, 4, 8), m, dtype=np.float32))
    paths.append(str(p))
agent = HyperspectralAnalysisAgent(api_key="sk-dummy", output_dir={out!r},
                                   enable_human_feedback=False, executor_timeout=120)
agent.model = _FakeModel(_Calls())
res = agent.analyze(paths, system_info=dict(AXIS), series_workers={workers},
                    series_metadata={{"variable": "temperature",
                                      "values": [300, 350, 400, 450, 500, 550], "unit": "K"}})
Path({summary!r}).write_text(json.dumps(res["summary"]))
'''


def _run_driver(tmp_path, name, workers, kill=None):
    root = tmp_path / name
    stub = root / "stub"
    stub.mkdir(parents=True)
    (stub / "_hs_spawn_stub.py").write_text(STUB)
    (stub / "sitecustomize.py").write_text(SITECUSTOMIZE)
    files = {k: str(root / v) for k, v in dict(mark="runs.txt", data="cubes", out="out",
                                                  summary="summary.json", log="calls.jsonl").items()}
    driver = root / "driver.py"
    driver.write_text(DRIVER.format(stub=str(stub), tests=str(TESTS), workers=workers, **files))
    env = {k: v for k, v in os.environ.items() if k not in ("HS_STUB_KILL", "PYTHONPATH")}
    env.update(HS_STUB_LOG=files["log"], ANTHROPIC_API_KEY="sk-dummy", UNSAFE_EXECUTION_OK="true",
               PYTHONPATH=str(REPO))
    env.pop("SCILINK_HS_SERIES_POOL", None)
    if kill:
        env["HS_STUB_KILL"] = kill
    proc = subprocess.run([sys.executable, str(driver)], cwd=root, env=env,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-4000:]
    calls = [json.loads(line) for line in Path(files["log"]).read_text().splitlines()]
    rows = json.loads((Path(files["out"]) / "series_analysis_results.json").read_text())["results"]
    record = {r["index"]: (r["success"], r["status"], r["role"], r.get("verified"),
                           r["extracted_features"]) for r in rows}
    return dict(runs=Path(files["mark"]).read_text().split(), calls=calls, record=record,
                summary=json.loads(Path(files["summary"]).read_text()), stderr=proc.stderr,
                out=Path(files["out"]))


@pytest.fixture(scope="module")
def serial(tmp_path_factory):
    return _run_driver(tmp_path_factory.mktemp("spawn"), "serial", workers=1)


def test_unguarded_driver_runs_once_and_its_replays_run_on_the_pool(tmp_path, serial):
    run = _run_driver(tmp_path, "pool", workers=2)
    parent = run["runs"]
    assert len(parent) == 1, f"the driver script ran {len(parent)} times"
    replays = [c for c in run["calls"] if c.get("role") == "replay"]
    assert len(replays) == 5 and all(str(c["pid"]) not in parent for c in replays)
    assert not [c for c in run["calls"] if c.get("model_call") and str(c["pid"]) not in parent]
    assert not [c for c in run["calls"] if c.get("role") == "refit"]
    assert "replays_rerun_in_process" not in run["summary"]
    assert run["record"] == serial["record"]


def test_a_killed_replay_worker_is_rerun_in_process_not_refit(tmp_path, serial):
    run = _run_driver(tmp_path, "killed", workers=2, kill="cube_T400")
    parent = run["runs"]
    assert len(parent) == 1
    t400 = [c for c in run["calls"] if c.get("name") == "cube_T400"]
    # once in a worker (killed), once more in the parent, as a replay both times
    assert [c["role"] for c in t400] == ["replay", "replay"]
    assert str(t400[0]["pid"]) not in parent and str(t400[1]["pid"]) in parent
    assert not [c for c in run["calls"] if c.get("role") == "refit"]
    assert "SIGKILL" in run["summary"]["replays_rerun_in_process"]["2"]
    assert run["record"] == serial["record"]
    # the killed attempt's leftovers are set aside; the folder holds the re-run
    unit = run["out"] / "dataset_0002"
    assert (unit / "lost_attempt" / "replay.log").is_file()
    assert (unit / "dynamic_analysis_records.json").is_file() and not (unit / "replay.log").exists()


SPAWN_SCRIPT = '''
import json, multiprocessing, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

OUT = Path({out!r})


def build_all(phase):
    from scilink.utils.child_process import SpawnBootstrapError
    from scilink.agents.exp_agents.analysis_orchestrator import AnalysisOrchestratorAgent
    from scilink.agents.planning_agents.planning_orchestrator import PlanningOrchestratorAgent
    from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent
    from scilink.agents.exp_agents import CurveFittingAgent
    from scilink.agents.exp_agents.preprocess import CurvePreprocessingAgent
    from scilink.wrappers.litellm_wrapper import LiteLLMGenerativeModel, LiteLLMEmbeddingModel
    from scilink.wrappers import openai_wrapper, openai_wrapper_tools, openai_wrapper_embeddings
    tag = multiprocessing.current_process().name + "_" + phase
    base = OUT / tag
    builders = {{
        "AnalysisOrchestratorAgent": lambda: AnalysisOrchestratorAgent(api_key="sk-dummy", base_dir=str(base / "an")),
        "PlanningOrchestratorAgent": lambda: PlanningOrchestratorAgent(api_key="sk-dummy", data_dir=str(OUT), base_dir=str(base / "pl")),
        "MetaOrchestratorAgent": lambda: MetaOrchestratorAgent(api_key="sk-dummy", base_dir=str(base / "meta")),
        "CurveFittingAgent": lambda: CurveFittingAgent(api_key="sk-dummy", output_dir=str(base / "cf")),
        "CurvePreprocessingAgent": lambda: CurvePreprocessingAgent(api_key="sk-dummy", output_dir=str(base / "pre")),
        "LiteLLMGenerativeModel": lambda: LiteLLMGenerativeModel(model="claude-opus-4-6", api_key="sk-dummy"),
        "LiteLLMEmbeddingModel": lambda: LiteLLMEmbeddingModel(model="gemini-embedding-001", api_key="sk-dummy"),
        "OpenAIAsGenerativeModel": lambda: openai_wrapper.OpenAIAsGenerativeModel(model="m", api_key="k", base_url="http://localhost:9"),
        "OpenAIAsGenerativeModel(tools)": lambda: openai_wrapper_tools.OpenAIAsGenerativeModel(model="m", api_key="k", base_url="http://localhost:9"),
        "OpenAIAsEmbeddingModel": lambda: openai_wrapper_embeddings.OpenAIAsEmbeddingModel(model="m", api_key="k", base_url="http://localhost:9"),
    }}
    try:
        from scilink.agents.sim_agents.simulation_orchestrator import SimulationOrchestratorAgent
        builders["SimulationOrchestratorAgent"] = lambda: SimulationOrchestratorAgent(api_key="sk-dummy", base_dir=str(base / "si"))
    except ImportError:
        pass
    seen = {{}}
    for name, build in builders.items():
        try:
            build()
            seen[name] = "built"
        except SpawnBootstrapError as e:
            seen[name] = "refused: " + str(e)
        except Exception as e:  # noqa: BLE001
            seen[name] = f"other: {{type(e).__name__}}: {{e}}"
    files = sorted(str(p.relative_to(base)) for p in base.rglob("*")) if base.exists() else []
    (OUT / f"{{tag}}.json").write_text(json.dumps({{"seen": seen, "files": files}}))
    return tag


def task(_):
    return build_all("task")


# NO __main__ guard: the top level runs again in every spawned worker's bootstrap.
build_all("top")
if multiprocessing.current_process().name == "MainProcess":
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as ex:
        try:
            print("TASK", ex.submit(task, 0).result())
        except Exception as e:  # noqa: BLE001
            print("POOL", type(e).__name__)
'''


def test_every_entry_point_refuses_in_a_spawn_bootstrap_and_runs_in_a_task(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    script = tmp_path / "unguarded.py"
    script.write_text(SPAWN_SCRIPT.format(out=str(out)))
    env = {**os.environ, "ANTHROPIC_API_KEY": "sk-dummy", "UNSAFE_EXECUTION_OK": "true",
           "PYTHONPATH": str(REPO)}
    proc = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=600)
    reports = {p.stem: json.loads(p.read_text()) for p in out.glob("*.json")}
    assert set(reports) == {"MainProcess_top", "SpawnProcess-1_top", "SpawnProcess-1_task"}, (
        sorted(reports), proc.stderr[-3000:])
    # the driver itself: nothing refused
    main = reports["MainProcess_top"]
    assert main["seen"] and all(v == "built" for v in main["seen"].values()), main
    # the worker re-importing the script: every entry point refused, with the
    # fix in the message, before writing a file
    boot = reports["SpawnProcess-1_top"]
    assert boot["seen"].keys() == main["seen"].keys()
    assert all(v.startswith("refused: ") and "if __name__" in v for v in boot["seen"].values()), boot
    assert boot["files"] == []
    # the same worker, once bootstrapped, runs the pool's task: SciLink starts
    task = reports["SpawnProcess-1_task"]
    assert all(v == "built" for v in task["seen"].values()), task
    assert "TASK SpawnProcess-1_task" in proc.stdout, (proc.stdout, proc.stderr[-3000:])


def test_run_in_child_returns_and_reports_what_went_wrong():
    assert run_in_child("os.path:basename", "/a/b/c.txt") == "c.txt"
    with pytest.raises(ChildLost) as raised:
        run_in_child("json:loads", "{not json")
    # one line names the exception; the traceback stays in the detail
    assert raised.value.reason.startswith("raised in the child: json.decoder.JSONDecodeError")
    assert "\n" not in raised.value.reason and "Traceback" in raised.value.detail
    with pytest.raises(ChildLost, match="SIGKILL"):
        run_in_child("signal:raise_signal", 9)      # the child kills itself


def test_the_target_runs_with_the_callers_pythonpath(monkeypatch):
    """The parent's sys.path reaches the child's START only; what the target
    runs (a replay's generated script, through the sandbox's env allow-list)
    sees the caller's own PYTHONPATH, as a serial run does."""
    monkeypatch.setenv("PYTHONPATH", "/nowhere/a")
    assert run_in_child("os:getenv", "PYTHONPATH") == "/nowhere/a"
    monkeypatch.delenv("PYTHONPATH")
    assert run_in_child("os:getenv", "PYTHONPATH") is None


SHADOWED_DRIVER = """
import sys
sys.path.insert(1, {repo!r})
from scilink.utils.child_process import run_in_child
import scilink
print("PARENT", scilink.__file__)
print("CHILD", run_in_child("os.path:basename", "/a/b/c.txt"))
"""


def test_a_package_in_the_working_directory_does_not_shadow_the_parents(tmp_path):
    """``python -m`` puts the working directory first; spawn never did. A
    driver run from a folder holding another ``scilink/`` (a second checkout)
    must still have its workers import the parent's SciLink."""
    work = tmp_path / "work"
    (work / "scilink").mkdir(parents=True)
    (work / "scilink" / "__init__.py").write_text("raise ImportError('the shadow scilink was imported')\n")
    drv = tmp_path / "drv"
    drv.mkdir()
    (drv / "driver.py").write_text(SHADOWED_DRIVER.format(repo=str(REPO)))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run([sys.executable, str(drv / "driver.py")], cwd=work, env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert f"PARENT {REPO / 'scilink' / '__init__.py'}" in proc.stdout
    assert "CHILD c.txt" in proc.stdout, (proc.stdout, proc.stderr[-3000:])
