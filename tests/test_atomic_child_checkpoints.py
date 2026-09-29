"""The mode orchestrators' checkpoints and histories survive a failed write,
and concurrent delegations never share a ledger index.

The three children wrote ``checkpoint.json`` and ``chat_history.json`` with
``open(path, "w")`` + ``json.dump``: the file is truncated first and filled as
serialization proceeds, so a crash mid-write, or a value that does not
serialize, left a truncated file that restore cannot read — the last good
checkpoint was destroyed along with the failed one. The meta's own checkpoint
was already atomic.
"""

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scilink.utils.text_io import atomic_write_json


def _circular():
    a = []
    a.append(a)
    return a


def test_atomic_write_json_leaves_the_previous_file_whole_on_failure(tmp_path):
    p = tmp_path / "state.json"
    atomic_write_json(p, {"good": 1})
    with pytest.raises(ValueError):
        atomic_write_json(p, {"bad": _circular()})
    assert json.loads(p.read_text()) == {"good": 1}
    assert [f.name for f in tmp_path.iterdir()] == ["state.json"]    # no temp left


def _analysis(tmp_path):
    from scilink.agents.exp_agents.analysis_orchestrator import AnalysisMode, AnalysisOrchestratorAgent
    o = AnalysisOrchestratorAgent.__new__(AnalysisOrchestratorAgent)
    o.checkpoint_path = tmp_path / "checkpoint.json"
    o.history_path = tmp_path / "chat_history.json"
    o.current_metadata, o.current_metadata_owner = {}, None
    o.current_data_path = o.current_data_type = o.selected_agent_id = None
    o.analysis_results, o._analysis_run_counter, o.message_count = [], 0, 0
    o.analysis_mode = AnalysisMode.AUTONOMOUS
    o.active_knowledge, o._graduated_skill_sources = [], []
    o.messages = [{"role": "user", "content": "hello"}]
    return o, "analysis_results"


def _planning(tmp_path):
    from scilink.agents.planning_agents.planning_orchestrator import (
        AutonomyLevel, PlanningOrchestratorAgent)
    o = PlanningOrchestratorAgent.__new__(PlanningOrchestratorAgent)
    o.checkpoint_path = tmp_path / "checkpoint.json"
    o.history_path = tmp_path / "chat_history.json"
    o.objective = "obj"
    o.active_scalarizer_script = None
    o.expected_input_columns = o.expected_target_columns = None
    o.target_directions = o.expected_input_types = o.expected_input_levels = None
    o.input_bounds_override = o.fidelity_spec = None
    o.bo_data_path = tmp_path / "bo.csv"
    o.planner = SimpleNamespace(state={})
    o.message_count, o.latest_tea_results, o._delegation_counter = 0, None, 0
    o.autonomy_level = AutonomyLevel.AUTONOMOUS
    o.data_dir = o.knowledge_dir = o.code_dir = None
    o.active_knowledge, o._graduated_skill_sources, o._custom_skills = [], [], {}
    o.messages = [{"role": "user", "content": "hello"}]
    return o, "latest_tea_results"


def _simulation(tmp_path):
    pytest.importorskip("ase")
    from scilink.agents.sim_agents.simulation_orchestrator import SimulationOrchestratorAgent
    o = SimulationOrchestratorAgent.__new__(SimulationOrchestratorAgent)
    o.checkpoint_path = tmp_path / "checkpoint.json"
    o.history_path = tmp_path / "chat_history.json"
    o.generated_structures, o.default_calc_params = [], {}
    o.message_count = o.last_checkpoint_message_count = 0
    o.CHECKPOINT_INTERVAL = 1
    o.logger = SimpleNamespace(warning=lambda *a, **k: None, info=lambda *a, **k: None)
    o.messages = [{"role": "user", "content": "hello"}]
    return o, "generated_structures"


def _checkpoint(o):
    if "force" in o._auto_checkpoint.__code__.co_varnames:
        o._auto_checkpoint(force=True, quiet=True)
    else:
        o._auto_checkpoint(quiet=True)


@pytest.mark.parametrize("make", [_analysis, _planning, _simulation],
                         ids=["analysis", "planning", "simulation"])
def test_a_failed_checkpoint_keeps_the_last_good_one(make, tmp_path):
    o, field = make(tmp_path)
    _checkpoint(o)
    good = json.loads(o.checkpoint_path.read_text())
    setattr(o, field, _circular())       # does not serialize, even with default=str
    _checkpoint(o)                       # logged, not raised
    assert json.loads(o.checkpoint_path.read_text()) == good
    assert sorted(f.name for f in tmp_path.iterdir()) == ["checkpoint.json"]


@pytest.mark.parametrize("make", [_analysis, _planning, _simulation],
                         ids=["analysis", "planning", "simulation"])
def test_a_failed_history_save_keeps_the_last_good_one(make, tmp_path):
    o, _ = make(tmp_path)
    o._save_history()
    good = json.loads(o.history_path.read_text())
    loop = {}
    loop["self"] = loop                  # survives the image sanitizer, never serializes
    o.messages = o.messages + [{"role": "assistant", "content": "x", "tool_calls": loop}]
    o._save_history()
    assert json.loads(o.history_path.read_text()) == good


def test_concurrent_delegations_get_distinct_ledger_indices():
    """Index allocation reads the ledger length and appends later; with a
    slow step in between (provenance inference) two unlocked callers read the
    same length."""
    from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent
    m = MetaOrchestratorAgent.__new__(MetaOrchestratorAgent)
    m._delegation_ledger = []
    m._fanout_lock = threading.RLock()

    def slow_infer(index, task, context):
        time.sleep(0.02)
        return set()

    m._infer_context_sources = slow_infer
    threads = [threading.Thread(target=m._open_delegation,
                                args=("analysis", f"task {i}", {}, None)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(e["index"] for e in m._delegation_ledger) == list(range(1, 13))


def test_the_ledger_lock_is_reentrant_for_fanout_preallocation():
    """Fan-out opens its branch entries while already holding the lock."""
    from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent
    m = MetaOrchestratorAgent.__new__(MetaOrchestratorAgent)
    m._delegation_ledger = []
    m._fanout_lock = threading.RLock()
    m._infer_context_sources = lambda index, task, context: set()
    done = threading.Event()

    def preallocate():
        with m._fanout_lock:
            m._open_delegation("analysis", "a", {}, None)
            m._open_delegation("analysis", "b", {}, None)
        done.set()

    t = threading.Thread(target=preallocate, daemon=True)
    t.start()
    assert done.wait(5), "deadlocked"
    assert [e["index"] for e in m._delegation_ledger] == [1, 2]


# ── review follow-ups ─────────────────────────────────────────────────────

def test_a_failed_replace_keeps_the_previous_file_and_leaves_no_temp(tmp_path, monkeypatch):
    """The earlier failure test fails inside json.dumps, before any temp file
    exists; this one fails at the publish step itself."""
    from scilink.utils import text_io
    p = tmp_path / "state.json"
    atomic_write_json(p, {"good": 1})

    def boom(src, dst):
        raise OSError("disk went away")

    monkeypatch.setattr(text_io, "_replace", boom)
    with pytest.raises(OSError):
        atomic_write_json(p, {"new": 2})
    assert json.loads(p.read_text()) == {"good": 1}
    assert [f.name for f in tmp_path.iterdir()] == ["state.json"]


def test_windows_retries_a_replace_blocked_by_an_open_reader(tmp_path, monkeypatch):
    from scilink.utils import text_io
    p = tmp_path / "chat_history.json"
    real = text_io.os.replace
    blocked = [2]

    def held_open(src, dst):
        if blocked[0]:
            blocked[0] -= 1
            raise PermissionError("in use by another process")
        return real(src, dst)

    monkeypatch.setattr(text_io, "_WINDOWS", True)
    monkeypatch.setattr(text_io.os, "replace", held_open)
    atomic_write_json(p, {"ok": True})
    assert json.loads(p.read_text()) == {"ok": True} and blocked[0] == 0


@pytest.mark.parametrize("make", [_analysis, _planning, _simulation],
                         ids=["analysis", "planning", "simulation"])
def test_a_failed_history_save_leaves_no_temp_file(make, tmp_path):
    o, _ = make(tmp_path)
    o._save_history()
    loop = {}
    loop["self"] = loop
    o.messages = o.messages + [{"role": "assistant", "content": "x", "tool_calls": loop}]
    o._save_history()
    assert sorted(f.name for f in tmp_path.iterdir()) == ["chat_history.json"]


def test_the_analysis_save_checkpoint_tool_uses_the_one_writer(tmp_path):
    from scilink.agents.exp_agents.analysis_orchestrator_tools import AnalysisOrchestratorTools
    o, _ = _analysis(tmp_path)
    o.analysis_results = [{"analysis_id": "a1"}]
    tools = AnalysisOrchestratorTools(o)
    out = json.loads(tools.functions_map["save_checkpoint"]())
    assert out["status"] == "success" and out["analyses_saved"] == 1
    good = json.loads(o.checkpoint_path.read_text())
    assert good["analysis_results"] == [{"analysis_id": "a1"}]
    o.analysis_results = _circular()
    out = json.loads(tools.functions_map["save_checkpoint"]())
    assert out["status"] == "error" and "previous one is kept" in out["message"]
    assert json.loads(o.checkpoint_path.read_text()) == good


def test_the_planning_save_checkpoint_tool_keeps_the_delegation_counter(tmp_path):
    """The tool wrote a second, smaller schema without delegation_counter, so
    a restore restarted the counter and the next delegation reused
    delegations/01_<slug>/."""
    from scilink.agents.planning_agents.orchestrator_tools import OrchestratorTools
    o, _ = _planning(tmp_path)
    o.base_dir = tmp_path
    o.use_openai = True
    o.active_knowledge = []
    o._delegation_counter = 3
    o.target_directions = {"y": "maximize"}
    tools = OrchestratorTools(o)
    out = json.loads(tools.functions_map["save_checkpoint"]())
    assert out["status"] == "success"
    saved = json.loads(o.checkpoint_path.read_text())
    assert saved["delegation_counter"] == 3
    assert saved["target_directions"] == {"y": "maximize"}
    assert "custom_skills" in saved and "graduated_skill_sources" in saved


def test_a_meta_is_built_with_a_reentrant_ledger_lock(tmp_path, monkeypatch):
    import threading as _threading
    from scilink.agents.meta_agent.meta_orchestrator import MetaMode, MetaOrchestratorAgent
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    m = MetaOrchestratorAgent(base_dir=str(tmp_path / "meta"), model_name="anthropic/claude-sonnet-4-5",
                              meta_mode=MetaMode.AUTONOMOUS, launch_dir=str(tmp_path))
    assert isinstance(m._fanout_lock, type(_threading.RLock()))
