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


def test_a_real_meta_has_a_reentrant_ledger_lock():
    import inspect
    from scilink.agents.meta_agent import meta_orchestrator as mo
    src = inspect.getsource(mo.MetaOrchestratorAgent.__init__)
    assert "self._fanout_lock = threading.RLock()" in src
