"""A planning agent keeps its workspace paths when its autonomy level changes.

``set_autonomy_level`` rebuilt the system prompt from ``get_system_prompt``
alone and dropped the workspace block (data, knowledge and code directories)
that the constructor and ``_rebuild_system_prompt`` append. ``run_task``
switches the level on every call, and the meta builds its planning child in
CO_PILOT, so a delegated planning run never saw its knowledge directory: live,
the agent answered that no knowledge base was accessible although its planner
had loaded one.
"""

from pathlib import Path

import pytest

from scilink.agents.planning_agents.planning_orchestrator import (
    AutonomyLevel, PlanningOrchestratorAgent, get_system_prompt)


def _agent(tmp_path, level=AutonomyLevel.CO_PILOT):
    a = PlanningOrchestratorAgent.__new__(PlanningOrchestratorAgent)
    a.autonomy_level = level
    a._external_tools = None
    a.objective = "check phase purity of batch A7"
    a.data_dir = tmp_path / "data"
    a.knowledge_dir = tmp_path / "knowledge"
    a.code_dir = tmp_path / "code"
    a.messages = [{"role": "system", "content": "stale"}]
    return a


@pytest.mark.parametrize("target", [AutonomyLevel.AUTOPILOT, AutonomyLevel.AUTONOMOUS])
def test_switching_up_from_copilot_names_the_workspace(tmp_path, target):
    a = _agent(tmp_path)
    a.set_autonomy_level(target)
    prompt = a.messages[0]["content"]
    for d in (a.data_dir, a.knowledge_dir, a.code_dir):
        assert str(d) in prompt
    assert prompt == a._system_prompt


def test_a_switch_between_non_copilot_levels_keeps_the_workspace(tmp_path):
    a = _agent(tmp_path, AutonomyLevel.AUTOPILOT)
    a._rebuild_system_prompt()                      # what the constructor produces
    assert str(a.knowledge_dir) in a.messages[0]["content"]
    a.set_autonomy_level(AutonomyLevel.AUTONOMOUS)  # what run_task does
    assert str(a.knowledge_dir) in a.messages[0]["content"]


def test_copilot_still_leaves_the_workspace_to_the_human(tmp_path):
    a = _agent(tmp_path, AutonomyLevel.AUTONOMOUS)
    a.set_autonomy_level(AutonomyLevel.CO_PILOT)
    assert str(a.knowledge_dir) not in a.messages[0]["content"]
    assert a.messages[0]["content"] == get_system_prompt(AutonomyLevel.CO_PILOT, None)


def test_human_feedback_still_follows_the_level(tmp_path):
    a = _agent(tmp_path)
    a.set_autonomy_level(AutonomyLevel.AUTONOMOUS)
    assert a._enable_human_feedback is False
    a.set_autonomy_level(AutonomyLevel.AUTOPILOT)
    assert a._enable_human_feedback is True


@pytest.mark.parametrize("objective", ["Delegated by meta-agent", "Undefined Research Goal"])
def test_a_placeholder_objective_is_not_presented_as_the_objective(tmp_path, objective):
    a = _agent(tmp_path)
    a.objective = objective
    a.set_autonomy_level(AutonomyLevel.AUTONOMOUS)
    prompt = a.messages[0]["content"]
    assert "Research objective" not in prompt
    assert str(a.knowledge_dir) in prompt


def test_a_knowledge_dir_attached_mid_session_reaches_the_next_delegation(tmp_path):
    """The meta's attach_knowledge_dir sets child.knowledge_dir without a
    rebuild; run_task's level switch on entry now carries it into the
    prompt."""
    a = _agent(tmp_path)
    a.knowledge_dir = tmp_path / "attached_kb"
    a.set_autonomy_level(AutonomyLevel.AUTONOMOUS)            # run_task on entry
    assert str(tmp_path / "attached_kb") in a.messages[0]["content"]


def test_the_bo_objective_is_the_targets_not_the_placeholder(tmp_path):
    from types import SimpleNamespace
    from scilink.agents.planning_agents.orchestrator_tools import OrchestratorTools
    orch = SimpleNamespace(base_dir=tmp_path, planner=SimpleNamespace(),
                           objective="Delegated by meta-agent", target_directions={"yield": "maximize"})
    t = OrchestratorTools(orch)
    assert t._distill_objective_for_bo(["yield", "cost"]) == "Optimize yield (maximize), cost."
    orch.objective = "Maximize yield of NMC811 cathodes"
    assert t._distill_objective_for_bo(["yield"]) == "Maximize yield of NMC811 cathodes"


def test_scalarizer_prompts_leave_out_a_placeholder_objective():
    import inspect
    from scilink.agents.planning_agents import orchestrator_tools as ot
    src = inspect.getsource(ot)
    assert 'self.orch.objective != "Undefined Research Goal"' not in src
    assert src.count("not in _placeholder_objectives()") == 2
