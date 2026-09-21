"""The mode adapters' seed turns: what the old per-mode CLIs ran before the
first prompt, and what they did not run."""

from types import SimpleNamespace

from scilink.cli.shell.modes import PlanAdapter


def _plan_args(**kw):
    base = dict(data_dir="data", knowledge_dir=None, code_dir=None, restore=False)
    base.update(kw)
    return SimpleNamespace(**base)


def _agent(level):
    from scilink.agents.planning_agents.planning_orchestrator import AutonomyLevel
    return SimpleNamespace(autonomy_level=AutonomyLevel[level], objective="Optimize yield")


def test_plan_autopilot_surveys_the_workspace_on_a_fresh_session():
    turns = PlanAdapter().initial_turns(_plan_args(), _agent("AUTOPILOT"))
    assert len(turns) == 2 and "Survey the workspace" in turns[0] and "next steps" in turns[1]


def test_plan_seed_turns_are_skipped_on_restore():
    """A restored campaign continues where it left off (the old CLI printed
    'Session restored from checkpoint' instead of re-running the survey)."""
    assert PlanAdapter().initial_turns(_plan_args(restore=True), _agent("AUTOPILOT")) == []
    assert PlanAdapter().initial_turns(_plan_args(restore=True), _agent("AUTONOMOUS")) == []


def test_plan_co_pilot_starts_empty():
    assert PlanAdapter().initial_turns(_plan_args(), _agent("CO_PILOT")) == []
