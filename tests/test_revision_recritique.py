"""Offline tests: caveat bookkeeping after non-human-path plan revisions.

The advisory critic's findings must describe the CURRENT plan. The refinement
LLM echoes the prior plan's ``critic_findings`` into revised JSON, so without
a resolution-check re-critique, an orchestrator-driven revision (e.g.
``adjust_plan_for_constraints`` fixing a caveat) carries the stale — now
false — caveat into plan_history, reports, and run_task warnings. Observed
live in meta session 2026-07-20 09:24 (ICP-MS TDS dilution caveat retained
verbatim after the adjustment that resolved it).

Contract pinned here:
- both revision paths (results-driven ``refine_plan``, constraint-driven
  ``adjust_plan_for_constraints``) pop echoed caveats and re-critique with
  the before/request/after picture (prior_plan, prior_findings, request);
- resolved caveats disappear from the plan AND the latest history snapshot;
- fresh findings are recorded critical-first;
- the re-critique is annotation-only: no extra refinement calls happen in
  response to findings (advisory-only preserved).
"""

import builtins
import json
from types import SimpleNamespace

import pytest

from scilink.agents.planning_agents.planning_agent import PlanningAgent
from scilink.agents.planning_agents.base_agent import BaseAgent
from scilink.agents.planning_agents import planning_agent as pa_mod


TDS_FINDING = {"dimension": "physics", "severity": "minor",
               "experiment": "Experiment 1",
               "issue": "Supernatants require ~100-1000x dilution before "
                        "ICP-MS — a step the plan never specifies."}


def make_plan(name="Base Plan", findings=None):
    plan = {
        "proposed_experiments": [{
            "hypothesis": "H", "experiment_name": name,
            "experimental_steps": ["step"], "required_equipment": ["eq"],
            "expected_outcome": "out", "justification": "j",
            "source_documents": [],
        }],
        "iteration": 1, "stage": "Science Draft",
    }
    if findings is not None:
        plan["critic_findings"] = list(findings)
    return plan


def make_agent(tmp_path):
    a = PlanningAgent.__new__(PlanningAgent)
    BaseAgent.__init__(a, str(tmp_path))
    a.agent_type = "planning"
    a.model = SimpleNamespace(generate_content=lambda *args, **kw: (_ for _ in ()).throw(
        AssertionError("unexpected raw LLM call")))
    a.generation_config = None
    a.kb_docs = SimpleNamespace(index=None)
    a.kb_code = SimpleNamespace(index=None)
    a.lit_agent = None
    a.state = {
        "session_id": "t", "agent_type": "planning", "action_history": [],
        "objective": "obj", "iteration_index": 1,
        "current_plan": make_plan(findings=[TDS_FINDING]),
        "plan_history": [make_plan(findings=[TDS_FINDING])],
        "experimental_results": [], "human_feedback_history": [],
        "status": "planned",
    }
    return a


@pytest.fixture
def spies(monkeypatch):
    """Patch refine + critic with spies. The refiner ECHOES critic_findings
    (the real failure mode); the critic's verdict is settable per-test."""
    calls = {"refine": [], "critic": [], "critic_verdict": {"findings": []}}

    def fake_refine(original_result, feedback, **kw):
        calls["refine"].append(feedback)
        new = json.loads(json.dumps(original_result))  # deep copy, echoes caveats
        new["proposed_experiments"][0]["experiment_name"] = "Revised Plan"
        return new

    def fake_critic(objective, result, model, generation_config, **kw):
        calls["critic"].append(kw)
        return json.loads(json.dumps(calls["critic_verdict"]))

    monkeypatch.setattr(pa_mod, "refine_plan_with_feedback", fake_refine)
    monkeypatch.setattr(pa_mod, "critique_plan", fake_critic)
    return calls


def test_adjust_drops_resolved_caveat(tmp_path, spies):
    agent = make_agent(tmp_path)
    out = agent.adjust_plan_for_constraints(
        "Add two-stage dilution to satisfy ICP-MS TDS tolerance.",
        enable_human_feedback=False)

    # stale echoed caveat removed from plan, current_plan, and latest snapshot
    assert "critic_findings" not in out
    assert "critic_findings" not in agent.state["current_plan"]
    assert "critic_findings" not in agent.state["plan_history"][-1]
    # the ORIGINAL pre-revision snapshot keeps its historical record
    assert agent.state["plan_history"][0]["critic_findings"] == [TDS_FINDING]
    # critic saw the full before/request/after picture
    ck = spies["critic"][-1]
    assert ck["prior_findings"] == [TDS_FINDING]
    assert ck["prior_plan"]["proposed_experiments"][0]["experiment_name"] == "Base Plan"
    assert "two-stage dilution" in ck["human_feedback"]
    # advisory-only: exactly one refine (the adjustment), no reaction to critic
    assert len(spies["refine"]) == 1


def test_adjust_records_fresh_findings_sorted(tmp_path, spies):
    spies["critic_verdict"] = {"findings": [
        {"dimension": "consistency", "severity": "minor", "experiment": "E", "issue": "m"},
        {"dimension": "physics", "severity": "critical", "experiment": "E", "issue": "c"},
    ]}
    agent = make_agent(tmp_path)
    out = agent.adjust_plan_for_constraints("constraint", enable_human_feedback=False)
    sev = [f["severity"] for f in out["critic_findings"]]
    assert sev == ["critical", "minor"]
    assert agent.state["plan_history"][-1]["critic_findings"] == out["critic_findings"]


def test_refine_plan_drops_resolved_caveat(tmp_path, spies):
    agent = make_agent(tmp_path)
    out = agent.refine_plan("Yield improved to 85%; dilution scheme worked.",
                            enable_human_feedback=False)
    assert "critic_findings" not in out
    assert "critic_findings" not in agent.state["plan_history"][-1]
    ck = spies["critic"][-1]
    assert ck["prior_findings"] == [TDS_FINDING]
    assert "experimental results" in ck["human_feedback"]
    assert len(spies["refine"]) == 1


def test_adjust_human_branch_recritiques_final_plan(tmp_path, spies, monkeypatch):
    agent = make_agent(tmp_path)
    replies = iter(["tighten the pH grid", ""])
    monkeypatch.setattr(builtins, "input", lambda *a: next(replies))
    out = agent.adjust_plan_for_constraints("constraint", enable_human_feedback=True)
    # two refines (adjustment + human), two re-critiques — every plan that
    # became current_plan was re-checked
    assert len(spies["refine"]) == 2
    assert len(spies["critic"]) == 2
    assert spies["critic"][-1]["human_feedback"] == "tighten the pH grid"
    assert "critic_findings" not in out


def test_critic_failure_fails_open(tmp_path, spies, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("critic down")
    monkeypatch.setattr(pa_mod, "critique_plan", boom)
    agent = make_agent(tmp_path)
    # helper is annotation-only; the real critique_plan fails open internally,
    # but even a raising spy must not lose the revision
    try:
        out = agent.adjust_plan_for_constraints("c", enable_human_feedback=False)
    except RuntimeError:
        pytest.fail("re-critique crash blocked the revision")
    assert out["proposed_experiments"][0]["experiment_name"] == "Revised Plan"
