"""The planning ``run_task`` reports how the plan was settled: a human-approved
plan, an unattended gate and any standing blocking finding travel as
``plan_review`` to the meta and over MCP, with whether the plan was written
in this call."""

import json


def test_planning_run_task_reports_how_the_plan_was_settled(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-dummy")
    from scilink.agents.planning_agents.planning_orchestrator import PlanningOrchestratorAgent
    orch = PlanningOrchestratorAgent(api_key="sk-dummy", data_dir=str(tmp_path),
                                     base_dir=str(tmp_path / "pl"))

    pending = {}

    def fake_chat(_prompt):
        orch._last_chat_hit_iter_cap = False
        orch._last_chat_error = None
        if pending:                      # the turn writes the plan
            orch.planner.state["current_plan"] = pending.pop("plan")
        return "Plan ready."
    orch.chat = fake_chat
    pending["plan"] = {
        "iteration": 1, "human_review": {"status": "accepted", "iteration": 1},
        "proposed_experiments": [{"experiment_name": "Purity check", "hypothesis": "A7 is single-phase anatase"},
                                 {"experiment_name": "Anneal"}],
        "critic_findings": [{"severity": "blocking", "issue": "650 C exceeds the furnace limit",
                             "conflict": "650 C vs 600 C"}]}
    r = orch.run_task("plan it")
    assert r["plan_review"]["human_review"] == {"status": "accepted", "iteration": 1}
    assert r["plan_review"]["hypotheses"] == ["Experiment 'Purity check': A7 is single-phase anatase",
                                              "Experiment 'Anneal': no hypothesis stated"]
    assert r["plan_review"]["written_here"] is True          # the plan appeared during this call
    r = orch.run_task("a TEA on the same plan")               # the child's plan is unchanged
    assert r["plan_review"]["written_here"] is False and r["plan_review"]["human_review"]
    # an edit that is not a rewrite (a literature search restored, a code-gen copy) is not a new plan
    orch.planner.state["current_plan"]["literature_search"] = {"restored": True}
    orch.planner.state["current_plan"]["implementation_code"] = "print(1)"
    r = orch.run_task("write the white paper")
    assert r["plan_review"]["written_here"] is False
    revised = json.loads(json.dumps(orch.planner.state["current_plan"]))
    revised["proposed_experiments"][0]["hypothesis"] = "A7 is 5% rutile"
    pending["plan"] = revised                                  # the turn rewrites a hypothesis
    r = orch.run_task("revise")
    assert r["plan_review"]["written_here"] is True
    assert r["plan_review"]["unattended_gate"] is None
    assert r["plan_review"]["blocking_findings"] == [{"issue": "650 C exceeds the furnace limit",
                                                      "conflict": "650 C vs 600 C"}]
    pending["plan"] = {"iteration": 2, "unattended_gate": {"would_have_been": "accepted"},
                       "directions": [{"title": "Doping series", "hypothesis": "Nb widens the gap"}]}
    r = orch.run_task("plan again")
    assert r["plan_review"]["written_here"] is True
    assert r["plan_review"]["human_review"] is None and r["plan_review"]["unattended_gate"]["would_have_been"] == "accepted"
    assert r["plan_review"]["blocking_findings"] == [] and r["plan_review"]["hypotheses"] == ["Direction: Nb widens the gap"]
    # an autonomous protocol revision keeps the hypotheses and iteration: its steps and blocking
    # issues change, and that is a plan written here too
    revised2 = {"iteration": 3, "proposed_experiments": [{"experiment_name": "Anneal", "hypothesis": "h",
                                                          "experimental_steps": ["anneal at 650 C"]}]}
    pending["plan"] = revised2
    orch.run_task("plan the anneal")
    revised2 = json.loads(json.dumps(revised2))
    revised2["proposed_experiments"][0]["experimental_steps"] = ["anneal at 550 C", "measure"]
    revised2["critic_findings"] = [{"severity": "blocking", "issue": "550 C still above", "conflict": "550 vs 500"}]
    pending["plan"] = revised2
    r = orch.run_task("fix the blocking defect")
    assert r["plan_review"]["written_here"] is True
