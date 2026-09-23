"""Offline tests: a plan the human approved is settled.

Observed live (meta session 2026-09-19 19:29, autopilot): the human picked the
Cu-nanoparticle candidate and approved it; the planning orchestrator then
called ``refine_plan_with_results`` three times on its own reading of the
advisory critic ("DESIGN REVIEW ... These are design corrections, not
experimental outcomes"), switching the catalyst to Au — the candidate the
human had passed over — and spending the next two rounds repairing what the
previous round broke. Each round was logged as experimental results, advanced
the iteration to 4 with no experiment run, and re-opened a gate the human
waved through in 10-36 s. A final ``edit_file`` on plan.json then forked the
disk copy from the state and the HTML report.

Contract pinned here:
- a gate records the human's review on the plan (``human_review``) and the tool
  result says the plan is settled;
- a rewrite names its ``trigger``; a ``user_request`` with no user message since
  the approval, a ``blocking_defect`` with no reason, a second self-initiated
  rewrite, and any rewrite after the human declined one are refused;
- an agent-initiated revision of an approved plan goes through a gate whose
  ENTER keeps the approved plan and leaves the campaign state as it was;
- only ``new_results`` advances the iteration and is logged as results;
- the critic's ``blocking`` tier — and only that tier — gets one in-place
  repair before the plan is shown, accepted only if it stayed local and the
  re-critique confirms it; everything else keeps the plan as authored;
- file tools refuse the files that mirror planner state.

All LLM traffic is mocked; no network.
"""

import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scilink.agents.planning_agents.planning_agent import PlanningAgent
from scilink.agents.planning_agents.base_agent import BaseAgent
from scilink.agents.planning_agents.orchestrator_tools import OrchestratorTools
from scilink.agents.planning_agents import planning_agent as pa_mod
from scilink.agents.planning_agents import planning_rag as pr
from scilink.agents.planning_agents.user_interface import (
    format_caveats, format_auto_repair)


# ---------------------------------------------------------------- helpers

def experiment(name="Cu-nanoparticle CO2RR reconstruction",
               hypothesis="Cu nanoparticles reduce from Cu-oxide to Cu0 under "
                          "cathodic bias.",
               steps=None):
    return {
        "hypothesis": hypothesis, "experiment_name": name,
        "experimental_steps": steps or [
            "Step 1: deposit a sparse Cu nanoparticle array on glassy carbon",
            "Step 2: hold the aqueous bicarbonate cell at 150 C and 1 atm",
            "Step 3: record operando UV-Vis during a cathodic staircase",
            "Step 4: emerse under Ar and map the same location by KPFM"],
        "required_equipment": ["potentiostat", "UV-Vis", "KPFM"],
        "expected_outcome": "LSPR shift tracks the work-function shift",
        "justification": "Tests ensemble optics against single-particle SPM.",
        "source_documents": [],
    }


def make_plan(findings=None, reviewed=None, **exp_kw):
    plan = {"proposed_experiments": [experiment(**exp_kw)],
            "iteration": 1, "stage": "Science Draft"}
    if findings is not None:
        plan["critic_findings"] = list(findings)
    if reviewed:
        plan["human_review"] = {"status": reviewed, "iteration": 1}
    return plan


REVISED_STEP = "Step 3: record operando UV-Vis on ITO during the staircase"


def revised(plan):
    return REVISED_STEP in plan["proposed_experiments"][0]["experimental_steps"]


BLOCKING = {"dimension": "physics", "severity": "blocking",
            "experiment": "plan-wide",
            "issue": "An aqueous cell cannot be held at 150 C at 1 atm.",
            "conflict": "150 C at 1 atm vs a ~100 C boiling point"}
CRITICAL = {"dimension": "method", "severity": "critical",
            "experiment": "plan-wide",
            "issue": "Air KPFM cannot validate an in-liquid oxidation state."}


def make_agent(tmp_path, plan=None):
    a = PlanningAgent.__new__(PlanningAgent)
    BaseAgent.__init__(a, str(tmp_path))
    a.agent_type = "planning"
    a.model = SimpleNamespace(generate_content=lambda *args, **kw: (_ for _ in ()).throw(
        AssertionError("unexpected raw LLM call")))
    a.generation_config = None
    a.kb_docs = SimpleNamespace(index=None, chunks=[])
    a.kb_code = SimpleNamespace(index=None)
    a.lit_agent = None
    plan = plan or make_plan(reviewed="accepted")
    a.state = {
        "session_id": "t", "agent_type": "planning", "action_history": [],
        "objective": "obj", "iteration_index": 1,
        "current_plan": plan,
        "plan_history": [json.loads(json.dumps(plan))],
        "experimental_results": [], "human_feedback_history": [],
        "status": "planned",
    }
    return a


@pytest.fixture
def spies(monkeypatch):
    calls = {"refine": [], "critic": [], "critic_verdict": {"findings": []},
             "rename_to": None}

    def fake_refine(original_result, feedback, **kw):
        calls["refine"].append(feedback)
        new = json.loads(json.dumps(original_result))
        exp = new["proposed_experiments"][0]
        exp["experimental_steps"][2] = REVISED_STEP          # a local edit
        if calls["rename_to"]:
            exp["experiment_name"] = calls["rename_to"]      # a redesign
        return new

    def fake_critic(objective, result, model, generation_config, **kw):
        calls["critic"].append(kw)
        return json.loads(json.dumps(calls["critic_verdict"]))

    monkeypatch.setattr(pa_mod, "refine_plan_with_feedback", fake_refine)
    monkeypatch.setattr(pa_mod, "critique_plan", fake_critic)
    return calls


def answers(monkeypatch, *replies):
    it = iter(replies)
    monkeypatch.setattr(builtins, "input", lambda *a: next(it))


def make_tools(tmp_path, agent, message_count=1, feedback=True):
    orch = SimpleNamespace(planner=agent, base_dir=tmp_path,
                           objective="obj", knowledge_dir=None,
                           message_count=message_count,
                           _enable_human_feedback=feedback,
                           _active_output_subdir=None,
                           latest_tea_results=None)
    t = OrchestratorTools(orch)
    t._emit_plan_report = lambda *a, **k: None
    t._adopt_literature = lambda *a, **k: None
    t._load_campaign_literature = lambda *a, **k: None
    t._collect_scalarizer_context = lambda *a, **k: []
    return t


# ------------------------------------------------- 1. the review is recorded

def test_gate_records_the_review(tmp_path, spies, monkeypatch):
    agent = make_agent(tmp_path, make_plan())
    answers(monkeypatch, "")                       # ENTER: accept the revision
    out = agent.refine_plan("Yield was 12%.", enable_human_feedback=True)
    assert out["human_review"]["status"] == "accepted"
    assert agent.state["plan_history"][-1]["human_review"]["status"] == "accepted"

    answers(monkeypatch, "use ITO instead")
    out = agent.refine_plan("Yield was 30%.", enable_human_feedback=True)
    assert out["human_review"]["status"] == "revised"


def test_no_gate_no_review(tmp_path, spies):
    agent = make_agent(tmp_path, make_plan())
    out = agent.refine_plan("Yield was 12%.", enable_human_feedback=False)
    assert "human_review" not in out


def test_a_rewrite_never_inherits_the_old_review():
    """The refiner echoes the plan it was given; an echoed stamp would mark a
    revision nobody has seen as approved."""
    plan = make_plan(reviewed="accepted")
    plan["auto_repair"] = {"status": "applied"}
    seen = {}

    class Model:
        def generate_content(self, parts, generation_config=None):
            seen["prompt"] = parts[0]
            return SimpleNamespace(text=json.dumps(plan))

    out = pr.refine_plan_with_feedback(plan, "fb", "obj", Model(), None)
    assert "human_review" not in out and "auto_repair" not in out
    assert "human_review" not in seen["prompt"]


# ------------------------------------------- 3. what may reopen a settled plan

def test_tool_result_says_the_plan_is_settled(tmp_path, spies):
    agent = make_agent(tmp_path)
    t = make_tools(tmp_path, agent, message_count=3)
    summary = t._review_summary()
    assert summary["human_review"] == "accepted"
    assert "SETTLED" in summary["plan_status"]
    assert agent.state["current_plan"]["human_review"]["turn"] == 3


def test_trigger_is_required(tmp_path, spies):
    t = make_tools(tmp_path, make_agent(tmp_path))
    out = json.loads(t.functions_map["refine_plan_with_results"](
        result_data="DESIGN REVIEW — resolve the critic's findings"))
    assert out["status"] == "error" and "trigger" in out["message"]
    assert spies["refine"] == []


def test_the_live_session_is_declined(tmp_path, spies, monkeypatch):
    """The three refinements of 2026-09-19, replayed against an approved plan
    in the turn it was approved in. None of them rewrites the plan."""
    agent = make_agent(tmp_path)
    t = make_tools(tmp_path, agent, message_count=1)
    t._review_summary()                                  # approval, turn 1
    refine = t.functions_map["refine_plan_with_results"]
    before = json.dumps(agent.state["current_plan"]["proposed_experiments"])

    # a) the honest label for a self-initiated design review
    answers(monkeypatch, "")                             # the 12-second ENTER
    a = json.loads(refine(
        result_data="DESIGN REVIEW — switch Cu to Au nanoparticles ...",
        trigger="blocking_defect",
        reason="Cu is a weak SERS enhancer, so the SERS leg cannot work."))
    assert a["status"] == "declined_by_human"
    # b) a second go is refused outright — the human already kept the plan
    b = json.loads(refine(
        result_data="SECOND DESIGN REVIEW ...", trigger="blocking_defect",
        reason="the Au switch broke the KPFM pairing"))
    assert b["status"] == "declined" and "kept" in b["message"]
    # c) claiming the user asked, in the turn the user approved
    c = json.loads(refine(result_data="FINAL DESIGN CORRECTION ...",
                          trigger="user_request"))
    assert c["status"] == "declined" and "not a user request" in c["message"]

    state = agent.state
    assert json.dumps(state["current_plan"]["proposed_experiments"]) == before
    assert state["iteration_index"] == 1
    assert state["experimental_results"] == []
    assert len(state["plan_history"]) == 1
    assert len(spies["refine"]) == 1          # only (a) ever reached the model


def test_user_request_in_a_later_turn_goes_through(tmp_path, spies, monkeypatch):
    agent = make_agent(tmp_path)
    t = make_tools(tmp_path, agent, message_count=1)
    t._review_summary()
    t.orch.message_count = 2                             # the user spoke again
    answers(monkeypatch, "")                             # normal gate: ENTER accepts
    out = json.loads(t.functions_map["refine_plan_with_results"](
        result_data="swap glassy carbon for ITO", trigger="user_request"))
    assert out["status"] == "success"
    assert out["human_review"] == "accepted"
    assert revised(agent.state["current_plan"])


def test_blocking_defect_needs_a_reason(tmp_path, spies):
    t = make_tools(tmp_path, make_agent(tmp_path))
    out = json.loads(t.functions_map["adjust_plan_for_constraints"](
        constraint_description="", trigger="blocking_defect"))
    assert out["status"] == "error" and "reason" in out["message"]


def test_one_self_initiated_rewrite_per_plan(tmp_path, spies):
    """No human gate (autonomous): the budget is what stops the agent chasing
    a critic that finds something new every round."""
    agent = make_agent(tmp_path, make_plan())
    t = make_tools(tmp_path, agent, feedback=False)
    adjust = t.functions_map["adjust_plan_for_constraints"]
    first = json.loads(adjust(constraint_description="8-channel pipette cannot "
                              "address a 384-well plate", trigger="blocking_defect"))
    second = json.loads(adjust(constraint_description="another thought",
                               trigger="blocking_defect"))
    assert first["status"] == "success"
    assert second["status"] == "declined" and "one" in second["message"]
    assert len(spies["refine"]) == 1


def test_portfolio_edits_carry_a_trigger(tmp_path, spies):
    t = make_tools(tmp_path, make_agent(tmp_path, make_plan()), feedback=False)
    out = json.loads(t.functions_map["refine_portfolio"](request="drop DIR-4"))
    assert out["status"] == "error" and "trigger" in out["message"]
    out = json.loads(t.functions_map["refine_portfolio"](
        request="drop DIR-4", trigger="user_request"))
    assert out["status"] == "success"


# --------------------------------------------------------- 4. the reopen gate

def test_enter_keeps_the_approved_plan(tmp_path, spies, monkeypatch, capsys):
    agent = make_agent(tmp_path)
    approved = json.dumps(agent.state["current_plan"]["proposed_experiments"])
    answers(monkeypatch, "")
    out = agent.refine_plan("the cell cannot hold 150 C",
                            trigger="blocking_defect",
                            reopen_reason="aqueous cell at 150 C, 1 atm")
    assert json.dumps(out["proposed_experiments"]) == approved
    assert agent.state["status"] == "reopen_declined"
    assert out["human_review"]["reopen_declined"] == 1
    assert len(agent.state["plan_history"]) == 1
    assert agent.state["experimental_results"] == []
    assert agent.state["action_history"][-1]["action"] == "reopen_declined"
    shown = capsys.readouterr().out
    assert "REOPENING A PLAN YOU APPROVED" in shown
    assert "aqueous cell at 150 C, 1 atm" in shown


def test_accept_adopts_the_revision(tmp_path, spies, monkeypatch):
    agent = make_agent(tmp_path)
    answers(monkeypatch, "accept")
    out = agent.refine_plan("the cell cannot hold 150 C",
                            trigger="blocking_defect")
    assert revised(out)
    assert out["human_review"]["status"] == "adopted"
    assert agent.state["status"] == "refined"


def test_the_web_widget_reply_adopts(tmp_path, spies, monkeypatch):
    """The keep/revert widget sends 'keep' for its primary button."""
    agent = make_agent(tmp_path)
    answers(monkeypatch, "keep")
    out = agent.adjust_plan_for_constraints("pipette cannot reach the plate",
                                            trigger="blocking_defect")
    assert out["human_review"]["status"] == "adopted"


def test_results_and_user_constraints_use_the_ordinary_gate(tmp_path, spies,
                                                            monkeypatch, capsys):
    agent = make_agent(tmp_path)
    answers(monkeypatch, "")
    out = agent.refine_plan("Yield was 12%.")            # results: ENTER accepts
    assert revised(out)
    answers(monkeypatch, "")
    out = agent.adjust_plan_for_constraints("200 uL wells only",
                                            trigger="user_request")
    assert out["stage"] == "Constraint Adjusted"
    assert "REOPENING" not in capsys.readouterr().out


def test_reopen_gate_has_its_own_words_on_the_web(tmp_path):
    from scilink.hitl import FeedbackRequest
    from scilink.server.presenter import present_question
    req = FeedbackRequest(prompt="> Decision: ", kind="keep_or_revert",
                          options=["keep", "revert"], default="", context="",
                          origin={"stage": "plan_reopen",
                                  "reason": "aqueous cell at 150 C, 1 atm"})
    payload = present_question(req, "REOPENING A PLAN YOU APPROVED",
                               session_dir=str(tmp_path))
    assert payload["notice"]["lines"] == [
        "Reason given: aqueous cell at 150 C, 1 atm"]
    assert payload["widget"] == "keep_revert"
    assert payload["labels"]["keep"] == "Adopt the revision"
    assert payload["labels"]["revert"] == "Keep my approved plan"
    assert payload["labels"]["submit"] == "Adopt with changes"   # the third reply
    # a curve-fit keep/revert question keeps its two buttons and its words
    fit = FeedbackRequest(prompt="> ", kind="keep_or_revert",
                          options=["keep", "revert"], default="", context="",
                          origin={})
    labels = present_question(fit, "", session_dir=str(tmp_path))["labels"]
    assert labels == {"keep": "Keep user-guided fit",
                      "revert": "Revert to original fit"}


def test_plan_gate_offers_a_revert_button_after_an_auto_repair(tmp_path):
    """Found in the browser: the button was keyed on the printed repair notice,
    which a real plan's nine caveats pushed out of the presenter's 1500-char
    context tail. The question now says so itself."""
    from scilink.hitl import FeedbackRequest
    from scilink.server.presenter import present_question
    gate = "\n📝 REQUESTING FEEDBACK\nReview the plan and any caveats above."
    long_ctx = ("🔧 Auto-corrected before review (type 'revert' to restore the "
                "plan as authored)\n  • 150 C -> 60 C\n"
                + "  • [method] a long caveat\n" * 200 + gate)

    def ask(auto_repaired):
        return FeedbackRequest(prompt="> Instruction: ", kind="approve_or_revise",
                               options=None, default="", context="",
                               origin={"stage": "plan_review",
                                       "auto_repair": auto_repaired})
    shown = present_question(ask(["150 C -> 60 C (below the boiling point)"]),
                             long_ctx, session_dir=str(tmp_path))
    assert shown["labels"]["revert_repair"] == "Revert auto-correction"
    assert shown["labels"]["accept"] == "Approve plan"
    # the reviewer is told WHAT the button would revert, beside the button
    assert shown["notice"] == {"title": "Auto-corrected before review",
                               "lines": ["150 C -> 60 C (below the boiling point)"]}
    plain = present_question(ask([]), long_ctx, session_dir=str(tmp_path))
    assert "revert_repair" not in plain["labels"] and "notice" not in plain
    # several changes, one of them quoting a whole step: counted and clipped
    many = present_question(ask(["a -> b (w)", "x" * 900]), long_ctx,
                            session_dir=str(tmp_path))["notice"]
    assert many["title"] == "Auto-corrected before review (2 changes)"
    assert many["lines"][0] == "a -> b (w)"
    assert len(many["lines"][1]) < 360 and many["lines"][1].endswith("(full text above)")


def test_the_plan_gate_tells_the_front_end_about_the_repair(monkeypatch):
    from scilink.agents.planning_agents import user_interface as ui
    seen = {}
    monkeypatch.setattr(ui, "request_human_feedback",
                        lambda prompt, **kw: seen.update(kw) or "")
    ui.get_user_feedback(auto_repair={"status": "applied", "notes": [
        {"was": "150 C", "now": "60 C", "why": "below the boiling point"}]})
    assert seen["origin"] == {"stage": "plan_review", "auto_repair": [
        "150 C -> 60 C (below the boiling point)"]}
    ui.get_user_feedback(auto_repair={"status": "discarded", "reason": "r"})
    assert seen["origin"]["auto_repair"] == []       # nothing to revert
    ui.get_user_feedback()
    assert seen["origin"]["auto_repair"] == []


# ------------------------------------- 7. results and feedback are told apart

def test_only_results_advance_the_iteration(tmp_path, spies):
    agent = make_agent(tmp_path, make_plan())
    agent.refine_plan("swap the support", enable_human_feedback=False,
                      trigger="user_request")
    entry = agent.state["experimental_results"][-1]
    assert entry["kind"] == "feedback" and entry["trigger"] == "user_request"
    assert agent.state["iteration_index"] == 1
    assert agent.state["current_plan"]["iteration"] == 1
    assert agent.state["current_plan"]["stage"] == "Feedback Revision"
    assert "NOT been executed" in spies["refine"][-1]
    assert "not executed" in spies["critic"][-1]["human_feedback"]

    agent.refine_plan("Yield was 12%.", enable_human_feedback=False)
    entry = agent.state["experimental_results"][-1]
    assert entry["kind"] == "results"
    assert agent.state["iteration_index"] == 2
    assert agent.state["current_plan"]["stage"] == "Reasoning Draft"
    assert "We executed the previous plan" in spies["refine"][-1]


def test_html_box_prefers_real_results(tmp_path):
    from scilink.agents.planning_agents.html_generator import HTMLReportGenerator
    plan = make_plan()
    state = {"objective": "o", "plan_history": [plan], "current_plan": plan,
             "action_history": [], "experimental_results": [
                 {"iteration": 1, "kind": "feedback", "raw_input": "swap support"},
                 {"iteration": 1, "kind": "results", "raw_input": "run_005.csv"}]}
    out = tmp_path / "plan.html"
    HTMLReportGenerator(state).generate(str(out))
    h = out.read_text()
    assert "run_005.csv" in h and "swap support" not in h


# ------------------------------------------------- 5. the blocking-tier repair

def test_blocking_needs_a_stated_conflict():
    """The tier triggers a rewrite, so it is held to a checkable claim."""
    found = pr._normalise_blocking([
        dict(BLOCKING), {**BLOCKING, "conflict": ""}])
    assert [f["severity"] for f in found] == ["blocking", "critical"]
    assert format_caveats(found)[0].startswith("BLOCKING: [physics]")
    assert [f["severity"] for f in pr.sort_findings(
        [CRITICAL, {**CRITICAL, "severity": "minor"}, BLOCKING])] == [
            "blocking", "critical", "minor"]


def test_critic_prompt_defines_the_blocking_tier():
    seen = {}

    class Model:
        def generate_content(self, parts, generation_config=None):
            seen["prompt"] = parts[0]
            return SimpleNamespace(text=json.dumps({"findings": [BLOCKING]}))

    out = pr.critique_plan("obj", make_plan(), Model(), None)
    assert out["findings"][0]["severity"] == "blocking"
    assert "cannot be run as written" in seen["prompt"]
    assert '"conflict"' in seen["prompt"]


def test_the_critic_reads_the_protocol():
    """Live: the critic called replication 'never defined' and an inert
    transfer 'not specified' on a plan whose steps specified both — it was
    shown the name, hypothesis and justification only. It can neither confirm
    a control nor see an unrunnable step without the protocol."""
    plan = make_plan()
    plan["proposed_experiments"][0]["optimization_params"] = [
        {"parameter_name": "Temperature", "min_value": 20.0, "max_value": 80.0}]
    view = pr.summarize_plan_for_critic(plan)
    assert "hold the aqueous bicarbonate cell at 150 C and 1 atm" in view
    assert "Equipment: potentiostat; UV-Vis; KPFM" in view
    assert "Temperature: 20.0 to 80.0" in view
    assert "Expected outcome: LSPR shift" in view
    # the conformance pass keeps its coverage-and-identity view
    assert "150 C" not in pr.summarize_experiment(plan["proposed_experiments"][0], 1)
    # a portfolio's experiment entry is a shim: its design is the directions
    portfolio = make_plan()
    portfolio["directions"] = [{"id": "D1", "title": "T", "details": "the design"}]
    pview = pr.summarize_plan_for_critic(portfolio)
    assert "the design" in pview and "PROTOCOL of Experiment" not in pview


def _repairing(spies, steps_change=True, **override):
    """Make the fake refiner behave like a fix-only repair."""
    def fake_refine(original_result, feedback, **kw):
        spies["refine"].append(feedback)
        new = json.loads(json.dumps(original_result))
        exp = new["proposed_experiments"][0]
        if steps_change:
            exp["experimental_steps"][1] = (
                "Step 2: hold the aqueous bicarbonate cell at 60 C and 1 atm")
        exp.update(override)
        new["repair_notes"] = [{"finding": 1, "was": "150 C", "now": "60 C",
                                "why": "below the boiling point"}]
        return new
    return fake_refine


def test_blocking_defect_is_repaired_in_place(tmp_path, spies, monkeypatch):
    monkeypatch.setattr(pr, "refine_plan_with_feedback", _repairing(spies))
    agent = make_agent(tmp_path, make_plan(findings=[BLOCKING, CRITICAL]))
    spies["critic_verdict"] = {"findings": [CRITICAL]}      # conflict resolved
    out = agent._auto_repair_blocking(agent.state["current_plan"], iteration=1)

    assert out["stage"] == "Auto-Corrected (critic)"
    assert "60 C" in out["proposed_experiments"][0]["experimental_steps"][1]
    assert out["auto_repair"]["status"] == "applied"
    assert out["critic_findings"] == [CRITICAL]             # stays advisory
    assert "repair_notes" not in out
    assert format_auto_repair(out["auto_repair"]) == [
        "150 C -> 60 C (below the boiling point)"]
    # the contract the author was given, and the evidence the verifier got
    assert "Fix ONLY the cited conflicts" in spies["refine"][0]
    assert "150 C at 1 atm vs" in spies["refine"][0]
    assert spies["critic"][-1]["prior_findings"] == [BLOCKING, CRITICAL]
    assert agent.state["plan_history"][-1]["stage"] == "Auto-Corrected (critic)"
    assert len(spies["refine"]) == 1                        # one attempt, no loop


@pytest.mark.parametrize("override, verdict, why", [
    ({"hypothesis": "Au nanoparticles ..."}, {"findings": []}, "hypothesis changed"),
    ({"experiment_name": "Au-nanoparticle CO2RR"}, {"findings": []},
     "experiment name changed"),
    ({"experimental_steps": ["a wholly different protocol"],
      "justification": "rewritten", "expected_outcome": "rewritten",
      "required_equipment": ["something else"]}, {"findings": []},
     "largely rewritten"),
    ({}, {"findings": [BLOCKING]}, "not resolved"),
    ({}, {"findings": [CRITICAL, {**CRITICAL, "issue": "a new one",
                                  "introduced": True}]},
     "new critical finding"),
    ({}, {"findings": [], "failed": True}, "could not be verified"),
])
def test_a_repair_that_is_not_a_repair_is_discarded(tmp_path, spies, monkeypatch,
                                                    override, verdict, why):
    monkeypatch.setattr(pr, "refine_plan_with_feedback",
                        _repairing(spies, **override))
    agent = make_agent(tmp_path, make_plan(findings=[BLOCKING, CRITICAL]))
    authored = json.dumps(agent.state["current_plan"]["proposed_experiments"])
    spies["critic_verdict"] = verdict
    out = agent._auto_repair_blocking(agent.state["current_plan"], iteration=1)

    assert json.dumps(out["proposed_experiments"]) == authored
    assert out["auto_repair"]["status"] == "discarded"
    assert why in out["auto_repair"]["reason"]
    assert out["critic_findings"][0]["severity"] == "blocking"   # caveat stands
    assert len(agent.state["plan_history"]) == 1
    assert "discarded" in format_auto_repair(out["auto_repair"])[0]


def test_advisory_findings_the_repair_did_not_create_do_not_sink_it(
        tmp_path, spies, monkeypatch):
    """Live: a verified repair was discarded because the re-critique raised
    one more advisory finding than the first pass did — it raises a different
    set every time. Only a finding the repair itself created counts."""
    monkeypatch.setattr(pr, "refine_plan_with_feedback", _repairing(spies))
    agent = make_agent(tmp_path, make_plan(findings=[BLOCKING]))
    spies["critic_verdict"] = {"findings": [
        CRITICAL, {**CRITICAL, "issue": "another pre-existing weakness"}]}
    out = agent._auto_repair_blocking(agent.state["current_plan"], iteration=1)
    assert out["auto_repair"]["status"] == "applied"
    assert len(out["critic_findings"]) == 2


def test_author_may_decline_the_repair(tmp_path, spies, monkeypatch):
    def declines(original_result, feedback, **kw):
        new = json.loads(json.dumps(original_result))
        new["repair_declined"] = "the hypothesis itself names 150 C"
        return new
    monkeypatch.setattr(pr, "refine_plan_with_feedback", declines)
    agent = make_agent(tmp_path, make_plan(findings=[BLOCKING]))
    out = agent._auto_repair_blocking(agent.state["current_plan"], iteration=1)
    assert out["auto_repair"]["status"] == "discarded"
    assert "names 150 C" in out["auto_repair"]["reason"]
    assert spies["critic"] == []                            # nothing to verify


def test_critical_and_minor_findings_are_never_acted_on(tmp_path, spies):
    """The five critical findings of the live session trigger nothing."""
    agent = make_agent(tmp_path, make_plan(findings=[CRITICAL] * 5))
    out = agent._auto_repair_blocking(agent.state["current_plan"], iteration=1)
    assert "auto_repair" not in out
    assert spies["refine"] == [] and spies["critic"] == []


def test_reviewer_can_revert_the_repair(tmp_path, spies, monkeypatch):
    monkeypatch.setattr(pr, "refine_plan_with_feedback", _repairing(spies))
    agent = make_agent(tmp_path, make_plan(findings=[BLOCKING]))
    repaired = agent._auto_repair_blocking(agent.state["current_plan"], iteration=1)
    assert agent._is_revert_request(" Revert. ", repaired)
    assert not agent._is_revert_request("revert", make_plan())
    restored = agent._revert_auto_repair(repaired)
    assert "150 C" in restored["proposed_experiments"][0]["experimental_steps"][1]
    assert restored["auto_repair"]["status"] == "reverted"
    assert restored["critic_findings"][0]["severity"] == "blocking"


def test_repair_runs_before_the_plan_is_shown(tmp_path, monkeypatch):
    """End to end through generate_plan: authored -> critic says blocking ->
    repaired -> verified -> THEN the gate, which shows what was changed."""
    authored = {"proposed_experiments": [experiment()]}
    fixed = json.loads(json.dumps(authored))
    fixed["proposed_experiments"][0]["experimental_steps"][1] = (
        "Step 2: hold the aqueous bicarbonate cell at 60 C and 1 atm")
    fixed["repair_notes"] = [{"finding": 1, "was": "150 C", "now": "60 C",
                              "why": "below the boiling point"}]
    replies = [json.dumps(authored), json.dumps(fixed)]
    prompts = []

    class Model:
        def generate_content(self, parts, generation_config=None):
            prompts.append(parts[0] if isinstance(parts, list) else parts)
            return SimpleNamespace(text=replies.pop(0))

    verdicts = [{"findings": [BLOCKING]}, {"findings": []}]
    monkeypatch.setattr(pa_mod, "verify_plan_relevance", lambda *a, **k: (True, ""))
    monkeypatch.setattr(pa_mod, "critique_plan", lambda *a, **k: verdicts.pop(0))
    shown = []
    monkeypatch.setattr(pa_mod, "display_plan_summary",
                        lambda res, **k: shown.append(json.loads(json.dumps(res))))
    answers(monkeypatch, "")

    a = PlanningAgent.__new__(PlanningAgent)
    BaseAgent.__init__(a, str(tmp_path))
    a.agent_type, a.model, a.generation_config = "planning", Model(), None
    a.kb_docs = a.kb_code = a.lit_agent = None
    a._ensure_kb_is_ready = lambda *x, **k: False
    res = a.generate_plan("obj", enable_human_feedback=True)

    assert shown[0]["auto_repair"]["status"] == "applied"   # the gate saw the fix
    assert "60 C" in res["proposed_experiments"][0]["experimental_steps"][1]
    assert res["human_review"]["status"] == "accepted"
    assert [p["stage"] for p in a.state["plan_history"]] == [
        "Science Draft", "Auto-Corrected (critic)"]
    assert a.state["self_revisions"] == 0


def test_selection_prompt_says_the_pick_was_auto_corrected(tmp_path, monkeypatch):
    """Candidate cards show plans as authored, so the repair of the judge's
    pick has to be announced beside its caveats."""
    def cand(name, temp):
        e = experiment(name=name)
        e["experimental_steps"][1] = f"Step 2: hold the aqueous cell at {temp} and 1 atm"
        return {"proposed_experiments": [e]}
    fixed = cand("P1", "60 C")
    fixed["repair_notes"] = [{"finding": 1, "was": "150 C", "now": "60 C", "why": "w"}]
    replies = [json.dumps(cand("P1", "150 C")), json.dumps(cand("P2", "25 C")),
               json.dumps({"scores": [], "selected_candidate": 1, "reasoning": "r"}),
               json.dumps(fixed)]

    class Model:
        def generate_content(self, parts, generation_config=None):
            return SimpleNamespace(text=replies.pop(0))

    verdicts = [{"findings": [BLOCKING]}, {"findings": []}]
    monkeypatch.setattr(pa_mod, "verify_plan_relevance", lambda *a, **k: (True, ""))
    monkeypatch.setattr(pa_mod, "critique_plan", lambda *a, **k: verdicts.pop(0))
    seen = {}
    monkeypatch.setattr(pa_mod, "display_plan_candidates",
                        lambda *a, **k: seen.update(k))
    answers(monkeypatch, "", "")

    a = PlanningAgent.__new__(PlanningAgent)
    BaseAgent.__init__(a, str(tmp_path))
    a.agent_type, a.model, a.generation_config = "planning", Model(), None
    a.kb_docs = a.kb_code = a.lit_agent = None
    a._ensure_kb_is_ready = lambda *x, **k: False
    a.generate_plan("obj", enable_human_feedback=True, n_candidates=2)
    assert seen["pick_caveats"] == ["Auto-repair: 150 C -> 60 C (w)"]


def test_revert_is_a_declined_reopen(tmp_path, monkeypatch):
    """Live: after the human typed 'revert', the orchestrator proposed the
    same fix again through the reopen gate. They had already refused it."""
    authored = {"proposed_experiments": [experiment()]}
    fixed = json.loads(json.dumps(authored))
    fixed["proposed_experiments"][0]["experimental_steps"][1] = (
        "Step 2: hold the aqueous bicarbonate cell at 60 C and 1 atm")
    replies = [json.dumps(authored), json.dumps(fixed)]

    class Model:
        def generate_content(self, parts, generation_config=None):
            return SimpleNamespace(text=replies.pop(0))

    verdicts = [{"findings": [BLOCKING]}, {"findings": []}]
    monkeypatch.setattr(pa_mod, "verify_plan_relevance", lambda *a, **k: (True, ""))
    monkeypatch.setattr(pa_mod, "critique_plan", lambda *a, **k: verdicts.pop(0))
    answers(monkeypatch, "revert")
    a = PlanningAgent.__new__(PlanningAgent)
    BaseAgent.__init__(a, str(tmp_path))
    a.agent_type, a.model, a.generation_config = "planning", Model(), None
    a.kb_docs = a.kb_code = a.lit_agent = None
    a._ensure_kb_is_ready = lambda *x, **k: False
    res = a.generate_plan("obj", enable_human_feedback=True)
    assert "150 C" in res["proposed_experiments"][0]["experimental_steps"][1]
    assert res["human_review"]["reopen_declined"] == 1

    t = make_tools(tmp_path, a)
    out = json.loads(t.functions_map["refine_plan_with_results"](
        result_data="lower the cell temperature", trigger="blocking_defect",
        reason="an aqueous cell cannot be held at 150 C at 1 atm"))
    assert out["status"] == "declined"


# ------------------------- 8. selection, repair and what happens when it fails

def test_judge_does_not_mark_down_what_the_repair_fixes():
    """The repair runs on the winner only, so a fixable defect must not decide
    who wins: the judge scores the design as it would stand once fixed."""
    from scilink.agents.planning_agents import instruct
    prompt = instruct.HYPOTHESIS_BEST_OF_N_SELECTION_INSTRUCTIONS
    assert "NOT a reason to score a candidate down" in prompt
    assert '"fixable_defects"' in prompt
    assert "changing\nthe hypothesis or the technique is not fixable" in prompt


def _bestofn_agent(tmp_path, monkeypatch, verdict_for, judge_scores=None):
    """Three candidates, judge picks 1; the critic's verdict is per candidate
    name; every repair is declined by its author."""
    def cand(name):
        return json.dumps({"proposed_experiments": [experiment(name=name)]})
    judge = {"scores": judge_scores or [
        {"candidate": 1, "feasibility": 5, "information_gain": 5},
        {"candidate": 2, "feasibility": 2, "information_gain": 2},
        {"candidate": 3, "feasibility": 4, "information_gain": 4}],
        "selected_candidate": 1, "reasoning": "r"}
    replies = [cand("P1"), cand("P2"), cand("P3"), json.dumps(judge)]

    class Model:
        def generate_content(self, parts, generation_config=None):
            return SimpleNamespace(text=replies.pop(0))

    critiqued = []

    def critic(objective, result, *a, **k):
        name = result["proposed_experiments"][0]["experiment_name"]
        critiqued.append(name)
        return {"findings": list(verdict_for(name))}

    def declines(plan, *a, **k):
        out = json.loads(json.dumps(plan))
        out["repair_declined"] = "the hypothesis names the impossible condition"
        return out

    monkeypatch.setattr(pa_mod, "verify_plan_relevance", lambda *a, **k: (True, ""))
    monkeypatch.setattr(pa_mod, "critique_plan", critic)
    monkeypatch.setattr(pa_mod, "repair_blocking_defects", declines)
    a = PlanningAgent.__new__(PlanningAgent)
    BaseAgent.__init__(a, str(tmp_path))
    a.agent_type, a.model, a.generation_config = "planning", Model(), None
    a.kb_docs = a.kb_code = a.lit_agent = None
    a._ensure_kb_is_ready = lambda *x, **k: False
    return a, critiqued


def test_unrunnable_pick_falls_back_to_the_best_runnable_candidate(tmp_path,
                                                                   monkeypatch):
    a, critiqued = _bestofn_agent(
        tmp_path, monkeypatch,
        lambda name: [BLOCKING] if name == "P1" else [CRITICAL])
    res = a.generate_plan("obj", enable_human_feedback=False, n_candidates=3)
    # by the judge's own scores the next best is 3, not 2
    assert res["proposed_experiments"][0]["experiment_name"] == "P3"
    assert critiqued == ["P1", "P3"]                  # 2 was never needed
    pc = a.state["plan_candidates"]
    assert pc["selected_index"] == 3
    assert pc["fallback"]["from"] == 1 and pc["fallback"]["to"] == 3
    assert res["critic_findings"] == [CRITICAL]       # advisory caveats ride along
    assert a.state["current_plan"] is res


def test_an_advisory_caveat_never_switches_the_selection(tmp_path, monkeypatch):
    a, critiqued = _bestofn_agent(tmp_path, monkeypatch,
                                  lambda name: [CRITICAL] * 5)
    res = a.generate_plan("obj", enable_human_feedback=False, n_candidates=3)
    assert res["proposed_experiments"][0]["experiment_name"] == "P1"
    assert critiqued == ["P1"] and "fallback" not in a.state["plan_candidates"]


def test_no_runnable_candidate_keeps_the_pick_and_tells_the_orchestrator(
        tmp_path, monkeypatch, spies):
    """Nothing is enforced on the critic's word: the finding and the failed
    repair are put in front of the orchestrator, which decides."""
    a, critiqued = _bestofn_agent(tmp_path, monkeypatch, lambda name: [BLOCKING])
    res = a.generate_plan("obj", enable_human_feedback=False, n_candidates=3)
    assert critiqued == ["P1", "P3", "P2"]            # bounded by the candidates
    assert res["proposed_experiments"][0]["experiment_name"] == "P1"
    assert a.state["plan_candidates"]["selected_index"] == 1
    assert a.state["plan_history"][-1]["stage"] == (
        "Science Draft (no runnable candidate)")

    t = make_tools(tmp_path, a, feedback=False)
    told = t._review_summary()["unresolved_blocking_finding"]
    assert "150 C at 1 atm" in told["conflict"]
    assert "critic can be wrong" in told["note"]
    # no hard stop: code generation is not refused on the finding
    src = Path("scilink/agents/planning_agents/orchestrator_tools.py").read_text()
    assert '"status": "blocked"' not in src
    orch = Path("scilink/agents/planning_agents/planning_orchestrator.py").read_text()
    assert "_standing_blocker" not in orch             # run_task does not overrule


def test_a_human_review_settles_the_finding(tmp_path, spies):
    """The reviewer was shown the finding; it is theirs, not unresolved."""
    agent = make_agent(tmp_path, make_plan(findings=[BLOCKING], reviewed="accepted"))
    t = make_tools(tmp_path, agent)
    assert t._standing_blocker() is None
    assert "unresolved_blocking_finding" not in t._review_summary()
    agent.state["current_plan"].pop("human_review")
    assert t._standing_blocker()["severity"] == "blocking"


def test_human_at_the_gate_means_no_automatic_fallback(tmp_path, monkeypatch):
    a, critiqued = _bestofn_agent(
        tmp_path, monkeypatch,
        lambda name: [BLOCKING] if name == "P1" else [])
    answers(monkeypatch, "", "")                      # accept pick, approve plan
    res = a.generate_plan("obj", enable_human_feedback=True, n_candidates=3)
    assert res["proposed_experiments"][0]["experiment_name"] == "P1"
    assert critiqued == ["P1"]


def test_the_agents_own_revision_is_held_to_the_repair_contract(tmp_path, spies):
    """Autonomous: no gate, so the scope guard is what stops a 'defect fix'
    from switching the material system."""
    agent = make_agent(tmp_path, make_plan())
    before = json.dumps(agent.state["current_plan"]["proposed_experiments"])
    t = make_tools(tmp_path, agent, feedback=False)
    spies["rename_to"] = "Au-nanoparticle CO2RR-to-CO"          # a redesign
    out = json.loads(t.functions_map["refine_plan_with_results"](
        result_data="Cu is a weak SERS enhancer; switch to Au",
        trigger="blocking_defect", reason="the SERS leg cannot work on Cu"))
    assert out["status"] == "declined"
    assert "experiment name changed" in out["message"]
    state = agent.state
    assert json.dumps(state["current_plan"]["proposed_experiments"]) == before
    assert len(state["plan_history"]) == 1
    assert state["experimental_results"] == []
    assert state["self_revisions"] == 1               # the budget is spent


def test_a_local_self_revision_stands(tmp_path, spies):
    agent = make_agent(tmp_path, make_plan())
    out = agent.adjust_plan_for_constraints(
        "8-channel pipette cannot address a 384-well plate",
        enable_human_feedback=False, trigger="blocking_defect")
    assert revised(out) and agent.state["status"] == "constraint_adjusted"
    spies["critic_verdict"] = {"findings": [{**CRITICAL, "introduced": True}]}
    agent.state["self_revisions"] = 0
    out = agent.adjust_plan_for_constraints(
        "another", enable_human_feedback=False, trigger="blocking_defect")
    assert agent.state["status"] == "revision_discarded"
    assert "introduced" in agent.state["last_discard_reason"]


# ------------------------------------------ 6. state-backed files are not edited

@pytest.mark.parametrize("name", ["plan.json", "plan.html", "planning_state.json",
                                  "tea_analysis.json", "x.state.json"])
def test_file_tools_refuse_state_backed_files(tmp_path, spies, name):
    (tmp_path / name).write_text('{"hypothesis": "Cu"}')
    t = make_tools(tmp_path, make_agent(tmp_path))
    out = json.loads(t.functions_map["edit_file"](
        path=str(tmp_path / name), old_text="Cu", new_text="Au"))
    assert out["status"] == "declined"
    assert "refine_plan_with_results" in out["message"]
    assert (tmp_path / name).read_text() == '{"hypothesis": "Cu"}'
    for tool, kw in (("save_file", {"filename": name, "content": "x"}),
                     ("append_file", {"filename": name, "content": "x"}),
                     ("rename_file", {"path": str(tmp_path / name),
                                      "new_name": "old.json"})):
        assert json.loads(t.functions_map[tool](**kw))["status"] == "declined"


def test_documents_are_still_editable(tmp_path, spies):
    doc = tmp_path / "white_paper.md"
    doc.write_text("The catalyst is Cu.")
    t = make_tools(tmp_path, make_agent(tmp_path))
    t._refresh_pdf_twin = lambda *a, **k: False
    out = json.loads(t.functions_map["edit_file"](
        path=str(doc), old_text="Cu", new_text="Cu (sparse array)"))
    assert out["status"] == "success"
    assert doc.read_text() == "The catalyst is Cu (sparse array)."


# ------------------------------------------------------ 2. the stated principle

def test_autopilot_directive_states_the_principle():
    from scilink.agents.planning_agents import planning_orchestrator as po
    assert "SETTLED" in po._AUTOPILOT_DIRECTIVE
    assert "impossible or unsafe to run" in po._AUTOPILOT_DIRECTIVE
    # nobody approves a plan in an autonomous run, so nothing is "settled";
    # the same bar is stated on its own there
    assert "caveats are advisory" in po._AUTONOMOUS_DIRECTIVE
    assert "impossible or unsafe to run" in po._AUTONOMOUS_DIRECTIVE
