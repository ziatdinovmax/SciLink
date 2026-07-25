"""Offline tests for best-of-N hypothesis generation in plan mode.

Covers the contract pinned in issue #377:
- n_candidates=1 is a no-op (single-plan path, no candidate state);
- sequential candidates are diversity-conditioned on prior hypotheses;
- retrieval evidence is shared (same retrieved-context block in every prompt);
- N is a cap: a decline (error JSON) stops generation early and skips judging;
- fallback parity: the tier is decided at candidate 1 and pinned for the run;
- the judge selects; invalid/broken judge output fails open to candidate 1;
- human override switches the selection and re-runs the (advisory) critic —
  the lazy-critique invariant;
- the critic is never acted on automatically (no auto-refine, no auto-switch);
- state["plan_candidates"] is JSON-serializable (checkpoint rides along);
- the CLI candidate block round-trips through the UI parser (CLI→UI contract).

All LLM traffic is a scripted mock; no network.
"""

import ast
import builtins
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from scilink.agents.planning_agents.planning_agent import PlanningAgent
from scilink.agents.planning_agents.base_agent import BaseAgent
from scilink.agents.planning_agents import planning_agent as pa_mod
from scilink.agents.planning_agents.user_interface import display_plan_candidates


# ---------------------------------------------------------------- helpers

def plan_json(name, hypothesis):
    return json.dumps({
        "proposed_experiments": [{
            "hypothesis": hypothesis,
            "experiment_name": name,
            "experimental_steps": ["Step 1: do the thing", "Step 2: measure"],
            "required_equipment": ["Instrument A"],
            "optimization_params": [
                {"parameter_name": "Temperature", "parameter_type": "continuous",
                 "min_value": 20.0, "max_value": 80.0, "rationale": "range"}],
            "expected_outcome": f"Outcome supporting {name}",
            "justification": f"Justified by evidence for {name}",
            "source_documents": ["doc1.pdf"],
        }]
    })


def judge_json(pick, n):
    return json.dumps({
        "scores": [
            {"candidate": i, "groundedness": 4, "testability": 4,
             "actionability": 4, "feasibility": 4, "information_gain": 3 + (i == pick),
             "comment": f"Candidate {i} comment."}
            for i in range(1, n + 1)
        ],
        "selected_candidate": pick,
        "reasoning": f"Candidate {pick} best balances the criteria.",
    })


DECLINE = json.dumps({"error": "Insufficient context: no additional distinct "
                               "approach is supported by the provided evidence."})


class ScriptedModel:
    """Returns canned responses in order; records every prompt it saw."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_content(self, prompt_parts, generation_config=None):
        if isinstance(prompt_parts, str):
            prompt_parts = [prompt_parts]
        self.calls.append("\n".join(p for p in prompt_parts if isinstance(p, str)))
        if not self.responses:
            raise AssertionError("ScriptedModel exhausted — unexpected extra LLM call")
        return SimpleNamespace(text=self.responses.pop(0))


def make_agent(tmp_path, model):
    agent = PlanningAgent.__new__(PlanningAgent)
    BaseAgent.__init__(agent, str(tmp_path))
    agent.agent_type = "planning"
    agent.model = model
    agent.generation_config = None
    agent.kb_docs = None
    agent.kb_code = None
    agent.lit_agent = None
    agent._ensure_kb_is_ready = lambda *a, **k: False
    return agent


@pytest.fixture
def no_verify_no_critic(monkeypatch):
    """Patch conformance + critic to spies so the ScriptedModel only serves
    generation and judge calls. Returns (verify_calls, critic_calls) lists."""
    verify_calls, critic_calls = [], []
    monkeypatch.setattr(pa_mod, "verify_plan_relevance",
                        lambda *a, **k: (verify_calls.append(a), (True, ""))[1])
    monkeypatch.setattr(pa_mod, "critique_plan",
                        lambda *a, **k: (critic_calls.append(a), {"findings": []})[1])
    return verify_calls, critic_calls


# ---------------------------------------------------------------- tests

def test_n1_is_single_plan_noop(tmp_path, no_verify_no_critic):
    model = ScriptedModel([plan_json("Solo", "H-solo")])
    agent = make_agent(tmp_path, model)
    res = agent.generate_plan("obj", enable_human_feedback=False)
    assert res["proposed_experiments"][0]["experiment_name"] == "Solo"
    assert len(model.calls) == 1
    assert "plan_candidates" not in agent.state
    assert "PRIOR CANDIDATE HYPOTHESES" not in model.calls[0]


def test_sequential_conditioning_and_shared_evidence(tmp_path, no_verify_no_critic):
    model = ScriptedModel([
        plan_json("P1", "H1: mechanism alpha"),
        plan_json("P2", "H2: mechanism beta"),
        plan_json("P3", "H3: mechanism gamma"),
        judge_json(2, 3),
    ])
    agent = make_agent(tmp_path, model)
    res = agent.generate_plan("obj", enable_human_feedback=False, n_candidates=3)

    gen1, gen2, gen3, judge = model.calls
    # candidate 1 is unconditioned; 2 and 3 see all prior hypotheses
    assert "PRIOR CANDIDATE HYPOTHESES" not in gen1
    assert "PRIOR CANDIDATE HYPOTHESES" in gen2 and "H1: mechanism alpha" in gen2
    assert "H1: mechanism alpha" in gen3 and "H2: mechanism beta" in gen3
    assert "DIFFERENT mechanistic hypothesis" in gen3
    # shared evidence: identical retrieved-context block in every author prompt
    marker = "No specific documents found in Knowledge Base."
    assert all(marker in g for g in (gen1, gen2, gen3))
    # judge saw all three candidates and the pick won
    assert "CANDIDATE 3" in judge and "selector, not an editor" in judge
    assert res["proposed_experiments"][0]["experiment_name"] == "P2"

    pc = agent.state["plan_candidates"]
    assert pc["selected_index"] == 2
    assert pc["human_override"] is False
    assert pc["tier"] == "strict"
    assert len(pc["candidates"]) == 3
    json.dumps(pc)  # checkpoint rides planner_state — must serialize


def test_decline_stops_early_and_skips_judge(tmp_path, no_verify_no_critic):
    model = ScriptedModel([
        plan_json("P1", "H1"),
        DECLINE,  # candidate 2 declines -> stop; judge never called
    ])
    agent = make_agent(tmp_path, model)
    res = agent.generate_plan("obj", enable_human_feedback=False, n_candidates=4)
    assert len(model.calls) == 2
    assert res["proposed_experiments"][0]["experiment_name"] == "P1"
    pc = agent.state["plan_candidates"]
    assert len(pc["candidates"]) == 1 and pc["selected_index"] == 1
    assert pc["judge"] is None


def test_fallback_tier_pinned_for_whole_run(tmp_path, no_verify_no_critic):
    model = ScriptedModel([
        json.dumps({"error": "Insufficient context for a specific plan."}),
        plan_json("F1", "H1-fallback"),   # candidate 1 fallback retry
        plan_json("F2", "H2-fallback"),   # candidate 2, authored in fallback tier
        judge_json(1, 2),
    ])
    agent = make_agent(tmp_path, model)
    agent.generate_plan("obj", enable_human_feedback=False, n_candidates=2)

    strict1, fb1, gen2, judge = model.calls
    assert "FALLBACK MODE ACTIVATED" not in strict1
    assert "FALLBACK MODE ACTIVATED" in fb1
    # tier pinned: candidate 2 authored directly under fallback instructions,
    # with the distinctness conditioning on top
    assert "FALLBACK MODE ACTIVATED" in gen2
    assert "PRIOR CANDIDATE HYPOTHESES" in gen2 and "H1-fallback" in gen2
    assert agent.state["plan_candidates"]["tier"] == "fallback"
    # judge told to compare within the fallback tier
    assert "fallback mode" in judge


def test_judge_invalid_selection_fails_open(tmp_path, no_verify_no_critic):
    model = ScriptedModel([
        plan_json("P1", "H1"),
        plan_json("P2", "H2"),
        json.dumps({"scores": [], "selected_candidate": 7, "reasoning": "bad"}),
    ])
    agent = make_agent(tmp_path, model)
    res = agent.generate_plan("obj", enable_human_feedback=False, n_candidates=2)
    pc = agent.state["plan_candidates"]
    assert pc["selected_index"] == 1
    assert "error" in pc["judge"]
    assert res["proposed_experiments"][0]["experiment_name"] == "P1"


def test_human_override_switches_and_recritiques(tmp_path, monkeypatch,
                                                 no_verify_no_critic):
    verify_calls, critic_calls = no_verify_no_critic
    model = ScriptedModel([
        plan_json("P1", "H1"),
        plan_json("P2", "H2"),
        plan_json("P3", "H3"),
        judge_json(1, 3),
    ])
    agent = make_agent(tmp_path, model)
    replies = iter(["3", ""])  # stage 1: override to 3; stage 2: approve as-is
    monkeypatch.setattr(builtins, "input", lambda *a: next(replies))

    res = agent.generate_plan("obj", enable_human_feedback=True, n_candidates=3)

    pc = agent.state["plan_candidates"]
    assert pc["selected_index"] == 3 and pc["human_override"] is True
    assert res["proposed_experiments"][0]["experiment_name"] == "P3"
    assert agent.state["current_plan"]["proposed_experiments"][0]["experiment_name"] == "P3"
    # lazy-critique invariant: judge's pick AND the overridden choice were
    # each conformance-checked and critiqued; nothing else was
    assert len(verify_calls) == 2 and len(critic_calls) == 2
    # advisory-only: no extra generation happened in response to criticism
    assert len(model.calls) == 4


def test_enter_accepts_judge_pick(tmp_path, monkeypatch, no_verify_no_critic):
    verify_calls, critic_calls = no_verify_no_critic
    model = ScriptedModel([
        plan_json("P1", "H1"),
        plan_json("P2", "H2"),
        judge_json(2, 2),
    ])
    agent = make_agent(tmp_path, model)
    replies = iter(["", ""])  # accept pick; approve plan
    monkeypatch.setattr(builtins, "input", lambda *a: next(replies))

    res = agent.generate_plan("obj", enable_human_feedback=True, n_candidates=2)
    pc = agent.state["plan_candidates"]
    assert pc["selected_index"] == 2 and pc["human_override"] is False
    assert res["proposed_experiments"][0]["experiment_name"] == "P2"
    assert len(verify_calls) == 1 and len(critic_calls) == 1


def test_candidate_html_reports_written(tmp_path, no_verify_no_critic):
    model = ScriptedModel([
        plan_json("P1", "H1"),
        plan_json("P2", "H2"),
        judge_json(1, 2),
    ])
    agent = make_agent(tmp_path, model)
    report_dir = tmp_path / "plan_candidates"
    agent.generate_plan("obj", enable_human_feedback=False, n_candidates=2,
                        candidate_report_dir=str(report_dir))
    files = sorted(p.name for p in report_dir.glob("*.html"))
    assert files == ["candidate_1.html", "candidate_2.html"]
    html = (report_dir / "candidate_2.html").read_text()
    assert "H2" in html and "P2" in html
    assert agent.state["plan_candidates"]["reports"] == [
        str(report_dir / "candidate_1.html"), str(report_dir / "candidate_2.html")]


def test_selection_profile_reaches_judge(tmp_path, no_verify_no_critic):
    """The judge prompt carries the LAB weighting by default and the
    IDEATION weighting when requested; candidates/authors are unaffected
    (profile changes the pick's weighting, not generation)."""
    def responses():
        return [plan_json("P1", "H1"), plan_json("P2", "H2"), judge_json(1, 2)]

    model = ScriptedModel(responses())
    agent = make_agent(tmp_path / "lab", model)
    agent.generate_plan("obj", enable_human_feedback=False, n_candidates=2)
    judge_prompt = model.calls[-1]
    assert "SELECTION WEIGHTING — LAB PROFILE" in judge_prompt
    assert "IDEATION PROFILE" not in judge_prompt
    assert all("SELECTION WEIGHTING" not in c for c in model.calls[:-1])

    model2 = ScriptedModel(responses())
    agent2 = make_agent(tmp_path / "disc", model2)
    agent2.generate_plan("obj", enable_human_feedback=False, n_candidates=2,
                         selection_profile="ideation")
    assert "SELECTION WEIGHTING — IDEATION PROFILE" in model2.calls[-1]
    assert agent2.state["plan_candidates"]["profile"] == "ideation"
    # ideation also relaxes AUTHOR-side derivability — every author prompt
    # carries the grounding-latitude override; lab authors never see it
    assert all("IDEATION MODE — GROUNDING LATITUDE" in c
               for c in model2.calls[:-1])
    assert all("GROUNDING LATITUDE" not in c for c in model.calls)


def test_ideation_profile_is_noop_at_n1(tmp_path, no_verify_no_critic):
    """DOCUMENTED behavior: ideation requires n_candidates >= 2. The
    single-plan path ignores the profile entirely — no grounding-latitude
    override, no weighting note, byte-identical to the pre-profile path."""
    model = ScriptedModel([plan_json("Solo", "H")])
    agent = make_agent(tmp_path, model)
    agent.generate_plan("obj", enable_human_feedback=False, n_candidates=1,
                        selection_profile="ideation")
    assert len(model.calls) == 1
    assert "GROUNDING LATITUDE" not in model.calls[0]
    assert "SELECTION WEIGHTING" not in model.calls[0]
    assert "plan_candidates" not in agent.state


def test_resolve_n_candidates_default_policy():
    """First plan of a campaign defaults to best-of-3; follow-ups to 1;
    explicit values always win (clamped); junk degrades to 1."""
    from scilink.agents.planning_agents.orchestrator_tools import resolve_n_candidates
    assert resolve_n_candidates(None, {}) == 3
    assert resolve_n_candidates(None, None) == 3
    assert resolve_n_candidates(None, {"current_plan": {"stage": "x"}}) == 1
    assert resolve_n_candidates(1, {}) == 1          # explicit single plan wins
    assert resolve_n_candidates(4, {"current_plan": {}}) == 4
    assert resolve_n_candidates(7, {}) == 4          # clamp
    assert resolve_n_candidates("junk", {}) == 1     # degrade, never crash


def test_all_authors_get_one_of_n_note(tmp_path, no_verify_no_critic):
    """EVERY author (candidate 1 included) is told it writes ONE of N —
    otherwise an objective phrased 'explore several alternatives' makes the
    first author pack all strategies into one plan (observed live). Absent
    entirely at n_candidates=1."""
    model = ScriptedModel([
        plan_json("P1", "H1"),
        plan_json("P2", "H2"),
        judge_json(1, 2),
    ])
    agent = make_agent(tmp_path, model)
    agent.generate_plan("explore several alternative strategies for X",
                        enable_human_feedback=False, n_candidates=2)
    gen1, gen2, _ = model.calls
    assert "BEST-OF-N AUTHORING NOTE" in gen1
    assert "BEST-OF-N AUTHORING NOTE" in gen2

    model2 = ScriptedModel([plan_json("Solo", "H")])
    agent2 = make_agent(tmp_path / "b", model2)
    agent2.generate_plan("obj", enable_human_feedback=False)
    assert "BEST-OF-N AUTHORING NOTE" not in model2.calls[0]


def test_constraints_reach_all_authors_and_judge(tmp_path, no_verify_no_critic):
    """Lab constraints (additional_context) must reach EVERY candidate's
    authoring prompt AND the judge — feasibility is judged against them."""
    model = ScriptedModel([
        plan_json("P1", "H1"),
        plan_json("P2", "H2"),
        judge_json(1, 2),
    ])
    agent = make_agent(tmp_path, model)
    agent.generate_plan(
        "obj", enable_human_feedback=False, n_candidates=2,
        additional_context={"Laboratory Equipment Constraints":
                            "All experiments run on an Opentrons Flex 2 with "
                            "96-well plates."})
    gen1, gen2, judge = model.calls
    assert all("Opentrons Flex 2" in p for p in (gen1, gen2, judge))
    # and on the judge it arrives as evidence, not as authoring instructions
    assert "Additional Context" in judge


# ------------------------------------------------- CLI -> UI parser contract

def _load_ui_parser():
    """Extract _parse_plan_candidate_review from app.py without importing
    streamlit (app.py has top-level st.* calls)."""
    src = Path(__file__).resolve().parents[1] / "scilink" / "ui" / "app.py"
    tree = ast.parse(src.read_text())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef)
              and n.name == "_parse_plan_candidate_review")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(src), "exec"), ns)
    return ns["_parse_plan_candidate_review"]


def test_cli_block_roundtrips_through_ui_parser(capsys):
    candidates = [json.loads(plan_json(f"Plan {i}", f"H{i}")) for i in (1, 2, 3)]
    judge = json.loads(judge_json(2, 3))
    judge = {"scores": judge["scores"], "selected_candidate": 2,
             "reasoning": judge["reasoning"]}
    display_plan_candidates(candidates, judge, selected=2,
                            report_paths=["a.html", "b.html", "c.html"],
                            pick_caveats=["[physics] minor caveat"])
    block = capsys.readouterr().out

    parse = _load_ui_parser()
    prompt = "\n> Selection (ENTER to accept plan candidate 2): "
    parsed = parse(block, prompt)
    assert parsed is not None
    cands, pick = parsed
    assert pick == 2
    assert [c["idx"] for c in cands] == [1, 2, 3]
    assert cands[0]["label"].startswith("Candidate 1 — Plan 1")

    # long names survive un-truncated up to the generous cap (the radio row
    # has full content width; 60-char clipping hid the distinguishing part)
    long_name = "Two-Stage Selective Li Recovery from Permian Produced Water " \
                "via Ion-Sieve Sorption with Staged Elution"
    cands2 = [json.loads(plan_json(long_name, "H1")),
              json.loads(plan_json("Short", "H2"))]
    display_plan_candidates(cands2, {"scores": [], "reasoning": ""}, selected=1)
    block2 = capsys.readouterr().out
    parsed2 = parse(block2, "accept plan candidate 1")
    assert parsed2 is not None
    assert long_name in parsed2[0][0]["label"]

    # disjoint gating: the analysis best-of-N prompt must NOT match this parser
    assert parse(block, "accept candidate 2") is None
    # and a non-candidate buffer must not match either
    assert parse("REQUESTING FEEDBACK\nReview the plan", prompt) is None


def test_candidate_reports_excluded_from_chat_sweep(tmp_path):
    """plan.html reaches the chat message; plan_candidates/*.html do not
    (side artifacts — selection prompt + File Explorer), but are marked
    known so they never leak into a later message."""
    src = Path(__file__).resolve().parents[1] / "scilink" / "ui" / "app.py"
    tree = ast.parse(src.read_text())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef)
              and n.name == "_find_new_html_reports")
    session = tmp_path / "planning_session_x"
    (session / "plan_candidates").mkdir(parents=True)
    (session / "plan.html").write_text("<html>winner</html>")
    for i in (1, 2, 3):
        (session / "plan_candidates" / f"candidate_{i}.html").write_text("<html>c</html>")

    known = set()
    st_stub = SimpleNamespace(session_state=SimpleNamespace(
        session_dir=str(session), known_images=known))
    ns = {"st": st_stub, "Path": Path}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(src), "exec"), ns)
    new = ns["_find_new_html_reports"]()

    assert [Path(p).name for p in new] == ["plan.html"]
    assert len(known) == 4  # all four marked seen
    assert ns["_find_new_html_reports"]() == []  # nothing leaks later


def test_stale_buffer_takes_latest_block(capsys):
    """Two successive reviews in one session buffer -> parser reads the last."""
    c2 = [json.loads(plan_json(f"Old {i}", f"HO{i}")) for i in (1, 2)]
    display_plan_candidates(c2, {"scores": [], "reasoning": ""}, selected=1)
    c3 = [json.loads(plan_json(f"New {i}", f"HN{i}")) for i in (1, 2, 3)]
    display_plan_candidates(c3, {"scores": [], "reasoning": ""}, selected=3)
    buffer = capsys.readouterr().out

    parse = _load_ui_parser()
    cands, pick = parse(buffer, "accept plan candidate 3")
    assert len(cands) == 3 and pick == 3
    assert cands[0]["label"] == "Candidate 1 — New 1"


# --------------------------------------------------- constraint coverage

def _capture_ctx(monkeypatch):
    """Capture the additional_context that reaches the RAG engine."""
    from scilink.agents.planning_agents import planning_rag as pr
    seen = {}

    def fake_run_rag(**kw):
        seen["additional_context"] = kw.get("additional_context")
        return {"plan": "x"}

    monkeypatch.setattr(pr, "run_rag", fake_run_rag)
    return pr, seen


def test_coverage_note_injected_only_with_literature_and_constraints(monkeypatch):
    """Literature degrades constraint COVERAGE (omissions, not violations), so
    the constraint->step mapping demand rides along whenever both are present —
    and stays out of the prompt otherwise, which is the condition it was
    measured in."""
    pr, seen = _capture_ctx(monkeypatch)
    cases = {
        ("constraints", "lit"): True,
        ("constraints", None): False,
        (None, "lit"): False,
        (None, None): False,
    }
    for (add, ext), expected in cases.items():
        seen.clear()
        pr.perform_science_rag(
            objective="o", instructions="i", task_name="t", kb_docs=None,
            model=None, generation_config=None,
            additional_context=add, external_context=ext)
        got = pr.CONSTRAINT_COVERAGE_NOTE in (seen["additional_context"] or "")
        assert got is expected, f"add={add!r} ext={ext!r}: coverage note {got}"


def test_coverage_note_preserves_the_original_context(monkeypatch):
    """The note is additive — the caller's constraints must survive verbatim."""
    pr, seen = _capture_ctx(monkeypatch)
    pr.perform_science_rag(
        objective="o", instructions="i", task_name="t", kb_docs=None,
        model=None, generation_config=None,
        additional_context="max 2 mL/well, aqueous only", external_context="lit")
    assert "max 2 mL/well, aqueous only" in seen["additional_context"]


def test_candidate_cards_are_visually_separated(capsys):
    """Each candidate's body runs to several hundred words, so boundaries
    must be findable at a glance: a banded 'CANDIDATE i of n' rule, the pick
    starred in that band, and wrapped/indented field bodies. The '── Candidate
    i: name ──' marker line keeps its exact grammar (UI parser contract)."""
    long_text = ("word " * 400).strip()
    cands = [json.loads(plan_json(f"Plan {i}", long_text)) for i in (1, 2, 3)]
    for c in cands:
        c["proposed_experiments"][0]["justification"] = long_text
    judge = json.loads(judge_json(2, 3))
    display_plan_candidates(cands, judge, selected=2)
    out = capsys.readouterr().out

    assert out.count("━" * 80) == 6  # a rule above and below each banner
    for i in (1, 2, 3):
        assert f"  CANDIDATE {i} of 3" in out
    assert "CANDIDATE 2 of 3   ★ JUDGE PICK" in out
    assert "CANDIDATE 1 of 3   ★" not in out
    # labels sit on their own line; bodies are wrapped and indented under them
    assert "\n🎯 Hypothesis:\n   word" in out
    assert "\n💡 Justification:\n   word" in out
    body_lines = [ln for ln in out.splitlines() if ln.startswith("   word")]
    assert body_lines and max(len(ln) for ln in body_lines) <= 80

    # pick is still readable from the BLOCK alone (prompt carries no index)
    parse = _load_ui_parser()
    cands_p, pick = parse(out, "accept plan candidate")
    assert pick == 2
    assert [c["idx"] for c in cands_p] == [1, 2, 3]


# ------------------------------------------------- plan summary presentation

def _summary(plan, ideation, capsys):
    from scilink.agents.planning_agents.user_interface import display_plan_summary
    display_plan_summary(plan, ideation=ideation)
    return capsys.readouterr().out


def _portfolio_plan():
    return {"proposed_experiments": [{
        "experiment_name": "Use-case portfolio",
        "hypothesis": "PORTFOLIO THESIS: capture structure and function together.",
        "experimental_steps": [
            "==================== DOMAIN 1: PRECISION SYNTHESIS ====================",
            "",
            "PS-1 (TIER-1) — catch the metastable polymorph before it equilibrates.",
            "",
            "PS-2 (TIER-1) — steer the redox state to a target.",
        ],
        "required_equipment": ["XRD", "XAS"],
        "expected_outcome": "SUPPORT if state and function co-emerge.",
        "justification": "Grounded in the retrieved context.",
    }]}


def test_blank_and_banner_entries_are_not_numbered_steps(capsys):
    """Blank entries are spacing and '==== ... ====' entries are section
    dividers — numbering them produced empty '2.'/'4.' rows in live output."""
    out = _summary(_portfolio_plan(), False, capsys)
    assert "▸ DOMAIN 1: PRECISION SYNTHESIS" in out
    assert " 1. PS-1" in out and " 2. PS-2" in out  # numbering skips the gaps
    assert " 3. " not in out
    assert not re.search(r"^ \d+\.\s*$", out, re.M)  # no empty numbered rows


def test_ideation_view_drops_lab_protocol_vocabulary(capsys):
    """An ideation portfolio is not a bench protocol: no imposed step
    numbering (the entries carry their own PS-/BA-/BRIDGE- labels) and no
    'EXPERIMENT n' ordinal on a single portfolio entry."""
    out = _summary(_portfolio_plan(), True, capsys)
    assert "PROPOSED RESEARCH DIRECTIONS" in out
    assert "💡 RESEARCH DIRECTION: Use-case portfolio" in out
    assert "RESEARCH DIRECTION 1" not in out  # single entry -> no ordinal
    assert "--- 🧭 Proposed Program ---" in out
    assert "--- 🛠️  Key Capabilities ---" in out
    assert "EXPERIMENT" not in out and "Experimental Steps" not in out
    assert "  • PS-1" in out and "  • PS-2" in out
    assert not re.search(r"^ \d+\. PS-", out, re.M)  # author labels, not 1..N
    assert "▸ DOMAIN 1: PRECISION SYNTHESIS" in out  # dividers still divide


def test_lab_view_unchanged_in_shape(capsys):
    out = _summary(_portfolio_plan(), False, capsys)
    assert "PROPOSED EXPERIMENTAL PLAN" in out
    assert "--- 🧪 Experimental Steps ---" in out
    assert "--- 🛠️  Required Equipment ---" in out
    assert "RESEARCH DIRECTION" not in out


def test_multiple_entries_keep_their_ordinals(capsys):
    plan = _portfolio_plan()
    plan["proposed_experiments"].append(
        dict(plan["proposed_experiments"][0], experiment_name="Second"))
    out_lab = _summary(plan, False, capsys)
    assert "🔬 EXPERIMENT 1:" in out_lab and "🔬 EXPERIMENT 2:" in out_lab
    out_id = _summary(plan, True, capsys)
    assert "💡 RESEARCH DIRECTION 1:" in out_id
    assert "💡 RESEARCH DIRECTION 2:" in out_id


# ------------------------------------------------- ideation authoring guards

def test_ideation_run_still_resolves_a_fallback_instruction_set(monkeypatch):
    """The ideation override is CONCATENATED onto the canonical instruction
    block, so the old identity test (`instructions == HYPOTHESIS_...`) left
    ideation runs with fallback_instructions=None: an 'Insufficient context'
    author response then aborted the run instead of regenerating from
    general knowledge (rag_engine returns the error verbatim when no
    fallback set is available)."""
    from scilink.agents.planning_agents import planning_rag as pr

    seen = {}
    monkeypatch.setattr(pr, "run_rag",
                        lambda **kw: (seen.update(kw), {"proposed_experiments": []})[1])
    pr.perform_science_rag(
        objective="obj",
        instructions=pr.HYPOTHESIS_GENERATION_INSTRUCTIONS + pr.IDEATION_AUTHOR_OVERRIDE,
        task_name="Candidate Plan 1", kb_docs=None, model=None,
        generation_config=None)
    assert seen["fallback_instructions"] == pr.HYPOTHESIS_GENERATION_INSTRUCTIONS_FALLBACK

    # lab (unconcatenated) and TEA keep resolving as before
    seen.clear()
    pr.perform_science_rag(objective="o", instructions=pr.HYPOTHESIS_GENERATION_INSTRUCTIONS,
                           task_name="t", kb_docs=None, model=None, generation_config=None)
    assert seen["fallback_instructions"] == pr.HYPOTHESIS_GENERATION_INSTRUCTIONS_FALLBACK
    seen.clear()
    pr.perform_science_rag(objective="o", instructions=pr.TEA_INSTRUCTIONS,
                           task_name="t", kb_docs=None, model=None, generation_config=None)
    assert seen["fallback_instructions"] == pr.TEA_INSTRUCTIONS_FALLBACK

    # an explicit argument still wins over the prefix inference
    seen.clear()
    pr.perform_science_rag(objective="o", instructions=pr.HYPOTHESIS_GENERATION_INSTRUCTIONS,
                           task_name="t", kb_docs=None, model=None, generation_config=None,
                           fallback_instructions="CUSTOM")
    assert seen["fallback_instructions"] == "CUSTOM"


def test_ideation_override_bars_invented_optimization_params():
    """optimization_params is the schema's only optional field and describes
    knobs of an executable campaign; an ideation portfolio was filling it
    with defensible-looking ranges for a system nobody had chosen."""
    from scilink.agents.planning_agents.instruct import (
        IDEATION_AUTHOR_OVERRIDE, IDEATION_OUTPUT_RULES)
    assert "optimization_params" in IDEATION_OUTPUT_RULES
    assert "omit the\nfield entirely" in IDEATION_OUTPUT_RULES
    # shape rules live OUTSIDE the grounding override, which the fallback
    # tier drops on purpose
    assert "optimization_params" not in IDEATION_AUTHOR_OVERRIDE
    assert "concepts" not in IDEATION_AUTHOR_OVERRIDE


def test_lab_and_tea_dispatch_identical_to_pre_fix_oracle(monkeypatch):
    """No-op proof for the paths that were already correct: for every
    instruction string the codebase actually passes, the new prefix-based
    fallback resolution must agree with the OLD identity logic — except for
    the ideation case, which is exactly the bug (old: None)."""
    from scilink.agents.planning_agents import planning_rag as pr

    def old_identity_logic(instructions):
        if instructions == pr.HYPOTHESIS_GENERATION_INSTRUCTIONS:
            return pr.HYPOTHESIS_GENERATION_INSTRUCTIONS_FALLBACK
        elif instructions == pr.TEA_INSTRUCTIONS:
            return pr.TEA_INSTRUCTIONS_FALLBACK
        return None

    def new_resolution(instructions):
        seen = {}
        monkeypatch.setattr(pr, "run_rag",
                            lambda **kw: (seen.update(kw), {})[1])
        pr.perform_science_rag(objective="o", instructions=instructions,
                               task_name="t", kb_docs=None, model=None,
                               generation_config=None)
        return seen["fallback_instructions"]

    lab_like = [pr.HYPOTHESIS_GENERATION_INSTRUCTIONS, pr.TEA_INSTRUCTIONS]
    for instr in lab_like:
        assert new_resolution(instr) == old_identity_logic(instr), \
            "lab/TEA dispatch changed — this fix must be a no-op for them"

    ideation = pr.HYPOTHESIS_GENERATION_INSTRUCTIONS + pr.IDEATION_AUTHOR_OVERRIDE
    assert old_identity_logic(ideation) is None          # the bug
    assert new_resolution(ideation) is not None          # the fix


def test_ideation_prompt_text_never_reaches_a_lab_run(monkeypatch):
    """The optimization_params ban and the grounding latitude ride
    IDEATION_AUTHOR_OVERRIDE, which is concatenated only for the ideation
    profile — a lab author's instruction block must be byte-identical to the
    canonical constant."""
    from scilink.agents.planning_agents import planning_rag as pr

    captured = {}

    def fake_rag(**kw):
        captured[kw["task_name"]] = kw["instructions"]
        return ({"proposed_experiments": [{"hypothesis": "h"}]},
                {"retrieved_context": "", "primary_data": ""})

    monkeypatch.setattr(pr, "perform_science_rag", fake_rag)
    monkeypatch.setattr(pr, "run_rag", lambda **kw: {"error": "stop"})

    pr.generate_plan_candidates(objective="o", kb_docs=None, model=None,
                                generation_config=None, n_candidates=2,
                                selection_profile="lab")
    lab_instr = captured["Candidate Plan 1"]
    assert lab_instr == pr.HYPOTHESIS_GENERATION_INSTRUCTIONS  # byte-identical
    assert "IDEATION MODE" not in lab_instr
    assert "optimization_params describes knobs" not in lab_instr

    captured.clear()
    pr.generate_plan_candidates(objective="o", kb_docs=None, model=None,
                                generation_config=None, n_candidates=2,
                                selection_profile="ideation")
    id_instr = captured["Candidate Plan 1"]
    assert id_instr.startswith(pr.HYPOTHESIS_GENERATION_INSTRUCTIONS)
    assert "IDEATION MODE" in id_instr


def test_lab_schema_block_still_offers_optimization_params():
    """The ban is ideation-only prose; the canonical schema (and its
    fallback twin) must still describe the field for lab campaigns, which
    is where run_optimization gets its scientific bounds."""
    from scilink.agents.planning_agents.instruct import (
        HYPOTHESIS_GENERATION_INSTRUCTIONS,
        HYPOTHESIS_GENERATION_INSTRUCTIONS_FALLBACK,
    )
    for block in (HYPOTHESIS_GENERATION_INSTRUCTIONS,
                  HYPOTHESIS_GENERATION_INSTRUCTIONS_FALLBACK):
        assert "optimization_params" in block
        assert "min_value" in block and "max_value" in block
        assert "omit the field" not in block


# ------------------------------------------------- concepts portfolio (P1-P3)

CONCEPTS = [
    {"id": "PS-1", "tier": "1", "title": "Catch the metastable polymorph",
     "hypothesis": "It nucleates first and relaxes within minutes.",
     "rationale": "Only visible operando.", "novelty": "Steered, not post-hoc.",
     "details": ["probes: XRD, PDF", "trigger: quench at max fraction"]},
    {"id": "BA-1", "tier": "1", "title": "Living biohybrid interphase",
     "hypothesis": "A charge-transfer interphase co-emerges with product.",
     "rationale": "Resolves direct-vs-H2.", "novelty": "Dual pumping."},
    {"id": "BR-1", "title": "Shared belief-state control",
     "rationale": "One controller spans both domains."},
]


def _concepts_plan(concepts=None, steps=None):
    exp = {"experiment_name": "Portfolio", "hypothesis": "Thesis.",
           "expected_outcome": "Support if co-emergence.",
           "justification": "Grounded.",
           "experimental_steps": steps if steps is not None else ["S1. Shared prep"]}
    if concepts is not None:
        exp["concepts"] = concepts
    return {"proposed_experiments": [exp]}


def test_concepts_render_as_directions_not_steps(capsys):
    from scilink.agents.planning_agents.user_interface import display_plan_summary
    display_plan_summary(_concepts_plan(CONCEPTS), ideation=True)
    out = capsys.readouterr().out
    assert "--- 🧠 Research Directions (3) ---" in out
    assert "▸ PS-1: Catch the metastable polymorph  ·  tier 1" in out
    assert "▸ BR-1: Shared belief-state control" in out       # partial entry ok
    assert "probes: XRD, PDF" in out                          # details kept
    # steps demoted to shared protocol, not presented as the whole plan
    assert "--- 🧭 Shared Protocol ---" in out
    assert "--- 🧭 Proposed Program ---" not in out


def test_plan_without_concepts_renders_exactly_as_before(capsys):
    from scilink.agents.planning_agents.user_interface import display_plan_summary
    display_plan_summary(_concepts_plan(), ideation=True)
    out = capsys.readouterr().out
    assert "Research Directions" not in out
    assert "--- 🧭 Proposed Program ---" in out   # unchanged path
    display_plan_summary(_concepts_plan(), ideation=False)
    lab = capsys.readouterr().out
    assert "--- 🧪 Experimental Steps ---" in lab and "Research Directions" not in lab


def test_unknown_concept_keys_are_not_silently_dropped(capsys):
    """Authors add elements the objective asked for; dropping them loses
    the answer."""
    from scilink.agents.planning_agents.user_interface import display_plan_summary
    display_plan_summary(_concepts_plan(
        [{"title": "T", "operando_necessity": "state vanishes at rest"}]),
        ideation=True)
    out = capsys.readouterr().out
    assert "operando_necessity" in out and "state vanishes at rest" in out


def test_reasoning_passes_see_every_direction():
    """Conformance/critic summarized a 13-concept portfolio as ONE experiment,
    so a 'propose 4-6 directions' objective read as unsatisfied — the failure
    that pushed portfolios into experimental_steps."""
    from scilink.agents.planning_agents.planning_rag import summarize_experiment
    s = summarize_experiment(_concepts_plan(CONCEPTS)["proposed_experiments"][0], 1)
    assert "Research directions in this plan (3 total)" in s
    for label in ("Catch the metastable polymorph", "Living biohybrid interphase",
                  "Shared belief-state control"):
        assert label in s
    assert "[PS-1 1]" in s
    # unchanged shape when there is no portfolio
    plain = summarize_experiment(_concepts_plan()["proposed_experiments"][0], 1)
    assert "Research directions" not in plain
    assert plain.startswith("Experiment 1: Portfolio")


def test_refinement_prompt_preserves_the_portfolio():
    """refine_plan_with_feedback is the only plan-mutation path (8 call
    sites); an unnamed field is what gets dropped on rewrite."""
    import inspect
    from scilink.agents.planning_agents import planning_rag as pr
    src = inspect.getsource(pr.refine_plan_with_feedback)
    assert '"concepts"' in src
    assert "Never collapse it into prose" in src


def test_ideation_type_stamp_survives_rewrite_passes(tmp_path, no_verify_no_critic):
    model = ScriptedModel([plan_json("P1", "H1"), plan_json("P2", "H2"),
                           judge_json(1, 2)])
    agent = make_agent(tmp_path, model)
    res = agent.generate_plan("ideate directions on X", enable_human_feedback=False,
                              n_candidates=2, selection_profile="ideation")
    assert res["type"] == "ideation"
    assert agent.state["plan_kind"] == "ideation"
    assert agent._is_ideation_campaign() is True
    # a pass that re-emits the plan without `type` gets it back on snapshot
    stripped = {k: v for k, v in res.items() if k != "type"}
    assert agent._stamp_campaign(stripped)["type"] == "ideation"
    # ... and TEA's own kind is never overwritten
    tea = agent._stamp_campaign({"type": "technoeconomic_analysis"})
    assert tea["type"] == "technoeconomic_analysis"


def test_lab_run_is_not_stamped_ideation(tmp_path, no_verify_no_critic):
    model = ScriptedModel([plan_json("P1", "H1"), plan_json("P2", "H2"),
                           judge_json(1, 2)])
    agent = make_agent(tmp_path, model)
    res = agent.generate_plan("obj", enable_human_feedback=False,
                              n_candidates=2, selection_profile="lab")
    assert "type" not in res
    assert agent._is_ideation_campaign() is False


def test_new_campaign_does_not_inherit_ideation_kind(tmp_path, no_verify_no_critic):
    model = ScriptedModel([plan_json("I1", "H1"), plan_json("I2", "H2"),
                           judge_json(1, 2), plan_json("LabPlan", "H3")])
    agent = make_agent(tmp_path, model)
    agent.generate_plan("ideate research directions on perovskite stability",
                        enable_human_feedback=False, n_candidates=2,
                        selection_profile="ideation")
    assert agent._is_ideation_campaign() is True
    lab = agent.generate_plan(
        "optimize suzuki coupling yield on the flow platform",
        enable_human_feedback=False, n_candidates=1, selection_profile="lab")
    assert agent.state["campaign_id"] == 2
    assert "type" not in lab
    assert agent._is_ideation_campaign() is False


def test_white_paper_leads_with_the_portfolio():
    """Concepts sit deep in the plan JSON where the 20k truncation can cut
    them; the paper must be written from all N directions."""
    agent = PlanningAgent.__new__(PlanningAgent)
    agent.state = {"objective": "o", "campaign_id": 1,
                   "current_plan": dict(_concepts_plan(CONCEPTS)["proposed_experiments"][0]
                                        and _concepts_plan(CONCEPTS), campaign_id=1)}
    agent.state["plan_history"] = [agent.state["current_plan"]]
    agent.model = ScriptedModel(["paper"])
    agent.generation_config = None
    agent.generate_white_paper()
    prompt = agent.model.calls[0]
    assert "RESEARCH DIRECTIONS IN THE SELECTED PLAN (3)" in prompt
    assert prompt.index("RESEARCH DIRECTIONS") < prompt.index("SELECTED Campaign Plan")
    for label in ("PS-1", "BA-1", "BR-1"):
        assert label in prompt


def test_dossier_renders_the_flagship_as_shipped_not_as_authored(tmp_path):
    """Conformance/critic rewrite the selected plan AFTER the candidate set
    is frozen; rendering the stored copy shipped a dossier describing a plan
    the user never got (live: dossier carried none of the corrected
    portfolio's directions while the white paper carried them all)."""
    from scilink.agents.planning_agents.orchestrator_tools import OrchestratorTools

    def exp(name, concepts=None):
        e = {"experiment_name": name, "hypothesis": f"H-{name}",
             "justification": "J"}
        if concepts:
            e["concepts"] = concepts
        return {"proposed_experiments": [e]}

    state = {"objective": "brainstorm",
             "current_plan": exp("Corrected Portfolio", concepts=CONCEPTS),
             "plan_candidates": {
                 "candidates": [exp("Authored A"), exp("Authored B")],
                 "selected_index": 2,
                 "judge": {"scores": [], "reasoning": "r"}}}
    tools = OrchestratorTools.__new__(OrchestratorTools)
    tools.orch = SimpleNamespace(planner=SimpleNamespace(state=state),
                                 base_dir=tmp_path, _active_output_subdir=None)
    tools._output_dir = lambda: tmp_path
    text = Path(tools._write_ideation_report()).read_text()

    assert "## Candidate 2 — SELECTED (flagship): Corrected Portfolio" in text
    assert "Shown as shipped" in text
    assert "### Research directions (3)" in text
    assert "**PS-1. Catch the metastable polymorph** *(tier 1)*" in text
    assert "## Candidate 1: Authored A" in text          # runner-up as authored
    assert text.count("Shown as shipped") == 1

    # unrevised flagship -> no note, no behavior change
    state["current_plan"] = state["plan_candidates"]["candidates"][1]
    text2 = Path(tools._write_ideation_report()).read_text()
    assert "Shown as shipped" not in text2


def test_html_report_renders_the_portfolio(tmp_path):
    """The ideation run itself skips plan.html, but every later refine
    regenerates it — without concepts rendering, the HTML showed only the
    shared protocol and silently dropped the research directions."""
    from scilink.agents.planning_agents.html_generator import HTMLReportGenerator

    plan = _concepts_plan(CONCEPTS)
    plan["stage"] = "Reasoning Draft"
    state = {"objective": "o", "plan_history": [plan], "current_plan": plan,
             "experimental_results": [], "action_history": []}
    out = tmp_path / "plan.html"
    HTMLReportGenerator(state).generate(str(out))
    h = out.read_text()

    assert "Research Directions (3)" in h
    for cid in ("PS-1", "BA-1", "BR-1"):
        assert cid in h
    assert "Catch the metastable polymorph" in h
    assert "probes: XRD, PDF" in h          # details survive
    assert "Shared Protocol" in h           # steps demoted, not dropped
    assert "&lt;" not in "Catch the metastable polymorph"   # sanity


def test_html_report_unchanged_without_concepts(tmp_path):
    from scilink.agents.planning_agents.html_generator import HTMLReportGenerator
    plan = _concepts_plan()
    plan["stage"] = "Science Draft"
    state = {"objective": "o", "plan_history": [plan], "current_plan": plan,
             "experimental_results": [], "action_history": []}
    out = tmp_path / "plan.html"
    HTMLReportGenerator(state).generate(str(out))
    h = out.read_text()
    assert "Research Directions" not in h
    assert "🧪 Steps" in h and "Shared Protocol" not in h


def test_html_escapes_hostile_concept_content(tmp_path):
    """Concept text is model-authored; ids/titles must not inject markup."""
    from scilink.agents.planning_agents.html_generator import HTMLReportGenerator
    plan = _concepts_plan([{"id": "<script>x</script>", "title": "A & B <b>",
                            "tier": "<i>1</i>", "rationale": "r"}])
    plan["stage"] = "Science Draft"
    state = {"objective": "o", "plan_history": [plan], "current_plan": plan,
             "experimental_results": [], "action_history": []}
    out = tmp_path / "plan.html"
    HTMLReportGenerator(state).generate(str(out))
    h = out.read_text()
    assert "<script>x</script>" not in h
    assert "&lt;script&gt;" in h
    assert "A &amp; B" in h


def test_output_rules_ride_both_tiers(monkeypatch):
    """Grounding latitude is tier-dependent (the fallback set already grants
    it, so the override is dropped there by design) — but the OUTPUT shape
    rules must survive into the fallback set, or a fallback ideation run
    goes back to cramming its portfolio into experimental_steps."""
    from scilink.agents.planning_agents import planning_rag as pr

    seen = {}

    def fake_psr(**kw):
        seen.update(kw)
        return ({"proposed_experiments": [{"hypothesis": "h"}]},
                {"retrieved_context": "", "primary_data": ""})

    monkeypatch.setattr(pr, "perform_science_rag", fake_psr)
    monkeypatch.setattr(pr, "run_rag", lambda **kw: {"error": "stop"})

    pr.generate_plan_candidates(objective="o", kb_docs=None, model=None,
                                generation_config=None, n_candidates=2,
                                selection_profile="ideation")
    author, fb = seen["instructions"], seen["fallback_instructions"]
    assert "PORTFOLIO OUTPUT" in author and "GROUNDING LATITUDE" in author
    assert "PORTFOLIO OUTPUT" in fb            # shape survives the tier swap
    assert "GROUNDING LATITUDE" not in fb      # latitude deliberately does not
    assert fb.startswith(pr.HYPOTHESIS_GENERATION_INSTRUCTIONS_FALLBACK)

    # lab is untouched in both tiers
    seen.clear()
    pr.generate_plan_candidates(objective="o", kb_docs=None, model=None,
                                generation_config=None, n_candidates=2,
                                selection_profile="lab")
    assert seen["instructions"] == pr.HYPOTHESIS_GENERATION_INSTRUCTIONS
    assert seen["fallback_instructions"] == pr.HYPOTHESIS_GENERATION_INSTRUCTIONS_FALLBACK
