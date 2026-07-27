"""Ideation campaigns keep their kind across the whole session.

Replays the cdoc session (meta_session_20260726_141723): nine delegations,
an ideation campaign established by a best-of-N call, then consolidation
follow-ups that asked for a single plan. Those follow-ups scored as LAB runs
— `selection_profile` weights the best-of-N judge and is a documented no-op
at n_candidates=1 — so each one wrote a protocol report, skipped the
ideation dossier, and skipped the white paper, while the console it printed
still used ideation vocabulary. The display asked the campaign; the
artifacts asked the judge knob.
"""

import json
from types import SimpleNamespace

import pytest

from scilink.agents.planning_agents.orchestrator_tools import OrchestratorTools
from scilink.agents.planning_agents.planning_agent import PlanningAgent


def _agent(state):
    a = PlanningAgent.__new__(PlanningAgent)
    a.state = state
    return a


def _tools(tmp_path, state, generated):
    """OrchestratorTools wired to a recording HTML generator."""
    tools = OrchestratorTools.__new__(OrchestratorTools)
    tools.orch = SimpleNamespace(base_dir=tmp_path, _active_output_subdir=None,
                                 planner=_agent(state))
    tools._output_dir = lambda: tmp_path
    return tools


# ── the kind predicate ───────────────────────────────────────────────

@pytest.mark.parametrize("state,expected", [
    ({"current_plan": {"type": "ideation"}}, True),
    ({"plan_kind": "ideation", "current_plan": {}}, True),
    # inherited by a follow-up plan via _stamp_campaign — the live case
    ({"plan_kind": "ideation", "current_plan": {"type": "ideation"}}, True),
    # the runnable protocol for a direction the ideation chose
    ({"plan_kind": "ideation", "current_plan": {"type": "lab"}}, False),
    # TEA still defers to the campaign, unchanged
    ({"plan_kind": "ideation",
      "current_plan": {"type": "technoeconomic_analysis"}}, True),
    ({"current_plan": {}}, False),
])
def test_campaign_kind(state, expected):
    assert _agent(state)._is_ideation_campaign() is expected


# ── the report decision, now shared by all five sites ────────────────

def test_no_protocol_report_for_a_dossier(tmp_path):
    state = {"plan_kind": "ideation", "current_plan": {"type": "ideation"}}
    tools = _tools(tmp_path, state, [])
    assert tools._emit_plan_report() is None
    assert not (tmp_path / "plan.html").exists()


def test_lab_campaigns_still_get_their_report(tmp_path):
    state = {"current_plan": {"type": "lab", "proposed_experiments": []}}
    tools = _tools(tmp_path, state, [])
    out = tools._emit_plan_report()
    assert out is not None and out.exists()

    from scilink.agents.planning_agents.user_interface import load_deliverables
    titles = [e["title"] for e in load_deliverables(tmp_path)]
    assert "Experimental plan (report)" in titles


def test_refinement_cannot_resurrect_the_report(tmp_path):
    """The bug's second half: suppression lived on the initial-plan path
    only, so every later refine re-rendered the protocol view."""
    state = {"plan_kind": "ideation", "current_plan": {"type": "ideation"}}
    tools = _tools(tmp_path, state, [])
    for name in ("plan.html", "plan.html", "plan_refined.html"):
        assert tools._emit_plan_report(name) is None
    assert list(tmp_path.glob("*.html")) == []


def test_every_report_site_routes_through_the_helper():
    """Five call sites had the render inlined; a sixth copy would silently
    reintroduce the leak."""
    from pathlib import Path
    src = Path("scilink/agents/planning_agents/orchestrator_tools.py").read_text()
    assert src.count("generator.generate(str(html_path))") == 0
    assert src.count("self._emit_plan_report(") == 5
    # and the render itself exists exactly once
    assert src.count("HTMLReportGenerator(self.orch.planner.state).generate") == 1


# ── the artifact branch ──────────────────────────────────────────────

def _ideation_run(plan, selection_profile, n_cand):
    """The predicate as written in the initial-plan path."""
    return (plan.get("type") == "ideation"
            or (selection_profile == "ideation" and n_cand > 1))


@pytest.mark.parametrize("plan,profile,n,expected", [
    # delegation 01: the best-of-N call that establishes the campaign
    ({"type": "ideation"}, "ideation", 3, True),
    # delegations 04/05 live: consolidation, single plan, profile omitted.
    # Was False -> protocol report, no dossier, no white paper.
    ({"type": "ideation"}, None, 1, True),
    # a bench plan inside the same campaign opts out
    ({"type": "lab"}, "lab", 1, False),
    # an ordinary lab campaign is untouched
    ({}, None, 1, False),
    ({}, "lab", 3, False),
    # first call of an ideation campaign, before the stamp lands
    ({}, "ideation", 3, True),
])
def test_artifact_branch_follows_the_plan_not_the_judge(plan, profile, n,
                                                        expected):
    assert _ideation_run(plan, profile, n) is expected


def test_the_recorded_session_would_now_behave(tmp_path):
    """Replays the real plan.json shapes from the cdoc session.

    The build-roadmap / footprint / merge delegations wrote markdown
    documents and never called the plan tool at all — correct, since a
    roadmap is not experimental design — so they must stay absent here.
    """
    from pathlib import Path
    rec = Path("meta_session_20260726_141723/planning/delegations")
    if not rec.exists():
        pytest.skip("recorded session not present")

    seen = {}
    for d in sorted(rec.iterdir()):
        pj = d / "plan.json"
        if not pj.exists():
            continue
        plan = json.loads(pj.read_text())
        # every consolidation call in that session used the defaults
        seen[d.name] = _ideation_run(plan, None, 1)

    assert seen, "expected recorded plans"
    assert all(seen.values()), f"still scored as lab: {seen}"
    # the roadmap/footprint/merge delegations produced no plan, and must
    # not start producing one
    assert not any("roadmap" in n or "staged" in n or "merge" in n
                   for n in seen), seen


# ── the three artifacts do NOT share one condition ───────────────────

def _artifacts(plan, profile, n_cand):
    """What the initial-plan path decides, as written."""
    ideation = _ideation_run(plan, profile, n_cand)
    return {"plan_html": not ideation,
            "white_paper": ideation,
            "dossier": ideation and n_cand > 1}


def test_dossier_needs_a_candidate_set_from_THIS_call():
    """`plan_candidates` survives across delegations, so a single-plan
    follow-up in an ideation campaign would otherwise render its own
    flagship beside an EARLIER question's runner-ups."""
    follow_up = _artifacts({"type": "ideation"}, None, 1)
    assert follow_up == {"plan_html": False, "white_paper": True,
                         "dossier": False}

    best_of_n = _artifacts({"type": "ideation"}, "ideation", 3)
    assert best_of_n["dossier"] is True


def test_the_dossier_renderer_refuses_without_candidates(tmp_path):
    """It raises rather than inventing one — the gate above is what keeps
    that from becoming a logged warning on every follow-up."""
    tools = _tools(tmp_path, {"plan_candidates": {}}, [])
    with pytest.raises(ValueError, match="No candidate set"):
        tools._write_ideation_report()


def test_the_gate_is_wired_not_just_computed():
    from pathlib import Path
    src = Path("scilink/agents/planning_agents/orchestrator_tools.py").read_text()
    assert "if _ideation_run and n_cand > 1:" in src


# ── the portfolio CONTRACT follows the campaign too ──────────────────

def test_single_plan_ideation_gets_the_portfolio_rules():
    """The `concepts` contract was injected only on best-of-N calls, so a
    single-plan follow-up authored without it. Live: 01/03/04 emitted 12/4/5
    concepts, delegation 05 emitted none and used 56 steps as sections."""
    from pathlib import Path
    src = Path("scilink/agents/planning_agents/planning_agent.py").read_text()
    assert "_ideation_out = (selection_profile == \"ideation\"" in src
    assert "or self._is_ideation_campaign())" in src
    # both tiers, or a fallback run reverts to cramming
    assert src.count("IDEATION_OUTPUT_RULES if _ideation_out else \"\"") == 1
    assert "HYPOTHESIS_GENERATION_INSTRUCTIONS_FALLBACK\n" \
           "                    + IDEATION_OUTPUT_RULES) if _ideation_out" in src


# ── one card, both surfaces ──────────────────────────────────────────

def _render(fn, *a, **kw):
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*a, **kw)
    return buf.getvalue()


EXP = {"experiment_name": "Rare-site localization",
       "hypothesis": "Active sites are a reactive minority.",
       "expected_outcome": "A localized map of active sites.",
       "justification": "Ensemble averages hide the minority.",
       "experimental_steps": [f"=== SECTION {i} — text ===" for i in range(1, 57)],
       "required_equipment": ["SECCM"], "source_documents": []}


def test_both_surfaces_render_the_same_card():
    from scilink.agents.planning_agents import user_interface as ui
    card = _render(ui._print_direction_fields, EXP)
    cands = _render(ui.display_plan_candidates, [{"proposed_experiments": [EXP]}],
                    {"scores": [], "reasoning": ""}, 1)
    for line in ("🎯 Hypothesis:", "📈 Expected outcome:", "💡 Justification:"):
        assert line in card and line in cands


def test_ideation_review_shows_the_card_not_the_document():
    from scilink.agents.planning_agents import user_interface as ui
    out = _render(ui.display_plan_summary, {"proposed_experiments": [EXP]},
                  ideation=True, report_path="/tmp/plan_preview.html")

    assert "🎯 Hypothesis:" in out and "📈 Expected outcomes:" in out
    assert "💡 Rationale:" in out
    # the 56 sections do not page past the reviewer
    assert "SECTION 20" not in out
    assert "44 more" in out and "report above" in out
    # and the fields are not printed twice
    assert out.count("Expected outcome") == 1


def test_lab_review_still_prints_every_step():
    """The protocol IS the lab deliverable — never truncated."""
    from scilink.agents.planning_agents import user_interface as ui
    steps = [f"Step {i}" for i in range(1, 57)]
    out = _render(ui.display_plan_summary,
                  {"proposed_experiments": [dict(EXP, experimental_steps=steps)]},
                  ideation=False)
    assert "Step 56" in out and "more" not in out.split("Step 56")[-1]
    assert "--- 📈 Expected Outcome ---" in out


def test_a_portfolio_with_concepts_is_untouched():
    """When `concepts` arrives the portfolio renderer runs and the step list
    is shared protocol — no truncation logic involved."""
    from scilink.agents.planning_agents import user_interface as ui
    exp = dict(EXP, concepts=[{"id": "RS-1", "title": "One"},
                              {"id": "RS-2", "title": "Two"}],
               experimental_steps=["shared step"])
    out = _render(ui.display_plan_summary, {"proposed_experiments": [exp]},
                  ideation=True)
    assert "Research Directions (2)" in out and "RS-2" in out
    assert "Shared Protocol" in out and "shared step" in out
