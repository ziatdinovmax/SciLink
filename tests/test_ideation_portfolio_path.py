"""The portfolio authoring path — ideation off generate_initial_plan.

`generate_initial_plan` designs lab experiments. Ideation rode it, so a
portfolio was authored into one `proposed_experiments` entry and its
directions came back as pseudo-protocol steps (live: 56 of them). These
exercise the new contract end to end with a stubbed author, so the wiring is
proven without spending live tokens on it.
"""

import json
from pathlib import Path

import pytest

from scilink.agents.planning_agents import planning_rag as pr
from scilink.agents.planning_agents.parser_utils import (
    plan_directions, plan_is_portfolio, portfolio_to_experiment_shim)

PORTFOLIO = {
    "portfolio_title": "Rare-site localization",
    "thesis": "Rare sites dominate the outcome.",
    "directions": [
        {"id": "RS-1", "title": "Correlative SECCM+AFM", "tier": "flagship",
         "hypothesis": "H1", "rationale": "R1", "novelty": "N1"},
        {"id": "RS-2", "title": "Event-gated spectroscopy",
         "hypothesis": "H2", "rationale": "R2", "novelty": "N2"},
    ],
    "shared_protocol": ["co-register with >=3 fiducials"],
    "open_questions": ["what registration error is achievable"],
}


# ── the contract ─────────────────────────────────────────────────────

def test_contract_targets_directions_not_experiments():
    c = pr.portfolio_contract()
    assert c["key"] == "directions"
    assert "portfolio" in c["strict"].lower()
    # the authoring contract must forbid the fields that caused the mess
    for banned in ("experimental_steps", "required_equipment",
                   "optimization_params"):
        assert banned in c["strict"], "must explicitly rule the field out"


def test_contract_summary_feeds_distinctness_on_directions():
    """Candidate 2 must be told what candidate 1 proposed. For a portfolio
    that is the thesis plus the direction titles, not a hypothesis field
    that does not exist there."""
    s = pr.portfolio_contract()["summarise"](PORTFOLIO)
    assert "Rare sites dominate" in s
    assert "Correlative SECCM+AFM" in s and "Event-gated" in s


def test_candidate_loop_is_unchanged_without_a_contract():
    """The experiment path must be byte-identical — this is the regression
    that would break every lab campaign."""
    import inspect
    src = inspect.getsource(pr.generate_plan_candidates)
    assert '_key = (contract or {}).get("key", "proposed_experiments")' in src
    assert '_label = (contract or {}).get("label", "Candidate Plan")' in src
    # the tier/decline logic reads through the key, never a literal
    assert 'not first.get("proposed_experiments")' not in src
    assert "not res.get(_key)" in src


# ── the generated artifact ───────────────────────────────────────────

def test_a_portfolio_satisfies_every_legacy_reader():
    """Fifty-odd sites read `proposed_experiments`; the validity gates treat
    a missing key as a FAILED plan and abort the run."""
    p = portfolio_to_experiment_shim(dict(PORTFOLIO, type="ideation"))

    assert p.get("proposed_experiments"), "would read as a failed plan"
    exp = p["proposed_experiments"][0]
    assert exp["hypothesis"] == "Rare sites dominate the outcome."
    assert [c["id"] for c in exp["concepts"]] == ["RS-1", "RS-2"]
    # shared protocol lands where a legacy reader looks for steps — and is
    # SHARED protocol, not the portfolio flattened into one
    assert exp["experimental_steps"] == ["co-register with >=3 fiducials"]
    assert len(exp["experimental_steps"]) < len(plan_directions(p)) + 3

    # the real payload is intact and preferred
    assert plan_is_portfolio(p) and len(plan_directions(p)) == 2
    assert plan_directions(p)[0]["id"] == "RS-1"


def test_the_portfolio_never_carries_bench_fields():
    """The failure this whole change exists to prevent: invented protocol
    and invented tunables for directions nobody has chosen yet."""
    p = portfolio_to_experiment_shim(dict(PORTFOLIO, type="ideation"))
    exp = p["proposed_experiments"][0]
    assert not exp.get("optimization_params")
    assert exp["required_equipment"] == []


def test_directions_survive_a_json_round_trip():
    """plan.json is written and re-read by the dossier, the white paper and
    a restored checkpoint."""
    p = portfolio_to_experiment_shim(dict(PORTFOLIO, type="ideation"))
    back = json.loads(json.dumps(p))
    assert [d["id"] for d in plan_directions(back)] == ["RS-1", "RS-2"]
    assert plan_is_portfolio(back)


# ── wiring ───────────────────────────────────────────────────────────

def test_generate_plan_routes_kind_to_the_portfolio_contract():
    import inspect
    from scilink.agents.planning_agents.planning_agent import PlanningAgent
    src = inspect.getsource(PlanningAgent.generate_plan)
    flat = " ".join(src.split())
    assert '_portfolio_run = (kind == "portfolio")' in flat
    assert "contract=(portfolio_contract() if _portfolio_run else None)" in flat
    assert "if _portfolio_run: res, author_context = author_portfolio(" in flat
    # stamped as ideation, and shimmed for the legacy readers
    assert "if _ideation_run or _portfolio_run:" in flat
    assert "res = portfolio_to_experiment_shim(res)" in flat
    # candidates shimmed BEFORE the judge reads them
    assert flat.index("candidates = [portfolio_to_experiment_shim(c)") < \
        flat.index("bestofn_judge = judge_plan_candidates(")


def test_the_tool_exists_and_points_the_other_way():
    src = Path("scilink/agents/planning_agents/orchestrator_tools.py").read_text()
    assert 'name="generate_ideation_portfolio"' in src
    i = src.index('name="generate_ideation_portfolio"')
    desc = src[i:i + 1800]
    assert "USE THIS INSTEAD OF" in desc and "generate_initial_plan" in desc
    # and it routes a picked direction back to the plan tool
    assert "selection_profile='lab'" in desc
    # kind is threaded, not hardcoded in two places
    assert 'kind="portfolio"' in src and 'kind: str = "experiment"' in src


# ── routing ──────────────────────────────────────────────────────────

def test_the_deprecated_ideation_profile_forwards_rather_than_authors():
    """It must not author a portfolio into the experiment schema one more
    time — that is the bug being retired."""
    src = Path("scilink/agents/planning_agents/orchestrator_tools.py").read_text()
    flat = " ".join(src.split())
    assert 'if selection_profile == "ideation" and kind != "portfolio":' in flat
    assert 'kind = "portfolio"' in flat
    assert "deprecated" in flat


def test_the_orchestrator_prompt_teaches_the_new_path():
    """The system prompt taught selection_profile='ideation' for years; the
    tool description alone will not outvote it."""
    src = Path("scilink/agents/planning_agents/planning_orchestrator.py").read_text()
    assert "generate_ideation_portfolio(..., literature_context=...)" in src
    assert 'generate_initial_plan(..., literature_context=..., selection_profile="ideation")' \
        not in src
    assert "flattened into protocol steps" in src


# ── refinement verbs ─────────────────────────────────────────────────

def test_portfolio_refinement_uses_portfolio_verbs():
    """A bench plan is refined against RESULTS; a portfolio against
    JUDGEMENT. The generic block speaks of experiments and results, which
    reads as an instruction to turn the portfolio into a protocol."""
    from scilink.agents.planning_agents.instruct import (
        PORTFOLIO_REFINEMENT_RULES as R)
    for verb in ("HARDEN", "DROP", "ADD", "RE-RANK", "CONSOLIDATE"):
        assert verb in R
    assert "Preserve every direction the feedback did not touch" in R
    # and it re-states the ban, because refinement is where invented
    # protocol crept back in before
    for banned in ("experimental_steps", "required_equipment",
                   "optimization_params"):
        assert banned in R


def test_the_verbs_are_injected_only_for_portfolios():
    import inspect
    src = inspect.getsource(pr.refine_plan_with_feedback)
    flat = " ".join(src.split())
    assert "if plan_is_portfolio(original_result) or any(" in flat
    assert "refinement_prompt += PORTFOLIO_REFINEMENT_RULES" in flat
    # the shim shape counts too — a portfolio in flight carries concepts
    assert '(e or {}).get("concepts")' in flat


# ── the three renderers, on one portfolio ────────────────────────────

def _shimmed():
    return portfolio_to_experiment_shim(dict(PORTFOLIO, type="ideation"))


def test_console_renders_directions_not_a_protocol():
    import io, contextlib
    from scilink.agents.planning_agents import user_interface as ui
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ui.display_plan_summary(_shimmed(), ideation=True,
                                report_path="/x/portfolio.html")
    out = buf.getvalue()
    assert "PROPOSED RESEARCH DIRECTIONS" in out
    assert out.count("  ▸ ") == 2, "one block per direction"
    assert "RS-1" in out and "RS-2" in out
    # the shared protocol is labelled as shared, not as the whole plan
    assert "Shared Protocol" in out
    assert "Experimental Steps" not in out


def test_html_renders_direction_cards():
    from scilink.agents.planning_agents.html_generator import HTMLReportGenerator
    import tempfile, pathlib
    plan = _shimmed()
    out = pathlib.Path(tempfile.mkdtemp()) / "p.html"
    HTMLReportGenerator({"current_plan": plan, "plan_history": [plan],
                         "objective": "o"}).generate_single_plan(
        plan, str(out), title="Portfolio")
    h = out.read_text()
    assert "Organizing thesis" in h and "Research directions (2)" in h
    assert "RS-1" in h and "Correlative SECCM+AFM" in h
    assert "Shared protocol" in h and "Open questions" in h
    # never the protocol view
    assert "Experimental Steps" not in h


def test_html_still_renders_an_experiment_as_a_protocol():
    """The regression that would matter most."""
    from scilink.agents.planning_agents.html_generator import HTMLReportGenerator
    import tempfile, pathlib
    lab = {"proposed_experiments": [{"experiment_name": "E", "hypothesis": "H",
                                     "experimental_steps": ["step one"],
                                     "required_equipment": ["RDE"]}]}
    out = pathlib.Path(tempfile.mkdtemp()) / "l.html"
    HTMLReportGenerator({"current_plan": lab, "plan_history": [lab],
                         "objective": "o"}).generate_single_plan(
        lab, str(out), title="Plan")
    h = out.read_text()
    assert "step one" in h, "the protocol is the lab deliverable"
    assert "Organizing thesis" not in h
    # NB: required_equipment is absent from the single-plan template on main
    # too — a pre-existing gap, not something this change introduced.


def test_a_portfolio_edit_has_a_findable_tool():
    """Live: "drop the weakest direction" went to write_technical_document,
    which wrote a document ABOUT the revised portfolio and left the
    portfolio itself untouched. refine_plan_with_results reaches the right
    engine but its name promises results, which a drop does not have."""
    src = Path("scilink/agents/planning_agents/orchestrator_tools.py").read_text()
    assert 'name="refine_portfolio"' in src
    i = src.index('name="refine_portfolio"')
    desc = src[i:i + 1500]
    for verb in ("HARDEN", "DROP", "ADD", "RE-RANK", "CONSOLIDATE"):
        assert verb in desc
    assert "preserved verbatim" in desc
    # it must warn off both wrong turns seen live
    assert "do NOT write a document" in desc
    assert "generate_ideation_portfolio" in desc
    # and the document tool declines the job
    j = src.index('name="write_technical_document"')
    assert "belong in refine_portfolio" in src[j:j + 3000]


# ── argument tolerance ───────────────────────────────────────────────

def test_portfolio_tool_tolerates_the_profile_it_has_to_mention():
    """Live: the model called generate_ideation_portfolio(selection_profile=
    'ideation') and got a TypeError, costing a round trip. The tool's own
    description names selection_profile='lab' to say where a CHOSEN
    direction goes next, which puts the parameter in scope — so it must not
    be an error to pass it. It is meaningless here: a portfolio is ideation
    by construction, and it stays out of the schema."""
    import inspect
    from scilink.agents.planning_agents.orchestrator_tools import OrchestratorTools
    src = inspect.getsource(OrchestratorTools._register_all_tools)
    i = src.index("def generate_ideation_portfolio")
    sig = src[i:src.index("):", i)]
    assert "selection_profile" in sig, "must be accepted"
    # ...but not advertised
    j = src.index('name="generate_ideation_portfolio"')
    params = src[j:src.index("required=", j)]
    assert '"selection_profile"' not in params, "must not be in the schema"
    # and the call still forces the portfolio contract
    body = src[i:j]
    assert 'selection_profile="ideation"' in body and 'kind="portfolio"' in body


def test_an_unknown_argument_gets_told_what_is_accepted(tmp_path):
    """A bare TypeError names the bad argument but not the good ones, so the
    model has to guess its way back."""
    from types import SimpleNamespace
    from scilink.agents.planning_agents.orchestrator_tools import OrchestratorTools

    t = OrchestratorTools.__new__(OrchestratorTools)
    t.functions_map = {"demo": lambda alpha=None, beta=None: "ok"}
    t.openai_schemas = []          # no schema -> nothing required
    out = json.loads(OrchestratorTools.execute_tool(t, "demo", gamma=1))
    assert out["status"] == "error" and out["tool"] == "demo"
    assert "unexpected keyword argument" in out["message"]
    assert "alpha" in out["message"] and "beta" in out["message"]

    # a genuine failure inside a tool is still reported as itself
    def boom():
        raise ValueError("real failure")
    t.functions_map["boom"] = boom
    out2 = json.loads(OrchestratorTools.execute_tool(t, "boom"))
    assert out2["status"] == "error" and "real failure" in out2["message"]
    assert "Accepted arguments" not in out2["message"]


def test_wrapped_bullets_do_not_repeat_their_marker():
    """Live: every continuation line was prefixed with '- ', so a single
    sentence read as four bullets and consecutive entries were
    indistinguishable."""
    from scilink.agents.planning_agents.user_interface import _wrap_field
    long = ("FUNCTION AND RARE SITE: reaching a designated target polymorph "
            "by design, where the rare active site is the transient "
            "nucleation locus that selects which phase becomes competent.")
    out = _wrap_field(long, indent="       - ")
    lines = out.splitlines()
    assert len(lines) > 1, "needs to wrap for this to mean anything"
    assert lines[0].startswith("       - ")
    assert all(not l.lstrip().startswith("- ") for l in lines[1:]), out
    # continuations align under the text, not under the marker
    assert all(l.startswith(" " * 9) for l in lines[1:]), out


def test_plain_indents_are_unchanged_by_that_fix():
    """Every other caller passes whitespace; their output must not move."""
    from scilink.agents.planning_agents.user_interface import _wrap_field
    out = _wrap_field("word " * 40, indent="   ")
    assert all(l.startswith("   ") and not l.startswith("    ")
               for l in out.splitlines())


def test_details_are_labelled():
    """Unlabelled they read as a continuation of the field above them."""
    import io, contextlib
    from scilink.agents.planning_agents import user_interface as ui
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ui._print_concepts([{"id": "A", "title": "T", "novelty": "N",
                             "details": ["one", "two"]}])
    out = buf.getvalue()
    assert "Details:" in out
    assert out.index("Novelty:") < out.index("Details:")
