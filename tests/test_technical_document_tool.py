"""`write_technical_document` — the honest home for non-experiment requests.

Live (cdoc replay): "outline a plan for how the platform gets built" routed
to generate_initial_plan, which filled the experiment schema by invention —
a build sequence as `hypothesis`, notes-to-self as `experimental_steps`, and
six fabricated `optimization_params` with numeric ranges and citations for a
facility that did not exist. Three such documents were then starred as
"Experimental plan (report)".
"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scilink.agents.planning_agents import orchestrator_tools as ot
from scilink.agents.planning_agents.orchestrator_tools import OrchestratorTools
from scilink.agents.planning_agents.planning_rag import document_to_markdown

SECTIONS = [{"heading": "Assumptions", "body": "Lab-based; no beamline."},
            {"heading": "Stage 0", "body": "Freeze requirements.\n\n- a\n- b"}]


@pytest.fixture
def tools(tmp_path, monkeypatch):
    captured = {}

    def fake_author(request, kb_docs, model, generation_config, **kw):
        captured.update(request=request, **kw)
        return {"sections": SECTIONS}

    monkeypatch.setattr(ot, "author_technical_document", fake_author)
    t = OrchestratorTools.__new__(OrchestratorTools)
    t.functions_map, t.gemini_functions, t.openai_tools = {}, [], []
    t.orch = SimpleNamespace(
        base_dir=tmp_path, _active_output_subdir=None,
        planner=SimpleNamespace(kb_docs=None, model=None,
                                generation_config=None,
                                _build_skill_context=lambda s: None))
    t._output_dir = lambda: tmp_path
    t._latest_literature_file = lambda: None
    t._register_tool = lambda func, name, **kw: t.functions_map.setdefault(name, func)
    t._register_document_tool = None
    t.captured = captured
    return t


def _fn(tools):
    """Register just this tool by executing its definition block."""
    OrchestratorTools._register_all_tools(tools) if hasattr(
        OrchestratorTools, "_register_all_tools") else None
    return tools.functions_map.get("write_technical_document")


def test_markdown_assembly_is_deterministic():
    md = document_to_markdown("Build roadmap", SECTIONS)
    assert md.startswith("# Build roadmap")
    assert "## Assumptions" in md and "## Stage 0" in md
    assert "- a" in md          # body markdown survives verbatim
    assert md.endswith("\n")


def test_assembly_tolerates_a_ragged_section_list():
    md = document_to_markdown("T", [{"body": "no heading"}, "bare string",
                                    {"heading": "H"}, None])
    assert "no heading" in md and "bare string" in md and "## H" in md


def test_the_tool_is_registered_with_the_routing_boundary():
    """The boundary has to sit where routing is decided — in the tool
    descriptions the model reads when choosing — not in a parameter blurb."""
    src = Path("scilink/agents/planning_agents/orchestrator_tools.py").read_text()
    assert 'name="write_technical_document"' in src
    # generate_initial_plan names the alternative and the test for it
    i = src.index('name="generate_initial_plan"')
    desc = src[i:i + 1400]
    assert "write_technical_document" in desc
    assert "no hypothesis to" in desc and "nothing to measure" in desc


def test_document_authoring_never_touches_campaign_state():
    """A document is not a plan: no plan.json, no plan_history, no
    plan_kind, no protocol report."""
    src = Path("scilink/agents/planning_agents/orchestrator_tools.py").read_text()
    i = src.index("def write_technical_document")
    body = src[i:src.index('name="write_technical_document"')]
    for forbidden in ("plan_history", "_emit_plan_report", "plan.json",
                      "plan_kind", "_stamp_campaign", "current_plan"):
        assert forbidden not in body, forbidden


def test_anti_fabrication_rule_now_rides_every_plan():
    """It was ideation-only; the roadmap that invented six BO ranges was
    typed lab, so the rule never reached it."""
    from scilink.agents.planning_agents.instruct import (
        HYPOTHESIS_GENERATION_INSTRUCTIONS,
        HYPOTHESIS_GENERATION_INSTRUCTIONS_FALLBACK)
    for block in (HYPOTHESIS_GENERATION_INSTRUCTIONS,
                  HYPOTHESIS_GENERATION_INSTRUCTIONS_FALLBACK):
        assert "Do not invent optimization parameters" in block
        assert "authoritative-looking numbers" in block


def test_the_document_contract_forbids_invented_figures():
    from scilink.agents.planning_agents.instruct import (
        TECHNICAL_DOCUMENT_INSTRUCTIONS)
    t = TECHNICAL_DOCUMENT_INSTRUCTIONS
    assert "NEVER invent" in t
    assert "sections" in t and "heading" in t and "body" in t
    # and it must not smuggle the experiment schema back in
    assert "not an experimental plan" in t
