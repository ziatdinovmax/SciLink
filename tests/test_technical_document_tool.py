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


def test_an_experimental_protocol_is_not_a_document():
    """The mirror of the original bug, seen live: after ideation, "give me
    the runnable bench protocol" was authored as prose, so it had no
    conformance check, no critic, and could never be refined with results."""
    src = Path("scilink/agents/planning_agents/orchestrator_tools.py").read_text()
    i = src.index('name="write_technical_document"')
    desc = src[i:i + 2600]
    assert "NOT for an EXPERIMENTAL PROTOCOL" in desc
    assert "runnable bench" in desc
    assert "A document cannot be refined with results." in desc
    # save_file must not advertise protocols as its own either
    j = src.index('name="save_file"')
    assert "protocols, notes" not in src[j - 1200:j + 400]


# ── revision in place ────────────────────────────────────────────────

ORIGINAL = ("# CDOC Class 1 White Paper\n\n## Significance\n"
            + "Body text that must survive the revision. " * 40
            + "\n\n## Approach\n" + "More body. " * 40 + "\n")


def _doc_tool(tmp_path, monkeypatch, sections, captured=None):
    from types import SimpleNamespace
    def fake_author(request, kb_docs, model, generation_config, **kw):
        if captured is not None:
            captured.update(request=request, **kw)
        return {"sections": sections}
    monkeypatch.setattr(ot, "author_technical_document", fake_author)
    t = OrchestratorTools.__new__(OrchestratorTools)
    t.functions_map = {}
    t.orch = SimpleNamespace(
        base_dir=tmp_path, _active_output_subdir=None,
        planner=SimpleNamespace(kb_docs=None, model=None, generation_config=None,
                                _build_skill_context=lambda s: None))
    t._output_dir = lambda: tmp_path / "delegations" / "04_revise"
    t._output_dir().mkdir(parents=True, exist_ok=True)
    t._latest_literature_file = lambda: None
    t._register_tool = lambda func, name, **kw: t.functions_map.setdefault(name, func)
    return t


def test_revision_writes_back_over_the_same_path(tmp_path, monkeypatch):
    """Live: 'revise the paper you wrote' authored into the CURRENT
    delegation folder while the original sat untouched, and the agent then
    rebuilt the file by hand in chunks."""
    orig = tmp_path / "delegations" / "02_author" / "paper.md"
    orig.parent.mkdir(parents=True)
    orig.write_text(ORIGINAL)

    cap = {}
    t = _doc_tool(tmp_path, monkeypatch,
                  [{"heading": "Significance", "body": "Body " * 900},
                   {"heading": "References", "body": "[1] Maeda 2012"}], cap)
    OrchestratorTools._register_all_tools(t)
    out = json.loads(t.functions_map["write_technical_document"](
        request="add references", revise_path=str(orig)))

    assert out["status"] == "success" and out["revised_in_place"] is True
    assert Path(out["path"]) == orig.resolve(), "must not spawn a new copy"
    assert "[1] Maeda 2012" in orig.read_text()
    # the author saw the document it was revising
    assert cap["revise_document"].startswith("# CDOC Class 1 White Paper")


def test_a_shrinking_revision_is_refused(tmp_path, monkeypatch):
    """A revision that comes back much shorter is the model summarising
    instead of revising — overwriting the good copy with it is the failure
    the live session hit by hand (5,816 bytes left of a 22 KB paper)."""
    orig = tmp_path / "delegations" / "02_author" / "paper.md"
    orig.parent.mkdir(parents=True)
    orig.write_text(ORIGINAL)

    t = _doc_tool(tmp_path, monkeypatch,
                  [{"heading": "Summary", "body": "tiny"}])
    OrchestratorTools._register_all_tools(t)
    out = json.loads(t.functions_map["write_technical_document"](
        request="add references", revise_path=str(orig)))

    assert out["status"] == "error" and "Revision aborted" in out["message"]
    assert orig.read_text() == ORIGINAL, "the original must be untouched"


def test_revision_cannot_escape_the_session(tmp_path, monkeypatch):
    t = _doc_tool(tmp_path, monkeypatch, [{"heading": "H", "body": "B"}])
    OrchestratorTools._register_all_tools(t)
    out = json.loads(t.functions_map["write_technical_document"](
        request="x", revise_path="/etc/hosts"))
    assert out["status"] == "error" and "session directory" in out["message"]


def test_missing_revision_target_is_reported(tmp_path, monkeypatch):
    t = _doc_tool(tmp_path, monkeypatch, [{"heading": "H", "body": "B"}])
    OrchestratorTools._register_all_tools(t)
    out = json.loads(t.functions_map["write_technical_document"](
        request="x", revise_path=str(tmp_path / "nope.md")))
    assert out["status"] == "error" and "No such document" in out["message"]


def test_the_schema_steers_away_from_hand_rebuilding():
    src = Path("scilink/agents/planning_agents/orchestrator_tools.py").read_text()
    i = src.index('"revise_path"')
    desc = src[i:i + 1400]
    assert "IN PLACE" in desc and "SAME path" in desc
    assert "save_file" in desc and "truncates" in desc


def test_citations_must_be_carried_not_dropped():
    """Splitting a referenced paper in two produced two unreferenced papers:
    the contract said only NEVER INVENT, so dropping them read as caution."""
    from scilink.agents.planning_agents.instruct import (
        TECHNICAL_DOCUMENT_INSTRUCTIONS as T,
        TECHNICAL_DOCUMENT_REVISION_RULES as R)
    flat = " ".join(T.split())
    assert "CARRY ITS REFERENCES THROUGH" in flat
    assert 'numbered "References" section' in flat
    assert "NEVER invent" in T          # the prohibition survives
    # a revision returns the whole document, because it overwrites
    assert "COMPLETE revised document" in R and "verbatim" in R


def test_the_revising_delegation_keeps_what_it_replaced(tmp_path, monkeypatch):
    """Revising in place crosses delegation isolation, which exists so a
    reused child cannot clobber earlier outputs by accident. An explicit
    revision is not that — but the record still has to survive, so the
    delegation that made the change keeps the version it replaced."""
    orig = tmp_path / "delegations" / "02_author" / "paper.md"
    orig.parent.mkdir(parents=True)
    orig.write_text(ORIGINAL)

    t = _doc_tool(tmp_path, monkeypatch,
                  [{"heading": "Significance", "body": "Body " * 900}])
    OrchestratorTools._register_all_tools(t)
    out = json.loads(t.functions_map["write_technical_document"](
        request="add references", revise_path=str(orig)))
    assert out["status"] == "success"

    bak = t._output_dir() / "paper.before_revision.md"
    assert bak.exists(), "the replaced version must survive somewhere"
    assert bak.read_text() == ORIGINAL
    # ...next to the delegation that changed it, not next to the original
    assert bak.parent != orig.parent
    assert orig.read_text() != ORIGINAL, "the canonical file was updated"


def test_repeated_revisions_never_lose_the_original(tmp_path, monkeypatch):
    """The hole in a single fixed backup name: revise twice from one
    delegation and the ORIGINAL — the version the user approved — is the
    copy that vanishes."""
    orig = tmp_path / "delegations" / "02_author" / "paper.md"
    orig.parent.mkdir(parents=True)
    orig.write_text(ORIGINAL)

    t = _doc_tool(tmp_path, monkeypatch,
                  [{"heading": "S", "body": "Body " * 900}])
    OrchestratorTools._register_all_tools(t)
    fn = t.functions_map["write_technical_document"]

    first = json.loads(fn(request="add references", revise_path=str(orig)))
    v2 = orig.read_text()
    second = json.loads(fn(request="tighten it", revise_path=str(orig)))

    assert first["status"] == second["status"] == "success"
    baks = sorted(p.name for p in t._output_dir().glob("*before_revision*"))
    assert len(baks) == 2, baks
    contents = {(t._output_dir() / b).read_text() for b in baks}
    assert ORIGINAL in contents, "the original must still exist somewhere"
    assert v2 in contents


def test_the_revision_is_traceable_without_reading_the_disk(tmp_path,
                                                            monkeypatch):
    """Who changed what, visible in the result and in the files ledger."""
    from scilink.agents.planning_agents.user_interface import load_deliverables
    orig = tmp_path / "delegations" / "02_author" / "paper.md"
    orig.parent.mkdir(parents=True)
    orig.write_text(ORIGINAL)

    t = _doc_tool(tmp_path, monkeypatch,
                  [{"heading": "S", "body": "Body " * 900}])
    OrchestratorTools._register_all_tools(t)
    out = json.loads(t.functions_map["write_technical_document"](
        request="add references", revise_path=str(orig)))

    assert out["revised_by"] == "04_revise"
    assert out["previous_version"] and Path(out["previous_version"]).exists()

    entries = {e["title"]: e for e in load_deliverables(tmp_path)}
    trail = [k for k in entries if k.startswith("Pre-revision copy")]
    assert trail, entries
    assert "revised by 04_revise" in trail[0]
    # the audit copy is listed, never starred over the real deliverable
    assert entries[trail[0]]["deliverable"] is False


def test_a_revision_keeps_the_documents_own_name(tmp_path, monkeypatch):
    """Live: with no explicit title the deliverable was renamed after the
    INSTRUCTION — "Revise this brief with two targeted changes, keeping ALL
    other content" replaced its real name in the files list."""
    from scilink.agents.planning_agents.user_interface import load_deliverables
    orig = tmp_path / "delegations" / "02_author" / "paper.md"
    orig.parent.mkdir(parents=True)
    orig.write_text("# Commissioning Brief\n\n" + "Body text. " * 400)

    t = _doc_tool(tmp_path, monkeypatch,
                  [{"heading": "S", "body": "Body " * 900}])
    OrchestratorTools._register_all_tools(t)
    out = json.loads(t.functions_map["write_technical_document"](
        request="Revise this brief with two targeted changes, keeping ALL "
                "other content exactly as it is",
        revise_path=str(orig)))

    assert out["title"] == "Commissioning Brief"
    titles = [e["title"] for e in load_deliverables(tmp_path)]
    assert "Commissioning Brief" in titles
    assert not any(x.startswith("Revise this brief") for x in titles)


def test_a_titleless_document_falls_back_to_its_filename(tmp_path,
                                                          monkeypatch):
    orig = tmp_path / "delegations" / "02_author" / "build_roadmap.md"
    orig.parent.mkdir(parents=True)
    orig.write_text("No heading here.\n" + "Body text. " * 400)

    t = _doc_tool(tmp_path, monkeypatch,
                  [{"heading": "S", "body": "Body " * 900}])
    OrchestratorTools._register_all_tools(t)
    out = json.loads(t.functions_map["write_technical_document"](
        request="tighten it", revise_path=str(orig)))
    assert out["title"] == "build roadmap"


def test_an_explicit_title_still_wins(tmp_path, monkeypatch):
    orig = tmp_path / "delegations" / "02_author" / "paper.md"
    orig.parent.mkdir(parents=True)
    orig.write_text("# Old Name\n\n" + "Body text. " * 400)
    t = _doc_tool(tmp_path, monkeypatch,
                  [{"heading": "S", "body": "Body " * 900}])
    OrchestratorTools._register_all_tools(t)
    out = json.loads(t.functions_map["write_technical_document"](
        request="rename it", title="New Name", revise_path=str(orig)))
    assert out["title"] == "New Name"
