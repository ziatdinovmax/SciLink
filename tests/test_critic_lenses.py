"""Reviewer-grade critic lenses (offline).

Phase 0 — the critic sees the whole plan (direction details, shared
protocol, open questions), not the clipped conformance summary.
Phase 1 — the critic prompt carries the six lenses and the widened
dimension set; downstream rendering is unchanged.
Phase 3a — deterministic document hygiene checks and their wiring into
write_technical_document as advisory review_notes.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scilink.agents.planning_agents import planning_rag as pr
from scilink.agents.planning_agents import orchestrator_tools as ot
from scilink.agents.planning_agents.user_interface import format_caveats
from scilink.utils import doc_hygiene as dh
from tests.test_edit_file_tool import make_tools


# ------------------------------------------------------------ fixtures (synthetic)

def _portfolio():
    return {
        "proposed_experiments": [{
            "experiment_name": "Portfolio: state-triggered control",
            "hypothesis": "Timing on state beats timing on a clock.",
            "justification": "shim",
            "concepts": [{"id": "D1", "tier": "primary", "title": "Direction one",
                          "hypothesis": "H1", "novelty": "N1"}],
        }],
        "directions": [{
            "id": "D1", "tier": "primary", "title": "Direction one",
            "hypothesis": "H1", "novelty": "N1",
            "details": ["(a) System: film on a reusable coupon.",
                        "(b) Policies: P1..P5; five policies x 6-8 trajectories = 30-40 episodes.",
                        "(c) Track 2 on QUENCHED coupons."],
        }],
        "shared_protocol": ["Every state-triggered policy paired with a YOKED clock control."],
        "open_questions": ["Is S estimable with low enough latency?"],
    }


def _lab_plan():
    return {"proposed_experiments": [{
        "experiment_name": "Single experiment", "hypothesis": "H",
        "justification": "J", "experimental_steps": ["s1", "s2"]}]}


# ------------------------------------------------------------ Phase 0

def test_critic_summary_includes_details_protocol_and_open_questions():
    s = pr.summarize_plan_for_critic(_portfolio())
    assert "five policies x 6-8 trajectories" in s
    assert "QUENCHED" in s
    assert "YOKED clock control" in s
    assert "OPEN QUESTIONS THE AUTHOR ALREADY RAISED" in s
    assert "Is S estimable" in s
    # the conformance summary is a strict subset
    assert pr.summarize_experiment(_portfolio()["proposed_experiments"][0], 1) in s


def test_critic_summary_clips_long_details_with_marker():
    p = _portfolio()
    p["directions"][0]["details"] = ["x" * (pr._CRITIC_DIRECTION_CLIP + 500)]
    s = pr.summarize_plan_for_critic(p)
    assert "[clipped]" in s
    assert len(s) < pr._CRITIC_DIRECTION_CLIP + 1500


def test_critic_summary_for_lab_plan_equals_conformance_summary():
    p = _lab_plan()
    assert pr.summarize_plan_for_critic(p) == pr.summarize_experiment(p["proposed_experiments"][0], 1)


def test_conformance_summary_unchanged_for_portfolios():
    # verify_plan_relevance keeps the coverage-and-identity view: no details leak
    s = pr.summarize_experiment(_portfolio()["proposed_experiments"][0], 1)
    assert "QUENCHED" not in s and "YOKED" not in s


# ------------------------------------------------------------ Phase 1

class _CaptureModel:
    def __init__(self, reply):
        self.reply = reply; self.prompts = []

    def generate_content(self, parts, generation_config=None):
        self.prompts.append("".join(p for p in parts if isinstance(p, str)))
        return SimpleNamespace(text=self.reply)


def test_critic_prompt_carries_all_lenses_and_full_plan():
    reply = json.dumps({"findings": [
        {"dimension": "design", "severity": "critical", "experiment": "D1",
         "issue": "Pairing every policy with a yoked control makes 7 arms, not 5."},
        {"dimension": "physics", "severity": "minor", "experiment": "D1", "issue": "x"}]})
    m = _CaptureModel(reply)
    out = pr.critique_plan("obj", _portfolio(), m, None,
                           retrieved_context="Chronocoulometry overestimates absorbed hydrogen.")
    prompt = m.prompts[0]
    for dim in pr.CRITIC_DIMENSIONS:
        assert f"• {dim}" in prompt, dim
    assert "physics|consistency|design|statistics|method|evidence" in prompt
    assert "QUENCHED" in prompt and "YOKED" in prompt          # Phase 0 reaches the prompt
    assert "OPEN QUESTIONS THE AUTHOR ALREADY RAISED" in prompt
    assert "read it for stated limitations" in prompt  # evidence framing
    assert "most 10 findings" in prompt
    assert [f["dimension"] for f in out["findings"]] == ["design", "physics"]


def test_new_dimensions_render_through_format_caveats_unchanged():
    lines = format_caveats([
        {"dimension": "statistics", "severity": "critical", "issue": "unit not named"},
        {"dimension": "method", "severity": "minor", "issue": "KPFM in electrolyte"},
    ])
    assert lines == ["[statistics] unit not named", "Minor: [method] KPFM in electrolyte"]


def test_critic_fails_open_on_bad_json():
    m = _CaptureModel("not json at all")
    assert pr.critique_plan("obj", _portfolio(), m, None) == {"findings": []}


# ------------------------------------------------------------ Phase 3a: validators

def test_image_link_check(tmp_path):
    (tmp_path / "ok.png").write_bytes(b"x")
    notes = dh.check_image_links("![a](ok.png) ![b](missing.png) ![c](https://x/y.png)", tmp_path)
    assert [n["note"] for n in notes] == ["Image link does not resolve: missing.png"]


def test_meta_language_check():
    notes = dh.find_meta_language("Values as retrieved in campaign literature; the specialist wrote it.")
    assert {n["note"] for n in notes} >= {
        "Agent meta-language in the document: 'as retrieved in'",
        "Agent meta-language in the document: 'the specialist'"}
    assert dh.find_meta_language("A clean sentence.") == []


@pytest.mark.parametrize("text,expect", [
    ("We run 7 arms × 6–8 replicates. The campaign is ≈ 30–40 episodes.", True),
    ("three arms × ~10–12 coupons; ~30–40 episodes total", False),
    ("five policies × six to eight trajectories; roughly 30–40 episodes", False),
    ("five policies × six to eight trajectories; roughly 20–25 episodes", True),
    ("no numbers here", False),
    ("3 × 12 coupons but no total stated", False),
])
def test_design_arithmetic(text, expect):
    notes = dh.check_design_arithmetic(text)
    assert bool(notes) is expect, notes


def test_acronym_fidelity():
    src = ["We propose an Operando Materials Observation & Control Platform (OMOC) that ..."]
    bad = "The Optimal Model of Operating Conditions (OMOC) thesis holds ..."
    good = "The Operando Materials Observation & Control Platform (OMOC) thesis holds ..."
    assert dh.check_acronym_fidelity(bad, src)[0]["lens"] == "alignment"
    assert dh.check_acronym_fidelity(good, src) == []
    # acronyms the sources never expand are not judged
    assert dh.check_acronym_fidelity("Scanning Electrochemical Cell Microscopy (SECCM)", src) == []


def test_hygiene_orders_critical_first(tmp_path):
    txt = "the specialist said; ![f](nope.png)"
    notes = dh.check_document_hygiene(txt, tmp_path, [])
    assert [n["severity"] for n in notes] == ["critical", "minor"]


# ------------------------------------------------------------ Phase 3a: wiring

def test_write_technical_document_returns_review_notes(tmp_path, monkeypatch):
    tools, cap, deleg = make_tools(tmp_path)
    tools._maybe_embed_workflow_diagram = lambda text, d, stem=None: text
    secs = [{"heading": "Design", "body": "We run 7 arms × 6–8 replicates; ≈ 30–40 episodes.\n\n"
                                            "![fig](missing.png)\n\nAs retrieved in campaign literature."}]
    monkeypatch.setattr(ot, "author_technical_document",
                        lambda request, kb_docs, model, generation_config, **kw: {"sections": secs})
    out = json.loads(cap["write_technical_document"](
        request="doc", filename="d.md", use_literature=False))
    assert out["status"] == "success"
    lenses = {n["lens"] for n in out["review_notes"]}
    assert {"artifact", "hygiene", "design"} <= lenses
    assert "nothing was changed" in out["review_note"]
    # advisory: the file is exactly what the author returned
    assert "![fig](missing.png)" in Path(out["path"]).read_text()


def test_write_technical_document_clean_doc_has_no_notes(tmp_path, monkeypatch):
    tools, cap, deleg = make_tools(tmp_path)
    tools._maybe_embed_workflow_diagram = lambda text, d, stem=None: text
    monkeypatch.setattr(ot, "author_technical_document",
                        lambda request, kb_docs, model, generation_config, **kw:
                        {"sections": [{"heading": "A", "body": "Plain prose."}]})
    out = json.loads(cap["write_technical_document"](
        request="doc", filename="d.md", use_literature=False))
    assert "review_notes" not in out


# ------------------------------------------------------------ Phase 2: adversarial retrieval

def test_literature_agent_has_adversarial_leg():
    from scilink.agents.lit_agents.literature_agent import LiteratureSearchAgent as LiteratureAgent
    calls = []
    agent = LiteratureAgent.__new__(LiteratureAgent)
    agent._execute_crow_task = lambda q, task_type=None: calls.append((q, task_type)) or {"ok": True}
    out = agent.search_for_technique_limitations("KPFM in 0.1 M HClO4; coulometric H:Pd")
    q, tt = calls[0]
    assert out == {"ok": True} and tt == "Limitations"
    assert "LIMITATIONS, ARTIFACTS and FAILURE MODES" in q
    assert "KPFM in 0.1 M HClO4" in q
    assert "Do NOT review what the technique does well" in q


def test_search_literature_tool_registers_technique_limitations(tmp_path):
    tools, cap, deleg = make_tools(tmp_path)
    tools.orch.lit_agent = SimpleNamespace(
        search_for_hypothesis_context=lambda o: {}, search_for_cross_domain=lambda o: {},
        search_for_technique_limitations=lambda o: {}, search_for_economic_data=lambda o: {},
        search_for_fitting_models=lambda o: {})
    out = json.loads(cap["search_literature"]("x", search_type="no_such_type"))
    assert out["status"] == "error" and "technique_limitations" in out["message"]
