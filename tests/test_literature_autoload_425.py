"""Plural literature auto-load (issue #425).

A campaign accumulates literature across several searches, but auto-load
used only the NEWEST file — a live refine reasoned over 77 KB while
silently ignoring the 256 KB corpus holding most of the evidence, and the
log line read like success. Covers:

- _campaign_literature_files: full campaign-scoped list, oldest first;
  legacy pre-registry glob fallback stays plural too;
- _latest_literature_file compat wrapper (newest);
- _load_campaign_literature: ONE file loads whole and byte-identical
  (registry AND legacy paths — the empirical no-op proof), several files
  union oldest-first, verbatim-duplicate sections deduped, over-budget
  whole sections dropped with the omission logged — no silent caps;
- the question-heading contract: writer/splitter lockstep, round-trip;
- '<path>#qN' section refs in _resolve_context_text/_context_file_paths;
- list_literature_searches: per-question index with previews, sizes and
  section refs; registry label/questions stamping.

All offline; no LLM, no network.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scilink.agents.planning_agents import orchestrator_tools as ot
from scilink.agents.planning_agents.orchestrator_tools import (
    OrchestratorTools,
    _format_lit_question_heading,
)


FOUNDATION = (
    "# Literature Search Results (hypothesis_context+cross_domain)\n\n"
    + _format_lit_question_heading(1, "What is known about defect kinetics?")
    + "\n\n## ESTABLISHED IN THIS FIELD\nFoundational corpus on defect "
      "kinetics. " + "Broad evidence. " * 40 + "\n\n"
    + _format_lit_question_heading(2, "Capture a state that exists only "
                                      "under drive?")
    + "\n\n## TRANSFERABLE MECHANISMS\nCross-domain nucleation one-shot "
      "path selection. " + "Analogy evidence. " * 40 + "\n"
)

TOPUP = (
    "# Literature Search Results (hypothesis_context)\n\n"
    "## ESTABLISHED IN THIS FIELD\nNarrow top-up on annealing schedules. "
    + "Recent evidence. " * 30 + "\n"
)


def make_tools(tmp_path, state):
    tools = OrchestratorTools.__new__(OrchestratorTools)
    tools.orch = SimpleNamespace(
        planner=SimpleNamespace(state=state),
        base_dir=Path(tmp_path),
        _active_output_subdir=None,
    )
    tools._prestate_lit = []
    return tools


def _lit_file(tmp_path, name, text, mtime=None):
    p = Path(tmp_path) / name
    p.write_text(text)
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


def _registry_state(*entries, cid=1):
    return {"campaign_id": cid, "current_plan": {"x": 1},
            "campaign_literature": list(entries)}


T0 = 1_700_000_000  # arbitrary fixed epoch; deltas are what matter


# ------------------------------------------------- plural resolver

def test_all_campaign_files_oldest_first(tmp_path):
    old = _lit_file(tmp_path, "literature_search_a.md", FOUNDATION, T0)
    new = _lit_file(tmp_path, "literature_search_b.md", TOPUP, T0 + 100)
    other = _lit_file(tmp_path, "literature_search_c.md", "other", T0 + 50)
    state = _registry_state(
        {"path": str(new), "campaign_id": 1},
        {"path": str(old), "campaign_id": 1},
        {"path": str(other), "campaign_id": 2},
    )
    tools = make_tools(tmp_path, state)
    assert tools._campaign_literature_files() == [old, new]
    assert tools._latest_literature_file() == new  # wrapper: newest


def test_legacy_glob_fallback_is_plural_and_oldest_first(tmp_path):
    old = _lit_file(tmp_path, "literature_search_x.md", FOUNDATION, T0)
    sub = tmp_path / "delegations" / "02_refine"
    sub.mkdir(parents=True)
    new = _lit_file(sub, "literature_search_y.md", TOPUP, T0 + 100)
    tools = make_tools(tmp_path, {"campaign_id": 1, "current_plan": {"x": 1}})
    assert tools._campaign_literature_files() == [old, new]
    # after a campaign transition, still no session-wide scraping
    tools2 = make_tools(tmp_path, {"campaign_id": 2, "current_plan": {"x": 1}})
    assert tools2._campaign_literature_files() == []


def test_duplicate_registry_entries_yield_one_file(tmp_path):
    f = _lit_file(tmp_path, "literature_search_a.md", FOUNDATION, T0)
    state = _registry_state({"path": str(f), "campaign_id": 1},
                            {"path": str(f), "campaign_id": 1})
    tools = make_tools(tmp_path, state)
    assert tools._campaign_literature_files() == [f]


# ------------------------------------------------- single-file fast path

def test_single_file_loads_byte_identical_registry(tmp_path):
    f = _lit_file(tmp_path, "literature_search_a.md", FOUNDATION, T0)
    tools = make_tools(tmp_path,
                       _registry_state({"path": str(f), "campaign_id": 1}))
    lit = tools._load_campaign_literature()
    assert lit["text"] == FOUNDATION          # byte-identical, uncapped
    assert lit["n_files"] == 1 and lit["dropped"] == []


def test_single_file_loads_byte_identical_legacy_glob(tmp_path):
    _lit_file(tmp_path, "literature_search_a.md", FOUNDATION, T0)
    tools = make_tools(tmp_path, {"campaign_id": 1, "current_plan": {"x": 1}})
    lit = tools._load_campaign_literature()
    assert lit["text"] == FOUNDATION


def test_single_file_ignores_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(ot, "_LIT_AUTOLOAD_MAX_CHARS", 100)
    f = _lit_file(tmp_path, "literature_search_a.md", FOUNDATION, T0)
    tools = make_tools(tmp_path,
                       _registry_state({"path": str(f), "campaign_id": 1}))
    assert tools._load_campaign_literature()["text"] == FOUNDATION


def test_no_literature_yields_none(tmp_path):
    tools = make_tools(tmp_path, _registry_state(cid=2))
    assert tools._load_campaign_literature() is None


# ------------------------------------------------- multi-file union

def test_two_files_union_oldest_first(tmp_path):
    old = _lit_file(tmp_path, "literature_search_a.md", FOUNDATION, T0)
    new = _lit_file(tmp_path, "literature_search_b.md", TOPUP, T0 + 100)
    state = _registry_state({"path": str(new), "campaign_id": 1},
                            {"path": str(old), "campaign_id": 1})
    tools = make_tools(tmp_path, state)
    lit = tools._load_campaign_literature()
    assert lit["n_files"] == 2
    assert lit["files"] == [old.name, new.name]
    i_found = lit["text"].index("Foundational corpus")
    i_top = lit["text"].index("Narrow top-up")
    assert i_found < i_top                     # foundation leads
    assert "nucleation one-shot" in lit["text"]  # nothing silently dropped


def test_verbatim_duplicate_sections_deduped(tmp_path, capsys):
    dup_q = (_format_lit_question_heading(1, "Same question?")
             + "\n\n## ESTABLISHED IN THIS FIELD\nIdentical answer text.\n")
    a = ("# Literature Search Results (hypothesis_context)\n\n" + dup_q
         + "\n" + _format_lit_question_heading(2, "Only in A?")
         + "\n\nUnique-to-A evidence.\n")
    b = ("# Literature Search Results (hypothesis_context)\n\n" + dup_q
         + "\n" + _format_lit_question_heading(2, "Only in B?")
         + "\n\nUnique-to-B evidence.\n")
    fa = _lit_file(tmp_path, "literature_search_a.md", a, T0)
    fb = _lit_file(tmp_path, "literature_search_b.md", b, T0 + 100)
    state = _registry_state({"path": str(fa), "campaign_id": 1},
                            {"path": str(fb), "campaign_id": 1})
    tools = make_tools(tmp_path, state)
    lit = tools._load_campaign_literature()
    assert lit["text"].count("Identical answer text.") == 1
    assert "Unique-to-A" in lit["text"] and "Unique-to-B" in lit["text"]
    assert "duplicate section" in capsys.readouterr().out


def test_over_budget_drops_whole_sections_and_logs(tmp_path, monkeypatch,
                                                   capsys):
    monkeypatch.setattr(ot, "_LIT_AUTOLOAD_MAX_CHARS",
                        len(FOUNDATION) + 50)
    old = _lit_file(tmp_path, "literature_search_a.md", FOUNDATION, T0)
    new = _lit_file(tmp_path, "literature_search_b.md", TOPUP, T0 + 100)
    state = _registry_state({"path": str(old), "campaign_id": 1},
                            {"path": str(new), "campaign_id": 1})
    tools = make_tools(tmp_path, state)
    lit = tools._load_campaign_literature()
    # the older foundation survives whole; the top-up is dropped whole
    assert "Foundational corpus" in lit["text"]
    assert "nucleation one-shot" in lit["text"]
    assert "Narrow top-up" not in lit["text"]
    assert lit["dropped"]                      # reported, not silent
    out = capsys.readouterr().out
    assert "Literature budget" in out and "dropped" in out
    # never mid-section truncation: kept text holds only complete sections
    assert lit["text"].rstrip().endswith("Analogy evidence.")


# ------------------------------------------------- heading contract

def test_split_round_trip_and_writer_lockstep():
    sections = OrchestratorTools._split_literature_sections(FOUNDATION)
    assert "".join(chunk for _q, chunk in sections) == FOUNDATION
    questions = [q for q, _c in sections if q is not None]
    assert questions == ["What is known about defect kinetics?",
                         "Capture a state that exists only under drive?"]
    # a single-question corpus has no headings: one chunk, question=None
    assert OrchestratorTools._split_literature_sections(TOPUP) == [
        (None, TOPUP)]


# ------------------------------------------------- section refs

def test_section_ref_resolves_one_question(tmp_path):
    f = _lit_file(tmp_path, "literature_search_a.md", FOUNDATION, T0)
    text = OrchestratorTools._resolve_context_text(f"{f}#q2")
    assert text.startswith("# Question 2:")
    assert "nucleation one-shot" in text
    assert "Foundational corpus" not in text


def test_section_ref_mixes_with_paths_comma_separated(tmp_path):
    fa = _lit_file(tmp_path, "literature_search_a.md", FOUNDATION, T0)
    fb = _lit_file(tmp_path, "literature_search_b.md", TOPUP, T0 + 100)
    text = OrchestratorTools._resolve_context_text(f"{fa}#q1,{fb}")
    assert "Foundational corpus" in text and "Narrow top-up" in text
    assert "nucleation one-shot" not in text   # q2 not selected
    paths = OrchestratorTools._context_file_paths(f"{fa}#q1,{fb}")
    assert paths == [str(fa.resolve()), str(fb.resolve())]  # ref → base file


def test_headingless_file_q1_resolves_to_whole_body(tmp_path):
    # live: the model selected '<file>#q1' for a single-question corpus
    # written without question headings — that must mean "its only
    # section", not an empty skip
    f = _lit_file(tmp_path, "literature_search_b.md", TOPUP, T0)
    assert OrchestratorTools._resolve_context_text(f"{f}#q1") == TOPUP
    assert OrchestratorTools._resolve_context_text(f"{f}#q2") is None


def test_missing_section_ref_never_becomes_raw_text(tmp_path, capsys):
    f = _lit_file(tmp_path, "literature_search_a.md", FOUNDATION, T0)
    text = OrchestratorTools._resolve_context_text(f"{f}#q9")
    assert text is None                        # skipped, not the ref string
    assert "does not contain" in capsys.readouterr().out


def test_raw_text_and_plain_paths_unchanged(tmp_path):
    f = _lit_file(tmp_path, "literature_search_a.md", FOUNDATION, T0)
    assert OrchestratorTools._resolve_context_text(str(f)) == FOUNDATION
    raw = "Yield was 12%, precipitation observed"
    assert OrchestratorTools._resolve_context_text(raw) == raw
    assert OrchestratorTools._context_file_paths(raw) == []


# ------------------------------------------------- registry stamping

def test_record_literature_file_stamps_label_and_questions(tmp_path):
    f = _lit_file(tmp_path, "literature_search_a.md", FOUNDATION, T0)
    state = _registry_state()
    tools = make_tools(tmp_path, state)
    tools._record_literature_file(f, label="hypothesis_context+cross_domain",
                                  questions=["Q one", "Q two"])
    entry = state["campaign_literature"][-1]
    assert entry["label"] == "hypothesis_context+cross_domain"
    assert entry["questions"] == ["Q one", "Q two"]


def test_record_literature_file_without_metadata_keeps_legacy_shape(tmp_path):
    f = _lit_file(tmp_path, "literature_search_a.md", FOUNDATION, T0)
    state = _registry_state()
    tools = make_tools(tmp_path, state)
    tools._record_literature_file(f)
    assert state["campaign_literature"][-1] == {
        "path": str(f.resolve()), "campaign_id": 1}


# ------------------------------------------------- index tool

def _registered(tools):
    cap = {}
    tools._register_tool = (
        lambda func, name, description, parameters, required=None:
        cap.update({name: func}))
    ot.OrchestratorTools._register_all_tools(tools)
    return cap


def _full_tools(tmp_path, state):
    tools = make_tools(tmp_path, state)
    tools.orch.planner.model = None
    tools.orch.lit_agent = None
    return tools


def test_list_literature_searches_index(tmp_path):
    old = _lit_file(tmp_path, "literature_search_a.md", FOUNDATION, T0)
    new = _lit_file(tmp_path, "literature_search_b.md", TOPUP, T0 + 100)
    state = _registry_state(
        {"path": str(old), "campaign_id": 1,
         "label": "hypothesis_context+cross_domain",
         "questions": ["What is known about defect kinetics?",
                       "Capture a state that exists only under drive?"]},
        {"path": str(new), "campaign_id": 1, "label": "hypothesis_context",
         "questions": ["Which annealing schedule?"]},
    )
    tools = _full_tools(tmp_path, state)
    out = json.loads(_registered(tools)["list_literature_searches"]())
    assert out["status"] == "success" and out["count"] == 2
    first, second = out["files"]
    assert first["path"] == str(old)           # oldest first
    qs = [s["question"] for s in first["sections"]]
    assert qs == ["What is known about defect kinetics?",
                  "Capture a state that exists only under drive?"]
    assert all(s["answer_preview"] for s in first["sections"])
    assert first["sections"][1]["section_ref"] == f"{old}#q2"
    # single-question file: no heading in the file, question from registry,
    # and its whole body addressable as '#q1' (uniform selection syntax)
    assert second["sections"][0]["question"] == "Which annealing schedule?"
    assert second["sections"][0]["section_ref"] == f"{new}#q1"
    assert "auto-load" in out["hint"].lower() or "Omit" in out["hint"]


def test_list_literature_searches_empty_campaign(tmp_path):
    tools = _full_tools(tmp_path, _registry_state(cid=2))
    out = json.loads(_registered(tools)["list_literature_searches"]())
    assert out["status"] == "success" and out["count"] == 0
