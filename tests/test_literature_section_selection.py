"""Targeted literature selection for documents, and the decline index.

write_technical_document could only take the campaign's literature
all-or-nothing, and the agent chose knowing nothing about what the corpus
covered. These cover the three branches:

  literature_context set   -> exactly those sections
  use_literature=True      -> whole campaign corpus (unchanged default)
  use_literature=False     -> no literature, but the index comes back

The hard case throughout is a campaign whose literature accumulated across
delegations: each search writes its own file, EVERY file restarts at
'# Question 1', and under the meta the files can share a basename. Refs
derived from list position instead of the file's own heading numbers
resolve to the wrong section exactly there.
"""
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scilink.agents.planning_agents.orchestrator_tools import OrchestratorTools

RESULTS = {}


def _tools(root, entries, campaign_id=1):
    t = OrchestratorTools.__new__(OrchestratorTools)
    t._prestate_lit = []
    t.orch = types.SimpleNamespace(base_dir=str(root), planner=None)
    state = {"campaign_id": campaign_id, "campaign_literature": entries}
    t._planner_state = lambda: state
    return t


def _write_search(path, marker, questions):
    """A saved search: title chunk, then '# Question N' sections from 1."""
    body = "# Literature Search Results\n\n"
    for i, q in enumerate(questions, start=1):
        body += f"# Question {i}: {q}\n\n{marker}-BODY-{i} answer text here.\n\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return {"path": str(path), "campaign_id": 1, "label": "hypothesis_context",
            "questions": questions}


def _fixture(root):
    """Two searches, two delegations, SAME basename — the meta's shape."""
    name = "literature_search_hypothesis_context.md"
    a = _write_search(root / "delegations" / "01_first" / name, "AAA",
                      ["What are the timescales of Pd nucleation?",
                       "How does BO time an intervention?"])
    b = _write_search(root / "delegations" / "02_second" / name, "BBB",
                      ["What throughput do autonomous colloidal campaigns reach?",
                       "How does carryover degrade reproducibility?"])
    return _tools(root, [a, b])


def test_index_spans_delegations_and_shared_basenames():
    root = Path(tempfile.mkdtemp(prefix="litsel_"))
    try:
        topics = _fixture(root)._campaign_literature_topics()
        assert len(topics) == 4, f"expected 4 sections, got {len(topics)}"
        refs = [t["section_ref"] for t in topics]
        assert len(set(refs)) == 4, "shared basenames collapsed the index"
        # Both files restart at q1, so #q1 must appear twice under
        # different paths rather than one shadowing the other.
        assert sum(r.endswith("#q1") for r in refs) == 2
        delegs = {r.split("delegations/")[1].split("/")[0] for r in refs}
        assert delegs == {"01_first", "02_second"}, delegs
    finally:
        shutil.rmtree(root)


def test_every_ref_resolves_to_its_own_files_own_section():
    root = Path(tempfile.mkdtemp(prefix="litsel_"))
    try:
        for t in _fixture(root)._campaign_literature_topics():
            text = OrchestratorTools._resolve_context_text(t["section_ref"])
            n = t["section_ref"].rsplit("#q", 1)[1]
            mine = "AAA" if "01_first" in t["section_ref"] else "BBB"
            other = "BBB" if mine == "AAA" else "AAA"
            assert f"{mine}-BODY-{n}" in text, f"wrong section: {t['section_ref']}"
            assert other not in text, f"cross-file bleed: {t['section_ref']}"
    finally:
        shutil.rmtree(root)


def test_selection_is_a_strict_subset_and_composes_across_delegations():
    root = Path(tempfile.mkdtemp(prefix="litsel_"))
    try:
        tools = _fixture(root)
        topics = tools._campaign_literature_topics()
        whole = tools._load_campaign_literature()["text"]

        thr = next(t for t in topics if "throughput" in t["question"].lower())
        one = OrchestratorTools._resolve_context_text(thr["section_ref"])
        assert "BBB-BODY-1" in one and "BBB-BODY-2" not in one
        assert "AAA" not in one
        assert len(one) < len(whole), "selection did not narrow the corpus"

        # A document may need one section from each search.
        pair = OrchestratorTools._resolve_context_text(
            f"{topics[0]['section_ref']},{thr['section_ref']}")
        assert "AAA-BODY-1" in pair and "BBB-BODY-1" in pair
        assert len(pair) > len(one)
    finally:
        shutil.rmtree(root)


def test_default_path_still_loads_the_whole_corpus():
    """use_literature=True with no selection must be unchanged."""
    root = Path(tempfile.mkdtemp(prefix="litsel_"))
    try:
        lit = _fixture(root)._load_campaign_literature()
        assert lit["n_files"] == 2
        for marker in ("AAA-BODY-1", "AAA-BODY-2", "BBB-BODY-1", "BBB-BODY-2"):
            assert marker in lit["text"], f"{marker} missing from auto-load"
    finally:
        shutil.rmtree(root)


def test_decline_index_carries_questions_not_answers():
    """The whole point of reporting a decline is that it stays cheap."""
    root = Path(tempfile.mkdtemp(prefix="litsel_"))
    try:
        topics = _fixture(root)._campaign_literature_topics()
        blob = json.dumps(topics)
        for marker in ("AAA-BODY", "BBB-BODY"):
            assert marker not in blob, "answer bodies leaked into the index"
        assert all(t["question"] and t["file"] for t in topics)
    finally:
        shutil.rmtree(root)


def test_campaign_without_literature_yields_no_index():
    """No literature must produce no decline block, not an empty one."""
    root = Path(tempfile.mkdtemp(prefix="litsel_"))
    try:
        assert _tools(root, [], campaign_id=7)._campaign_literature_topics() == []
    finally:
        shutil.rmtree(root)


def test_headingless_single_question_file_is_addressable_as_q1():
    """A one-question search is saved WITHOUT headings; the resolver has a
    special case for '#q1' and the index must agree with it."""
    root = Path(tempfile.mkdtemp(prefix="litsel_"))
    try:
        f = root / "delegations" / "01_only" / "literature_search_solo.md"
        f.parent.mkdir(parents=True)
        f.write_text("Solo corpus body with no question headings.\n")
        entry = {"path": str(f), "campaign_id": 1, "label": "solo",
                 "questions": ["What is the solo question?"]}
        topics = _tools(root, [entry])._campaign_literature_topics()
        assert len(topics) == 1 and topics[0]["section_ref"].endswith("#q1")
        assert topics[0]["question"] == "What is the solo question?"
        text = OrchestratorTools._resolve_context_text(topics[0]["section_ref"])
        assert "Solo corpus body" in text, "index advertised an unresolvable ref"
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
