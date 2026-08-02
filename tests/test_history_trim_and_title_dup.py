"""Two live failures from one long session.

1. History trimming sliced a tool_use away from its tool_result and the
   repair pass ran BEFORE the trim, so it cleaned an already-valid history
   and left the fresh damage. Bedrock rejected the turn with "Expected
   toolResult blocks at messages.10" — index 10 being exactly the splice
   boundary of `history[:10] + history[-90:]`.

2. Revising a document stacked its title: the assembler always prepends
   `# {title}`, while the revision contract tells the model to return the
   whole document verbatim, so it returns the title as a section too. Six
   copies had accumulated, readable as the title's own edit history.
"""

import re
from pathlib import Path

import pytest

from scilink.agents.planning_agents.planning_rag import document_to_markdown
from scilink.utils.tool_media import repair_dangling_tool_calls

PLANNING = Path("scilink/agents/planning_agents/planning_orchestrator.py")
META = Path("scilink/agents/meta_agent/meta_orchestrator.py")


# ── 1 · repair must run AFTER the trim ───────────────────────────────

@pytest.mark.parametrize("src", [PLANNING, META])
def test_repair_follows_every_trim(src):
    """Repairing before the trim fixes a history that was already fine."""
    text = src.read_text()
    trims = [m.start() for m in
             re.finditer(r"self\.messages = \[system_msg\] \+ recent_msgs", text)]
    assert trims, f"no trim site found in {src.name}"
    for pos in trims:
        after = text[pos:pos + 700]
        assert "repair_dangling_tool_calls(self.messages)" in after, (
            f"{src.name}: a trim at offset {pos} is not followed by a repair")


@pytest.mark.parametrize("src", [PLANNING, META])
def test_the_reason_is_recorded_at_the_call_site(src):
    assert "AFTER the trim, not before" in src.read_text()


def test_repair_fixes_exactly_the_damage_a_trim_causes():
    """The splice keeps head+tail, so it can orphan a result at the head
    boundary and strand a call at the tail boundary. Both must survive."""
    history = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "tool_calls": [
            {"id": "call_A", "type": "function",
             "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_A", "content": "A done"},
        {"role": "assistant", "tool_calls": [
            {"id": "call_B", "type": "function",
             "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_B", "content": "B done"},
    ]
    # the exact shape a head/tail splice produces: call_A's result dropped,
    # call_B's result kept without its call
    sliced = [history[0], history[1], history[4]]
    fixed = repair_dangling_tool_calls(sliced)

    pending = []
    for m in fixed:
        for tc in (m.get("tool_calls") or []):
            pending.append(tc["id"])
        if m.get("role") == "tool":
            cid = m.get("tool_call_id")
            assert cid in pending, "orphan tool result survived the repair"
            pending.remove(cid)
    assert not pending, f"unanswered tool_use survived the repair: {pending}"


# ── 2 · the title is written once, however many revisions ────────────

def test_a_title_echoed_as_a_section_is_dropped():
    md = document_to_markdown("My Doc", [{"heading": "My Doc", "body": ""},
                                         {"heading": "Intro", "body": "text"}])
    assert md.count("My Doc") == 1
    assert md.startswith("# My Doc")
    assert "## Intro" in md and "text" in md


def test_successive_revisions_do_not_stack_titles():
    """The live failure: one extra copy per revision, six by the end."""
    title = "Staging CDOC: Mini-CDOC MVPs"
    sections = [{"heading": "Framing", "body": "body text"}]
    doc = document_to_markdown(title, sections)
    for _ in range(5):
        # what a revision returns: the title echoed back, then the content
        returned = [{"heading": title, "body": ""}] + sections
        doc = document_to_markdown(title, returned)
        assert doc.count(title) == 1, doc[:200]


def test_a_real_section_that_merely_starts_with_the_title_is_kept():
    """Only an EMPTY leading echo is dropped — never content."""
    md = document_to_markdown("Roadmap", [
        {"heading": "Roadmap", "body": "This section has real content."},
        {"heading": "Next", "body": "more"}])
    assert "This section has real content." in md
    assert md.count("Roadmap") == 2       # the title, and the kept section


def test_only_a_LEADING_echo_is_dropped():
    """A title-like heading later in the document is the author's choice."""
    md = document_to_markdown("Doc", [{"heading": "Intro", "body": "a"},
                                      {"heading": "Doc", "body": ""}])
    assert md.count("## Doc") == 1


def test_case_and_hash_variation_still_counts_as_an_echo():
    for h in ("my doc", "MY DOC", "## My Doc", "  My Doc  "):
        md = document_to_markdown("My Doc", [{"heading": h, "body": ""},
                                             {"heading": "S", "body": "b"}])
        assert md.count("My Doc") == 1, h


def test_the_contract_tells_the_author_not_to_emit_a_title():
    """Dropping it in the assembler alone leaves the model emitting a copy
    every revision that is then silently swallowed."""
    from scilink.agents.planning_agents.instruct import (
        TECHNICAL_DOCUMENT_REVISION_RULES as R)
    flat = " ".join(R.split())
    assert "Do NOT return the document's TITLE as a section" in flat
