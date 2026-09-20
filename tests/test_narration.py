"""The narration reader — the Python side of a contract the frontend's
``webui/src/narration.ts`` shares through ``tests/fixtures/
narration_activity.json`` (``npm run check:vocabulary`` runs the same cases
through the TS twin)."""

import json
from pathlib import Path

import pytest

from scilink.ui import vocabulary
from scilink.ui.narration import (LineClassifier, classify, current_activity,
                                  strip_ansi)

FIXTURE = Path(__file__).parent / "fixtures" / "narration_activity.json"
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_current_activity_matches_fixture(case):
    assert current_activity(case["log"]) == case["expected"]


def test_fixture_labels_come_from_the_vocabulary():
    """The fixed (non-templated) labels the fixture expects are vocabulary
    strings, so the words cannot drift from what the web UI shows."""
    fixed = {v for v in vocabulary.ACTIVITY_LABELS.values() if "{" not in v}
    seen = {c["expected"] for c in CASES if c["expected"] in fixed}
    assert {"Writing response…", "Waiting for your input…", "Wrapping up…"} <= seen


def test_strip_ansi():
    assert strip_ansi("\x1b[1;96m🤖 Agent:\x1b[0m") == "🤖 Agent:"


def test_classifier_kinds_and_visibility():
    c = LineClassifier()
    tool = c.push("  🔧 Calling tool: delegate_to_analysis")
    assert (tool.kind, tool.verbose) == ("tool_call", False)
    wait = c.push("  ⏳ Waiting for meta-orchestrator response ...")
    assert (wait.kind, wait.verbose) == ("waiting", True)
    th = c.push("  \x1b[2;3;36m💭 first line\x1b[0m")
    assert (th.kind, th.specialist, th.verbose) == ("thought", False, False)
    cont = c.push("     continued reasoning")
    assert (cont.kind, cont.text) == ("thought", "continued reasoning")
    plain = c.push("some progress line")
    assert (plain.kind, plain.verbose) == ("plain", True)
    warn = c.push("  ⚠️  Context window getting full")
    assert (warn.kind, warn.verbose) == ("warning", False)
    hand = c.push("  🧪 Delegating to analysis specialist: task")
    assert (hand.kind, hand.verbose) == ("handoff", False)
    sim = c.push("  ⚛️ Delegating to simulation specialist: task")
    assert sim.kind == "handoff"
    cand = c.push("[cand_02] Verification 1/7")
    assert (cand.kind, cand.verbose) == ("candidate", True)
    rule = c.push("=" * 60)
    assert (rule.kind, rule.verbose) == ("rule", True)


def test_classifier_specialist_marks_and_answer_body():
    mark = vocabulary.THOUGHT_MARK
    c = LineClassifier()
    th = c.push(f"  💭{mark} specialist thought")
    assert (th.kind, th.specialist, th.text) == ("thought", True, "💭 specialist thought")
    head = c.push(f"\x1b[1;33m🤖{mark} Analysis specialist:\x1b[0m")
    assert (head.kind, head.specialist) == ("answer_header", True)
    body = c.push("The fit converged.")
    assert (body.kind, body.specialist, body.verbose) == ("answer_body", True, False)
    blank = c.push("")
    assert blank.kind == "answer_body"
    # A recognisable kind closes the answer.
    tool = c.push("  🔧 Calling tool: summarize_session_state")
    assert tool.kind == "tool_call"
    assert c.push("after the tool").kind == "plain"


def test_classify_convenience_is_stateful_per_call():
    kinds = [ln.kind for ln in classify("  💭 a\n     b\nc")]
    assert kinds == ["thought", "thought", "plain"]


def test_bookkeeping_lines_close_a_specialist_answer():
    """After a delegated specialist's 🤖 answer, its housekeeping lines
    (mode change, auto-checkpoint) are not part of the answer and stay verbose."""
    mark = vocabulary.THOUGHT_MARK
    c = LineClassifier()
    c.push(f"🤖{mark} Analysis specialist:")
    assert c.push("The sample is a grain mosaic.").kind == "answer_body"
    m = c.push("  🔄 Analysis mode changed: autopilot → co-pilot")
    assert (m.kind, m.verbose) == ("bookkeeping", True)
    assert c.push("     Human feedback enabled: True").kind == "bookkeeping"
    assert c.push("      ✅ Auto-checkpoint saved").kind == "bookkeeping"
    assert c.push("plain after").kind == "plain"
