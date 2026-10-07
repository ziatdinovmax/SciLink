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


def test_a_swarm_items_tag_is_read_off_the_line_and_the_coordinators_lines_are_visible():
    """A swarm item's lines carry "[label] "; the kind is read from what
    follows, the label kept on the line. The coordinators' own lines (the
    launch, a hold, the guard, a rerun, the heartbeat, an item finished)
    are the visible fan-out kind, not plain."""
    from scilink.ui.narration import split_worker_tag
    c = LineClassifier()
    th = c.push("[XRD 300 K]   💭 I'll examine the data first")
    assert (th.kind, th.worker, th.text, th.verbose) == ("thought", "XRD 300 K", "💭 I'll examine the data first", False)
    cont = c.push("[XRD 300 K]      and then fit")
    assert (cont.kind, cont.worker, cont.text) == ("thought", "XRD 300 K", "and then fit")
    tool = c.push("[purity plan]   🔧 Calling tool: generate_initial_plan")
    assert (tool.kind, tool.worker) == ("tool_call", "purity plan")
    cand = c.push("[fit A] [cand_02] Verification 1/7")
    assert (cand.kind, cand.worker, cand.text) == ("candidate", "fit A", "[cand_02] Verification 1/7")
    plain = c.push("[XRD 300 K] 2026-10-07 10:04:27,563 - INFO -   Points: 3000")
    assert (plain.kind, plain.worker, plain.verbose) == ("plain", "XRD 300 K", True)
    assert c.push("  💭 the meta's own thought").worker is None
    from scilink.agents.meta_agent.fanout import coordinator_line
    for text in ("  🐝 swarm_x: 2 item(s), up to 2 at a time",
                 "  ⏸  holding branch 'purity plan' for memory headroom",
                 "  🧯 free memory is low (0.10 GB) — cancelling 'heavy'",
                 "  🔁 running 'heavy' again, alone",
                 "  ⏳ 2 swarm item(s) still running ...",
                 "  ✅ swarm item finished: XRD 300 K (success)",
                 "  ⛔ not started: big cube — needs about 40.0 GB",
                 "  ⏱️  swarm item 'slow one' exceeded its wall-clock budget (0s)"):
        ln = c.push(coordinator_line(text))              # as the coordinators print them
        assert (ln.kind, ln.verbose) == ("fanout", False), text
    assert c.push("  ⏳ Waiting for orchestrator response ...").kind == "waiting"
    assert c.push("  ✅ Sandbox approval already granted this session").kind == "bookkeeping"
    assert split_worker_tag("[cand_01] x") == (None, "[cand_01] x")       # a candidate tag is not a worker


def test_the_writers_label_is_always_one_the_reader_accepts():
    """Round trip: whatever label the model wrote, ``worker_tag`` makes a tag
    that ``split_worker_tag`` reads back (review of #774, item 1)."""
    from scilink.ui.narration import WORKER_TAG_MAX, split_worker_tag, worker_tag
    long = "Raman A7 anatase fraction vs annealing temperature series, second pass"
    for label in (long, "XRD [300 K]", "cand_02", "cand-7 fit", "", "   ", "a\nb", "x" * 200):
        tag = worker_tag(label)
        assert 1 <= len(tag) <= WORKER_TAG_MAX and "[" not in tag and "]" not in tag and "\n" not in tag
        got, rest = split_worker_tag(f"[{tag}] 💭 hello")
        assert (got, rest) == (tag, "💭 hello"), label
    assert worker_tag("XRD [300 K]") == "XRD (300 K)"
    assert worker_tag(long).endswith("…") and len(worker_tag(long)) == WORKER_TAG_MAX
    assert worker_tag("cand_02") == "item cand_02" and worker_tag("") == "item"
    c = LineClassifier()
    ln = c.push(f"[{worker_tag(long)}]   💭 a thought")
    assert (ln.kind, ln.verbose) == ("thought", False) and ln.worker == worker_tag(long)


def test_coordinator_lines_are_known_by_their_mark_not_their_wording():
    """Item 4 (round 2): the coordinators mark their own lines
    (``fanout.coordinator_line``); the reader shows a marked line and nothing
    by its emoji, so an agent's "⏱ Nobody answered…" stays plain and a
    reworded coordinator print needs no change here."""
    from scilink.agents.meta_agent.fanout import coordinator_line
    from scilink.ui.vocabulary import COORDINATOR_MARK
    c = LineClassifier()
    agents = ("  ⏱ Nobody answered the review in time; the plan is marked unattended",
              "    🔁 Diagram render error (attempt 2): …",
              "  ⏱️ Deep literature searches take a few minutes",
              "  ⏳ Waiting for orchestrator response ...",
              "  ✅ Sandbox approval already granted this session",
              "  ⛔ Fan-out over this dataset set was already declined")
    for text in agents:
        ln = c.push(text)
        assert ln.kind != "fanout" and ln.verbose, text
    coordinators = ("  🐝 swarm_x: 2 item(s), up to 2 at a time",
                    "  ⏸  holding branch 'purity plan' for memory headroom (needs ~0.5 GB)",
                    "  🧯 free memory is low (0.10 GB) — cancelling 'heavy'",
                    "  🔁 running 'heavy' again, alone",
                    "  ⛔ not started: big cube — needs about 40.0 GB",
                    "  ⛔ Fan-out declined: 'x' is a RAW instrument container",      # a refusal, whatever its words
                    "  ⏱️  swarm item 'slow one' exceeded its wall-clock budget (0s)",
                    "  ⏳ 2 swarm item(s) still running ...",
                    "  ✅ swarm item finished: XRD 300 K (success)",
                    "  🔁 Resuming fan-out branches in their original sessions")
    for text in coordinators:
        raw = c.push(text)
        assert raw.kind != "fanout", text                        # the same words unmarked: an agent's line
        marked = coordinator_line(text)
        assert marked.startswith("  " + COORDINATOR_MARK) and COORDINATOR_MARK not in marked[3:]
        ln = c.push(marked)
        assert (ln.kind, ln.verbose, ln.text) == ("fanout", False, text.strip()), text
    assert coordinator_line("🐝 SWARM — 2 item(s)") == COORDINATOR_MARK + "🐝 SWARM — 2 item(s)"
    assert current_activity(coordinator_line("  🐝 swarm_x: 2 item(s), up to 2 at a time")) == "Swarm · 2 items"


def test_every_coordinator_print_goes_through_the_marking_helper():
    """The guard against drift: a new or reworded coordinator line printed
    raw would be hidden again. No ``print`` of a line that opens with a
    coordinator emoji is left in the coordinator modules; they all go
    through ``_cprint`` (the meta's tools included)."""
    import re
    from pathlib import Path
    import scilink.agents.meta_agent as pkg
    root = Path(pkg.__file__).parent
    raw = re.compile(r'(?<![\w.])print\((?=f?"\s*(?:🐝|⏸|🧯|🔁|⛔|⏱|⏳|✅|🔀))')
    offenders = []
    for name in ("fanout.py", "swarm.py", "meta_orchestrator_tools.py"):
        for i, line in enumerate((root / name).read_text(encoding="utf-8").splitlines(), 1):
            if raw.search(line):
                offenders.append(f"{name}:{i}: {line.strip()[:80]}")
    assert not offenders, "\n".join(offenders)


def test_thought_and_answer_continuation_is_kept_per_worker():
    """Item 5: item A's open thought never swallows item B's indented line,
    and the meta's own stream is a writer of its own."""
    c = LineClassifier()
    assert c.push("[A]   💭 A thinks").kind == "thought"
    b = c.push("[B]      an indented line of B")
    assert (b.kind, b.worker) == ("plain", "B")
    a2 = c.push("[A]      and A continues")
    assert (a2.kind, a2.worker, a2.text) == ("thought", "A", "and A continues")
    meta = c.push("     the meta's indented line")
    assert (meta.kind, meta.worker) == ("plain", None)
    assert c.push("[A] 🤖 Specialist:").kind == "answer_header"
    assert c.push("[B] some result text of B").kind == "plain"           # not A's answer body
    assert c.push("[A] the body of A's answer").kind == "answer_body"
