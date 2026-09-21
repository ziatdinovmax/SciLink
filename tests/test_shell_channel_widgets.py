"""The question widgets follow the presenter's vocabulary and the
"Enter accepts" convention, read through a prompt_toolkit pipe input."""

import io

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from contextlib import ExitStack
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from scilink.cli.shell.channel import AutoAcceptChannel, Widgets, ask_secret, enter_hint
from scilink.hitl import FeedbackRequest


def _widgets(keys: str):
    stack = ExitStack()
    pipe = stack.enter_context(create_pipe_input())
    pipe.send_text(keys)
    session = PromptSession(input=pipe, output=DummyOutput())
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100, highlight=False)
    return Widgets(console, session, session_dir="/tmp/s"), buf, stack


GENERIC = {"widget": "generic", "prompt": "Review the plan.",
           "labels": {"input": "Your plan feedback (optional):",
                      "submit": "Request changes", "accept": "Approve plan"},
           "preview_images": [], "code_files": [], "candidate_captions": {}}


def test_generic_enter_accepts():
    w, buf, stack = _widgets("\r")
    with stack:
        assert w.ask(GENERIC) == ""
    out = buf.getvalue()
    assert "Review the plan." in out
    assert "Your plan feedback (optional):" in out
    assert "Enter = Approve plan" in out


def test_generic_text_is_the_feedback():
    w, _, stack = _widgets("use a wider window\r")
    with stack:
        assert w.ask(GENERIC) == "use a wider window"


def test_code_review_shows_files():
    q = dict(GENERIC, widget="code_review",
             labels={"input": "Your code feedback (optional):", "submit": "Request changes",
                     "accept": "Approve code"},
             code_files=[{"name": "fit.py", "content": "import numpy as np\n"}])
    w, buf, stack = _widgets("\r")
    with stack:
        assert w.ask(q) == ""
    out = buf.getvalue()
    assert "fit.py" in out and "import numpy" in out and "Enter = Approve code" in out


def test_keep_revert():
    q = {"widget": "keep_revert", "prompt": "", "labels": {"keep": "Keep user-guided fit",
                                                            "revert": "Revert to original fit"},
         "preview_images": [], "code_files": [], "candidate_captions": {}}
    w, buf, stack = _widgets("\x1b[A\r")          # up from the default (revert) to keep
    with stack:
        assert w.ask(q) == "keep"
    w, _, stack = _widgets("\r")                    # Enter = the default, revert
    with stack:
        assert w.ask(q) == ""
    w, _, stack = _widgets("k")                      # the letter jumps
    with stack:
        assert w.ask(q) == "keep"


def test_fanout_confirm():
    q = {"widget": "fanout_confirm", "prompt": "",
         "labels": {"confirm": "🔀 Launch parallel analysis", "cancel": "Cancel"},
         "fanout": {"verdict": "complementary", "join_axis": "temperature",
                    "rationale": "same sample", "branches": ["a.csv", "b.csv"]},
         "preview_images": [], "code_files": [], "candidate_captions": {}}
    w, buf, stack = _widgets("\x1b[B\r")          # down to "launch", Enter
    with stack:
        assert w.ask(q) == "y"
    out = buf.getvalue()
    assert "Complementarity" in out and "a.csv" in out
    w, _, stack = _widgets("\r")                    # Enter = the default, cancel
    with stack:
        assert w.ask(q) == "no"


def test_bestofn_pick_and_judge_default():
    q = {"widget": "bestofn", "prompt": "",
         "labels": {"select": "Select the candidate to lock:", "use": "Use selected",
                    "accept": "Accept judge's pick (Candidate 2)"},
         "candidates": [{"idx": 1, "label": "Candidate 1 — chi2=0.1"},
                        {"idx": 2, "label": "Candidate 2 — chi2=0.04"}],
         "judge_pick": 2, "preview_images": ["figs/bestofn_candidate_1_review.png"],
         "code_files": [], "candidate_captions": {"bestofn_candidate_1_review.png": "Candidate 1"}}
    w, buf, stack = _widgets("\x1b[A\r")          # highlight starts on the judge's pick (2); up -> 1
    with stack:
        assert w.ask(q) == "1"
    assert "Select the candidate to lock:" in buf.getvalue()
    w, _, stack = _widgets("\r")                    # Enter on the judge's pick = accept ("")
    with stack:
        assert w.ask(q) == ""
    w, _, stack = _widgets("1")                      # a digit jumps
    with stack:
        assert w.ask(q) == "1"


def test_picker_escape_stops_the_turn():
    import pytest as _pytest
    q = {"widget": "keep_revert", "prompt": "", "labels": {"keep": "Keep", "revert": "Revert"},
         "preview_images": [], "code_files": [], "candidate_captions": {}}
    w, _, stack = _widgets("\x1b")
    with stack, _pytest.raises(KeyboardInterrupt):
        w.ask(q)


def test_enter_hint_and_auto_accept():
    assert enter_hint({"accept": "Approve plan"}) == "Enter = Approve plan"
    assert enter_hint({}) == ""
    assert AutoAcceptChannel().ask(FeedbackRequest(prompt="x", default="keep")) == "keep"


def test_ask_secret_reads_and_defaults():
    stack = ExitStack()
    pipe = stack.enter_context(create_pipe_input())
    pipe.send_text("sk-123\r")
    session = PromptSession(input=pipe, output=DummyOutput())
    with stack:
        assert ask_secret(session, "key: ", secret=True) == "sk-123"
    pipe2 = stack.enter_context(create_pipe_input())
    pipe2.send_text("\r")
    session2 = PromptSession(input=pipe2, output=DummyOutput())
    with stack:
        assert ask_secret(session2, "key: ", default="dflt") == "dflt"


def test_context_under_review_is_shown():
    q = dict(GENERIC, context_display="📋 PROPOSED PLAN\nStep 1: threshold\nStep 2: label grains",
             prompt="Review the plan and press Enter to approve:")
    w, buf, stack = _widgets("\r")
    with stack:
        assert w.ask(q) == ""
    out = buf.getvalue()
    assert "PROPOSED PLAN" in out and "Step 2: label grains" in out
    assert "Review the plan and press Enter to approve:" in out
    assert out.index("PROPOSED PLAN") < out.index("Review the plan")   # context first, then the ask


def test_long_context_is_capped_with_a_pointer():
    q = dict(GENERIC, context_display="\n".join(f"line {i}" for i in range(200)))
    w, buf, stack = _widgets("\r")
    with stack:
        w.ask(q)
    out = buf.getvalue()
    assert "line 199" in out and "line 80 " in out and "line 79 " not in out   # last 120 lines kept
    assert "80 earlier lines (Ctrl+O shows them)" in out


def test_ctrl_o_at_a_question_shows_earlier_lines_and_keeps_the_prompt():
    q = dict(GENERIC, context_display="\n".join(f"line {i}" for i in range(200)))
    w, buf, stack = _widgets("\x0f\r")          # Ctrl+O, then Enter
    with stack:
        assert w.ask(q) == ""                     # accepted by Enter, not by Ctrl+O
    out = buf.getvalue()
    assert "80 earlier lines" in out and "line 0" in out and "line 79" in out


def test_ctrl_o_at_a_picker_shows_earlier_lines_and_keeps_the_picker():
    q = {"widget": "bestofn", "prompt": "",
         "labels": {"select": "Select the candidate to lock:", "use": "Use selected",
                    "accept": "Accept judge's pick (Candidate 2)"},
         "candidates": [{"idx": 1, "label": "Candidate 1"}, {"idx": 2, "label": "Candidate 2"}],
         "judge_pick": 2, "preview_images": [], "code_files": [], "candidate_captions": {},
         "context_display": "\n".join(f"line {i}" for i in range(150))}
    w, buf, stack = _widgets("\x0f\x1b[A\r")     # Ctrl+O, up, Enter
    with stack:
        assert w.ask(q) == "1"                    # the picker survived Ctrl+O
    out = buf.getvalue()
    assert "30 earlier lines" in out and "line 0" in out and "line 29" in out


def test_notice_callout_and_revert_repair_hint():
    """The plan gate after an auto-correction: the notice beside the answer
    and the one-word reply that restores the plan as authored (the web
    panel's callout and "Revert auto-correction" button)."""
    q = dict(GENERIC, notice={"title": "Auto-corrected before review (2 changes)",
                              "lines": ["Step 3: 950 C -> 850 C", "Step 5: added a control"]},
             labels=dict(GENERIC["labels"], revert_repair="Revert auto-correction"))
    w, buf, stack = _widgets("revert\r")
    with stack:
        assert w.ask(q) == "revert"
    out = buf.getvalue()
    assert "Auto-corrected before review (2 changes)" in out and "950 C -> 850 C" in out
    assert "revert = Revert auto-correction" in out


def test_reopen_gate_offers_adopt_with_changes():
    """The reopen gate is the keep/revert widget with a third reply: adopt
    the agent's revision with changes (free text)."""
    q = {"widget": "keep_revert", "prompt": "",
         "labels": {"keep": "Adopt the revision", "revert": "Keep my approved plan",
                    "input": "Adopt it with changes (optional):", "submit": "Adopt with changes"},
         "notice": {"title": "The agent proposes to revise a plan you approved",
                    "lines": ["Reason given: the anneal step exceeds the furnace limit"]},
         "preview_images": [], "code_files": [], "candidate_captions": {}}
    w, buf, stack = _widgets("\r")                       # Enter on the default: keep the approved plan
    with stack:
        assert w.ask(q) == ""
    assert "proposes to revise" in buf.getvalue()
    w, _, stack = _widgets("3use a 900 C anneal\r")     # the third option, then the changes
    with stack:
        assert w.ask(q) == "use a 900 C anneal"
    w, _, stack = _widgets("3\r")                        # chose it, typed nothing: adopt as-is
    with stack:
        assert w.ask(q) == "keep"
