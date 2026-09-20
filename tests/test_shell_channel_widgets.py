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
    w, buf, stack = _widgets("k\r")
    with stack:
        assert w.ask(q) == "keep"
    assert "Keep user-guided fit" in buf.getvalue()
    w, _, stack = _widgets("\r")
    with stack:
        assert w.ask(q) == ""


def test_fanout_confirm():
    q = {"widget": "fanout_confirm", "prompt": "",
         "labels": {"confirm": "🔀 Launch parallel analysis", "cancel": "Cancel"},
         "fanout": {"verdict": "complementary", "join_axis": "temperature",
                    "rationale": "same sample", "branches": ["a.csv", "b.csv"]},
         "preview_images": [], "code_files": [], "candidate_captions": {}}
    w, buf, stack = _widgets("y\r")
    with stack:
        assert w.ask(q) == "y"
    out = buf.getvalue()
    assert "Complementarity" in out and "a.csv" in out and "Launch parallel analysis" in out
    w, _, stack = _widgets("\r")
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
    w, buf, stack = _widgets("1\r")
    with stack:
        assert w.ask(q) == "1"
    out = buf.getvalue()
    assert "judge's pick" in out and "Enter = Accept judge's pick (Candidate 2)" in out
    assert "bestofn_candidate_1_review.png" in out
    w, _, stack = _widgets("\r")
    with stack:
        assert w.ask(q) == ""


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
