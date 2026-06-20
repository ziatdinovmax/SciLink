"""Meta vs. meta-delegated-specialist output is told apart by COLOR while the
visible 💭 glyph stays identical:

  * UI — driven by an invisible U+2063 source marker (emitted only when
    delegated); `_log_to_html` colors meta cool cyan / specialist warm amber for
    both the 💭 reasoning lines AND the 🤖 answer header, and strips the marker.
  * CLI — the reasoning/answer ANSI color itself differs (cyan vs amber), since
    captured non-tty output carries no ANSI for the UI to read.

Both are gated on `_agent_label`; a standalone session is unchanged.
"""
import io
import sys
from contextlib import redirect_stdout

import pytest

from scilink.agents.exp_agents.analysis_orchestrator import AnalysisOrchestratorAgent
from scilink.agents.planning_agents.planning_orchestrator import PlanningOrchestratorAgent
from scilink.ui.app import (
    _log_to_html, _THOUGHT_MARK, _META_THOUGHT_COLOR, _SPECIALIST_THOUGHT_COLOR,
    _HANDOFF_COLOR,
)

THOUGHT = "\U0001F4AD"  # 💭
ROBOT = "\U0001F916"    # 🤖
ESC = "\x1b"            # ANSI escape — what the CLI emits on a real tty

ALL = [AnalysisOrchestratorAgent, PlanningOrchestratorAgent]


class _Stub:
    pass


class _FakeTTY(io.StringIO):
    """A StringIO that reports as a terminal, so the printers emit ANSI."""

    def isatty(self):
        return True


def _emit(cls, label, text="reasoning here"):
    s = _Stub()
    if label is not None:
        s._agent_label = label
    buf = io.StringIO()
    with redirect_stdout(buf):
        cls._print_assistant_reasoning(s, text)
    return buf.getvalue()


def _emit_tty(cls, method, label, text="text"):
    """Capture a printer's output as if writing to a real terminal (ANSI on)."""
    s = _Stub()
    if label is not None:
        s._agent_label = label
    buf = _FakeTTY()
    old = sys.stdout
    sys.stdout = buf
    try:
        getattr(cls, method)(s, text)
    finally:
        sys.stdout = old
    return buf.getvalue()


@pytest.mark.parametrize("cls", [AnalysisOrchestratorAgent, PlanningOrchestratorAgent])
def test_standalone_has_no_marker(cls):
    out = _emit(cls, None)  # default label "Agent" -> standalone
    assert _THOUGHT_MARK not in out
    assert THOUGHT in out


@pytest.mark.parametrize("cls", [AnalysisOrchestratorAgent, PlanningOrchestratorAgent])
def test_delegated_carries_invisible_marker(cls):
    out = _emit(cls, "Analysis specialist")
    assert _THOUGHT_MARK in out
    # The visible glyph is unchanged: stripping the marker leaves a plain 💭.
    assert THOUGHT in out.replace(_THOUGHT_MARK, "")
    # The marker sits adjacent to the glyph, not as a separate visible token.
    assert THOUGHT + _THOUGHT_MARK in out


def test_renderer_colors_meta_cool():
    html = _log_to_html(_emit(AnalysisOrchestratorAgent, None))
    assert _META_THOUGHT_COLOR in html
    assert _SPECIALIST_THOUGHT_COLOR not in html


def test_renderer_colors_specialist_warm_and_strips_marker():
    html = _log_to_html(_emit(AnalysisOrchestratorAgent, "Analysis specialist"))
    assert _SPECIALIST_THOUGHT_COLOR in html
    assert _META_THOUGHT_COLOR not in html
    assert _THOUGHT_MARK not in html  # invisible tag never reaches the DOM


def test_specialist_color_persists_across_continuation_lines():
    html = _log_to_html(_emit(AnalysisOrchestratorAgent, "Analysis specialist",
                              "line one\nline two\nline three"))
    # Each line of the multi-line thought is wrapped in the specialist color.
    assert html.count(_SPECIALIST_THOUGHT_COLOR) == 3


def test_structural_headers_not_thought_colored():
    # A 📋/🔬 header is not a 💭 line, so it must stay unstyled (no thought color).
    for header in ("📋 PROPOSED FITTING PLAN", "🔬 SINGLE SPECTRUM INTERPRETATION"):
        html = _log_to_html(f"  {header}")
        assert _META_THOUGHT_COLOR not in html
        assert _SPECIALIST_THOUGHT_COLOR not in html


@pytest.mark.parametrize("banner", [
    "  🧪 Delegating to analysis specialist: fit the edge...",
    "  📋 Delegating to planning specialist: design campaign...",
    "  🧬 Fusing delegations [0, 1]...",
])
def test_handoff_banner_is_pronounced(banner):
    html = _log_to_html(banner)
    assert _HANDOFF_COLOR in html and "font-weight:bold" in html


def test_structural_emoji_headers_are_not_handoffs():
    # A bare 📋/🧪 header (plan/step) must NOT be styled as a handoff banner.
    for header in ("  📋 PROPOSED FITTING PLAN", "  🧪 Step 1: ForceFieldAgent"):
        assert _HANDOFF_COLOR not in _log_to_html(header)


# ── 🤖 answer header: specialist matches its thought color (UI) ──────────────

def _emit_answer(cls, label, text="the answer"):
    s = _Stub()
    if label is not None:
        s._agent_label = label
    buf = io.StringIO()
    with redirect_stdout(buf):
        cls._print_agent_answer(s, text)
    return buf.getvalue()


@pytest.mark.parametrize("cls", ALL)
def test_specialist_answer_header_marked_visible_glyph_unchanged(cls):
    out = _emit_answer(cls, "Analysis specialist")
    assert _THOUGHT_MARK in out
    assert ROBOT in out.replace(_THOUGHT_MARK, "")  # plain 🤖 still visible


@pytest.mark.parametrize("cls", ALL)
def test_standalone_answer_header_has_no_marker(cls):
    assert _THOUGHT_MARK not in _emit_answer(cls, None)


def test_ui_specialist_answer_header_is_specialist_color():
    html = _log_to_html(_emit_answer(AnalysisOrchestratorAgent, "Analysis specialist"))
    assert _SPECIALIST_THOUGHT_COLOR in html   # same color as its thoughts
    assert "#7fdfff" not in html               # not the meta cyan
    assert _THOUGHT_MARK not in html           # marker stripped


def test_ui_meta_answer_header_stays_cyan():
    html = _log_to_html(_emit_answer(AnalysisOrchestratorAgent, None))
    assert "#7fdfff" in html
    assert _SPECIALIST_THOUGHT_COLOR not in html


# ── CLI: the ANSI color itself differs (cyan meta / amber specialist) ────────

@pytest.mark.parametrize("cls", ALL)
def test_cli_reasoning_color_differs(cls):
    meta = _emit_tty(cls, "_print_assistant_reasoning", None)
    spec = _emit_tty(cls, "_print_assistant_reasoning", "Analysis specialist")
    assert f"{ESC}[2;3;36m" in meta   # dim italic cyan
    assert f"{ESC}[2;3;33m" in spec   # dim italic amber


@pytest.mark.parametrize("cls", ALL)
def test_cli_answer_header_color_differs(cls):
    meta = _emit_tty(cls, "_print_agent_answer", None)
    spec = _emit_tty(cls, "_print_agent_answer", "Analysis specialist")
    assert f"{ESC}[1;96m" in meta     # bold bright cyan
    assert f"{ESC}[1;33m" in spec     # bold amber


# ── handoff banner: a multi-line task must not bleed into the banner ─────────

def test_handoff_task_summary_collapses_multiline():
    # The reported CLI bug: a multi-line task carried its newline + the
    # "Primary data file:" line into the bold banner span. The summary must be
    # one line so the styling can't span past it.
    from scilink.agents.meta_agent.meta_orchestrator_tools import _task_summary
    task = ("Analyze a STEM HAADF microscopy image of an LLTO thin film.\n\n"
            "Primary data file: /path/to/data.npy")
    s = _task_summary(task)
    assert "\n" not in s
    assert "Primary data file" not in s
    assert s.startswith("Analyze a STEM HAADF")


def test_handoff_task_summary_no_ellipsis_when_complete():
    from scilink.agents.meta_agent.meta_orchestrator_tools import _task_summary
    assert _task_summary("Fit the decay") == "Fit the decay"  # short, single line


def test_handoff_ends_preceding_thought_block():
    # A 💭 line immediately followed by a handoff: the banner must render as a
    # handoff (gold), not inherit the thought color.
    html = _log_to_html("  💭 I will delegate\n"
                        "  🧪 Delegating to analysis specialist: x")
    banner_line = next(l for l in html.split("\n") if "Delegating" in l)
    assert _HANDOFF_COLOR in banner_line
    assert _META_THOUGHT_COLOR not in banner_line


def test_handoff_matches_ansi_wrapped_terminal_form():
    # The terminal emits a bold-ANSI-wrapped banner; after ANSI strip it must
    # still be recognized as a handoff.
    html = _log_to_html("  \x1b[1;33m🧪 Delegating to analysis specialist: x\x1b[0m")
    assert _HANDOFF_COLOR in html
