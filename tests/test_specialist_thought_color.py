"""Meta vs. meta-delegated-specialist reasoning are the SAME visible 💭 glyph
but render in different UI colors, driven by an invisible U+2063 source marker.

Guards: the marker is emitted only when delegated (gated on `_agent_label`),
the visible glyph is unchanged, and `_log_to_html` colors + strips it.
"""
import io
from contextlib import redirect_stdout

import pytest

from scilink.agents.exp_agents.analysis_orchestrator import AnalysisOrchestratorAgent
from scilink.agents.planning_agents.planning_orchestrator import PlanningOrchestratorAgent
from scilink.ui.app import (
    _log_to_html, _THOUGHT_MARK, _META_THOUGHT_COLOR, _SPECIALIST_THOUGHT_COLOR,
    _HANDOFF_COLOR,
)

THOUGHT = "\U0001F4AD"  # 💭


class _Stub:
    pass


def _emit(cls, label, text="reasoning here"):
    s = _Stub()
    if label is not None:
        s._agent_label = label
    buf = io.StringIO()
    with redirect_stdout(buf):
        cls._print_assistant_reasoning(s, text)
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
