"""A stopped turn is closed before the next user message.

A mid-run Stop ends a turn on a tool result with no assistant reply; the
next user message then follows a tool block, and litellm's Bedrock adapter
prints "Potential consecutive user/tool blocks. Trying to merge." on every
later call. close_interrupted_turn appends a stub assistant note at the
turn-start seam so alternation holds — and does nothing on healthy
histories.
"""

import re
from pathlib import Path

import pytest

from scilink.utils.tool_media import close_interrupted_turn

ASSISTANT_CALL = {"role": "assistant", "tool_calls": [
    {"id": "c1", "type": "function",
     "function": {"name": "f", "arguments": "{}"}}]}
TOOL_RESULT = {"role": "tool", "tool_call_id": "c1", "content": "done"}


def test_tool_tail_gets_a_stub_assistant():
    msgs = [{"role": "user", "content": "go"}, ASSISTANT_CALL, TOOL_RESULT]
    out = close_interrupted_turn(list(msgs))
    assert out[-1]["role"] == "assistant"
    assert "interrupted" in out[-1]["content"]
    assert out[:-1] == msgs


def test_idempotent():
    msgs = close_interrupted_turn(
        [{"role": "user", "content": "go"}, ASSISTANT_CALL, TOOL_RESULT])
    assert close_interrupted_turn(list(msgs)) == msgs


@pytest.mark.parametrize("tail", [
    [],
    [{"role": "user", "content": "hi"}],
    [{"role": "user", "content": "hi"},
     {"role": "assistant", "content": "reply"}],
])
def test_healthy_histories_untouched(tail):
    assert close_interrupted_turn(list(tail)) == tail


def test_note_names_the_last_completed_tool_call():
    """Phase B of the resume plan: the closure note tells the model WHERE the
    turn stopped, so a restored session continues from the recorded results
    instead of restarting the turn's work."""
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c9", "type": "function",
                             "function": {"name": "run_analysis",
                                          "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c9", "content": "big result"}]
    out = close_interrupted_turn(list(msgs))
    note = out[-1]["content"]
    assert "run_analysis" in note
    assert "do not redo completed work" in note


def test_note_names_all_calls_of_the_last_batch():
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": None,
             "tool_calls": [
                 {"id": "a", "type": "function",
                  "function": {"name": "load_metadata", "arguments": "{}"}},
                 {"id": "b", "type": "function",
                  "function": {"name": "run_analysis", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "a", "content": "r1"},
            {"role": "tool", "tool_call_id": "b", "content": "r2"}]
    note = close_interrupted_turn(list(msgs))[-1]["content"]
    assert "load_metadata" in note and "run_analysis" in note


def test_malformed_tool_calls_fall_back_to_generic_note():
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"bad": "shape"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "r"}]
    out = close_interrupted_turn(list(msgs))
    assert out[-1]["role"] == "assistant"
    assert "interrupted" in out[-1]["content"]


ORCHESTRATORS = [
    Path("scilink/agents/exp_agents/analysis_orchestrator.py"),
    Path("scilink/agents/planning_agents/planning_orchestrator.py"),
    Path("scilink/agents/sim_agents/simulation_orchestrator.py"),
    Path("scilink/agents/meta_agent/meta_orchestrator.py"),
]

# The seam: every place a NEW user message enters the history must close
# an interrupted turn first (both litellm and openai chat loops).
_SEAM = re.compile(
    r"close_interrupted_turn\(self\.messages\)\s*\n\s*"
    r"self\.messages\.append\(\{\"role\": \"user\", \"content\": user_input\}\)")


@pytest.mark.parametrize("src", ORCHESTRATORS, ids=lambda p: p.stem)
def test_every_turn_start_closes_interrupted_turns(src):
    text = src.read_text()
    appends = len(re.findall(
        r"self\.messages\.append\(\{\"role\": \"user\", "
        r"\"content\": user_input\}\)", text))
    assert appends == 2, f"{src.name}: expected both chat loops"
    assert len(_SEAM.findall(text)) == appends


def test_litellm_bedrock_warning_is_actually_silenced(caplog):
    """The real litellm transform: a tool-tailed history warns, a closed
    one does not."""
    pytest.importorskip("litellm")
    from litellm.litellm_core_utils.prompt_templates.factory import (
        _bedrock_converse_messages_pt)

    def convert(msgs):
        caplog.clear()
        import logging
        with caplog.at_level(logging.WARNING):
            _bedrock_converse_messages_pt(
                messages=msgs, model="anthropic.claude", llm_provider="bedrock")
        return " ".join(r.message for r in caplog.records)

    broken = [{"role": "user", "content": "go"}, ASSISTANT_CALL, TOOL_RESULT,
              {"role": "user", "content": "next request"}]
    healthy = close_interrupted_turn(
        [{"role": "user", "content": "go"}, ASSISTANT_CALL, TOOL_RESULT])
    healthy.append({"role": "user", "content": "next request"})

    assert "consecutive" in convert(broken).lower(), \
        "probe failed: the unclosed history should trigger the merge warning"
    assert "consecutive" not in convert(healthy).lower()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
