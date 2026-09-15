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


# All four chat-driven orchestrators (the three modes + the meta agent — see
# CLAUDE.md "The mode universe is fixed at three" / "The meta agent") are now
# on the LangGraph backbone: a single chat() -> _invoke_graph() path per
# orchestrator, no more hand-rolled dual (openai + litellm) loops. The
# equivalent protection against a mid-run-Stop dangling tool call is
# _repair_graph_state(), which heals it in the CHECKPOINTED graph state
# before each invoke — see AnalysisOrchestratorAgent._repair_graph_state for
# the rationale. There is no self.messages textual seam to check any more;
# self.messages is a lightweight parallel log, not what the LLM sees on the
# next turn.
GRAPH_BACKED_ORCHESTRATORS = [
    Path("scilink/agents/exp_agents/analysis_orchestrator.py"),
    Path("scilink/agents/planning_agents/planning_orchestrator.py"),
    Path("scilink/agents/sim_agents/simulation_orchestrator.py"),
    Path("scilink/agents/meta_agent/meta_orchestrator.py"),
]

_GRAPH_SEAM = re.compile(
    r"def _invoke_graph\(self, user_input: str\) -> str:.*?"
    r"self\._repair_graph_state\(\)", re.DOTALL)


@pytest.mark.parametrize("src", GRAPH_BACKED_ORCHESTRATORS, ids=lambda p: p.stem)
def test_every_graph_turn_repairs_state_first(src):
    text = src.read_text()
    assert "def _repair_graph_state(self) -> None:" in text, (
        f"{src.name}: expected _repair_graph_state to be defined")
    assert _GRAPH_SEAM.search(text), (
        f"{src.name}: expected _invoke_graph to call self._repair_graph_state() "
        "before appending the new turn")


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
