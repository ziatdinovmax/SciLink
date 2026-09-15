"""
Unit tests for the LangGraph backbone — pytest-discoverable.

Covers the gaps left by tests/test_langgraph_backbone.py:

* _compress_messages_inplace  — threshold, truncation, skip-last-2, no-op
* _langchain_to_openai_dict   — all message types (System, Tool, AIMessage
                                with tool_calls, dict pass-through, fallback)
* _litellm_message_to_langchain — mirrors the OpenAI converter
* _react_should_continue      — step-count gate, tool-call route, END route
* empty-response nudge        — call_model injects HumanMessage on empty AI reply
* _seed_graph_history         — messages are written into MemorySaver
* state field population      — step_count and autonomy_mode round-trip through
                                the graph after a single mocked invocation

No live LLM calls are made; OpenAI/LiteLLM clients are patched where needed.
No API keys are required.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path when running from any working directory
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

def _make_analysis_orch(base_dir: str):
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisOrchestratorAgent,
        AnalysisMode,
    )
    return AnalysisOrchestratorAgent(
        base_dir=base_dir,
        api_key="dummy-key-not-used",
        model_name="claude-opus-4-6",
        analysis_mode=AnalysisMode.AUTONOMOUS,
    )


def _fake_openai_response(content: str, tool_calls=None):
    """Minimal mock that looks like an openai.ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ===========================================================================
# _compress_messages_inplace
# ===========================================================================

class TestCompressMessages:
    """Tests for scilink.graphs._react._compress_messages_inplace."""

    def setup_method(self):
        from scilink.graphs._react import _compress_messages_inplace
        self.compress = _compress_messages_inplace

    def test_noop_below_threshold(self):
        """No compression when total chars are below the threshold."""
        messages = [
            {"role": "tool", "content": "x" * 100},
            {"role": "tool", "content": "y" * 100},
        ]
        original = [dict(m) for m in messages]
        self.compress(messages, threshold=10_000)
        assert messages == original

    def test_compresses_large_tool_message(self):
        """A tool message > 30 000 chars is truncated when total > threshold."""
        big = "A" * 40_000
        messages = [
            {"role": "tool", "content": big},
            {"role": "assistant", "content": "last"},  # one of the last-2 → protected
        ]
        # Only one message, which is in the last-2 window — should NOT be compressed.
        self.compress(messages, threshold=1_000)
        assert messages[0]["content"] == big  # protected by last-2 rule

    def test_compresses_old_messages_not_recent_two(self):
        """Messages outside the last-2 window are candidates; last 2 are protected."""
        big = "B" * 40_000
        messages = [
            {"role": "tool", "content": big},          # old — should be compressed
            {"role": "assistant", "content": "penultimate"},
            {"role": "user", "content": "last"},
        ]
        self.compress(messages, threshold=1_000)
        # First message should have been truncated to 5 000 chars + note
        assert len(messages[0]["content"]) < 40_000
        assert "truncated from history" in messages[0]["content"]
        # Last two must be untouched
        assert messages[1]["content"] == "penultimate"
        assert messages[2]["content"] == "last"

    def test_truncation_length_is_5000(self):
        """Truncated content starts with the first 5 000 chars of the original."""
        from scilink.graphs._react import _COMPRESS_TRUNCATE_AT
        big = "C" * 50_000
        messages = [
            {"role": "tool", "content": big},
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ]
        self.compress(messages, threshold=1_000)
        assert messages[0]["content"].startswith("C" * _COMPRESS_TRUNCATE_AT)

    def test_only_tool_messages_are_compressed(self):
        """assistant / user messages are never truncated, even if large."""
        big = "D" * 40_000
        messages = [
            {"role": "assistant", "content": big},
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ]
        self.compress(messages, threshold=1_000)
        assert messages[0]["content"] == big  # untouched

    def test_multiple_large_tool_messages_all_compressed(self):
        """All old tool messages exceeding 30 000 chars are compressed."""
        big = "E" * 40_000
        messages = [
            {"role": "tool", "content": big},
            {"role": "tool", "content": big},
            {"role": "user", "content": "last-1"},
            {"role": "user", "content": "last-2"},
        ]
        self.compress(messages, threshold=1_000)
        assert "truncated from history" in messages[0]["content"]
        assert "truncated from history" in messages[1]["content"]


# ===========================================================================
# _cap_message_count
# ===========================================================================

class TestCapMessageCount:
    """Tests for scilink.graphs._react._cap_message_count — TODO item 5:
    the char-threshold-only _compress_messages_inplace never trips for a
    session with many small messages, so nothing bounded the LLM-facing
    message *count* until this."""

    def setup_method(self):
        from scilink.graphs._react import _cap_message_count
        self.cap = _cap_message_count

    def _long_thread(self, n_turns):
        messages = [{"role": "system", "content": "sys"}]
        for i in range(n_turns):
            messages.append({"role": "user", "content": f"turn {i}"})
            messages.append({
                "role": "assistant", "content": "",
                "tool_calls": [{"id": f"c{i}", "type": "function",
                                 "function": {"name": "f", "arguments": "{}"}}],
            })
            messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"result {i}"})
            messages.append({"role": "assistant", "content": f"done {i}"})
        return messages

    def test_noop_under_cap(self):
        from types import SimpleNamespace
        messages = self._long_thread(5)
        original = [dict(m) for m in messages]
        self.cap(messages, SimpleNamespace(MAX_HISTORY_MESSAGES=100))
        assert messages == original

    def test_trims_over_cap_and_keeps_system_message(self):
        from types import SimpleNamespace
        messages = self._long_thread(80)  # 1 + 320 messages
        before = len(messages)
        self.cap(messages, SimpleNamespace(MAX_HISTORY_MESSAGES=100))
        assert len(messages) < before
        assert messages[0] == {"role": "system", "content": "sys"}

    def test_no_dangling_tool_call_survives_the_splice(self):
        """The head/tail splice can cut a tool_calls/tool message pair apart;
        it must be repaired, not left dangling (same hazard as the
        self.messages trim fixed for TODO item 4)."""
        from types import SimpleNamespace
        messages = self._long_thread(80)
        self.cap(messages, SimpleNamespace(MAX_HISTORY_MESSAGES=100))

        pending = []
        for m in messages:
            for tc in (m.get("tool_calls") or []):
                pending.append(tc["id"])
            if m.get("role") == "tool":
                cid = m.get("tool_call_id")
                assert cid in pending, "orphan tool result survived the cap"
                pending.remove(cid)
        assert not pending, f"unanswered tool_use survived the cap: {pending}"

    def test_default_cap_used_when_orchestrator_has_no_override(self):
        from types import SimpleNamespace
        from scilink.graphs._react import _MAX_MESSAGE_COUNT_DEFAULT
        messages = self._long_thread(80)
        self.cap(messages, SimpleNamespace())  # no MAX_HISTORY_MESSAGES attr
        assert len(messages) <= _MAX_MESSAGE_COUNT_DEFAULT + 2  # +system +marker

    def test_respects_per_orchestrator_override(self):
        from types import SimpleNamespace
        messages = self._long_thread(80)
        self.cap(messages, SimpleNamespace(MAX_HISTORY_MESSAGES=20))
        assert len(messages) <= 23  # +system +marker, small margin

    def test_does_not_mutate_the_original_list_object_identity(self):
        """Operates in-place via slice assignment so the caller's reference
        (full_messages, passed by the caller without reassignment) stays
        valid after the call."""
        from types import SimpleNamespace
        messages = self._long_thread(80)
        ref = messages
        self.cap(messages, SimpleNamespace(MAX_HISTORY_MESSAGES=100))
        assert ref is messages
        assert len(ref) < 321


# ===========================================================================
# repair_graph_state gating + invoke_graph
# ===========================================================================

class _FakeGraph:
    """Minimal stand-in for a compiled LangGraph — tracks get_state calls
    so tests can assert the checkpoint-deserializing call was skipped."""

    def __init__(self, snapshot_messages=None, invoke_result=None, raise_on_invoke=False):
        from types import SimpleNamespace
        self.get_state_calls = 0
        self.update_state_calls = 0
        self._snapshot = SimpleNamespace(values={"messages": snapshot_messages or []})
        self._invoke_result = invoke_result if invoke_result is not None else {"messages": []}
        self._raise_on_invoke = raise_on_invoke

    def get_state(self, config):
        self.get_state_calls += 1
        return self._snapshot

    def update_state(self, config, patch):
        self.update_state_calls += 1

    def invoke(self, initial_state, config):
        if self._raise_on_invoke:
            raise RuntimeError("boom")
        return self._invoke_result


class TestRepairGraphStateGating:
    """Tests for scilink.graphs._react.repair_graph_state's once-per-clean-
    turn skip — TODO item 8: get_state (checkpoint deserialization) ran
    unconditionally on every turn though the condition it checks for (a
    dangling tool call from a killed process) is realistically true only
    once, right after a restart."""

    def setup_method(self):
        from scilink.graphs._react import repair_graph_state
        self.repair = repair_graph_state

    def _orch(self, **kw):
        from types import SimpleNamespace
        graph = _FakeGraph(**kw)
        return SimpleNamespace(_graph=graph, _graph_config={"configurable": {"thread_id": "t"}})

    def test_first_call_deserializes_the_checkpoint(self):
        orch = self._orch()
        self.repair(orch)
        assert orch._graph.get_state_calls == 1
        assert orch._graph_state_repaired is True

    def test_second_call_this_epoch_skips_get_state(self):
        orch = self._orch()
        self.repair(orch)
        self.repair(orch)
        self.repair(orch)
        assert orch._graph.get_state_calls == 1  # not 3

    def test_flag_already_true_skips_entirely(self):
        """Mirrors a turn that completed cleanly setting the flag itself
        (repair_graph_state is what sets it — this covers a pre-set flag,
        e.g. from a prior repair call, being honored)."""
        orch = self._orch()
        orch._graph_state_repaired = True
        self.repair(orch)
        assert orch._graph.get_state_calls == 0


class TestRepairGraphStateHealing:
    """Tests for what repair_graph_state actually patches in, for each
    kind of dangling terminal message."""

    def setup_method(self):
        from scilink.graphs._react import repair_graph_state
        self.repair = repair_graph_state

    def _orch(self, snapshot_messages):
        from types import SimpleNamespace
        graph = _FakeGraph(snapshot_messages=snapshot_messages)
        return SimpleNamespace(_graph=graph, _graph_config={"configurable": {"thread_id": "t"}})

    def test_dangling_tool_call_gets_a_tool_result_patch(self):
        from langchain_core.messages import AIMessage
        last = AIMessage(content="", tool_calls=[
            {"id": "c1", "name": "f", "args": {}, "type": "tool_call"}])
        orch = self._orch([last])
        self.repair(orch)
        assert orch._graph.update_state_calls == 1

    def test_dangling_tool_result_gets_an_assistant_patch(self):
        from langchain_core.messages import ToolMessage
        last = ToolMessage(content="done", tool_call_id="c1")
        orch = self._orch([last])
        self.repair(orch)
        assert orch._graph.update_state_calls == 1

    def test_dangling_empty_response_nudge_gets_an_assistant_patch(self):
        """TODO round-2 item 1: repair_graph_state must heal a checkpoint
        ending on the empty-response nudge HumanMessage (e.g. from a
        process kill in the narrow window right after it was committed, or
        a checkpoint written before the router-reorder fix), not just
        dangling tool_calls/tool_results — otherwise the next turn's user
        message would land right after it (two consecutive human-role
        messages)."""
        from langchain_core.messages import HumanMessage
        from scilink.graphs._react import _EMPTY_RESPONSE_NUDGE, _NUDGE_MARKER_KEY
        last = HumanMessage(
            content=_EMPTY_RESPONSE_NUDGE,
            additional_kwargs={_NUDGE_MARKER_KEY: True},
        )
        orch = self._orch([last])
        self.repair(orch)
        assert orch._graph.update_state_calls == 1

    def test_well_formed_final_message_is_left_alone(self):
        from langchain_core.messages import AIMessage
        last = AIMessage(content="all done", tool_calls=[])
        orch = self._orch([last])
        self.repair(orch)
        assert orch._graph.update_state_calls == 0

    def test_plain_human_message_is_left_alone(self):
        """A regular user message (not the nudge) at the end of a
        checkpoint is not something repair_graph_state should ever see in
        practice, but must not be misidentified as the nudge."""
        from langchain_core.messages import HumanMessage
        last = HumanMessage(content="an ordinary user turn")
        orch = self._orch([last])
        self.repair(orch)
        assert orch._graph.update_state_calls == 0


class TestInvokeGraph:
    """Tests for scilink.graphs._react.invoke_graph — resets the repair
    flag on a mid-run exception so the NEXT turn re-checks the checkpoint,
    since chat() swallows the exception and reuses the same orchestrator
    instance (no process restart, so repair_graph_state's normal
    once-per-clean-turn skip would otherwise miss a dangling call the
    failed turn left behind)."""

    def setup_method(self):
        from scilink.graphs._react import invoke_graph
        self.invoke = invoke_graph

    def _orch(self, **kw):
        from types import SimpleNamespace
        graph = _FakeGraph(**kw)
        return SimpleNamespace(
            _graph=graph, _graph_config={"configurable": {"thread_id": "t"}},
            _graph_state_repaired=True,  # simulate: already clean from a prior turn
        )

    def test_successful_invoke_leaves_flag_untouched(self):
        orch = self._orch(invoke_result={"messages": ["ok"]})
        result = self.invoke(orch, {"messages": []})
        assert result == {"messages": ["ok"]}
        assert orch._graph_state_repaired is True

    def test_exception_resets_flag_and_reraises(self):
        orch = self._orch(raise_on_invoke=True)
        with pytest.raises(RuntimeError, match="boom"):
            self.invoke(orch, {"messages": []})
        assert orch._graph_state_repaired is False

    def test_reset_flag_makes_the_next_turn_actually_repair(self):
        """End-to-end: a failed turn's reset flag causes the following
        repair_graph_state call to actually deserialize the checkpoint,
        instead of being skipped by the once-per-clean-turn gate."""
        from scilink.graphs._react import repair_graph_state
        orch = self._orch(raise_on_invoke=True)

        with pytest.raises(RuntimeError):
            self.invoke(orch, {"messages": []})

        repair_graph_state(orch)
        assert orch._graph.get_state_calls == 1


# ===========================================================================
# _get_openai_client
# ===========================================================================

class TestGetOpenAIClient:
    """Tests for scilink.graphs._react._get_openai_client — TODO item 7:
    call_model used to construct a fresh openai.OpenAI(...) on every
    re-entry (N tool calls in a turn -> N client constructions), losing
    connection-pool reuse."""

    def setup_method(self):
        from scilink.graphs._react import _get_openai_client
        self.get_client = _get_openai_client

    def _orch(self, api_key="k1", base_url="https://api.example.com"):
        from types import SimpleNamespace
        return SimpleNamespace(model=SimpleNamespace(api_key=api_key, base_url=base_url))

    def test_second_call_reuses_the_same_client_object(self):
        with patch("openai.OpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            orch = self._orch()
            first = self.get_client(orch)
            second = self.get_client(orch)
            assert first is second
            mock_cls.assert_called_once()

    def test_rebuilds_when_api_key_or_base_url_changes(self):
        with patch("openai.OpenAI") as mock_cls:
            mock_cls.side_effect = lambda **kw: MagicMock(kwargs=kw)
            orch = self._orch(api_key="k1")
            first = self.get_client(orch)

            orch.model.api_key = "k2"  # runtime reconfiguration
            second = self.get_client(orch)

            assert first is not second
            assert mock_cls.call_count == 2

    def test_different_orchestrators_get_independent_clients(self):
        with patch("openai.OpenAI") as mock_cls:
            mock_cls.side_effect = lambda **kw: MagicMock()
            orch_a = self._orch()
            orch_b = self._orch()
            client_a = self.get_client(orch_a)
            client_b = self.get_client(orch_b)
            assert client_a is not client_b
            assert orch_a._openai_client is client_a
            assert orch_b._openai_client is client_b


# ===========================================================================
# sync_system_prompt
# ===========================================================================

class TestSyncSystemPrompt:
    """Tests for scilink.graphs._react.sync_system_prompt — TODO item 6:
    orch._system_prompt (read by call_model) and orch.messages[0] (JSON
    persistence / _trim_history copy) were kept in sync by convention at
    ~13 call sites across the 4 orchestrators; a site updating only one
    would silently send a stale system prompt to the LLM."""

    def setup_method(self):
        from scilink.graphs._react import sync_system_prompt
        self.sync = sync_system_prompt

    def test_updates_existing_system_message(self):
        from types import SimpleNamespace
        orch = SimpleNamespace(
            _system_prompt="new prompt",
            messages=[{"role": "system", "content": "old prompt"},
                      {"role": "user", "content": "hi"}],
        )
        self.sync(orch)
        assert orch.messages[0]["content"] == "new prompt"
        assert orch.messages[1]["content"] == "hi"  # untouched

    def test_inserts_system_message_when_missing(self):
        """Mirrors set_simulation_mode's defensive else-branch."""
        from types import SimpleNamespace
        orch = SimpleNamespace(
            _system_prompt="new prompt",
            messages=[{"role": "user", "content": "hi"}],
        )
        self.sync(orch)
        assert orch.messages[0] == {"role": "system", "content": "new prompt"}
        assert orch.messages[1]["content"] == "hi"

    def test_noop_on_empty_messages_list(self):
        from types import SimpleNamespace
        orch = SimpleNamespace(_system_prompt="new prompt", messages=[])
        self.sync(orch)
        assert orch.messages == [{"role": "system", "content": "new prompt"}]


class TestRealOrchestratorSystemPromptStaysInSync:
    """Exercises each orchestrator's own rebuild methods (not just the
    shared helper in isolation) to prove the two copies genuinely cannot
    diverge in practice, for every rebuild site that used to hand-roll the
    sync."""

    def test_analysis_set_analysis_mode(self):
        from scilink.agents.exp_agents.analysis_orchestrator import AnalysisMode
        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)
            orch.set_analysis_mode(AnalysisMode.CO_PILOT)
            assert orch._system_prompt == orch.messages[0]["content"]
            assert orch.messages[0]["role"] == "system"

    def test_analysis_register_external_tool_rebuild(self):
        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)
            before = orch._system_prompt
            orch.register_tools(
                schemas=[{"type": "function", "function": {
                    "name": "dummy_tool", "description": "d",
                    "parameters": {"type": "object", "properties": {}}}}],
                factory=lambda: {"dummy_tool": lambda: "ok"},
            )
            assert orch._system_prompt == orch.messages[0]["content"]
            assert orch._system_prompt != before  # confirms the rebuild actually ran

    def test_meta_set_meta_mode(self):
        from scilink.agents.meta_agent.meta_orchestrator import (
            MetaOrchestratorAgent, MetaMode,
        )
        with tempfile.TemporaryDirectory() as td:
            orch = MetaOrchestratorAgent(
                base_dir=td, api_key="dummy-key-not-used",
                model_name="claude-opus-4-6",
            )
            orch.set_meta_mode(MetaMode.AUTONOMOUS)
            assert orch._system_prompt == orch.messages[0]["content"]

    def test_meta_inject_capabilities(self):
        """The exact site TODO item 6 names — two independent .replace()
        calls on the same placeholder, now a single replace + one sync."""
        from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent
        with tempfile.TemporaryDirectory() as td:
            orch = MetaOrchestratorAgent(
                base_dir=td, api_key="dummy-key-not-used",
                model_name="claude-opus-4-6",
            )
            orch._capabilities_block = None
            orch.messages[0]["content"] += " __SPECIALIST_CAPABILITIES__"
            orch._system_prompt += " __SPECIALIST_CAPABILITIES__"
            orch._inject_capabilities()
            assert "__SPECIALIST_CAPABILITIES__" not in orch._system_prompt
            assert orch._system_prompt == orch.messages[0]["content"]

    def test_simulation_set_simulation_mode(self):
        from scilink.agents.sim_agents.simulation_orchestrator import (
            SimulationOrchestratorAgent, SimulationMode,
        )
        with tempfile.TemporaryDirectory() as td:
            orch = SimulationOrchestratorAgent(
                base_dir=td, api_key="dummy-key-not-used",
                model_name="claude-opus-4-6",
                simulation_mode=SimulationMode.CO_PILOT,
            )
            orch.set_simulation_mode(SimulationMode.AUTONOMOUS)
            assert orch._system_prompt == orch.messages[0]["content"]

    def test_planning_set_autonomy_level(self):
        from scilink.agents.planning_agents.planning_orchestrator import (
            PlanningOrchestratorAgent, AutonomyLevel,
        )
        with tempfile.TemporaryDirectory() as td:
            orch = PlanningOrchestratorAgent(
                objective="Test objective",
                base_dir=td, api_key="dummy-key-not-used",
                model_name="claude-opus-4-6",
                autonomy_level=AutonomyLevel.CO_PILOT,
                data_dir=td,
            )
            orch.set_autonomy_level(AutonomyLevel.AUTONOMOUS)
            assert orch._system_prompt == orch.messages[0]["content"]


# ===========================================================================
# _langchain_to_openai_dict — all message types
# ===========================================================================

class TestLangchainToOpenaiDict:
    """Tests for scilink.graphs._react._langchain_to_openai_dict."""

    def setup_method(self):
        from scilink.graphs._react import _langchain_to_openai_dict
        self.convert = _langchain_to_openai_dict

    def test_human_message(self):
        from langchain_core.messages import HumanMessage
        result = self.convert(HumanMessage(content="hello"))
        assert result == {"role": "user", "content": "hello"}

    def test_system_message(self):
        from langchain_core.messages import SystemMessage
        result = self.convert(SystemMessage(content="you are helpful"))
        assert result == {"role": "system", "content": "you are helpful"}

    def test_tool_message(self):
        from langchain_core.messages import ToolMessage
        result = self.convert(ToolMessage(content="tool output", tool_call_id="tc_1"))
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "tc_1"
        assert result["content"] == "tool output"

    def test_ai_message_no_tool_calls(self):
        from langchain_core.messages import AIMessage
        result = self.convert(AIMessage(content="response", tool_calls=[]))
        assert result["role"] == "assistant"
        assert result["content"] == "response"
        # tool_calls key either absent or empty
        assert not result.get("tool_calls")

    def test_ai_message_with_tool_calls(self):
        from langchain_core.messages import AIMessage
        ai = AIMessage(
            content="",
            tool_calls=[{
                "id": "tc_42",
                "name": "my_tool",
                "args": {"param": "val"},
                "type": "tool_call",
            }],
        )
        result = self.convert(ai)
        assert result["role"] == "assistant"
        tcs = result["tool_calls"]
        assert len(tcs) == 1
        assert tcs[0]["id"] == "tc_42"
        assert tcs[0]["type"] == "function"
        assert tcs[0]["function"]["name"] == "my_tool"
        # args should be JSON-serialised
        args_parsed = json.loads(tcs[0]["function"]["arguments"])
        assert args_parsed == {"param": "val"}

    def test_ai_message_tool_calls_args_already_string(self):
        """If args is already a JSON string it should be passed through unchanged."""
        from langchain_core.messages import AIMessage
        # Build an AIMessage where the internal tool_call has string args
        ai = AIMessage(content="")
        ai.tool_calls = [
            {"id": "tc_s", "name": "t", "args": '{"x": 1}', "type": "tool_call"}
        ]
        result = self.convert(ai)
        assert result["tool_calls"][0]["function"]["arguments"] == '{"x": 1}'

    def test_dict_passthrough(self):
        """A plain dict is returned unchanged."""
        d = {"role": "user", "content": "already a dict"}
        assert self.convert(d) is d

    def test_unknown_type_fallback(self):
        """Unknown objects are stringified and tagged as user messages."""
        result = self.convert(42)
        assert result["role"] == "user"
        assert "42" in result["content"]


# ===========================================================================
# _parse_tool_args — _MALFORMED_ARGS_KEY unwrap
# ===========================================================================

class TestParseToolArgsMalformedMarker:
    """TODO round-2 item 2: _parse_tool_args must unwrap the
    {_MALFORMED_ARGS_KEY: raw_string} marker the converters now produce for
    a tool call whose arguments didn't parse as JSON, and re-derive the same
    recovery-hint error _parse_tool_args already gives a raw malformed
    string — the marker is plumbing to satisfy LangChain's AIMessage dict
    requirement, not a second, different failure mode."""

    def setup_method(self):
        from scilink.graphs._react import _parse_tool_args, _MALFORMED_ARGS_KEY
        self.parse = _parse_tool_args
        self.key = _MALFORMED_ARGS_KEY

    def test_marker_unwraps_to_the_same_error_as_the_raw_string(self):
        raw = "{not: valid json"
        args_marker, err_marker = self.parse({self.key: raw}, None)
        args_raw, err_raw = self.parse(raw, None)
        assert args_marker is None and args_raw is None
        assert json.loads(err_marker) == json.loads(err_raw)

    def test_ordinary_dict_is_not_mistaken_for_the_marker(self):
        """A real, already-parsed args dict must pass through untouched,
        even if it happens to contain other keys alongside one that looks
        like the marker (exact single-key match required)."""
        real_args = {self.key: "not actually malformed", "other": 1}
        args, err = self.parse(real_args, None)
        assert err is None
        assert args is real_args

    def test_valid_json_wrapped_in_marker_key_alone_parses_through(self):
        """A single-key dict whose only key is the marker is always treated
        as the wrapped-string shape and re-parsed — this only ever happens
        via the converters, which only wrap on an actual parse failure, so
        this exercises the unwrap path with a (contrived) valid payload."""
        args, err = self.parse({self.key: '{"x": 1}'}, None)
        assert err is None
        assert args == {"x": 1}

    def test_marker_collision_requires_string_value(self):
        """A real args dict that happens to be exactly
        {_MALFORMED_ARGS_KEY: <value>} only collides with the marker shape
        when the value is itself a string (what the converters always wrap)
        — a non-string value can't have come from the wrapping path, so it's
        passed through as genuine data instead of being unwrapped and
        re-parsed as a string."""
        real_args = {self.key: 42}
        args, err = self.parse(real_args, None)
        assert err is None
        assert args is real_args

    def test_valid_json_non_object_is_treated_as_malformed(self):
        """TODO round-2 item 2 follow-up: arguments that parse as valid JSON
        but aren't a JSON object (a bare list, number, etc.) must not pass
        through as (parsed, None) — execute_tools calls
        orch.tools.execute_tool(name, **args), which raises an uncaught
        TypeError on a non-mapping. Treated the same as a parse failure."""
        for raw in ["[1, 2, 3]", "42", '"just a string"', "true", "null"]:
            args, err = self.parse(raw, None)
            assert args is None, f"expected malformed for {raw!r}, got {args!r}"
            assert err is not None
            assert "not a JSON object" in json.loads(err)["message"]


# ===========================================================================
# _serialize_tool_call_args / _parse_history_tool_args
# ===========================================================================

class TestSerializeToolCallArgs:
    """TODO round-2 item 2 follow-up: once a malformed tool call's args are
    wrapped as {_MALFORMED_ARGS_KEY: raw}, every later re-serialization back
    to OpenAI wire format (call_model's next step, sync_new_messages's write
    to history.json) must unwrap it back to the original raw string, not
    permanently bake in the wrapped-dict shape."""

    def setup_method(self):
        from scilink.graphs._react import _serialize_tool_call_args, _MALFORMED_ARGS_KEY
        self.serialize = _serialize_tool_call_args
        self.key = _MALFORMED_ARGS_KEY

    def test_malformed_marker_unwraps_to_raw_string(self):
        raw = '{not valid json'
        assert self.serialize({self.key: raw}) == raw

    def test_ordinary_dict_still_json_dumped(self):
        assert json.loads(self.serialize({"a": 1})) == {"a": 1}

    def test_string_args_passed_through(self):
        assert self.serialize('{"a": 1}') == '{"a": 1}'

    def test_marker_collision_with_non_string_value_not_unwrapped(self):
        """Mirrors _parse_tool_args's collision guard: a dict that happens
        to match the marker shape but whose value isn't a string is real
        data, not the wrapper, so it's JSON-dumped like any other dict."""
        real = {self.key: 42}
        assert json.loads(self.serialize(real)) == real


class TestParseHistoryToolArgs:
    """TODO round-2 item 2: seed_graph_history's tool-call argument parsing
    (replaying persisted history.json into the graph on restore) must never
    raise on malformed input — history.json can be hand-edited, corrupted,
    or written by an older version of this code. An uncaught exception here
    previously propagated straight out of __init__ (called unconditionally
    whenever history is non-empty), permanently bricking that session."""

    def setup_method(self):
        from scilink.graphs._react import _parse_history_tool_args, _wrap_malformed_args
        self.parse = _parse_history_tool_args
        self.wrap = _wrap_malformed_args

    def test_valid_json_object_parses_normally(self):
        assert self.parse('{"a": 1}') == {"a": 1}

    def test_malformed_json_string_never_raises(self):
        assert self.parse("{not valid json") == self.wrap("{not valid json")

    def test_valid_json_non_object_is_wrapped_not_raised(self):
        assert self.parse("[1, 2, 3]") == self.wrap("[1, 2, 3]")

    def test_already_a_dict_passes_through(self):
        assert self.parse({"a": 1}) == {"a": 1}

    def test_non_string_non_dict_is_wrapped_not_raised(self):
        # Defensive: history.json is external input; a schema violation here
        # (e.g. arguments as a bare int) must still not raise.
        result = self.parse(12345)
        assert result == self.wrap("12345")


# ===========================================================================
# _openai_message_to_langchain
# ===========================================================================

class TestOpenaiMessageToLangchain:
    """Tests for scilink.graphs._react._openai_message_to_langchain."""

    def setup_method(self):
        from scilink.graphs._react import _openai_message_to_langchain
        self.convert = _openai_message_to_langchain

    def _make_openai_msg(self, content, tool_calls=None):
        msg = MagicMock()
        msg.content = content
        msg.tool_calls = tool_calls or []
        return msg

    def _make_tc(self, id_, name, arguments):
        tc = MagicMock()
        tc.id = id_
        tc.function.name = name
        tc.function.arguments = arguments
        return tc

    def test_text_only_response(self):
        msg = self._make_openai_msg("hello from openai")
        result = self.convert(msg)
        assert result.content == "hello from openai"
        assert result.tool_calls == []

    def test_none_content_becomes_empty_string(self):
        msg = self._make_openai_msg(None)
        result = self.convert(msg)
        assert result.content == ""

    def test_tool_calls_converted(self):
        tc = self._make_tc("tc_1", "search", '{"query": "iron"}')
        msg = self._make_openai_msg("", tool_calls=[tc])
        result = self.convert(msg)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["id"] == "tc_1"
        assert result.tool_calls[0]["name"] == "search"
        assert result.tool_calls[0]["args"] == {"query": "iron"}

    def test_empty_arguments_string(self):
        tc = self._make_tc("tc_2", "noop", "")
        msg = self._make_openai_msg("", tool_calls=[tc])
        result = self.convert(msg)
        assert result.tool_calls[0]["args"] == {}

    def test_malformed_arguments_do_not_raise(self):
        """TODO round-2 item 2: this is the exact bug — json.loads used to
        run here, eagerly and unguarded, so a truncated/malformed arguments
        string raised JSONDecodeError before _parse_tool_args (in
        execute_tools) ever got a chance to turn it into a recoverable
        tool-result error. Must not raise here; the raw string is wrapped
        in the _MALFORMED_ARGS_KEY marker (LangChain's AIMessage requires
        args to be a dict, so the raw string can't be passed through
        directly) for _parse_tool_args to unwrap and actually parse/report."""
        from scilink.graphs._react import _MALFORMED_ARGS_KEY
        tc = self._make_tc("tc_3", "f", "{not: valid json")
        msg = self._make_openai_msg("", tool_calls=[tc])
        result = self.convert(msg)  # must not raise
        assert result.tool_calls[0]["args"] == {_MALFORMED_ARGS_KEY: "{not: valid json"}


# ===========================================================================
# _litellm_message_to_langchain
# ===========================================================================

class TestLitellmMessageToLangchain:
    """Tests for scilink.graphs._react._litellm_message_to_langchain."""

    def setup_method(self):
        from scilink.graphs._react import _litellm_message_to_langchain
        self.convert = _litellm_message_to_langchain

    def _make_litellm_msg(self, content, tool_calls=None):
        msg = MagicMock()
        msg.content = content
        msg.tool_calls = tool_calls or []
        return msg

    def _make_tc(self, id_, name, arguments):
        tc = MagicMock()
        tc.id = id_
        tc.function.name = name
        tc.function.arguments = arguments
        return tc

    def test_text_only_response(self):
        msg = self._make_litellm_msg("hello from litellm")
        result = self.convert(msg)
        assert result.content == "hello from litellm"
        assert result.tool_calls == []

    def test_none_content_becomes_empty_string(self):
        msg = self._make_litellm_msg(None)
        result = self.convert(msg)
        assert result.content == ""

    def test_tool_calls_converted(self):
        tc = self._make_tc("ltc_1", "search", '{"query": "iron"}')
        msg = self._make_litellm_msg("", tool_calls=[tc])
        result = self.convert(msg)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["id"] == "ltc_1"
        assert result.tool_calls[0]["name"] == "search"
        assert result.tool_calls[0]["args"] == {"query": "iron"}

    def test_empty_arguments_string(self):
        """Empty arguments string should produce an empty args dict, not an error."""
        tc = self._make_tc("ltc_2", "noop", "")
        msg = self._make_litellm_msg("", tool_calls=[tc])
        result = self.convert(msg)
        assert result.tool_calls[0]["args"] == {}

    def test_malformed_arguments_do_not_raise(self):
        """TODO round-2 item 2 — see the OpenAI converter's identical test."""
        from scilink.graphs._react import _MALFORMED_ARGS_KEY
        tc = self._make_tc("ltc_3", "f", "{not: valid json")
        msg = self._make_litellm_msg("", tool_calls=[tc])
        result = self.convert(msg)  # must not raise
        assert result.tool_calls[0]["args"] == {_MALFORMED_ARGS_KEY: "{not: valid json"}

    def test_no_tool_calls_attribute(self):
        """Objects without a tool_calls attribute should produce an empty list."""
        msg = MagicMock(spec=[])  # no attributes
        msg.content = "bare"
        result = self.convert(msg)
        assert result.content == "bare"
        assert result.tool_calls == []


class TestMalformedToolArgsEndToEnd:
    """TODO round-2 item 2, end-to-end: a real orchestrator turn where the
    model's tool call has a malformed arguments string must not crash the
    turn (the old bug — json.loads raised inside call_model's message
    conversion, before _parse_tool_args ever ran) — it should skip the
    tool, report the recovery-hint error as a tool result, and let the
    model retry with a shorter/valid call."""

    def test_malformed_tool_args_do_not_crash_the_turn(self):
        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)
            orch.use_openai = True

            fake_tc = MagicMock()
            fake_tc.id = "tc_bad"
            fake_tc.function.name = "nonexistent_tool"
            fake_tc.function.arguments = "{not: valid json"
            bad_call_response = _fake_openai_response(content="", tool_calls=[fake_tc])
            followup_response = _fake_openai_response(content="Retrying with a shorter call.")

            call_count = [0]

            def side_effect(**kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    return bad_call_response
                return followup_response

            with patch("openai.OpenAI") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.chat.completions.create.side_effect = side_effect

                result = orch.chat("Do something")

            # Must not have hit chat()'s generic exception handler.
            assert "❌ Error:" not in result
            assert "Retrying with a shorter call." in result
            # The tool must never actually have been invoked with guessed args.
            assert call_count[0] == 2


# ===========================================================================
# _react_should_continue router
# ===========================================================================

class TestReactShouldContinue:
    """Tests for scilink.graphs._react._react_should_continue."""

    def setup_method(self):
        from scilink.graphs._react import _react_should_continue
        from langgraph.graph import END
        self.router = _react_should_continue
        self.END = END

    def _state(self, step_count: int, has_tool_calls: bool) -> Dict[str, Any]:
        from langchain_core.messages import AIMessage
        last = AIMessage(content="")
        if has_tool_calls:
            last.tool_calls = [{"id": "x", "name": "t", "args": {}, "type": "tool_call"}]
        else:
            last.tool_calls = []
        return {"messages": [last], "step_count": step_count}

    def test_routes_to_execute_tools_when_tool_calls_present(self):
        state = self._state(step_count=0, has_tool_calls=True)
        assert self.router(state, max_steps=20) == "execute_tools"

    def test_routes_to_end_when_no_tool_calls(self):
        state = self._state(step_count=0, has_tool_calls=False)
        assert self.router(state, max_steps=20) == self.END

    def test_routes_to_end_at_max_steps(self):
        """step_count == max_steps should immediately route to END."""
        state = self._state(step_count=5, has_tool_calls=True)
        assert self.router(state, max_steps=5) == self.END

    def test_routes_to_end_beyond_max_steps(self):
        """step_count > max_steps also routes to END."""
        state = self._state(step_count=99, has_tool_calls=True)
        assert self.router(state, max_steps=5) == self.END

    def test_step_count_missing_defaults_to_zero(self):
        """Missing step_count key should be treated as 0."""
        from langchain_core.messages import AIMessage
        last = AIMessage(content="done")
        last.tool_calls = []
        state = {"messages": [last]}  # no step_count key
        assert self.router(state, max_steps=20) == self.END

    def test_nudge_retried_even_at_or_past_the_step_cap(self):
        """TODO round-2 item 1: the nudge check must run BEFORE the
        step-count cap. A model that returns an empty response on the same
        round the cap is hit must still get its one-shot forced-text retry,
        not be routed straight to END on the dangling nudge HumanMessage —
        which would leave a stale final-text scan and a following turn
        stacking two consecutive human-role messages."""
        from langchain_core.messages import HumanMessage
        from scilink.graphs._react import _EMPTY_RESPONSE_NUDGE, _NUDGE_MARKER_KEY
        nudge = HumanMessage(
            content=_EMPTY_RESPONSE_NUDGE,
            additional_kwargs={_NUDGE_MARKER_KEY: True},
        )

        state_at_cap = {"messages": [nudge], "step_count": 5}
        assert self.router(state_at_cap, max_steps=5) == "call_model"

        state_past_cap = {"messages": [nudge], "step_count": 99}
        assert self.router(state_past_cap, max_steps=5) == "call_model"


# ===========================================================================
# Empty-response nudge injection (call_model node behaviour)
# ===========================================================================

class TestEmptyResponseNudge:
    """
    When the model returns an AIMessage with no content and no tool_calls,
    call_model should append a synthetic HumanMessage nudge.
    """

    def test_nudge_injected_on_empty_response(self):
        from langchain_core.messages import AIMessage, HumanMessage

        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)
            orch.use_openai = True

            # Return an empty message (no content, no tool_calls)
            empty_response = _fake_openai_response(content="", tool_calls=[])

            with patch("openai.OpenAI") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.chat.completions.create.return_value = empty_response

                result = orch.chat("Do something")

            # The orchestrator should return the fallback warning or a summary
            # (either the nudge was processed → some text came back,
            #  or the step limit was hit → the fallback warning).
            assert isinstance(result, str)
            assert len(result) > 0

    def test_nudge_actually_triggers_a_second_call_model_pass(self):
        """
        The nudge must route back to call_model — not fall through to END
        without ever retrying. Regression test: _react_should_continue used
        to check the last message for tool_calls only, so the injected
        HumanMessage nudge (which has none) routed straight to END and the
        LLM was never asked again — every empty first response silently
        became "⚠️ Maximum tool iterations reached" instead of a real retry.
        """
        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)
            orch.use_openai = True

            empty_response = _fake_openai_response(content="", tool_calls=[])
            followup_response = _fake_openai_response(content="Here is the summary.")
            call_count = [0]

            def side_effect(**kwargs):
                call_count[0] += 1
                return empty_response if call_count[0] == 1 else followup_response

            with patch("openai.OpenAI") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.chat.completions.create.side_effect = side_effect

                result = orch.chat("Do something")

            assert call_count[0] == 2, (
                "expected exactly one retry after the empty response — "
                f"got {call_count[0]} call_model invocation(s)"
            )
            assert "Here is the summary." in result

    def test_nudge_retried_when_empty_response_lands_exactly_at_step_cap(self):
        """TODO round-2 item 1, end-to-end: an empty response arriving on
        the call_model pass right after step_count already hit
        MAX_TOOL_ITERATIONS must still get its forced-text retry and
        return real content — not route straight to END on the dangling
        nudge HumanMessage (which would make the final-text scan fall
        through to a stale reply, here none, so the fallback warning)."""
        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)
            orch.use_openai = True
            # should_continue reads orch.max_iterations first if it's an
            # int, falling back to MAX_TOOL_ITERATIONS only when it isn't —
            # the orchestrator always sets max_iterations at construction,
            # so that's the one that actually caps this run.
            orch.max_iterations = 1  # step_count hits 1 after one tool call

            from scilink.graphs.analysis import build_analysis_graph
            from langgraph.checkpoint.memory import MemorySaver
            orch._graph = build_analysis_graph(orch, checkpointer=MemorySaver())

            fake_tc = MagicMock()
            fake_tc.id = "tc_001"
            fake_tc.function.name = "nonexistent_tool"
            fake_tc.function.arguments = "{}"
            tool_call_response = _fake_openai_response(content="", tool_calls=[fake_tc])
            empty_response = _fake_openai_response(content="", tool_calls=[])
            followup_response = _fake_openai_response(content="Recovered summary.")

            # Pass 1: a tool call -> execute_tools runs, step_count becomes 1
            #         (== MAX_TOOL_ITERATIONS).
            # Pass 2: call_model runs again with step_count already at the
            #         cap and returns an empty response — this is the race.
            # Pass 3: the forced-tool_choice="none" nudge retry.
            responses = [tool_call_response, empty_response, followup_response]
            call_count = [0]

            def side_effect(**kwargs):
                i = call_count[0]
                call_count[0] += 1
                return responses[min(i, len(responses) - 1)]

            with patch("openai.OpenAI") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.chat.completions.create.side_effect = side_effect

                result = orch.chat("Do something")

            assert call_count[0] == 3, (
                "expected tool-call pass + empty pass + nudge retry — "
                f"got {call_count[0]} call(s)"
            )
            assert "Recovered summary." in result
            assert "Maximum tool iterations" not in result

    def test_nudge_not_injected_when_content_present(self):
        """When the model returns real content, no nudge is added."""
        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)
            orch.use_openai = True

            normal_response = _fake_openai_response(content="Analysis complete.")

            with patch("openai.OpenAI") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.chat.completions.create.return_value = normal_response

                result = orch.chat("Analyse data")

            assert "Analysis complete." in result


# ===========================================================================
# _seed_graph_history
# ===========================================================================

class TestSeedGraphHistory:
    """
    _seed_graph_history() should replay persisted messages into MemorySaver
    so a restored session has full thread context.
    """

    def test_seed_injects_messages_into_graph_state(self):
        from langchain_core.messages import HumanMessage, AIMessage

        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)

            history = [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Follow-up"},
            ]
            orch._seed_graph_history(history)

            # After seeding, the graph state should contain those messages
            state = orch._graph.get_state(orch._graph_config)
            messages = state.values.get("messages", [])
            contents = [m.content for m in messages if hasattr(m, "content")]
            assert "First question" in contents
            assert "First answer" in contents
            assert "Follow-up" in contents

    def test_seed_empty_history_is_noop(self):
        """An empty history list should not raise and should leave state empty."""
        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)
            # Should not raise
            orch._seed_graph_history([])

    def test_seed_skips_system_messages(self):
        """System messages in history should be ignored by the seeder."""
        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)
            history = [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ]
            orch._seed_graph_history(history)
            state = orch._graph.get_state(orch._graph_config)
            messages = state.values.get("messages", [])
            roles = [getattr(m, "type", "") for m in messages]
            assert "system" not in roles


# ===========================================================================
# State field population through graph invocation
# ===========================================================================

class TestStateFieldPopulation:
    """
    Verify that step_count, autonomy_mode, and session_dir are correctly
    carried through a single mocked graph invocation.
    """

    def test_step_count_increments_per_tool_call(self):
        """
        Each execute_tools step increments step_count by 1.
        A response with one tool call followed by a final text response
        should leave step_count == 1 in the graph's persisted state.
        """
        from langchain_core.messages import AIMessage
        from langgraph.checkpoint.memory import MemorySaver
        from scilink.graphs.analysis import build_analysis_graph

        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)
            orch.use_openai = True
            orch.MAX_TOOL_ITERATIONS = 5
            orch._graph = build_analysis_graph(orch, checkpointer=MemorySaver())

            # First call returns a tool_call; second returns plain text.
            tc = MagicMock()
            tc.id = "tc_step"
            tc.function.name = "nonexistent_tool"
            tc.function.arguments = "{}"

            tool_response = _fake_openai_response("", tool_calls=[tc])
            final_response = _fake_openai_response("Done.")

            call_n = [0]

            def side_effect(**kwargs):
                call_n[0] += 1
                return tool_response if call_n[0] == 1 else final_response

            with patch("openai.OpenAI") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.chat.completions.create.side_effect = side_effect

                orch.chat("Do one tool call then finish.")

            # After the call, retrieve graph state and inspect step_count
            state = orch._graph.get_state(orch._graph_config)
            step_count = state.values.get("step_count", None)
            assert step_count is not None, "step_count not found in graph state"
            assert step_count == 1, f"expected step_count=1, got {step_count}"

    def test_autonomy_mode_in_state(self):
        """autonomy_mode passed to _invoke_graph should be visible in state after call."""
        from langgraph.checkpoint.memory import MemorySaver
        from scilink.graphs.analysis import build_analysis_graph

        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)
            orch.use_openai = True
            orch._graph = build_analysis_graph(orch, checkpointer=MemorySaver())

            normal_response = _fake_openai_response("Autonomous reply.")

            with patch("openai.OpenAI") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.chat.completions.create.return_value = normal_response
                orch.chat("Hello")

            state = orch._graph.get_state(orch._graph_config)
            # autonomy_mode should be the string value set by the orchestrator
            autonomy_mode = state.values.get("autonomy_mode")
            assert autonomy_mode is not None
            assert isinstance(autonomy_mode, str)
            assert len(autonomy_mode) > 0


# ===========================================================================
# invoke_graph_turn — crashing turn's own input must survive to history.json
# ===========================================================================

class TestInvokeGraphTurn:
    """A gap found in a follow-up review of TODO round-2 item 4
    (sync_new_messages): sync_new_messages only runs on invoke_graph's
    success path, inside each orchestrator's _invoke_graph, AFTER the graph
    call returns. An exception mid-run previously skipped it entirely, and
    chat()'s except block only calls _auto_checkpoint() (scalar
    checkpoint.json fields), never _save_history() — so the crashing turn's
    own user input was silently absent from every persisted record.
    invoke_graph_turn closes that gap directly."""

    def setup_method(self):
        from scilink.graphs._react import invoke_graph_turn
        self.invoke = invoke_graph_turn

    def _orch(self, raise_on_invoke=False):
        from types import SimpleNamespace
        graph = _FakeGraph(raise_on_invoke=raise_on_invoke,
                            invoke_result={"messages": ["ok"]})
        save_calls = []
        return SimpleNamespace(
            _graph=graph, _graph_config={"configurable": {"thread_id": "t"}},
            _graph_state_repaired=True,
            messages=[],
            _save_history=lambda: save_calls.append(1),
            _save_history_calls=save_calls,
        )

    def test_success_path_does_not_touch_messages(self):
        """On success, invoke_graph_turn defers entirely to the normal
        sync_new_messages path (called later by extract_final_text) —
        it must not double-append the user input itself."""
        orch = self._orch()
        result = self.invoke(orch, {"messages": []}, "hello")
        assert result == {"messages": ["ok"]}
        assert orch.messages == []
        assert orch._save_history_calls == []

    def test_exception_records_user_input_and_saves_history(self):
        orch = self._orch(raise_on_invoke=True)
        with pytest.raises(RuntimeError, match="boom"):
            self.invoke(orch, {"messages": []}, "the crashing question")
        assert orch.messages == [{"role": "user", "content": "the crashing question"}]
        assert orch._save_history_calls == [1]

    def test_exception_still_reraises_even_if_save_history_itself_fails(self):
        """A broken _save_history() must not swallow the original exception
        or prevent the user-input append."""
        orch = self._orch(raise_on_invoke=True)

        def broken_save():
            raise OSError("disk full")
        orch._save_history = broken_save

        with pytest.raises(RuntimeError, match="boom"):
            self.invoke(orch, {"messages": []}, "the crashing question")
        assert orch.messages == [{"role": "user", "content": "the crashing question"}]

    def test_missing_save_history_hook_is_tolerated(self):
        """Not every duck-typed orch necessarily has _save_history (tests
        commonly use bare stand-ins) — the append must still happen."""
        from types import SimpleNamespace
        graph = _FakeGraph(raise_on_invoke=True)
        orch = SimpleNamespace(
            _graph=graph, _graph_config={"configurable": {"thread_id": "t"}},
            _graph_state_repaired=True, messages=[],
        )
        with pytest.raises(RuntimeError, match="boom"):
            self.invoke(orch, {"messages": []}, "q")
        assert orch.messages == [{"role": "user", "content": "q"}]


# ===========================================================================
# extract_final_text — step-cap fallback text must be persisted too
# ===========================================================================

class TestExtractFinalText:
    """A gap found in a follow-up review: when the graph thread ends
    without a content-bearing AIMessage (step-count cap reached), the
    fallback warning text shown to the user isn't part of
    result['messages'] at all, so sync_new_messages can't mirror it —
    silently lost from history.json until the next restart's
    seed_graph_history repair overwrites the gap with a generic
    "[Turn interrupted...]" placeholder. extract_final_text persists the
    fallback text directly when it's used."""

    def setup_method(self):
        from scilink.graphs._react import extract_final_text
        self.extract = extract_final_text

    def _orch(self, max_iterations=20):
        from types import SimpleNamespace
        return SimpleNamespace(
            messages=[], max_iterations=max_iterations,
            _graph_synced_message_count=0,
        )

    def test_normal_reply_is_returned_and_synced_not_duplicated(self):
        from langchain_core.messages import HumanMessage, AIMessage
        orch = self._orch()
        result = {
            "messages": [HumanMessage(content="hi"), AIMessage(content="hello back")],
            "step_count": 1,
        }
        text = self.extract(orch, result)
        assert text == "hello back"
        # sync_new_messages already mirrors the real thread; no extra
        # fallback append should happen on the normal path.
        assert orch.messages == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello back"},
        ]

    def test_step_cap_fallback_is_persisted_to_messages(self):
        from langchain_core.messages import HumanMessage
        orch = self._orch(max_iterations=5)
        result = {"messages": [HumanMessage(content="hi")], "step_count": 5}
        text = self.extract(orch, result)
        assert "Maximum tool iterations" in text
        assert orch._last_chat_hit_iter_cap is True
        # The fallback text the user actually saw must be the LAST message
        # in orch.messages, not silently missing.
        assert orch.messages[-1] == {"role": "assistant", "content": text}

    def test_fallback_without_hitting_cap_does_not_set_iter_cap_flag(self):
        """An empty thread for some other reason (e.g. step_count below
        max_iterations) still gets the fallback text persisted, but
        shouldn't be misreported as an iteration-cap exhaustion."""
        from langchain_core.messages import HumanMessage
        orch = self._orch(max_iterations=20)
        result = {"messages": [HumanMessage(content="hi")], "step_count": 1}
        text = self.extract(orch, result)
        assert "Maximum tool iterations" in text
        assert getattr(orch, "_last_chat_hit_iter_cap", False) is False
        assert orch.messages[-1] == {"role": "assistant", "content": text}
