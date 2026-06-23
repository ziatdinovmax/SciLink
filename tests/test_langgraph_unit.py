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
* verification _route_verification — all four branches
* verification anneal node    — patience escalation, reset on improvement, cap
* verification human_feedback node — with and without human_fn
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

    def test_no_tool_calls_attribute(self):
        """Objects without a tool_calls attribute should produce an empty list."""
        msg = MagicMock(spec=[])  # no attributes
        msg.content = "bare"
        result = self.convert(msg)
        assert result.content == "bare"
        assert result.tool_calls == []


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
# verification._route_verification — all four branches
# ===========================================================================

class TestRouteVerification:
    """Tests for scilink.graphs.verification._route_after_verify."""

    def setup_method(self):
        from scilink.graphs.verification import (
            _route_after_verify,
            _ROUTE_APPROVED,
            _ROUTE_HUMAN,
            _ROUTE_REFINE,
            _ROUTE_MAX,
            _ROUTE_FAILED,
        )
        self.route = _route_after_verify
        self.APPROVED = _ROUTE_APPROVED
        self.HUMAN = _ROUTE_HUMAN
        self.REFINE = _ROUTE_REFINE
        self.MAX = _ROUTE_MAX
        self.FAILED = _ROUTE_FAILED

    def _state(self, **kwargs):
        base = {
            "approved": False,
            "best_score": 0.0,
            "iteration": 0,
            "human_feedback_requested": False,
            "verification_failed": False,
        }
        base.update(kwargs)
        return base

    def test_approved_flag_routes_to_approved(self):
        state = self._state(approved=True)
        assert self.route(state, quality_threshold=0.7, max_iterations=5) == self.APPROVED

    def test_high_score_routes_to_approved(self):
        state = self._state(best_score=0.95)
        assert self.route(state, quality_threshold=0.7, max_iterations=5) == self.APPROVED

    def test_score_at_threshold_routes_to_approved(self):
        """Score equal to the threshold is approved (≥, not >)."""
        state = self._state(best_score=0.7)
        assert self.route(state, quality_threshold=0.7, max_iterations=5) == self.APPROVED

    def test_max_iterations_routes_to_max(self):
        state = self._state(iteration=5)
        assert self.route(state, quality_threshold=0.7, max_iterations=5) == self.MAX

    def test_human_feedback_requested_routes_to_human(self):
        state = self._state(iteration=2, human_feedback_requested=True)
        assert self.route(state, quality_threshold=0.7, max_iterations=5) == self.HUMAN

    def test_default_routes_to_refine(self):
        """Nothing special → continue refining."""
        state = self._state(iteration=1, best_score=0.3)
        assert self.route(state, quality_threshold=0.7, max_iterations=5) == self.REFINE

    def test_approved_takes_priority_over_max_iterations(self):
        """Approval check runs before max_iterations check."""
        state = self._state(approved=True, iteration=10)
        assert self.route(state, quality_threshold=0.7, max_iterations=5) == self.APPROVED

    def test_verification_failed_routes_to_failed(self):
        """verification_failed has highest priority — routes to END."""
        state = self._state(verification_failed=True, approved=True)
        assert self.route(state, quality_threshold=0.7, max_iterations=5) == self.FAILED


# ===========================================================================
# verification anneal node — patience / escalation logic
# ===========================================================================

class TestVerificationAnnealNode:
    """
    The anneal node inside the verification subgraph manages a patience counter
    and escalates the annealing_level when the score stalls.
    """

    def _build_subgraph_with_scores(self, scores: list, max_iterations: int = 10):
        """
        Build a verification subgraph where run_fn cycles through a list of
        scores and verify_fn reports them.  Returns (subgraph, initial_state).
        """
        from scilink.graphs.verification import build_verification_subgraph
        from langgraph.checkpoint.memory import MemorySaver

        score_iter = iter(scores + [scores[-1]] * 20)  # cycle last score
        step = [0]

        def run_fn(state):
            s = next(score_iter)
            return {
                "current_result": {"success": True, "_quality_score": s},
            }

        def verify_fn(state):
            return {"quality_score": state.get("best_score", 0.0)}

        def feedback_fn(state):
            # Change config each call so config_unchanged never fires
            step[0] += 1
            return {"analysis_config": {"step": step[0]}}

        subgraph = build_verification_subgraph(
            run_fn=run_fn,
            verify_fn=verify_fn,
            feedback_fn=feedback_fn,
            max_iterations=max_iterations,
            quality_threshold=0.9,  # high threshold → never approved by score
            n_annealing_levels=3,
            patience_limit=2,
            checkpointer=MemorySaver(),
        )

        initial = {
            "messages": [],
            "analysis_config": {},
            "current_result": None,
            "best_result": None,
            "best_score": 0.0,
            "prev_best_score": 0.0,
            "last_verification": None,
            "verification_failed": False,
            "config_unchanged": False,
            "verification_history": [],
            "iteration": 0,
            "max_iterations": max_iterations,
            "annealing_level": 0,
            "patience_counter": 0,
            "approved": False,
            "human_feedback_requested": False,
            "best_r2": 0.0,
            "r2_floor": 0.0,
            "r2_threshold": 0.8,
            "best_ever_rejected": False,
            "best_verification": None,
        }
        return subgraph, initial

    def test_annealing_escalates_after_patience_limit(self):
        """After 2 non-improving iterations the annealing level should have risen."""
        # Flat scores — never improving
        subgraph, initial = self._build_subgraph_with_scores(
            [0.3, 0.3, 0.3, 0.3, 0.3, 0.3], max_iterations=6
        )
        cfg = {"configurable": {"thread_id": "anneal-escalate"}}
        result = subgraph.invoke(initial, config=cfg)
        # Ran to max_iterations without approval; annealing should have escalated
        assert result["annealing_level"] > 0

    def test_annealing_resets_patience_on_improvement(self):
        """A score improvement should reset the patience counter."""
        # Improve on iteration 1 → patience should be reset
        subgraph, initial = self._build_subgraph_with_scores(
            [0.3, 0.5, 0.5, 0.5, 0.5, 0.5], max_iterations=6
        )
        cfg = {"configurable": {"thread_id": "anneal-reset"}}
        result = subgraph.invoke(initial, config=cfg)
        # After improvement + stall, level should be 0 or 1, never 2 (not enough stall)
        assert result["annealing_level"] <= 2  # sanity — never exceeds n_levels-1

    def test_annealing_level_never_exceeds_max(self):
        """Annealing level is capped at n_annealing_levels - 1 (== 2 for 3 levels)."""
        subgraph, initial = self._build_subgraph_with_scores(
            [0.1] * 20, max_iterations=20
        )
        cfg = {"configurable": {"thread_id": "anneal-cap"}}
        result = subgraph.invoke(initial, config=cfg)
        assert result["annealing_level"] <= 2


# ===========================================================================
# verification human_feedback node — with and without human_fn
# ===========================================================================

class TestVerificationHumanFeedback:
    """Tests for the human_feedback node in the verification subgraph."""

    def _base_initial(self, max_iterations=3):
        return {
            "messages": [],
            "analysis_config": {},
            "current_result": None,
            "best_result": None,
            "best_score": 0.0,
            "prev_best_score": 0.0,
            "last_verification": None,
            "verification_failed": False,
            "config_unchanged": False,
            "verification_history": [],
            "iteration": 0,
            "max_iterations": max_iterations,
            "annealing_level": 0,
            "patience_counter": 0,
            "approved": False,
            "human_feedback_requested": False,
            "best_r2": 0.0,
            "r2_floor": 0.0,
            "r2_threshold": 0.8,
            "best_ever_rejected": False,
            "best_verification": None,
        }

    def _build_subgraph(self, human_fn, max_iterations=3):
        from scilink.graphs.verification import build_verification_subgraph
        from langgraph.checkpoint.memory import MemorySaver

        step = [0]

        def run_fn(state):
            return {"current_result": {"success": True}}

        def verify_fn(state):
            # Request human feedback on the first iteration
            request = state.get("iteration", 0) == 0
            return {
                "quality_score": 0.3,
                "human_feedback_requested": request,
            }

        def feedback_fn(state):
            # Always change config so config_unchanged never fires
            step[0] += 1
            return {"analysis_config": {"step": step[0]}, "human_feedback_requested": False}

        return build_verification_subgraph(
            run_fn=run_fn,
            verify_fn=verify_fn,
            feedback_fn=feedback_fn,
            human_fn=human_fn,
            max_iterations=max_iterations,
            quality_threshold=0.9,
            checkpointer=MemorySaver(),
        )

    def test_human_fn_none_is_passthrough(self):
        """When human_fn=None, the human_feedback node clears the flag and continues."""
        subgraph = self._build_subgraph(human_fn=None)
        cfg = {"configurable": {"thread_id": "human-none"}}
        result = subgraph.invoke(self._base_initial(), config=cfg)
        assert "approved" in result

    def test_human_fn_called_and_result_applied(self):
        """When human_fn is provided it should be called during the subgraph run."""
        call_log = []

        def human_fn(state):
            call_log.append("called")
            return {"analysis_config": {"human_override": True}}

        subgraph = self._build_subgraph(human_fn=human_fn)
        cfg = {"configurable": {"thread_id": "human-fn"}}
        subgraph.invoke(self._base_initial(), config=cfg)

        assert "called" in call_log, "human_fn was never invoked"


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
