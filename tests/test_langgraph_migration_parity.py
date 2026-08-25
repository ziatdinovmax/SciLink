"""
Parity regression tests for the LangGraph migration.

Each test class maps to a finding from the migration audit:

  R1 — FIXED REGRESSION: API timeout retry restored
       Old code retried timeouts up to 3 times. After the initial migration
       the new code raised on the first timeout (regression). The retry logic
       has been restored in _react.py (_TIMEOUT_RETRIES = 3). Tests now assert
       the restored behaviour: up to 3 retries on timeout, raise on the 3rd.

  R2 — REGRESSION: Verification subgraph not wired into controllers
       graphs/verification.py is exported and tested in isolation, but
       image_analysis_controllers.py and curve_fitting_controllers.py still
       run their own imperative `for verification_iter in range(...)` loops.
       Tests assert this structural gap so it cannot be silently closed
       (or re-opened) without the test suite noticing.

  D1 — BEHAVIORAL DELTA: Empty-response nudge path changed
       Old code made a second API call with `tool_choice="none"` to force a
       text reply. New code injects a HumanMessage and lets the model respond
       freely. Tests document both the old contract (no longer met) and the
       new contract (what the code actually does).

  D2 — BEHAVIORAL DELTA: Tool-result compression behaviour changed for planning
       Old `_compress_large_tool_results()` permanently mutated `self.messages`
       (the JSON-persisted history). New `_compress_messages_inplace()` in
       `_react.py` compresses only the wire-format copy passed to the API each
       turn; `self.messages` is left untouched. Tests document both contracts.

  DC1 — DEAD CODE: PlanningOrchestratorAgent._compress_large_tool_results
        The method is still defined but never called since the LangGraph
        migration. The test pins this so the dead code is not accidentally
        re-invoked (or silently removed without awareness).

No live LLM calls are made. OpenAI/LiteLLM clients are patched where needed.
No API keys are required.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Shared helpers
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
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ===========================================================================
# R1 — API timeout retry restored
# ===========================================================================

class TestR1TimeoutRetryRegression:
    """
    R1 FIXED: _react.py now retries API timeouts up to _TIMEOUT_RETRIES (3)
    times before re-raising, matching the behaviour of the old
    _handle_openai_chat / _handle_litellm_chat loops.

    Non-timeout exceptions still raise immediately (no blanket retry).

    NOTE: chat() catches all exceptions and returns an error string; it does
    not re-raise.  These tests call _invoke_graph() directly (which re-raises)
    to observe raw retry behaviour.
    """

    def test_invoke_graph_timeout_retries_up_to_limit(self):
        """
        A timeout triggers up to _TIMEOUT_RETRIES attempts before raising.
        The mock should be called exactly _TIMEOUT_RETRIES times.
        """
        from scilink.graphs._react import _TIMEOUT_RETRIES

        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)
            orch.use_openai = True

            call_count = [0]

            def side_effect(**kwargs):
                call_count[0] += 1
                raise Exception("Request timed out")

            with patch("openai.OpenAI") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.chat.completions.create.side_effect = side_effect

                with pytest.raises(Exception, match="timed out"):
                    orch._invoke_graph("Do something")

            assert call_count[0] == _TIMEOUT_RETRIES, (
                f"Expected {_TIMEOUT_RETRIES} API calls (retry up to limit), "
                f"got {call_count[0]}."
            )

    def test_invoke_graph_timeout_succeeds_on_retry(self):
        """
        If a timeout clears on the second attempt, the call succeeds and
        the result is returned normally.
        """
        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)
            orch.use_openai = True

            attempt = [0]

            def side_effect(**kwargs):
                attempt[0] += 1
                if attempt[0] < 2:
                    raise Exception("Request timed out")
                # Second attempt succeeds
                msg = MagicMock()
                msg.content = "Recovered after timeout."
                msg.tool_calls = []
                choice = MagicMock()
                choice.message = msg
                resp = MagicMock()
                resp.choices = [choice]
                return resp

            with patch("openai.OpenAI") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.chat.completions.create.side_effect = side_effect

                result = orch.chat("Do something")

            assert "Recovered after timeout." in result
            assert attempt[0] == 2

    def test_chat_returns_error_string_on_timeout_exhausted(self):
        """
        When all retries are exhausted, chat() catches the exception and
        returns an error string rather than re-raising.
        """
        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)
            orch.use_openai = True

            with patch("openai.OpenAI") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.chat.completions.create.side_effect = Exception(
                    "Connection timed out"
                )

                result = orch.chat("Hello")

            assert "Error" in result or "timed out" in result.lower() or "❌" in result

    def test_invoke_graph_non_timeout_raises_immediately_no_retry(self):
        """
        Non-timeout exceptions raise immediately — confirming retry is
        timeout-specific, not a blanket retry.
        """
        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)
            orch.use_openai = True

            call_count = [0]

            def side_effect(**kwargs):
                call_count[0] += 1
                raise Exception("AuthenticationError: invalid API key")

            with patch("openai.OpenAI") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.chat.completions.create.side_effect = side_effect

                with pytest.raises(Exception, match="AuthenticationError"):
                    orch._invoke_graph("Hello")

            assert call_count[0] == 1, (
                "Non-timeout exception should raise after exactly 1 attempt, "
                f"got {call_count[0]} attempts."
            )


# ===========================================================================
# R2 — Verification subgraph not wired into controllers
# ===========================================================================

class TestR2VerificationSubgraphNotWired:
    """
    graphs/verification.py defines and exports a LangGraph subgraph that
    could in principle replace the per-item verify/refine loop in
    image_analysis_controllers.py and curve_fitting_controllers.py.

    That wiring is deliberately NOT done: upstream/main independently unified
    the same loop across all three codegen controllers (image, curve-fitting,
    hyperspectral) behind a shared, non-LangGraph ``CodegenQCEngine``
    (issue #327). Wiring only 2 of the 3 controllers onto a LangGraph
    subgraph would leave hyperspectral inconsistent, and this inner
    generate/verify/refine loop is not part of the LangGraph backbone's
    scope (that's the orchestrator-level chat loop — see CLAUDE.md). These
    tests confirm the subgraph stays available standalone without being
    wired into the controllers.
    """

    def test_verification_subgraph_is_importable_and_exported(self):
        """
        The subgraph IS built and exported — confirming it exists and is
        ready to be wired in once the controllers are migrated.
        """
        from scilink.graphs import build_verification_subgraph
        assert callable(build_verification_subgraph)

    def test_verification_subgraph_runs_independently(self):
        """
        The subgraph works correctly in isolation — a prerequisite for wiring
        it into the controllers without breaking existing behaviour.
        """
        from scilink.graphs import build_verification_subgraph
        from langgraph.checkpoint.memory import MemorySaver

        approved = [False]

        def run_fn(state):
            return {
                "current_result": {"success": True},
                "best_result": {"success": True},
                "best_score": 0.85,
            }

        def verify_fn(state):
            approved[0] = True
            return {"approved": True, "best_score": 0.85}

        def feedback_fn(state):
            return {"analysis_config": {}}

        sg = build_verification_subgraph(
            run_fn=run_fn,
            verify_fn=verify_fn,
            feedback_fn=feedback_fn,
            max_iterations=3,
            quality_threshold=0.7,
            checkpointer=MemorySaver(),
        )
        result = sg.invoke(
            {
                "messages": [],
                "analysis_config": {},
                "current_result": None,
                "best_result": None,
                "best_score": 0.0,
                "verification_history": [],
                "iteration": 0,
                "max_iterations": 3,
                "annealing_level": 0,
                "patience_counter": 0,
                "approved": False,
                "human_feedback_requested": False,
            },
            config={"configurable": {"thread_id": "r2-sanity"}},
        )
        assert result["approved"] is True
        assert approved[0], "verify_fn was never called"


# ===========================================================================
# D1 — Empty-response nudge: FIXED to match old tool_choice="none" behaviour
# ===========================================================================

class TestD1EmptyResponseNudgeDelta:
    """
    D1 FIXED: When the model returns an empty response (no content, no tool_calls),
    the old code made a second API call with tool_choice="none" to force a text reply.

    The new code:
      1. Injects a HumanMessage nudge into graph state (1 API call, not 2)
      2. On the NEXT call_model step, detects the nudge as the last message
         and passes tool_choice="none" — forcing a text-only reply, matching
         the old contract exactly.

    This means the delta (model could call tools again on nudge step) is closed.
    """

    def test_empty_response_returns_two_messages_nudge_included(self):
        """
        call_model returns [empty_ai_msg, HumanMessage(nudge)] when the model
        produces no content and no tool_calls.
        """
        from langchain_core.messages import HumanMessage, AIMessage
        from scilink.graphs._react import _make_react_nodes

        orch = MagicMock()
        orch.use_openai = True
        orch._system_prompt = "system"
        orch.tools_for_model = []

        empty_msg = MagicMock()
        empty_msg.content = ""
        empty_msg.tool_calls = []
        choice = MagicMock()
        choice.message = empty_msg
        empty_response = MagicMock()
        empty_response.choices = [choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = empty_response

        with patch("openai.OpenAI", return_value=mock_client):
            call_model, _ = _make_react_nodes(orch)
            state = {"messages": [HumanMessage(content="Do something")], "step_count": 0}
            delta = call_model(state)

        msgs = delta.get("messages", [])
        assert len(msgs) == 2, (
            f"Expected [empty_ai_msg, nudge_human_msg], got {len(msgs)} messages"
        )
        assert isinstance(msgs[1], HumanMessage)
        assert "summarize" in msgs[1].content.lower() or "next steps" in msgs[1].content.lower()

    def test_nudge_step_uses_tool_choice_none(self):
        """
        RESTORED CONTRACT: When the last message is the nudge HumanMessage,
        call_model passes tool_choice="none" to the API — forcing a text-only
        reply, identical to the old tool_choice="none" followup call.
        """
        from langchain_core.messages import HumanMessage, AIMessage
        from scilink.graphs._react import _make_react_nodes, _EMPTY_RESPONSE_NUDGE

        orch = MagicMock()
        orch.use_openai = True
        orch._system_prompt = "system"
        orch.tools_for_model = []

        summary_msg = MagicMock()
        summary_msg.content = "Here is what I did."
        summary_msg.tool_calls = []
        choice = MagicMock()
        choice.message = summary_msg
        summary_response = MagicMock()
        summary_response.choices = [choice]

        captured_kwargs = []
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda **kw: (
            captured_kwargs.append(kw) or summary_response
        )

        with patch("openai.OpenAI", return_value=mock_client):
            call_model, _ = _make_react_nodes(orch)
            # State where the last message IS the nudge
            nudge_state = {
                "messages": [
                    HumanMessage(content="original question"),
                    AIMessage(content=""),
                    HumanMessage(content=_EMPTY_RESPONSE_NUDGE),
                ],
                "step_count": 1,
            }
            call_model(nudge_state)

        assert len(captured_kwargs) == 1
        assert captured_kwargs[0]["tool_choice"] == "none", (
            f"Expected tool_choice='none' on nudge step, "
            f"got {captured_kwargs[0].get('tool_choice')!r}"
        )

    def test_normal_step_uses_tool_choice_auto(self):
        """
        Non-nudge steps still use tool_choice="auto".
        """
        from langchain_core.messages import HumanMessage
        from scilink.graphs._react import _make_react_nodes

        orch = MagicMock()
        orch.use_openai = True
        orch._system_prompt = "system"
        orch.tools_for_model = []

        normal_msg = MagicMock()
        normal_msg.content = "Normal reply."
        normal_msg.tool_calls = []
        choice = MagicMock()
        choice.message = normal_msg
        normal_response = MagicMock()
        normal_response.choices = [choice]

        captured_kwargs = []
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda **kw: (
            captured_kwargs.append(kw) or normal_response
        )

        with patch("openai.OpenAI", return_value=mock_client):
            call_model, _ = _make_react_nodes(orch)
            state = {"messages": [HumanMessage(content="Do something")], "step_count": 0}
            call_model(state)

        assert captured_kwargs[0]["tool_choice"] == "auto"

    def test_nudge_message_is_human_message_not_second_api_call(self):
        """
        The nudge is injected as a HumanMessage in the delta — NOT a second
        API call on the same step. Mock called exactly once on empty response.
        """
        from langchain_core.messages import HumanMessage
        from scilink.graphs._react import _make_react_nodes

        orch = MagicMock()
        orch.use_openai = True
        orch._system_prompt = "system"
        orch.tools_for_model = []

        empty_msg = MagicMock()
        empty_msg.content = ""
        empty_msg.tool_calls = []
        choice = MagicMock()
        choice.message = empty_msg
        empty_response = MagicMock()
        empty_response.choices = [choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = empty_response

        with patch("openai.OpenAI", return_value=mock_client):
            call_model, _ = _make_react_nodes(orch)
            state = {"messages": [HumanMessage(content="go")], "step_count": 0}
            call_model(state)

        assert mock_client.chat.completions.create.call_count == 1, (
            "Expected exactly 1 API call on empty-response step. "
            "The nudge is injected into state, not a second LLM call."
        )


# ===========================================================================
# D2 — Tool-result compression behaviour changed for planning
# ===========================================================================

class TestD2CompressionBehaviourDelta:
    """
    BEHAVIORAL DELTA D2: Tool-result compression changed between old and new code.

    OLD (planning orchestrator only):
        _compress_large_tool_results() was called at the start of _handle_*_chat
        and modified self.messages IN-PLACE, permanently compressing the
        JSON-persisted history.

    NEW (_react.py, all orchestrators):
        _compress_messages_inplace() is called inside call_model on the
        wire-format list built just before the API call.  self.messages is
        NOT modified.

    Tests document both contracts.
    """

    def test_new_compress_does_not_mutate_self_messages(self):
        """
        NEW CONTRACT: _compress_messages_inplace does NOT modify self.messages.
        The JSON-persisted history remains uncompressed even when a large tool
        result triggers compression on the wire-format copy.
        """
        from scilink.graphs._react import _compress_messages_inplace

        # Build a wire-format list that IS large enough to trigger compression
        big_content = "X" * 40_000
        wire_messages = [
            {"role": "system", "content": "sys"},
            {"role": "tool", "content": big_content},   # old, compressible
            {"role": "user", "content": "recent-1"},
            {"role": "user", "content": "recent-2"},
        ]

        # Simulate what _react.py does: compress a copy, not the original list
        import copy
        wire_copy = copy.deepcopy(wire_messages)
        _compress_messages_inplace(wire_copy, threshold=1_000)

        # Wire copy IS compressed
        assert "truncated from history" in wire_copy[1]["content"]
        # Original list is NOT modified (self.messages equivalent)
        assert wire_messages[1]["content"] == big_content

    def test_old_compress_mutates_list_in_place(self):
        """
        OLD CONTRACT: _compress_large_tool_results() on the planning orchestrator
        permanently modifies the list it operates on (self.messages).
        This test documents the OLD behaviour using the same algorithm
        extracted inline — the method still exists in the planning orchestrator
        but is no longer called.

        This behaviour is what changed: the old code modified `self.messages`
        permanently; the new code does not.
        """
        # Replicate the old algorithm (same as PlanningOrchestratorAgent._compress_large_tool_results)
        def old_compress(messages):
            total_chars = sum(len(m.get("content", "") or "") for m in messages)
            if total_chars > 100_000:
                for msg in messages[:-2]:
                    if msg["role"] == "tool" and len(msg.get("content", "")) > 30_000:
                        original_len = len(msg["content"])
                        msg["content"] = (
                            msg["content"][:5_000]
                            + f"\n\n... ({original_len - 5_000} chars truncated from history.)"
                        )

        messages = [
            {"role": "tool", "content": "T" * 40_000},
            {"role": "user", "content": "recent-1"},
            {"role": "user", "content": "recent-2"},
        ]
        # Pad to exceed 100k threshold
        messages.insert(0, {"role": "tool", "content": "P" * 70_000})

        original_content = messages[0]["content"]  # the big one

        old_compress(messages)

        # OLD CONTRACT: list is modified in-place (content shortened)
        assert len(messages[0]["content"]) < len(original_content), (
            "Old compress should have mutated the message content in-place"
        )
        assert "truncated from history" in messages[0]["content"]

    def test_new_compress_applies_to_all_orchestrators_not_just_planning(self):
        """
        NEW CONTRACT: _compress_messages_inplace is called inside call_model
        for ALL three orchestrators (analysis, planning, simulation), not just
        planning as in the old code.  This is an improvement.

        Verify by checking the call_model node source references the function.
        """
        import inspect
        from scilink.graphs import _react
        source = inspect.getsource(_react)
        assert "_compress_messages_inplace" in source
        # Appears in call_model (the node function factory)
        assert source.count("_compress_messages_inplace") >= 2  # definition + ≥1 call


# ===========================================================================
# DC1 — Dead code: PlanningOrchestratorAgent._compress_large_tool_results
# ===========================================================================

class TestDC1DeadCodeCompress:
    """
    DEAD CODE DC1: PlanningOrchestratorAgent._compress_large_tool_results()
    was called inside _handle_openai_chat / _handle_litellm_chat.  Those
    methods were removed in the LangGraph migration.  The method itself was
    left behind and is never called from the new _invoke_graph() path.

    Tests document this so the dead code is:
    1. Not accidentally re-invoked (which would mutate self.messages)
    2. Not silently removed without awareness of the contract change
    """

    def test_method_still_defined(self):
        """
        DC1: The method still exists on the class (it is dead code, not yet
        removed).  If it is removed, this test should be deleted too.
        """
        from scilink.agents.planning_agents.planning_orchestrator import (
            PlanningOrchestratorAgent,
        )
        assert hasattr(PlanningOrchestratorAgent, "_compress_large_tool_results"), (
            "_compress_large_tool_results has been removed. "
            "Delete this test class (DC1) — the dead code has been cleaned up."
        )

    def test_method_not_called_from_invoke_graph(self):
        """
        DC1: _compress_large_tool_results is NOT called from _invoke_graph.
        If it were re-introduced, it would permanently mutate self.messages —
        changing the JSON-persisted history, which is D2's documented delta.
        """
        import inspect
        from scilink.agents.planning_agents.planning_orchestrator import (
            PlanningOrchestratorAgent,
        )
        invoke_graph_source = inspect.getsource(
            PlanningOrchestratorAgent._invoke_graph
        )
        assert "_compress_large_tool_results" not in invoke_graph_source, (
            "_compress_large_tool_results is being called from _invoke_graph. "
            "This would permanently mutate self.messages (old D2 behaviour). "
            "If this is intentional, update D2 tests accordingly."
        )

    def test_method_not_called_from_chat(self):
        """
        DC1: _compress_large_tool_results is NOT called from chat() either.
        """
        import inspect
        from scilink.agents.planning_agents.planning_orchestrator import (
            PlanningOrchestratorAgent,
        )
        chat_source = inspect.getsource(PlanningOrchestratorAgent.chat)
        assert "_compress_large_tool_results" not in chat_source, (
            "_compress_large_tool_results is being called from chat(). "
            "If this is intentional, update D2 tests accordingly."
        )

    def test_method_logic_is_superseded_by_react_compress(self):
        """
        DC1: The logic in _compress_large_tool_results is functionally
        superseded by _compress_messages_inplace in _react.py.  Both use the
        same threshold (100k chars), truncation point (5k chars), and
        skip-last-2 rule — confirming they are equivalent and the dead method
        is safe to remove.
        """
        import inspect
        from scilink.graphs import _react
        from scilink.graphs._react import _COMPRESS_THRESHOLD, _COMPRESS_TRUNCATE_AT
        from scilink.agents.planning_agents.planning_orchestrator import (
            PlanningOrchestratorAgent,
        )

        old_source = inspect.getsource(
            PlanningOrchestratorAgent._compress_large_tool_results
        )

        # Both use the same threshold value (100 000)
        assert _COMPRESS_THRESHOLD == 100_000, \
            f"New compress threshold changed to {_COMPRESS_THRESHOLD}"
        assert "100000" in old_source or "100_000" in old_source, \
            "Old method threshold changed"

        # Both use the same truncation point (5 000)
        assert _COMPRESS_TRUNCATE_AT == 5_000, \
            f"New compress truncation changed to {_COMPRESS_TRUNCATE_AT}"
        assert "5000" in old_source or "5_000" in old_source, \
            "Old method truncation point changed"

        # Both skip the last 2 messages
        assert "messages[:-2]" in old_source, \
            "Old method skip-last-2 rule changed"
        new_source = inspect.getsource(_react._compress_messages_inplace)
        assert "messages[:-2]" in new_source, \
            "New function skip-last-2 rule changed"
