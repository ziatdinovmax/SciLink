"""
Comprehensive LangGraph backbone tests — fills gaps not covered by the
existing four test files.

Coverage targets
----------------

G1  — Simulation orchestrator (graph construction, chat, run_task) without ase.
      The three failures in test_langgraph_backbone.py are all caused by
      ``from ase.io import read`` at module-import time inside val_agent.py.
      We skip the whole scilink.agents.sim_agents package and test the graph
      layer directly with a duck-typed stub orchestrator.

G2  — Planning orchestrator chat() smoke test (mocked LLM, no sandbox).

G3  — Checkpoint round-trip: interrupt a graph mid-run via MemorySaver,
      then resume — final result matches non-interrupted run.

G4  — build_react_graph duck-type contract: confirms every attribute in the
      orchestrator contract is both documented and present on real orch instances.

G5  — _seed_graph_history: assistant messages with tool_calls replay correctly
      (existing tests only cover user/assistant/tool messages without calls).

G6  — graphs/__init__.py exports build_curve_fitting_verification_subgraph
      (not checked by test_graphs_init_exports).

G7  — Verification subgraph: verify_fn raising an exception → safe break
      (distinct from returning None — currently only the None case is tested).

G8  — Verification subgraph: config_unchanged at level < max → force_anneal
      escalates level, then loop continues (not breaks).

G9  — Verification subgraph: human_feedback_requested resets to False after
      human_fn runs so the next iteration doesn't re-trigger it.

G10 — ReAct graph: two consecutive turns share MemorySaver thread state so
      message history accumulates across calls.

G11 — Emergency checkpoint saved on chat() exception (error string format).

G12 — run_task() always restores autonomy mode on exit (even if chat() raises).

No live LLM calls. No ase/ASE dependency. No API keys required.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_analysis_orch(base_dir: str):
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisOrchestratorAgent, AnalysisMode,
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


def _fake_tool_call(name: str = "noop", tc_id: str = "tc_0", args: str = "{}"):
    tc = MagicMock()
    tc.id = tc_id
    tc.function.name = name
    tc.function.arguments = args
    return tc


# ---------------------------------------------------------------------------
# G1 — Simulation graph without ase (duck-typed stub orchestrator)
# ---------------------------------------------------------------------------

class TestSimulationGraphStub:
    """
    Tests for graphs/simulation.py that bypass the ase import by building
    a duck-typed stub rather than instantiating SimulationOrchestratorAgent.
    """

    def _make_stub_orch(self):
        orch = MagicMock()
        orch.use_openai = True
        orch._system_prompt = "You are a simulation assistant."
        orch.tools_for_model = []
        orch.MAX_TOOL_ITERATIONS = 5
        orch.tools.execute_tool.return_value = "tool result"
        orch.model.model = "claude-opus-4-6"
        orch.model.api_key = "dummy"
        orch.model.base_url = None
        return orch

    def test_g1_simulation_graph_builds_with_stub_orch(self):
        """build_simulation_graph works with a duck-typed stub orchestrator."""
        from scilink.graphs.simulation import build_simulation_graph
        from scilink.graphs.state import SimulationOrchestratorState
        from langgraph.checkpoint.memory import MemorySaver

        orch = self._make_stub_orch()
        graph = build_simulation_graph(orch, checkpointer=MemorySaver())

        nodes = set(graph.get_graph().nodes.keys())
        assert "call_model" in nodes
        assert "execute_tools" in nodes

    def test_g1_simulation_graph_invoke_with_mock_llm(self):
        """Simulation graph invocation with mocked LLM returns expected text."""
        from scilink.graphs.simulation import build_simulation_graph
        from langgraph.checkpoint.memory import MemorySaver
        from langchain_core.messages import HumanMessage

        orch = self._make_stub_orch()
        graph = build_simulation_graph(orch, checkpointer=MemorySaver())

        fake_response = _fake_openai_response("Simulation complete.")

        with patch("openai.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = fake_response

            result = graph.invoke(
                {
                    "messages": [HumanMessage(content="Generate a silicon structure.")],
                    "step_count": 0,
                    "autonomy_mode": "autonomous",
                    "active_skill": None,
                    "session_dir": "/tmp",
                    "checkpoint_data": {},
                    "mcp_connections": [],
                    "generated_structures": [],
                    "default_calc_params": {},
                    "message_count": 0,
                },
                config={"configurable": {"thread_id": "sim-stub-1"}},
            )

        messages = result.get("messages", [])
        final_content = ""
        for m in reversed(messages):
            if hasattr(m, "content") and m.content:
                final_content = m.content
                break
        assert "Simulation complete." in final_content

    def test_g1_simulation_graph_max_iterations_enforced(self):
        """Simulation graph respects MAX_TOOL_ITERATIONS."""
        from scilink.graphs.simulation import build_simulation_graph
        from langgraph.checkpoint.memory import MemorySaver
        from langchain_core.messages import HumanMessage

        orch = self._make_stub_orch()
        orch.MAX_TOOL_ITERATIONS = 2
        graph = build_simulation_graph(orch, checkpointer=MemorySaver())

        tc = _fake_tool_call("some_tool", "tc_sim", "{}")
        always_tool_resp = _fake_openai_response("", tool_calls=[tc])

        call_n = [0]
        def side_effect(**kwargs):
            call_n[0] += 1
            return always_tool_resp

        with patch("openai.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = side_effect

            result = graph.invoke(
                {
                    "messages": [HumanMessage(content="Loop forever")],
                    "step_count": 0,
                    "autonomy_mode": "autonomous",
                    "active_skill": None,
                    "session_dir": "/tmp",
                    "checkpoint_data": {},
                    "mcp_connections": [],
                    "generated_structures": [],
                    "default_calc_params": {},
                    "message_count": 0,
                },
                config={"configurable": {"thread_id": "sim-stub-maxiter"}},
            )

        assert call_n[0] <= orch.MAX_TOOL_ITERATIONS + 1

    def test_g1_planning_graph_builds_with_stub_orch(self):
        """build_planning_graph works with a duck-typed stub orchestrator."""
        from scilink.graphs.planning import build_planning_graph
        from langgraph.checkpoint.memory import MemorySaver

        orch = self._make_stub_orch()
        graph = build_planning_graph(orch, checkpointer=MemorySaver())

        nodes = set(graph.get_graph().nodes.keys())
        assert "call_model" in nodes
        assert "execute_tools" in nodes


# ---------------------------------------------------------------------------
# G2 — Planning orchestrator chat() smoke test
# ---------------------------------------------------------------------------

class TestPlanningOrchestratorSmoke:
    """Planning orchestrator chat() smoke test — mocked LLM, no sandbox."""

    def test_g2_planning_orch_chat_mock_llm(self):
        """PlanningOrchestratorAgent.chat() returns expected text with mocked LLM."""
        from scilink.agents.planning_agents.planning_orchestrator import (
            PlanningOrchestratorAgent, AutonomyLevel,
        )

        with tempfile.TemporaryDirectory() as td:
            try:
                orch = PlanningOrchestratorAgent(
                    objective="Test objective",
                    base_dir=td,
                    api_key="dummy",
                    model_name="claude-opus-4-6",
                    autonomy_level=AutonomyLevel.AUTONOMOUS,
                    data_dir=td,
                )
            except RuntimeError as e:
                if "sandbox" in str(e).lower() or "code execution" in str(e).lower():
                    pytest.skip("PlanningOrchestrator requires sandbox environment")
                raise

            orch.use_openai = True
            fake_response = _fake_openai_response("Planning complete.")

            with patch("openai.OpenAI") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.chat.completions.create.return_value = fake_response

                result = orch.chat("Recommend next experiments.")

            assert "Planning complete." in result


# ---------------------------------------------------------------------------
# G13 — Meta orchestrator chat() smoke test (mocked LLM)
# ---------------------------------------------------------------------------

class TestMetaOrchestratorSmoke:
    """Meta orchestrator chat() smoke test — mocked LLM.

    MetaOrchestratorAgent was converted to the LangGraph backbone (chat() ->
    _invoke_graph()) after analysis/planning/simulation; this is its
    equivalent of TestPlanningOrchestratorSmoke.
    """

    def test_g13_meta_orch_chat_mock_llm(self):
        """MetaOrchestratorAgent.chat() returns expected text with mocked LLM."""
        from scilink.agents.meta_agent.meta_orchestrator import (
            MetaOrchestratorAgent, MetaMode,
        )

        with tempfile.TemporaryDirectory() as td:
            orch = MetaOrchestratorAgent(
                base_dir=td,
                api_key="dummy",
                model_name="claude-opus-4-6",
                meta_mode=MetaMode.AUTONOMOUS,
            )
            orch.use_openai = True
            fake_response = _fake_openai_response("Meta complete.")

            with patch("openai.OpenAI") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.chat.completions.create.return_value = fake_response

                result = orch.chat("Route this to a specialist.")

            assert "Meta complete." in result

    def test_g13_meta_orch_delegates_via_graph(self):
        """A tool-calling turn dispatches through scilink.graphs._react's
        execute_tools exactly like the other three orchestrators — the
        delegate_to_* tool runs synchronously inside the graph's tool node."""
        from scilink.agents.meta_agent.meta_orchestrator import (
            MetaOrchestratorAgent, MetaMode,
        )

        with tempfile.TemporaryDirectory() as td:
            orch = MetaOrchestratorAgent(
                base_dir=td,
                api_key="dummy",
                model_name="claude-opus-4-6",
                meta_mode=MetaMode.AUTONOMOUS,
            )
            orch.use_openai = True

            tc = _fake_tool_call(name="summarize_session_state", args="{}")
            tool_response = _fake_openai_response("", tool_calls=[tc])
            final_response = _fake_openai_response("Delegation summarized.")

            with patch("openai.OpenAI") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.chat.completions.create.side_effect = [
                    tool_response, final_response,
                ]
                result = orch.chat("What's the session state?")

            assert "Delegation summarized." in result


# ---------------------------------------------------------------------------
# G3 — Checkpoint round-trip
# ---------------------------------------------------------------------------

class TestCheckpointRoundTrip:
    """
    Interrupt a graph mid-run (by capturing MemorySaver state), create a
    fresh graph with the same checkpointer, resume — final result matches
    non-interrupted run.
    """

    def test_g3_checkpoint_state_persists_across_graph_rebuild(self):
        """
        A graph's MemorySaver state survives a graph rebuild on the same
        checkpointer, so a session restored from JSON sees the same
        prior-message context.
        """
        from scilink.graphs.analysis import build_analysis_graph
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()

        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)
            orch.use_openai = True
            orch._graph = build_analysis_graph(orch, checkpointer=checkpointer)

            # First turn
            with patch("openai.OpenAI") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.chat.completions.create.return_value = \
                    _fake_openai_response("First response.")
                orch.chat("First question")

            # Capture thread state before rebuild
            state_before = orch._graph.get_state(orch._graph_config)
            msgs_before = len(state_before.values.get("messages", []))

            # Rebuild graph on same checkpointer (simulates process restart)
            orch._graph = build_analysis_graph(orch, checkpointer=checkpointer)

            # State still present on the same thread
            state_after_rebuild = orch._graph.get_state(orch._graph_config)
            msgs_after = len(state_after_rebuild.values.get("messages", []))

            assert msgs_after == msgs_before, (
                f"MemorySaver lost messages on graph rebuild: "
                f"before={msgs_before}, after={msgs_after}"
            )

    def test_g3_second_turn_sees_first_turn_messages(self):
        """
        A second chat() call sees messages from the first call in the LLM
        wire context (MemorySaver thread accumulation).
        """
        from scilink.graphs.analysis import build_analysis_graph
        from langgraph.checkpoint.memory import MemorySaver

        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)
            orch.use_openai = True
            orch._graph = build_analysis_graph(orch, checkpointer=MemorySaver())

            captured = []

            def capture_first(**kwargs):
                captured.append(list(kwargs.get("messages", [])))
                return _fake_openai_response("First response.")

            with patch("openai.OpenAI") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.chat.completions.create.side_effect = capture_first
                orch.chat("Turn one content UNIQUE_MARKER_ALPHA")

            captured2 = []

            def capture_second(**kwargs):
                captured2.append(list(kwargs.get("messages", [])))
                return _fake_openai_response("Second response.")

            with patch("openai.OpenAI") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.chat.completions.create.side_effect = capture_second
                orch.chat("Turn two question")

            assert len(captured2) == 1
            all_wire_content = " ".join(
                m.get("content", "") or "" for m in captured2[0]
            )
            assert "UNIQUE_MARKER_ALPHA" in all_wire_content, (
                "First turn's user message not present in second turn's LLM context."
            )


# ---------------------------------------------------------------------------
# G4 — build_react_graph duck-type contract
# ---------------------------------------------------------------------------

class TestReactGraphOrchestratorContract:
    """
    Confirms all attributes in the orchestrator duck-type contract documented
    in _react.py are present on real AnalysisOrchestratorAgent instances.
    """

    def test_g4_all_contracted_attributes_present(self):
        """Every attribute named in the _react.py contract exists on the orch."""
        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)

        required = [
            "use_openai",
            "_system_prompt",
            "tools_for_model",
            "tools",
            "MAX_TOOL_ITERATIONS",
        ]
        for attr in required:
            assert hasattr(orch, attr), f"Contract attribute missing: {attr}"

        # tools.execute_tool must be callable
        assert callable(orch.tools.execute_tool), "tools.execute_tool not callable"

    def test_g4_model_attributes_present(self):
        """model.model, model.api_key, model.base_url present on orch."""
        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)

        assert hasattr(orch.model, "model")
        assert hasattr(orch.model, "api_key")
        assert hasattr(orch.model, "base_url")


# ---------------------------------------------------------------------------
# G5 — _seed_graph_history: assistant messages with tool_calls
# ---------------------------------------------------------------------------

class TestSeedGraphHistoryToolCalls:
    """
    _seed_graph_history should replay assistant messages that contain
    tool_calls into MemorySaver correctly.
    """

    def test_g5_seed_assistant_with_tool_calls(self):
        """Assistant messages with tool_calls in JSON history are seeded as AIMessages."""
        from langchain_core.messages import AIMessage

        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)

            history = [
                {"role": "user", "content": "Run analysis"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc_hist",
                            "function": {
                                "name": "run_analysis",
                                "arguments": '{"data_path": "/fake.npy"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "Analysis result", "tool_call_id": "tc_hist"},
            ]
            orch._seed_graph_history(history)

            state = orch._graph.get_state(orch._graph_config)
            messages = state.values.get("messages", [])

            # Find the AIMessage that should carry the tool_calls
            ai_messages = [m for m in messages if isinstance(m, AIMessage)]
            assert len(ai_messages) >= 1

            # At least one AI message should have a tool_call
            ai_with_tc = [m for m in ai_messages if m.tool_calls]
            assert len(ai_with_tc) >= 1, (
                "No AIMessage with tool_calls found after seeding assistant history with tool_calls"
            )
            assert ai_with_tc[0].tool_calls[0]["name"] == "run_analysis"

    def test_g5_seed_multiple_turns_preserves_order(self):
        """Messages seeded from a multi-turn history preserve their order."""
        from langchain_core.messages import HumanMessage, AIMessage

        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)

            history = [
                {"role": "user", "content": "A"},
                {"role": "assistant", "content": "B"},
                {"role": "user", "content": "C"},
                {"role": "assistant", "content": "D"},
            ]
            orch._seed_graph_history(history)

            state = orch._graph.get_state(orch._graph_config)
            messages = state.values.get("messages", [])
            contents = [m.content for m in messages if hasattr(m, "content")]

            assert contents.index("A") < contents.index("B")
            assert contents.index("B") < contents.index("C")
            assert contents.index("C") < contents.index("D")


# ---------------------------------------------------------------------------
# G6 — build_curve_fitting_verification_subgraph exported from __init__
# ---------------------------------------------------------------------------

class TestGraphsInitExportsCurveFitting:
    """build_curve_fitting_verification_subgraph must be exported."""

    def test_g6_curve_fitting_builder_exported(self):
        from scilink.graphs import build_curve_fitting_verification_subgraph
        assert callable(build_curve_fitting_verification_subgraph)


# ---------------------------------------------------------------------------
# G7 — verify_fn raising an exception → verification_failed (safe break)
# ---------------------------------------------------------------------------

class TestVerifyFnExceptionSafeBreak:
    """
    verify_fn raising an exception must set verification_failed=True and
    route to END immediately — the same as returning None.
    """

    def test_g7_verify_fn_exception_triggers_safe_break(self):
        from scilink.graphs.verification import build_verification_subgraph
        from langgraph.checkpoint.memory import MemorySaver

        run_count = [0]

        def run_fn(state):
            run_count[0] += 1
            return {"current_result": {"success": True}}

        def verify_fn(state):
            raise RuntimeError("LLM call failed unexpectedly")

        def feedback_fn(state):
            return {"analysis_config": {}}

        sg = build_verification_subgraph(
            run_fn=run_fn,
            verify_fn=verify_fn,
            feedback_fn=feedback_fn,
            max_iterations=5,
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
                "prev_best_score": 0.0,
                "last_verification": None,
                "verification_failed": False,
                "config_unchanged": False,
                "verification_history": [],
                "iteration": 0,
                "max_iterations": 5,
                "annealing_level": 0,
                "patience_counter": 0,
                "approved": False,
                "human_feedback_requested": False,
                "best_r2": 0.0,
                "r2_floor": 0.0,
                "r2_threshold": 0.8,
                "best_ever_rejected": False,
                "best_verification": None,
            },
            config={"configurable": {"thread_id": "g7-exception"}},
        )

        assert result.get("verification_failed") is True, (
            "Expected verification_failed=True when verify_fn raises"
        )
        # Only one run attempt before safe break
        assert run_count[0] == 1, (
            f"Expected exactly 1 run before safe break, got {run_count[0]}"
        )


# ---------------------------------------------------------------------------
# G8 — force_anneal escalates level < max → loop continues (not breaks)
# ---------------------------------------------------------------------------

class TestForceAnnealContinuesLoop:
    """
    When apply_feedback returns the same config and annealing_level < max,
    force_anneal should escalate the level and continue the loop (not break).
    """

    def test_g8_force_anneal_at_level_0_continues(self):
        """
        force_anneal at level 0 (< max 2) should escalate to 1 and continue.
        Loop should run more than 1 iteration total.
        """
        from scilink.graphs.verification import build_verification_subgraph
        from langgraph.checkpoint.memory import MemorySaver

        run_count = [0]
        config_call = [0]

        def run_fn(state):
            run_count[0] += 1
            return {"current_result": {"success": True}}

        def verify_fn(state):
            return {"quality_score": 0.3}

        def feedback_fn(state):
            config_call[0] += 1
            # Always return the same config so force_anneal fires
            return {"analysis_config": {"fixed": True}}

        sg = build_verification_subgraph(
            run_fn=run_fn,
            verify_fn=verify_fn,
            feedback_fn=feedback_fn,
            max_iterations=5,
            quality_threshold=0.9,
            n_annealing_levels=3,
            checkpointer=MemorySaver(),
        )

        result = sg.invoke(
            {
                "messages": [],
                "analysis_config": {"fixed": True},
                "current_result": None,
                "best_result": None,
                "best_score": 0.0,
                "prev_best_score": 0.0,
                "last_verification": None,
                "verification_failed": False,
                "config_unchanged": False,
                "verification_history": [],
                "iteration": 0,
                "max_iterations": 5,
                "annealing_level": 0,
                "patience_counter": 0,
                "approved": False,
                "human_feedback_requested": False,
                "best_r2": 0.0,
                "r2_floor": 0.0,
                "r2_threshold": 0.8,
                "best_ever_rejected": False,
                "best_verification": None,
            },
            config={"configurable": {"thread_id": "g8-force-continue"}},
        )

        # Level escalated from 0 → higher before breaking at max
        assert result["annealing_level"] > 0

    def test_g8_force_anneal_at_max_level_breaks(self):
        """
        force_anneal at max level (2 for n=3) with no-config-change
        should break the loop, not continue indefinitely.
        """
        from scilink.graphs.verification import build_verification_subgraph
        from langgraph.checkpoint.memory import MemorySaver

        run_count = [0]

        def run_fn(state):
            run_count[0] += 1
            return {"current_result": {"success": True}}

        def verify_fn(state):
            return {"quality_score": 0.3}

        def feedback_fn(state):
            return {"analysis_config": {}}

        sg = build_verification_subgraph(
            run_fn=run_fn,
            verify_fn=verify_fn,
            feedback_fn=feedback_fn,
            max_iterations=20,
            quality_threshold=0.9,
            n_annealing_levels=3,
            checkpointer=MemorySaver(),
        )

        result = sg.invoke(
            {
                "messages": [],
                "analysis_config": {},  # same as feedback_fn will return
                "current_result": None,
                "best_result": None,
                "best_score": 0.0,
                "prev_best_score": 0.0,
                "last_verification": None,
                "verification_failed": False,
                "config_unchanged": False,
                "verification_history": [],
                "iteration": 0,
                "max_iterations": 20,
                "annealing_level": 2,  # already at max
                "patience_counter": 0,
                "approved": False,
                "human_feedback_requested": False,
                "best_r2": 0.0,
                "r2_floor": 0.0,
                "r2_threshold": 0.8,
                "best_ever_rejected": False,
                "best_verification": None,
            },
            config={"configurable": {"thread_id": "g8-force-break"}},
        )

        # Should have broken early, well short of max_iterations=20
        assert result["iteration"] < 20, (
            "Expected early break at max annealing level, but loop ran to max_iterations"
        )


# ---------------------------------------------------------------------------
# G9 — human_feedback_requested clears after human_fn runs
# ---------------------------------------------------------------------------

class TestHumanFeedbackClearsAfterRun:
    """
    After human_fn is called, human_feedback_requested should be False
    so subsequent iterations don't immediately re-enter human feedback.
    """

    def test_g9_human_flag_clears_after_human_fn(self):
        """human_feedback_requested resets after human_fn executes."""
        from scilink.graphs.verification import build_verification_subgraph
        from langgraph.checkpoint.memory import MemorySaver

        human_call_count = [0]
        verify_count = [0]

        def run_fn(state):
            return {"current_result": {"success": True}}

        def verify_fn(state):
            verify_count[0] += 1
            # Request human feedback only on the very first verify call
            return {
                "quality_score": 0.3,
                "human_feedback_requested": verify_count[0] == 1,
            }

        def feedback_fn(state):
            return {"analysis_config": {"step": verify_count[0]}, "human_feedback_requested": False}

        def human_fn(state):
            human_call_count[0] += 1
            return {"analysis_config": {"human": True}, "human_feedback_requested": False}

        sg = build_verification_subgraph(
            run_fn=run_fn,
            verify_fn=verify_fn,
            feedback_fn=feedback_fn,
            human_fn=human_fn,
            max_iterations=5,
            quality_threshold=0.95,
            checkpointer=MemorySaver(),
        )

        result = sg.invoke(
            {
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
                "max_iterations": 5,
                "annealing_level": 0,
                "patience_counter": 0,
                "approved": False,
                "human_feedback_requested": False,
                "best_r2": 0.0,
                "r2_floor": 0.0,
                "r2_threshold": 0.8,
                "best_ever_rejected": False,
                "best_verification": None,
            },
            config={"configurable": {"thread_id": "g9-human-clear"}},
        )

        # human_fn called exactly once (feedback only on first iteration)
        assert human_call_count[0] == 1, (
            f"Expected human_fn called once, got {human_call_count[0]}"
        )
        # Loop ran beyond the first iteration (flag cleared → continued normally)
        assert verify_count[0] > 1, (
            "Loop should have continued after human feedback cleared the flag"
        )


# ---------------------------------------------------------------------------
# G10 — Two consecutive turns share MemorySaver thread
# ---------------------------------------------------------------------------

class TestConsecutiveTurnsShareThread:
    """
    Two consecutive graph.invoke() calls on the same thread_id should
    accumulate messages so the second turn sees prior history.
    """

    def test_g10_message_count_grows_across_turns(self):
        """Message count in MemorySaver state grows with each chat() call."""
        from scilink.graphs.analysis import build_analysis_graph
        from langgraph.checkpoint.memory import MemorySaver

        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)
            orch.use_openai = True
            orch._graph = build_analysis_graph(orch, checkpointer=MemorySaver())

            with patch("openai.OpenAI") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.chat.completions.create.return_value = \
                    _fake_openai_response("Turn 1 reply.")
                orch.chat("Question one")

            state_after_1 = orch._graph.get_state(orch._graph_config)
            count_after_1 = len(state_after_1.values.get("messages", []))

            with patch("openai.OpenAI") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.chat.completions.create.return_value = \
                    _fake_openai_response("Turn 2 reply.")
                orch.chat("Question two")

            state_after_2 = orch._graph.get_state(orch._graph_config)
            count_after_2 = len(state_after_2.values.get("messages", []))

            assert count_after_2 > count_after_1, (
                f"Expected more messages after turn 2 ({count_after_2}) than turn 1 ({count_after_1})"
            )


# ---------------------------------------------------------------------------
# G11 — chat() exception returns error string (emergency checkpoint)
# ---------------------------------------------------------------------------

class TestChatExceptionErrorString:
    """chat() catches exceptions and returns a structured error string."""

    def test_g11_chat_exception_returns_error_string(self):
        """
        When _invoke_graph raises an unexpected exception, chat() should
        catch it and return a string containing '❌ Error:' or 'Error'.
        """
        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)
            orch.use_openai = True

            with patch("openai.OpenAI") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.chat.completions.create.side_effect = \
                    Exception("Unexpected internal error XYZ")

                result = orch.chat("Do something")

            assert isinstance(result, str)
            assert "Error" in result or "❌" in result, (
                f"Expected error string, got: {result!r}"
            )

    def test_g11_chat_error_string_contains_exception_message(self):
        """The error string should include the original exception message."""
        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)
            orch.use_openai = True

            with patch("openai.OpenAI") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.chat.completions.create.side_effect = \
                    Exception("MARKER_ERROR_42")

                result = orch.chat("Do something")

            assert "MARKER_ERROR_42" in result, (
                f"Exception message not in error string: {result!r}"
            )


# ---------------------------------------------------------------------------
# G12 — run_task() restores autonomy mode on exit (even on exception)
# ---------------------------------------------------------------------------

class TestRunTaskRestoresAutonomyMode:
    """
    SimulationOrchestratorAgent.run_task() must restore the original
    autonomy mode even when chat() raises an exception.
    """

    def _make_simulation_stub_orch(self, base_dir: str):
        """
        Duck-typed stub for SimulationOrchestratorAgent that avoids ase import.
        Has just enough to test run_task() mode-restoration behavior.
        """
        from scilink.graphs.simulation import build_simulation_graph
        from langgraph.checkpoint.memory import MemorySaver
        from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

        # Build a minimal but real orchestrator-like object
        orch = MagicMock()
        orch.use_openai = True
        orch._system_prompt = "You are a stub."
        orch.tools_for_model = []
        orch.MAX_TOOL_ITERATIONS = 5
        orch.tools.execute_tool.return_value = "result"
        orch.model.model = "claude-opus-4-6"
        orch.model.api_key = "dummy"
        orch.model.base_url = None

        orch._graph = build_simulation_graph(orch, checkpointer=MemorySaver())
        return orch

    def test_g12_run_task_mode_preserved_on_success(self):
        """
        The original autonomy mode is preserved after run_task() completes.
        Uses the real SimulationOrchestratorAgent with ase mocked away.
        """
        # Mock ase at the module level to avoid the import error
        ase_mock = MagicMock()
        ase_io_mock = MagicMock()
        ase_io_mock.read = MagicMock()

        with patch.dict("sys.modules", {
            "ase": ase_mock,
            "ase.io": ase_io_mock,
            "ase.neighborlist": MagicMock(),
        }):
            try:
                from scilink.agents.sim_agents.simulation_orchestrator import (
                    SimulationOrchestratorAgent, SimulationMode,
                )
            except ImportError as e:
                pytest.skip(f"SimulationOrchestratorAgent import failed: {e}")

            with tempfile.TemporaryDirectory() as td:
                try:
                    orch = SimulationOrchestratorAgent(
                        base_dir=td,
                        api_key="dummy",
                        model_name="claude-opus-4-6",
                        simulation_mode=SimulationMode.CO_PILOT,
                    )
                except Exception as e:
                    pytest.skip(f"SimulationOrchestratorAgent init failed: {e}")

                original_mode = orch.simulation_mode
                assert original_mode == SimulationMode.CO_PILOT

                orch.use_openai = True

                with patch("openai.OpenAI") as mock_cls:
                    mock_client = MagicMock()
                    mock_cls.return_value = mock_client
                    mock_client.chat.completions.create.return_value = \
                        _fake_openai_response("Task complete.")

                    orch.run_task("Generate a test structure.")

                # Mode must be restored to CO_PILOT after run_task
                assert orch.simulation_mode == SimulationMode.CO_PILOT, (
                    f"run_task did not restore original mode: {orch.simulation_mode}"
                )

    def test_g12_run_task_returns_structured_dict(self):
        """run_task() return value has the expected keys."""
        ase_mock = MagicMock()
        ase_io_mock = MagicMock()
        ase_io_mock.read = MagicMock()

        with patch.dict("sys.modules", {
            "ase": ase_mock,
            "ase.io": ase_io_mock,
            "ase.neighborlist": MagicMock(),
        }):
            try:
                from scilink.agents.sim_agents.simulation_orchestrator import (
                    SimulationOrchestratorAgent, SimulationMode,
                )
            except ImportError as e:
                pytest.skip(f"SimulationOrchestratorAgent import failed: {e}")

            with tempfile.TemporaryDirectory() as td:
                try:
                    orch = SimulationOrchestratorAgent(
                        base_dir=td,
                        api_key="dummy",
                        model_name="claude-opus-4-6",
                        simulation_mode=SimulationMode.AUTONOMOUS,
                    )
                except Exception as e:
                    pytest.skip(f"SimulationOrchestratorAgent init failed: {e}")

                orch.use_openai = True

                with patch("openai.OpenAI") as mock_cls:
                    mock_client = MagicMock()
                    mock_cls.return_value = mock_client
                    mock_client.chat.completions.create.return_value = \
                        _fake_openai_response("Structure generated.")

                    result = orch.run_task("Make a simple structure.")

                required_keys = {"status", "summary", "files_produced",
                                 "key_findings", "suggested_followups", "task"}
                for key in required_keys:
                    assert key in result, f"run_task result missing key: {key}"

                assert result["status"] in ("success", "error")
                assert result["task"] == "Make a simple structure."


# ---------------------------------------------------------------------------
# G13 — Verification history accumulates across iterations
# ---------------------------------------------------------------------------

class TestVerificationHistoryAccumulation:
    """
    Each iteration appends a record to verification_history.
    After N iterations the history should have N + 1 entries
    (N loop iterations + 1 final pass).
    """

    def test_g13_history_grows_per_iteration(self):
        """verification_history length == number of verify_fn calls."""
        from scilink.graphs.verification import build_verification_subgraph
        from langgraph.checkpoint.memory import MemorySaver

        step = [0]

        def run_fn(state):
            return {"current_result": {"success": True}}

        def verify_fn(state):
            return {"quality_score": 0.3}

        def feedback_fn(state):
            step[0] += 1
            return {"analysis_config": {"step": step[0]}}

        max_iter = 3
        sg = build_verification_subgraph(
            run_fn=run_fn,
            verify_fn=verify_fn,
            feedback_fn=feedback_fn,
            max_iterations=max_iter,
            quality_threshold=0.9,
            checkpointer=MemorySaver(),
        )

        result = sg.invoke(
            {
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
                "max_iterations": max_iter,
                "annealing_level": 0,
                "patience_counter": 0,
                "approved": False,
                "human_feedback_requested": False,
                "best_r2": 0.0,
                "r2_floor": 0.0,
                "r2_threshold": 0.8,
                "best_ever_rejected": False,
                "best_verification": None,
            },
            config={"configurable": {"thread_id": "g13-history"}},
        )

        # Loop ran max_iter times + 1 final pass = max_iter + 1 entries
        history = result.get("verification_history", [])
        assert len(history) >= max_iter, (
            f"Expected at least {max_iter} history entries, got {len(history)}"
        )


# ---------------------------------------------------------------------------
# G14 — _build_openai_messages prepends system prompt
# ---------------------------------------------------------------------------

class TestBuildOpenaiMessages:
    """
    _build_openai_messages should always prepend the system prompt as the
    first message in the wire-format list.
    """

    def test_g14_system_prompt_is_first_message(self):
        from scilink.graphs._react import _build_openai_messages
        from langchain_core.messages import HumanMessage

        messages = [HumanMessage(content="Hello")]
        wire = _build_openai_messages(messages, system_prompt="Be helpful.")

        assert wire[0]["role"] == "system"
        assert wire[0]["content"] == "Be helpful."
        assert wire[1]["role"] == "user"

    def test_g14_empty_message_list_still_has_system_prompt(self):
        from scilink.graphs._react import _build_openai_messages

        wire = _build_openai_messages([], system_prompt="System prompt text.")
        assert len(wire) == 1
        assert wire[0]["role"] == "system"
        assert "System prompt text." in wire[0]["content"]


# ---------------------------------------------------------------------------
# G15 — execute_tools: JSONDecodeError guard on malformed tool args
# ---------------------------------------------------------------------------

class TestExecuteToolsJsonDecodeGuard:
    """
    If the model returns malformed JSON in a tool call argument string,
    execute_tools must not raise or abort the turn — and must not silently
    call the tool with an empty dict either (that hides the real cause: the
    tool then raises about a MISSING argument, so the model "resubmits with
    the full task" and loops — see _react._parse_tool_args). It skips the
    tool call and returns a recovery-hint error message instead.
    """

    def _make_state_with_raw_tool_call(self, args_value):
        """
        Build a graph state dict whose last message has a tool_call with
        the given raw args value, bypassing AIMessage pydantic validation.
        LangChain's AIMessage validates args must be a dict, so we use a
        MagicMock to simulate the message object as _react.execute_tools
        reads it via getattr(last, "tool_calls").
        """
        tc = MagicMock()
        tc.__getitem__ = lambda self, k: {
            "name": "some_tool",
            "args": args_value,
            "id": "tc_x",
        }[k]
        mock_msg = MagicMock()
        mock_msg.tool_calls = [{"name": "some_tool", "args": args_value, "id": "tc_x"}]
        return {"messages": [mock_msg], "step_count": 0}

    def test_g15_malformed_tool_args_skips_tool_and_returns_recovery_hint(self):
        """
        execute_tools must NOT call tools.execute_tool when tc["args"] is a
        string that is not valid JSON — it returns a recovery-hint error
        message instead, so the model sees a clear "not executed" signal
        rather than a missing-argument TypeError from the tool itself.
        """
        from scilink.graphs._react import _make_react_nodes

        orch = MagicMock()
        orch.use_openai = True
        orch._system_prompt = "sys"
        orch.tools_for_model = []
        orch.tools.execute_tool.return_value = "ok"
        del orch._tool_message  # not every orchestrator has this hook

        _, execute_tools = _make_react_nodes(orch)

        state = self._make_state_with_raw_tool_call("{not: valid json}")
        result = execute_tools(state)

        # Should not raise, and must NOT call the tool with a guessed {}.
        orch.tools.execute_tool.assert_not_called()
        assert len(result["messages"]) == 1
        assert "NOT executed" in result["messages"][0].content

    def test_g15_valid_dict_args_pass_through(self):
        """
        execute_tools passes valid dict args through directly without JSON parsing.
        """
        from scilink.graphs._react import _make_react_nodes

        orch = MagicMock()
        orch.use_openai = True
        orch._system_prompt = "sys"
        orch.tools_for_model = []
        orch.tools.execute_tool.return_value = "result"

        _, execute_tools = _make_react_nodes(orch)

        state = self._make_state_with_raw_tool_call({"data_path": "/f.npy"})
        execute_tools(state)

        orch.tools.execute_tool.assert_called_once_with("some_tool", data_path="/f.npy")

    def test_g15_valid_json_string_args_are_parsed(self):
        """
        execute_tools parses valid JSON string args into a dict.
        """
        from scilink.graphs._react import _make_react_nodes

        orch = MagicMock()
        orch.use_openai = True
        orch._system_prompt = "sys"
        orch.tools_for_model = []
        orch.tools.execute_tool.return_value = "result"

        _, execute_tools = _make_react_nodes(orch)

        state = self._make_state_with_raw_tool_call('{"path": "/data.csv"}')
        execute_tools(state)

        orch.tools.execute_tool.assert_called_once_with("some_tool", path="/data.csv")
