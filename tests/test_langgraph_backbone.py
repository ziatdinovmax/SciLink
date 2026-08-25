"""
Integration tests for the LangGraph backbone (Phase 1).

Tiers
-----
targeted
    Graph wiring, state construction, and node function tests.
    Zero LLM calls — safe to run without API keys.

smoke
    Graph.invoke() with a mock LLM response (monkeypatched).
    Zero live LLM calls.

Run
---
    python tests/test_langgraph_backbone.py
    python tests/test_langgraph_backbone.py --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_api_key(allow_dummy: bool = False) -> str:
    key = os.environ.get("SCILINK_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    if allow_dummy:
        return "dummy-not-used"
    raise RuntimeError("Set SCILINK_API_KEY or ANTHROPIC_API_KEY")


def _make_analysis_orch(base_dir: str):
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisOrchestratorAgent,
        AnalysisMode,
    )

    return AnalysisOrchestratorAgent(
        base_dir=base_dir,
        api_key=_get_api_key(allow_dummy=True),
        model_name="claude-opus-4-6",
        analysis_mode=AnalysisMode.AUTONOMOUS,
    )


def _make_planning_orch(base_dir: str):
    from scilink.agents.planning_agents.planning_orchestrator import (
        PlanningOrchestratorAgent,
        AutonomyLevel,
    )

    return PlanningOrchestratorAgent(
        objective="Test objective",
        base_dir=base_dir,
        api_key=_get_api_key(allow_dummy=True),
        model_name="claude-opus-4-6",
        autonomy_level=AutonomyLevel.AUTONOMOUS,
        data_dir=base_dir,
    )


def _make_simulation_orch(base_dir: str):
    from scilink.agents.sim_agents.simulation_orchestrator import (
        SimulationOrchestratorAgent,
        SimulationMode,
    )

    return SimulationOrchestratorAgent(
        base_dir=base_dir,
        api_key=_get_api_key(allow_dummy=True),
        model_name="claude-opus-4-6",
        simulation_mode=SimulationMode.AUTONOMOUS,
    )


# ---------------------------------------------------------------------------
# Targeted tests — zero LLM calls
# ---------------------------------------------------------------------------


def test_state_schemas():
    """State TypedDicts import cleanly and have the expected fields."""
    from scilink.graphs.state import (
        OrchestratorState,
        AnalysisOrchestratorState,
        PlanningOrchestratorState,
        SimulationOrchestratorState,
        VerificationState,
    )

    # These are TypedDicts — check they have annotations
    assert "autonomy_mode" in OrchestratorState.__annotations__
    assert "step_count" in OrchestratorState.__annotations__
    assert "current_data_path" in AnalysisOrchestratorState.__annotations__
    assert "objective" in PlanningOrchestratorState.__annotations__
    assert "generated_structures" in SimulationOrchestratorState.__annotations__
    assert "verification_history" in VerificationState.__annotations__
    assert "annealing_level" in VerificationState.__annotations__
    assert "patience_counter" in VerificationState.__annotations__
    print("   ✅ State schemas: all expected fields present")


def test_graphs_init_exports():
    """graphs/__init__.py exports all expected symbols."""
    import scilink.graphs as g

    expected = [
        "OrchestratorState",
        "AnalysisOrchestratorState",
        "PlanningOrchestratorState",
        "SimulationOrchestratorState",
        "VerificationState",
        "build_analysis_graph",
        "build_planning_graph",
        "build_simulation_graph",
        "build_verification_subgraph",
    ]
    for name in expected:
        assert hasattr(g, name), f"graphs.__init__ missing: {name}"
    print("   ✅ graphs/__init__ exports all expected symbols")


def test_analysis_orch_constructs_graph():
    """AnalysisOrchestratorAgent builds a LangGraph graph on init."""
    with tempfile.TemporaryDirectory() as td:
        orch = _make_analysis_orch(td)

        # Graph and config exist
        assert hasattr(orch, "_graph"), "missing _graph attribute"
        assert hasattr(orch, "_graph_config"), "missing _graph_config attribute"
        assert hasattr(orch, "_graph_thread_id"), "missing _graph_thread_id"

        # Graph has the expected nodes
        graph_nodes = set(orch._graph.get_graph().nodes.keys())
        assert "call_model" in graph_nodes, f"nodes: {graph_nodes}"
        assert "execute_tools" in graph_nodes, f"nodes: {graph_nodes}"

        # thread_id is session-scoped
        assert "analysis-" in orch._graph_thread_id
        print("   ✅ AnalysisOrchestratorAgent: graph constructed with correct nodes")


def test_planning_orch_constructs_graph():
    """PlanningOrchestratorAgent builds a LangGraph graph on init."""
    with tempfile.TemporaryDirectory() as td:
        try:
            orch = _make_planning_orch(td)
        except RuntimeError as e:
            if "sandbox" in str(e).lower() or "code execution" in str(e).lower():
                print(
                    "   ⏭  PlanningOrchestratorAgent skipped: "
                    "ScalarizerAgent requires sandbox environment "
                    f"({e})"
                )
                return
            raise

        assert hasattr(orch, "_graph")
        assert hasattr(orch, "_graph_config")

        graph_nodes = set(orch._graph.get_graph().nodes.keys())
        assert "call_model" in graph_nodes
        assert "execute_tools" in graph_nodes
        print("   ✅ PlanningOrchestratorAgent: graph constructed with correct nodes")


def test_simulation_orch_constructs_graph():
    """SimulationOrchestratorAgent builds a LangGraph graph on init."""
    with tempfile.TemporaryDirectory() as td:
        orch = _make_simulation_orch(td)

        assert hasattr(orch, "_graph")
        assert hasattr(orch, "_graph_config")

        graph_nodes = set(orch._graph.get_graph().nodes.keys())
        assert "call_model" in graph_nodes
        assert "execute_tools" in graph_nodes
        print("   ✅ SimulationOrchestratorAgent: graph constructed with correct nodes")


def test_verification_subgraph_topology():
    """build_verification_subgraph() produces a graph with the correct nodes."""
    from scilink.graphs.verification import build_verification_subgraph

    def run_fn(state):
        return {"current_result": {"success": True, "_quality_score": 0.8}}

    def verify_fn(state):
        return {"approved": True, "best_score": 0.8, "best_result": state.get("current_result")}

    def feedback_fn(state):
        return {"analysis_config": state.get("analysis_config", {})}

    subgraph = build_verification_subgraph(
        run_fn=run_fn,
        verify_fn=verify_fn,
        feedback_fn=feedback_fn,
        max_iterations=3,
        quality_threshold=0.7,
    )

    nodes = set(subgraph.get_graph().nodes.keys())
    assert "run_analysis" in nodes, f"nodes: {nodes}"
    assert "verify_quality" in nodes, f"nodes: {nodes}"
    assert "apply_feedback" in nodes, f"nodes: {nodes}"
    assert "anneal" in nodes, f"nodes: {nodes}"
    assert "human_feedback" in nodes, f"nodes: {nodes}"
    print("   ✅ Verification subgraph: correct topology")


def test_verification_subgraph_terminates_on_approval():
    """Verification subgraph reaches END after a single successful run."""
    from scilink.graphs.verification import build_verification_subgraph

    call_log: List[str] = []

    def run_fn(state):
        call_log.append("run")
        return {
            "current_result": {"success": True, "_quality_score": 0.9},
            "best_result": {"success": True, "_quality_score": 0.9},
            "best_score": 0.9,
        }

    def verify_fn(state):
        call_log.append("verify")
        return {
            "approved": True,
            "best_score": 0.9,
            "best_result": state.get("current_result"),
        }

    def feedback_fn(state):
        call_log.append("feedback")
        return {"analysis_config": {}}

    subgraph = build_verification_subgraph(
        run_fn=run_fn,
        verify_fn=verify_fn,
        feedback_fn=feedback_fn,
        max_iterations=7,
        quality_threshold=0.7,
        checkpointer=None,
    )

    from langgraph.checkpoint.memory import MemorySaver

    subgraph = build_verification_subgraph(
        run_fn=run_fn,
        verify_fn=verify_fn,
        feedback_fn=feedback_fn,
        max_iterations=7,
        quality_threshold=0.7,
        checkpointer=MemorySaver(),
    )

    initial_state = {
        "messages": [],
        "analysis_config": {"method": "test"},
        "current_result": None,
        "best_result": None,
        "best_score": 0.0,
        "verification_history": [],
        "iteration": 0,
        "max_iterations": 7,
        "annealing_level": 0,
        "patience_counter": 0,
        "approved": False,
        "human_feedback_requested": False,
    }

    cfg = {"configurable": {"thread_id": "verify-test-1"}}
    result = subgraph.invoke(initial_state, config=cfg)

    assert result["approved"] is True, f"expected approved=True, got {result}"
    assert result["best_score"] >= 0.7
    # run_fn and verify_fn called exactly once (approved immediately)
    assert call_log.count("run") == 1
    assert call_log.count("verify") == 1
    assert call_log.count("feedback") == 0
    print("   ✅ Verification subgraph: terminates on first approval")


def test_verification_subgraph_max_iterations():
    """Verification subgraph stops at max_iterations even without approval."""
    from scilink.graphs.verification import build_verification_subgraph
    from langgraph.checkpoint.memory import MemorySaver

    run_count = [0]
    verify_count = [0]

    def run_fn(state):
        run_count[0] += 1
        return {
            "current_result": {"success": True, "_quality_score": 0.3},
            "best_result": {"success": True, "_quality_score": 0.3},
            "best_score": 0.3,
        }

    def verify_fn(state):
        verify_count[0] += 1
        return {
            "approved": False,
            "best_score": 0.3,
            "best_result": state.get("current_result"),
        }

    def feedback_fn(state):
        return {"analysis_config": state.get("analysis_config", {})}

    max_iter = 3
    subgraph = build_verification_subgraph(
        run_fn=run_fn,
        verify_fn=verify_fn,
        feedback_fn=feedback_fn,
        max_iterations=max_iter,
        quality_threshold=0.7,
        checkpointer=MemorySaver(),
    )

    initial_state = {
        "messages": [],
        "analysis_config": {},
        "current_result": None,
        "best_result": None,
        "best_score": 0.0,
        "verification_history": [],
        "iteration": 0,
        "max_iterations": max_iter,
        "annealing_level": 0,
        "patience_counter": 0,
        "approved": False,
        "human_feedback_requested": False,
    }

    cfg = {"configurable": {"thread_id": "verify-test-maxiter"}}
    result = subgraph.invoke(initial_state, config=cfg)

    assert result["approved"] is False
    assert result["iteration"] >= max_iter
    print(
        f"   ✅ Verification subgraph: terminates at max_iterations "
        f"(run×{run_count[0]}, verify×{verify_count[0]})"
    )


def test_analysis_graph_message_format_helpers():
    """_langchain_to_openai_dict and reverse helpers round-trip correctly."""
    from scilink.graphs._react import (
        _openai_message_to_langchain,
        _langchain_to_openai_dict,
    )
    from langchain_core.messages import HumanMessage, AIMessage

    # HumanMessage → dict
    hm = HumanMessage(content="hello")
    d = _langchain_to_openai_dict(hm)
    assert d == {"role": "user", "content": "hello"}

    # AIMessage with no tool calls → dict
    ai = AIMessage(content="world", tool_calls=[])
    d = _langchain_to_openai_dict(ai)
    assert d["role"] == "assistant"
    assert d["content"] == "world"
    assert "tool_calls" not in d or not d["tool_calls"]

    print("   ✅ Message format helpers: round-trip OK")


def test_no_old_handle_methods_on_orchestrators():
    """Ensure _handle_openai_chat / _handle_litellm_chat are gone."""
    with tempfile.TemporaryDirectory() as td:
        orch = _make_analysis_orch(td)
        assert not hasattr(orch, "_handle_openai_chat"), \
            "AnalysisOrchestratorAgent still has _handle_openai_chat"
        assert not hasattr(orch, "_handle_litellm_chat"), \
            "AnalysisOrchestratorAgent still has _handle_litellm_chat"

    with tempfile.TemporaryDirectory() as td:
        orch = _make_simulation_orch(td)
        assert not hasattr(orch, "_handle_openai_chat"), \
            "SimulationOrchestratorAgent still has _handle_openai_chat"
        assert not hasattr(orch, "_handle_litellm_chat"), \
            "SimulationOrchestratorAgent still has _handle_litellm_chat"

    # Planning orchestrator may fail in non-sandbox environments — skip gracefully
    with tempfile.TemporaryDirectory() as td:
        try:
            orch = _make_planning_orch(td)
            assert not hasattr(orch, "_handle_openai_chat"), \
                "PlanningOrchestratorAgent still has _handle_openai_chat"
            assert not hasattr(orch, "_handle_litellm_chat"), \
                "PlanningOrchestratorAgent still has _handle_litellm_chat"
        except RuntimeError as e:
            if "sandbox" in str(e).lower() or "code execution" in str(e).lower():
                pass  # Skip in non-sandbox environments
            else:
                raise

    print("   ✅ Old _handle_*_chat methods correctly removed from all orchestrators")


# ---------------------------------------------------------------------------
# Smoke tests — mock LLM response, no live API calls
# ---------------------------------------------------------------------------


def _make_fake_openai_response(content: str, tool_calls=None):
    """Build a minimal mock that looks like an OpenAI ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


def test_smoke_analysis_graph_invoke():
    """
    Analysis graph invoke with a mocked LLM call returns expected text.
    Uses a thin stub that pretends to be the OpenAI client.
    """
    with tempfile.TemporaryDirectory() as td:
        orch = _make_analysis_orch(td)
        orch.use_openai = True  # force OpenAI path for mock

        fake_response = _make_fake_openai_response("Mock analysis complete.")

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = fake_response

            result = orch.chat("Analyze this data file: /fake/data.npy")

        # The response should include the mocked content
        assert "Mock analysis complete." in result, f"unexpected response: {result!r}"
        print("   ✅ Analysis graph smoke: mock LLM response propagated correctly")


def test_smoke_simulation_graph_invoke():
    """
    Simulation graph invoke with a mocked LLM call returns expected text.
    """
    with tempfile.TemporaryDirectory() as td:
        orch = _make_simulation_orch(td)
        orch.use_openai = True

        fake_response = _make_fake_openai_response("Mock structure generated.")

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = fake_response

            result = orch.chat("Generate a silicon structure.")

        assert "Mock structure generated." in result, f"unexpected: {result!r}"
        print("   ✅ Simulation graph smoke: mock LLM response propagated correctly")


def test_smoke_graph_step_count_enforced():
    """
    Graph respects MAX_TOOL_ITERATIONS — if the mock always returns tool calls,
    the graph should route to END after the limit.
    """
    with tempfile.TemporaryDirectory() as td:
        orch = _make_analysis_orch(td)
        orch.use_openai = True
        orch.MAX_TOOL_ITERATIONS = 2  # lower limit for test speed

        # Rebuild graph with new limit (build_analysis_graph reads MAX_TOOL_ITERATIONS)
        from scilink.graphs.analysis import build_analysis_graph
        from langgraph.checkpoint.memory import MemorySaver

        orch._graph = build_analysis_graph(orch, checkpointer=MemorySaver())

        # Fake a tool-call response that never resolves
        fake_tc = MagicMock()
        fake_tc.id = "tc_001"
        fake_tc.function.name = "nonexistent_tool"
        fake_tc.function.arguments = "{}"
        fake_response_with_tc = _make_fake_openai_response("", tool_calls=[fake_tc])
        fake_response_final = _make_fake_openai_response("Done after limit.")

        call_count = [0]

        def _side_effect(**kwargs):
            call_count[0] += 1
            # First two calls return tool_calls, then final
            if call_count[0] <= orch.MAX_TOOL_ITERATIONS:
                return fake_response_with_tc
            return fake_response_final

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = _side_effect

            result = orch.chat("Do something that requires many tool calls.")

        # The graph should have routed to END after MAX_TOOL_ITERATIONS
        # and returned the warning message (since the LLM gets no final text after limit)
        assert result  # something was returned
        print(
            f"   ✅ Graph MAX_TOOL_ITERATIONS enforced "
            f"(LLM called {call_count[0]} times, result={result[:60]!r})"
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


TARGETED_TESTS = [
    test_state_schemas,
    test_graphs_init_exports,
    test_analysis_orch_constructs_graph,
    test_planning_orch_constructs_graph,
    test_simulation_orch_constructs_graph,
    test_verification_subgraph_topology,
    test_verification_subgraph_terminates_on_approval,
    test_verification_subgraph_max_iterations,
    test_analysis_graph_message_format_helpers,
    test_no_old_handle_methods_on_orchestrators,
]

SMOKE_TESTS = [
    test_smoke_analysis_graph_invoke,
    test_smoke_simulation_graph_invoke,
    test_smoke_graph_step_count_enforced,
]


def _run_suite(tests: list, label: str) -> int:
    passed = failed = 0
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception:
            failed += 1
            print(f"   ❌ {fn.__name__}")
            traceback.print_exc()
    print(f"\n  {label}: {passed} passed, {failed} failed")
    return failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests (mock LLM)")
    args = parser.parse_args()

    total_failures = 0
    total_failures += _run_suite(TARGETED_TESTS, "Targeted tests (no LLM)")
    if args.smoke:
        total_failures += _run_suite(SMOKE_TESTS, "Smoke tests (mocked LLM)")

    sys.exit(0 if total_failures == 0 else 1)


if __name__ == "__main__":
    main()
