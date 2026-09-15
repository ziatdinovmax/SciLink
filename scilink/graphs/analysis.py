"""
scilink.graphs.analysis
=======================

LangGraph ReAct graph for ``AnalysisOrchestratorAgent``.

Graph topology
--------------

    [START]
       │
       ▼
  [call_model]  ── tool_calls present? ──► [execute_tools]
       ▲                                          │
       └──────────────────────────────────────────┘
       │
       └── no tool_calls OR step_count >= MAX_STEPS ──► [END]

Usage
-----

    from scilink.graphs.analysis import build_analysis_graph

    graph = build_analysis_graph(orchestrator_instance)
    result = graph.invoke(
        {"messages": [{"role": "user", "content": user_input}], ...},
        config={"configurable": {"thread_id": "session-123"}},
    )

Implementation
--------------

All node logic, routing, and LLM dispatch live in ``scilink.graphs._react``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scilink.graphs._react import build_react_graph
from scilink.graphs.state import AnalysisOrchestratorState

if TYPE_CHECKING:
    from scilink.agents.exp_agents.analysis_orchestrator import AnalysisOrchestratorAgent


def build_analysis_graph(
    orch: "AnalysisOrchestratorAgent",
    checkpointer: Any = None,
) -> Any:
    """Construct and compile the analysis ReAct graph for *orch*."""
    return build_react_graph(orch, AnalysisOrchestratorState, checkpointer)
