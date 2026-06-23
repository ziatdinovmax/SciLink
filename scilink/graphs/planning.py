"""
scilink.graphs.planning
=======================

LangGraph ReAct graph for ``PlanningOrchestratorAgent``.

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

    from scilink.graphs.planning import build_planning_graph

    graph = build_planning_graph(orchestrator_instance)
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
from scilink.graphs.state import PlanningOrchestratorState

if TYPE_CHECKING:
    from scilink.agents.planning_agents.planning_orchestrator import PlanningOrchestratorAgent


def build_planning_graph(
    orch: "PlanningOrchestratorAgent",
    checkpointer: Any = None,
) -> Any:
    """Construct and compile the planning ReAct graph for *orch*."""
    return build_react_graph(orch, PlanningOrchestratorState, checkpointer)
