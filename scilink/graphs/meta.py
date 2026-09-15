"""
scilink.graphs.meta
====================

LangGraph ReAct graph for ``MetaOrchestratorAgent``.

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

    from scilink.graphs.meta import build_meta_graph

    graph = build_meta_graph(orchestrator_instance)
    result = graph.invoke(
        {"messages": [{"role": "user", "content": user_input}], ...},
        config={"configurable": {"thread_id": "session-123"}},
    )

Implementation
--------------

All node logic, routing, and LLM dispatch live in ``scilink.graphs._react``.
A ``delegate_to_*`` tool call runs synchronously inside ``execute_tools``
(via ``orch.tools.execute_tool``), exactly as it did in the old hand-rolled
loop — the child's own ``run_task`` call, human-feedback prompts included,
is unaffected by the meta's chat loop now being graph-backed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scilink.graphs._react import build_react_graph
from scilink.graphs.state import MetaOrchestratorState

if TYPE_CHECKING:
    from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent


def build_meta_graph(
    orch: "MetaOrchestratorAgent",
    checkpointer: Any = None,
) -> Any:
    """Construct and compile the meta ReAct graph for *orch*."""
    return build_react_graph(orch, MetaOrchestratorState, checkpointer)
