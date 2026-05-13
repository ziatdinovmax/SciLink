"""
scilink.graphs.analysis
=======================

LangGraph graph skeleton for AnalysisOrchestratorAgent.

This module defines the *shape* of the analysis graph — nodes, edges, and
the compiled graph object — but none of the nodes contain real logic yet.
The existing orchestrator chat loop is untouched; this sits alongside it
as infrastructure ready to be wired in.

Graph topology
--------------

    [START]
       │
       ▼
  [call_model]  ── tool_calls present? ──► [execute_tools]
       ▲                                          │
       └──────────────────────────────────────────┘
       │
       └── no tool_calls ──► [END]

This is a standard ReAct loop.  The conditional edge uses LangGraph's
built-in ``tools_condition`` from ``langgraph.prebuilt``, which handles
AIMessage objects, raw dicts, and BaseModel state consistently.

Usage (not yet wired into the orchestrator)::

    from scilink.graphs.analysis import graph, AnalysisState

    # Invoke for a single turn (stateless — no checkpointer attached yet)
    result = graph.invoke(initial_state)
"""

from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

from scilink.graphs.state import AnalysisState

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def call_model(state: AnalysisState) -> Dict[str, Any]:
    """
    LLM call node.

    Will call the LLM with the current message history and registered tools,
    then return the assistant message to be appended to state.

    Not yet implemented — placeholder only.
    """
    raise NotImplementedError(
        "call_model node is not yet wired to the LLM. "
        "This is scaffolding; the live orchestrator still runs its own loop."
    )


def execute_tools(state: AnalysisState) -> Dict[str, Any]:
    """
    Tool execution node.

    Will iterate over the tool calls in the last assistant message, dispatch
    each via AnalysisOrchestratorTools.execute_tool(), and return the tool
    result messages to be appended to state.

    Not yet implemented — placeholder only.
    """
    raise NotImplementedError(
        "execute_tools node is not yet wired to the tool registry. "
        "This is scaffolding; the live orchestrator still runs its own loop."
    )


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_analysis_graph():
    """
    Construct and compile the analysis StateGraph.

    Returns the compiled graph.  A checkpointer can be attached at call time::

        from langgraph.checkpoint.memory import MemorySaver
        graph = build_analysis_graph(checkpointer=MemorySaver())
    """
    builder = StateGraph(AnalysisState)

    # Nodes
    builder.add_node("call_model", call_model)
    builder.add_node("execute_tools", execute_tools)

    # Entry point
    builder.add_edge(START, "call_model")

    # After LLM: tools_condition routes to "tools" node or END.
    # We name our node "execute_tools" so we remap the default "tools" key.
    builder.add_conditional_edges(
        "call_model",
        tools_condition,
        {
            "tools": "execute_tools",
            END: END,
        },
    )

    # After tools: always loop back to the LLM
    builder.add_edge("execute_tools", "call_model")

    return builder.compile()


# Module-level compiled graph instance.
# Import this when you want a ready-to-use (but not-yet-functional) graph::
#
#     from scilink.graphs.analysis import graph
#
graph = build_analysis_graph()
