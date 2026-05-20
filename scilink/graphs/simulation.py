"""
scilink.graphs.simulation
=========================

LangGraph graph for SimulationOrchestratorAgent.

Same ReAct topology as the analysis and planning graphs; nodes close over the
orchestrator instance so no new LLM dependencies are introduced.

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

    from scilink.graphs.simulation import build_simulation_graph

    # Interactive — passes user input then streams responses
    graph = build_simulation_graph(orchestrator_instance)
    result = graph.invoke(
        {"messages": [{"role": "user", "content": user_input}], ...},
        config={"configurable": {"thread_id": "session-123"}},
    )

    # Non-interactive (run_task) — uses the same graph with autonomy_mode=autonomous
    result = graph.invoke(
        {"messages": [{"role": "user", "content": task_prompt}],
         "autonomy_mode": "autonomous", ...},
        config={"configurable": {"thread_id": "task-<uuid>"}},
    )
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from scilink.graphs.state import SimulationOrchestratorState

if TYPE_CHECKING:
    from scilink.agents.sim_agents.simulation_orchestrator import (
        SimulationOrchestratorAgent,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------


def _make_nodes(orch: "SimulationOrchestratorAgent"):
    """Return node functions that close over *orch*."""

    def call_model(state: SimulationOrchestratorState) -> Dict[str, Any]:
        """Invoke the LLM with the current message history and bound tools."""
        messages = list(state["messages"])
        print("  ⏳ Waiting for orchestrator response ...")

        if orch.use_openai:
            from openai import OpenAI
            from scilink.graphs.analysis import _build_openai_messages, _openai_message_to_langchain

            client = OpenAI(
                api_key=orch.model.api_key,
                base_url=orch.model.base_url,
                timeout=120.0,
            )
            full_messages = _build_openai_messages(messages, orch._system_prompt)

            try:
                response = client.chat.completions.create(
                    model=orch.model.model,
                    messages=full_messages,
                    tools=orch.tools_for_model,
                    tool_choice="auto",
                )
            except Exception as e:
                if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    logger.warning("API timeout in call_model — raising to let graph retry")
                raise

            message = response.choices[0].message
            # Handle empty-content follow-up (mirrors old loop behavior)
            ai_msg = _openai_message_to_langchain(message)
            if not ai_msg.content and not ai_msg.tool_calls:
                # Ask for summary — append a synthetic user message so the
                # next call_model invocation returns a human-readable response.
                from langchain_core.messages import HumanMessage
                return {
                    "messages": [
                        ai_msg,
                        HumanMessage(
                            content="Please briefly summarize what you just did and suggest next steps."
                        ),
                    ]
                }
            return {"messages": [ai_msg]}

        else:
            from scilink.wrappers.litellm_wrapper import litellm_completion
            from scilink.graphs.analysis import _build_openai_messages, _litellm_message_to_langchain

            full_messages = _build_openai_messages(messages, orch._system_prompt)

            try:
                response = litellm_completion(
                    model=orch.model.model,
                    messages=full_messages,
                    tools=orch.tools_for_model,
                    tool_choice="auto",
                    api_key=orch.model.api_key,
                    api_base=orch.model.base_url,
                    timeout=120,
                    request_timeout=120,
                )
            except Exception as e:
                if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    logger.warning("API timeout in call_model — raising to let graph retry")
                raise

            ai_msg = _litellm_message_to_langchain(response.choices[0].message)
            if not ai_msg.content and not ai_msg.tool_calls:
                from langchain_core.messages import HumanMessage
                return {
                    "messages": [
                        ai_msg,
                        HumanMessage(
                            content="Please briefly summarize what you just did and suggest next steps."
                        ),
                    ]
                }
            return {"messages": [ai_msg]}

    def execute_tools(state: SimulationOrchestratorState) -> Dict[str, Any]:
        """Execute tool calls from the last AI message."""
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []

        results = []
        for tc in tool_calls:
            func_name = tc["name"]
            args = tc["args"] if isinstance(tc["args"], dict) else json.loads(tc["args"])
            print(f"  🔧 Calling tool: {func_name}")
            result = orch.tools.execute_tool(func_name, **args)
            results.append(
                ToolMessage(content=result, tool_call_id=tc["id"], name=func_name)
            )

        current_step = state.get("step_count", 0)
        return {
            "messages": results,
            "step_count": current_step + 1,
        }

    return call_model, execute_tools


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _should_continue(state: SimulationOrchestratorState, max_steps: int) -> str:
    step_count = state.get("step_count", 0)
    if step_count >= max_steps:
        logger.warning("⚠️ Maximum tool iterations (%d) reached. Routing to END.", max_steps)
        return END

    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "execute_tools"
    return END


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_simulation_graph(
    orch: "SimulationOrchestratorAgent",
    checkpointer: Any = None,
) -> Any:
    """
    Construct and compile the simulation StateGraph for *orch*.

    Parameters
    ----------
    orch:
        The ``SimulationOrchestratorAgent`` instance.
    checkpointer:
        LangGraph checkpointer.  Defaults to a new ``MemorySaver`` instance.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    call_model, execute_tools = _make_nodes(orch)

    def should_continue(state: SimulationOrchestratorState) -> str:
        # Read live: orch.max_iterations is per-call (run_task can override
        # it for the duration of one delegation).
        max_steps = getattr(orch, "max_iterations", None) or getattr(
            orch, "MAX_TOOL_ITERATIONS", 20)
        return _should_continue(state, max_steps)

    builder = StateGraph(SimulationOrchestratorState)
    builder.add_node("call_model", call_model)
    builder.add_node("execute_tools", execute_tools)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        should_continue,
        {"execute_tools": "execute_tools", END: END},
    )
    builder.add_edge("execute_tools", "call_model")

    return builder.compile(checkpointer=checkpointer)
