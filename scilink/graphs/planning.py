"""
scilink.graphs.planning
=======================

LangGraph graph for PlanningOrchestratorAgent.

Same ReAct topology as the analysis graph; nodes close over the orchestrator
instance so no new LLM dependencies are introduced.

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
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from scilink.graphs.state import PlanningOrchestratorState

if TYPE_CHECKING:
    from scilink.agents.planning_agents.planning_orchestrator import (
        PlanningOrchestratorAgent,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------


def _make_nodes(orch: "PlanningOrchestratorAgent"):
    """Return node functions that close over *orch*."""

    def call_model(state: PlanningOrchestratorState) -> Dict[str, Any]:
        """Invoke the LLM with the current message history and bound tools."""
        messages = list(state["messages"])
        print("  ⏳ Waiting for orchestrator response ...")

        if orch.use_openai:
            from openai import OpenAI
            from scilink.graphs.analysis import _build_openai_messages, _openai_message_to_langchain

            client = OpenAI(
                api_key=orch.model.api_key,
                base_url=orch.model.base_url,
            )
            full_messages = _build_openai_messages(messages, orch._system_prompt)

            # Compress large tool results before sending to avoid context overflow.
            _compress_messages_inplace(full_messages)

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
            if getattr(message, "tool_calls", None):
                orch._print_assistant_reasoning(message.content)
            return {"messages": [_openai_message_to_langchain(message)]}

        else:
            from scilink.wrappers.litellm_wrapper import litellm_completion
            from scilink.graphs.analysis import _build_openai_messages, _litellm_message_to_langchain

            full_messages = _build_openai_messages(messages, orch._system_prompt)
            _compress_messages_inplace(full_messages)

            try:
                response = litellm_completion(
                    model=orch.model.model,
                    messages=full_messages,
                    tools=orch.tools_for_model,
                    tool_choice="auto",
                    api_key=orch.model.api_key,
                    api_base=orch.model.base_url,
                )
            except Exception as e:
                if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    logger.warning("API timeout in call_model — raising to let graph retry")
                raise

            message = response.choices[0].message
            if getattr(message, "tool_calls", None):
                orch._print_assistant_reasoning(getattr(message, "content", None))
            return {"messages": [_litellm_message_to_langchain(message)]}

    def execute_tools(state: PlanningOrchestratorState) -> Dict[str, Any]:
        """Execute tool calls from the last AI message."""
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        # finish_reason isn't carried on the LangChain AIMessage by default;
        # a truncated tool call (finish_reason="length") shows up here as a
        # json.loads failure below, same recovery path either way.

        results = []
        for tc in tool_calls:
            func_name = tc["name"]
            try:
                args = tc["args"] if isinstance(tc["args"], dict) else json.loads(tc["args"])
            except (TypeError, json.JSONDecodeError):
                print(f"  🔧 Calling tool: {func_name}")
                print("    ⚠️ Malformed/truncated tool arguments — "
                      "returning recovery hint to the model")
                result = json.dumps({
                    "status": "error",
                    "message": (
                        "Tool call discarded: the arguments string was not "
                        "valid JSON — typically truncation at the output-token "
                        "limit or broken escaping of quotes/newlines inside a "
                        "large string value. The tool was NOT executed. Do not "
                        "retry with one large argument. For large text content, "
                        "write the file in chunks: save_file with the first "
                        "chunk, then append_file for each remaining chunk."
                    ),
                })
            else:
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


def _should_continue(state: PlanningOrchestratorState, max_steps: int) -> str:
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


def build_planning_graph(
    orch: "PlanningOrchestratorAgent",
    checkpointer: Any = None,
) -> Any:
    """
    Construct and compile the planning StateGraph for *orch*.

    Parameters
    ----------
    orch:
        The ``PlanningOrchestratorAgent`` instance.
    checkpointer:
        LangGraph checkpointer.  Defaults to a new ``MemorySaver`` instance.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    call_model, execute_tools = _make_nodes(orch)

    def should_continue(state: PlanningOrchestratorState) -> str:
        # Read live: orch.max_iterations is per-call (run_task can override
        # it for the duration of one delegation).
        max_steps = getattr(orch, "max_iterations", None) or 20
        return _should_continue(state, max_steps)

    builder = StateGraph(PlanningOrchestratorState)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compress_messages_inplace(messages: list, threshold: int = 100_000) -> None:
    """
    Mirror ``PlanningOrchestratorAgent._compress_large_tool_results``.

    Truncates tool messages > 30 K chars when total context exceeds 100 K.
    Operates in-place on the wire-format dict list produced by
    ``_build_openai_messages``.
    """
    total = sum(len(m.get("content", "") or "") for m in messages)
    if total <= threshold:
        return

    compressed = 0
    for msg in messages[:-2]:
        if msg.get("role") == "tool" and len(msg.get("content", "")) > 30_000:
            original_len = len(msg["content"])
            msg["content"] = (
                msg["content"][:5_000]
                + f"\n\n... ({original_len - 5_000} chars truncated from history. "
                "Use read_file to re-read the full content only if "
                "the truncated portion above is insufficient for your current task.)"
            )
            compressed += 1

    if compressed:
        new_total = sum(len(m.get("content", "") or "") for m in messages)
        logger.info(
            "Compressed %d large tool result(s) (%d → %d chars)",
            compressed, total, new_total,
        )
