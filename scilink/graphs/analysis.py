"""
scilink.graphs.analysis
=======================

LangGraph graph for AnalysisOrchestratorAgent.

This module is responsible solely for graph topology and node wiring.
All LLM calls and tool dispatch delegate back to the orchestrator instance
that constructs the graph — no new LLM dependencies are introduced.

Graph topology (ReAct loop)
---------------------------

    [START]
       │
       ▼
  [call_model]  ── tool_calls present? ──► [execute_tools]
       ▲                                          │
       └──────────────────────────────────────────┘
       │
       └── no tool_calls OR step_count >= MAX_STEPS ──► [END]

The ``step_count`` field in ``AnalysisOrchestratorState`` replaces the
``while iteration < MAX_TOOL_ITERATIONS`` counter from the old loop.  The
routing function increments the count and routes to ``END`` when it reaches
the configured maximum.

Usage
-----

    from scilink.graphs.analysis import build_analysis_graph

    # Pass the orchestrator instance; nodes close over it.
    graph = build_analysis_graph(orchestrator_instance)

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

from scilink.graphs.state import AnalysisOrchestratorState

if TYPE_CHECKING:
    from scilink.agents.exp_agents.analysis_orchestrator import AnalysisOrchestratorAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node factory — returns node functions that close over the orchestrator
# ---------------------------------------------------------------------------


def _make_nodes(orch: "AnalysisOrchestratorAgent"):
    """
    Return the ``call_model`` and ``execute_tools`` node functions, each
    closing over *orch* so they can access the LLM and tool registry without
    accepting them as arguments (which LangGraph does not support).
    """

    # ------------------------------------------------------------------
    # call_model
    # ------------------------------------------------------------------

    def call_model(state: AnalysisOrchestratorState) -> Dict[str, Any]:
        """Invoke the LLM with the current message history and bound tools."""
        # Sync any orchestrator-side state changes back into the graph state
        # (e.g., system-prompt refreshes triggered by register_tools).
        messages = list(state["messages"])

        print("  ⏳ Waiting for orchestrator response ...")

        if orch.use_openai:
            from openai import OpenAI

            client = OpenAI(
                api_key=orch.model.api_key,
                base_url=orch.model.base_url,
                timeout=120.0,
            )
            # Build the full message list including the system prompt.
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
            if getattr(message, "tool_calls", None):
                orch._print_assistant_reasoning(message.content)
            return {"messages": [_openai_message_to_langchain(message)]}

        else:
            from ..wrappers.litellm_wrapper import litellm_completion

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

            message = response.choices[0].message
            if getattr(message, "tool_calls", None):
                orch._print_assistant_reasoning(getattr(message, "content", None))
            return {"messages": [_litellm_message_to_langchain(message)]}

    # ------------------------------------------------------------------
    # execute_tools
    # ------------------------------------------------------------------

    def execute_tools(state: AnalysisOrchestratorState) -> Dict[str, Any]:
        """Execute tool calls from the last AI message."""
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []

        results = []
        for tc in tool_calls:
            func_name = tc["name"]
            args = tc["args"] if isinstance(tc["args"], dict) else json.loads(tc["args"])

            print(f"  🔧 Calling tool: {func_name}")
            result = orch.tools.execute_tool(func_name, **args)

            # orch._tool_message() upgrades an image-bearing result to a
            # multimodal content list for providers that render images in
            # tool results (see scilink/utils/tool_media.py); every other
            # result stays the plain-string content it always was.
            tm = orch._tool_message(tc["id"], result)
            results.append(
                ToolMessage(
                    content=tm["content"],
                    tool_call_id=tc["id"],
                    name=func_name,
                )
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


def _should_continue(state: AnalysisOrchestratorState, max_steps: int) -> str:
    """
    Route to ``execute_tools`` if the last message has tool calls AND we have
    not exceeded the step limit.  Otherwise route to ``END``.
    """
    step_count = state.get("step_count", 0)
    if step_count >= max_steps:
        logger.warning(
            "⚠️ Maximum tool iterations (%d) reached. Routing to END.", max_steps
        )
        return END

    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        return "execute_tools"
    return END


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_analysis_graph(
    orch: "AnalysisOrchestratorAgent",
    checkpointer: Any = None,
) -> Any:
    """
    Construct and compile the analysis StateGraph for *orch*.

    Parameters
    ----------
    orch:
        The ``AnalysisOrchestratorAgent`` instance.  Nodes close over it.
    checkpointer:
        LangGraph checkpointer.  Defaults to a new ``MemorySaver`` instance.

    Returns
    -------
    Compiled LangGraph ``CompiledGraph``.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    call_model, execute_tools = _make_nodes(orch)

    def should_continue(state: AnalysisOrchestratorState) -> str:
        # Read live: orch.max_iterations is per-call (run_task can override
        # it for the duration of one delegation), so a value baked in at
        # build time would go stale after the first override/restore.
        max_steps = getattr(orch, "max_iterations", None) or getattr(
            orch, "MAX_TOOL_ITERATIONS", 20)
        return _should_continue(state, max_steps)

    builder = StateGraph(AnalysisOrchestratorState)

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
# Helpers — message format conversion
# ---------------------------------------------------------------------------


def _build_openai_messages(
    langchain_messages: list,
    system_prompt: str,
) -> list:
    """
    Convert a LangChain message list to the OpenAI-wire dict format,
    prepending the system prompt.
    """
    result = [{"role": "system", "content": system_prompt}]
    for msg in langchain_messages:
        result.append(_langchain_to_openai_dict(msg))
    return result


def _langchain_to_openai_dict(msg: Any) -> dict:
    """Convert a LangChain message object to an OpenAI-wire dict."""
    from langchain_core.messages import (
        AIMessage as AI,
        HumanMessage,
        SystemMessage,
        ToolMessage as TM,
    )

    if isinstance(msg, HumanMessage):
        return {"role": "user", "content": msg.content}
    elif isinstance(msg, SystemMessage):
        return {"role": "system", "content": msg.content}
    elif isinstance(msg, TM):
        return {
            "role": "tool",
            "tool_call_id": msg.tool_call_id,
            "content": msg.content,
        }
    elif isinstance(msg, AI):
        d: dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": (
                            json.dumps(tc["args"])
                            if not isinstance(tc["args"], str)
                            else tc["args"]
                        ),
                    },
                }
                for tc in msg.tool_calls
            ]
        return d
    elif isinstance(msg, dict):
        # Already in wire format
        return msg
    else:
        # Best-effort fallback
        return {"role": "user", "content": str(msg)}


def _openai_message_to_langchain(msg: Any) -> AIMessage:
    """Convert an OpenAI ``ChatCompletionMessage`` to a LangChain ``AIMessage``."""
    tool_calls_raw = getattr(msg, "tool_calls", None) or []
    lc_tool_calls = [
        {
            "id": tc.id,
            "name": tc.function.name,
            "args": json.loads(tc.function.arguments)
            if tc.function.arguments
            else {},
            "type": "tool_call",
        }
        for tc in tool_calls_raw
    ]
    return AIMessage(
        content=msg.content or "",
        tool_calls=lc_tool_calls,
    )


def _litellm_message_to_langchain(msg: Any) -> AIMessage:
    """Convert a LiteLLM response message to a LangChain ``AIMessage``."""
    tool_calls_raw = getattr(msg, "tool_calls", None) or []
    lc_tool_calls = [
        {
            "id": tc.id,
            "name": tc.function.name,
            "args": json.loads(tc.function.arguments)
            if tc.function.arguments
            else {},
            "type": "tool_call",
        }
        for tc in tool_calls_raw
    ]
    content = getattr(msg, "content", None) or ""
    return AIMessage(
        content=content,
        tool_calls=lc_tool_calls,
    )
