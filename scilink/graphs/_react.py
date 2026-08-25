"""
scilink.graphs._react
=====================

Shared ReAct graph backbone for all three SciLink orchestrators.

This module is intentionally absent from ``graphs/__init__.py`` exports.
Callers use ``build_analysis_graph`` / ``build_planning_graph`` /
``build_simulation_graph`` in the sibling modules.

Public interface
----------------

    build_react_graph(orch, state_type, checkpointer=None) -> CompiledGraph

All LLM routing, tool dispatch, context compression, message-format
conversion, and step-limit enforcement are hidden behind that one call.

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

Orchestrator contract
---------------------

The ``orch`` argument is duck-typed.  It must expose::

    orch.use_openai          bool
    orch.model.model         str    — model identifier
    orch.model.api_key       str
    orch.model.base_url      str | None
    orch._system_prompt      str
    orch.tools_for_model     list   — OpenAI-format tool schemas
    orch.tools.execute_tool  callable(name, **kwargs) -> str
    orch.MAX_TOOL_ITERATIONS int    (optional; defaults to 20)
    orch.max_iterations      int    (optional; per-call override, read live —
                                     takes precedence over MAX_TOOL_ITERATIONS)
    orch._print_assistant_reasoning(content)  (optional; interim 💭 display)
    orch._tool_message(tool_call_id, result)  (optional; upgrades an image-
                                     bearing tool result to a multimodal
                                     message — see scilink/utils/tool_media.py)

Behavioral notes
----------------

* ``_compress_messages_inplace`` is applied on every ``call_model`` step
  for all orchestrators.  It is a no-op unless total context exceeds
  100 K chars, so it is safe to enable unconditionally.

* When the model returns an empty message (no content, no tool calls),
  a synthetic user nudge is injected so the next step can produce a
  human-readable summary.  This guards against silent dead-ends in any
  orchestrator mode.

* ``MAX_TOOL_ITERATIONS`` is always read from the orchestrator instance
  (``getattr`` with a default of 20), so per-session overrides take
  effect without rebuilding the graph.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)

_TIMEOUT = 120.0
_MAX_STEPS_DEFAULT = 20
_COMPRESS_THRESHOLD = 100_000
_COMPRESS_TRUNCATE_AT = 5_000
_TIMEOUT_RETRIES = 3
_EMPTY_RESPONSE_NUDGE = "Please briefly summarize what you just did and suggest next steps."


# ---------------------------------------------------------------------------
# Message format helpers
# ---------------------------------------------------------------------------


def _build_openai_messages(langchain_messages: list, system_prompt: str) -> list:
    """Convert a LangChain message list to OpenAI wire format, prepending the system prompt."""
    result = [{"role": "system", "content": system_prompt}]
    for msg in langchain_messages:
        result.append(_langchain_to_openai_dict(msg))
    return result


def _langchain_to_openai_dict(msg: Any) -> dict:
    """Convert a LangChain message object to an OpenAI wire dict."""
    from langchain_core.messages import (
        AIMessage as AI,
        HumanMessage as HM,
        SystemMessage,
        ToolMessage as TM,
    )

    if isinstance(msg, HM):
        return {"role": "user", "content": msg.content}
    if isinstance(msg, SystemMessage):
        return {"role": "system", "content": msg.content}
    if isinstance(msg, TM):
        return {"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content}
    if isinstance(msg, AI):
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
    if isinstance(msg, dict):
        return msg
    return {"role": "user", "content": str(msg)}


def _openai_message_to_langchain(msg: Any) -> AIMessage:
    """Convert an OpenAI ``ChatCompletionMessage`` to a LangChain ``AIMessage``."""
    tool_calls_raw = getattr(msg, "tool_calls", None) or []
    return AIMessage(
        content=msg.content or "",
        tool_calls=[
            {
                "id": tc.id,
                "name": tc.function.name,
                "args": json.loads(tc.function.arguments) if tc.function.arguments else {},
                "type": "tool_call",
            }
            for tc in tool_calls_raw
        ],
    )


def _litellm_message_to_langchain(msg: Any) -> AIMessage:
    """Convert a LiteLLM response message to a LangChain ``AIMessage``."""
    tool_calls_raw = getattr(msg, "tool_calls", None) or []
    return AIMessage(
        content=getattr(msg, "content", None) or "",
        tool_calls=[
            {
                "id": tc.id,
                "name": tc.function.name,
                "args": json.loads(tc.function.arguments) if tc.function.arguments else {},
                "type": "tool_call",
            }
            for tc in tool_calls_raw
        ],
    )


# ---------------------------------------------------------------------------
# Context compression
# ---------------------------------------------------------------------------


def _compress_messages_inplace(messages: list, threshold: int = _COMPRESS_THRESHOLD) -> None:
    """
    Truncate oversized tool messages in a wire-format list when total context is large.

    Operates in-place on the list produced by ``_build_openai_messages``.
    Skips the two most recent messages so the model always has full current context.
    No-op when total char count is below *threshold*.
    """
    total = sum(len(m.get("content", "") or "") for m in messages)
    if total <= threshold:
        return

    compressed = 0
    for msg in messages[:-2]:
        if msg.get("role") == "tool" and len(msg.get("content", "")) > 30_000:
            original_len = len(msg["content"])
            msg["content"] = (
                msg["content"][:_COMPRESS_TRUNCATE_AT]
                + f"\n\n... ({original_len - _COMPRESS_TRUNCATE_AT} chars truncated from history. "
                "Use read_file to re-read the full content only if "
                "the truncated portion above is insufficient for your current task.)"
            )
            compressed += 1

    if compressed:
        new_total = sum(len(m.get("content", "") or "") for m in messages)
        logger.info(
            "Compressed %d large tool result(s) (%d → %d chars)",
            compressed,
            total,
            new_total,
        )


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------


def _parse_tool_args(raw_args: Any, finish_reason: Optional[str]) -> tuple:
    """Parse a tool call's JSON arguments, failing loud on bad input.

    Returns ``(args, None)`` on success, or ``(None, error_json)`` when the
    arguments are malformed or truncated. Ported from the planning
    orchestrator's hand-rolled loop (#270): a silent ``args = {}`` fallback
    hides the real cause — the tool then raises about a MISSING argument, so
    the model "resubmits with the full task" (fixing the wrong thing) and
    loops. Shared here so every backbone orchestrator gets the same recovery
    hint instead of each hand-rolling (or losing) it independently.
    """
    if isinstance(raw_args, dict):
        return raw_args, None
    try:
        return json.loads(raw_args), None
    except (json.JSONDecodeError, TypeError):
        raw = raw_args if isinstance(raw_args, str) else ""
        if finish_reason == "length":
            cause = ("the arguments JSON was truncated — the response hit "
                      "the output-token limit")
        else:
            cause = ("the arguments string was not valid JSON — typically "
                      "broken escaping of quotes or newlines inside a large "
                      "string value")
        return None, json.dumps({
            "status": "error",
            "message": (
                f"Tool call discarded: {cause} ({len(raw)} characters "
                "received). The tool was NOT executed, and the arguments "
                "you sent were never seen — this is NOT a missing-argument "
                "error, so re-sending the same call will fail the same way. "
                "Send a SHORTER call, or split the work across several "
                "smaller tool calls."
            ),
        })


def _print_reasoning(orch: Any, content: Any) -> None:
    """Surface interim reasoning via the orchestrator's own printer, when it
    has one (analysis, planning). Simulation has no such hook yet — no-op."""
    printer = getattr(orch, "_print_assistant_reasoning", None)
    if printer is not None:
        printer(content)


def _make_react_nodes(orch: Any):
    """Return ``(call_model, execute_tools)`` node functions that close over *orch*."""

    def call_model(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = list(state["messages"])
        print("  ⏳ Waiting for orchestrator response ...")

        # If the previous step injected an empty-response nudge, force text-only
        # reply (no tool calls) — matches old code's tool_choice="none" followup.
        last_msg = messages[-1] if messages else None
        is_nudge_step = (
            isinstance(last_msg, HumanMessage)
            and last_msg.content == _EMPTY_RESPONSE_NUDGE
        )
        tool_choice = "none" if is_nudge_step else "auto"

        if orch.use_openai:
            from openai import OpenAI

            client = OpenAI(
                api_key=orch.model.api_key,
                base_url=orch.model.base_url,
                timeout=_TIMEOUT,
            )
            full_messages = _build_openai_messages(messages, orch._system_prompt)
            _compress_messages_inplace(full_messages)

            response = None
            for attempt in range(1, _TIMEOUT_RETRIES + 1):
                try:
                    response = client.chat.completions.create(
                        model=orch.model.model,
                        messages=full_messages,
                        tools=orch.tools_for_model,
                        tool_choice=tool_choice,
                    )
                    break
                except Exception as e:
                    if ("timeout" in str(e).lower() or "timed out" in str(e).lower()) and attempt < _TIMEOUT_RETRIES:
                        logger.warning("API timeout in call_model (attempt %d/%d) — retrying...", attempt, _TIMEOUT_RETRIES)
                        continue
                    raise

            message = response.choices[0].message
            if getattr(message, "tool_calls", None):
                _print_reasoning(orch, message.content)
            ai_msg = _openai_message_to_langchain(message)
            _fr = getattr(response.choices[0], "finish_reason", None)
            ai_msg.additional_kwargs["finish_reason"] = _fr if isinstance(_fr, str) else None

        else:
            from ..wrappers.litellm_wrapper import litellm_completion

            full_messages = _build_openai_messages(messages, orch._system_prompt)
            _compress_messages_inplace(full_messages)

            response = None
            for attempt in range(1, _TIMEOUT_RETRIES + 1):
                try:
                    response = litellm_completion(
                        model=orch.model.model,
                        messages=full_messages,
                        tools=orch.tools_for_model,
                        tool_choice=tool_choice,
                        api_key=orch.model.api_key,
                        api_base=orch.model.base_url,
                        timeout=int(_TIMEOUT),
                        request_timeout=int(_TIMEOUT),
                    )
                    break
                except Exception as e:
                    if ("timeout" in str(e).lower() or "timed out" in str(e).lower()) and attempt < _TIMEOUT_RETRIES:
                        logger.warning("API timeout in call_model (attempt %d/%d) — retrying...", attempt, _TIMEOUT_RETRIES)
                        continue
                    raise

            message = response.choices[0].message
            if getattr(message, "tool_calls", None):
                _print_reasoning(orch, getattr(message, "content", None))
            ai_msg = _litellm_message_to_langchain(message)
            _fr = getattr(response.choices[0], "finish_reason", None)
            ai_msg.additional_kwargs["finish_reason"] = _fr if isinstance(_fr, str) else None

        # Guard against empty responses — inject a nudge so the next step
        # is forced to produce a text summary via tool_choice="none".
        if not ai_msg.content and not ai_msg.tool_calls:
            return {
                "messages": [
                    ai_msg,
                    HumanMessage(content=_EMPTY_RESPONSE_NUDGE),
                ]
            }

        return {"messages": [ai_msg]}

    def execute_tools(state: Dict[str, Any]) -> Dict[str, Any]:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        finish_reason = last.additional_kwargs.get("finish_reason") \
            if hasattr(last, "additional_kwargs") else None

        results = []
        for tc in tool_calls:
            func_name = tc["name"]
            args, arg_error = _parse_tool_args(tc["args"], finish_reason)
            if arg_error is not None:
                print(f"  ⚠️  {func_name}: arguments discarded (malformed/truncated)")
                content = orch._tool_message(tc["id"], arg_error)["content"] \
                    if hasattr(orch, "_tool_message") else arg_error
                results.append(ToolMessage(content=content, tool_call_id=tc["id"], name=func_name))
                continue

            print(f"  🔧 Calling tool: {func_name}")
            result = orch.tools.execute_tool(func_name, **args)
            # orch._tool_message() upgrades an image-bearing result to a
            # multimodal content list for providers that render images in
            # tool results (see scilink/utils/tool_media.py); orchestrators
            # without that hook (planning, simulation) keep the plain string.
            content = orch._tool_message(tc["id"], result)["content"] \
                if hasattr(orch, "_tool_message") else result
            results.append(ToolMessage(content=content, tool_call_id=tc["id"], name=func_name))

        return {
            "messages": results,
            "step_count": state.get("step_count", 0) + 1,
        }

    return call_model, execute_tools


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def _react_should_continue(state: Dict[str, Any], max_steps: int) -> str:
    step_count = state.get("step_count", 0)
    if step_count >= max_steps:
        logger.warning("⚠️ Maximum tool iterations (%d) reached. Routing to END.", max_steps)
        return END

    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "execute_tools"
    return END


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def build_react_graph(orch: Any, state_type: type, checkpointer: Any = None) -> Any:
    """
    Build and compile a ReAct ``StateGraph`` for *orch*.

    Parameters
    ----------
    orch:
        Orchestrator instance (see module docstring for the required attribute
        contract).
    state_type:
        A subclass of ``OrchestratorState`` to use as the graph's state schema.
    checkpointer:
        LangGraph checkpointer.  Defaults to a new ``MemorySaver`` instance.

    Returns
    -------
    Compiled LangGraph ``CompiledGraph``.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    call_model, execute_tools = _make_react_nodes(orch)

    def should_continue(state: Dict[str, Any]) -> str:
        # Read live: orch.max_iterations is per-call (run_task can override
        # it for the duration of one delegation), so a value baked in at
        # build time would go stale after the first override/restore.
        per_call = getattr(orch, "max_iterations", None)
        max_steps = per_call if isinstance(per_call, int) else getattr(
            orch, "MAX_TOOL_ITERATIONS", _MAX_STEPS_DEFAULT)
        return _react_should_continue(state, max_steps)

    builder = StateGraph(state_type)
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
