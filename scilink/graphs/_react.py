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
    orch.MAX_HISTORY_MESSAGES int    (optional; defaults to 100 — also caps
                                     the LLM-facing wire message count, see
                                     ``_cap_message_count``)

Behavioral notes
----------------

* ``_compress_messages_inplace`` is applied on every ``call_model`` step
  for all orchestrators.  It is a no-op unless total context exceeds
  100 K chars, so it is safe to enable unconditionally.

* ``_cap_message_count`` is applied right after it, also on every
  ``call_model`` step.  It bounds the *count* of the wire-format message
  list (default 100, or ``orch.MAX_HISTORY_MESSAGES`` if set) independent
  of char size — a session with many small turns never trips the char
  threshold above but still sends an ever-growing history without this.
  Only trims the copy sent to the LLM this call; the graph's checkpointed
  state and ``self.messages``/``history.json`` keep the full thread.

* When the model returns an empty message (no content, no tool calls),
  a synthetic user nudge is injected so the next step can produce a
  human-readable summary.  This guards against silent dead-ends in any
  orchestrator mode.

* ``MAX_TOOL_ITERATIONS`` is always read from the orchestrator instance
  (``getattr`` with a default of 20), so per-session overrides take
  effect without rebuilding the graph.

* ``_get_openai_client`` caches the ``openai.OpenAI`` client on *orch*
  (``orch._openai_client`` / ``orch._openai_client_key``), reused across
  every ``call_model`` step for that orchestrator's lifetime instead of
  constructing one per LLM call. Keyed on ``(api_key, base_url)`` so a
  runtime model reconfiguration still gets a fresh client. The litellm
  branch has no equivalent — ``litellm_completion`` is a stateless
  function call, not a constructed client object.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ..utils.tool_media import close_interrupted_turn, repair_dangling_tool_calls
from ..wrappers.litellm_wrapper import litellm_completion

logger = logging.getLogger(__name__)

_TIMEOUT = 120.0
_MAX_STEPS_DEFAULT = 20
_COMPRESS_THRESHOLD = 100_000
_COMPRESS_TRUNCATE_AT = 5_000
_TIMEOUT_RETRIES = 3
_EMPTY_RESPONSE_NUDGE = "Please briefly summarize what you just did and suggest next steps."
# additional_kwargs marker identifying a synthetic nudge HumanMessage, so
# detection doesn't rely on exact content equality — a genuine user message
# that happens to type this exact sentence would otherwise be misidentified
# as the internal nudge (tools force-disabled, step-cap bypassed, spurious
# restart-repair). additional_kwargs lives only in the in-memory LangGraph
# checkpoint (MemorySaver) and isn't carried through _langchain_to_openai_dict,
# so a nudge that round-trips through history.json/seed_graph_history loses
# the marker — acceptable, since repair_graph_state's nudge-healing branch is
# already a rare belt-and-suspenders case for that path.
_NUDGE_MARKER_KEY = "_scilink_nudge"


def _is_nudge_message(msg: Any) -> bool:
    """True if *msg* is the synthetic empty-response nudge — see
    ``_NUDGE_MARKER_KEY``."""
    return (
        isinstance(msg, HumanMessage)
        and bool(getattr(msg, "additional_kwargs", None))
        and msg.additional_kwargs.get(_NUDGE_MARKER_KEY) is True
    )
_MAX_MESSAGE_COUNT_DEFAULT = 100
# Sentinel wrapper key: LangChain's AIMessage requires tool_calls[i]["args"]
# to be a dict, so a tool call whose arguments string doesn't parse as JSON
# is wrapped as {_MALFORMED_ARGS_KEY: raw_string} instead of raising at
# message-construction time. _parse_tool_args unwraps it before parsing.
_MALFORMED_ARGS_KEY = "__malformed_tool_args__"
_TRIM_HEAD_WINDOW = 10


# ---------------------------------------------------------------------------
# Message format helpers
# ---------------------------------------------------------------------------


def _build_openai_messages(langchain_messages: list, system_prompt: str) -> list:
    """Convert a LangChain message list to OpenAI wire format, prepending the system prompt."""
    result = [{"role": "system", "content": system_prompt}]
    for msg in langchain_messages:
        result.append(_langchain_to_openai_dict(msg))
    return result


def _serialize_tool_call_args(args: Any) -> str:
    """Serialize a LangChain ``ToolCall``'s ``args`` back to an OpenAI-wire
    ``arguments`` string — unwrapping the ``{_MALFORMED_ARGS_KEY: raw}``
    marker back to the original raw string instead of ``json.dumps``-ing
    the wrapper dict itself.

    Without this, once one malformed tool call is committed to the graph's
    checkpointed thread, every later ``call_model`` step (which rebuilds
    the full wire-format history via ``_build_openai_messages``) and every
    ``sync_new_messages`` write to ``history.json`` would re-serialize that
    historical call's arguments as ``'{"__malformed_tool_args__": "..."}'``
    — a distorted, permanently-baked-in version of what the model actually
    sent — instead of either the clean original string or an unwrapped
    equivalent.
    """
    if isinstance(args, dict) and set(args) == {_MALFORMED_ARGS_KEY} \
            and isinstance(args[_MALFORMED_ARGS_KEY], str):
        return args[_MALFORMED_ARGS_KEY]
    return args if isinstance(args, str) else json.dumps(args)


def _langchain_to_openai_dict(msg: Any) -> dict:
    """Convert a LangChain message object to an OpenAI wire dict."""
    if isinstance(msg, HumanMessage):
        return {"role": "user", "content": msg.content}
    if isinstance(msg, SystemMessage):
        return {"role": "system", "content": msg.content}
    if isinstance(msg, ToolMessage):
        return {"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content}
    if isinstance(msg, AIMessage):
        d: dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": _serialize_tool_call_args(tc["args"]),
                    },
                }
                for tc in msg.tool_calls
            ]
        return d
    if isinstance(msg, dict):
        return msg
    return {"role": "user", "content": str(msg)}


def _wrap_malformed_args(raw: str) -> dict:
    """Wrap a raw arguments string that can't be used as-is into the
    single-key marker shape ``_parse_tool_args`` (and ``_langchain_to_openai_dict``)
    know how to unwrap. Centralized so both call sites that produce the
    marker (a JSON-parse failure, and valid-JSON-but-not-an-object) do it
    identically."""
    return {_MALFORMED_ARGS_KEY: raw}


def _convert_tool_call(tc: Any) -> dict:
    """Convert one OpenAI/LiteLLM tool-call object to a LangChain ``ToolCall``
    dict, shared by ``_openai_message_to_langchain`` and
    ``_litellm_message_to_langchain`` (identical shape from both providers).

    Does not let a malformed/truncated arguments string raise here:
    LangChain's ``AIMessage`` requires ``args`` to be a dict (a raw string
    fails pydantic validation at message-construction time), so this parses
    just far enough to satisfy that — on failure, wraps the raw string as
    ``{_MALFORMED_ARGS_KEY: raw}`` rather than raising, deferring the actual
    *handling* of the bad JSON to ``_parse_tool_args`` (in ``execute_tools``),
    which unwraps the marker and turns it into a recoverable tool-result
    error the model can retry from — never a crashed turn.

    Also wraps arguments that parse as valid JSON but aren't a JSON object
    (e.g. ``"[1,2,3]"`` or ``"42"``) — those used to pass through as
    ``parsed`` unchanged, satisfying ``AIMessage``'s dict requirement only
    because the earlier check was "did json.loads raise", not "is this a
    dict"; ``execute_tools`` would then call ``orch.tools.execute_tool(name,
    **parsed)`` on a non-mapping and crash the turn with an uncaught
    ``TypeError`` — exactly what this wrapper exists to prevent.
    """
    raw = tc.function.arguments
    if not raw:
        args: Any = {}
    else:
        try:
            parsed = json.loads(raw)
            args = parsed if isinstance(parsed, dict) else _wrap_malformed_args(raw)
        except (json.JSONDecodeError, TypeError):
            args = _wrap_malformed_args(raw)
    return {"id": tc.id, "name": tc.function.name, "args": args, "type": "tool_call"}


def _parse_history_tool_args(raw: Any) -> Any:
    """Parse a persisted ``history.json`` tool-call's ``arguments`` field for
    ``seed_graph_history``, mirroring ``_convert_tool_call``'s safety
    contract: never raise, and never let a non-dict result through
    unwrapped — a dict already (older schema, or already-parsed), a string
    that parses as a JSON object, and everything else (parse failure OR
    valid-JSON-non-object) all become AIMessage-safe.

    Unlike the live-response path, ``history.json`` can be hand-edited or
    written by a version of this code that predates a bugfix, so this must
    not assume the persisted arguments string is well-formed.
    """
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return _wrap_malformed_args(str(raw))
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return _wrap_malformed_args(raw)
    return parsed if isinstance(parsed, dict) else _wrap_malformed_args(raw)


def _openai_message_to_langchain(msg: Any) -> AIMessage:
    """Convert an OpenAI ``ChatCompletionMessage`` to a LangChain ``AIMessage``."""
    tool_calls_raw = getattr(msg, "tool_calls", None) or []
    return AIMessage(
        content=msg.content or "",
        tool_calls=[_convert_tool_call(tc) for tc in tool_calls_raw],
    )


def _litellm_message_to_langchain(msg: Any) -> AIMessage:
    """Convert a LiteLLM response message to a LangChain ``AIMessage``."""
    tool_calls_raw = getattr(msg, "tool_calls", None) or []
    return AIMessage(
        content=getattr(msg, "content", None) or "",
        tool_calls=[_convert_tool_call(tc) for tc in tool_calls_raw],
    )


# ---------------------------------------------------------------------------
# Restart / interruption recovery
# ---------------------------------------------------------------------------


def repair_graph_state(orch: Any) -> None:
    """Heal a dangling tool call left by a mid-run interruption (process
    stopped between ``execute_tools`` committing its result and the loop
    continuing) — the LangGraph-state analogue of ``repair_dangling_tool_calls``
    / ``close_interrupted_turn``.

    The checkpointer only ever appends, so the realistic failure mode is
    purely additive: the persisted thread ends on an AIMessage whose
    tool_calls have no matching ToolMessage, or ends on a ToolMessage with
    no assistant reply. Both are healed by appending the missing
    message(s) via ``update_state`` — never by rewriting history.

    Shared across all four backbone orchestrators; each calls this as
    ``self._repair_graph_state()``, a thin wrapper that passes ``self``.

    Skips the (checkpoint-deserializing) ``get_state`` call once a turn has
    already completed cleanly this "epoch" (``orch._graph_state_repaired``
    is True) — a clean completion always leaves the checkpoint well-formed,
    since the graph only reaches END on a final text AIMessage with no
    unanswered tool_calls, so re-checking every turn re-deserializes state
    that can't have changed shape. The realistic source of a dangling call
    is either a fresh process's first turn touching a possibly-stale
    restored checkpoint (flag starts False at construction), or a turn
    whose ``invoke_graph`` call raised mid-run — ``chat()`` swallows that
    exception and keeps the process alive, so the very next turn on the
    same orchestrator instance must re-check (``invoke_graph`` resets the
    flag to False in its except branch on exactly that path).
    """
    if getattr(orch, "_graph_state_repaired", False):
        return
    orch._graph_state_repaired = True
    try:
        snapshot = orch._graph.get_state(orch._graph_config)
    except Exception:
        return
    if not snapshot or not snapshot.values:
        return
    messages = snapshot.values.get("messages") or []
    if not messages:
        return

    last = messages[-1]
    patch = []
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        for tc in last.tool_calls:
            patch.append(ToolMessage(
                content=(
                    "⚠️ Tool execution was interrupted before a result "
                    "was produced (the run was stopped). Re-run this "
                    "tool if its output is still needed."
                ),
                tool_call_id=tc["id"],
            ))
    elif isinstance(last, ToolMessage):
        patch.append(AIMessage(
            content="[Turn interrupted before a reply was produced.]"
        ))
    elif _is_nudge_message(last):
        # Belt-and-suspenders: _react_should_continue now always retries the
        # nudge regardless of the step-count cap (see the reorder there), so
        # this shouldn't happen live any more — but a checkpoint written
        # before that fix, or a genuine process kill in the narrow window
        # right after this message was committed, can still leave it as the
        # terminal message. Healed the same way as a dangling ToolMessage:
        # append a placeholder reply so the next turn's user message doesn't
        # follow it directly (two consecutive human-role messages) and the
        # final-text scan doesn't fall through to a stale earlier turn.
        patch.append(AIMessage(
            content="[Turn interrupted before a reply was produced.]"
        ))

    if patch:
        log = getattr(orch, "logger", None) or logger
        log.info(
            "  🔧 Repaired %d dangling message(s) from an interrupted run",
            len(patch),
        )
        orch._graph.update_state(orch._graph_config, {"messages": patch})


def invoke_graph(orch: Any, initial_state: Dict[str, Any]) -> Dict[str, Any]:
    """Run the graph for one turn (``orch._graph.invoke``), tracking whether
    it completed cleanly.

    On an exception mid-run, resets ``orch._graph_state_repaired`` to False
    so the next turn's ``repair_graph_state`` actually re-deserializes the
    checkpoint and heals any dangling tool call this turn may have left —
    ``chat()`` swallows the exception (logs it, saves an emergency
    checkpoint, returns an error string) rather than crashing the process,
    so the same orchestrator instance is reused for the next turn. Re-raises
    unchanged so each orchestrator's existing ``chat()`` exception handling
    is unaffected.
    """
    try:
        return orch._graph.invoke(initial_state, config=orch._graph_config)
    except Exception:
        orch._graph_state_repaired = False
        raise


def invoke_graph_turn(orch: Any, initial_state: Dict[str, Any], user_input: str) -> Dict[str, Any]:
    """``invoke_graph`` wrapper for a chat turn that also guarantees the
    turn's own user input survives to ``history.json`` even when the graph
    raises mid-run.

    ``sync_new_messages`` — the only thing that appends a turn's messages
    (including its user input) to ``orch.messages`` — runs only after
    ``invoke_graph`` returns successfully, inside each orchestrator's
    ``_invoke_graph``. An exception here previously skipped it entirely, and
    ``chat()``'s ``except`` block only calls ``_auto_checkpoint()`` (scalar
    checkpoint.json fields), never ``_save_history()`` — so the crashing
    turn's own question was silently absent from every persisted record.
    This appends it directly and saves history before re-raising, so the
    exact input that triggered the crash is recoverable after the fact.
    """
    try:
        return invoke_graph(orch, initial_state)
    except Exception:
        orch.messages.append({"role": "user", "content": user_input})
        save_history = getattr(orch, "_save_history", None)
        if callable(save_history):
            try:
                save_history()
            except Exception:
                log = getattr(orch, "logger", None) or logger
                log.warning("Failed to save history after a crashed turn", exc_info=True)
        raise


def extract_final_text(orch: Any, result: Dict[str, Any]) -> str:
    """Extract the final assistant reply from a completed graph turn and
    mirror the turn into ``orch.messages``.

    Scans ``result["messages"]`` in reverse for the last content-bearing
    ``AIMessage``. When none is found (the graph hit the tool-iteration
    step cap, or otherwise ended without a text reply), falls back to a
    fixed warning string and sets ``orch._last_chat_hit_iter_cap`` when the
    cap was the actual cause — mirrored by ``run_task`` to report iteration
    exhaustion as an error to programmatic callers.

    Always calls ``sync_new_messages`` to mirror the graph's own thread.
    When the fallback fires, that thread has no AIMessage carrying the
    fallback text at all, so ``sync_new_messages`` can't mirror it —
    appended directly here so ``history.json`` matches what the user
    actually saw, instead of losing it to a generic "[Turn interrupted...]"
    placeholder on the next restart's ``seed_graph_history`` repair.
    """
    final_text = ""
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            final_text = msg.content
            break

    used_fallback = not final_text
    if used_fallback:
        max_iterations = getattr(orch, "max_iterations", None)
        if isinstance(max_iterations, int) and result.get("step_count", 0) >= max_iterations:
            orch._last_chat_hit_iter_cap = True
        final_text = "⚠️ Maximum tool iterations reached. Please simplify your request."

    sync_new_messages(orch, result)
    if used_fallback:
        orch.messages.append({"role": "assistant", "content": final_text})

    return final_text


def seed_graph_history(orch: Any, history: List[Dict]) -> None:
    """Replay persisted history into the graph's MemorySaver so a restored
    session has full context.

    Only called once at init when history is non-empty. Uses a bulk
    ``update_state`` so the graph thread accumulates all prior messages
    without making any LLM calls. System messages are skipped — the graph
    handles the system prompt internally via the ``call_model`` closure.

    Shared across all four backbone orchestrators; each calls this as
    ``self._seed_graph_history(history)``, a thin wrapper that passes ``self``.
    """
    history = close_interrupted_turn(repair_dangling_tool_calls(list(history)))

    lc_messages = []
    for m in history:
        role = m.get("role", "")
        content = m.get("content") or ""
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            tool_calls_raw = m.get("tool_calls", [])
            lc_tc = [
                {
                    "id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", ""),
                    "args": _parse_history_tool_args(tc.get("function", {}).get("arguments", "{}")),
                    "type": "tool_call",
                }
                for tc in tool_calls_raw
            ]
            lc_messages.append(AIMessage(content=content, tool_calls=lc_tc))
        elif role == "tool":
            lc_messages.append(
                ToolMessage(content=content, tool_call_id=m.get("tool_call_id", ""))
            )

    if not lc_messages:
        return

    log = getattr(orch, "logger", None) or logger
    try:
        orch._graph.update_state(orch._graph_config, {"messages": lc_messages})
        # sync_new_messages() below only mirrors the *delta* since the last
        # sync into orch.messages — seed the counter to what the graph now
        # holds so the next turn's sync doesn't re-append this seeded history.
        orch._graph_synced_message_count = len(lc_messages)
        log.info(
            "  🧠 Graph history seeded: %d messages loaded into MemorySaver",
            len(lc_messages),
        )
    except Exception as e:
        log.warning("Failed to seed graph history: %s", e)


def sync_new_messages(orch: Any, result: Dict[str, Any]) -> None:
    """Mirror this turn's full tool-call trace into ``orch.messages`` — not
    just the user text and final assistant text — so it round-trips through
    ``history.json`` / ``seed_graph_history`` across a process restart.

    Without this, ``orch.messages`` (and thus ``history.json``) only ever
    held a flattened ``user`` + final ``assistant`` pair per turn; every
    intermediate tool call and tool result the graph made along the way
    lived only in the in-memory ``MemorySaver`` checkpoint, gone for good on
    restart even though ``seed_graph_history`` faithfully replays whatever
    ``history.json`` does contain.

    ``result["messages"]`` is the graph's full accumulated thread for this
    checkpointer thread (not just this turn), so only the slice since the
    last sync is appended — tracked via ``orch._graph_synced_message_count``
    (seeded by ``seed_graph_history`` on restore, else 0 from a fresh
    session's graph construction).

    Call this in place of manually appending a ``{"role": "user", ...}`` /
    ``{"role": "assistant", ...}`` pair around ``orch._graph.invoke(...)``.
    """
    all_lc = result.get("messages", [])
    start = getattr(orch, "_graph_synced_message_count", 0)
    new = all_lc[start:]
    orch.messages.extend(_langchain_to_openai_dict(m) for m in new)
    orch._graph_synced_message_count = len(all_lc)


def sync_system_prompt(orch: Any) -> None:
    """Push ``orch._system_prompt`` into ``orch.messages[0]`` so the two
    copies never diverge.

    ``call_model`` reads ``orch._system_prompt`` exclusively (see this
    module's orchestrator contract); ``orch.messages[0]`` is a separate copy
    kept only for JSON persistence (``_save_history``) and ``_trim_history``.
    Call this immediately after assigning ``orch._system_prompt`` instead of
    hand-rolling the ``if orch.messages and orch.messages[0]["role"] ==
    "system": orch.messages[0]["content"] = ...`` guard at each site — a
    future rebuild site that updates only one of the two copies is exactly
    the bug this closes (previously duplicated ~13 times across the 4
    orchestrators, one update-only-one-side away from a silently stale
    system prompt reaching the LLM).

    Inserts a system message at index 0 if ``orch.messages`` doesn't have
    one yet (mirrors the one call site — ``set_simulation_mode`` — that
    needed this defensively).
    """
    if orch.messages and orch.messages[0].get("role") == "system":
        orch.messages[0]["content"] = orch._system_prompt
    elif orch.messages is not None:
        orch.messages.insert(0, {"role": "system", "content": orch._system_prompt})


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


def _cap_message_count(messages: list, orch: Any) -> None:
    """Bound the LLM-facing wire message count so a long session with many
    small turns doesn't resend an ever-growing history every call.

    ``_compress_messages_inplace`` only shrinks individual oversized tool
    results once total chars exceed its threshold — a session that
    accumulates hundreds of small messages (no single one ever large) never
    trips it. Before the LangGraph migration, the hand-rolled loop sent
    ``self.messages`` directly to the LLM, so trimming it (still done today
    for JSON-persistence bookkeeping — see ``AnalysisOrchestratorAgent
    ._trim_history``) also capped the wire payload as a side effect. The
    graph now keeps its own accumulating thread independent of
    ``self.messages``, so that trim no longer bounds what's actually sent —
    this restores an equivalent cap directly on the wire copy.

    Operates in-place on *messages* (index 0 assumed to be the system
    message, per ``_build_openai_messages``'s output shape). Only trims the
    copy passed to the LLM this call — never mutates the graph's
    checkpointed state (``state["messages"]``), so no history is lost, only
    re-sent less often; the full thread stays available via
    ``self.messages`` / ``history.json`` (see ``sync_new_messages``) and the
    checkpoint itself.
    """
    cap = getattr(orch, "MAX_HISTORY_MESSAGES", None)
    if not isinstance(cap, int) or cap <= 0:
        cap = _MAX_MESSAGE_COUNT_DEFAULT
    if len(messages) <= cap + 1:  # +1 for the system message at index 0
        return

    system_msg, rest = messages[0], messages[1:]
    head_window = min(_TRIM_HEAD_WINDOW, cap // 4)
    tail_window = cap - head_window
    omitted = len(rest) - (head_window + tail_window)
    trimmed = rest[:head_window] + rest[-tail_window:]
    trimmed.insert(head_window, {
        "role": "system",
        "content": f"[{omitted} messages omitted for context management]",
    })
    # The head/tail splice can orphan a tool_use/tool_result pair right at
    # its seam — repair AFTER the splice (see TODO item 4's fix for the same
    # hazard on self.messages: repairing beforehand only re-validates an
    # already-fine list and misses the fresh splice damage).
    trimmed = repair_dangling_tool_calls(trimmed)
    messages[:] = [system_msg] + trimmed
    logger.info(
        "Capped LLM-facing message count: %d → %d (%d omitted)",
        len(rest) + 1, len(messages), omitted,
    )


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------


def _parse_tool_args(
    raw_args: Any, finish_reason: Optional[str], extra_hint: Optional[str] = None
) -> tuple:
    """Parse a tool call's JSON arguments, failing loud on bad input.

    Returns ``(args, None)`` on success, or ``(None, error_json)`` when the
    arguments are malformed or truncated. Ported from the planning
    orchestrator's hand-rolled loop (#270): a silent ``args = {}`` fallback
    hides the real cause — the tool then raises about a MISSING argument, so
    the model "resubmits with the full task" (fixing the wrong thing) and
    loops. Shared here so every backbone orchestrator gets the same recovery
    hint instead of each hand-rolling (or losing) it independently.

    *extra_hint*, if given, is an orchestrator-specific remediation sentence
    (e.g. "for a delegation, keep the essential instruction in `task`…")
    appended after the generic advice.

    Unwraps ``{_MALFORMED_ARGS_KEY: raw_string}`` back to the raw string
    before parsing — the shape ``_openai_message_to_langchain`` /
    ``_litellm_message_to_langchain`` produce when the model's arguments
    string doesn't parse as JSON, since LangChain's ``AIMessage`` requires
    ``tool_calls[i]["args"]`` to be a dict (a raw string there raises a
    pydantic ``ValidationError``, so it can't be passed through directly).
    Any other dict is assumed already-parsed and returned as-is.

    The marker match also requires the single value to be a ``str`` — a
    real (if unlikely) tool argument dict that happens to be exactly
    ``{"__malformed_tool_args__": <value>}`` only collides with the marker
    shape if that value is itself a string; requiring the type narrows the
    false-positive window without adding another sentinel field to every
    genuinely malformed case.

    A parsed-but-non-dict result (valid JSON that isn't a JSON object, e.g.
    a bare list or number) is treated the same as a parse failure — a tool
    call's arguments must be a mapping, and passing anything else through
    as ``(parsed, None)`` would let ``execute_tools`` call
    ``**parsed`` on a non-mapping and crash the turn.
    """
    if isinstance(raw_args, dict):
        if set(raw_args) == {_MALFORMED_ARGS_KEY} and isinstance(raw_args[_MALFORMED_ARGS_KEY], str):
            raw_args = raw_args[_MALFORMED_ARGS_KEY]
        else:
            return raw_args, None
    parsed_non_dict = False
    try:
        parsed = json.loads(raw_args)
        if isinstance(parsed, dict):
            return parsed, None
        parsed_non_dict = True
    except (json.JSONDecodeError, TypeError):
        pass

    raw = raw_args if isinstance(raw_args, str) else ""
    if finish_reason == "length":
        cause = ("the arguments JSON was truncated — the response hit "
                  "the output-token limit")
    elif parsed_non_dict:
        cause = ("the arguments parsed as valid JSON but were not a "
                  "JSON object (tool arguments must be a mapping of "
                  "parameter name to value)")
    else:
        cause = ("the arguments string was not valid JSON — typically "
                  "broken escaping of quotes or newlines inside a large "
                  "string value")
    message = (
        f"Tool call discarded: {cause} ({len(raw)} characters "
        "received). The tool was NOT executed, and the arguments "
        "you sent were never seen — this is NOT a missing-argument "
        "error, so re-sending the same call will fail the same way. "
        "Send a SHORTER call, or split the work across several "
        "smaller tool calls."
    )
    if extra_hint:
        message += " " + extra_hint
    return None, json.dumps({"status": "error", "message": message})


def _get_openai_client(orch: Any) -> Any:
    """Return a cached ``openai.OpenAI`` client, reused across ``call_model``
    steps instead of constructing a fresh one on every re-entry (a turn with
    N tool calls previously made N client constructions, losing
    connection-pool reuse).

    Cached on *orch* itself — keyed on ``(api_key, base_url)`` so a runtime
    model reconfiguration (a different key/proxy) still gets a fresh client
    rather than silently reusing a stale one pointed at the old endpoint.

    The ``from openai import OpenAI`` + construction still happen inside
    this function (not hoisted to module scope) for the same reason the
    call-local import existed before: tests ``patch("openai.OpenAI")``,
    which only intercepts a lookup made at call time — the first
    ``call_model`` invocation made while a patch is active still triggers a
    fresh construction here (nothing was cached yet, or the endpoint
    "changed" from whatever a prior unrelated orchestrator/test cached).
    """
    key = (orch.model.api_key, orch.model.base_url)
    if getattr(orch, "_openai_client", None) is not None \
            and getattr(orch, "_openai_client_key", None) == key:
        return orch._openai_client

    from openai import OpenAI

    client = OpenAI(api_key=orch.model.api_key, base_url=orch.model.base_url, timeout=_TIMEOUT)
    orch._openai_client = client
    orch._openai_client_key = key
    return client


def _print_reasoning(orch: Any, content: Any) -> None:
    """Surface interim reasoning via the orchestrator's own printer, when it
    has one (analysis, planning). Simulation has no such hook yet — no-op."""
    printer = getattr(orch, "_print_assistant_reasoning", None)
    if printer is not None:
        printer(content)


def _run_call_model_branch(
    orch: Any,
    messages: list,
    make_request: Callable[[list], Any],
    convert_fn: Callable[[Any], AIMessage],
) -> AIMessage:
    """Shared body of ``call_model``'s openai/litellm branches: prepare the
    wire-format messages, run the timeout-retry loop, convert the response
    to a LangChain ``AIMessage``, and stamp its ``finish_reason``.

    The two branches previously duplicated this whole sequence (>15 lines)
    near-verbatim, differing only in which client/conversion function is
    invoked — a future fix to the retry count, the timeout-detection string
    match, or a new retryable exception type had to be applied in both
    places, and missing one would silently reintroduce the bug in whichever
    provider path wasn't updated.

    *make_request* takes the prepared wire-format message list and returns
    the raw provider response (an OpenAI ``ChatCompletion`` or LiteLLM
    equivalent — both expose ``.choices[0].message`` /
    ``.choices[0].finish_reason``); *convert_fn* is
    ``_openai_message_to_langchain`` or ``_litellm_message_to_langchain``.
    """
    full_messages = _build_openai_messages(messages, orch._system_prompt)
    _compress_messages_inplace(full_messages)
    _cap_message_count(full_messages, orch)

    response = None
    for attempt in range(1, _TIMEOUT_RETRIES + 1):
        try:
            response = make_request(full_messages)
            break
        except Exception as e:
            if ("timeout" in str(e).lower() or "timed out" in str(e).lower()) and attempt < _TIMEOUT_RETRIES:
                logger.warning("API timeout in call_model (attempt %d/%d) — retrying...", attempt, _TIMEOUT_RETRIES)
                continue
            raise

    message = response.choices[0].message
    if getattr(message, "tool_calls", None):
        _print_reasoning(orch, getattr(message, "content", None))
    ai_msg = convert_fn(message)
    _fr = getattr(response.choices[0], "finish_reason", None)
    ai_msg.additional_kwargs["finish_reason"] = _fr if isinstance(_fr, str) else None
    return ai_msg


def _make_react_nodes(orch: Any):
    """Return ``(call_model, execute_tools)`` node functions that close over *orch*."""

    def call_model(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = list(state["messages"])
        print("  ⏳ Waiting for orchestrator response ...")

        # If the previous step injected an empty-response nudge, force text-only
        # reply (no tool calls) — matches old code's tool_choice="none" followup.
        last_msg = messages[-1] if messages else None
        is_nudge_step = _is_nudge_message(last_msg)
        tool_choice = "none" if is_nudge_step else "auto"

        if orch.use_openai:
            client = _get_openai_client(orch)
            ai_msg = _run_call_model_branch(
                orch, messages,
                make_request=lambda wire: client.chat.completions.create(
                    model=orch.model.model,
                    messages=wire,
                    tools=orch.tools_for_model,
                    tool_choice=tool_choice,
                ),
                convert_fn=_openai_message_to_langchain,
            )

        else:
            ai_msg = _run_call_model_branch(
                orch, messages,
                make_request=lambda wire: litellm_completion(
                    model=orch.model.model,
                    messages=wire,
                    tools=orch.tools_for_model,
                    tool_choice=tool_choice,
                    api_key=orch.model.api_key,
                    api_base=orch.model.base_url,
                    timeout=int(_TIMEOUT),
                    request_timeout=int(_TIMEOUT),
                ),
                convert_fn=_litellm_message_to_langchain,
            )

        # Empty response: nudge once for a forced tool_choice="none" retry.
        # Don't nudge again if the retry is also empty (avoids looping).
        if not ai_msg.content and not ai_msg.tool_calls and not is_nudge_step:
            return {
                "messages": [
                    ai_msg,
                    HumanMessage(
                        content=_EMPTY_RESPONSE_NUDGE,
                        additional_kwargs={_NUDGE_MARKER_KEY: True},
                    ),
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
            args, arg_error = _parse_tool_args(
                tc["args"], finish_reason,
                extra_hint=getattr(orch, "_tool_arg_error_hint", None),
            )
            if arg_error is not None:
                print(f"  ⚠️  {func_name}: arguments discarded (malformed/truncated)")
                content = orch._tool_message(tc["id"], arg_error)["content"] \
                    if hasattr(orch, "_tool_message") else arg_error
                results.append(ToolMessage(content=content, tool_call_id=tc["id"], name=func_name))
                continue

            print(f"  🔧 Calling tool: {func_name}")
            result = orch.tools.execute_tool(func_name, **args)
            # orch._tool_message() upgrades an image-bearing result to a
            # multimodal message on providers that support it (see
            # scilink/utils/tool_media.py); not every orchestrator has this hook.
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
    last = state["messages"][-1]
    # Checked BEFORE the step-count cap, not after: the nudge is a one-shot
    # forced-text-only retry (call_model's own is_nudge_step guard prevents
    # a second nudge, so this can add at most one extra call_model beyond
    # the cap — bounded). If the cap check ran first, hitting max_steps in
    # the same round the nudge fires would route to END on the dangling
    # nudge HumanMessage instead of letting the retry run: the checkpoint
    # would end on that HumanMessage (repair_graph_state now heals it, but
    # only on the NEXT turn — this turn's final-text scan would already
    # have fallen through to a stale earlier-turn reply), and the following
    # turn's user message would land right after it, violating strict
    # human/human alternation.
    if _is_nudge_message(last):
        return "call_model"

    step_count = state.get("step_count", 0)
    if step_count >= max_steps:
        logger.warning("⚠️ Maximum tool iterations (%d) reached. Routing to END.", max_steps)
        return END

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
        # Read live, not baked in at build time: run_task overrides
        # orch.max_iterations per-call.
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
        {"call_model": "call_model", "execute_tools": "execute_tools", END: END},
    )
    builder.add_edge("execute_tools", "call_model")

    return builder.compile(checkpointer=checkpointer)
