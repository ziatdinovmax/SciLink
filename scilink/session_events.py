"""Append-only per-session event log + bounded programmatic access (#462).

Long chat sessions lose their own middle: ``_trim_history`` evicts older
turns from context and the LLM cannot get them back — ``chat_history.json``
is a single JSON array with full tool results, so a naive read pulls the
entire evicted history straight back into the window. This module gives each
session a compact, line-addressable record instead: one bounded JSON line
per tool call, appended at every orchestrator's ``execute_tool`` chokepoint,
searched grep-style and never read whole. The pattern (not the harness)
follows PRO-LONG (arXiv:2607.20064).

Thread-local binding mirrors ``scilink/hitl.py``'s feedback log exactly:
``chat()`` binds ``<base_dir>/events.jsonl``; ``run_task`` saves and
restores the caller's binding so a meta delegation logs child events to the
child's session and the meta's own log resumes afterward. An unbound thread
makes ``append_event`` a silent no-op (library callers, bare agents, tests
unaffected), and the writer never raises into the tool path — a broken
event log must not break a run.

Every field is bounded at write time, so grep hits stay small no matter how
long the session runs. ``search_events`` / ``read_events`` are the pure
functions the orchestrator tools wrap (cheap wide search over summaries,
bounded narrow read of specifics).
"""
from __future__ import annotations

import json
import logging
import re
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Bounds — summaries by construction. Loosening these silently grows every
# future search hit, so treat them as part of the log format.
ARGS_MAX_CHARS = 200
SUMMARY_MAX_CHARS = 300
FILES_MAX = 8
SEARCH_MAX_HITS = 50
SEARCH_MAX_CHARS = 6_000
READ_MAX_EVENTS = 40
READ_MAX_CHARS = 8_000

_REDACT_KEY_RE = re.compile(
    r"api[_-]?key|token|secret|password|credential", re.IGNORECASE
)
# Result keys whose values are worth surfacing in the one-line summary,
# in preference order.
_SUMMARY_KEYS = ("message", "summary", "error", "warning", "verdict", "action")
# Result keys that carry produced-file paths.
_FILES_KEYS = ("files_produced", "output_files", "files")
_PATHLIKE_KEY_RE = re.compile(r"(_path|_file|_dir|_directory)$")

_thread_local = threading.local()
_counter_lock = threading.Lock()
_next_n: Dict[str, int] = {}


# --------------------------------------------------------------- binding

def set_thread_event_log(path) -> None:
    """Bind this thread's tool calls to a session event log.

    ``path`` is the JSONL file (conventionally ``<session>/events.jsonl``).
    ``None`` unbinds. Orchestrators bind their session file at each chat
    turn; ``run_task`` restores the caller's binding on exit.
    """
    _thread_local.event_log = str(path) if path else None


def get_thread_event_log() -> Optional[str]:
    return getattr(_thread_local, "event_log", None)


@contextmanager
def use_event_log(path):
    """Scoped thread-local event-log binding (re-entrant safe)."""
    previous = get_thread_event_log()
    set_thread_event_log(path)
    try:
        yield
    finally:
        set_thread_event_log(previous)


# --------------------------------------------------------------- gisting

def _clip(text: str, limit: int) -> str:
    """One physical line, at most ``limit`` chars."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _gist_args(kwargs: Optional[Dict[str, Any]]) -> str:
    if not kwargs:
        return ""
    parts = []
    for key, value in kwargs.items():
        if _REDACT_KEY_RE.search(str(key)):
            parts.append(f"{key}=<redacted>")
        else:
            parts.append(f"{key}={_clip(value, 80)}")
    return _clip(" ".join(parts), ARGS_MAX_CHARS)


def _clip_path(text: str, limit: int = 160) -> str:
    """Clip a path from the LEFT — the tail (dirs near the file + filename)
    is the searchable, drill-down-able part; a head-clipped path is useless."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else "…" + flat[-(limit - 1):]


def _collect_files(parsed: Dict[str, Any]) -> List[str]:
    files: List[str] = []
    for key in _FILES_KEYS:
        val = parsed.get(key)
        if isinstance(val, list):
            files.extend(str(v) for v in val if isinstance(v, (str, Path)))
    for key, val in parsed.items():
        if isinstance(val, str) and _PATHLIKE_KEY_RE.search(key):
            files.append(val)
    # De-dup preserving order, bounded.
    seen: Dict[str, None] = {}
    for f in files:
        seen.setdefault(_clip_path(f))
    return list(seen)[:FILES_MAX]


def _gist_result(result: Any) -> Tuple[str, str, List[str]]:
    """Derive (status, summary, files) from a tool result.

    A JSON-object result contributes its ``status`` plus the first
    informative summary-like keys; anything else contributes the head of
    its string form.
    """
    raw = result if isinstance(result, str) else json.dumps(
        result, default=str, ensure_ascii=False
    )
    parsed: Optional[Dict[str, Any]] = None
    try:
        candidate = json.loads(raw)
        if isinstance(candidate, dict):
            parsed = candidate
    except (ValueError, TypeError):
        parsed = None

    if parsed is None:
        return "success", _clip(raw, SUMMARY_MAX_CHARS), []

    status = str(parsed.get("status") or (
        "error" if parsed.get("error") else "success"
    ))
    bits = []
    for key in _SUMMARY_KEYS:
        val = parsed.get(key)
        if isinstance(val, (str, int, float)) and str(val).strip():
            bits.append(f"{key}={_clip(val, 120)}")
    summary = _clip("; ".join(bits), SUMMARY_MAX_CHARS) if bits else _clip(
        raw, SUMMARY_MAX_CHARS
    )
    return status, summary, _collect_files(parsed)


# --------------------------------------------------------------- writer

def _claim_n(path: str) -> int:
    """Next monotonic event index for ``path`` (counts existing lines once)."""
    with _counter_lock:
        if path not in _next_n:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    _next_n[path] = sum(1 for _ in f) + 1
            except OSError:
                _next_n[path] = 1
        n = _next_n[path]
        _next_n[path] = n + 1
        return n


def append_event(tool: str, kwargs: Optional[Dict[str, Any]] = None,
                 result: Any = None, *, branch: Optional[str] = None) -> None:
    """Append one bounded event line for a tool call. Never raises.

    No-op when the thread has no bound event log.
    """
    path = get_thread_event_log()
    if not path:
        return
    try:
        status, summary, files = _gist_result(result)
        record = {
            "n": _claim_n(path),
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tool": str(tool),
            "args": _gist_args(kwargs),
            "status": status,
            "summary": summary,
        }
        if files:
            record["files"] = files
        if branch:
            record["branch"] = str(branch)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as e:  # noqa: BLE001 - the log must never break a run
        logger.debug(f"session event append failed: {e}")


# --------------------------------------------------------------- readers

def _iter_events(path) -> List[Dict[str, Any]]:
    """Parsed event lines; malformed lines are skipped, never fatal."""
    events: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict):
                    events.append(rec)
    except OSError:
        pass
    return events


def search_events(path, pattern: str, max_hits: int = 10,
                  exclude_tools: frozenset = frozenset()) -> Dict[str, Any]:
    """Case-insensitive grep over an event log.

    ``pattern`` is treated as a regex when it compiles, else as a plain
    substring. Returns ``{"total_matches", "returned", "hits"}`` where each
    hit is a compact event dict; when matches exceed the cap, the MOST
    RECENT ones are returned (refine the pattern to reach older ones).
    ``exclude_tools`` drops events for the named tools (the history tools
    exclude their own past invocations so a search never matches itself).
    """
    max_hits = max(1, min(int(max_hits), SEARCH_MAX_HITS))
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error:
        rx = re.compile(re.escape(pattern), re.IGNORECASE)

    matches = [rec for rec in _iter_events(path)
               if rec.get("tool") not in exclude_tools
               and rx.search(json.dumps(rec, ensure_ascii=False, default=str))]
    hits, used = [], 0
    for rec in reversed(matches):  # most recent first, then restore order
        line = json.dumps(rec, ensure_ascii=False, default=str)
        if len(hits) >= max_hits or used + len(line) > SEARCH_MAX_CHARS:
            break
        hits.append(rec)
        used += len(line)
    hits.reverse()
    return {"total_matches": len(matches), "returned": len(hits),
            "hits": hits}


def read_events(path, start_n: int, end_n: int) -> Dict[str, Any]:
    """Verbatim event lines for a bounded index range (drill-down)."""
    start_n, end_n = int(start_n), int(end_n)
    if end_n < start_n:
        start_n, end_n = end_n, start_n
    end_n = min(end_n, start_n + READ_MAX_EVENTS - 1)
    events, used = [], 0
    for rec in _iter_events(path):
        n = rec.get("n")
        if isinstance(n, int) and start_n <= n <= end_n:
            line = json.dumps(rec, ensure_ascii=False, default=str)
            if used + len(line) > READ_MAX_CHARS:
                break
            events.append(rec)
            used += len(line)
    return {"start_n": start_n, "end_n": end_n, "returned": len(events),
            "events": events}


# ------------------------------------------------------- tool registration

HISTORY_TOOL_NAMES = ("search_session_history", "get_history_events")


def register_history_tools(register_tool, events_path_fn) -> None:
    """Register the two bounded history-access tools on an orchestrator.

    ``register_tool`` is the tools class's ``_register_tool`` bound method
    (same signature on all four orchestrators); ``events_path_fn`` returns
    the session's ``events.jsonl`` path at call time (base_dir can change
    on restore). Shared engine, thin per-mode registration — the
    ``utils/file_edit.py`` pattern.
    """

    def search_session_history(pattern: str, max_hits: int = 10) -> str:
        print(f"  ⚡ Tool: Searching session history for '{pattern}'...")
        out = search_events(events_path_fn(), pattern, max_hits,
                            exclude_tools=frozenset(HISTORY_TOOL_NAMES))
        if out["total_matches"] == 0:
            return json.dumps({
                "status": "success", "total_matches": 0,
                "message": "No matching events in this session's history.",
            })
        return json.dumps({"status": "success", **out}, ensure_ascii=False,
                          default=str)

    def get_history_events(start_n: int, end_n: int) -> str:
        print(f"  ⚡ Tool: Reading history events {start_n}-{end_n}...")
        out = read_events(events_path_fn(), start_n, end_n)
        if out["returned"] == 0:
            return json.dumps({
                "status": "success", "returned": 0,
                "message": f"No events with n in [{start_n}, {end_n}].",
            })
        return json.dumps({"status": "success", **out}, ensure_ascii=False,
                          default=str)

    register_tool(
        func=search_session_history,
        name="search_session_history",
        description=(
            "Search this session's own action history — one compact record "
            "per past tool call (tool, arguments gist, outcome, files, "
            "event index n). Earlier turns may have been trimmed from the "
            "chat context; search here BEFORE asking the user to repeat "
            "information or redoing prior work. Matches are case-"
            "insensitive (regex or plain substring); when there are more "
            "matches than max_hits, the most recent are returned."
        ),
        parameters={
            "pattern": {
                "type": "string",
                "description": (
                    "Regex or plain substring to search for, e.g. a file "
                    "name, tool name, parameter, or result fragment."
                ),
            },
            "max_hits": {
                "type": "integer",
                "description": "Maximum matching events to return "
                               "(default 10, capped at 50).",
            },
        },
        required=["pattern"],
    )
    register_tool(
        func=get_history_events,
        name="get_history_events",
        description=(
            "Retrieve the verbatim history records for a small range of "
            "event indices (the `n` values reported by "
            "search_session_history) — the drill-down for when a search "
            "hit's one-line summary isn't enough."
        ),
        parameters={
            "start_n": {"type": "integer",
                        "description": "First event index (inclusive)."},
            "end_n": {"type": "integer",
                      "description": "Last event index (inclusive; at most "
                                     "40 events per call)."},
        },
        required=["start_n", "end_n"],
    )
