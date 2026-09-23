"""
Opt-in tracing of LLM calls.

By default SciLink records nothing about individual LLM calls — no prompts, responses,
or token usage. Enabling tracing appends one JSON record per call to a JSONL file,
capturing the model, prompt messages, response text, token usage, and latency. This is
useful for cost accounting, model comparisons, and variability / reproducibility studies.

Enable in code::

    import scilink
    scilink.enable_tracing("run/llm_trace.jsonl")
    ...
    scilink.disable_tracing()

or via the environment (auto-enabled on first call)::

    export SCILINK_TRACE_FILE=run/llm_trace.jsonl

Each line of the trace file is a JSON object::

    {"timestamp", "model", "messages", "response_text", "finish_reason",
     "usage": {"prompt_tokens", "completion_tokens", "total_tokens"}, "latency_s"}

Tracing is global and opt-in: the LLM wrapper has no per-session output directory, so a
single sink keeps it simple, and nothing is recorded unless tracing is enabled.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

_lock = threading.Lock()
_trace_path: Optional[str] = None
_env_checked = False


# ── Always-on call counters ──────────────────────────────────────
# Independent of the opt-in file tracer: every LLM call through the
# wrappers is counted, so a front-end can show "N calls · P/C tokens" per
# turn by diffing two snapshots. Process-wide, so concurrent sessions in
# one process (the web backend) cross-charge each other; a front-end that
# runs one turn at a time (the terminal shell) reads them exactly.
_counters: Dict[str, float] = {
    "calls": 0, "seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
}
_counters_lock = threading.Lock()


def note_llm_call(latency_s: Optional[float] = None,
                  prompt_tokens: Optional[int] = None,
                  completion_tokens: Optional[int] = None) -> None:
    """Count one completed LLM call (missing token counts count as 0)."""
    with _counters_lock:
        _counters["calls"] += 1
        _counters["seconds"] += float(latency_s or 0.0)
        _counters["prompt_tokens"] += int(prompt_tokens or 0)
        _counters["completion_tokens"] += int(completion_tokens or 0)


def llm_counters() -> Dict[str, float]:
    """A snapshot of the process-wide counters."""
    with _counters_lock:
        return dict(_counters)


def counters_delta(before: Dict[str, float]) -> Dict[str, float]:
    """What happened since ``before`` (a prior ``llm_counters()``)."""
    now = llm_counters()
    return {k: now[k] - before.get(k, 0) for k in now}


def enable_tracing(path: str) -> None:
    """Append a JSON record (one line) for every subsequent LLM call to ``path``."""
    global _trace_path
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    _trace_path = path


def disable_tracing() -> None:
    """Turn off LLM-call tracing."""
    global _trace_path
    _trace_path = None


# ──────────────────────────────────────────────────────────────
# Always-on call counters
#
# Distinct from the opt-in trace above: no prompt, response or file is
# involved — only how many LLM calls completed in this process, how long they
# took and how many tokens they moved. Stage timing
# (``exp_agents/_stage_timing.py``) reads deltas of these to split a pipeline
# stage's wall-clock into LLM time and everything else.
# ──────────────────────────────────────────────────────────────

_counters: Dict[str, float] = {
    "calls": 0, "seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
    "off_path_calls": 0,
}
_off_path = threading.local()


class off_path:
    """Context manager: LLM calls made on this thread inside the block are
    counted as ``off_path_calls`` and NOT in ``calls``.

    For work that runs beside a timed pipeline rather than inside it — a live
    loop's slow recommender thinking on its own thread while frames are being
    analysed. Without it the process-wide ``calls`` counter would charge that
    call to whichever pipeline stage happened to be running, and a frame that
    called no model would be reported as having used one.
    """

    def __enter__(self):
        _off_path.depth = getattr(_off_path, "depth", 0) + 1
        return self

    def __exit__(self, *exc):
        _off_path.depth -= 1
        return False


def note_llm_call(latency_s: Optional[float] = None,
                  prompt_tokens: Optional[int] = None,
                  completion_tokens: Optional[int] = None) -> None:
    """Count one completed LLM call. Never raises."""
    try:
        if getattr(_off_path, "depth", 0):
            with _lock:
                _counters["off_path_calls"] += 1
            return
        with _lock:
            _counters["calls"] += 1
            _counters["seconds"] += float(latency_s or 0.0)
            _counters["prompt_tokens"] += int(prompt_tokens or 0)
            _counters["completion_tokens"] += int(completion_tokens or 0)
    except Exception:
        pass


def llm_counters() -> Dict[str, float]:
    """Snapshot of the process-wide LLM-call counters.

    Process-wide by design: a stage that fans work out to worker threads still
    sees their calls. The cost is that two pipelines running concurrently in
    one process (a meta fan-out) see each other's calls; calls made in a
    spawned worker process are not seen at all.
    """
    with _lock:
        return dict(_counters)


def is_enabled() -> bool:
    """Whether a trace destination is active (set via enable_tracing or $SCILINK_TRACE_FILE)."""
    return _active_path() is not None


def _active_path() -> Optional[str]:
    """Resolve the active trace path, honoring $SCILINK_TRACE_FILE on first use."""
    global _trace_path, _env_checked
    if _trace_path is None and not _env_checked:
        _env_checked = True
        env = os.environ.get("SCILINK_TRACE_FILE")
        if env:
            enable_tracing(env)
    return _trace_path


def record(model: str,
           messages: Any,
           response_text: str,
           finish_reason: Optional[str] = None,
           usage: Optional[Dict[str, Any]] = None,
           latency_s: Optional[float] = None,
           extra: Optional[Dict[str, Any]] = None) -> None:
    """Append one LLM-call record. No-op when tracing is disabled; never raises."""
    path = _active_path()
    if not path:
        return
    rec: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "messages": _sanitize_messages(messages),
        "response_text": response_text,
        "finish_reason": finish_reason,
        "usage": usage,
        "latency_s": latency_s,
    }
    if extra:
        rec.update(extra)
    try:
        line = json.dumps(rec, default=str)
        with _lock:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception:
        # Tracing must never break a generation.
        pass


def _sanitize_messages(messages: Any) -> Any:
    """Keep prompt text but replace inline image payloads with a marker, so traces stay small."""
    if not isinstance(messages, list):
        return messages
    out: List[Dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            out.append({"role": None, "content": str(m)})
            continue
        role, content = m.get("role"), m.get("content")
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict):
                    if p.get("type") == "text":
                        parts.append(p.get("text", ""))
                    elif p.get("type") in ("image_url", "image"):
                        parts.append("<image>")
                    else:
                        parts.append(f"<{p.get('type', 'part')}>")
                else:
                    parts.append(str(p))
            content = " ".join(parts)
        out.append({"role": role, "content": content})
    return out
