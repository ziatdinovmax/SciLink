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
