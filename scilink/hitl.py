"""Human-in-the-loop chokepoint.

Every agent-level human-feedback prompt in SciLink routes through
``request_human_feedback`` instead of calling ``input()`` directly. The
default channel simply calls ``builtins.input`` (resolved at call time, so
front-ends that intercept ``builtins.input`` keep working), which makes the
console behavior byte-identical to the pre-chokepoint code. Front-ends can
install richer channels — process-wide via ``set_default_channel`` or for
the current thread via ``set_thread_channel`` — to serve prompts through a
UI, a queue, or a remote transport.

Out of scope by design: CLI session-bootstrap/REPL prompts, destructive-op
confirmations in ``scilink memory`` / ``scilink kb``, and the sandbox
consent prompt in ``executors.py`` (TTY-gated security surface). Those are
not agent decision points and keep calling ``input()`` directly.
"""

from __future__ import annotations

import builtins
import itertools
import queue as _queue_mod
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

__all__ = [
    "FeedbackRequest",
    "FeedbackChannel",
    "ConsoleChannel",
    "QueueChannel",
    "request_human_feedback",
    "get_channel",
    "set_default_channel",
    "set_thread_channel",
    "use_channel",
    "set_thread_feedback_log",
    "get_thread_feedback_log",
    "use_feedback_log",
]

_counter = itertools.count(1)


def _next_id() -> str:
    return f"q_{int(time.time())}_{next(_counter):04d}"


@dataclass
class FeedbackRequest:
    """A structured human-feedback question.

    ``kind`` identifies the decision type so channels can render the right
    surface (a radio for ``bestofn_select``, buttons for ``confirm``, a text
    box for ``free_text``) without sniffing the prompt string. ``default``
    is the answer that means "accept as-is" — for almost every SciLink
    prompt that is the empty string (press Enter). ``origin`` carries
    routing metadata (agent label, pipeline stage, fan-out branch thread
    id) and must stay JSON-serializable.
    """

    prompt: str
    kind: str = "free_text"
    options: Optional[List[str]] = None
    default: str = ""
    context: str = ""
    origin: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_next_id)
    created_at: float = field(default_factory=time.time)


@runtime_checkable
class FeedbackChannel(Protocol):
    def ask(self, req: FeedbackRequest) -> str:  # pragma: no cover - protocol
        ...


class ConsoleChannel:
    """Blocking console prompt — the process default.

    Calls ``builtins.input`` dynamically so a front-end that monkeypatches
    ``builtins.input`` (the Streamlit UI does, until it installs a channel
    of its own) still intercepts every prompt.
    """

    def ask(self, req: FeedbackRequest) -> str:
        return builtins.input(req.prompt)


class QueueChannel:
    """Parks requests from concurrent worker threads for serial serving.

    Worker threads (fan-out branches) install this as their thread channel;
    ``ask`` enqueues the request and blocks until a coordinator thread —
    the one that owns the human — answers it via ``serve_pending``, which
    relays each queued request through the coordinator's own active
    channel (console prompt, UI modal, ...) one at a time.

    ``timeout_s`` bounds how long a worker waits for an answer; on timeout
    the request's ``default`` answer is returned so an unattended branch
    degrades to accept-as-is instead of hanging forever.
    """

    def __init__(self, timeout_s: Optional[float] = None) -> None:
        self._queue: _queue_mod.Queue = _queue_mod.Queue()
        self.timeout_s = timeout_s

    def ask(self, req: FeedbackRequest) -> str:
        event = threading.Event()
        holder: Dict[str, str] = {}
        self._queue.put((req, holder, event))
        if not event.wait(self.timeout_s):
            holder.setdefault("answer", req.default)
        return holder.get("answer", req.default)

    def serve_pending(self, through: Optional[FeedbackChannel] = None) -> int:
        """Serve every queued request now; returns how many were answered.

        Called periodically from the coordinator's wait loop. A label from
        ``origin['branch_label']`` is prefixed onto the prompt so the human
        knows which branch is asking. If the serving channel raises (EOF,
        stop, interrupt), the waiting worker is unblocked with the
        request's default answer before the exception propagates.
        """
        served = 0
        while True:
            try:
                req, holder, event = self._queue.get_nowait()
            except _queue_mod.Empty:
                return served
            label = req.origin.get("branch_label")
            to_serve = (replace(req, prompt=f"\n[branch: {label}]{req.prompt}")
                        if label else req)
            try:
                holder["answer"] = (through or get_channel()).ask(to_serve)
            except BaseException:
                holder.setdefault("answer", req.default)
                event.set()
                raise
            event.set()
            served += 1


_default_channel: FeedbackChannel = ConsoleChannel()
_thread_local = threading.local()


def set_default_channel(channel: Optional[FeedbackChannel]) -> None:
    """Install the process-wide channel (``None`` restores the console)."""
    global _default_channel
    _default_channel = channel if channel is not None else ConsoleChannel()


def set_thread_channel(channel: Optional[FeedbackChannel]) -> None:
    """Install (or with ``None`` clear) the channel for the current thread.

    Thread-local overrides win over the process default; fan-out branches
    and UI worker threads use this so concurrent prompts route to their
    own owner instead of colliding on one console.
    """
    _thread_local.channel = channel


@contextmanager
def use_channel(channel: FeedbackChannel):
    """Scoped thread-local channel override."""
    previous = getattr(_thread_local, "channel", None)
    _thread_local.channel = channel
    try:
        yield channel
    finally:
        _thread_local.channel = previous


def get_channel() -> FeedbackChannel:
    channel = getattr(_thread_local, "channel", None)
    return channel if channel is not None else _default_channel


# --------------------------------------------------------------- durability

def set_thread_feedback_log(path) -> None:
    """Bind this thread's prompts to a session feedback log.

    ``path`` is the JSONL file (conventionally
    ``<session>/feedback_log.jsonl``); every request asked on this thread
    is appended at ask time and again at answer time, and a
    ``pending_question.json`` sidecar next to the log marks the question
    currently awaiting an answer (removed once answered — if a run dies
    while blocked, the sidecar says what it was waiting for). ``None``
    unbinds. Orchestrators bind their session file around each chat turn.
    """
    _thread_local.feedback_log = path


def get_thread_feedback_log():
    return getattr(_thread_local, "feedback_log", None)


@contextmanager
def use_feedback_log(path):
    """Scoped thread-local feedback-log binding (re-entrant safe)."""
    previous = get_thread_feedback_log()
    _thread_local.feedback_log = path
    try:
        yield
    finally:
        _thread_local.feedback_log = previous


def _append_record(path, record: Dict[str, Any]) -> None:
    """Best-effort append — the log must never break the prompt itself."""
    import json

    try:
        p = __import__("pathlib").Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:  # noqa: BLE001
        pass


def _write_pending(path, req: FeedbackRequest) -> None:
    import json

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"id": req.id, "kind": req.kind, "prompt": req.prompt,
                       "origin": req.origin, "asked_at": req.created_at},
                      f, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        pass


def _clear_pending(path) -> None:
    try:
        __import__("pathlib").Path(path).unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


def request_human_feedback(
    prompt: str,
    *,
    kind: str = "free_text",
    options: Optional[List[str]] = None,
    default: str = "",
    context: str = "",
    origin: Optional[Dict[str, Any]] = None,
) -> str:
    """Ask the human a question through the active feedback channel.

    Returns the raw answer string exactly as ``input()`` would (no
    stripping — call sites keep their own normalization so behavior stays
    identical). Exceptions from the channel (``EOFError``,
    ``KeyboardInterrupt``, stop signals) propagate to the call site, which
    keeps its existing fallback handling.
    """
    req = FeedbackRequest(
        prompt=prompt,
        kind=kind,
        options=options,
        default=default,
        context=context,
        origin=origin or {},
    )
    log = get_thread_feedback_log()
    pending = None
    if log:
        from pathlib import Path as _Path

        _append_record(log, {"id": req.id, "event": "asked",
                             "kind": req.kind, "prompt": req.prompt,
                             "options": req.options, "default": req.default,
                             "origin": req.origin, "t": req.created_at})
        pending = _Path(log).parent / "pending_question.json"
        _write_pending(pending, req)
    try:
        answer = get_channel().ask(req)
    except BaseException as exc:
        if log:
            _append_record(log, {
                "id": req.id, "event": "interrupted",
                "error": type(exc).__name__,
                "elapsed_s": round(time.time() - req.created_at, 3)})
        raise
    finally:
        if pending is not None:
            _clear_pending(pending)
    if log:
        _append_record(log, {
            "id": req.id, "event": "answered", "answer": answer,
            "elapsed_s": round(time.time() - req.created_at, 3)})
    return answer
