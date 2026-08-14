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
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

__all__ = [
    "FeedbackRequest",
    "FeedbackChannel",
    "ConsoleChannel",
    "request_human_feedback",
    "get_channel",
    "set_default_channel",
    "set_thread_channel",
    "use_channel",
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
    return get_channel().ask(req)
