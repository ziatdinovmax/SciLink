"""Thread-routing stdout/stderr capture for concurrent sessions.

The Streamlit-inherited ``OutputCapture`` swaps the PROCESS-GLOBAL
``sys.stdout``, so with two sessions' turns running at once each capture
tees everything the other prints — session B's verbose panel narrated
session A's analysis (observed live once Detach made concurrent sessions an
everyday path).

This module installs one dispatching proxy over the real stdout/stderr
(idempotent, process-lifetime) and routes each ``write()`` by the CALLING
thread: a turn registers its chat thread with a buffer + stop event, and
``log_context.effective_thread`` maps the agents' fan-out worker threads
back to their owning turn — so a print lands in exactly its own session's
capture (and still reaches the real console). Unrouted threads (HTTP
handlers, the main thread) pass straight through.

``RoutedCapture`` keeps the surface of ``OutputCapture`` that the turn
runner uses (context manager, ``getvalue``, ``request_stop`` with the
print-driven ``AgentStoppedError`` + subprocess kill), so the runner swaps
classes and nothing else.
"""

from __future__ import annotations

import io
import sys
import threading
from typing import Dict, Optional, TextIO, Tuple

from scilink.ui.output_capture import AgentStoppedError
from scilink.utils.log_context import effective_thread

# chat thread id -> (buffer, buffer lock, stop event)
_ROUTES: Dict[int, Tuple[io.StringIO, threading.Lock, threading.Event]] = {}
_ROUTES_LOCK = threading.Lock()
# Reentrancy guard: if something later wraps OUR proxy and install() adds a
# second router around it, only the outermost may write to the buffer, or
# every line would capture twice.
_tls = threading.local()


class _RoutingStream:
    """Tee to the real stream plus the calling thread's registered buffer."""

    def __init__(self, original: TextIO) -> None:
        self._original = original

    def write(self, data: str) -> int:
        route = _ROUTES.get(effective_thread(threading.get_ident()))
        if route is not None:
            buffer, lock, stop = route
            if stop.is_set():
                raise AgentStoppedError("Agent stopped by user")
            if not getattr(_tls, "routing", False):
                _tls.routing = True
                try:
                    with lock:
                        buffer.write(data)
                finally:
                    _tls.routing = False
        self._original.write(data)
        return len(data)

    def flush(self) -> None:
        self._original.flush()

    # Delegate attribute access so .isatty() etc. keep working.
    def __getattr__(self, name: str):
        return getattr(self._original, name)


def install() -> None:
    """Ensure sys.stdout/sys.stderr are routing proxies.

    Checked against the LIVE streams rather than a done-once flag: anything
    that replaces the streams later (pytest's per-test capture is the known
    case) silently unwinds the proxy, and a flag would leave every
    subsequent turn uncaptured."""
    if not isinstance(sys.stdout, _RoutingStream):
        sys.stdout = _RoutingStream(sys.stdout)
    if not isinstance(sys.stderr, _RoutingStream):
        sys.stderr = _RoutingStream(sys.stderr)


class RoutedCapture:
    """Per-turn capture registered with the router instead of swapping the
    global streams — drop-in for the runner's use of ``OutputCapture``."""

    def __init__(self) -> None:
        self._buffer = io.StringIO()
        self._buffer_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._agent_thread_id: Optional[int] = None

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def request_stop(self) -> None:
        """Abort the turn on its next print/log; kill its subprocesses."""
        self._stop_event.set()
        if self._agent_thread_id is not None:
            try:
                from scilink.executors import kill_subprocesses_for_thread
                kill_subprocesses_for_thread(self._agent_thread_id)
            except Exception:
                pass  # Best-effort; the stop event still propagates.

    def __enter__(self) -> "RoutedCapture":
        install()
        self._agent_thread_id = threading.get_ident()
        with _ROUTES_LOCK:
            _ROUTES[self._agent_thread_id] = (
                self._buffer, self._buffer_lock, self._stop_event)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        with _ROUTES_LOCK:
            _ROUTES.pop(self._agent_thread_id, None)

    def getvalue(self) -> str:
        with self._buffer_lock:
            return self._buffer.getvalue()

    @property
    def log_stream(self) -> "_LockedWriter":
        """Stream for the turn's logging handler — same buffer, same lock,
        so logging and print writes never interleave mid-write."""
        return _LockedWriter(self)


class _LockedWriter:
    def __init__(self, cap: RoutedCapture) -> None:
        self._cap = cap

    def write(self, data: str) -> int:
        with self._cap._buffer_lock:
            self._cap._buffer.write(data)
        return len(data)

    def flush(self) -> None:
        pass
