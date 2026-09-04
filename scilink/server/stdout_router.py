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

class _Route:
    """One turn's capture target + console-attribution state."""

    __slots__ = ("buffer", "lock", "stop", "tag", "at_line_start")

    def __init__(self, buffer: io.StringIO, lock: threading.Lock,
                 stop: threading.Event, tag: str) -> None:
        self.buffer = buffer
        self.lock = lock
        self.stop = stop
        self.tag = tag
        self.at_line_start = True  # guarded by ``lock``


# chat thread id -> _Route
_ROUTES: Dict[int, _Route] = {}
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
        out = data
        route = _ROUTES.get(effective_thread(threading.get_ident()))
        if route is not None:
            if route.stop.is_set():
                raise AgentStoppedError("Agent stopped by user")
            if not getattr(_tls, "routing", False):
                _tls.routing = True
                try:
                    with route.lock:
                        route.buffer.write(data)
                        # Console attribution: with several sessions live,
                        # prefix each line with the session tag so the
                        # terminal firehose stays readable. The session's
                        # own buffer stays untagged. Single session: no
                        # prefix noise.
                        if route.tag and len(_ROUTES) > 1:
                            parts = []
                            for seg in data.splitlines(keepends=True):
                                if route.at_line_start:
                                    parts.append(f"[{route.tag}] ")
                                parts.append(seg)
                                route.at_line_start = seg.endswith("\n")
                            out = "".join(parts)
                finally:
                    _tls.routing = False
        self._original.write(out)
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

    def __init__(self, tag: str = "") -> None:
        self._buffer = io.StringIO()
        self._buffer_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._agent_thread_id: Optional[int] = None
        self._tag = tag

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
            _ROUTES[self._agent_thread_id] = _Route(
                self._buffer, self._buffer_lock, self._stop_event, self._tag)
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
