"""Per-session event stream: bounded ring buffer + SSE framing.

Producers are plain threads (the turn runner, the HITL channel, endpoint
handlers); consumers are SSE connections. A ``threading.Condition`` bridges
the two — the SSE generator runs synchronously (FastAPI iterates sync
generators in a threadpool), so no asyncio hand-off is needed.

Every event gets a monotonically increasing ``id`` so a reconnecting
``EventSource`` can replay from ``Last-Event-ID``; events older than the
ring buffer are gone, and the client falls back to the session snapshot
endpoint for state (mirrors the Streamlit UI, which also only shows a tail).
"""

from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

_RING_SIZE = 5000
_HEARTBEAT_S = 15.0


@dataclass(frozen=True)
class Event:
    id: int
    type: str
    data: Dict[str, Any]


class EventBuffer:
    """Thread-safe append-only ring of events for one session."""

    def __init__(self, ring_size: int = _RING_SIZE) -> None:
        self._ring: deque[Event] = deque(maxlen=ring_size)
        self._cond = threading.Condition()
        self._next_id = 1
        self._closed = False

    @property
    def cursor(self) -> int:
        """Id of the newest event (0 when empty) — a client that starts its
        stream with ``after=cursor`` sees only what happens next."""
        with self._cond:
            return self._next_id - 1

    def emit(self, type: str, data: Optional[Dict[str, Any]] = None) -> Event:
        with self._cond:
            ev = Event(id=self._next_id, type=type, data=data or {})
            self._next_id += 1
            self._ring.append(ev)
            self._cond.notify_all()
            return ev

    def close(self) -> None:
        """Wake all streams so they can terminate (session teardown)."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def _events_after(self, after_id: int) -> list[Event]:
        return [ev for ev in self._ring if ev.id > after_id]

    def sse_stream(self, last_event_id: Optional[int] = None) -> Iterator[str]:
        """Yield SSE frames forever (until the buffer is closed).

        Replays buffered events newer than ``last_event_id``, then blocks for
        new ones, emitting a comment-line heartbeat while idle so proxies and
        the browser keep the connection alive.
        """
        cursor = last_event_id or 0
        while True:
            with self._cond:
                pending = self._events_after(cursor)
                if not pending:
                    if self._closed:
                        return
                    self._cond.wait(timeout=_HEARTBEAT_S)
                    pending = self._events_after(cursor)
                    if not pending:
                        if self._closed:
                            return
                        yield ": heartbeat\n\n"
                        continue
            for ev in pending:
                cursor = ev.id
                payload = json.dumps(ev.data, ensure_ascii=False, default=str)
                yield f"id: {ev.id}\nevent: {ev.type}\ndata: {payload}\n\n"
