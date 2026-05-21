"""Data sources for live-monitoring sessions.

A :class:`LiveDataSource` knows how to detect "new data" and hand the
current payload to the tick loop. Each tick the session calls
``source.read_latest()``; the source returns a :class:`LatestData`
object when something new exists and ``None`` otherwise.

Three concrete sources ship in v1:

  - :class:`MtimePollFileSource` — watches a file by ``stat().st_mtime``
    and returns the full content on every change. Universal for any
    instrument that rewrites a single output file as it scans.
  - :class:`AppendOnlyFileSource` — tracks a read offset and returns
    only the appended chunk on each call. Right shape for grow-by-
    append formats (some diffractometers and many streaming data
    pipelines).
  - :class:`CallbackSource` — in-process. Caller pushes payloads via
    ``push(...)``; the tick loop consumes them via ``read_latest()``.
    Used by tests and by future instrument-API integrations that have
    their own callback / websocket / message-queue layer.

Adding a new source is one class implementing the protocol — no
framework changes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

_logger = logging.getLogger(__name__)


@dataclass
class LatestData:
    """The payload :class:`LiveDataSource.read_latest` hands to the tick loop.

    The `source_kind`, `path`, `text` fields are convention; tick
    functions read the fields they understand. Anything source-specific
    that doesn't fit the common shape lives in ``extras`` — e.g.
    :class:`AppendOnlyFileSource` records ``offset_start`` / ``offset_end``
    there so a downstream consumer can replay just the new chunk.
    """

    timestamp: float
    source_kind: str
    path: Optional[Path] = None
    text: Optional[str] = None
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LiveDataSource(Protocol):
    """Minimal contract every data source satisfies.

    ``read_latest`` must be safe to call from a background thread at
    high frequency. It must NOT block on slow I/O — file sources should
    return ``None`` and let the next tick try again rather than stall
    the tick loop on a transient OSError.
    """

    name: str

    def read_latest(self) -> Optional[LatestData]: ...

    def reset(self) -> None: ...


# ---------------------------------------------------------------------------
# Concrete sources
# ---------------------------------------------------------------------------


class MtimePollFileSource:
    """Watches a single file by mtime; returns the full content on change.

    Universal for instruments that rewrite their output file as the
    scan progresses (any diffractometer or spectrometer with a "save
    pattern" button that overwrites). First read always returns the
    current content; subsequent reads return ``None`` until mtime
    advances.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._last_mtime: Optional[float] = None
        self.name = "mtime_poll"

    def read_latest(self) -> Optional[LatestData]:
        if not self.path.is_file():
            return None
        try:
            mtime = self.path.stat().st_mtime
        except OSError as e:
            _logger.debug("stat failed for %s: %s", self.path, e)
            return None
        if self._last_mtime is not None and mtime <= self._last_mtime:
            return None
        try:
            text = self.path.read_text()
        except OSError as e:
            _logger.debug("read_text failed for %s: %s", self.path, e)
            return None
        self._last_mtime = mtime
        return LatestData(
            timestamp=time.time(),
            source_kind=self.name,
            path=self.path,
            text=text,
            extras={"file_mtime": mtime},
        )

    def reset(self) -> None:
        self._last_mtime = None


class AppendOnlyFileSource:
    """Tracks a read offset; returns only the newly-appended chunk per call.

    For data formats where each measurement step appends a line (or a
    block of bytes) to the same file rather than rewriting the whole
    file. First read returns the entire file as the "first append";
    subsequent reads return only what has been added since.

    If the file is truncated below the tracked offset, treats it as
    "nothing new" — caller can ``reset()`` to start over.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._offset = 0
        self.name = "append_only"

    def read_latest(self) -> Optional[LatestData]:
        if not self.path.is_file():
            return None
        try:
            size = self.path.stat().st_size
        except OSError as e:
            _logger.debug("stat failed for %s: %s", self.path, e)
            return None
        if size <= self._offset:
            return None
        try:
            with self.path.open("rb") as f:
                f.seek(self._offset)
                chunk_bytes = f.read()
        except OSError as e:
            _logger.debug("read failed for %s: %s", self.path, e)
            return None
        old_offset = self._offset
        self._offset = size
        try:
            text: Optional[str] = chunk_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = None
        return LatestData(
            timestamp=time.time(),
            source_kind=self.name,
            path=self.path,
            text=text,
            extras={
                "offset_start": old_offset,
                "offset_end": size,
                "bytes": chunk_bytes,
            },
        )

    def reset(self) -> None:
        self._offset = 0


class CallbackSource:
    """In-process source. Caller pushes payloads; the tick loop consumes them.

    Used by:
      - tests (the test pushes synthetic data on a schedule)
      - future instrument-API integrations whose vendor SDK delivers
        new data via its own callback / event-loop and you want to
        bridge into SciLink's tick loop without polling a file

    Holds one pending payload at a time. A push while a previous push
    is still un-consumed replaces it — live mode cares about the most
    recent snapshot, not the full history. The JSONL stream preserves
    every observed tick anyway.
    """

    def __init__(self) -> None:
        self._buffer: Optional[LatestData] = None
        self._consumed = False
        self.name = "callback"

    def push(self, *, text: Optional[str] = None,
              path: Optional[Path] = None,
              extras: Optional[dict] = None) -> None:
        self._buffer = LatestData(
            timestamp=time.time(),
            source_kind=self.name,
            path=path,
            text=text,
            extras=dict(extras) if extras else {},
        )
        self._consumed = False

    def read_latest(self) -> Optional[LatestData]:
        if self._buffer is None or self._consumed:
            return None
        self._consumed = True
        return self._buffer

    def reset(self) -> None:
        self._buffer = None
        self._consumed = False


__all__ = [
    "LatestData",
    "LiveDataSource",
    "MtimePollFileSource",
    "AppendOnlyFileSource",
    "CallbackSource",
]
