"""Data sources for live-monitoring sessions.

A :class:`LiveDataSource` knows how to detect "new data" and hand the
current payload to the reading loop. Each reading the session calls
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
    ``push(...)``; the reading loop consumes them via ``read_latest()``.
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
    """The payload :class:`LiveDataSource.read_latest` hands to the reading loop.

    The `source_kind`, `path`, `text` fields are convention; reading
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
    return ``None`` and let the next reading try again rather than stall
    the reading loop on a transient OSError.
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


class DirectoryWatchSource:
    """Watch a directory for new files; return the contents of the newest each reading.

    Right shape for "each new datapoint is a new file" scenarios — time-
    resolved XRD where the instrument writes ``scan_0001.txt``,
    ``scan_0002.txt``, ... as the experiment progresses, or a Raman
    series where every acquisition is its own file in a session
    directory.

    Each reading:
      - lists files in ``directory`` matching ``pattern``
      - selects the latest by ``sort_by`` ("mtime" default; "name" for
        zero-padded sequential filenames where alphabetical = chronological)
      - returns its text content the first time it's seen
      - returns None on subsequent readings until a newer file appears

    For workflows where you want EVERY new file (not just the latest)
    surfaced one-at-a-time, use ``strategy='unseen'`` — each reading the
    source returns the next-unseen file in sort order, so a burst of N
    files spawns N readings before going quiet.
    """

    def __init__(
        self,
        directory: str | Path,
        pattern: str = "*.txt",
        *,
        strategy: str = "newest",
        sort_by: str = "mtime",
    ):
        if strategy not in ("newest", "unseen"):
            raise ValueError(
                f"strategy must be 'newest' or 'unseen'; got {strategy!r}"
            )
        if sort_by not in ("mtime", "name"):
            raise ValueError(
                f"sort_by must be 'mtime' or 'name'; got {sort_by!r}"
            )
        self.directory = Path(directory)
        self.pattern = pattern
        self.strategy = strategy
        self.sort_by = sort_by
        # 'newest' tracks just the last-returned path.
        # 'unseen' tracks the full set of returned paths.
        self._last_returned: Optional[Path] = None
        self._seen: set[Path] = set()
        self.name = "directory_watch"

    def _list_files(self) -> list[Path]:
        try:
            files = list(self.directory.glob(self.pattern))
        except OSError:
            return []
        if self.sort_by == "mtime":
            def _key(p: Path) -> float:
                try:
                    return p.stat().st_mtime
                except OSError:
                    return 0.0
            files.sort(key=_key)
        else:  # "name"
            files.sort(key=lambda p: p.name)
        return files

    def read_latest(self) -> Optional[LatestData]:
        if not self.directory.is_dir():
            return None
        files = self._list_files()
        if not files:
            return None
        if self.strategy == "newest":
            chosen = files[-1]
            if chosen == self._last_returned:
                return None
            self._last_returned = chosen
        else:  # "unseen"
            chosen = None
            for f in files:
                if f not in self._seen:
                    chosen = f
                    break
            if chosen is None:
                return None
            self._seen.add(chosen)
        try:
            text = chosen.read_text()
        except OSError as e:
            _logger.debug("read_text failed for %s: %s", chosen, e)
            return None
        return LatestData(
            timestamp=time.time(),
            source_kind=self.name,
            path=chosen,
            text=text,
            extras={
                "directory": str(self.directory),
                "pattern": self.pattern,
                "filename": chosen.name,
                "n_files_total": len(files),
                "strategy": self.strategy,
            },
        )

    def reset(self) -> None:
        self._last_returned = None
        self._seen.clear()


class CallbackSource:
    """In-process source. Caller pushes payloads; the reading loop consumes them.

    Used by:
      - tests (the test pushes synthetic data on a schedule)
      - future instrument-API integrations whose vendor SDK delivers
        new data via its own callback / event-loop and you want to
        bridge into SciLink's reading loop without polling a file

    Holds one pending payload at a time. A push while a previous push
    is still un-consumed replaces it — live mode cares about the most
    recent snapshot, not the full history. The JSONL stream preserves
    every observed reading anyway.
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
    "DirectoryWatchSource",
    "CallbackSource",
]
