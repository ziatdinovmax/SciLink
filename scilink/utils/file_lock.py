"""A cross-process, cross-thread exclusive lock on a path.

Shared stores under ``~/.scilink`` (model weights, knowledge bases, the
sessions index) are written by several analyses at once: fan-out branches and
best-of-N candidates are threads, series replays and separate CLI sessions are
processes. ``path_lock`` serializes a read-modify-write on one path across all
of them. It is the same ``flock`` pattern as the script bank's domain lock,
without the bank's re-entrancy bookkeeping.
"""
from __future__ import annotations

import contextlib
import threading
from pathlib import Path
from typing import Any, Iterator

# Without fcntl (Windows) the lock is per process only: threads still
# serialize, processes do not. Every fan-out and best-of-N worker is a thread.
_local_locks: dict = {}
_local_locks_guard = threading.Lock()


@contextlib.contextmanager
def path_lock(path: Any) -> Iterator[None]:
    """Exclusive lock on ``path`` held through ``<path>.lock`` beside it.

    ``flock`` on a freshly opened descriptor conflicts with every other
    descriptor on the file, in this process or another, so the one lock
    covers threads and processes alike. Not reentrant.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lock_path = p.parent / f"{p.name}.lock"
    try:
        import fcntl
    except ImportError:          # pragma: no cover - non-POSIX
        with _local_locks_guard:
            lock = _local_locks.setdefault(str(lock_path), threading.Lock())
        with lock:
            yield
        return
    with open(lock_path, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
