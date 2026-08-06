"""Small store I/O helpers: atomic publish and a cross-process file lock.

The persistent store under ``~/.scilink`` is shared across every session on a
machine. Nothing in the package previously guarded concurrent writers, so two
parallel sessions could silently lose each other's updates (issue #398).

Two primitives, both intentionally tiny:

- ``atomic_write_text`` — write to a sibling temp file then ``os.replace``,
  so a reader never observes a half-written file. This is the same publish
  the tool cache already uses (``_tool_cache.py``).
- ``file_lock`` — an exclusive advisory lock on a sidecar ``.lock`` file via
  ``fcntl.flock``, held for the duration of a read-modify-write. On platforms
  without ``fcntl`` (Windows) it degrades to a no-op context manager, so the
  store stays importable everywhere; atomicity still protects torn reads.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None


def atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically (tmp file + ``os.replace``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive advisory lock for a read-modify-write on ``path``.

    The lock file lives alongside ``path`` as ``<name>.lock``. No-op where
    ``fcntl`` is unavailable.
    """
    if fcntl is None:  # pragma: no cover - Windows
        yield
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with open(lock_path, "a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
