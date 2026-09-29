"""A cross-process, cross-thread exclusive lock on a path.

Shared stores under ``~/.scilink`` (model weights, knowledge bases, the
sessions index) are written by several analyses at once: fan-out branches and
best-of-N candidates are threads, series replays and separate CLI sessions are
processes. ``path_lock`` serializes a read-modify-write on one path across all
of them. It is the same ``flock`` pattern as the script bank's domain lock,
without the bank's re-entrancy bookkeeping.

Platforms:

- POSIX: ``flock`` on a freshly opened descriptor, which conflicts with every
  other descriptor on the file, in this process or another. The kernel
  releases it when the holder dies.
- Windows: ``msvcrt.locking`` on byte 0 of the lock file, also per handle, so
  it covers threads and processes alike.
- A filesystem that does not support locking (some HPC Lustre, GPFS or NFS
  mounts raise ``ENOTSUP`` / ``ENOLCK``): the caller runs unlocked, with one
  warning. Callers publish atomically, so the cost is a possible duplicate of
  the work, never a torn file.
"""
from __future__ import annotations

import contextlib
import errno
import logging
import threading
import time
from pathlib import Path
from typing import Any, Iterator, Optional

_logger = logging.getLogger(__name__)

# Neither fcntl nor msvcrt (an exotic platform): threads still serialize,
# processes do not.
_local_locks: dict = {}
_local_locks_guard = threading.Lock()

_UNSUPPORTED_ERRNOS = {errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOLCK, errno.EINVAL,
                       getattr(errno, "ENOSYS", errno.ENOTSUP)}
_warned_unsupported: set = set()


class _Unsupported(Exception):
    """The filesystem holding the lock file cannot lock."""


def lock_file_for(path: Any) -> Path:
    """The lock file that guards ``path``: ``<path>.lock`` beside it."""
    p = Path(path)
    return p.parent / f"{p.name}.lock"


def _try_acquire(fh) -> bool:
    """One non-blocking attempt. True when held, False when someone else
    holds it; raises ``_Unsupported`` when the filesystem cannot lock."""
    try:
        import fcntl
    except ImportError:
        fcntl = None
    if fcntl is not None:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EACCES):
                return False
            if exc.errno in _UNSUPPORTED_ERRNOS:
                raise _Unsupported(exc) from exc
            raise
    import msvcrt                                    # Windows
    fh.seek(0)
    try:
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        return True
    except OSError as exc:
        # Only "someone holds it" means held; anything else (a share that
        # cannot lock) falls back to running unlocked, as flock's ENOTSUP does.
        if exc.errno in (errno.EACCES, getattr(errno, "EDEADLOCK", errno.EDEADLK)):
            return False
        raise _Unsupported(exc) from exc


def _acquire_blocking(fh) -> None:
    try:
        import fcntl
    except ImportError:
        fcntl = None
    if fcntl is not None:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX)
            return
        except OSError as exc:
            if exc.errno in _UNSUPPORTED_ERRNOS:
                raise _Unsupported(exc) from exc
            raise
    while not _try_acquire(fh):                      # Windows: no blocking call
        time.sleep(0.2)


def _release(fh) -> None:
    try:
        import fcntl
    except ImportError:
        fcntl = None
    if fcntl is not None:
        fcntl.flock(fh, fcntl.LOCK_UN)
        return
    import msvcrt
    fh.seek(0)
    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)


def _has_os_lock() -> bool:
    for name in ("fcntl", "msvcrt"):
        try:
            __import__(name)
            return True
        except ImportError:
            continue
    return False


def _warn_unsupported(lock_path: Path, exc: BaseException) -> None:
    key = str(lock_path.parent)
    if key not in _warned_unsupported:
        _warned_unsupported.add(key)
        _logger.warning(f"File locking is not supported under {lock_path.parent} ({exc}); "
                        "continuing without a lock. Concurrent writers may duplicate work "
                        "there, but never see a partial file.")


@contextlib.contextmanager
def path_lock(path: Any, *, label: Optional[str] = None) -> Iterator[None]:
    """Exclusive lock on ``path`` held through ``<path>.lock`` beside it,
    across threads and processes. Not reentrant.

    ``label`` names what is being waited for in the log line written when
    another holder has it (a waiting fan-out branch otherwise spends its
    wall-clock budget without saying why).
    """
    lock_path = lock_file_for(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if not _has_os_lock():                           # pragma: no cover - exotic platform
        with _local_locks_guard:
            lock = _local_locks.setdefault(str(lock_path), threading.Lock())
        with lock:
            yield
        return
    fh = open(lock_path, "a+b")
    try:
        unsupported = None
        try:
            if not _try_acquire(fh):
                _logger.info(f"Waiting for another process working on {label or Path(path).name} ...")
                _acquire_blocking(fh)
        except _Unsupported as exc:
            unsupported = exc
        if unsupported is not None:
            # Yielded outside the except block, so an error raised by the
            # caller's body does not carry "During handling of ..." noise.
            _warn_unsupported(lock_path, unsupported.__cause__ or unsupported)
            yield
            return
        try:
            yield
        finally:
            _release(fh)
    finally:
        fh.close()


def is_locked(path: Any) -> bool:
    """Whether another holder has ``path_lock(path)`` right now. Never
    blocks; False where locking is unsupported or the lock file is absent."""
    lock_path = lock_file_for(path)
    if not lock_path.exists() or not _has_os_lock():
        return False
    try:
        with open(lock_path, "a+b") as fh:
            try:
                if _try_acquire(fh):
                    _release(fh)
                    return False
                return True
            except _Unsupported:
                return False
    except OSError:
        return False
