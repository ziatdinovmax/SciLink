"""Downloads into a shared cache: once per machine, never half-visible.

Model weights and reference files are cached per machine (``~/.scilink/models``
and friends) and fetched on first use. Several analyses can reach the same
missing file at the same moment — fan-out branches, best-of-N candidates,
series replays in spawned processes — and the cache used to be written the
simple way: straight to the final path (SAM's ``urlretrieve``) or through a
fixed ``.part`` name (the DCNN ensemble). Two writers then share one file, and
a reader that checks ``os.path.exists`` can load a checkpoint that is still
arriving.

``download_once`` closes both windows: the destination is locked for the whole
download (across threads and processes), the existence check is repeated once
the lock is held so the second caller reuses the first caller's file, the
bytes go to a unique temp file in the destination directory, and the file is
published with ``os.replace`` only when complete. A reader never sees a
partial file, only no file or the whole one.
"""
from __future__ import annotations

import contextlib
import logging
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Optional

from scilink.utils.file_lock import path_lock

_logger = logging.getLogger(__name__)


class DownloadError(OSError):
    """A download that did not produce a complete file."""


def _usable(p: Path) -> bool:
    try:
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


def fetch_to(url: str, dest: Any, *, timeout: float = 60.0) -> Path:
    """Stream ``url`` into ``dest`` through a unique same-directory temp file,
    published with ``os.replace``. Raises ``DownloadError`` (removing the
    temp file) when the transfer fails or ends short of its Content-Length.
    No lock: callers that share ``dest`` use ``download_once``."""
    p = Path(dest)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as fh, urllib.request.urlopen(url, timeout=timeout) as resp:
            expected = resp.headers.get("Content-Length")
            shutil.copyfileobj(resp, fh, length=1 << 20)
            written = fh.tell()
        if expected is not None and expected.isdigit() and written != int(expected):
            raise DownloadError(f"{url}: received {written} of {expected} bytes")
        # mkstemp creates 0600; a shared cache file gets what a plain write
        # would have produced (the umask), so other users can read it.
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, p)
        return p
    except BaseException as exc:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        if isinstance(exc, DownloadError) or not isinstance(exc, Exception):
            raise
        raise DownloadError(f"{url}: {exc}") from exc


def download_once(url: str, dest: Any, *, timeout: float = 60.0,
                  logger: Optional[logging.Logger] = None) -> Path:
    """Return ``dest``, downloading it from ``url`` first if it is missing.

    Concurrent callers for the same ``dest`` download it once: the rest wait
    on the lock and then find the file. Raises ``DownloadError`` on failure,
    leaving no file at ``dest``.
    """
    log = logger or _logger
    p = Path(dest)
    if _usable(p):
        return p
    with path_lock(p):
        if _usable(p):          # another worker finished it while we waited
            return p
        log.info(f"Downloading {url} ...")
        fetch_to(url, p, timeout=timeout)
        log.info(f"Saved {p.stat().st_size / 1e6:.0f} MB to '{p}'.")
        return p
