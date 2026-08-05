"""Tests for the shared store I/O helpers (issue #398).

Covers:
  - atomic_write_text publishes the file and never leaves a temp file;
  - file_lock is reentrant-safe across sequential acquires and releases
    the lock file, and two concurrent processes cannot both hold it.

No LLM calls anywhere.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

from scilink.skills._shared._store_io import atomic_write_text, file_lock


def test_atomic_write_text_creates_parent_and_writes(tmp_path):
    target = tmp_path / "sub" / "out.json"
    atomic_write_text(target, '{"a": 1}')
    assert target.read_text() == '{"a": 1}'
    # No stray temp files left behind (directories are expected).
    leftovers = [p.name for p in tmp_path.rglob("*") if p.is_file() and p != target]
    assert leftovers == []


def test_atomic_write_text_overwrites_existing(tmp_path):
    target = tmp_path / "out.json"
    target.write_text("old")
    atomic_write_text(target, "new")
    assert target.read_text() == "new"


def test_file_lock_acquires_and_releases(tmp_path):
    lock_target = tmp_path / "data.json"
    with file_lock(lock_target):
        assert lock_target.with_name(lock_target.name + ".lock").exists()
    # Lock file remains (flock releases on close) but a second acquire works.
    with file_lock(lock_target):
        pass


def test_file_lock_serializes_processes(tmp_path):
    """Two processes cannot both hold the lock: the second blocks until
    the first releases. Proves the lock is cross-process, not just reentrant."""
    lock_target = tmp_path / "data.json"
    holder = (
        "import time\n"
        "from scilink.skills._shared._store_io import file_lock\n"
        f"p = {str(lock_target)!r}\n"
        "with file_lock(p):\n"
        "    print('HELD', flush=True)\n"
        "    time.sleep(3)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", holder], stdout=subprocess.PIPE, text=True
    )
    try:
        # Wait until the holder prints HELD (it holds the lock inside the with).
        assert proc.stdout.readline().strip() == "HELD"
        started = time.time()
        with file_lock(lock_target):
            elapsed = time.time() - started
        # We only got the lock after the holder released (3s sleep), so
        # elapsed must be >= ~2.5s. If the lock were a no-op, elapsed < 1s.
        assert elapsed >= 2.5, f"lock did not block: acquired in {elapsed:.2f}s"
    finally:
        proc.terminate()
        proc.wait(timeout=5)
