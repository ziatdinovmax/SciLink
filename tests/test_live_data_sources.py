"""Tests for the three LiveDataSource implementations.

For each source we exercise the same contract:
  - First read returns the initial state.
  - Subsequent reads with no new data return None.
  - After an update (file touch, append, push), the next read surfaces it.
  - reset() returns the source to its just-constructed state.

Plus source-specific invariants (offset tracking for append-only,
ignoring the same payload across consumes for callback, etc.).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from scilink.agents.exp_agents.live_data_sources import (
    AppendOnlyFileSource,
    CallbackSource,
    DirectoryWatchSource,
    LatestData,
    LiveDataSource,
    MtimePollFileSource,
)


# --- Protocol conformance -----------------------------------------------------

def test_all_three_sources_satisfy_protocol(tmp_path):
    (tmp_path / "f").write_text("x")
    sources = [
        MtimePollFileSource(tmp_path / "f"),
        AppendOnlyFileSource(tmp_path / "f"),
        CallbackSource(),
    ]
    for s in sources:
        assert isinstance(s, LiveDataSource)
        assert isinstance(s.name, str) and s.name


# --- MtimePollFileSource ------------------------------------------------------

def test_mtime_first_read_returns_content(tmp_path):
    p = tmp_path / "scan.csv"
    p.write_text("two_theta,intensity\n20,100\n21,150\n")
    src = MtimePollFileSource(p)
    out = src.read_latest()
    assert out is not None
    assert out.source_kind == "mtime_poll"
    assert out.path == p
    assert out.text is not None
    assert "two_theta" in out.text
    assert "file_mtime" in out.extras


def test_mtime_no_change_returns_none(tmp_path):
    p = tmp_path / "scan.csv"
    p.write_text("x,y\n1,2\n")
    src = MtimePollFileSource(p)
    assert src.read_latest() is not None
    assert src.read_latest() is None  # no new mtime


def test_mtime_touch_surfaces_new_data(tmp_path):
    p = tmp_path / "scan.csv"
    p.write_text("x,y\n1,2\n")
    src = MtimePollFileSource(p)
    src.read_latest()  # prime
    time.sleep(0.02)
    p.write_text("x,y\n1,2\n3,4\n")  # rewrite advances mtime
    out = src.read_latest()
    assert out is not None
    assert "3,4" in (out.text or "")


def test_mtime_missing_file_returns_none(tmp_path):
    src = MtimePollFileSource(tmp_path / "not_here.csv")
    assert src.read_latest() is None
    # When file appears later, next read surfaces it
    (tmp_path / "not_here.csv").write_text("now exists")
    out = src.read_latest()
    assert out is not None


def test_mtime_reset_clears_state(tmp_path):
    p = tmp_path / "f"
    p.write_text("a")
    src = MtimePollFileSource(p)
    assert src.read_latest() is not None
    assert src.read_latest() is None
    src.reset()
    assert src.read_latest() is not None  # state cleared → emits again


# --- AppendOnlyFileSource -----------------------------------------------------

def test_append_first_read_returns_full_content(tmp_path):
    p = tmp_path / "scan.csv"
    p.write_text("header\nline1\nline2\n")
    src = AppendOnlyFileSource(p)
    out = src.read_latest()
    assert out is not None
    assert out.source_kind == "append_only"
    assert "header" in (out.text or "")
    assert out.extras["offset_start"] == 0
    assert out.extras["offset_end"] == len(p.read_bytes())


def test_append_no_growth_returns_none(tmp_path):
    p = tmp_path / "scan.csv"
    p.write_text("header\nline1\n")
    src = AppendOnlyFileSource(p)
    assert src.read_latest() is not None
    assert src.read_latest() is None


def test_append_only_returns_new_chunk(tmp_path):
    p = tmp_path / "scan.csv"
    p.write_text("header\nline1\n")
    src = AppendOnlyFileSource(p)
    src.read_latest()  # prime — full content consumed

    # Append a new line
    with p.open("a") as f:
        f.write("line2\n")

    out = src.read_latest()
    assert out is not None
    assert out.text == "line2\n"  # ONLY the new chunk
    assert out.extras["offset_start"] > 0
    assert out.extras["offset_end"] == len(p.read_bytes())


def test_append_offset_tracks_across_multiple_appends(tmp_path):
    p = tmp_path / "scan.csv"
    p.write_text("a")
    src = AppendOnlyFileSource(p)
    src.read_latest()  # consume "a"
    for chunk in ("b", "c", "d"):
        with p.open("a") as f:
            f.write(chunk)
        out = src.read_latest()
        assert out is not None and out.text == chunk


def test_append_truncated_file_returns_none(tmp_path):
    """If the file shrinks below the tracked offset, treat as 'no new data'
    rather than crashing or returning stale content."""
    p = tmp_path / "scan.csv"
    p.write_text("aaaaaaaaaa")
    src = AppendOnlyFileSource(p)
    src.read_latest()
    p.write_text("b")  # shorter than offset
    assert src.read_latest() is None


def test_append_handles_binary(tmp_path):
    p = tmp_path / "binary.bin"
    p.write_bytes(b"\x00\x01\x02\xff")
    src = AppendOnlyFileSource(p)
    out = src.read_latest()
    assert out is not None
    assert out.text is None  # non-UTF8 → text stays None
    assert out.extras["bytes"] == b"\x00\x01\x02\xff"


def test_append_reset_returns_full_content_again(tmp_path):
    p = tmp_path / "scan.csv"
    p.write_text("header\nline1\n")
    src = AppendOnlyFileSource(p)
    src.read_latest()
    assert src.read_latest() is None
    src.reset()
    out = src.read_latest()
    assert out is not None
    assert out.text == "header\nline1\n"


# --- CallbackSource -----------------------------------------------------------

def test_callback_read_before_push_returns_none():
    src = CallbackSource()
    assert src.read_latest() is None


def test_callback_push_then_read_returns_payload():
    src = CallbackSource()
    src.push(text="some-data", extras={"step": 1})
    out = src.read_latest()
    assert out is not None
    assert out.source_kind == "callback"
    assert out.text == "some-data"
    assert out.extras == {"step": 1}


def test_callback_consumed_payload_does_not_repeat():
    src = CallbackSource()
    src.push(text="x")
    src.read_latest()
    assert src.read_latest() is None


def test_callback_second_push_replaces_first():
    src = CallbackSource()
    src.push(text="old")
    src.push(text="new")
    out = src.read_latest()
    assert out is not None and out.text == "new"


def test_callback_reset_clears_buffer():
    src = CallbackSource()
    src.push(text="x")
    src.reset()
    assert src.read_latest() is None


# --- DirectoryWatchSource -----------------------------------------------------

def test_directory_watch_missing_dir_returns_none(tmp_path):
    src = DirectoryWatchSource(tmp_path / "no_such_dir")
    assert src.read_latest() is None


def test_directory_watch_empty_dir_returns_none(tmp_path):
    src = DirectoryWatchSource(tmp_path)
    assert src.read_latest() is None


def test_directory_watch_returns_newest_file_by_mtime(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    time.sleep(0.02)
    (tmp_path / "b.txt").write_text("beta")
    src = DirectoryWatchSource(tmp_path, pattern="*.txt")
    out = src.read_latest()
    assert out is not None
    assert out.path.name == "b.txt"
    assert out.text == "beta"
    assert out.extras["filename"] == "b.txt"
    assert out.extras["n_files_total"] == 2


def test_directory_watch_pattern_filters_files(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "b.csv").write_text("beta")
    src = DirectoryWatchSource(tmp_path, pattern="*.csv")
    out = src.read_latest()
    assert out is not None
    assert out.path.name == "b.csv"


def test_directory_watch_same_newest_returns_none_second_call(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    src = DirectoryWatchSource(tmp_path)
    assert src.read_latest() is not None
    assert src.read_latest() is None


def test_directory_watch_new_file_surfaced(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    src = DirectoryWatchSource(tmp_path)
    src.read_latest()  # prime on 'a'
    time.sleep(0.02)
    (tmp_path / "b.txt").write_text("beta")
    out = src.read_latest()
    assert out is not None and out.path.name == "b.txt"


def test_directory_watch_sort_by_name_lex_order(tmp_path):
    """When sort_by='name', alphabetical order picks the newest. Useful for
    zero-padded sequential filenames where mtime might be unreliable."""
    # Create out-of-order on disk
    (tmp_path / "scan_0003.txt").write_text("third")
    time.sleep(0.02)
    (tmp_path / "scan_0001.txt").write_text("first")
    time.sleep(0.02)
    (tmp_path / "scan_0002.txt").write_text("second")

    src = DirectoryWatchSource(tmp_path, pattern="scan_*.txt", sort_by="name")
    out = src.read_latest()
    assert out is not None
    assert out.path.name == "scan_0003.txt"  # alphabetically last
    assert out.text == "third"


def test_directory_watch_strategy_unseen_walks_each_file_once(tmp_path):
    """In 'unseen' strategy, three files in the directory produce three
    distinct ticks (in sort order), then None thereafter."""
    (tmp_path / "a.txt").write_text("alpha")
    time.sleep(0.01)
    (tmp_path / "b.txt").write_text("beta")
    time.sleep(0.01)
    (tmp_path / "c.txt").write_text("gamma")
    src = DirectoryWatchSource(tmp_path, strategy="unseen", sort_by="mtime")
    out1 = src.read_latest(); out2 = src.read_latest(); out3 = src.read_latest()
    out4 = src.read_latest()
    assert [o.path.name for o in (out1, out2, out3)] == ["a.txt", "b.txt", "c.txt"]
    assert out4 is None


def test_directory_watch_reset_re_surfaces_files(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    src = DirectoryWatchSource(tmp_path)
    assert src.read_latest() is not None
    assert src.read_latest() is None
    src.reset()
    out = src.read_latest()
    assert out is not None
    assert out.path.name == "a.txt"


def test_directory_watch_validates_strategy():
    with pytest.raises(ValueError, match="strategy"):
        DirectoryWatchSource("/tmp", strategy="random")


def test_directory_watch_validates_sort_by():
    with pytest.raises(ValueError, match="sort_by"):
        DirectoryWatchSource("/tmp", sort_by="size")


def test_directory_watch_protocol_conformance(tmp_path):
    src = DirectoryWatchSource(tmp_path)
    assert isinstance(src, LiveDataSource)
    assert src.name == "directory_watch"
