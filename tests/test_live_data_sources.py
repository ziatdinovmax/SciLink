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
