"""Session discovery (shared with the web) and the terminal resume picker."""

import json

from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from contextlib import ExitStack
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from scilink.cli.shell.sessions import pick_session, print_sessions
from scilink.sessions import discover_resumable


def _mk(root, name, *, checkpoint=True, chat=False):
    d = root / name
    d.mkdir()
    if checkpoint:
        (d / "checkpoint.json").write_text(json.dumps(
            {"analysis_results": [1, 2], "current_data_path": "/x/grains.tif"}))
    if chat:
        (d / "chat_history.json").write_text(json.dumps(
            [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]))
    return d


def test_discover_lists_canonical_and_legacy_prefixes(tmp_path):
    _mk(tmp_path, "planning_session_20260102_000000")
    _mk(tmp_path, "campaign_session_20260103_000000", checkpoint=False, chat=True)
    _mk(tmp_path, "planning_session_20260101_000000", checkpoint=False)  # nothing to resume
    (tmp_path / "planning_session_notadir").write_text("x")
    found = discover_resumable(tmp_path, "plan")
    assert [s["id"] for s in found] == ["campaign_session_20260103_000000",
                                        "planning_session_20260102_000000"]
    assert found[0]["summary"] == {"message_count": 1}
    assert found[1]["summary"] == {}          # analyses count only in analyze mode
    assert found[1]["label"].startswith("2026-01-02 00:00:00")


def test_discover_excludes_live(tmp_path):
    _mk(tmp_path, "meta_session_20260102_000000")
    assert discover_resumable(tmp_path, "meta", exclude={"meta_session_20260102_000000"}) == []


def _pick(tmp_path, keys):
    import os; os.environ.setdefault("SCILINK_HOME", str(tmp_path / "home"))
    stack = ExitStack()
    pipe = stack.enter_context(create_pipe_input())
    pipe.send_text(keys)
    session = PromptSession(input=pipe, output=DummyOutput())
    console = Console(file=open("/dev/null", "w"), force_terminal=False)
    with stack:
        return pick_session(console, session, tmp_path, "meta")


def test_picker_enter_picks_newest_and_numbers_pick(tmp_path):
    _mk(tmp_path, "meta_session_20260102_000000")
    _mk(tmp_path, "meta_session_20260103_000000")
    assert _pick(tmp_path, "\r").endswith("meta_session_20260103_000000")   # a path now
    assert _pick(tmp_path, "2\r").endswith("meta_session_20260102_000000")
    assert _pick(tmp_path, "meta_session_20260102_000000\r").endswith("meta_session_20260102_000000")
    assert _pick(tmp_path, "9\r") is None


def test_picker_empty(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    console = Console(force_terminal=False)
    assert print_sessions(console, tmp_path, "analyze") == []
    assert "No Analyze sessions" in capsys.readouterr().out
