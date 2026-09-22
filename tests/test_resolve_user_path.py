"""Relative paths a user types in the terminal resolve against the working
directory when the session directory has no such file (the web UI's uploads
still win when both exist)."""

from pathlib import Path

from scilink.utils.file_io import resolve_user_path


def test_absolute_is_returned_as_is(tmp_path):
    assert resolve_user_path(str(tmp_path / "x.json"), tmp_path / "session") == tmp_path / "x.json"


def test_session_dir_wins_when_present(tmp_path, monkeypatch):
    session = tmp_path / "session"; (session / "data").mkdir(parents=True)
    (session / "data" / "a.json").write_text("{}")
    cwd = tmp_path / "cwd"; (cwd / "data").mkdir(parents=True)
    (cwd / "data" / "a.json").write_text("{}")
    monkeypatch.chdir(cwd)
    assert resolve_user_path("data/a.json", session) == session / "data" / "a.json"


def test_working_directory_fallback(tmp_path, monkeypatch):
    session = tmp_path / "session"; session.mkdir()
    cwd = tmp_path / "cwd"; (cwd / "examples").mkdir(parents=True)
    (cwd / "examples" / "image.json").write_text("{}")
    monkeypatch.chdir(cwd)
    assert resolve_user_path("examples/image.json", session) == cwd / "examples" / "image.json"


def test_missing_everywhere_names_the_session_location(tmp_path, monkeypatch):
    session = tmp_path / "session"; session.mkdir()
    monkeypatch.chdir(tmp_path)
    assert resolve_user_path("nope.txt", session) == session / "nope.txt"


def test_tilde_expands(tmp_path):
    assert resolve_user_path("~/x", tmp_path).is_absolute()
