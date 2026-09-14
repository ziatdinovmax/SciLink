"""`scilink ui` launches the React web UI by default.

The default UI flipped from Streamlit to the React web app: `scilink ui`
(and `scilink-ui`) launch the FastAPI-served React bundle, `--streamlit`
forces the classic Streamlit app, and a source checkout without a built
bundle falls back to Streamlit with a note. These tests pin that dispatch
without launching a real server or Streamlit process.
"""
import tomllib
from pathlib import Path

import pytest

from scilink.cli import ui as ui_mod


@pytest.fixture
def spy(monkeypatch):
    """Record which launcher fired and with which argv, launching neither."""
    calls = {"streamlit": None, "web": None}

    def fake_streamlit(argv):
        calls["streamlit"] = list(argv)
        return 0

    monkeypatch.setattr(ui_mod, "_run_streamlit", fake_streamlit)

    # web_main is imported inside main() from scilink.server.cli.
    import scilink.server.cli as server_cli

    def fake_web(argv):
        calls["web"] = list(argv)
        return 0

    monkeypatch.setattr(server_cli, "main", fake_web)
    return calls


def _run(monkeypatch, argv):
    monkeypatch.setattr(ui_mod.sys, "argv", ["scilink ui", *argv])
    return ui_mod.main()


def test_react_by_default_when_bundle_present(monkeypatch, spy):
    monkeypatch.setattr(ui_mod, "_react_bundle_present", lambda: True)
    rc = _run(monkeypatch, ["--port", "9000"])
    assert rc == 0
    assert spy["web"] == ["--port", "9000"]
    assert spy["streamlit"] is None


def test_streamlit_flag_forces_streamlit(monkeypatch, spy):
    # Even with a bundle present, --streamlit wins, and the flag is stripped.
    monkeypatch.setattr(ui_mod, "_react_bundle_present", lambda: True)
    rc = _run(monkeypatch, ["--streamlit", "--server.port", "8501"])
    assert rc == 0
    assert spy["streamlit"] == ["--server.port", "8501"]
    assert spy["web"] is None


def test_falls_back_to_streamlit_without_bundle(monkeypatch, spy, capsys):
    monkeypatch.setattr(ui_mod, "_react_bundle_present", lambda: False)
    rc = _run(monkeypatch, [])
    assert rc == 0
    assert spy["streamlit"] == []
    assert spy["web"] is None
    assert "Streamlit" in capsys.readouterr().err


def test_web_deps_are_core_and_web_extra_empty():
    """The React backend deps moved into core; the [web] extra is a kept
    but empty back-compat alias."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    core = " ".join(pyproject["project"]["dependencies"])
    assert "fastapi" in core
    assert "uvicorn" in core
    assert "python-multipart" in core
    assert pyproject["project"]["optional-dependencies"]["web"] == []
