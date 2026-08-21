"""Onboarding + narration features: shared credentials file, foreground
narration streaming, --print-mcp-json."""
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("mcp")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

from scilink.mcp_server import (  # noqa: E402
    _load_shared_credentials, _to_thread_streaming, _MCP_SESSION,
)


def test_credentials_setdefault(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    d = tmp_path / ".scilink"; d.mkdir()
    (d / "credentials.env").write_text(
        "# comment\nAWS_BEARER_TOKEN_BEDROCK=abc123\n"
        'QUOTED_KEY="qv"\nALREADY_SET=from_file\nBADLINE\n=novalue\n')
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.setenv("ALREADY_SET", "from_env")
    loaded = _load_shared_credentials()
    assert set(loaded) == {"AWS_BEARER_TOKEN_BEDROCK", "QUOTED_KEY"}
    assert os.environ["AWS_BEARER_TOKEN_BEDROCK"] == "abc123"
    assert os.environ["QUOTED_KEY"] == "qv"
    assert os.environ["ALREADY_SET"] == "from_env"     # explicit env wins


def test_credentials_missing_file_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert _load_shared_credentials() == []


class _FakeSession:
    def __init__(self): self.msgs = []
    async def send_log_message(self, level, data, logger=None, **kw):
        self.msgs.append((level, data, logger))


def _worker(a, b, log_lines):
    for i in range(3):
        if log_lines is not None:
            log_lines.append(f"step {i}")
        time.sleep(0.4)
    return json.dumps({"status": "success", "sum": a + b})


def test_narration_streams_lines():
    sess = _FakeSession()
    tok = _MCP_SESSION.set(sess)
    try:
        out = asyncio.run(_to_thread_streaming(_worker, 1, 2))
    finally:
        _MCP_SESSION.reset(tok)
    assert json.loads(out)["sum"] == 3
    assert [m[1] for m in sess.msgs] == ["step 0", "step 1", "step 2"]
    assert all(m[0] == "info" and m[2] == "scilink" for m in sess.msgs)


def test_narration_disabled_and_no_session_are_silent(monkeypatch):
    monkeypatch.setenv("SCILINK_MCP_NARRATION", "0")
    out = asyncio.run(_to_thread_streaming(_worker, 2, 3))
    assert json.loads(out)["sum"] == 5
    monkeypatch.delenv("SCILINK_MCP_NARRATION")
    tok = _MCP_SESSION.set(None)
    try:
        out = asyncio.run(_to_thread_streaming(_worker, 3, 4))
    finally:
        _MCP_SESSION.reset(tok)
    assert json.loads(out)["sum"] == 7


class _DyingSession(_FakeSession):
    async def send_log_message(self, *a, **k):
        await super().send_log_message(*a, **k)
        if len(self.msgs) >= 1:
            raise RuntimeError("client gone")


def test_notification_failure_never_breaks_the_call():
    sess = _DyingSession()
    tok = _MCP_SESSION.set(sess)
    try:
        out = asyncio.run(_to_thread_streaming(_worker, 5, 6))
    finally:
        _MCP_SESSION.reset(tok)
    assert json.loads(out)["sum"] == 11
    assert len(sess.msgs) == 1          # stopped streaming after the failure


def test_print_mcp_json_forms():
    env = dict(os.environ)
    r = subprocess.run([sys.executable, "-m", "scilink.cli.main", "serve",
                        "--print-mcp-json", "--mode", "both",
                        "--model", "m", "--session-dir", "/tmp/x"],
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr[-400:]
    cfg = json.loads(r.stdout)
    e = cfg["mcpServers"]["scilink"]
    assert "env" not in e                      # secret-free by design
    assert "serve" in e["args"] and "--model" in e["args"]
    assert "credentials.env" in r.stderr
    r = subprocess.run([sys.executable, "-m", "scilink.cli.main", "serve",
                        "--print-mcp-json", "--transport", "sse", "--port", "8123"],
                       capture_output=True, text=True, env=env, timeout=120)
    e = json.loads(r.stdout)["mcpServers"]["scilink"]
    assert e == {"type": "sse", "url": "http://127.0.0.1:8123/sse"}
