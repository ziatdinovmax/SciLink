"""The meta's launch directory is explicit: the CLI leaves it as the process
cwd, a server passes the session root. The server's own cwd belongs to nobody,
so a ./kb_storage there must never be offered to a session as "your KB"."""
from pathlib import Path

import pytest


def _meta(tmp_path, **kw):
    from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent
    return MetaOrchestratorAgent(api_key="sk-dummy", base_dir=str(tmp_path / "s"), **kw)


def _kb(where: Path):
    where.mkdir(parents=True, exist_ok=True)
    (where / "index.faiss").write_bytes(b"x")


def test_cli_default_is_the_process_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _kb(tmp_path / "kb_storage")
    m = _meta(tmp_path)
    assert m.launch_dir == tmp_path.resolve()
    assert m._shared_kb_candidate == (tmp_path / "kb_storage").resolve()


def test_server_passes_the_session_root_and_ignores_its_own_cwd(tmp_path, monkeypatch):
    server_cwd = tmp_path / "server"; server_cwd.mkdir()
    monkeypatch.chdir(server_cwd)
    _kb(server_cwd / "kb_storage")                       # the operator's, not the user's
    root = tmp_path / "users" / "alice"
    m = _meta(tmp_path, launch_dir=str(root))
    assert m.launch_dir == root.resolve()
    assert m._shared_kb_candidate is None
    _kb(root / "kb_storage")
    m = _meta(tmp_path, launch_dir=str(root))
    assert m._shared_kb_candidate == (root / "kb_storage").resolve()


def test_web_session_manager_wires_the_session_root(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from scilink.server import session_manager as sm
    seen = {}

    class Fake:
        def __init__(self, **kw):
            seen.update(kw)
    monkeypatch.setattr("scilink.agents.meta_agent.meta_orchestrator.MetaOrchestratorAgent", Fake)
    sdir = tmp_path / "users" / "bob" / "meta_session_20260101_000000"
    sdir.mkdir(parents=True)
    sm._init_meta_agent(sdir, "k", "m", "", "autonomous", "")
    assert seen["launch_dir"] == str(sdir.parent)
