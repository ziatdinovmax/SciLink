"""The ops surface a control plane drives: health (open), status (busy or
idle, with reasons), drain (no new work), and the workspace manifest."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from scilink.server.app import create_app  # noqa: E402
from scilink.server.auth import AuthConfig  # noqa: E402
from scilink.server.runner import TurnState  # noqa: E402
from scilink.server.session_manager import WebSession  # noqa: E402

TOK = "t" * 32


def _fake_session(app, user="default", name="analysis_session_20260101_090909"):
    mgr = app.state.manager_for_user(user)
    sdir = Path(mgr.session_root) / name
    sdir.mkdir(exist_ok=True)
    s = WebSession(id=sdir.name, session_dir=str(sdir), mode="analyze", model="m",
                   autonomy="autonomous", agent=SimpleNamespace())
    mgr._sessions[sdir.name] = s
    return s


def test_health_is_open_and_names_the_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SCILINK_WORKSPACE", raising=False)
    (tmp_path / "workspace.json").write_text(json.dumps({"id": "ws-42", "name": "Perovskites"}))
    root = tmp_path / "sessions"; root.mkdir()
    app = create_app(root, serve_frontend=False, auth=AuthConfig.single(TOK))
    anon = TestClient(app)
    h = anon.get("/api/v1/ops/health")
    assert h.status_code == 200 and h.json()["ok"] and h.json()["workspace"] == "ws-42"
    assert "version" in h.json()
    # status and drain are not open
    assert anon.get("/api/v1/ops/status").status_code == 401
    assert anon.post("/api/v1/ops/drain", json={"drain": True}).status_code == 401
    user = TestClient(app, headers={"Authorization": f"Bearer {TOK}"})
    assert user.get("/api/v1/workspace").json()["workspace"]["name"] == "Perovskites"


def test_status_reports_busy_reasons_and_idle_time(tmp_path, monkeypatch):
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    app = create_app(tmp_path, serve_frontend=False)
    c = TestClient(app)
    st = c.get("/api/v1/ops/status").json()
    assert st["state"] == "idle" and st["busy"] == [] and st["sessions_live"] == 0
    s = _fake_session(app)
    s.turn = TurnState(is_running=True)
    st = c.get("/api/v1/ops/status").json()
    assert st["state"] == "busy" and st["busy"] == [f"turn:{s.id}"] and st["idle_for_s"] == 0
    s.turn.is_running = False
    st = c.get("/api/v1/ops/status").json()
    assert st["state"] == "idle" and st["busy"] == [] and st["sessions_live"] == 1
    # a live run and a memory job count too
    from scilink.server import live_api, memory_api
    live_api._RUNS[s.id] = SimpleNamespace(state="finishing")
    with memory_api._jobs_lock:
        memory_api._jobs["j1"] = {"id": "j1", "status": "running"}
    try:
        assert set(c.get("/api/v1/ops/status").json()["busy"]) == {f"live:{s.id}", "memory_job:j1"}
    finally:
        live_api._RUNS.pop(s.id, None)
        with memory_api._jobs_lock:
            memory_api._jobs.pop("j1", None)


def test_drain_refuses_new_work_and_reopens(tmp_path, monkeypatch):
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    app = create_app(tmp_path, serve_frontend=False)
    c = TestClient(app)
    s = _fake_session(app)
    st = c.post("/api/v1/ops/drain", json={"drain": True}).json()
    assert st["state"] == "draining" and st["draining"]
    r = c.post(f"/api/v1/sessions/{s.id}/messages", json={"content": "hi"})
    assert r.status_code == 503 and "draining" in r.json()["detail"]
    r = c.post("/api/v1/sessions", json={"mode": "analyze", "model": "m", "autonomy": "autonomous",
                                         "api_key": "k", "consent": True})
    assert r.status_code == 503
    assert c.post(f"/api/v1/sessions/{s.id}/live/start", json={}).status_code == 503
    # running work is untouched: stop still works, status still answers
    assert c.post(f"/api/v1/sessions/{s.id}/stop").status_code == 200
    st = c.post("/api/v1/ops/drain", json={"drain": False}).json()
    assert st["state"] == "idle"
    r = c.post(f"/api/v1/sessions/{s.id}/messages", json={"content": "hi"})
    assert r.status_code != 503


def test_shared_server_drain_needs_the_ops_token(tmp_path, monkeypatch):
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    users = tmp_path / "users.json"
    users.write_text(json.dumps({"alice": "a" * 32, "bob": "b" * 32}))
    app = create_app(tmp_path, serve_frontend=False, auth=AuthConfig.from_users_file(users))
    alice = TestClient(app, headers={"Authorization": "Bearer " + "a" * 32})
    assert alice.get("/api/v1/ops/status").status_code == 200
    assert alice.post("/api/v1/ops/drain", json={"drain": True}).status_code == 403
    monkeypatch.setenv("SCILINK_OPS_TOKEN", "ops-secret")
    ctl = TestClient(app, headers={"X-Ops-Token": "ops-secret"})
    assert ctl.get("/api/v1/ops/status").status_code == 200
    assert ctl.post("/api/v1/ops/drain", json={"drain": True}).json()["draining"] is True
    assert TestClient(app, headers={"X-Ops-Token": "wrong"}).get("/api/v1/ops/status").status_code == 401


def test_workspace_env_override_and_absence(tmp_path, monkeypatch):
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    named = tmp_path / "elsewhere.json"
    named.write_text(json.dumps({"id": "named"}))
    monkeypatch.setenv("SCILINK_WORKSPACE", str(named))
    app = create_app(tmp_path, serve_frontend=False)
    assert TestClient(app).get("/api/v1/ops/health").json()["workspace"] == "named"
    monkeypatch.delenv("SCILINK_WORKSPACE")
    app = create_app(tmp_path / "bare", serve_frontend=False)
    h = TestClient(app).get("/api/v1/ops/health").json()
    assert h["ok"] and "workspace" not in h
    assert TestClient(app).get("/api/v1/workspace").json() == {"workspace": None}
