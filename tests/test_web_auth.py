"""Token auth + per-user isolation for the web backend (multi-user hardening).

Default posture is unchanged (no auth → one implicit user, sessions in the
root). With tokens: every /api/v1 call needs a bearer header or the login
cookie; per-user tokens give isolated session roots and registries; the
local-machine folder endpoints are refused on a remote deployment.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from scilink.server.app import create_app  # noqa: E402
from scilink.server.auth import COOKIE_NAME, AuthConfig, AuthConfigError  # noqa: E402
from scilink.server.session_manager import WebSession  # noqa: E402

TOK_A = "alice-token-0123456789abcdef"
TOK_B = "bob-token-0123456789abcdefgh"


def _users_file(tmp_path):
    f = tmp_path / "users.json"
    f.write_text(json.dumps({"alice": TOK_A, "bob": TOK_B}))
    return f


def _fake_session(app, user, sdir_name="analysis_session_20260101_090909"):
    mgr = app.state.manager_for_user(user)
    sdir = Path(mgr.session_root) / sdir_name
    sdir.mkdir(exist_ok=True)
    session = WebSession(id=sdir.name, session_dir=str(sdir), mode="analyze",
                         model="gpt-5.4", autonomy="autonomous",
                         agent=SimpleNamespace())
    mgr._sessions[sdir.name] = session
    return session


# ── config validation ────────────────────────────────────────────

def test_auth_config_validation(tmp_path):
    with pytest.raises(AuthConfigError):
        AuthConfig.single("short")
    single = AuthConfig.single(TOK_A)
    assert single.user_for_token(TOK_A) == "default" and not single.multi_user
    assert single.user_for_token("nope") is None
    f = tmp_path / "u.json"
    f.write_text(json.dumps({"alice": TOK_A, "bad/name": TOK_B}))
    with pytest.raises(AuthConfigError):
        AuthConfig.from_users_file(f)
    f.write_text(json.dumps({"alice": TOK_A, "bob": TOK_A}))   # duplicate token
    with pytest.raises(AuthConfigError):
        AuthConfig.from_users_file(f)
    f.write_text("[]")
    with pytest.raises(AuthConfigError):
        AuthConfig.from_users_file(f)
    multi = AuthConfig.from_users_file(_users_file(tmp_path))
    assert multi.multi_user and multi.user_for_token(TOK_B) == "bob"


# ── no auth: unchanged posture ────────────────────────────────────

def test_no_auth_is_open_and_single_root(tmp_path):
    client = TestClient(create_app(tmp_path, serve_frontend=False))
    me = client.get("/api/v1/auth/me").json()
    assert me == {"auth_required": False, "user": "default",
                  "multi_user": False, "local_files": True}
    cfg = client.get("/api/v1/config").json()
    assert cfg["auth"] == {"required": False, "user": "default", "multi_user": False}
    assert cfg["local_files"] is True
    assert Path(client.app.state.manager.session_root) == tmp_path.resolve()


# ── single shared token ───────────────────────────────────────────

def test_single_token_gate_bearer_and_cookie(tmp_path):
    app = create_app(tmp_path, serve_frontend=False, auth=AuthConfig.single(TOK_A))
    client = TestClient(app)
    # API is closed without a credential; the SPA shell path is not an API path
    r = client.get("/api/v1/config")
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    assert client.get("/api/v1/auth/me").json()["user"] is None
    # bearer header
    r = client.get("/api/v1/config", headers={"Authorization": f"Bearer {TOK_A}"})
    assert r.status_code == 200 and r.json()["auth"]["user"] == "default"
    # cookie flow
    assert client.post("/api/v1/auth/login", json={"token": "wrong"}).status_code == 401
    r = client.post("/api/v1/auth/login", json={"token": TOK_A})
    assert r.status_code == 200 and r.json()["user"] == "default"
    assert COOKIE_NAME in r.cookies
    assert client.get("/api/v1/sessions").status_code == 200   # cookie now set
    assert client.get("/api/v1/auth/me").json()["user"] == "default"
    # logout clears it
    client.post("/api/v1/auth/logout")
    assert client.get("/api/v1/sessions").status_code == 401
    # single token: sessions live in the root itself (no users/ subtree)
    assert Path(app.state.manager.session_root) == tmp_path.resolve()
    assert not (tmp_path / "users").exists()


# ── per-user tokens: isolation ────────────────────────────────────

def test_multi_user_isolation(tmp_path):
    auth = AuthConfig.from_users_file(_users_file(tmp_path))
    app = create_app(tmp_path, serve_frontend=False, auth=auth)
    assert app.state.manager is None            # nothing to single out
    alice = TestClient(app, headers={"Authorization": f"Bearer {TOK_A}"})
    bob = TestClient(app, headers={"Authorization": f"Bearer {TOK_B}"})
    # first touch creates each user's root
    assert alice.get("/api/v1/sessions").json() == {"live": [], "resumable": []}
    assert bob.get("/api/v1/sessions").json() == {"live": [], "resumable": []}
    assert (tmp_path / "users" / "alice").is_dir() and (tmp_path / "users" / "bob").is_dir()
    # a live session of alice's is invisible and unreachable to bob
    s = _fake_session(app, "alice")
    assert [x["id"] for x in alice.get("/api/v1/sessions").json()["live"]] == [s.id]
    assert bob.get("/api/v1/sessions").json()["live"] == []
    assert bob.get(f"/api/v1/sessions/{s.id}").status_code == 404
    assert bob.get(f"/api/v1/sessions/{s.id}/tree").status_code == 404
    assert bob.delete(f"/api/v1/sessions/{s.id}").status_code == 404
    assert alice.get(f"/api/v1/sessions/{s.id}").status_code == 200
    # resumable discovery is per root too
    (tmp_path / "users" / "bob" / "analysis_session_20260101_010101").mkdir()
    (tmp_path / "users" / "bob" / "analysis_session_20260101_010101" / "checkpoint.json").write_text("{}")
    assert [x["id"] for x in bob.get("/api/v1/sessions?mode=analyze").json()["resumable"]] == \
        ["analysis_session_20260101_010101"]
    assert alice.get("/api/v1/sessions?mode=analyze").json()["resumable"] == []
    # shared server: quit is refused; config names the user
    assert alice.post("/api/v1/quit").status_code == 403
    assert alice.get("/api/v1/config").json()["auth"] == {
        "required": True, "user": "alice", "multi_user": True}


# ── remote deployment: no server-machine folders ─────────────────

def test_local_files_gate(tmp_path):
    app = create_app(tmp_path, serve_frontend=False, local_files=False)
    client = TestClient(app)
    s = _fake_session(app, "default")
    assert client.get("/api/v1/config").json()["local_files"] is False
    r = client.post(f"/api/v1/sessions/{s.id}/folders", json={"paths": ["/etc"]})
    assert r.status_code == 403
    r = client.post(f"/api/v1/sessions/{s.id}/plan_dirs", json={"knowledge": "/etc"})
    assert r.status_code == 403
    # uploads still work remotely
    r = client.post(f"/api/v1/sessions/{s.id}/uploads", data={"category": "meta"},
                    files=[("files", ("a.csv", b"1"))])
    assert r.status_code == 200
