"""Trusted-header sign-in: an authenticating proxy in front names the user.

The header is honoured only on requests from a trusted proxy address, the
name is sanitized (it is a name, not a token), each user gets a root as
with --users, and nothing else signs anyone in.
"""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from scilink.server.app import create_app  # noqa: E402
from scilink.server.auth import AuthConfig, AuthConfigError  # noqa: E402

H = "X-Auth-Request-User"


def _app(tmp_path, proxies=()):
    return create_app(tmp_path, serve_frontend=False,
                      auth=AuthConfig.from_header(H, proxies))


def test_header_from_the_proxy_names_the_user_and_roots_are_per_user(tmp_path, monkeypatch):
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    app = _app(tmp_path)                                  # default: loopback only
    alice = TestClient(app, headers={H: "Alice"}, client=("127.0.0.1", 5000))
    me = alice.get("/api/v1/auth/me").json()
    assert (me["auth_required"], me["user"], me["multi_user"]) == (True, "alice", True)
    assert alice.get("/api/v1/sessions").json() == {"live": [], "resumable": []}
    assert (tmp_path / "users" / "alice").is_dir()
    bob = TestClient(app, headers={H: "bob"}, client=("127.0.0.1", 5001))
    assert bob.get("/api/v1/sessions").status_code == 200
    assert app.state.manager_for_user("alice").confined
    assert app.state.manager_for_user("bob").session_root != app.state.manager_for_user("alice").session_root


def test_header_from_anywhere_else_is_ignored(tmp_path):
    app = _app(tmp_path, ["10.0.0.0/8"])
    ok = TestClient(app, headers={H: "alice"}, client=("10.1.2.3", 5000))
    assert ok.get("/api/v1/auth/me").json()["user"] == "alice"
    spoof = TestClient(app, headers={H: "alice"}, client=("203.0.113.9", 5000))
    assert spoof.get("/api/v1/auth/me").json()["user"] is None
    assert spoof.get("/api/v1/sessions").status_code == 401
    # the bare test client host is not an address at all: not trusted either
    assert TestClient(app, headers={H: "alice"}).get("/api/v1/sessions").status_code == 401


def test_bad_names_tokens_and_cookies_do_not_sign_in(tmp_path):
    app = _app(tmp_path)
    for bad in ("", "..", "a/b", "x y", "../etc"):
        c = TestClient(app, headers={H: bad}, client=("127.0.0.1", 5000))
        assert c.get("/api/v1/sessions").status_code == 401, bad
    no_header = TestClient(app, headers={"Authorization": "Bearer " + "t" * 32},
                           client=("127.0.0.1", 5000))
    assert no_header.get("/api/v1/sessions").status_code == 401
    r = no_header.post("/api/v1/auth/login", json={"token": "t" * 32})
    assert r.status_code == 401 and "proxy" in r.json()["detail"]


def test_config_validation():
    with pytest.raises(AuthConfigError):
        AuthConfig.from_header("")
    with pytest.raises(AuthConfigError):
        AuthConfig.from_header("X-User", ["not-an-address"])
    cfg = AuthConfig.from_header("X-User", ["10.0.0.5", "192.168.0.0/16"])
    assert cfg.multi_user and cfg.users == {} and len(cfg.trusted_proxies) == 2
