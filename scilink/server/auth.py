"""Token authentication and per-user identity for the web backend.

Posture
-------
The default is unchanged: no auth, loopback bind, single implicit user —
the local tool. Auth turns on when the operator configures tokens, and the
CLI refuses a non-loopback bind without it (anyone who can reach the port
can run code through the agents).

Two shapes, both bearer tokens:

* ``--token T`` / ``SCILINK_WEB_TOKEN``: one shared token, one user
  (``default``) whose session root is ``--session-root`` itself — the
  "share my machine's UI over a tunnel or proxy" case.
* ``--users FILE``: a JSON object ``{"alice": "<token>", "bob": "<token>"}``.
  Each user gets an isolated session root ``<session-root>/users/<name>/``,
  their own live-session registry, and cannot see or attach to anyone
  else's sessions.

How a request authenticates
---------------------------
* ``Authorization: Bearer <token>`` on any request (scripts, curl), or
* the ``scilink_web`` cookie, minted by ``POST /api/v1/auth/login`` from a
  token — this is what the browser uses, because ``EventSource``, ``<img>``
  and download links cannot carry headers. Cookie sessions live in memory
  (a server restart logs everyone out) and are ``HttpOnly`` + ``SameSite=
  Lax``; ``Secure`` is set when the request arrived over HTTPS (a reverse
  proxy terminating TLS forwards ``X-Forwarded-Proto``).

The middleware guards ``/api/v1/*`` only; the SPA shell and its assets are
public so the login screen can load. ``/api/v1/auth/*`` is public by
construction.
"""

from __future__ import annotations

import hmac
import json
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

COOKIE_NAME = "scilink_web"
DEFAULT_USER = "default"
_PUBLIC_PREFIXES = ("/api/v1/auth/",)
_USERNAME_OK = set("abcdefghijklmnopqrstuvwxyz0123456789_-.")


class AuthConfigError(ValueError):
    """Bad token / users configuration (reported at startup, never at runtime)."""


@dataclass
class AuthConfig:
    """Token → user table plus the in-memory cookie sessions."""

    users: Dict[str, str]                       # user name -> token
    multi_user: bool = False                    # per-user session roots
    _cookies: Dict[str, str] = field(default_factory=dict)  # cookie id -> user

    # -- construction ---------------------------------------------------
    @classmethod
    def single(cls, token: str) -> "AuthConfig":
        token = (token or "").strip()
        if len(token) < 16:
            raise AuthConfigError(
                "The access token must be at least 16 characters; generate "
                "one with: python -c 'import secrets; print(secrets.token_urlsafe(32))'")
        return cls(users={DEFAULT_USER: token}, multi_user=False)

    @classmethod
    def from_users_file(cls, path: Path) -> "AuthConfig":
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, ValueError) as exc:
            raise AuthConfigError(f"Cannot read users file {path}: {exc}")
        if not isinstance(data, dict) or not data:
            raise AuthConfigError(
                f"{path} must be a non-empty JSON object of {{\"name\": \"token\"}}")
        users: Dict[str, str] = {}
        for name, token in data.items():
            n = str(name).strip()
            if not n or set(n.lower()) - _USERNAME_OK or n in (".", ".."):
                raise AuthConfigError(
                    f"Bad user name {name!r}: letters, digits, '_', '-', '.' only")
            t = str(token).strip()
            if len(t) < 16:
                raise AuthConfigError(f"Token for {n!r} must be at least 16 characters")
            if t in users.values():
                raise AuthConfigError(f"Token for {n!r} is not unique")
            users[n] = t
        return cls(users=users, multi_user=True)

    # -- checks ---------------------------------------------------------
    def user_for_token(self, token: str) -> Optional[str]:
        """Constant-time match of a presented token against every user."""
        presented = (token or "").encode()
        match = None
        for name, t in self.users.items():
            if hmac.compare_digest(presented, t.encode()):
                match = name       # keep scanning: no early exit on a hit
        return match

    def login(self, token: str) -> Optional[str]:
        """Mint a cookie session for the token's user; returns the cookie
        value, or None for an unknown token."""
        user = self.user_for_token(token)
        if user is None:
            return None
        cookie = secrets.token_urlsafe(32)
        self._cookies[cookie] = user
        return cookie

    def logout(self, cookie: Optional[str]) -> None:
        if cookie:
            self._cookies.pop(cookie, None)

    def user_for_cookie(self, cookie: Optional[str]) -> Optional[str]:
        return self._cookies.get(cookie or "")

    def user_for_request(self, request: Request) -> Optional[str]:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            return self.user_for_token(auth[7:].strip())
        return self.user_for_cookie(request.cookies.get(COOKIE_NAME))


def user_root(session_root: Path, auth: Optional[AuthConfig], user: str) -> Path:
    """Where a user's sessions live: the shared root unless per-user roots
    are on, then ``<root>/users/<name>`` (created on first use)."""
    if auth is None or not auth.multi_user:
        return session_root
    root = (session_root / "users" / user).resolve()
    if root.parent != (session_root / "users").resolve():
        raise AuthConfigError(f"User name escapes the users directory: {user!r}")
    root.mkdir(parents=True, exist_ok=True)
    return root


class AuthMiddleware:
    """Pure-ASGI guard for ``/api/v1/*``: resolves the user onto
    ``scope["state"]["user"]`` (``default`` when auth is off) and answers
    401 JSON for unauthenticated API calls. Non-API paths pass through."""

    def __init__(self, app: ASGIApp, auth: Optional[AuthConfig]) -> None:
        self.app = app
        self.auth = auth

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        state = scope.setdefault("state", {})
        if self.auth is None:
            state["user"] = DEFAULT_USER
            await self.app(scope, receive, send)
            return
        if not path.startswith("/api/v1/") or path.startswith(_PUBLIC_PREFIXES):
            state["user"] = None
            await self.app(scope, receive, send)
            return
        user = self.auth.user_for_request(Request(scope))
        if user is None:
            response = JSONResponse(
                {"detail": "Not authenticated. Sign in with your access token."},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"})
            await response(scope, receive, send)
            return
        state["user"] = user
        await self.app(scope, receive, send)
