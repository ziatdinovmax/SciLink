"""SSH login for HPC connections: keys, passwords and one-time codes (MFA).

Paramiko's built-in login tries a key, then a plain password. It cannot answer
the keyboard-interactive prompts that multi-factor sites use (NERSC asks for
"Password + OTP"), and when a key is accepted but a second factor is still
required it reads the code from the terminal with ``input()``, which hangs a
web server. :class:`MFAAuth` is a paramiko ``AuthStrategy`` that runs the whole
chain itself, answers each prompt from what the user supplied, and says what
is missing when the server wants more.

Secrets are held only for the length of one login and never logged.
"""
from __future__ import annotations

import logging
import os
from typing import Iterable, List, Optional, Sequence, Tuple

import paramiko
from paramiko.auth_strategy import AuthStrategy
from paramiko.ssh_exception import (
    AuthenticationException,
    BadAuthenticationType,
    SSHException,
)

logger = logging.getLogger(__name__)

# Words in a keyboard-interactive prompt that ask for a one-time code.
_CODE_WORDS = (
    "otp", "one-time", "one time", "passcode", "verification", "token",
    "authenticator", "duo", "mfa", "2fa", "second factor", "security code",
    "code",
)
_PASSWORD_WORDS = ("password", "passphrase")


class LoginRefused(AuthenticationException):
    """The server answered and refused (or SciLink stopped because a needed
    secret is missing): another address of the same host would refuse too."""


class _MissingSecret(Exception):
    """A prompt needs a secret we don't have: stop before answering, so the
    server does not record a failed login (sites lock accounts on those)."""
_DEFAULT_KEYS = ("id_ed25519", "id_ecdsa", "id_rsa")


def classify_prompt(prompt: str) -> str:
    """``"password+code"``, ``"code"``, ``"password"`` or ``"unknown"`` for
    one keyboard-interactive prompt."""
    p = (prompt or "").lower()
    wants_code = any(w in p for w in _CODE_WORDS)
    wants_password = any(w in p for w in _PASSWORD_WORDS)
    if wants_code and wants_password:
        return "password+code"
    if wants_code:
        return "code"
    if wants_password:
        return "password"
    return "unknown"


class MFAAuth(AuthStrategy):
    """Log in with a key and/or password, answering one-time-code prompts.

    ``use_keys`` tries ``key_path``, then the SSH agent and the default
    ``~/.ssh`` keys (when ``look_for_keys``). After that, or instead in
    password mode, it answers what the server still asks for: a password, a
    keyboard-interactive round (password, code, or both joined, as in NERSC's
    "Password + OTP:"), or a password followed by a code.
    """

    def __init__(self, username: str, *, password: str = "", otp: str = "",
                 use_keys: bool = False, key_path: str = "",
                 key_passphrase: str = "", allow_agent: bool = True,
                 look_for_keys: bool = True) -> None:
        super().__init__(ssh_config=paramiko.SSHConfig())
        self.username = username
        self._password = password or ""
        self._otp = "".join((otp or "").split())   # "123 456" -> "123456"
        self._use_keys = use_keys
        self._key_path = key_path or ""
        # paramiko's key loaders want the passphrase as bytes
        self._key_passphrase = key_passphrase.encode() if key_passphrase else None
        self._allow_agent = allow_agent
        self._look_for_keys = look_for_keys
        # What happened, for the error message (prompt texts, never answers).
        self.prompts_seen: List[str] = []
        self.code_sent = False
        self.code_missing = False
        self.password_missing = False
        self.code_asked_again = False
        self.key_accepted = False
        self.key_load_failed: List[str] = []
        # Set once a password or code has gone to the server: from then on a
        # failure must not be retried on another address (codes are single-use).
        self.secret_sent = False
        self._agent: Optional[paramiko.Agent] = None

    # ── keyboard-interactive ─────────────────────────────────

    def _handler(self, title: str, instructions: str,
                 prompts: Sequence[Tuple[str, bool]]) -> List[str]:
        answers = []
        for text, _echo in prompts:
            self.prompts_seen.append(text.strip())
            kind = classify_prompt(text)
            needs_code = kind in ("password+code", "code")
            needs_password = kind in ("password+code", "password")
            if needs_code and not self._otp:
                self.code_missing = True
                raise _MissingSecret()
            if needs_code and self.code_sent:
                # a second code request: a code is single-use, never resend it
                self.code_asked_again = True
                raise _MissingSecret()
            if needs_password and not self._password:
                self.password_missing = True
                raise _MissingSecret()
            if kind == "password+code":
                answer = self._password + self._otp
            elif kind == "code":
                answer = self._otp
            elif kind == "password":
                answer = self._password
            else:
                # A prompt we can't read: a code if that is all we have.
                answer = self._otp if (self._otp and not self._password) else self._password
            self.code_sent = self.code_sent or needs_code or (
                kind == "unknown" and answer == self._otp and bool(self._otp))
            self.secret_sent = self.secret_sent or bool(answer)
            answers.append(answer)
        return answers

    # ── keys ─────────────────────────────────────────────────

    def _load(self, path: str) -> Optional[paramiko.PKey]:
        try:
            return paramiko.PKey.from_path(path, passphrase=self._key_passphrase)
        except (SSHException, OSError, TypeError, ValueError) as e:
            # encrypted without (or with the wrong) passphrase, unreadable, or
            # an unsupported type: skip it and try the next key
            logger.info("Could not load SSH key %s: %s", path, type(e).__name__)
            self.key_load_failed.append(path)
            return None

    def _keys(self) -> Iterable[paramiko.PKey]:
        if self._key_path:
            key = self._load(os.path.expanduser(self._key_path))
            if key is not None:
                yield key
        if self._allow_agent:
            try:
                self._agent = paramiko.Agent()
                yield from self._agent.get_keys()
            except SSHException:
                pass
        if self._look_for_keys:
            for name in _DEFAULT_KEYS:
                path = os.path.expanduser(f"~/.ssh/{name}")
                if os.path.isfile(path):
                    key = self._load(path)
                    if key is not None:
                        yield key

    # ── the chain ────────────────────────────────────────────

    def _allowed(self, transport) -> List[str]:
        try:
            transport.auth_none(self.username)
            return []                       # no login needed at all
        except BadAuthenticationType as e:
            return list(e.allowed_types)

    def authenticate(self, transport):  # noqa: D102 - paramiko AuthStrategy hook
        try:
            return self._authenticate(transport)
        finally:
            if self._agent is not None:
                try:
                    self._agent.close()
                except Exception:
                    pass
                self._agent = None

    def _authenticate(self, transport):
        remaining: Optional[List[str]] = None
        if self._use_keys:
            for key in self._keys():
                try:
                    rest = transport.auth_publickey(self.username, key)
                except BadAuthenticationType as e:
                    remaining = list(e.allowed_types)    # no publickey here
                    break
                except (AuthenticationException, SSHException):
                    continue        # refused, or the agent could not sign (no touch)
                self.key_accepted = True
                if not rest:
                    return []
                remaining = list(rest)                   # key + a second factor
                break
        if transport.is_authenticated():
            return []
        allowed = remaining if remaining is not None else self._allowed(transport)
        if transport.is_authenticated():
            return []

        password_sent = interactive_rounds = 0
        for _ in range(3):  # at most: password, then a code round, then one more
            if transport.is_authenticated():
                return []
            kbd = "keyboard-interactive" in allowed and interactive_rounds < 2
            pwd = "password" in allowed and bool(self._password) and not password_sent
            # A code goes through keyboard-interactive; so does a password when
            # the server offers no plain password method.
            if kbd and ((self._otp and not self.code_sent) or not pwd):
                if not (self._password or self._otp):
                    break
                interactive_rounds += 1
                method = "keyboard-interactive"
            elif pwd:
                password_sent += 1
                method = "password"
            else:
                break
            try:
                if method == "password":
                    # fallback=False: on "password not allowed" paramiko would
                    # otherwise answer the keyboard-interactive prompt itself
                    # with the bare password (a failed login where the prompt
                    # is "Password + OTP:"). The loop picks the next method.
                    # Counted as sent once it goes out (a drop mid-call may
                    # still have delivered it); undone below only when the
                    # server answers that password logins are not allowed.
                    sent_before, self.secret_sent = self.secret_sent, True
                    rest = transport.auth_password(self.username, self._password,
                                                   fallback=False)
                else:
                    rest = transport.auth_interactive(self.username, self._handler)
            except BadAuthenticationType as e:
                # The method is not allowed here: no login of that kind
                # happened, so a failure later may still try another address.
                if method == "password":
                    self.secret_sent = sent_before
                allowed = list(e.allowed_types)
                continue
            except AuthenticationException:
                break                               # refused: wrong secret or expired code
            except Exception:
                # the handler stopped before answering (a missing secret), so
                # the transport was closed with no failed attempt on record
                if self.code_missing or self.password_missing or self.code_asked_again:
                    break
                raise
            if not rest:
                return []
            allowed = list(rest)
        if transport.is_authenticated():
            return []
        raise LoginRefused(self.explain(allowed))

    def explain(self, allowed: Sequence[str]) -> str:
        """Why the login failed, in terms the user can act on."""
        asked = f" (the server asked: {self.prompts_seen[-1]!r})" if self.prompts_seen else ""
        if self.code_missing:
            return ("The server asks for a one-time code" + asked + ". Enter the "
                    "current code from your authenticator app and connect again.")
        if self.code_asked_again:
            return ("The server asked for a second one-time code" + asked + ". A code "
                    "is single-use, so none was resent: connect again with a fresh code.")
        if self.code_sent:
            return ("Login failed after sending the one-time code" + asked + ". Codes "
                    "expire quickly and are single-use: try again with a fresh code, "
                    "and check the password.")
        if self.password_missing:
            return "The server asks for a password" + asked + "."
        if self.key_accepted and "keyboard-interactive" in allowed and not self._otp:
            return ("The SSH key was accepted, but the server asks for a second "
                    "factor: enter your one-time code and connect again.")
        if self._use_keys and not self.key_accepted and self.key_load_failed:
            return ("No SSH key was accepted; " + ", ".join(self.key_load_failed)
                    + " could not be loaded (a passphrase-protected key needs its "
                    "passphrase). The server allows: "
                    + (", ".join(allowed) or "nothing we can offer") + ".")
        if self._use_keys and not self.key_accepted:
            return ("No SSH key was accepted. The server allows: "
                    + (", ".join(allowed) or "nothing we can offer") + ".")
        return ("Authentication failed. The server allows: "
                + (", ".join(allowed) or "nothing we can offer") + asked + ".")
