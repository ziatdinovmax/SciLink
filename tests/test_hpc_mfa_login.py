"""HPC login with one-time codes (MFA), against a real in-process SSH server.

Each fake server follows one site style: NERSC's single "Password + OTP:"
prompt, a key followed by a verification code, a password followed by a
code, and plain password / keyboard-interactive logins without MFA. No
network beyond localhost; no real cluster.
"""
import socket
import sys
import threading
from pathlib import Path

import paramiko
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scilink.hpc.auth import MFAAuth, classify_prompt  # noqa: E402
from scilink.hpc.connection import HPCConnection, HPCProfile  # noqa: E402

USER, PASSWORD, CODE = "alice", "s3cret", "123456"
_HOST_KEY = paramiko.RSAKey.generate(2048)
_CLIENT_KEY = paramiko.RSAKey.generate(2048)


class _Server(paramiko.ServerInterface):
    """``style`` picks the site behaviour."""

    def __init__(self, style, state=None):
        self.style = style
        self.stage = 0                         # factors completed so far
        self.state = state if state is not None else {}

    # what the server allows at each stage
    def get_allowed_auths(self, username):
        return {
            "nersc": "keyboard-interactive",
            "key+code": "publickey" if self.stage == 0 else "keyboard-interactive",
            "password+code": "password" if self.stage == 0 else "keyboard-interactive",
            "password": "password",
            "kbd-password": "keyboard-interactive",
            "drop": "keyboard-interactive",
            "two-codes": "keyboard-interactive",
            "eof-early": "password",
        }[self.style]

    def check_auth_none(self, username):
        if self.style == "eof-early":           # a half-dead node: drops before any secret
            self.state["transport"].sock.close()
        return paramiko.AUTH_FAILED

    def check_auth_password(self, username, password):
        if self.style not in ("password", "password+code") or password != PASSWORD:
            return paramiko.AUTH_FAILED
        if self.style == "password+code":
            self.stage = 1
            return paramiko.AUTH_PARTIALLY_SUCCESSFUL
        return paramiko.AUTH_SUCCESSFUL

    def check_auth_publickey(self, username, key):
        if self.style == "key+code" and key == _CLIENT_KEY:
            self.stage = 1
            return paramiko.AUTH_PARTIALLY_SUCCESSFUL
        return paramiko.AUTH_FAILED

    def check_auth_interactive(self, username, submethods):
        prompt = {"nersc": "Password + OTP: ", "kbd-password": "Password: "}.get(
            self.style, "Verification code: ")
        q = paramiko.InteractiveQuery("", "")
        q.add_prompt(prompt, False)
        return q

    def check_auth_interactive_response(self, responses):
        self.state.setdefault("responses", []).append(list(responses))
        if self.style == "drop":                  # the link dies after the code arrives
            self.state["transport"].sock.close()  # abruptly: no SSH disconnect message
            return paramiko.AUTH_FAILED
        if self.style == "two-codes":             # first code accepted, then asks again
            if len(self.state["responses"]) == 1:
                return paramiko.AUTH_PARTIALLY_SUCCESSFUL
            return paramiko.AUTH_SUCCESSFUL
        want = {"nersc": PASSWORD + CODE, "kbd-password": PASSWORD}.get(self.style, CODE)
        return paramiko.AUTH_SUCCESSFUL if list(responses) == [want] else paramiko.AUTH_FAILED

    def check_channel_request(self, kind, chanid):
        return paramiko.OPEN_SUCCEEDED


LAST_SERVER_STATE: dict = {}


@pytest.fixture
def server():
    """Start a one-connection SSH server; yield (port, set_style)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    state = LAST_SERVER_STATE            # the server thread and the tests share it
    state.clear()
    state["style"] = "password"
    transports = []

    def serve():
        try:
            conn, _ = sock.accept()
        except OSError:
            return
        t = paramiko.Transport(conn)
        t.add_server_key(_HOST_KEY)
        transports.append(t)
        state["transport"] = t
        try:
            t.start_server(server=_Server(state["style"], state))
        except Exception:
            pass

    th = threading.Thread(target=serve, daemon=True)
    th.start()
    yield sock.getsockname()[1], (lambda s: state.__setitem__("style", s))
    for t in transports:
        t.close()
    sock.close()


@pytest.fixture(autouse=True)
def no_real_keys(tmp_path, monkeypatch):
    """Keep the developer's own SSH agent and ~/.ssh keys out of the tests."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home" / ".ssh").mkdir(parents=True)
    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    monkeypatch.setattr(paramiko.Agent, "get_keys", lambda self: ())


def _conn(port, auth_method="password", key_path=""):
    return HPCConnection(HPCProfile(name="t", hostname="127.0.0.1", username=USER,
                                    port=port, auth_method=auth_method, key_path=key_path))


# ── prompt reading ───────────────────────────────────────────────────────

@pytest.mark.parametrize("prompt, kind", [
    ("Password + OTP: ", "password+code"),
    ("Verification code: ", "code"),
    ("Passcode or option (1-3): ", "code"),
    ("Enter your OTP: ", "code"),
    ("Password: ", "password"),
    ("Hello: ", "unknown"),
])
def test_prompts_are_read(prompt, kind):
    assert classify_prompt(prompt) == kind


# ── logins ───────────────────────────────────────────────────────────────

def test_nersc_style_password_plus_otp(server):
    port, style = server
    style("nersc")
    c = _conn(port)
    c.connect(password=PASSWORD, otp=CODE)
    assert c.is_connected
    c.disconnect()


def test_key_then_verification_code(server, tmp_path):
    port, style = server
    style("key+code")
    key = tmp_path / "id_test"
    _CLIENT_KEY.write_private_key_file(str(key))
    c = _conn(port, auth_method="key", key_path=str(key))
    c.connect(otp=CODE)
    assert c.is_connected
    c.disconnect()


def test_password_then_code(server):
    port, style = server
    style("password+code")
    c = _conn(port)
    c.connect(password=PASSWORD, otp=CODE)
    assert c.is_connected
    c.disconnect()


def test_plain_password_still_works(server):
    port, style = server
    style("password")
    c = _conn(port)
    c.connect(password=PASSWORD)
    assert c.is_connected
    c.disconnect()


def test_keyboard_interactive_password_without_mfa(server):
    port, style = server
    style("kbd-password")
    c = _conn(port)
    c.connect(password=PASSWORD)
    assert c.is_connected
    c.disconnect()


def test_a_missing_code_is_named(server):
    port, style = server
    style("nersc")
    with pytest.raises(Exception, match="asks for a one-time code"):
        _conn(port).connect(password=PASSWORD)


def test_a_wrong_code_suggests_a_fresh_one(server):
    port, style = server
    style("nersc")
    with pytest.raises(Exception, match="fresh code"):
        _conn(port).connect(password=PASSWORD, otp="000000")


def test_a_key_then_code_site_without_a_code_does_not_hang(server, tmp_path):
    # paramiko's own fallback would call input() here and block.
    port, style = server
    style("key+code")
    key = tmp_path / "id_test"
    _CLIENT_KEY.write_private_key_file(str(key))
    with pytest.raises(Exception, match="one-time code"):
        _conn(port, auth_method="key", key_path=str(key)).connect()


def test_secrets_never_appear_in_the_error(server):
    port, style = server
    style("nersc")
    with pytest.raises(Exception) as e:
        _conn(port).connect(password=PASSWORD, otp="999999")
    assert PASSWORD not in str(e.value) and "999999" not in str(e.value)


def test_the_strategy_records_prompts_not_answers():
    a = MFAAuth(USER, password=PASSWORD, otp=CODE)
    assert a._handler("", "", [("Password + OTP: ", False)]) == [PASSWORD + CODE]
    assert a.prompts_seen == ["Password + OTP:"] and a.code_sent


def _two_addresses(monkeypatch):
    """The host resolves to 127.0.0.1 (the server) and 192.0.2.1 (a TEST-NET
    address nobody answers); record every address a connection is tried on."""
    real = socket.getaddrinfo

    def gai(host, *a, **k):
        if host == "127.0.0.1":
            return real("127.0.0.1", *a, **k) + [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.1", a[0] if a else 22))]
        return real(host, *a, **k)
    monkeypatch.setattr(socket, "getaddrinfo", gai)
    attempts = []
    orig = socket.socket.connect
    monkeypatch.setattr(socket.socket, "connect",
                        lambda self, addr: (attempts.append(addr[0]), orig(self, addr))[1])
    return attempts


def test_a_refused_login_is_not_retried_on_other_addresses(server, monkeypatch):
    port, style = server
    style("nersc")
    attempts = _two_addresses(monkeypatch)
    with pytest.raises(paramiko.AuthenticationException, match="fresh code"):
        _conn(port).connect(password=PASSWORD, otp="000000")
    assert attempts == ["127.0.0.1"]


def test_a_code_is_not_resent_after_the_link_drops(server, monkeypatch):
    port, style = server
    style("drop")
    attempts = _two_addresses(monkeypatch)
    with pytest.raises(Exception, match="fresh"):   # the user is told the code is spent
        _conn(port).connect(password=PASSWORD, otp=CODE)
    assert attempts == ["127.0.0.1"]            # never sent again to 192.0.2.1


def test_a_drop_before_any_secret_still_tries_the_next_address(server, monkeypatch):
    port, style = server
    style("eof-early")
    attempts = _two_addresses(monkeypatch)
    with pytest.raises(Exception):
        _conn(port).connect(password=PASSWORD)
    assert attempts == ["127.0.0.1", "192.0.2.1"]


def test_a_second_code_request_gets_no_resent_code(server):
    port, style = server
    style("two-codes")
    with pytest.raises(paramiko.AuthenticationException, match="second one-time code"):
        _conn(port).connect(password=PASSWORD, otp=CODE)
    assert LAST_SERVER_STATE["responses"] == [[CODE]]     # sent once, never again


def test_the_jump_host_command(monkeypatch):
    from scilink.hpc import connection as cm
    assert cm._jump_command("alice@bastion", "perlmutter.nersc.gov", 22) == \
        "ssh -W perlmutter.nersc.gov:22 -- alice@bastion"
    assert cm._jump_command("alice@bastion:2222", "h", 2200) == "ssh -W h:2200 -p 2222 -- alice@bastion"
    seen = []

    class Stop(Exception):
        pass

    def fake_proxy(cmd):
        seen.append(cmd)
        raise Stop()
    monkeypatch.setattr(cm.paramiko, "ProxyCommand", fake_proxy)
    c = cm.HPCConnection(cm.HPCProfile(name="t", hostname="perlmutter.nersc.gov",
                                       username="u", proxy_jump="alice@bastion:2222"))
    with pytest.raises(Stop):
        c.connect(password="pw")
    assert seen == ["ssh -W perlmutter.nersc.gov:22 -p 2222 -- alice@bastion"]
    bad = cm.HPCConnection(cm.HPCProfile(name="t", hostname="x -oProxyCommand=touch /tmp/p",
                                         username="u", proxy_jump="alice@bastion"))
    with pytest.raises(ValueError):
        bad.connect(password="pw")



def test_a_missing_code_sends_no_failed_attempt(server):
    # The server would count a wrong "Password + OTP" toward an account lockout.
    port, style = server
    style("nersc")
    with pytest.raises(paramiko.AuthenticationException, match="one-time code"):
        _conn(port).connect(password=PASSWORD)
    assert LAST_SERVER_STATE.get("responses", []) == []


def _encrypted(path, key, passphrase):
    key.write_private_key_file(str(path), password=passphrase)


def test_an_encrypted_key_with_its_passphrase(server, tmp_path):
    port, style = server
    style("key+code")
    key = tmp_path / "id_enc"
    _encrypted(key, _CLIENT_KEY, "right")
    c = _conn(port, auth_method="key", key_path=str(key))
    c.connect(key_passphrase="right", otp=CODE)
    assert c.is_connected
    c.disconnect()


def test_an_unloadable_default_key_does_not_stop_the_next_one(server, tmp_path):
    port, style = server
    style("key+code")
    ssh = tmp_path / "home" / ".ssh"
    _encrypted(ssh / "id_ed25519", paramiko.RSAKey.generate(2048), "other")  # no passphrase given
    _CLIENT_KEY.write_private_key_file(str(ssh / "id_rsa"))
    c = _conn(port, auth_method="key")
    c.connect(otp=CODE)
    assert c.is_connected
    c.disconnect()


def test_a_code_with_spaces_is_joined():
    assert MFAAuth(USER, otp=" 123 456 ")._otp == "123456"


def test_a_jump_host_must_look_like_one():
    from scilink.hpc.connection import validate_proxy_jump
    assert validate_proxy_jump(" alice@bastion.nersc.gov:22 ") == "alice@bastion.nersc.gov:22"
    assert validate_proxy_jump("a+b@login_node") == "a+b@login_node"
    for bad in ("-oProxyCommand=touch /tmp/x x", "a b", "alice@host;rm -rf /", "alice@-oFoo"):
        with pytest.raises(ValueError):
            validate_proxy_jump(bad)
