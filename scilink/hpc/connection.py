"""Thread-safe SSH connection manager for HPC resources."""
from __future__ import annotations

import io
import re
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import paramiko
import logging
import socket


def _login_strategy():
    """The MFA login classes, imported on first use: they need paramiko >= 3.2
    (``auth_strategy``), and an older paramiko should fail here with a message
    naming the version rather than when the package is imported."""
    try:
        from .auth import LoginRefused, MFAAuth
    except ImportError as exc:
        raise ImportError(
            f"SciLink's HPC login needs paramiko >= 3.2 (found "
            f"{getattr(paramiko, '__version__', 'unknown')}): "
            "pip install -U 'paramiko>=3.2'") from exc
    return LoginRefused, MFAAuth


# Host names and jump hosts reach a local `ssh` command (the jump-host path):
# nothing that ssh could read as an option or a second argument.
_HOST = r"[A-Za-z0-9_](?:[A-Za-z0-9_.-]*[A-Za-z0-9_])?"
_HOSTNAME = re.compile(rf"^{_HOST}$")
_PROXY_JUMP = re.compile(rf"^(?:([A-Za-z0-9_][A-Za-z0-9._+-]*)@)?({_HOST})(?::(\d{{1,5}}))?$")


def validate_hostname(value: str) -> str:
    """The host name, stripped, or ValueError when it isn't a plain name/address."""
    v = (value or "").strip()
    if not _HOSTNAME.match(v):
        raise ValueError(f"not a valid host name: {value!r}")
    return v


def validate_proxy_jump(value: str) -> str:
    """The jump host, stripped, or ValueError when it is not ``[user@]host[:port]``
    (anything else, e.g. a leading ``-``, would reach ssh as an option)."""
    v = (value or "").strip()
    if v and not _PROXY_JUMP.match(v):
        raise ValueError(f"jump host must look like user@host or user@host:port, got {value!r}")
    return v


def _jump_command(jump: str, hostname: str, port: int) -> str:
    """``ssh -W host:port [-p P] -- [user@]jump``: ssh reads ``host:port`` after
    ``--`` as a host name, so a jump-host port goes in ``-p``."""
    m = _PROXY_JUMP.match(jump)
    user, host, jport = m.group(1), m.group(2), m.group(3)
    target = f"{user}@{host}" if user else host
    port_opt = f"-p {int(jport)} " if jport else ""
    return f"ssh -W {hostname}:{int(port)} {port_opt}-- {target}"

@dataclass
class HPCProfile:
    """Storable connection profile (no secrets)."""
    name: str
    hostname: str
    username: str
    port: int = 22
    auth_method: str = "key"        # "key" | "password" (either may add a one-time code)
    key_path: str = ""
    proxy_jump: str = ""            # e.g. "user@bastion.example.edu"

class HPCConnection:
    """Wraps a single Paramiko SSHClient with SFTP helpers.

    Stored in ``st.session_state`` — persists across reruns within
    the same Streamlit browser session.  NOT picklable, so it cannot
    survive a server restart.
    """

    def __init__(self, profile: HPCProfile) -> None:
        self.profile = profile
        self._client: Optional[paramiko.SSHClient] = None
        self._sftp: Optional[paramiko.SFTPClient] = None
        self._lock = threading.RLock()

    # ── lifecycle ─────────────────────────────────────────────

    def connect(
        self,
        password: str = "",
        key_passphrase: str = "",
        otp: str = "",
    ) -> None:
        """Open an SSH connection, trying all resolved IPs.

        ``otp`` is a one-time code for sites with multi-factor login (e.g.
        NERSC's "Password + OTP" prompt, or a key followed by a code). It is
        used for this login only; a dropped connection needs a fresh code.
        """
        def _new_client() -> paramiko.SSHClient:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            return c

        client = _new_client()

        kw: dict = dict(
            hostname=self.profile.hostname,
            port=self.profile.port,
            username=self.profile.username,
            timeout=30,
            banner_timeout=30,
            auth_timeout=30,
        )

        # One login chain for every route: key and/or password, answering
        # keyboard-interactive prompts (paramiko's own fallback reads a
        # second factor from the terminal, which hangs a server).
        LoginRefused, MFAAuth = _login_strategy()

        def _auth():
            return MFAAuth(
                self.profile.username,
                password=password,
                otp=otp,
                use_keys=self.profile.auth_method == "key",
                key_path=self.profile.key_path,
                key_passphrase=key_passphrase,
            )
        kw["auth_strategy"] = _auth()

        if self.profile.proxy_jump:
            jump = validate_proxy_jump(self.profile.proxy_jump)
            host = validate_hostname(self.profile.hostname)
            # paramiko's ProxyCommand does not expand %h/%p: fill them in.
            kw["sock"] = paramiko.ProxyCommand(
                _jump_command(jump, host, self.profile.port))
            try:
                client.connect(**kw)
            except Exception:
                client.close()
                raise
            self._client = client
            self._enable_keepalive()
            return
    
        # Resolve all IPs and try each with a short per-IP timeout
        addrs = socket.getaddrinfo(
            self.profile.hostname, self.profile.port,
            family=socket.AF_INET,
            type=socket.SOCK_STREAM,
        )
        # Deduplicate while preserving order
        seen = set()
        unique_ips = []
        for info in addrs:
            ip = info[4][0]
            if ip not in seen:
                seen.add(ip)
                unique_ips.append(ip)
    
        if not unique_ips:
            raise ConnectionError(
                f"Cannot resolve {self.profile.hostname}"
            )
    
        per_ip_timeout = 5  # seconds — fail fast, try next
        last_err = None
    
        for ip in unique_ips:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(per_ip_timeout)
                sock.connect((ip, self.profile.port))
    
                # Socket connected — hand to Paramiko (a fresh login chain:
                # a failed attempt must not carry its prompt state over)
                kw["sock"] = sock
                kw["timeout"] = 30
                kw["auth_strategy"] = _auth()
                client.connect(**kw)
                self._client = client
                self._enable_keepalive()
                return
            except Exception as exc:
                client.close()
                try:
                    sock.close()
                except Exception:
                    pass
                # The host answered and refused the login, or a password or
                # code already went out before the link failed: another
                # address of the same host would refuse it too, and would
                # spend a single-use code again. Report this failure as is.
                if isinstance(exc, LoginRefused):
                    raise
                if kw["auth_strategy"].secret_sent:
                    raise ConnectionError(
                        f"The connection to {self.profile.hostname} dropped after the "
                        f"password/one-time code was sent ({exc}). A code is single-use: "
                        "connect again with a fresh one.") from exc
                last_err = exc
                client = _new_client()
                continue
    
        raise ConnectionError(
            f"All IPs for {self.profile.hostname} failed. "
            f"Tried: {unique_ips}. Last error: {last_err}"
        )

    def _enable_keepalive(self, interval: int = 30) -> None:
        """Send an SSH keepalive every `interval` seconds so idle transports
        don't get silently dropped by the server / firewall / NAT.

        Without this, a transport can sit idle long enough that the remote
        forgets the session; the next exec_command opens a channel that
        never receives an exit status, and recv_exit_status blocks forever.
        """
        if self._client is None:
            return
        transport = self._client.get_transport()
        if transport is not None:
            transport.set_keepalive(interval)

    def disconnect(self) -> None:
        with self._lock:
            for res in (self._sftp, self._client):
                if res is not None:
                    try:
                        res.close()
                    except Exception:
                        pass
            self._client = None
            self._sftp = None

    @property
    def is_connected(self) -> bool:
        if self._client is None:
            return False
        t = self._client.get_transport()
        return t is not None and t.is_active()

    # ── command execution ─────────────────────────────────────

    def run(
        self,
        cmd: str,
        timeout: float = 300,
    ) -> tuple[str, str, int]:
        """Execute *cmd*, return ``(stdout, stderr, exit_code)``."""
        if not self.is_connected:
            raise ConnectionError("SSH session is not active")
        with self._lock:
            _, out, err = self._client.exec_command(cmd, timeout=timeout)
            channel = out.channel
            # status_event.wait honors the timeout; recv_exit_status doesn't.
            if not channel.status_event.wait(timeout):
                try:
                    channel.close()
                except Exception:
                    pass
                raise TimeoutError(
                    f"Command timed out after {timeout}s: {cmd!r}"
                )
            rc = channel.recv_exit_status()
            return (
                out.read().decode(errors="replace"),
                err.read().decode(errors="replace"),
                rc,
            )

# ── SFTP helpers ──────────────────────────────────────────

    @property
    def sftp(self) -> paramiko.SFTPClient:
        with self._lock:
            if self._sftp is None or self._sftp.get_channel().closed:
                if self._client is None:
                    raise ConnectionError("Not connected")
                self._sftp = self._client.open_sftp()
            return self._sftp

    def listdir(self, path: str) -> list[paramiko.SFTPAttributes]:
        return self.sftp.listdir_attr(path)

    def read_text(self, path: str, tail: int = 0) -> str:
        """Read a remote text file.  *tail* > 0 → last N lines only."""
        if tail > 0:
            out, _, _ = self.run(f"tail -n {tail} {_q(path)}", timeout=15)
            return out
        with self.sftp.open(path, "r") as fh:
            return fh.read().decode(errors="replace")

    def read_bytes(self, path: str) -> bytes:
        """Read a remote file into memory as bytes."""
        buf = io.BytesIO()
        self.sftp.getfo(path, buf)
        buf.seek(0)
        return buf.read()

    def upload(
        self,
        local: str,
        remote: str,
        callback: Optional[Callable] = None,
    ) -> None:
        self.sftp.put(local, remote, callback=callback)

    def download(
        self,
        remote: str,
        local: str,
        callback: Optional[Callable] = None,
    ) -> None:
        self.sftp.get(remote, local, callback=callback)

    def stat(self, path: str) -> paramiko.SFTPAttributes:
        return self.sftp.stat(path)

    def remove(self, path: str) -> None:
        """Delete a remote file; a file that is not there is not an error."""
        try:
            self.sftp.remove(path)
        except FileNotFoundError:   # paramiko's missing-file error
            pass

    def mkdir_p(self, path: str) -> None:
        """Remote equivalent of ``mkdir -p``."""
        self.run(f"mkdir -p {_q(path)}")

    def home_dir(self) -> str:
        out, _, _ = self.run("echo $HOME", timeout=10)
        return out.strip() or "~"


def _q(s: str) -> str:
    """Shell-quote a path."""
    return "'" + s.replace("'", "'\\''") + "'"
