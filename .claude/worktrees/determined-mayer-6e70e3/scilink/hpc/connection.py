"""Thread-safe SSH connection manager for HPC resources."""
from __future__ import annotations

import io
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import paramiko
import logging
import socket

@dataclass
class HPCProfile:
    """Storable connection profile (no secrets)."""
    name: str
    hostname: str
    username: str
    port: int = 22
    auth_method: str = "key"        # "key" | "password"
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
    ) -> None:
        """Open an SSH connection, trying all resolved IPs."""
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
        kw: dict = dict(
            hostname=self.profile.hostname,
            port=self.profile.port,
            username=self.profile.username,
            timeout=30,
            banner_timeout=30,
            auth_timeout=30,
        )
    
        if self.profile.auth_method == "key":
            if self.profile.key_path:
                kw["key_filename"] = self.profile.key_path
            if key_passphrase:
                kw["passphrase"] = key_passphrase
            kw["look_for_keys"] = True
            kw["allow_agent"] = True
        else:
            kw["password"] = password
            kw["look_for_keys"] = False
            kw["allow_agent"] = False
    
        if self.profile.proxy_jump:
            kw["sock"] = paramiko.ProxyCommand(
                f"ssh -W %h:%p {self.profile.proxy_jump}"
            )
            client.connect(**kw)
            self._client = client
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
    
                # Socket connected — hand to Paramiko
                kw["sock"] = sock
                kw["timeout"] = 30
                client.connect(**kw)
                self._client = client
                return
            except Exception as exc:
                last_err = exc
                try:
                    sock.close()
                except Exception:
                    pass
                # Reset client for next attempt
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                continue
    
        raise ConnectionError(
            f"All IPs for {self.profile.hostname} failed. "
            f"Tried: {unique_ips}. Last error: {last_err}"
        )

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
            rc = out.channel.recv_exit_status()
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

    def mkdir_p(self, path: str) -> None:
        """Remote equivalent of ``mkdir -p``."""
        self.run(f"mkdir -p {_q(path)}")

    def home_dir(self) -> str:
        out, _, _ = self.run("echo $HOME", timeout=10)
        return out.strip() or "~"


def _q(s: str) -> str:
    """Shell-quote a path."""
    return "'" + s.replace("'", "'\\''") + "'"
