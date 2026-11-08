import { useEffect, useState } from "react";
import { api, type HpcConnectBody, type HpcStatus } from "../api";

/** Connect the session's simulation specialist to an HPC cluster.
 *
 * Password logins (with an optional one-time code for multi-factor sites such
 * as NERSC) work everywhere. Key login and a jump host use the SSH keys of the
 * machine running SciLink, so they are offered only when it runs locally.
 * Secrets go to the server for one login and are cleared from the form; saved
 * profiles hold only host, user, port, method, key path and jump host. */

interface Profile {
  name: string;
  hostname: string;
  username: string;
  port: number;
  auth_method: "password" | "key";
  key_path: string;
  proxy_jump: string;
}

const PROFILES_KEY = "scilink.hpcProfiles";

function loadProfiles(): Profile[] {
  try {
    const raw = JSON.parse(localStorage.getItem(PROFILES_KEY) ?? "[]");
    return Array.isArray(raw) ? raw.filter((p) => p && typeof p.hostname === "string") : [];
  } catch {
    return [];
  }
}

function saveProfiles(ps: Profile[]): void {
  try {
    localStorage.setItem(PROFILES_KEY, JSON.stringify(ps));
  } catch {
    /* private mode */
  }
}

export function HpcPanel({ sessionId }: { sessionId: string }) {
  const [status, setStatus] = useState<HpcStatus>({ connected: false });
  const [localFiles, setLocalFiles] = useState(false);
  const [profiles, setProfiles] = useState<Profile[]>(loadProfiles);
  const [hostname, setHostname] = useState("");
  const [username, setUsername] = useState("");
  const [port, setPort] = useState(22);
  const [method, setMethod] = useState<"password" | "key">("password");
  const [password, setPassword] = useState("");
  const [keyPath, setKeyPath] = useState("");
  const [keyPass, setKeyPass] = useState("");
  const [otp, setOtp] = useState("");
  const [proxy, setProxy] = useState("");
  const [remember, setRemember] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // The server's own keys are offered only on a local server without
    // sign-in (a signed-in server is shared, even on loopback).
    api.authMe()
      .then((a) => setLocalFiles(a.local_files && !a.auth_required))
      .catch(() => setLocalFiles(false));
  }, []);

  // Status now, then every 30 s while connected (to notice a dropped link).
  useEffect(() => {
    let alive = true;
    const poll = () =>
      api.hpcStatus(sessionId).then((s) => alive && setStatus(s)).catch(() => {});
    poll();
    const t = window.setInterval(poll, 30_000);
    return () => {
      alive = false;
      window.clearInterval(t);
    };
  }, [sessionId]);

  const usePassword = method === "password" || !localFiles;

  function applyProfile(name: string) {
    const p = profiles.find((x) => x.name === name);
    if (!p) return;
    setHostname(p.hostname);
    setUsername(p.username);
    setPort(p.port || 22);
    setMethod(localFiles ? p.auth_method : "password");
    setKeyPath(p.key_path || "");
    setProxy(localFiles ? p.proxy_jump || "" : "");
  }

  async function connect() {
    setBusy(true);
    setError(null);
    const body: HpcConnectBody = {
      hostname: hostname.trim(),
      username: username.trim(),
      port,
      auth_method: usePassword ? "password" : "key",
      password: usePassword ? password : "",
      key_path: usePassword ? "" : keyPath.trim(),
      key_passphrase: usePassword ? "" : keyPass,
      otp: otp.trim(),
      proxy_jump: localFiles ? proxy.trim() : "",
    };
    try {
      setStatus(await api.hpcConnect(sessionId, body));
      if (remember) {
        const p: Profile = {
          name: body.hostname.split(".")[0] || body.hostname,
          hostname: body.hostname,
          username: body.username,
          port: body.port,
          auth_method: body.auth_method,
          key_path: body.key_path ?? "",
          proxy_jump: body.proxy_jump ?? "",
        };
        const next = [...profiles.filter((x) => x.name !== p.name), p];
        setProfiles(next);
        saveProfiles(next);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      // Secrets are single-use here: never kept in the form after an attempt.
      setPassword("");
      setKeyPass("");
      setOtp("");
      setBusy(false);
    }
  }

  async function disconnect() {
    setBusy(true);
    try {
      setStatus(await api.hpcDisconnect(sessionId));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  if (status.connected) {
    return (
      <div className="sidebar-section">
        <h3>HPC cluster</h3>
        <p className="caption">
          ● Connected to <b>{status.hostname}</b> as {status.username}
          {status.scheduler ? ` · ${status.scheduler}` : " · no scheduler detected"}
          {status.mfa ? " · MFA" : ""}
        </p>
        <p className="caption">The simulation specialist can now submit and track jobs here.</p>
        <button style={{ width: "100%" }} disabled={busy} onClick={() => void disconnect()}>
          Disconnect
        </button>
        {error && <p className="caption warn">{error}</p>}
      </div>
    );
  }

  return (
    <div className="sidebar-section">
      <details>
        <summary>
          <h3 style={{ display: "inline" }}>HPC cluster</h3>
        </summary>
        {status.dropped && (
          <p className="caption warn">
            The connection to {status.hostname} dropped. Reconnect below (a site with a
            one-time code needs a fresh one).
          </p>
        )}
        {profiles.length > 0 && (
          <label className="field">
            <span>Profile</span>
            <select defaultValue="" onChange={(e) => applyProfile(e.target.value)}>
              <option value="">New connection…</option>
              {profiles.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.name}
                </option>
              ))}
            </select>
          </label>
        )}
        <label className="field">
          <span>Host</span>
          <input value={hostname} placeholder="perlmutter.nersc.gov"
            onChange={(e) => setHostname(e.target.value)} />
        </label>
        <label className="field">
          <span>User</span>
          <input value={username} onChange={(e) => setUsername(e.target.value)} />
        </label>
        <label className="field">
          <span>Port</span>
          <input type="number" min={1} max={65535} value={port}
            onChange={(e) => setPort(Number(e.target.value) || 22)} />
        </label>
        <label className="field">
          <span>Login</span>
          <select value={usePassword ? "password" : "key"}
            onChange={(e) => setMethod(e.target.value as "password" | "key")}>
            <option value="password">Password</option>
            <option value="key" disabled={!localFiles}>
              SSH key{localFiles ? "" : " (only for SciLink on your own computer, no sign-in)"}
            </option>
          </select>
        </label>
        {usePassword ? (
          <label className="field">
            <span>Password</span>
            <input type="password" autoComplete="off" value={password}
              onChange={(e) => setPassword(e.target.value)} />
          </label>
        ) : (
          <>
            <label className="field">
              <span>Key path (blank: agent / default keys)</span>
              <input value={keyPath} placeholder="~/.ssh/nersc"
                onChange={(e) => setKeyPath(e.target.value)} />
            </label>
            <label className="field">
              <span>Key passphrase</span>
              <input type="password" autoComplete="off" value={keyPass}
                onChange={(e) => setKeyPass(e.target.value)} />
            </label>
          </>
        )}
        <label className="field">
          <span>One-time code (MFA, optional)</span>
          <input autoComplete="one-time-code" inputMode="numeric" value={otp}
            placeholder="from your authenticator app"
            onChange={(e) => setOtp(e.target.value)} />
        </label>
        {localFiles && (
          <label className="field">
            <span>Jump host (optional)</span>
            <input value={proxy} placeholder="user@bastion"
              onChange={(e) => setProxy(e.target.value)} />
          </label>
        )}
        <label className="field" style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <input type="checkbox" checked={remember} onChange={(e) => setRemember(e.target.checked)} />
          <span>Remember this profile (no passwords or codes)</span>
        </label>
        <button className="primary" style={{ width: "100%" }}
          disabled={busy || !hostname.trim() || !username.trim()}
          onClick={() => void connect()}>
          {busy ? "Connecting…" : "Connect"}
        </button>
        {error && <p className="caption warn">{error}</p>}
        <p className="caption">
          Codes are single-use: if the connection drops, reconnect with a fresh one.
        </p>
      </details>
    </div>
  );
}
