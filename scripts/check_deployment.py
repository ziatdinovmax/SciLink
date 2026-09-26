#!/usr/bin/env python
"""Acceptance test for one hosted SciLink workspace.

    python scripts/check_deployment.py http://host:8422 --token T [--ops-token O]
        [--model bedrock/us.anthropic.claude-opus-4-8 --api-key K --region us-east-1]
        [--data examples/eels_identification_demo]

Without --model it checks the ops surface, auth and drain only; with it, it
creates an analysis session, uploads the demo data and runs one real turn,
then checks the usage ledger and attribution. Prints PASS/FAIL lines and
exits non-zero on any failure. Keys are read from arguments or the
environment and never written anywhere.
"""
import argparse, json, os, sys, time, urllib.error, urllib.request
from pathlib import Path

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(("PASS " if ok else "FAIL ") + name + (f" — {detail}" if detail else ""), flush=True)


def req(base, path, method="GET", body=None, token=None, raw=None, headers=None):
    h = dict(headers or {})
    if token:
        h["Authorization"] = f"Bearer {token}"
    data = None
    if body is not None:
        data = json.dumps(body).encode(); h["Content-Type"] = "application/json"
    if raw is not None:
        data, h["Content-Type"] = raw
    r = urllib.request.Request(f"{base}/api/v1{path}", data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"null")
        except Exception:
            return e.code, None


def multipart(fields, files):
    b = "----scilinkcheck"; out = b""
    for k, v in fields.items():
        out += f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    for name, path in files:
        out += (f"--{b}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"{name}\"\r\n"
                f"Content-Type: application/octet-stream\r\n\r\n").encode() + Path(path).read_bytes() + b"\r\n"
    out += f"--{b}--\r\n".encode()
    return out, f"multipart/form-data; boundary={b}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--token", default=os.environ.get("SCILINK_WEB_TOKEN", ""))
    ap.add_argument("--ops-token", default=os.environ.get("SCILINK_OPS_TOKEN", ""))
    ap.add_argument("--model"); ap.add_argument("--api-key", default=os.environ.get("SCILINK_CHECK_API_KEY", ""))
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "examples" / "eels_identification_demo"))
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--prompt", default=None,
                    help="The turn to run; {path} is the uploaded data path. Default: a generic one-pass analysis.")
    a = ap.parse_args()
    base, tok = a.base.rstrip("/"), a.token

    st, h = req(base, "/ops/health")
    check("health answers without a token", st == 200 and bool(h and h.get("ok")), str(h))
    st, _ = req(base, "/sessions")
    check("sessions need a token", st == 401, str(st))
    st, _ = req(base, "/sessions", token="wrong-" * 6)
    check("a wrong token is refused", st == 401, str(st))
    st, s = req(base, "/ops/status", token=tok)
    check("status answers a signed-in user", st == 200 and s.get("state") in ("idle", "busy", "draining"), str(s)[:120])
    if a.ops_token:
        st, s = req(base, "/ops/status", headers={"X-Ops-Token": a.ops_token})
        check("the ops token alone reaches status", st == 200, str(st))
        st, d = req(base, "/ops/drain", "POST", {"drain": True}, headers={"X-Ops-Token": a.ops_token})
        check("drain reported", st == 200 and d.get("state") == "draining", str(d)[:120])
        st, _ = req(base, "/sessions", "POST", {"mode": "analyze", "model": "m", "autonomy": "autonomous",
                                                 "api_key": "k", "consent": True}, token=tok)
        check("draining refuses a new session with 503", st == 503, str(st))
        req(base, "/ops/drain", "POST", {"drain": False}, headers={"X-Ops-Token": a.ops_token})
    st, u = req(base, "/usage", token=tok)
    check("usage answers", st == 200 and "calls" in (u or {}), str(u)[:120])

    if a.model:
        st, sess = req(base, "/sessions", "POST", {
            "mode": "analyze", "model": a.model, "autonomy": "autonomous", "consent": True,
            "api_key": a.api_key, "provider_fields": {"region": a.region}}, token=tok)
        check("analysis session created", st == 200 and bool(sess and sess.get("id")), str(sess)[:160])
        if st == 200:
            sid = sess["id"]; data = Path(a.data)
            files = [(p.name, p) for p in sorted(data.iterdir()) if p.suffix in (".npy", ".csv", ".txt")][:1]
            body, ctype = multipart({"category": "data"}, files)
            st, up = req(base, f"/sessions/{sid}/uploads", "POST", raw=(body, ctype), token=tok)
            check("data uploaded", st == 200, str(up)[:160])
            for p in sorted(data.glob("*.json")):
                body, ctype = multipart({"category": "metadata"}, [(p.name, p)])
                req(base, f"/sessions/{sid}/uploads", "POST", raw=(body, ctype), token=tok)
            path = next((p for p in (up or {}).get("paths", [])), "uploads/" + files[0][0])
            msg = (a.prompt.format(path=path) if a.prompt else
                   f"Analyze the data at {path} (metadata alongside it) in one pass and summarize the result.")
            st, _ = req(base, f"/sessions/{sid}/messages", "POST", {"content": msg}, token=tok)
            check("turn accepted", st == 202, str(st))
            t0 = time.time(); status = None
            while time.time() - t0 < a.timeout:
                st, snap = req(base, f"/sessions/{sid}", token=tok)
                status = (snap or {}).get("status")
                if status == "idle":
                    break
                time.sleep(10)
            check("turn finished", status == "idle", f"{status} after {int(time.time() - t0)}s")
            st, u = req(base, "/usage", token=tok)
            check("usage recorded and attributed", st == 200 and u["calls"] > 0 and sid in u.get("by_session", {}),
                  json.dumps({"calls": u.get("calls"), "sessions": list(u.get("by_session", {}))}))
    n_fail = results.count(False)
    print(f"\n{len(results) - n_fail} passed, {n_fail} failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
