"""``scilink-web`` — run the web backend (and, when built, the React UI).

Local single-user posture by default: binds 127.0.0.1 with no auth.
Sharing beyond this machine requires authentication — ``--token`` (one
shared token, one user) or ``--users FILE`` (per-user tokens, isolated
session roots under ``<session-root>/users/<name>/``) — and a non-loopback
``--host`` without either is refused (anyone who can reach the port can run
code through the agents). Put TLS in front with a reverse proxy; see
docs/react_web_ui.md "Sharing on a lab server". Session directories are
created under — and file serving is fenced to — ``--session-root``
(default: cwd, matching the Streamlit convention of launching from your
data directory).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _headless_matplotlib() -> None:
    """Force a non-interactive backend before any agent imports pyplot.

    The agents draw diagnostics on the server's worker threads; on macOS
    matplotlib's default GUI backend refuses that ("Cannot create a GUI
    FigureManager outside the main thread"), so a plan session's BO step
    failed at its plot unless something else had switched to Agg first.
    """
    import os
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib
        matplotlib.use("Agg")
    except Exception:  # noqa: BLE001 - matplotlib absent or already locked
        pass


def main(argv=None) -> int:
    _headless_matplotlib()
    parser = argparse.ArgumentParser(
        prog="scilink-web",
        description="SciLink web backend (REST + SSE) for the React UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8422)
    parser.add_argument("--session-root", default=".",
                        help="Directory holding session dirs (default: cwd).")
    parser.add_argument("--no-open", action="store_true",
                        help="Do not open the browser automatically.")
    parser.add_argument("--token", default=None,
                        help="Require this access token (or set "
                             "SCILINK_WEB_TOKEN). One shared user; sessions "
                             "stay in --session-root.")
    parser.add_argument("--users", default=None, metavar="FILE",
                        help='JSON {"name": "token", ...}: per-user tokens '
                             "with isolated session roots under "
                             "<session-root>/users/<name>/.")
    parser.add_argument("--insecure-no-auth", action="store_true",
                        help="Allow a non-loopback --host WITHOUT any auth "
                             "(only behind a reverse proxy that "
                             "authenticates for you).")
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        print("scilink-web requires the web extra: pip install 'scilink[web]'",
              file=sys.stderr)
        return 1

    session_root = Path(args.session_root).expanduser().resolve()
    if not session_root.is_dir():
        print(f"Session root does not exist: {session_root}", file=sys.stderr)
        return 1

    import os

    from .auth import AuthConfig, AuthConfigError
    loopback = args.host in ("127.0.0.1", "localhost", "::1")
    auth = None
    try:
        if args.users and (args.token or os.environ.get("SCILINK_WEB_TOKEN")):
            print("Use either --token / SCILINK_WEB_TOKEN or --users, not both.",
                  file=sys.stderr)
            return 2
        if args.users:
            auth = AuthConfig.from_users_file(Path(args.users).expanduser())
        elif args.token or os.environ.get("SCILINK_WEB_TOKEN"):
            auth = AuthConfig.single(args.token or os.environ["SCILINK_WEB_TOKEN"])
    except AuthConfigError as exc:
        print(f"scilink-web: {exc}", file=sys.stderr)
        return 2

    if not loopback and auth is None and not args.insecure_no_auth:
        print("=" * 70, file=sys.stderr)
        print(f"Refusing to bind {args.host} without authentication: anyone "
              "who can reach the port\ncould execute code on this machine "
              "through the agents.\n\nPass --token <secret> (or set "
              "SCILINK_WEB_TOKEN), or --users FILE for per-user tokens.\n"
              "Generate a token:  python -c 'import secrets; "
              "print(secrets.token_urlsafe(32))'\nOnly behind a reverse "
              "proxy that authenticates for you: --insecure-no-auth.",
              file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        return 2
    if not loopback and auth is None:
        print("WARNING: non-loopback bind with NO authentication "
              "(--insecure-no-auth). Make sure a reverse proxy in front "
              "authenticates every request.", file=sys.stderr)

    from .app import NO_BUNDLE_MESSAGE, create_app
    # Pasted local folder paths only make sense when the browser and the
    # server share a machine.
    app = create_app(session_root, auth=auth, local_files=loopback)
    url = f"http://127.0.0.1:{args.port}"
    print(f"SciLink web backend on http://{args.host}:{args.port} "
          f"(sessions in {session_root})")
    if auth is not None:
        who = (f"{len(auth.users)} users, per-user session roots"
               if auth.multi_user else "one shared token")
        print(f"Authentication ON ({who}). Sign in with the token, or open "
              f"{url}/?token=<your token> once.")
    if not getattr(app.state, "frontend_dir", None):
        print("=" * 70, file=sys.stderr)
        print(NO_BUNDLE_MESSAGE.rstrip(), file=sys.stderr)
        print("=" * 70, file=sys.stderr)
    if not args.no_open and getattr(app.state, "frontend_dir", None):
        # Open the UI once the server is up (matches scilink-ui/Streamlit).
        # Loopback URL regardless of --host: the browser is on this machine.
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
