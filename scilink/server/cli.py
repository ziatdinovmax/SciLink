"""``scilink-web`` — run the web backend (and, when built, the React UI).

Local single-user posture: binds 127.0.0.1 with no auth by default; a
non-loopback ``--host`` gets a loud warning (anyone who can reach the port
can run code through the agents). Session directories are created under —
and file serving is fenced to — ``--session-root`` (default: cwd, matching
the Streamlit convention of launching from your data directory).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="scilink-web",
        description="SciLink web backend (REST + SSE) for the React UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8422)
    parser.add_argument("--session-root", default=".",
                        help="Directory holding session dirs (default: cwd).")
    parser.add_argument("--no-open", action="store_true",
                        help="Do not open the browser automatically.")
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

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("=" * 70, file=sys.stderr)
        print(f"WARNING: binding to {args.host} exposes the server beyond "
              "this machine.\nThere is NO authentication: anyone who can "
              "reach the port can execute\ncode on this machine through the "
              "agents. Use an SSH tunnel or a reverse\nproxy with auth "
              "instead of a raw public bind.", file=sys.stderr)
        print("=" * 70, file=sys.stderr)

    from .app import create_app
    app = create_app(session_root)
    url = f"http://127.0.0.1:{args.port}"
    print(f"SciLink web backend on http://{args.host}:{args.port} "
          f"(sessions in {session_root})")
    if not args.no_open:
        # Open the UI once the server is up (matches scilink-ui/Streamlit).
        # Loopback URL regardless of --host: the browser is on this machine.
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
