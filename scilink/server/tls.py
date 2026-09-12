"""Native TLS for the servers SciLink runs itself (#611).

Both ``scilink serve --transport sse`` and ``scilink-web`` bind an HTTP
port; a reverse proxy terminating TLS in front of it is the recommended
deployment (see docs), but when no proxy is available the same flags on
both commands hand a certificate and key straight to uvicorn.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


def add_tls_arguments(parser: argparse.ArgumentParser, what: str) -> None:
    g = parser.add_argument_group(
        "TLS", f"Serve {what} over HTTPS directly (when no reverse proxy terminates TLS).")
    g.add_argument("--ssl-certfile", default=None, metavar="PEM",
                   help="TLS certificate chain (PEM). Requires --ssl-keyfile.")
    g.add_argument("--ssl-keyfile", default=None, metavar="PEM",
                   help="TLS private key (PEM). Requires --ssl-certfile.")
    g.add_argument("--ssl-keyfile-password", default=None, metavar="PASSWORD",
                   help="Password of an encrypted private key (optional).")


def tls_kwargs(args: argparse.Namespace) -> dict:
    """The uvicorn keyword arguments for the parsed TLS flags — an empty
    dict for plain HTTP. Raises ``SystemExit(2)`` with a message when the
    flags are inconsistent or a file is missing."""
    cert, key = args.ssl_certfile, args.ssl_keyfile
    if not cert and not key:
        return {}
    if not (cert and key):
        raise SystemExit("--ssl-certfile and --ssl-keyfile must be given together.")
    for label, path in (("--ssl-certfile", cert), ("--ssl-keyfile", key)):
        if not Path(path).is_file():
            raise SystemExit(f"{label}: file not found: {path}")
    out = {"ssl_certfile": cert, "ssl_keyfile": key}
    if args.ssl_keyfile_password:
        out["ssl_keyfile_password"] = args.ssl_keyfile_password
    return out


def scheme(tls: Optional[dict]) -> str:
    return "https" if tls else "http"
