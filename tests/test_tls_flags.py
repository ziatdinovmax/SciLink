"""#611 — native TLS for the servers SciLink runs itself: the same three
flags on `scilink serve --transport sse` and `scilink-web`, validated
together, threaded into uvicorn, and reflected in the printed URLs."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scilink.server.tls import add_tls_arguments, tls_kwargs


def _ns(**kw):
    base = {"ssl_certfile": None, "ssl_keyfile": None, "ssl_keyfile_password": None}
    base.update(kw); return SimpleNamespace(**base)


def test_tls_kwargs_validation(tmp_path):
    cert = tmp_path / "c.pem"; key = tmp_path / "k.pem"; cert.write_text("c"); key.write_text("k")
    assert tls_kwargs(_ns()) == {}
    assert tls_kwargs(_ns(ssl_certfile=str(cert), ssl_keyfile=str(key))) == {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
    assert tls_kwargs(_ns(ssl_certfile=str(cert), ssl_keyfile=str(key), ssl_keyfile_password="pw"))["ssl_keyfile_password"] == "pw"
    with pytest.raises(SystemExit, match="together"):
        tls_kwargs(_ns(ssl_certfile=str(cert)))
    with pytest.raises(SystemExit, match="not found"):
        tls_kwargs(_ns(ssl_certfile=str(cert), ssl_keyfile=str(tmp_path / "missing.pem")))
    p = argparse.ArgumentParser(); add_tls_arguments(p, "x")
    a = p.parse_args(["--ssl-certfile", str(cert), "--ssl-keyfile", str(key)])
    assert tls_kwargs(a)["ssl_certfile"] == str(cert)


def test_run_sse_threads_tls_into_uvicorn(monkeypatch):
    import uvicorn
    from scilink import mcp_server
    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(kw))
    server = SimpleNamespace(run=None, create_initialization_options=lambda: None)
    mcp_server.run_sse(server, host="0.0.0.0", port=8443, ssl_certfile="c.pem", ssl_keyfile="k.pem")
    assert seen["ssl_certfile"] == "c.pem" and seen["ssl_keyfile"] == "k.pem" and seen["port"] == 8443
    seen.clear(); mcp_server.run_sse(server)
    assert "ssl_certfile" not in seen
    with pytest.raises(ValueError, match="together"):
        mcp_server.run_sse(server, ssl_certfile="c.pem")


def test_serve_print_mcp_json_reflects_https(tmp_path, monkeypatch, capsys):
    from scilink.cli import serve
    cert = tmp_path / "c.pem"; key = tmp_path / "k.pem"; cert.write_text("c"); key.write_text("k")
    monkeypatch.setattr(sys, "argv", ["scilink serve", "--print-mcp-json", "--transport", "sse", "--host", "0.0.0.0",
                                      "--port", "8443", "--ssl-certfile", str(cert), "--ssl-keyfile", str(key)])
    assert serve.main() == 0
    out = capsys.readouterr()
    assert json.loads(out.out)["mcpServers"]["scilink"]["url"] == "https://0.0.0.0:8443/sse"
    assert "--ssl-certfile" in out.err
    monkeypatch.setattr(sys, "argv", ["scilink serve", "--print-mcp-json", "--transport", "sse"])
    assert serve.main() == 0
    assert json.loads(capsys.readouterr().out)["mcpServers"]["scilink"]["url"] == "http://127.0.0.1:8000/sse"
    monkeypatch.setattr(sys, "argv", ["scilink serve", "--transport", "sse", "--ssl-certfile", str(cert)])
    with pytest.raises(SystemExit, match="together"):
        serve.main()


def test_web_cli_threads_tls_into_uvicorn_and_prints_https(tmp_path, monkeypatch, capsys):
    import uvicorn
    from scilink.server import cli
    cert = tmp_path / "c.pem"; key = tmp_path / "k.pem"; cert.write_text("c"); key.write_text("k")
    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(kw))
    rc = cli.main(["--port", "8499", "--session-root", str(tmp_path), "--no-open",
                   "--ssl-certfile", str(cert), "--ssl-keyfile", str(key)])
    assert rc == 0 and seen["ssl_certfile"] == str(cert) and seen["ssl_keyfile"] == str(key)
    assert "https://127.0.0.1:8499" in capsys.readouterr().out
    seen.clear()
    assert cli.main(["--port", "8499", "--session-root", str(tmp_path), "--no-open"]) == 0
    assert "ssl_certfile" not in seen and "http://127.0.0.1:8499" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="together"):
        cli.main(["--session-root", str(tmp_path), "--no-open", "--ssl-keyfile", str(key)])
