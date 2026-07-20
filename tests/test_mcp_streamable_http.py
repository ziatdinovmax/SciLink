"""Offline tests for MCPConnection's streamable HTTP transport.

Spins up a real in-process MCP server (FastMCP + uvicorn) behind a
bearer-token middleware, then connects through ``MCPConnection`` — so the
tests exercise the actual wire path, including that ``headers`` reach the
server on every request.
"""

import socket
import threading
import time

import pytest

pytest.importorskip("uvicorn")
pytest.importorskip("mcp.server.fastmcp")

from scilink.mcp_client import MCPConnection

TOKEN = "test-token-123"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def http_server_url():
    """A streamable-HTTP MCP server that rejects requests without the token."""
    import uvicorn
    from mcp.server.fastmcp import FastMCP
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    server = FastMCP("test-lab", stateless_http=True)

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    app = server.streamable_http_app()

    class RequireBearer(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if request.headers.get("authorization") != f"Bearer {TOKEN}":
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return await call_next(request)

    app.add_middleware(RequireBearer)

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    uv = uvicorn.Server(config)
    thread = threading.Thread(target=uv.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not uv.started:
        if time.time() > deadline:
            raise RuntimeError("uvicorn did not start within 15s")
        time.sleep(0.05)

    yield f"http://127.0.0.1:{port}/mcp"

    uv.should_exit = True
    thread.join(timeout=5)


def test_connect_list_and_call_with_headers(http_server_url):
    conn = MCPConnection(
        "lab",
        url=http_server_url,
        transport="http",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    try:
        schemas = conn.connect()
        names = {s["function"]["name"] for s in schemas}
        assert "add" in names
        assert conn.connected

        result = conn.call_tool("add", {"a": 2, "b": 3})
        assert "5" in result
    finally:
        conn.disconnect()


def test_connect_without_headers_is_rejected(http_server_url):
    conn = MCPConnection("lab-noauth", url=http_server_url, transport="http")
    try:
        with pytest.raises(Exception):
            conn.connect()
        assert not conn.connected
    finally:
        conn.disconnect()


def test_url_defaults_to_sse_transport():
    conn = MCPConnection("legacy", url="http://localhost:9/sse")
    assert conn.transport == "sse"
    conn.disconnect()


def test_unknown_transport_rejected():
    with pytest.raises(ValueError, match="Unknown transport"):
        MCPConnection("bad", url="http://localhost:9/mcp", transport="grpc")
