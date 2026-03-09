"""MCP (Model Context Protocol) client for connecting to external tool servers.

Provides a thin synchronous wrapper around the async MCP Python SDK so that
the analysis orchestrator can discover and call tools exposed by any
MCP-compatible server over stdio or SSE transports.

Requires the ``mcp`` optional dependency::

    pip install scilink[mcp]
"""

import asyncio
import json
import logging
import threading
from typing import Any, Dict, List, Optional

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    HAS_MCP = True
except ImportError:
    HAS_MCP = False

try:
    from mcp.client.sse import sse_client
    HAS_SSE = True
except ImportError:
    HAS_SSE = False


def _require_mcp():
    if not HAS_MCP:
        raise ImportError(
            "MCP client support requires the 'mcp' package. "
            "Install with: pip install scilink[mcp]"
        )


class MCPConnection:
    """Manages a single connection to an MCP server.

    Parameters
    ----------
    server_name : str
        Human-readable label for this server.
    command : list[str] | None
        Command + arguments for stdio transport,
        e.g. ``["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]``.
    url : str | None
        URL for SSE transport, e.g. ``"http://localhost:8080/sse"``.
    env : dict | None
        Extra environment variables passed to the stdio subprocess.
    """

    def __init__(
        self,
        server_name: str,
        command: Optional[List[str]] = None,
        url: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        _require_mcp()
        if not command and not url:
            raise ValueError("Provide either 'command' (stdio) or 'url' (SSE).")

        self.server_name = server_name
        self.command = command
        self.url = url
        self.env = env

        self._tool_schemas: List[dict] = []
        self._connected = False

        # Dedicated event loop on a background thread so sync callers
        # never conflict with an outer loop (e.g. Streamlit).
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True
        )
        self._thread.start()

        # Populated by connect(); kept alive until disconnect().
        self._session: Optional[ClientSession] = None
        self._cleanup_coros: List = []

    # ── Public sync API ──────────────────────────────────────────────

    def connect(self) -> List[dict]:
        """Connect to the MCP server and return OpenAI-format tool schemas."""
        return self._run(self._connect())

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Invoke a tool on the MCP server and return the result as a JSON string."""
        return self._run(self._call_tool(tool_name, arguments))

    def disconnect(self) -> None:
        """Disconnect from the MCP server and stop the background loop."""
        if self._connected:
            try:
                self._run(self._disconnect())
            except Exception as exc:
                logging.warning(f"MCP disconnect error ({self.server_name}): {exc}")
        self._connected = False
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def tool_schemas(self) -> List[dict]:
        return list(self._tool_schemas)

    # ── Async internals ──────────────────────────────────────────────

    async def _connect(self):
        if self.command:
            params = StdioServerParameters(
                command=self.command[0],
                args=self.command[1:],
                env=self.env,
            )
            transport_ctx = stdio_client(params)
        elif self.url:
            if not HAS_SSE:
                raise ImportError(
                    "SSE transport requires 'mcp[sse]'. "
                    "Install with: pip install mcp[sse]"
                )
            transport_ctx = sse_client(self.url)
        else:
            raise ValueError("No transport configured.")

        # Enter the transport context manager manually so it stays open.
        read_stream, write_stream = await transport_ctx.__aenter__()
        self._cleanup_coros.append(transport_ctx.__aexit__)

        session_ctx = ClientSession(read_stream, write_stream)
        self._session = await session_ctx.__aenter__()
        self._cleanup_coros.append(session_ctx.__aexit__)

        await self._session.initialize()

        # List tools and convert schemas.
        tools_result = await self._session.list_tools()
        self._tool_schemas = [
            _mcp_to_openai_schema(t) for t in tools_result.tools
        ]
        self._connected = True

        logging.info(
            f"MCP connected to '{self.server_name}': "
            f"{len(self._tool_schemas)} tool(s)"
        )
        return self._tool_schemas

    async def _call_tool(self, tool_name: str, arguments: dict) -> str:
        if not self._session:
            return json.dumps({
                "status": "error",
                "message": f"MCP server '{self.server_name}' not connected.",
            })
        try:
            result = await self._session.call_tool(tool_name, arguments)
            # Serialize content blocks to a single string.
            parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
                else:
                    parts.append(str(block))
            combined = "\n".join(parts)
            # Return as-is if it looks like JSON, otherwise wrap.
            try:
                json.loads(combined)
                return combined
            except (json.JSONDecodeError, ValueError):
                return json.dumps({"status": "success", "result": combined})
        except Exception as exc:
            logging.error(f"MCP tool call error ({tool_name}): {exc}")
            return json.dumps({
                "status": "error",
                "message": str(exc),
                "tool": tool_name,
            })

    async def _disconnect(self):
        for cleanup in reversed(self._cleanup_coros):
            try:
                await cleanup(None, None, None)
            except Exception:
                pass
        self._cleanup_coros.clear()
        self._session = None

    def _run(self, coro):
        """Schedule a coroutine on the background loop and wait for the result."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=60)

    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass


def _mcp_to_openai_schema(mcp_tool) -> dict:
    """Convert an MCP tool object to OpenAI function-calling format."""
    input_schema = {}
    if hasattr(mcp_tool, "inputSchema") and mcp_tool.inputSchema:
        input_schema = mcp_tool.inputSchema
    elif isinstance(mcp_tool, dict):
        input_schema = mcp_tool.get("inputSchema", {})

    return {
        "type": "function",
        "function": {
            "name": getattr(mcp_tool, "name", "") or mcp_tool.get("name", ""),
            "description": getattr(mcp_tool, "description", "") or mcp_tool.get("description", ""),
            "parameters": input_schema,
        },
    }
