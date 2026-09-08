"""Tool inventory + MCP server management for the web UI's Tools tab.

Read-only over what the orchestrators already expose — ``tools.openai_schemas``
(the registered tool surface), ``_external_tools`` (tools registered from
outside, today MCP), ``_mcp_connections`` (name -> MCPConnection) — plus
thin wrappers over ``connect_mcp_server`` / ``disconnect_mcp_server``.
Every orchestrator (analysis, planning, meta) has the same three methods;
a session whose agent lacks them reports ``mcp_supported: False``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

_DESC_MAX = 600


def _clip(text: Any) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= _DESC_MAX else s[:_DESC_MAX - 1] + "…"


def tool_inventory(agent: Any) -> Dict[str, Any]:
    """``{builtin, external, mcp_servers, mcp_supported}`` for one agent."""
    external = [{"name": str(t.get("name", "")), "description": _clip(t.get("description"))}
                for t in (getattr(agent, "_external_tools", None) or [])
                if isinstance(t, dict) and t.get("name")]
    external_names = {t["name"] for t in external}

    builtin: List[Dict[str, str]] = []
    tools = getattr(agent, "tools", None)
    for td in (getattr(tools, "openai_schemas", None) or []):
        fn = td.get("function", {}) if isinstance(td, dict) else {}
        name = fn.get("name")
        if name and name not in external_names:
            builtin.append({"name": name, "description": _clip(fn.get("description"))})
    builtin.sort(key=lambda t: t["name"])

    servers: List[Dict[str, Any]] = []
    for name, conn in (getattr(agent, "_mcp_connections", None) or {}).items():
        schemas = getattr(conn, "tool_schemas", None) or []
        tool_names = []
        for s in schemas:
            fn = s.get("function", {}) if isinstance(s, dict) else {}
            if fn.get("name"):
                tool_names.append(fn["name"])
        transport = ("stdio" if getattr(conn, "command", None)
                     else getattr(conn, "transport", None) or "sse")
        servers.append({"name": str(name), "transport": transport,
                        "tools": tool_names})
    return {
        "builtin": builtin,
        "external": external,
        "mcp_servers": servers,
        "mcp_supported": callable(getattr(agent, "connect_mcp_server", None)),
    }


class MCPError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def connect_mcp(agent: Any, *, name: str, transport: str, command: str = "",
                url: str = "", headers: Optional[Dict[str, str]] = None,
                expand_env: bool = True) -> int:
    """Connect an MCP server through the agent's own method. Returns the
    number of tools registered. ``expand_env`` expands ``${VAR}`` in header
    values from the server's environment (so tokens stay out of the form);
    off on a shared server, where a remote user must not read its env."""
    if not callable(getattr(agent, "connect_mcp_server", None)):
        raise MCPError(400, "This session's agent does not support MCP servers.")
    name = (name or "").strip()
    if not name:
        raise MCPError(400, "Provide a server name.")
    if name in (getattr(agent, "_mcp_connections", None) or {}):
        raise MCPError(409, f"'{name}' is already connected.")
    if transport not in ("stdio", "sse", "http"):
        raise MCPError(400, f"Unknown transport {transport!r}.")
    hdrs = None
    if headers:
        hdrs = {str(k): (os.path.expandvars(str(v)) if expand_env else str(v))
                for k, v in headers.items()}
    try:
        if transport == "stdio":
            parts = (command or "").split()
            if not parts:
                raise MCPError(400, "Provide the command to run.")
            return int(agent.connect_mcp_server(name, command=parts))
        if not (url or "").strip():
            raise MCPError(400, "Provide the server URL.")
        return int(agent.connect_mcp_server(name, url=url.strip(),
                                            transport=transport, headers=hdrs))
    except MCPError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface the connection failure
        raise MCPError(400, f"Failed to connect: {exc}")


def disconnect_mcp(agent: Any, name: str) -> None:
    if not callable(getattr(agent, "disconnect_mcp_server", None)):
        raise MCPError(400, "This session's agent does not support MCP servers.")
    if name not in (getattr(agent, "_mcp_connections", None) or {}):
        raise MCPError(404, f"No MCP server named '{name}'.")
    try:
        agent.disconnect_mcp_server(name)
    except Exception as exc:  # noqa: BLE001
        raise MCPError(400, f"Failed to disconnect: {exc}")
