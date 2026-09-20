"""A reference MCP server for an instrument: a simulator behind the protocol.

    python -m scilink.live.mcp_demo_server insitu_raman

Two tools, which is all :class:`~scilink.live.mcp_instrument.MCPInstrument`
needs: ``acquire`` (the acquisition parameters with their limits in the input
schema; replies ``{"x", "y", "x_label", "y_label", "meta"}``) and
``describe_instrument`` (what is measured and what is worth tracking). A real
instrument server has the same shape with the vendor API inside ``acquire``; it
can be written in any language.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, Dict, List

from .simulators import get_simulator


def tool_input_schema(sim: Any) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    for pname, spec in sim.schema.to_dict().items():
        p: Dict[str, Any] = {"description": spec.get("description", "")}
        if spec["kind"] == "choice":
            p["enum"] = list(spec["choices"])
        elif spec["kind"] == "bool":
            p["type"] = "boolean"
        else:
            p.update({"type": "integer" if spec["kind"] == "int" else "number",
                      "minimum": spec["low"], "maximum": spec["high"]})
        if spec.get("units"):
            p["x-units"] = spec["units"]
        if pname in sim.defaults:
            p["default"] = sim.defaults[pname]
        props[pname] = p
    return {"type": "object", "properties": props, "additionalProperties": False}


def main(argv: List[str] | None = None) -> int:
    from mcp import types
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    args = list(sys.argv[1:] if argv is None else argv)
    sim = get_simulator(args[0] if args else "insitu_raman", seed=int(args[1]) if len(args) > 1 else 0)
    server = Server(f"scilink-demo-{sim.name}")

    @server.list_tools()
    async def list_tools() -> List[types.Tool]:
        return [
            types.Tool(name="acquire", description=f"Acquire one {sim.system_info.get('technique')} measurement.",
                       inputSchema=tool_input_schema(sim)),
            types.Tool(name="describe_instrument", description="What this instrument measures.",
                       inputSchema={"type": "object", "properties": {}}),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
        if name == "describe_instrument":
            reply: Dict[str, Any] = {"name": sim.name, "system_info": sim.system_info,
                                     "outputs": sim.outputs, "targets": sim.targets}
        elif name == "acquire":
            try:
                frame = sim.acquire(dict(arguments or {}))
                reply = {"x": [float(v) for v in frame.x], "y": [float(v) for v in frame.y],
                         "x_label": frame.x_label, "y_label": frame.y_label, "meta": frame.meta}
            except Exception as e:  # noqa: BLE001 - the instrument says no
                reply = {"status": "error", "message": str(e)}
        else:
            reply = {"status": "error", "message": f"unknown tool {name!r}"}
        return [types.TextContent(type="text", text=json.dumps(reply))]

    async def run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    sys.exit(main())
