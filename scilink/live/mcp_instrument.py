"""An instrument behind an MCP server — onboarding without SciLink-specific code.

An instrument lab usually already has (or can write in any language) a small MCP
server in front of its controller: one tool that takes acquisition parameters
and returns a measurement. :class:`MCPInstrument` turns that tool into a live
loop's :class:`~scilink.live.instruments.Instrument`:

- the acquisition parameters and their limits are read from the tool's own
  ``inputSchema`` (``minimum`` / ``maximum`` / ``enum`` / ``default`` /
  ``description``, plus an optional ``x-units``). A numeric parameter with no
  declared limits is NOT steerable — the loop never invents safe limits — and is
  held at its default; ``limits=`` supplies them from the caller's side;
- ``acquire(params)`` calls the tool and reads the measurement from its JSON
  reply (see :func:`parse_measurement` for the accepted shapes);
- an optional description tool (``describe_instrument`` by default) supplies
  ``system_info`` / ``outputs`` / ``targets`` so the server can say what it
  measures; anything the caller passes wins.

What stays where: the DRIVER is the MCP server (vendor API, file formats, the
real limits); the KNOWLEDGE about steering this kind of measurement is an
acquisition skill per technique (``skills/acquisition/``), selected from
``system_info["technique"]``. Neither needs a SciLink Python class.

    inst = MCPInstrument.connect(command=["python", "my_raman_server.py"],
                                 tool="acquire_spectrum")
    loop = MeasurementLoop("run/loop", model_name=..., system_info=inst.system_info,
                           outputs=inst.outputs, schema=inst.schema)

``python -m scilink.live.mcp_demo_server insitu_raman`` is a reference server
(a simulator behind the protocol) to develop against.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .instruments import Frame, Instrument, read_curve
from .recommend import InstrumentSchema

DESCRIBE_TOOLS = ("describe_instrument", "describe")


def schema_from_tool(tool: Dict[str, Any], limits: Optional[Dict[str, Tuple[float, float]]] = None):
    """``(InstrumentSchema | None, defaults, held)`` from one tool's OpenAI-format
    schema (what :class:`~scilink.mcp_client.MCPConnection` returns).

    ``held`` lists the parameters that cannot be steered and why — typically a
    number with no declared limits. They are still SENT (at their default) when
    they have one."""
    fn = tool.get("function") or tool
    props = ((fn.get("parameters") or {}).get("properties") or {})
    spec: Dict[str, Dict[str, Any]] = {}
    defaults: Dict[str, Any] = {}
    held: List[str] = []
    for name, p in props.items():
        if not isinstance(p, dict):
            continue
        if "default" in p:
            defaults[name] = p["default"]
        kind = p.get("type")
        if isinstance(kind, list):                      # e.g. ["number", "null"]
            kind = next((k for k in kind if k != "null"), None)
        entry: Dict[str, Any] = {}
        if p.get("description"):
            entry["description"] = str(p["description"])
        units = p.get("x-units") or p.get("units")
        if units:
            entry["units"] = str(units)
        if p.get("enum"):
            spec[name] = {**entry, "kind": "choice", "choices": list(p["enum"])}
        elif kind == "boolean":
            spec[name] = {**entry, "kind": "bool"}
        elif kind in ("number", "integer"):
            lo, hi = (limits or {}).get(name, (p.get("minimum", p.get("exclusiveMinimum")),
                                               p.get("maximum", p.get("exclusiveMaximum"))))
            if lo is None or hi is None or not float(lo) < float(hi):
                held.append(f"{name}: no limits declared by the server (pass limits={{'{name}': "
                            "(low, high)}} to steer it)")
                continue
            spec[name] = {**entry, "kind": "int" if kind == "integer" else "float",
                          "low": float(lo), "high": float(hi)}
        else:
            held.append(f"{name}: type {kind!r} is not an acquisition parameter the loop can steer")
    schema = InstrumentSchema.from_dict(spec) if spec else None
    return schema, defaults, held


def _as_dict(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except ValueError:
            raise ValueError("the tool did not return JSON") from None
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object from the tool, got {type(payload).__name__}")
    # MCPConnection wraps a non-JSON text reply as {"status", "result": "<text>"}.
    inner = payload.get("result")
    if isinstance(inner, str) and set(payload) <= {"status", "result"}:
        try:
            return _as_dict(inner)
        except ValueError:
            pass
    return payload


def parse_measurement(payload: Any):
    """``(x, y, x_label, y_label, meta)`` from a tool reply. Accepted shapes:

    - ``{"x": [...], "y": [...]}``;
    - ``{"data": [[x, y], ...]}`` or ``{"data": {"x": [...], "y": [...]}}``;
    - ``{"path": "<two-column file>"}`` (also ``file``), for a controller that
      writes files — read with the same reader as a replayed folder;
    - otherwise, exactly two equal-length numeric arrays under any two keys, the
      first being x.

    Optional: ``x_label`` / ``y_label`` (or ``x_axis`` / ``y_axis``) and a
    ``meta`` (or ``metadata``) object. ``{"status": "error", ...}`` raises."""
    d = _as_dict(payload)
    if str(d.get("status") or "").lower() == "error":
        raise RuntimeError(f"the instrument reported an error: {d.get('message') or d.get('error')}")
    meta = d.get("meta") if isinstance(d.get("meta"), dict) else (
        d.get("metadata") if isinstance(d.get("metadata"), dict) else {})
    x_label = str(d.get("x_label") or d.get("x_axis") or "x")
    y_label = str(d.get("y_label") or d.get("y_axis") or "y")
    path = d.get("path") or d.get("file")
    if isinstance(path, str) and path:
        x, y, fx, fy = read_curve(str(Path(path).expanduser()))
        return x, y, (fx if x_label == "x" else x_label), (fy if y_label == "y" else y_label), meta
    data = d.get("data")
    if isinstance(data, dict):
        d = {**d, **data}
    elif isinstance(data, list) and data and isinstance(data[0], (list, tuple)):
        a = np.asarray(data, dtype=float)
        if a.ndim == 2 and a.shape[1] >= 2:
            return a[:, 0], a[:, 1], x_label, y_label, meta
    if "x" in d and "y" in d:
        x, y = np.asarray(d["x"], dtype=float), np.asarray(d["y"], dtype=float)
    else:
        arrays = [(k, v) for k, v in d.items()
                  if isinstance(v, list) and len(v) >= 3 and all(isinstance(i, (int, float)) for i in v[:5])]
        if len(arrays) != 2 or len(arrays[0][1]) != len(arrays[1][1]):
            raise ValueError("no measurement in the tool's reply: expected x and y arrays, "
                             "a `data` table, or a `path` to a two-column file")
        (kx, vx), (ky, vy) = arrays
        x, y = np.asarray(vx, dtype=float), np.asarray(vy, dtype=float)
        x_label, y_label = (kx if x_label == "x" else x_label), (ky if y_label == "y" else y_label)
    if x.shape != y.shape or x.ndim != 1 or x.size < 3:
        raise ValueError(f"x and y must be equal-length 1D arrays, got {x.shape} and {y.shape}")
    return x, y, x_label, y_label, meta


class MCPInstrument(Instrument):
    """A live loop's instrument whose ``acquire`` is one tool of an MCP server.
    See the module docstring."""

    def __init__(self, connection: Any, tool: str = "acquire", *,
                 system_info: Optional[Dict[str, Any]] = None,
                 outputs: Optional[Dict[str, str]] = None, targets: Optional[List[str]] = None,
                 limits: Optional[Dict[str, Tuple[float, float]]] = None,
                 fixed: Optional[Dict[str, Any]] = None, describe_tool: Optional[str] = None,
                 name: Optional[str] = None, owns_connection: bool = False) -> None:
        self.connection, self.tool = connection, tool
        self._owns = owns_connection
        schemas = {(t.get("function") or {}).get("name"): t
                   for t in (connection.tool_schemas or [])}
        if tool not in schemas:
            raise ValueError(f"the MCP server has no tool {tool!r}; it offers {sorted(k for k in schemas if k)}")
        self.schema, tool_defaults, self.held = schema_from_tool(schemas[tool], limits)
        self._properties = set((((schemas[tool].get("function") or {}).get("parameters") or {})
                                .get("properties") or {}))
        self.fixed = dict(fixed or {})
        self.defaults = {k: v for k, v in tool_defaults.items()
                         if self.schema is not None and self.schema.get(k) is not None}
        self._held_defaults = {k: v for k, v in tool_defaults.items() if k not in self.defaults}
        described: Dict[str, Any] = {}
        for candidate in ([describe_tool] if describe_tool else DESCRIBE_TOOLS):
            if candidate in schemas:
                try:
                    described = _as_dict(connection.call_tool(candidate, {}))
                except Exception:  # noqa: BLE001 - a description is optional
                    described = {}
                break
        self.name = name or str(described.get("name") or getattr(connection, "server_name", "mcp"))
        self.system_info = dict(system_info or described.get("system_info") or {})
        self.outputs = dict(outputs or described.get("outputs") or {})
        self.targets = list(targets or described.get("targets") or [])
        self.events = []

    @classmethod
    def connect(cls, *, command: Optional[List[str]] = None, url: Optional[str] = None,
                transport: Optional[str] = None, headers: Optional[Dict[str, str]] = None,
                env: Optional[Dict[str, str]] = None, server_name: str = "instrument",
                **kwargs: Any) -> "MCPInstrument":
        """Open the connection and build the instrument; ``close()`` ends both."""
        from ..mcp_client import MCPConnection
        conn = MCPConnection(server_name, command=command, url=url, env=env,
                             transport=transport, headers=headers)
        conn.connect()
        return cls(conn, owns_connection=True, **kwargs)

    def acquire(self, params: Dict[str, Any]) -> Frame:
        steer = self.check(params) if self.schema is not None else dict(params or {})
        args = {**self._held_defaults, **self.fixed, **steer}
        args = {k: v for k, v in args.items() if k in self._properties}
        x, y, x_label, y_label, meta = parse_measurement(self.connection.call_tool(self.tool, args))
        return Frame(x=x, y=y, params=steer, meta=dict(meta), x_label=x_label, y_label=y_label)

    def close(self) -> None:
        if self._owns:
            try:
                self.connection.disconnect()
            except Exception:  # noqa: BLE001
                pass
