"""An instrument behind an MCP server: the driver is the server, the limits are
the tool's own schema, and SciLink adds no code of its own per instrument."""

import json
import sys

import numpy as np
import pytest

from scilink.live.mcp_instrument import MCPInstrument, parse_measurement, schema_from_tool

TOOL = {"type": "function", "function": {"name": "acquire_spectrum", "parameters": {
    "type": "object", "properties": {
        "integration_s": {"type": "number", "minimum": 0.1, "maximum": 60, "default": 1.0,
                          "x-units": "s", "description": "detector integration time"},
        "accumulations": {"type": "integer", "minimum": 1, "maximum": 20, "default": 2},
        "grating": {"enum": ["600", "1800"], "default": "600"},
        "shutter_open": {"type": "boolean", "default": True},
        "stage_z_um": {"type": "number", "default": 12.5, "description": "no limits declared"},
        "sample_id": {"type": "string", "default": "A7"}}}}}


class FakeConnection:
    server_name = "raman-lab"

    def __init__(self, reply=None, tools=None):
        self.tool_schemas = tools or [TOOL]
        self.calls, self.reply = [], reply or {"x": [1, 2, 3, 4], "y": [5, 9, 6, 5]}

    def call_tool(self, name, args):
        self.calls.append((name, args))
        return json.dumps(self.reply if name == "acquire_spectrum" else {
            "name": "LabRAM", "system_info": {"technique": "Raman spectroscopy"},
            "outputs": {"g_position": "position of the G band"}, "targets": ["G band"]})


class TestSchemaFromTool:
    def test_limits_units_choices_and_defaults_come_from_the_tool(self):
        schema, defaults, held = schema_from_tool(TOOL)
        d = schema.to_dict()
        assert d["integration_s"] == {"kind": "float", "low": 0.1, "high": 60.0, "units": "s",
                                      "description": "detector integration time"}
        assert d["accumulations"]["kind"] == "int" and d["grating"]["choices"] == ["600", "1800"]
        assert d["shutter_open"]["kind"] == "bool"
        assert defaults["stage_z_um"] == 12.5

    def test_a_number_without_limits_is_held_never_steered(self):
        schema, _, held = schema_from_tool(TOOL)
        assert schema.get("stage_z_um") is None
        assert any(h.startswith("stage_z_um: no limits declared") for h in held)
        assert any(h.startswith("sample_id:") for h in held)
        steerable, _, held2 = schema_from_tool(TOOL, limits={"stage_z_um": (0, 50)})
        assert steerable.get("stage_z_um").high == 50 and not any("stage_z_um" in h for h in held2)


class TestParseMeasurement:
    def test_accepted_shapes(self, tmp_path):
        x, y, xl, yl, meta = parse_measurement({"x": [1, 2, 3], "y": [4, 5, 6], "x_label": "shift",
                                                "meta": {"T_K": 300}})
        assert list(y) == [4, 5, 6] and xl == "shift" and meta == {"T_K": 300}
        assert list(parse_measurement({"data": [[1, 4], [2, 5], [3, 6]]})[1]) == [4, 5, 6]
        assert list(parse_measurement({"data": {"x": [1, 2, 3], "y": [7, 8, 9]}})[1]) == [7, 8, 9]
        x, y, xl, yl, _ = parse_measurement({"two_theta": [10, 11, 12], "counts": [1, 9, 1]})
        assert (xl, yl) == ("two_theta", "counts")
        f = tmp_path / "scan.csv"
        f.write_text("bias_mV,dIdV\n-1,1\n0,0.1\n1,1\n")
        x, y, xl, yl, _ = parse_measurement(json.dumps({"path": str(f)}))
        assert list(x) == [-1, 0, 1] and xl == "bias_mV"
        wrapped = {"status": "success", "result": json.dumps({"x": [1, 2, 3], "y": [1, 2, 3]})}
        assert list(parse_measurement(wrapped)[0]) == [1, 2, 3]

    def test_what_is_refused(self):
        with pytest.raises(RuntimeError, match="shutter interlock"):
            parse_measurement({"status": "error", "message": "shutter interlock"})
        with pytest.raises(ValueError, match="no measurement"):
            parse_measurement({"ok": True})
        with pytest.raises(ValueError, match="equal-length"):
            parse_measurement({"x": [1, 2, 3], "y": [1, 2]})
        with pytest.raises(ValueError, match="did not return JSON"):
            parse_measurement("all good")


class TestMCPInstrument:
    def test_acquire_sends_steered_held_and_fixed_parameters(self):
        conn = FakeConnection()
        inst = MCPInstrument(conn, tool="acquire_spectrum", fixed={"sample_id": "B2"},
                             system_info={"technique": "Raman"})
        frame = inst.acquire({"integration_s": 5})
        name, args = conn.calls[-1]
        assert name == "acquire_spectrum"
        assert args == {"integration_s": 5, "accumulations": 2, "grating": "600", "shutter_open": True,
                        "stage_z_um": 12.5, "sample_id": "B2"}
        assert frame.params["integration_s"] == 5 and "stage_z_um" not in frame.params
        assert len(frame.x) == 4

    def test_limits_are_enforced_before_the_server_is_called(self):
        conn = FakeConnection()
        inst = MCPInstrument(conn, tool="acquire_spectrum")
        with pytest.raises(ValueError, match="outside"):
            inst.acquire({"integration_s": 600})
        assert conn.calls == []

    def test_the_server_may_describe_itself_and_the_caller_wins(self):
        describe = {"type": "function", "function": {"name": "describe_instrument", "parameters": {}}}
        inst = MCPInstrument(FakeConnection(tools=[TOOL, describe]), tool="acquire_spectrum")
        assert inst.name == "LabRAM" and inst.system_info == {"technique": "Raman spectroscopy"}
        assert inst.outputs == {"g_position": "position of the G band"} and inst.targets == ["G band"]
        mine = MCPInstrument(FakeConnection(tools=[TOOL, describe]), tool="acquire_spectrum",
                             system_info={"technique": "SERS"}, outputs={"x": "y"})
        assert mine.system_info == {"technique": "SERS"} and mine.outputs == {"x": "y"}

    def test_an_unknown_tool_is_named(self):
        with pytest.raises(ValueError, match="offers \\['acquire_spectrum'\\]"):
            MCPInstrument(FakeConnection(), tool="measure")


def test_pause_and_resume_map_onto_the_servers_tools():
    tool = lambda n: {"type": "function", "function": {"name": n, "parameters": {}}}   # noqa: E731
    conn = FakeConnection(tools=[TOOL, tool("pause"), tool("resume")])
    inst = MCPInstrument(conn, tool="acquire_spectrum")
    assert inst.can_pause
    inst.pause(); inst.resume()
    assert [c[0] for c in conn.calls[-2:]] == ["pause", "resume"]
    plain = MCPInstrument(FakeConnection(), tool="acquire_spectrum")
    assert plain.can_pause is False
    plain.pause()                                                # nothing to call: a no-op
    assert plain.connection.calls == []


def test_the_reference_server_end_to_end_over_stdio():
    pytest.importorskip("mcp")
    inst = MCPInstrument.connect(command=[sys.executable, "-m", "scilink.live.mcp_demo_server",
                                          "afm_force_curve"])
    try:
        assert inst.system_info["technique"].startswith("AFM") and "stiffness_N_per_m" in inst.outputs
        assert inst.id == "demo-afm_force_curve" and inst.can_pause
        inst.pause(); inst.resume()
        assert inst.schema.get("trigger_force_nN").high == 60 and inst.defaults["trigger_force_nN"] == 20
        a, b = inst.acquire({}), inst.acquire({"trigger_force_nN": 40})
        assert len(a.x) == len(a.y) > 50 and float(np.max(b.y)) > float(np.max(a.y))
        with pytest.raises(ValueError, match="outside"):
            inst.acquire({"trigger_force_nN": 5000})
    finally:
        inst.close()
