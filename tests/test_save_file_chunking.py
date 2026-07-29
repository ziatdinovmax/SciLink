"""Tests for #270 — reliable persistence of large content from planning mode.

Two halves:

1. ``PlanningOrchestratorAgent._parse_tool_args`` — malformed/truncated
   tool-call arguments must surface an actionable error the model can
   recover from, never a silent ``args = {}`` + raw TypeError.
2. ``append_file`` — the chunked-write companion to ``save_file`` so large
   files never ride in a single JSON tool-call argument.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scilink.agents.planning_agents.orchestrator_tools import OrchestratorTools
from scilink.agents.planning_agents.planning_orchestrator import (
    PlanningOrchestratorAgent,
)


def _tool_call(arguments: str):
    return SimpleNamespace(function=SimpleNamespace(name="save_file",
                                                    arguments=arguments))


# ---------------------------------------------------------------------------
# _parse_tool_args
# ---------------------------------------------------------------------------

def test_valid_args_parse_cleanly():
    args, err = PlanningOrchestratorAgent._parse_tool_args(
        _tool_call('{"filename": "a.md", "content": "hello"}'))
    assert err is None
    assert args == {"filename": "a.md", "content": "hello"}


def test_malformed_args_fail_loud_with_recovery_hint():
    """Broken escaping must NOT become args={}; the error must name the
    cause and point at the chunked-write path."""
    bad = '{"filename": "a.md", "content": "it\'s \n broken'
    args, err = PlanningOrchestratorAgent._parse_tool_args(_tool_call(bad))
    assert args is None
    payload = json.loads(err)
    assert payload["status"] == "error"
    assert "NOT executed" in payload["message"]
    assert "append_file" in payload["message"]
    assert "valid JSON" in payload["message"]


def test_truncated_args_report_truncation():
    """finish_reason == 'length' must be distinguished from bad escaping."""
    truncated = '{"filename": "a.md", "content": "very long protoc'
    args, err = PlanningOrchestratorAgent._parse_tool_args(
        _tool_call(truncated), finish_reason="length")
    assert args is None
    payload = json.loads(err)
    assert "truncated" in payload["message"]
    assert "append_file" in payload["message"]


# ---------------------------------------------------------------------------
# append_file
# ---------------------------------------------------------------------------

@pytest.fixture
def tools(tmp_path):
    return OrchestratorTools(SimpleNamespace(base_dir=tmp_path))


def test_append_file_registered(tools):
    assert "append_file" in tools.functions_map
    schema_names = [s["function"]["name"] for s in tools.openai_schemas]
    assert "append_file" in schema_names


def test_chunked_write_roundtrip(tools, tmp_path):
    """save_file + N append_file calls reassemble a large, escape-hostile
    artifact byte-for-byte."""
    chunks = [
        "# Protocol\n\nStep 1: mix 'A' with \"B\".\n",
        "```python\nfor i in range(10):\n    print({'k': i})\n```\n",
        "Final notes: 50 µL aliquots\n" * 50,
    ]
    r = json.loads(tools.execute_tool(
        "save_file", filename="protocol.md", content=chunks[0]))
    assert r["status"] == "success"
    for chunk in chunks[1:]:
        r = json.loads(tools.execute_tool(
            "append_file", filename="protocol.md", content=chunk))
        assert r["status"] == "success"
    assert (tmp_path / "protocol.md").read_text() == "".join(chunks)


def test_append_creates_missing_file(tools, tmp_path):
    r = json.loads(tools.execute_tool(
        "append_file", filename="fresh.txt", content="first"))
    assert r["status"] == "success"
    assert (tmp_path / "fresh.txt").read_text() == "first"


def test_append_respects_subfolder_and_sanitizes(tools, tmp_path):
    r = json.loads(tools.execute_tool(
        "append_file", filename="../../escape.txt", content="x",
        subfolder="protocols"))
    assert r["status"] == "success"
    # Path traversal stripped: lands inside the session dir, not outside.
    assert (tmp_path / "protocols" / "escape.txt").read_text() == "x"
    assert not (tmp_path.parent.parent / "escape.txt").exists()


def test_append_rejects_empty_filename(tools):
    r = json.loads(tools.execute_tool("append_file", filename="/", content="x"))
    assert r["status"] == "error"


# ---------------------------------------------------------------------------
# Loop integration: a bad tool call must produce a tool-role recovery
# message, not a TypeError from the tool.
# ---------------------------------------------------------------------------

def test_bad_args_never_reach_execute_tool(tmp_path):
    """Simulate what the chat loops now do with a malformed call: the
    recovery hint is returned and the tool function is never invoked."""
    tools = OrchestratorTools(SimpleNamespace(base_dir=tmp_path))
    args, err = PlanningOrchestratorAgent._parse_tool_args(
        _tool_call('{"filename": "p.md", "content": "trunc'))
    assert args is None and err is not None
    # The old behavior (args={}) produced this unrecoverable TypeError:
    old_result = tools.execute_tool("save_file")
    assert "missing" in json.loads(old_result)["message"]
    # The new behavior returns the actionable message instead.
    assert "append_file" in json.loads(err)["message"]
