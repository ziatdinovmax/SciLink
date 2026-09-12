"""#606 — tool schemas are normalized once, in the provider wrappers, into
the subset every provider accepts; and every registered tool of every
orchestrator passes the Gemini declaration rules after normalization, so a
non-portable schema fails here rather than at request time."""
import copy
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from scilink.wrappers.tool_schema import (
    gemini_schema_problems, normalize_schema, normalize_tools, tool_schema_problems)


def test_multi_type_with_shared_items_becomes_scoped_anyof():
    s = {"type": ["string", "array", "object"], "items": {"type": "string"},
         "properties": {"k": {"type": "number"}}, "description": "d"}
    out = normalize_schema(s)
    assert out == {"description": "d", "anyOf": [
        {"type": "string"},
        {"type": "array", "items": {"type": "string"}},
        {"type": "object", "properties": {"k": {"type": "number"}}}]}
    assert s["type"] == ["string", "array", "object"]            # input untouched
    assert gemini_schema_problems(out) == []
    assert any("'items' on a 'object'" in p for p in gemini_schema_problems(
        {"anyOf": [{"type": "object", "items": {"type": "string"}}]}))


def test_null_single_and_enum_type_lists():
    assert normalize_schema({"type": ["string", "null"]}) == {"anyOf": [{"type": "string"}, {"type": "null"}]}
    assert normalize_schema({"type": ["string"], "enum": ["a"]}) == {"type": "string", "enum": ["a"]}
    out = normalize_schema({"type": ["string", "integer"], "enum": ["a", 1]})
    assert out == {"anyOf": [{"type": "string", "enum": ["a"]}, {"type": "integer", "enum": [1]}]}


def test_oneof_becomes_anyof_recursively():
    s = {"type": "object", "properties": {"skill": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": ["string", "null"]}}]}}}
    out = normalize_schema(s)
    sk = out["properties"]["skill"]
    assert "oneOf" not in sk and sk["anyOf"][1]["items"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}
    assert gemini_schema_problems(out) == []
    assert gemini_schema_problems(s) and any("oneOf" in p for p in gemini_schema_problems(s))


def test_normalize_tools_handles_openai_and_google_shapes_and_passes_others():
    oa = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"p": {"type": ["string", "array"], "items": {"type": "string"}}}}}}]
    out = normalize_tools(oa)
    assert "anyOf" in out[0]["function"]["parameters"]["properties"]["p"]
    assert "anyOf" not in oa[0]["function"]["parameters"]["properties"]["p"]      # original untouched
    g = [{"function_declarations": [{"name": "f", "parameters": {"type": "object", "properties": {"p": {"oneOf": [{"type": "string"}]}}}}]}]
    assert "anyOf" in normalize_tools(g)[0]["function_declarations"][0]["parameters"]["properties"]["p"]
    assert normalize_tools(None) is None and normalize_tools(["not a dict"]) == ["not a dict"]


# ── every registered tool of every orchestrator ─────────────────────────

def _analysis():
    from scilink.agents.exp_agents.analysis_orchestrator_tools import AnalysisOrchestratorTools
    t = AnalysisOrchestratorTools.__new__(AnalysisOrchestratorTools)
    t.orch = SimpleNamespace(base_dir=Path("/tmp/x"), model=None, analysis_results=[], futurehouse_api_key=None,
                             current_metadata=None, _custom_skills={})
    t.functions_map, t.openai_schemas = {}, []; t._register_all_tools(); return t.openai_schemas


def _planning():
    from scilink.agents.planning_agents.orchestrator_tools import OrchestratorTools
    return OrchestratorTools(SimpleNamespace(base_dir=Path("/tmp/x"), planner=SimpleNamespace())).openai_schemas


def _simulation():
    from scilink.agents.sim_agents.simulation_orchestrator_tools import SimulationOrchestratorTools
    t = SimulationOrchestratorTools.__new__(SimulationOrchestratorTools)
    t.orch = SimpleNamespace(base_dir="/tmp/x", model=None, generated_structures=[])
    t.functions_map, t.openai_schemas = {}, []; t.logger = logging.getLogger("sim"); t._register_all_tools(); return t.openai_schemas


def _meta():
    from scilink.agents.meta_agent.meta_orchestrator_tools import MetaOrchestratorTools
    t = MetaOrchestratorTools.__new__(MetaOrchestratorTools)
    t.orch = SimpleNamespace(base_dir=Path("/tmp/x"), model=None, _delegation_ledger=[], _children={})
    t.functions_map, t.openai_schemas = {}, []; t._register_all_tools(); return t.openai_schemas


ORCHESTRATORS = {"analysis": _analysis, "planning": _planning, "simulation": _simulation, "meta": _meta}


@pytest.mark.parametrize("name", sorted(ORCHESTRATORS))
def test_every_registered_tool_is_portable_after_normalization(name):
    tools = ORCHESTRATORS[name]()
    assert len(tools) > 5
    normalized = normalize_tools(tools)
    assert tool_schema_problems(normalized) == []
    json.dumps(normalized)                                        # serializable for every provider
    # names, descriptions and required lists survive untouched
    for a, b in zip(tools, normalized):
        assert a["function"]["name"] == b["function"]["name"]
        assert a["function"]["description"] == b["function"]["description"]
        assert a["function"]["parameters"].get("required", []) == b["function"]["parameters"].get("required", [])
        assert set(a["function"]["parameters"]["properties"]) == set(b["function"]["parameters"]["properties"])


def test_the_authored_planning_schemas_are_the_ones_gemini_rejected():
    """The un-normalized planning set carries the #606 shapes; the check
    catches them (so the test would have failed before the fix)."""
    problems = tool_schema_problems(_planning())
    assert any("search_literature" in p and "objective" in p for p in problems)


def test_litellm_gemini_conversion_keeps_items_only_on_arrays():
    """After normalization LiteLLM's Gemini schema builder produces no
    `items` on a non-array variant and no empty schema from oneOf."""
    from litellm.llms.vertex_ai.common_utils import _build_vertex_schema

    def walk(node, path="p"):
        bad = []
        if isinstance(node, dict):
            if node == {} and path.endswith("]"):
                bad.append(path + ": empty schema")
            if "items" in node and node.get("type", "").lower() not in ("array",) and "anyOf" not in node:
                bad.append(path + ": items on " + str(node.get("type")))
            for k, v in node.items():
                bad += walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                bad += walk(v, f"{path}[{i}]")
        return bad

    for name, build in ORCHESTRATORS.items():
        for tool in normalize_tools(build()):
            params = copy.deepcopy(tool["function"]["parameters"])
            converted = _build_vertex_schema(params)
            for pname, prop in (converted.get("properties") or {}).items():
                assert prop != {}, f"{name}.{tool['function']['name']}.{pname} collapsed to an empty schema"
            assert walk(converted) == [], f"{name}.{tool['function']['name']}: {walk(converted)}"


def test_wrapper_normalizes_tools_and_sets_reasoning_off_for_openai_gpt5():
    from scilink.wrappers.litellm_wrapper import LiteLLMGenerativeModel, _openai_tools_need_no_reasoning
    tools = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {
        "type": "object", "properties": {"p": {"type": ["string", "array"], "items": {"type": "string"}}}}}}]
    m = LiteLLMGenerativeModel.__new__(LiteLLMGenerativeModel); m.model = "openai/gpt-5.6-sol"
    params = m._build_params(None, tools)
    assert "anyOf" in params["tools"][0]["function"]["parameters"]["properties"]["p"]
    assert params["reasoning_effort"] == "none"
    m.model = "gemini/gemini-3.8-flash"
    assert "reasoning_effort" not in m._build_params(None, tools)
    m.model = "bedrock/us.anthropic.claude-opus-4-8"
    assert "reasoning_effort" not in m._build_params(None, tools)
    assert not _openai_tools_need_no_reasoning("gpt-4o") and _openai_tools_need_no_reasoning("gpt-5.6-sol")
    assert m._build_params(None, None).get("tools") is None
