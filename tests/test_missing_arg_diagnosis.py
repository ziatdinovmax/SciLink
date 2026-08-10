"""A missing required argument must not be blamed on truncation by guess.

The old message asserted the arguments "were likely truncated by the
response length limit" and told the model to split its content across
save_file + append_file chunks — a cause it never checked, and a remedy that
abandons the tool rather than retrying it.

An inference from schema order was tried and is unsound: instrumentation
caught a save_file call with finish_reason='length' — genuinely cut — that
was VALID JSON missing `content` (declared second) while `deliverable` and
`title` (declared fourth and fifth) arrived. Arguments are emitted in the
model's chosen order, and a cut response loses the LONG value while short
ones already emitted survive. So the message claims nothing about the cause
and simply asks for the call again.
"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scilink.agents.planning_agents.orchestrator_tools import OrchestratorTools


def _tools():
    tmp = Path(tempfile.mkdtemp(prefix="missingargs_"))
    t = OrchestratorTools.__new__(OrchestratorTools)
    t.orch = SimpleNamespace(
        planner=SimpleNamespace(state={}, model=None, kb_docs=None,
                                generation_config=None,
                                _build_skill_context=lambda s: None),
        base_dir=tmp, _active_output_subdir=None, lit_agent=None)
    t.functions_map, t.gemini_functions, t.openai_schemas = {}, [], []
    OrchestratorTools._register_all_tools(t)
    return t


def _declared(t, tool):
    """Schema-declaration order, read straight off the registered schema."""
    for schema in t.openai_schemas:
        fn = schema["function"]
        if fn["name"] == tool:
            return list(fn["parameters"]["properties"].keys())
    return []


def _msg(t, tool, **kw):
    return json.loads(t.execute_tool(tool, **kw))["message"]


def test_schema_order_is_available_and_request_comes_first():
    """Declaration order is still read (the diagnostic records it), but it
    is no longer used to infer whether the call was truncated."""
    t = _tools()
    order = _declared(t, "write_technical_document")
    assert order[0] == "request", order
    assert {"filename", "title", "source_files"} <= set(order[1:])
    assert t._required_params("write_technical_document") == ["request"]


def test_it_claims_nothing_about_truncation():
    """The live shape: request missing, later-declared args all arrived."""
    t = _tools()
    m = _msg(t, "write_technical_document", filename="memo.md", title="T",
             use_literature=False, source_files="x.md")

    for claim in ("nothing was truncated", "complete JSON", "not emitted",
                  "response length limit"):
        assert claim not in m, f"still asserting a cause: {claim!r}"
    assert "append_file" not in m and "save_file" not in m


def test_the_counterexample_that_killed_schema_order_reasoning():
    """save_file, finish_reason='length', content missing while deliverable
    and title — both declared AFTER it — arrived. Any message telling the
    model this call was complete would be factually wrong."""
    t = _tools()
    order = _declared(t, "save_file")
    assert order.index("content") < order.index("deliverable")
    assert order.index("content") < order.index("title")

    m = _msg(t, "save_file", filename="m.md", deliverable=True, title="T")
    assert "content" in m
    assert "nothing was truncated" not in m
    assert "complete" not in m


def test_it_reports_what_did_arrive():
    """Useful without being a claim about why the rest did not."""
    t = _tools()
    m = _msg(t, "write_technical_document", filename="memo.md", title="T")
    assert "filename" in m and "title" in m


def test_it_says_resend_not_switch_tools():
    """The failure mode being fixed is abandonment of the tool."""
    t = _tools()
    m = _msg(t, "write_technical_document", filename="memo.md")
    assert "Resend the SAME call" in m
    # And it corrects the misuse that produced the omission: the brief is a
    # specification, not the document.
    assert "specification" in m
    assert "not the finished text" in m


def test_every_tool_gets_the_same_cause_neutral_message():
    """Sweep every registered tool: none may assert or deny truncation."""
    t = _tools()
    for schema in t.openai_schemas:
        name = schema["function"]["name"]
        if not t._required_params(name):
            continue
        m = _msg(t, name)
        assert "nothing was truncated" not in m, name
        assert "response length limit" not in m, name
        assert "Resend the SAME call" in m, name


def test_the_missing_argument_is_always_named():
    t = _tools()
    m = _msg(t, "write_technical_document", filename="memo.md")
    assert "request" in m


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
