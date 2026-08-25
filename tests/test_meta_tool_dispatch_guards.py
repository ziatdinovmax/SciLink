"""Meta tool dispatch: truncated and malformed calls get actionable errors.

Live, twice: `delegate_to_planning` arrived without its required `task`.
Its brief runs to thousands of words, so a call that hits the output-token
cap mid-generation is VALID but incomplete JSON — later keys simply absent —
and dispatching it raised a bare TypeError about a missing positional
argument. The model then spent a whole round trip re-emitting the same
oversized call. The planning orchestrator had guarded this; the meta had not.
"""

import json

import pytest

from scilink.agents.meta_agent.meta_orchestrator_tools import (
    MetaOrchestratorTools)


@pytest.fixture
def tools():
    t = MetaOrchestratorTools.__new__(MetaOrchestratorTools)
    t.openai_schemas = [{"function": {
        "name": "delegate_to_planning",
        "parameters": {"required": ["task", "label"]}}}]
    t.openai_schemas.append({"function": {
        "name": "view_document", "parameters": {"required": ["paths"]}}})
    t.functions_map = {
        "delegate_to_planning": lambda task, label, context=None: "ok",
        "view_document": lambda paths, page=None: "ok",
        "boom": lambda: (_ for _ in ()).throw(ValueError("real failure")),
    }
    return t


def _run(t, name, **kw):
    return json.loads(MetaOrchestratorTools.execute_tool(t, name, **kw))


def test_truncated_call_is_named_and_told_to_shorten(tools):
    out = _run(tools, "delegate_to_planning", label="x")
    assert out["status"] == "error" and out["tool"] == "delegate_to_planning"
    assert "task" in out["message"]
    assert "truncated" in out["message"]
    # re-sending the same text would truncate again — the advice must be to
    # restructure, not to retry
    assert "SHORTER" in out["message"] and "context" in out["message"]


def test_advice_fits_the_tool_it_is_given_to(tools):
    """The guards cover every meta tool — analysis and simulation
    delegations, fan-out, document viewing. `view_document` takes `paths`,
    so telling it to move detail into `context` would be nonsense."""
    delegation = _run(tools, "delegate_to_planning", label="x")["message"]
    other = _run(tools, "view_document")["message"]

    assert "`context`" in delegation
    assert "context" not in other
    assert "paths" in other and "split the work" in other


def test_every_missing_required_argument_is_listed(tools):
    out = _run(tools, "delegate_to_planning")
    assert "task" in out["message"] and "label" in out["message"]


def test_a_complete_call_still_dispatches(tools):
    assert MetaOrchestratorTools.execute_tool(
        tools, "delegate_to_planning", task="t", label="l") == "ok"


def test_optional_arguments_are_not_demanded(tools):
    """`context` is optional; requiring it would break every valid call."""
    assert MetaOrchestratorTools.execute_tool(
        tools, "delegate_to_planning", task="t", label="l") == "ok"


def test_unknown_argument_gets_the_accepted_list(tools):
    out = _run(tools, "delegate_to_planning", task="t", label="l", bogus=1)
    assert "unexpected keyword argument" in out["message"]
    for name in ("task", "label", "context"):
        assert name in out["message"]


def test_a_real_failure_inside_a_tool_reports_as_itself(tools):
    """The guards must not swallow genuine errors into argument advice."""
    out = _run(tools, "boom")
    assert out["status"] == "error" and "real failure" in out["message"]
    assert "truncated" not in out["message"]
    assert "Accepted arguments" not in out["message"]


def test_unknown_tool_is_still_reported(tools):
    out = _run(tools, "no_such_tool")
    assert out["status"] == "error" and "not found" in out["message"]


def test_guard_survives_a_toolset_without_schemas():
    """_required_params must not explode when schemas were never built."""
    t = MetaOrchestratorTools.__new__(MetaOrchestratorTools)
    t.functions_map = {"f": lambda: "ok"}
    assert MetaOrchestratorTools._required_params(t, "f") == []
    assert MetaOrchestratorTools.execute_tool(t, "f") == "ok"


# ── malformed arguments (#270, ported to the meta) ───────────────────

class _Fn:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments


class _Call:
    def __init__(self, arguments, name="delegate_to_planning"):
        self.id, self.function = "tc_1", _Fn(name, arguments)


def _parse(arguments, finish_reason=None):
    from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent
    return MetaOrchestratorAgent._parse_tool_args(
        _Call(arguments), finish_reason)


def test_valid_arguments_parse():
    args, err = _parse('{"task": "do it", "label": "x"}')
    assert err is None and args["task"] == "do it"


def test_malformed_arguments_are_not_silently_emptied():
    """The bug: json.JSONDecodeError -> args = {} -> the tool raises about a
    MISSING argument, so the model 'resubmits with the full task' — fixing
    the wrong thing. Live, that looped four times on one delegation."""
    args, err = _parse('{"task": "unterminated string...')
    assert args is None and err is not None
    msg = json.loads(err)["message"]
    assert "NOT a missing-argument error" in msg
    assert "re-sending the same call will fail the same way" in msg
    assert "SHORTER" in msg and "context" in msg


def test_truncation_and_bad_escaping_are_distinguished():
    """Different causes need different corrections."""
    truncated = json.loads(_parse('{"task": "abc', "length")[1])["message"]
    malformed = json.loads(_parse('{"task": "abc', None)[1])["message"]
    assert "output-token limit" in truncated
    assert "escaping" in malformed
    # both report how much actually arrived
    assert "characters received" in truncated and "characters received" in malformed


def test_meta_chat_is_graph_backed_with_a_single_dispatch_path():
    """
    The meta orchestrator used to have two hand-rolled dispatch loops
    (openai + litellm) — fixing one left the other looping, which is what
    this test used to guard against. It is now on the LangGraph backbone
    (chat() -> _invoke_graph() -> a single scilink.graphs._react.execute_tools
    path shared with analysis/planning/simulation), so there is exactly one
    dispatch path left, and the malformed-args recovery lives in
    scilink.graphs._react._parse_tool_args instead.
    """
    import inspect
    from scilink.agents.meta_agent import meta_orchestrator as mo
    from scilink.graphs import _react

    src = inspect.getsource(mo.MetaOrchestratorAgent)
    assert "_handle_openai_chat" not in src and "_handle_litellm_chat" not in src, (
        "expected the hand-rolled dual chat loops to be gone"
    )
    assert hasattr(mo.MetaOrchestratorAgent, "_invoke_graph")

    react_src = inspect.getsource(_react)
    assert "_parse_tool_args(" in react_src, (
        "expected the shared execute_tools to use the recovery-hint parser"
    )
    assert "args = {}" not in src.replace("``args = {}``", "")
