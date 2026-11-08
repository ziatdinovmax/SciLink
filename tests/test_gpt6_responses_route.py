"""GPT-6+ on OpenAI's endpoint is sent through LiteLLM's Responses bridge.

OpenAI's GPT-6 family refuses function tools on /v1/chat/completions ("To use
function tools, use /v1/responses"), and does not accept the
``reasoning_effort="none"`` escape the gpt-5 family uses, so every
orchestrator turn failed on its first tool call. The fix rewrites the model
string of the REQUEST only, at the one call point every SciLink completion
goes through. These pin that rule and that nothing else changes: other
models are sent byte-for-byte as before, a proxy (``api_base``) is left
alone, and callers keep the name they passed.
"""

from unittest import mock

import pytest

from scilink.wrappers import litellm_wrapper
from scilink.wrappers.litellm_wrapper import (
    LiteLLMGenerativeModel, _responses_api_model, litellm_completion,
)

MSG = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def _no_ambient_openai_base(monkeypatch):
    """The route reads the endpoint the way LiteLLM does; a developer's own
    OPENAI_BASE_URL must not decide these tests."""
    for var in litellm_wrapper._OPENAI_BASE_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(litellm_wrapper.litellm, "api_base", None, raising=False)
TOOLS = [{"type": "function", "function": {
    "name": "f", "parameters": {"type": "object", "properties": {}}}}]


@pytest.mark.parametrize("given, sent", [
    ("gpt-6-astra", "openai/responses/gpt-6-astra"),
    ("openai/gpt-6-astra", "openai/responses/gpt-6-astra"),
    ("gpt-6.1-sol", "openai/responses/gpt-6.1-sol"),
    ("GPT-7-preview", "openai/responses/GPT-7-preview"),
])
def test_gpt6_and_later_route_to_responses(given, sent):
    assert _responses_api_model(given) == sent


@pytest.mark.parametrize("model", [
    "gpt-5.6-sol", "openai/gpt-5.6-sol", "gpt-4o", "o3-mini",
    "claude-opus-5-5", "anthropic/claude-opus-5-5",
    "bedrock/us.anthropic.claude-opus-5-5", "gemini/gemini-4-argon",
    "azure/gpt-6-astra", "openai/responses/gpt-6-astra", "text-embedding-3-small",
    "", None,
])
def test_everything_else_is_untouched(model):
    assert _responses_api_model(model) == model


def test_a_proxy_is_left_alone():
    assert _responses_api_model("gpt-6-astra", "https://proxy.example/v1") == "gpt-6-astra"


@pytest.mark.parametrize("var", ["OPENAI_BASE_URL", "OPENAI_API_BASE"])
def test_a_proxy_set_in_the_environment_is_left_alone(monkeypatch, var):
    """LiteLLM sends an openai/ request to these when no api_base is passed."""
    monkeypatch.setenv(var, "https://proxy.example/v1")
    assert _responses_api_model("gpt-6-astra") == "gpt-6-astra"


def test_a_proxy_set_on_litellm_is_left_alone(monkeypatch):
    monkeypatch.setattr(litellm_wrapper.litellm, "api_base", "https://proxy.example/v1")
    assert _responses_api_model("gpt-6-astra") == "gpt-6-astra"


@pytest.mark.parametrize("base", ["https://api.openai.com/v1", "https://api.openai.com"])
def test_openais_own_url_is_not_a_proxy(monkeypatch, base):
    monkeypatch.setenv("OPENAI_BASE_URL", base)
    assert _responses_api_model("gpt-6-astra") == "openai/responses/gpt-6-astra"
    assert _responses_api_model("gpt-6-astra", base) == "openai/responses/gpt-6-astra"


def _sent(call, **kwargs):
    """Call through a stubbed litellm; return the kwargs litellm received."""
    seen = {}

    def fake_completion(*a, **kw):
        seen.update(kw)
        return object()

    with mock.patch.object(litellm_wrapper.litellm, "completion", side_effect=fake_completion):
        call(**kwargs)
    return seen


def test_the_orchestrator_entry_point_sends_the_responses_route():
    seen = _sent(litellm_completion, model="openai/gpt-6-astra", messages=MSG, tools=TOOLS)
    assert seen["model"] == "openai/responses/gpt-6-astra"
    assert "reasoning_effort" not in seen        # "none" is refused by GPT-6


_SENT_TOOLS = [{"type": "function", "function": {
    "name": "f", "parameters": {"type": "object", "properties": {}}}}]
_COMMON = {"messages": MSG, "num_retries": 0, "timeout": 1200.0, "tools": _SENT_TOOLS}


# The request each model gets on main, captured from main's own code: the
# route must leave every one of them exactly as it was.
@pytest.mark.parametrize("model, expected", [
    ("anthropic/claude-opus-5-5",
     {**_COMMON, "model": "anthropic/claude-opus-5-5", "max_tokens": 128000}),
    ("bedrock/us.anthropic.claude-opus-5-5",
     {**_COMMON, "model": "bedrock/us.anthropic.claude-opus-5-5", "max_tokens": 128000}),
    ("gemini/gemini-4-argon", {**_COMMON, "model": "gemini/gemini-4-argon"}),
    ("openai/gpt-5.6-sol",
     {**_COMMON, "model": "openai/gpt-5.6-sol", "reasoning_effort": "none"}),
])
def test_other_models_are_sent_exactly_as_before(model, expected):
    assert _sent(litellm_completion, model=model, messages=MSG, tools=TOOLS) == expected


def test_the_wrapper_class_sends_the_route_but_keeps_its_name():
    m = LiteLLMGenerativeModel("gpt-6-astra", api_key="sk-test")
    seen = _sent(lambda **kw: m.generate_content("hi"))
    assert seen["model"] == "openai/responses/gpt-6-astra"
    assert m.model == "openai/gpt-6-astra"      # limiter keys and logs unchanged


def _bridge_reply(*parts):
    """A ModelResponse shaped like the Responses bridge's: one choice per output
    item, ``("text", str)`` or ``("tool", name)``."""
    from litellm.types.utils import ChatCompletionMessageToolCall, Choices, Function, Message, ModelResponse
    choices = []
    for i, (kind, val) in enumerate(parts):
        if kind == "text":
            choices.append(Choices(index=i, finish_reason="stop",
                                   message=Message(role="assistant", content=val)))
        else:
            tc = ChatCompletionMessageToolCall(id=f"call_{i}", type="function",
                                               function=Function(name=val, arguments="{}"))
            choices.append(Choices(index=i, finish_reason="tool_calls",
                                   message=Message(role="assistant", content=None, tool_calls=[tc])))
    return ModelResponse(choices=choices)


def _via_entry_point(model, reply):
    with mock.patch.object(litellm_wrapper.litellm, "completion", return_value=reply):
        return litellm_completion(model=model, messages=MSG, tools=TOOLS)


def test_a_preamble_and_its_tool_call_come_back_as_one_message():
    """Caught live: GPT-6 said 'I'm sending the cube to the specialist' and called
    delegate_to_analysis in the same reply; the bridge split them and the meta
    loop, reading choices[0], ended its turn without delegating."""
    r = _via_entry_point("gpt-6-astra", _bridge_reply(("text", "I'll delegate this."),
                                                      ("tool", "delegate_to_analysis")))
    assert len(r.choices) == 1
    msg = r.choices[0].message
    assert [t.function.name for t in msg.tool_calls] == ["delegate_to_analysis"]
    assert msg.content == "I'll delegate this."
    assert r.choices[0].finish_reason == "tool_calls"


def test_several_tool_calls_are_all_kept():
    r = _via_entry_point("gpt-6-astra", _bridge_reply(("text", "Two lookups."), ("tool", "a"), ("tool", "b")))
    assert [t.function.name for t in r.choices[0].message.tool_calls] == ["a", "b"]


def test_a_reply_in_two_messages_comes_back_whole():
    """No tool call: the second message is often the answer itself."""
    r = _via_entry_point("gpt-6-astra", _bridge_reply(("text", "Looking at the data."),
                                                      ("text", '{"answer": 42}')))
    assert len(r.choices) == 1
    assert r.choices[0].message.content == 'Looking at the data.\n{"answer": 42}'
    assert r.choices[0].message.tool_calls is None
    assert r.choices[0].finish_reason == "stop"


def test_text_beside_a_tool_call_is_kept():
    from litellm.types.utils import ChatCompletionMessageToolCall, Choices, Function, Message, ModelResponse
    tc = ChatCompletionMessageToolCall(id="c", type="function", function=Function(name="f", arguments="{}"))
    reply = ModelResponse(choices=[
        Choices(index=0, finish_reason="stop", message=Message(role="assistant", content="First.")),
        Choices(index=1, finish_reason="tool_calls",
                message=Message(role="assistant", content="Then this.", tool_calls=[tc]))])
    msg = _via_entry_point("gpt-6-astra", reply).choices[0].message
    assert msg.content == "First.\nThen this."
    assert [t.function.name for t in msg.tool_calls] == ["f"]


def test_a_truncated_last_part_keeps_its_finish_reason():
    reply = _bridge_reply(("text", "a"), ("text", "b"))
    reply.choices[-1].finish_reason = "length"
    assert _via_entry_point("gpt-6-astra", reply).choices[0].finish_reason == "length"


def test_a_single_choice_is_untouched():
    reply = _bridge_reply(("text", "Just an answer."))
    r = _via_entry_point("gpt-6-astra", reply)
    assert r.choices[0].message.content == "Just an answer." and r.choices[0].finish_reason == "stop"


@pytest.mark.parametrize("model", ["anthropic/claude-opus-5-5", "openai/gpt-5.6-sol",
                                   "gemini/gemini-4-argon"])
def test_other_models_responses_are_never_merged(model):
    r = _via_entry_point(model, _bridge_reply(("text", "a"), ("tool", "f")))
    assert len(r.choices) == 2


def test_the_limiter_and_sampling_memory_see_the_callers_name():
    names = []

    def fake_call_dropping(call, kwargs, retries, model):
        names.append((model, kwargs["model"]))
        return object()

    with mock.patch.object(litellm_wrapper, "call_dropping_refused_sampling",
                           side_effect=fake_call_dropping):
        litellm_completion(model="openai/gpt-6-astra", messages=MSG)
    assert names == [("openai/gpt-6-astra", "openai/responses/gpt-6-astra")]


@pytest.mark.parametrize("model, refuses", [
    ("gpt-6-astra", True), ("openai/gpt-6.1-sol", True), ("gpt-5.6-sol", True),
    ("o3-mini", True), ("gpt-4o", False), ("gpt-4.1", False), ("claude-opus-5-5", False),
])
def test_sampling_is_omitted_up_front_for_gpt6_on_every_wrapper(model, refuses):
    """GPT-6 refuses temperature/top_p; sending them cost one 400 per process."""
    from scilink.wrappers import openai_wrapper, openai_wrapper_tools
    for mod in (litellm_wrapper, openai_wrapper, openai_wrapper_tools):
        assert mod._is_openai_reasoning_model(model) is refuses, mod.__name__


class _FailingRaw:
    def __init__(self):
        self.calls = 0

    def create(self, **kw):
        self.calls += 1
        raise RuntimeError("To use function tools, use /v1/responses")


def _proxy_call(model, tools=TOOLS):
    from scilink.wrappers.tool_schema import _PortableCompletions
    comp = _PortableCompletions(_FailingRaw(), model)
    kw = {"model": model, "messages": MSG}
    if tools:
        kw["tools"] = tools
    with pytest.raises(RuntimeError, match="v1/responses"):   # the provider's error, unchanged
        comp.create(**kw)


def test_a_gpt6_tool_failure_through_a_proxy_is_explained_once(caplog, monkeypatch):
    from scilink.wrappers import tool_schema
    monkeypatch.setattr(tool_schema, "_gpt6_tool_hint_given", set())
    with mock.patch.object(litellm_wrapper, "call_with_retries", side_effect=lambda f, r, model=None: f()):
        with caplog.at_level("WARNING", logger=tool_schema.__name__):
            _proxy_call("gpt-6-astra")
            _proxy_call("gpt-6-astra")
    hints = [r for r in caplog.records if "Responses API" in r.getMessage()]
    assert len(hints) == 1 and "base_url" in hints[0].getMessage()


@pytest.mark.parametrize("model, tools", [("gpt-5.6-sol", TOOLS), ("claude-opus-5-5", TOOLS),
                                          ("gpt-6-astra", None)])
def test_no_explanation_off_gpt6_tool_calls(caplog, monkeypatch, model, tools):
    from scilink.wrappers import tool_schema
    monkeypatch.setattr(tool_schema, "_gpt6_tool_hint_given", set())
    with mock.patch.object(litellm_wrapper, "call_with_retries", side_effect=lambda f, r, model=None: f()):
        with caplog.at_level("WARNING", logger=tool_schema.__name__):
            _proxy_call(model, tools)
    assert not [r for r in caplog.records if "Responses API" in r.getMessage()]
