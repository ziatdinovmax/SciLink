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


@pytest.mark.parametrize("model", [
    "anthropic/claude-opus-5-5", "bedrock/us.anthropic.claude-opus-5-5",
    "gemini/gemini-4-argon", "openai/gpt-5.6-sol",
])
def test_other_models_are_sent_exactly_as_before(model):
    """Request parity: the rewrite is the identity off GPT-6."""
    seen = _sent(litellm_completion, model=model, messages=MSG, tools=TOOLS)
    with mock.patch.object(litellm_wrapper, "_responses_api_model", side_effect=lambda m, b=None: m):
        before = _sent(litellm_completion, model=model, messages=MSG, tools=TOOLS)
    assert seen == before


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
