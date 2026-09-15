"""The orchestrator chat loops must not run on Bedrock's default output cap.

#238 injected a model-appropriate ``max_tokens`` because Bedrock omits
``maxTokens`` when unset and falls back to a low server-side default. That
injection went into ``LiteLLMGenerativeModel`` only — but every chat-driven
orchestrator (planning, meta, simulation) and ``structure_agent`` call
``litellm_completion`` instead, so their loops kept the low default.

Caught live: a tool call returned ``finish_reason='length'`` at exactly 4096
completion tokens with a required argument missing, and the fallback the
model chose hit the same wall. These pin the ceiling at the shared entry
point without disturbing other providers.
"""

import sys
from unittest import mock

import pytest

from scilink.wrappers import litellm_wrapper
from scilink.wrappers.litellm_wrapper import litellm_completion


def _captured(**kwargs):
    """Call through a stubbed litellm and return the params it received."""
    seen = {}

    def fake_completion(*a, **kw):
        seen.update(kw)
        seen["_positional"] = a
        return object()

    with mock.patch.object(litellm_wrapper.litellm, "completion",
                           side_effect=fake_completion):
        litellm_completion(**kwargs)
    return seen


def test_bedrock_gets_an_explicit_ceiling():
    seen = _captured(model="bedrock/us.anthropic.claude-opus-4-8",
                     messages=[{"role": "user", "content": "hi"}])
    assert "max_tokens" in seen, "the chat loop would run on Bedrock's default"
    assert seen["max_tokens"] > 4096, seen["max_tokens"]


def test_anthropic_direct_gets_one_too():
    seen = _captured(model="anthropic/claude-opus-4-8",
                     messages=[{"role": "user", "content": "hi"}])
    assert seen.get("max_tokens", 0) > 4096


def test_an_explicit_value_from_the_caller_still_wins():
    seen = _captured(model="bedrock/us.anthropic.claude-opus-4-8",
                     messages=[], max_tokens=256)
    assert seen["max_tokens"] == 256


def test_other_providers_are_untouched():
    """Scope check: litellm already gives these a sane ceiling, so injecting
    one here would be a behaviour change nobody asked for."""
    for model in ("gpt-4o", "openai/gpt-4o", "gemini/gemini-2.0-flash",
                  "vertex_ai/gemini-2.0-flash"):
        seen = _captured(model=model, messages=[])
        assert "max_tokens" not in seen, model


def test_the_model_can_be_positional():
    seen = {}

    def fake_completion(*a, **kw):
        seen.update(kw)
        return object()

    with mock.patch.object(litellm_wrapper.litellm, "completion",
                           side_effect=fake_completion):
        litellm_completion("bedrock/us.anthropic.claude-opus-4-8", messages=[])
    assert seen.get("max_tokens", 0) > 4096


def test_retries_still_defaulted():
    """Pre-existing behaviour must survive the change."""
    seen = _captured(model="gpt-4o", messages=[])
    assert seen["num_retries"] == 4


def test_it_sends_exactly_what_the_registry_reports():
    """Never a rounded-up or invented number — the registry value or nothing."""
    m = "bedrock/us.anthropic.claude-opus-4-8"
    seen = _captured(model=m, messages=[])
    assert seen["max_tokens"] == litellm_wrapper._registered_max_output_tokens(m)


@pytest.mark.parametrize("model", [
    # An ordinary Bedrock name litellm has no entry for — not exotic.
    "bedrock/us.anthropic.claude-sonnet-4-5-v1:0",
    "bedrock/some.custom.unmapped-model",
    "anthropic/my-private-deployment",
])
def test_unknown_models_are_left_alone(model):
    """Guessing high on a deployment nobody verified trades a silent
    truncation for an outright rejection, under every orchestrator loop."""
    assert litellm_wrapper._registered_max_output_tokens(model) is None
    seen = _captured(model=model, messages=[])
    assert "max_tokens" not in seen, model


def test_the_helper_reports_none_rather_than_a_default():
    """The distinction this whole tightening rests on."""
    assert litellm_wrapper._registered_max_output_tokens(
        "bedrock/definitely-not-a-real-model") is None
    assert litellm_wrapper._resolve_max_output_tokens(
        "bedrock/definitely-not-a-real-model") == \
        litellm_wrapper.DEFAULT_MAX_OUTPUT_TOKENS


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
