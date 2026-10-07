"""A model that refuses the sampling parameters is sent its request again
without them, and remembered.

Providers are retiring ``temperature`` / ``top_p`` / ``top_k``: a newer model
answers a request that sets one with a 400, which an agent reads as a failed
step — a verifier's call came back as an API error and every candidate scored
0.0. The name rules know some of these models; these pin the rule for the rest,
through the real HTTP paths (LiteLLM and the OpenAI-compatible proxy client)
against a local server that refuses the knobs the way a provider does.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from scilink.wrappers import litellm_wrapper
from scilink.wrappers.litellm_wrapper import LiteLLMGenerativeModel, _sampling_refusal

REFUSAL = "`temperature` is deprecated for this model."


class _Provider:
    """A local endpoint that answers Anthropic messages and OpenAI chat
    completions, refusing any request that sets a sampling parameter when
    ``refuses`` is on. Records every request body."""

    def __init__(self, refuses: bool, message: str = REFUSAL, always: bool = False):
        self.refuses, self.message, self.always, self.bodies = refuses, message, always, []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
                outer.bodies.append(body)
                anthropic = self.path.endswith("/messages")
                if outer.always or (outer.refuses
                                    and any(k in body for k in ("temperature", "top_p", "top_k"))):
                    err = ({"type": "error", "error": {"type": "invalid_request_error",
                                                       "message": outer.message}}
                           if anthropic else
                           {"error": {"message": outer.message, "type": "invalid_request_error",
                                      "param": "temperature", "code": "unsupported_value"}})
                    return self._send(400, err)
                if anthropic:
                    return self._send(200, {
                        "id": "msg_1", "type": "message", "role": "assistant", "model": body.get("model"),
                        "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                        "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}})
                return self._send(200, {
                    "id": "c1", "object": "chat.completion", "created": 0, "model": body.get("model"),
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

            def _send(self, status, payload):
                raw = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture(autouse=True)
def _fresh_memory(monkeypatch):
    monkeypatch.setattr(litellm_wrapper, "_sampling_refused", set())
    monkeypatch.setenv("SCILINK_LLM_RETRIES", "0")


@pytest.fixture
def provider(request):
    p = _Provider(**getattr(request, "param", {"refuses": True}))
    yield p
    p.close()


def _model(provider, name="anthropic/claude-next-9"):
    return LiteLLMGenerativeModel(name, api_key="x", base_url=provider.url)


def test_a_refused_temperature_is_dropped_and_the_model_remembered(provider):
    m = _model(provider)
    out = m.generate_content("hi", generation_config={"temperature": 0.0})
    assert out.text == "ok"
    assert len(provider.bodies) == 2
    assert provider.bodies[0]["temperature"] == 0.0
    assert "temperature" not in provider.bodies[1]
    # Remembered: the next request omits it up front, one request only.
    m.generate_content("again", generation_config={"temperature": 0.0, "top_p": 0.9})
    assert len(provider.bodies) == 3
    assert not {"temperature", "top_p"} & set(provider.bodies[2])


@pytest.mark.parametrize("provider", [{"refuses": False}], indirect=True)
def test_a_model_that_accepts_temperature_gets_the_same_single_request(provider):
    m = _model(provider, "anthropic/claude-sonnet-4-5")
    m.generate_content("hi", generation_config={"temperature": 0.0})
    m.generate_content("hi", generation_config={"temperature": 0.0})
    assert len(provider.bodies) == 2
    assert all(b["temperature"] == 0.0 for b in provider.bodies)
    assert litellm_wrapper._sampling_refused == set()


@pytest.mark.parametrize("provider", [{"refuses": True, "message": "prompt is too long: temperature"
                                       " readings table exceeds the context"}], indirect=True)
def test_another_bad_request_is_raised_and_not_resent(provider):
    m = _model(provider)
    with pytest.raises(Exception):
        m.generate_content("hi", generation_config={"temperature": 0.0})
    assert len(provider.bodies) == 1
    assert litellm_wrapper._sampling_refused == set()


@pytest.mark.parametrize("provider", [{"refuses": True, "always": True,
                                       "message": "Parameter 'temperature' echoed: tool schema not "
                                                  "supported for this deployment"}], indirect=True)
def test_a_match_whose_resend_also_fails_is_raised_and_not_remembered(provider):
    """A 400 about something else that quotes a sampling parameter passes the
    wording check; the resend fails the same way, the error is raised, and
    the model keeps its parameters on the next request."""
    m = _model(provider)
    with pytest.raises(Exception):
        m.generate_content("hi", generation_config={"temperature": 0.0})
    assert len(provider.bodies) == 2 and "temperature" not in provider.bodies[1]
    assert litellm_wrapper._sampling_refused == set()
    with pytest.raises(Exception):
        m.generate_content("again", generation_config={"temperature": 0.0})
    assert provider.bodies[2]["temperature"] == 0.0


def test_a_request_without_sampling_parameters_is_not_resent():
    # Nothing to drop: a 400 for a request that set no knob is raised as is.
    from scilink.wrappers.litellm_wrapper import call_dropping_refused_sampling

    calls = []

    def call(kw):
        calls.append(dict(kw))
        raise _bad_request(REFUSAL)

    with pytest.raises(Exception):
        call_dropping_refused_sampling(call, {"model": "m", "messages": []}, 0, "m")
    assert len(calls) == 1


def test_the_proxy_client_drops_a_refused_parameter_too(provider):
    from scilink.wrappers.openai_wrapper import OpenAIAsGenerativeModel

    provider.message = ("Unsupported value: 'temperature' does not support 0.0 with this model. "
                        "Only the default (1) value is supported.")
    m = OpenAIAsGenerativeModel(model="next-model", api_key="x", base_url=provider.url)
    cfg = type("Cfg", (), {"temperature": 0.0})()
    assert m.generate_content(["hi"], generation_config=cfg).text == "ok"
    assert len(provider.bodies) == 2 and "temperature" not in provider.bodies[1]
    assert "next-model" in litellm_wrapper._sampling_refused


@pytest.mark.parametrize("model", ["anthropic/claude-opus-4-8", "openai/gpt-5.5"])
def test_the_name_rules_omit_every_sampling_parameter(model):
    params = LiteLLMGenerativeModel(model, api_key="x")._build_params(
        {"temperature": 0.0, "top_p": 0.9, "top_k": 5, "max_output_tokens": 100}, None)
    assert not {"temperature", "top_p", "top_k"} & set(params)
    assert params["max_tokens"] == 100


def test_a_model_without_a_name_rule_keeps_its_sampling_parameters():
    params = LiteLLMGenerativeModel("anthropic/claude-sonnet-4-5", api_key="x")._build_params(
        {"temperature": 0.0, "top_p": 0.9, "top_k": 5}, None)
    assert (params["temperature"], params["top_p"], params["top_k"]) == (0.0, 0.9, 5)


def _bad_request(message):
    import httpx
    import openai
    req = httpx.Request("POST", "http://x")
    return openai.BadRequestError(message, response=httpx.Response(400, request=req), body=None)


@pytest.mark.parametrize("message", [
    "`temperature` is deprecated for this model.",
    "Unsupported value: 'temperature' does not support 0.0 with this model. Only the default (1) value is supported.",
    "Unsupported parameter: 'top_p' is not supported with this model.",
    "temperature is not supported for this model",
    "Parameter top_k is no longer supported.",
])
def test_refusal_wordings_are_recognised(message):
    assert _sampling_refusal(_bad_request(message))


@pytest.mark.parametrize("message", [
    "prompt is too long",
    "temperature must be between 0 and 1",
    "messages: at least one message is required",
])
def test_other_bad_requests_are_not_refusals(message):
    assert not _sampling_refusal(_bad_request(message))


def test_a_refusal_wording_on_a_non_400_is_not_a_refusal():
    import httpx
    import openai
    req = httpx.Request("POST", "http://x")
    exc = openai.InternalServerError("`temperature` is deprecated for this model.",
                                     response=httpx.Response(500, request=req), body=None)
    assert not _sampling_refusal(exc)
