"""SciLink retries what a retry can fix, backs off with jitter, and bounds
every attempt with a timeout.

LiteLLM's ``num_retries`` uses its "constant_retry" strategy by default: no
wait between attempts, and any openai.APIError is retried, a 400 or a 401
included. A throttled provider was hit again at once, and N concurrent callers
came back in step. The chat-session path had no retries or timeout at all, and
``litellm_completion`` (every orchestrator chat loop) no timeout: LiteLLM's
default is 6000 s.
"""

from unittest import mock

import httpx
import litellm
import pytest

from scilink.wrappers import litellm_wrapper as lw


def _rate_limited(retry_after=None):
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    return litellm.RateLimitError(
        message="throttled", llm_provider="bedrock", model="m",
        response=httpx.Response(429, headers=headers, request=httpx.Request("POST", "http://x")))


def _unavailable():
    return litellm.ServiceUnavailableError(message="busy", llm_provider="bedrock", model="m")


def _bad_request():
    return litellm.BadRequestError(message="bad", model="m", llm_provider="bedrock")


def _auth():
    return litellm.AuthenticationError(message="no", llm_provider="bedrock", model="m")


def _timeout():
    return litellm.Timeout(message="slow", model="m", llm_provider="bedrock")


class _Provider:
    """Stands in for litellm.completion: raises the scripted errors, then answers."""

    def __init__(self, *errors):
        self.errors = list(errors)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        return mock.MagicMock(usage=None, choices=[])


@pytest.fixture
def sleeps(monkeypatch):
    waited = []
    monkeypatch.setattr(lw.time, "sleep", waited.append)
    return waited


def _run(provider, **kw):
    with mock.patch.object(lw.litellm, "completion", side_effect=provider):
        return lw.litellm_completion(model="gpt-4o", messages=[], **kw)


def test_transient_errors_are_retried_with_waits_between(sleeps):
    p = _Provider(_rate_limited(), _unavailable())
    _run(p)
    assert len(p.calls) == 3
    assert len(sleeps) == 2 and all(w > 0 for w in sleeps)


def test_litellm_does_not_retry_on_top_and_every_attempt_has_a_timeout(sleeps):
    p = _Provider(_unavailable())
    _run(p)
    assert all(c["num_retries"] == 0 for c in p.calls)
    assert all(c["timeout"] == lw.LLM_TIMEOUT_S for c in p.calls)


@pytest.mark.parametrize("err", [_bad_request, _auth], ids=["400", "401"])
def test_errors_a_retry_cannot_fix_are_raised_at_once(err, sleeps):
    p = _Provider(err())
    with pytest.raises(type(err())):
        _run(p)
    assert len(p.calls) == 1 and sleeps == []


def test_retry_after_is_honoured(sleeps):
    _run(_Provider(_rate_limited(retry_after=7)))
    assert len(sleeps) == 1 and 7.0 <= sleeps[0] <= 8.0     # the server's wait plus jitter


def test_backoff_grows_and_is_jittered():
    for attempt in range(6):
        step = min(lw._BACKOFF_CAP_S, lw._BACKOFF_BASE_S * 2 ** attempt)
        waits = {lw._backoff_s(attempt) for _ in range(50)}
        assert all(step / 2 <= w <= step for w in waits)
        assert len(waits) > 1                      # not in lockstep


def test_the_budget_is_bounded_and_the_last_error_surfaces(sleeps):
    p = _Provider(*[_unavailable() for _ in range(10)])
    with pytest.raises(litellm.ServiceUnavailableError):
        _run(p)
    assert len(p.calls) == lw.LLM_RETRIES + 1


def test_a_timeout_is_retried_once_not_the_whole_budget(sleeps):
    p = _Provider(_timeout(), _timeout(), _timeout())
    with pytest.raises(litellm.Timeout):
        _run(p)
    assert len(p.calls) == 2


def test_a_caller_can_turn_retries_off_or_set_its_own_timeout(sleeps):
    p = _Provider(_unavailable())
    with pytest.raises(litellm.ServiceUnavailableError):
        _run(p, num_retries=0)
    assert len(p.calls) == 1
    p = _Provider()
    _run(p, timeout=30)
    assert p.calls[0]["timeout"] == 30


def test_generate_content_uses_the_same_policy(sleeps, monkeypatch):
    model = lw.LiteLLMGenerativeModel("gpt-4o", timeout=99)
    monkeypatch.setattr(model, "_to_legacy_response", lambda r: r)
    p = _Provider(_rate_limited(), _bad_request())
    with mock.patch.object(lw.litellm, "completion", side_effect=p):
        with pytest.raises(litellm.BadRequestError):
            model.generate_content("hi")
    assert len(p.calls) == 2 and p.calls[0]["timeout"] == 99 and p.calls[0]["num_retries"] == 0


def test_the_chat_session_path_now_retries_and_times_out(sleeps, monkeypatch):
    model = lw.LiteLLMGenerativeModel("gpt-4o", timeout=42)
    monkeypatch.setattr(model, "_to_legacy_response", lambda r: mock.MagicMock(text="ok"))
    chat = model.start_chat()
    p = _Provider(_unavailable())
    with mock.patch.object(lw.litellm, "completion", side_effect=p):
        chat.send_message("hi")
    assert len(p.calls) == 2 and p.calls[0]["timeout"] == 42


# ── review follow-ups ─────────────────────────────────────────────────────

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _FakeOpenAI:
    """An OpenAI-compatible endpoint that answers 429 (with the given
    headers) a set number of times, then succeeds: the exception reaches
    SciLink exactly as LiteLLM raises it, not hand-built."""

    def __init__(self, n_429=1, headers=None):
        self.left, self.headers, self.hits = n_429, headers or {}, 0
        server = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                server.hits += 1
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if server.left > 0:
                    server.left -= 1
                    body = json.dumps({"error": {"message": "slow down", "type": "rate_limit"}}).encode()
                    self.send_response(429)
                    for k, v in server.headers.items():
                        self.send_header(k, v)
                else:
                    body = json.dumps({
                        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": "ok"}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
                    self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.mark.parametrize("headers, lo, hi", [
    ({"retry-after": "3"}, 3.0, 6.0),
    ({"retry-after-ms": "1500"}, 1.5, 3.0),
])
def test_retry_after_from_a_real_litellm_429_is_honoured(sleeps, headers, lo, hi):
    srv = _FakeOpenAI(n_429=1, headers=headers)
    try:
        lw.litellm_completion(model="openai/gpt-4o", api_base=srv.base, api_key="k",
                              messages=[{"role": "user", "content": "hi"}])
    finally:
        srv.close()
    assert srv.hits == 2
    assert len(sleeps) == 1 and lo <= sleeps[0] <= hi       # the server's wait, plus jitter


class _Status(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status_code = status


def _raised_from(outer, inner):
    try:
        try:
            raise inner
        except Exception as e:
            raise outer from e
    except Exception as exc:
        return exc


def test_bad_bedrock_credentials_under_a_connection_error_are_not_retried(sleeps):
    """On Bedrock a malformed or wrong API key arrives as APIConnectionError
    500 whose cause is a 403."""
    err = _raised_from(litellm.APIConnectionError(message="Invalid API Key format", llm_provider="bedrock",
                                                  model="m"), _Status(403))
    p = _Provider(err)
    with pytest.raises(litellm.APIConnectionError):
        _run(p)
    assert len(p.calls) == 1 and sleeps == []


def test_a_dropped_connection_keeps_the_full_budget(sleeps):
    import httpx
    errs = [_raised_from(litellm.APIConnectionError(message="Server disconnected", llm_provider="bedrock",
                                                    model="m"), httpx.RemoteProtocolError("disconnected"))
            for _ in range(3)]
    p = _Provider(*errs)
    _run(p)
    assert len(p.calls) == 4


def test_an_unclassified_connection_error_gets_one_retry(sleeps):
    errs = [litellm.APIConnectionError(message="something odd", llm_provider="bedrock", model="m")
            for _ in range(3)]
    p = _Provider(*errs)
    with pytest.raises(litellm.APIConnectionError):
        _run(p)
    assert len(p.calls) == 2


@pytest.mark.parametrize("make", [
    lambda: litellm.ContextWindowExceededError(message="too long", model="m", llm_provider="bedrock"),
    lambda: litellm.ContentPolicyViolationError(message="policy", model="m", llm_provider="bedrock"),
    lambda: litellm.NotFoundError(message="no model", model="m", llm_provider="bedrock"),
    lambda: litellm.UnsupportedParamsError(message="bad param", model="m", llm_provider="bedrock"),
], ids=["context_window", "content_policy", "not_found", "unsupported_params"])
def test_litellms_final_errors_are_not_retried(make, sleeps):
    p = _Provider(make())
    with pytest.raises(Exception):
        _run(p)
    assert len(p.calls) == 1


def test_an_overloaded_529_is_retried(sleeps):
    p = _Provider(_Status(529))
    _run(p)
    assert len(p.calls) == 2


def test_a_stop_during_the_wait_lands_before_the_next_attempt(monkeypatch):
    """The warning before the wait is not enough: the Stop can arrive during
    it. The record logged after the wait raises it before another call."""
    import io
    import logging
    from types import SimpleNamespace
    from scilink.server.runner import turn_log_handler
    from scilink.ui.output_capture import AgentStoppedError
    cap = SimpleNamespace(log_stream=io.StringIO(), stop_requested=False)
    handler = turn_log_handler(cap, threading.get_ident())
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    monkeypatch.setattr(lw.time, "sleep", lambda s: setattr(cap, "stop_requested", True))
    p = _Provider(_unavailable(), _unavailable())
    try:
        with pytest.raises(AgentStoppedError):
            _run(p)
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
    assert len(p.calls) == 1


def test_timeout_and_retries_can_be_set_from_the_environment(sleeps, monkeypatch):
    monkeypatch.setenv("SCILINK_LLM_TIMEOUT_S", "45")
    monkeypatch.setenv("SCILINK_LLM_RETRIES", "1")
    p = _Provider(_unavailable(), _unavailable())
    with pytest.raises(litellm.ServiceUnavailableError):
        _run(p)
    assert len(p.calls) == 2 and p.calls[0]["timeout"] == 45.0
    assert lw.LiteLLMGenerativeModel("gpt-4o").timeout == 45.0


def test_num_retries_none_means_the_default(sleeps):
    p = _Provider(_unavailable())
    _run(p, num_retries=None)
    assert len(p.calls) == 2
