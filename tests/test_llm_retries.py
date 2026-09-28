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
    assert sleeps == [7.0]


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
