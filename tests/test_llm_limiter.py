"""At most ``llm_max_inflight()`` calls to one model are in flight in a process,
and every call made for a worker is charged to that worker.

A swarm puts many workers on one provider at once. The cap holds on every
path a call can take (the LiteLLM completion helper, embeddings, the internal
proxy's client), is never held across a retry's backoff sleep, and can be
lifted. Usage recorded by the ledger names the worker beside the session.
"""

import threading
import time
from types import SimpleNamespace

import pytest

from scilink import tracing
from scilink.usage import UsageLedger
from scilink.wrappers import litellm_wrapper as lw
from scilink.wrappers import llm_limiter
from scilink.wrappers.tool_schema import portable_openai_client


class Gauge:
    """A fake provider call that records how many run at once."""

    def __init__(self, seconds=0.05):
        self.now = self.peak = self.calls = 0
        self.seconds = seconds
        self._lock = threading.Lock()

    def __call__(self, *args, **kwargs):
        with self._lock:
            self.now += 1
            self.calls += 1
            self.peak = max(self.peak, self.now)
        time.sleep(self.seconds)
        with self._lock:
            self.now -= 1
        return SimpleNamespace(choices=[], usage=None, data=[])


def _in_threads(fn, n):
    threads = [threading.Thread(target=fn) for _ in range(n)]
    [t.start() for t in threads]
    [t.join(30) for t in threads]
    assert not any(t.is_alive() for t in threads)


@pytest.fixture()
def cap(monkeypatch):
    def set_cap(value):
        monkeypatch.setenv("SCILINK_LLM_MAX_INFLIGHT", str(value))
    return set_cap


def test_completions_to_one_model_never_exceed_the_cap(monkeypatch, cap):
    cap(3)
    gauge = Gauge()
    monkeypatch.setattr(lw.litellm, "completion", gauge)
    _in_threads(lambda: lw._completion_with_retries(0, model="bedrock/m1", messages=[]), 12)
    assert gauge.calls == 12 and gauge.peak == 3


def test_the_cap_is_per_model(monkeypatch, cap):
    cap(2)
    gauge = Gauge(seconds=0.1)
    monkeypatch.setattr(lw.litellm, "completion", gauge)
    models = iter(["m1", "m2"] * 4)
    lock = threading.Lock()

    def call():
        with lock:
            model = next(models)
        lw._completion_with_retries(0, model=model, messages=[])
    _in_threads(call, 8)
    assert gauge.peak == 4                  # two per model, two models


def test_a_retry_does_not_hold_the_slot_while_it_backs_off(monkeypatch, cap):
    cap(1)
    order, attempts = [], {"a": 0}

    class Throttled(Exception):
        status_code = 503

    def completion(**kwargs):
        who = kwargs["metadata"]
        if who == "a":
            attempts["a"] += 1
            if attempts["a"] == 1:
                raise Throttled("busy")
        order.append(who)
        return SimpleNamespace(choices=[], usage=None)

    monkeypatch.setattr(lw.litellm, "completion", completion)
    monkeypatch.setattr(lw, "_backoff_s", lambda attempt: 0.4)
    a = threading.Thread(target=lambda: lw._completion_with_retries(
        2, model="m", messages=[], metadata="a"))
    a.start()
    time.sleep(0.1)                         # a is sleeping in its backoff now
    t0 = time.time()
    lw._completion_with_retries(0, model="m", messages=[], metadata="b")
    assert time.time() - t0 < 0.3           # b did not wait for a's backoff
    a.join(5)
    assert order == ["b", "a"]


def test_the_cap_can_be_lifted(monkeypatch, cap):
    cap(0)
    gauge = Gauge()
    monkeypatch.setattr(lw.litellm, "completion", gauge)
    _in_threads(lambda: lw._completion_with_retries(0, model="m", messages=[]), 6)
    assert gauge.peak == 6


@pytest.mark.parametrize("raw,expected", [(None, llm_limiter.LLM_MAX_INFLIGHT), ("4", 4),
                                          ("0", None), ("-1", None), ("many", llm_limiter.LLM_MAX_INFLIGHT)])
def test_the_cap_comes_from_the_environment(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("SCILINK_LLM_MAX_INFLIGHT", raising=False)
    else:
        monkeypatch.setenv("SCILINK_LLM_MAX_INFLIGHT", raw)
    assert llm_limiter.llm_max_inflight() == expected


def test_embeddings_share_the_cap(monkeypatch, cap):
    cap(2)
    gauge = Gauge()
    monkeypatch.setattr(lw.litellm, "embedding", gauge)
    _in_threads(lambda: lw._embedding(model="emb", input=["x"]), 6)
    assert gauge.peak == 2


def test_the_internal_proxy_client_shares_the_cap(cap):
    cap(2)
    gauge = Gauge()
    raw = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=gauge)))
    client = portable_openai_client(raw, "proxy-model")
    _in_threads(lambda: client.chat.completions.create(model="proxy-model", messages=[]), 6)
    assert gauge.calls == 6 and gauge.peak == 2


def test_a_waiting_call_says_so(monkeypatch, cap, caplog):
    cap(1)
    monkeypatch.setattr(lw.litellm, "completion", Gauge(seconds=0.3))
    with caplog.at_level("WARNING"):
        _in_threads(lambda: lw._completion_with_retries(0, model="slow", messages=[]), 2)
    assert "Waiting for an LLM slot" in caplog.text


# ------------------------------------------------------------------ attribution

@pytest.fixture()
def ledger(tmp_path):
    led = UsageLedger(tmp_path / "usage.jsonl")
    tracing.set_usage_sink(led.record)
    yield led
    tracing.set_usage_sink(None)
    tracing.bind_session(None)


def test_calls_made_for_a_worker_are_charged_to_it(ledger, tmp_path):
    tracing.bind_session("s1")
    tracing.note_llm_call(prompt_tokens=10, completion_tokens=1, model="m")   # the session's own

    def worker(label):
        with tracing.attributed(session="s1", worker=label):
            tracing.note_llm_call(prompt_tokens=100, completion_tokens=10, model="m")
        tracing.note_llm_call(prompt_tokens=1, completion_tokens=0, model="m")  # after the block

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    s = ledger.summary()
    assert s["by_worker"]["s1/w0"] == {"calls": 1, "prompt_tokens": 100, "completion_tokens": 10}
    assert s["by_worker"]["s1/w1"] == {"calls": 1, "prompt_tokens": 100, "completion_tokens": 10}
    assert set(s["by_worker"]) == {"s1/w0", "s1/w1"}
    assert s["by_session"]["s1"]["calls"] == 3
    assert s["by_session"]["unattributed"]["calls"] == 2
    again = UsageLedger(tmp_path / "usage.jsonl")                             # a redeploy
    assert again.summary()["by_worker"] == s["by_worker"]


def test_the_block_restores_the_previous_tags():
    tracing.bind_session("outer")
    with tracing.attributed(session="inner", worker="w"):
        assert (tracing.current_session(), tracing.current_worker()) == ("inner", "w")
        with tracing.attributed(worker="w2"):
            assert (tracing.current_session(), tracing.current_worker()) == ("inner", "w2")
    assert (tracing.current_session(), tracing.current_worker()) == ("outer", None)
    tracing.bind_session(None)


def test_a_sink_without_a_worker_parameter_still_receives_every_call():
    got = []
    tracing.set_usage_sink(lambda model, p, c, lat, session: got.append((model, session)))
    try:
        with tracing.attributed(session="s", worker="w"):
            tracing.note_llm_call(prompt_tokens=1, completion_tokens=1, model="m")
    finally:
        tracing.set_usage_sink(None)
    assert got == [("m", "s")]


def test_a_fanout_branch_runs_under_the_coordinators_session():
    from scilink.agents.meta_agent.fanout import _attributed_branch
    tracing.bind_session("meta-session")
    try:
        wrapped = _attributed_branch(lambda: (tracing.current_session(), tracing.current_worker()))
    finally:
        tracing.bind_session(None)
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("tags", wrapped()))
    t.start()
    t.join()
    assert out["tags"] == ("meta-session", None)


def test_a_cancelled_worker_does_not_take_the_slot_it_waited_for(monkeypatch, cap):
    from scilink.utils.log_context import register_cancel, unregister_cancel
    from scilink.ui.output_capture import AgentStoppedError
    cap(1)
    gauge = Gauge(seconds=1.5)
    monkeypatch.setattr(lw.litellm, "completion", gauge)
    monkeypatch.setattr(llm_limiter, "_WAIT_SLICE_S", 0.1)
    holder = threading.Thread(target=lambda: lw._completion_with_retries(0, model="m", messages=[]))
    holder.start()
    time.sleep(0.2)                                  # the slot is taken
    stop, out = threading.Event(), {}

    def waiter():
        register_cancel(stop)
        try:
            lw._completion_with_retries(0, model="m", messages=[])
            out["r"] = "called"
        except AgentStoppedError:
            out["r"] = "stopped"
        finally:
            unregister_cancel()
    w = threading.Thread(target=waiter)
    w.start()
    time.sleep(0.3)
    stop.set()                                       # cancelled while waiting
    w.join(5)
    holder.join(5)
    assert out["r"] == "stopped" and gauge.calls == 1


def test_the_proxy_client_retries_transient_errors_itself_with_the_slot_released(monkeypatch, cap):
    """The SDK's own retries are off (they would sleep inside the slot); a
    503 is retried by SciLink's policy, a 400 is not."""
    import openai
    import httpx
    cap(1)
    monkeypatch.setattr(lw, "_backoff_s", lambda attempt: 0.2)

    class Raw:
        def __init__(self):
            self.max_retries = 2
            self.calls = []

    def status_error(code):
        resp = httpx.Response(code, request=httpx.Request("POST", "http://proxy/v1/chat"))
        return openai.APIStatusError("boom", response=resp, body=None)

    raw = Raw()
    seen = {"n": 0}

    def create(**kwargs):
        seen["n"] += 1
        if seen["n"] == 1:
            raise status_error(503)
        return SimpleNamespace(choices=[], usage=None)
    raw.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
    client = portable_openai_client(raw, "proxy-model")
    assert raw.max_retries == 0
    # another call gets the single slot during the backoff sleep
    order = []
    other_raw = SimpleNamespace(max_retries=2, chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: order.append("other") or SimpleNamespace(choices=[], usage=None))))
    other = portable_openai_client(other_raw, "proxy-model")
    t = threading.Thread(target=lambda: client.chat.completions.create(model="proxy-model", messages=[]))
    t.start()
    time.sleep(0.05)
    other.chat.completions.create(model="proxy-model", messages=[])
    t.join(5)
    assert seen["n"] == 2 and order == ["other"]

    def bad(**kwargs):
        raise status_error(400)
    raw.chat = SimpleNamespace(completions=SimpleNamespace(create=bad))
    client = portable_openai_client(raw, "proxy-model")
    with pytest.raises(openai.APIStatusError):
        client.chat.completions.create(model="proxy-model", messages=[])
