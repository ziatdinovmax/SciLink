"""A dollar estimate for model calls on a provider's own API (Anthropic,
OpenAI, Google Gemini, Amazon Bedrock).

The estimate is LiteLLM's list price for the model id as it is called, with
cached input priced apart; a route with no known price (a base URL, another
provider) gets none, and a total says how many calls it covers.
"""
import json
import tempfile
from pathlib import Path

import pytest

from scilink import pricing, tracing
from scilink.usage import UsageLedger

BEDROCK = "bedrock/us.anthropic.claude-opus-5-5"
VENDOR = "claude-opus-5-5"
UNPRICED = "azure/gpt-6-astra"        # a deployment name: not a route with a list price
DIRECT = ("claude-opus-5-5", "gpt-6-astra", "gemini-4-argon")


@pytest.fixture(autouse=True)
def _no_sink(monkeypatch):
    # a base URL of the developer's own (a corporate gateway, an agent behind
    # one) must not decide what these tests call a provider's own API
    from scilink.wrappers import litellm_wrapper
    for names, _host, _global in litellm_wrapper._PROVIDER_BASE.values():
        for name in names:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(litellm_wrapper.litellm, "api_base", None, raising=False)
    tracing.set_usage_sink(None)
    tracing.bind_session(None)
    yield
    tracing.set_usage_sink(None)
    tracing.bind_session(None)


def _rates(key):
    import litellm
    e = litellm.model_cost[key]
    return (e["input_cost_per_token"], e["output_cost_per_token"],
            e["cache_read_input_token_cost"], e["cache_creation_input_token_cost"])


# ── the estimate ─────────────────────────────────────────────────────────

def test_a_bedrock_call_is_priced_with_cached_input_apart():
    rin, rout, rread, rwrite = _rates("us.anthropic.claude-opus-5-5")
    cost = pricing.estimate_cost(BEDROCK, prompt_tokens=1_000_000, completion_tokens=100_000,
                                 cache_read_tokens=400_000, cache_write_tokens=100_000)
    # prompt_tokens includes what the cache read and wrote
    assert cost == pytest.approx(500_000 * rin + 400_000 * rread + 100_000 * rwrite + 100_000 * rout)
    plain = pricing.estimate_cost(BEDROCK, prompt_tokens=1_000_000, completion_tokens=100_000)
    assert plain == pytest.approx(1_000_000 * rin + 100_000 * rout) and plain > cost


def test_a_geo_profile_is_priced_by_its_own_entry_not_the_base_models():
    geo, base = _rates("us.anthropic.claude-opus-5-5"), _rates("anthropic.claude-opus-5-5")
    cost = pricing.estimate_cost(BEDROCK, prompt_tokens=1_000_000)
    assert cost == pytest.approx(1_000_000 * geo[0])
    if geo[0] != base[0]:
        # (LiteLLM 1.81's get_model_info resolved the id to the base model's
        # entry; the figure never came from there)
        assert cost != pytest.approx(1_000_000 * base[0])


def test_a_route_with_no_known_price_gets_no_estimate():
    assert pricing.priced(BEDROCK) and not pricing.priced(None) and not pricing.priced("")
    for model in (UNPRICED, "vertex_ai/gemini-4-argon", "ollama/llama3", "my-own-model"):
        assert not pricing.priced(model), model
        assert pricing.estimate_cost(model, 1000, 100) is None, model
    assert pricing.estimate_cost("bedrock/no-such-model", 1000, 100) is None   # not in the table
    assert pricing.estimate_cost("gpt-99-no-such-model", 1000, 100) is None
    assert pricing.estimate_cost(None, 1000, 100) is None


@pytest.mark.parametrize("model", DIRECT)
def test_a_providers_own_api_is_priced_bare_or_prefixed(model):
    """Anthropic, OpenAI and Gemini called directly: the name as a person
    types it and the name the wrapper sends give one figure, LiteLLM's."""
    import litellm
    from scilink.wrappers.litellm_wrapper import _normalize_model_name
    sent = _normalize_model_name(model)
    assert pricing.route(model) == pricing.route(sent) == sent.split("/")[0]
    assert pricing.route(model) in pricing.PRICED_ROUTES and pricing.priced(model) and pricing.priced(sent)
    cost = pricing.estimate_cost(model, prompt_tokens=100_000, completion_tokens=10_000, cache_read_tokens=40_000)
    assert cost == pytest.approx(pricing.estimate_cost(sent, 100_000, 10_000, cache_read_tokens=40_000))
    assert cost == pytest.approx(sum(litellm.cost_per_token(
        model=sent, prompt_tokens=100_000, completion_tokens=10_000, cache_read_input_tokens=40_000)))
    entry = litellm.model_cost[sent if sent in litellm.model_cost else model]
    # cached input is priced apart, below plain input
    assert cost == pytest.approx(60_000 * entry["input_cost_per_token"]
                                 + 40_000 * entry["cache_read_input_token_cost"]
                                 + 10_000 * entry["output_cost_per_token"])
    assert cost < pricing.estimate_cost(model, 100_000, 10_000)


@pytest.mark.parametrize("model", DIRECT)
def test_a_direct_model_sent_through_a_base_url_gets_no_estimate(model):
    assert not pricing.priced(model, proxied=True)
    assert pricing.estimate_cost(model, 1000, 100, proxied=True) is None
    before = tracing.llm_counters()
    tracing.note_llm_call(1.0, 10_000, 1_000, model=model, proxied=True, cost_usd=0.5)
    tracing.note_llm_call(1.0, 10_000, 1_000, model=model)
    d = tracing.counters_delta(before)
    assert d["calls"] == 2 and d["priced_calls"] == 1
    assert d["cost_usd"] == pytest.approx(pricing.estimate_cost(model, 10_000, 1_000))


@pytest.mark.parametrize("model, variable, own", [
    ("claude-opus-5-5", "ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
    ("claude-opus-5-5", "ANTHROPIC_API_BASE", "https://api.anthropic.com"),
    ("gpt-6-astra", "OPENAI_BASE_URL", "https://api.openai.com/v1"),
    ("gpt-6-astra", "OPENAI_API_BASE", "https://api.openai.com/v1"),
    ("gemini-4-argon", "GEMINI_API_BASE", "https://generativelanguage.googleapis.com"),
])
def test_a_base_url_the_environment_sets_is_a_proxy_too(monkeypatch, model, variable, own):
    """A call that passes no base URL still goes wherever the provider's
    client reads one from: that is a proxy's price, not the provider's."""
    assert pricing.priced(model)
    monkeypatch.setenv(variable, "http://proxy.internal/v1")
    assert not pricing.priced(model) and pricing.estimate_cost(model, 1000, 100) is None
    for other in DIRECT:
        if pricing.route(other) != pricing.route(model):
            assert pricing.priced(other)            # another provider's variable changes nothing
    assert pricing.priced(BEDROCK)
    monkeypatch.setenv(variable, own)               # the provider's own host is no proxy
    assert pricing.priced(model)


@pytest.mark.parametrize("model, variable, own_host", [
    ("claude-opus-5-5", "ANTHROPIC_BASE_URL", "api.anthropic.com"),
    ("gpt-6-astra", "OPENAI_BASE_URL", "api.openai.com"),
    ("gemini-4-argon", "GEMINI_API_BASE", "generativelanguage.googleapis.com"),
    (BEDROCK, "AWS_BEDROCK_RUNTIME_ENDPOINT", "bedrock-runtime.us-east-1.amazonaws.com"),
])
def test_the_providers_own_endpoint_is_told_by_the_urls_host(monkeypatch, model, variable, own_host):
    """A gateway whose address only CONTAINS the provider's host is a
    gateway; the provider's host in another case, with a port or a path, or
    a subdomain of it, is the provider."""
    for gateway in (f"https://{own_host}.gw.example.net", f"https://egress.corp/https://{own_host}/v1",
                    f"https://example.net/?to={own_host}", f"https://not{own_host.split('.', 2)[-1]}",
                    "not a url"):
        monkeypatch.setenv(variable, gateway)
        assert not pricing.priced(model), gateway
        assert pricing.estimate_cost(model, 1000, 100) is None, gateway
    for own in (f"https://{own_host}", f"https://{own_host.upper()}/v1", f"https://{own_host}:443/v1/",
                own_host, f"https://eu.{own_host}/v1"):
        monkeypatch.setenv(variable, own)
        assert pricing.priced(model), own


def test_the_route_and_the_price_read_one_rule(monkeypatch):
    """The Responses bridge and the estimate ask the same function whether
    an endpoint is OpenAI's own, so a fix to one cannot leave the other."""
    from scilink.wrappers import litellm_wrapper as w
    for base, own in (("https://api.openai.com.gw.example.net/v1", False), ("https://API.OPENAI.COM/v1", True)):
        monkeypatch.setenv("OPENAI_BASE_URL", base)
        assert (w._custom_openai_base() is None) is own
        assert pricing.priced("gpt-6-astra") is own
        assert (w._responses_api_model("gpt-6-astra") == "openai/responses/gpt-6-astra") is own


@pytest.mark.parametrize("model", ["o3", "o1", "o4-mini"])
def test_a_bare_openai_name_the_name_rule_does_not_know_is_still_priced(model):
    """LiteLLM sends these to OpenAI as they are; the route is read from
    its table when the wrapper's prefix rule leaves the name bare."""
    import litellm
    if model not in litellm.model_cost:
        pytest.skip(f"{model} is not in this LiteLLM's table")
    assert pricing.route(model) == "openai" and pricing.priced(model)
    assert pricing.estimate_cost(model, 1000, 100) == pytest.approx(
        sum(litellm.cost_per_token(model=model, prompt_tokens=1000, completion_tokens=100)))
    assert not pricing.priced(model, proxied=True)
    # a bare id the table files under another route stays unpriced
    assert not pricing.priced("my-own-model")


def test_litellms_global_base_url_is_a_proxy_for_every_direct_provider(monkeypatch):
    import litellm
    monkeypatch.setattr(litellm, "api_base", "http://proxy.internal/v1", raising=False)
    assert not any(pricing.priced(m) for m in DIRECT)
    assert pricing.priced(BEDROCK)


@pytest.mark.parametrize("model", ["anthropic/claude-opus-5-5", "gemini/gemini-4-argon", "openai/gpt-6-astra"])
def test_a_direct_call_counts_litellms_own_figure(model):
    """The whole path with no network: LiteLLM prices its own (mock) reply
    and the counters take that figure."""
    from scilink.wrappers import litellm_wrapper
    before = tracing.llm_counters()
    response = litellm_wrapper.litellm_completion(
        model=model, messages=[{"role": "user", "content": "hello there"}], mock_response="hi", api_key="x")
    figure = response._hidden_params["response_cost"]
    d = tracing.counters_delta(before)
    assert figure > 0 and d["calls"] == 1 and d["priced_calls"] == 1 and d["cost_usd"] == pytest.approx(figure)
    before = tracing.llm_counters()
    litellm_wrapper.litellm_completion(
        model=model, messages=[{"role": "user", "content": "hello there"}], mock_response="hi", api_key="x",
        api_base="http://proxy.internal/v1")
    d = tracing.counters_delta(before)
    assert d["calls"] == 1 and d["priced_calls"] == 0 and d["cost_usd"] == 0


def test_the_figure_says_how_many_calls_it_covers():
    assert pricing.cost_text(1.234, calls=6, priced_calls=6) == "≈ $1.23"
    assert pricing.cost_text(1.234, calls=6, priced_calls=4) == "≈ $1.23 for 4 of 6 calls"
    assert pricing.cost_text(1234.5, 3, 3) == "≈ $1,234.50"
    assert pricing.cost_text(0.0004, 1, 1) == "≈ <$0.01"
    # no call had a price: nothing, never "$0.00"
    assert pricing.cost_text(0.0, calls=6, priced_calls=0) == ""
    assert pricing.cost_text(None, None, None) == ""


# ── the counters every surface reads ─────────────────────────────────────

def test_the_counters_carry_the_estimate_and_the_calls_it_covers():
    before = tracing.llm_counters()
    tracing.note_llm_call(1.0, 10_000, 1_000, model=BEDROCK)
    tracing.note_llm_call(1.0, 10_000, 1_000, model=BEDROCK, cost_usd=0.25)     # the caller's own figure
    tracing.note_llm_call(1.0, 10_000, 1_000, model=UNPRICED)
    tracing.note_llm_call(1.0, 10_000, 1_000, model=UNPRICED, cost_usd=0.25)    # no price, whatever is passed
    d = tracing.counters_delta(before)
    assert d["calls"] == 4 and d["priced_calls"] == 2
    assert d["cost_usd"] == pytest.approx(pricing.estimate_cost(BEDROCK, 10_000, 1_000) + 0.25)


# ── a proxied call, and LiteLLM's own figure ─────────────────────────────

def test_a_bedrock_id_sent_through_a_base_url_gets_no_estimate():
    """The prefix alone decided, so a proxy's ``bedrock/...``
    alias was priced at Bedrock's list price."""
    assert not pricing.priced(BEDROCK, proxied=True)
    assert pricing.estimate_cost(BEDROCK, 1000, 100, proxied=True) is None
    before = tracing.llm_counters()
    tracing.note_llm_call(1.0, 10_000, 1_000, model=BEDROCK, proxied=True)
    tracing.note_llm_call(1.0, 10_000, 1_000, model=BEDROCK, proxied=True, cost_usd=0.5)   # a figure is not taken either
    d = tracing.counters_delta(before)
    assert d["calls"] == 2 and d["priced_calls"] == 0 and d["cost_usd"] == 0


def _response(prompt=10_000, completion=1_000, response_cost=None, **usage):
    from types import SimpleNamespace
    r = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, **usage),
                        choices=[])
    if response_cost is not None:
        r._hidden_params = {"response_cost": response_cost}
    return r


def test_the_openai_wrapper_is_the_base_url_path_and_prices_nothing():
    from scilink.wrappers import openai_wrapper
    before = tracing.llm_counters()
    openai_wrapper._record_trace(BEDROCK, [], _response(), 1.0)
    d = tracing.counters_delta(before)
    assert d["calls"] == 1 and d["priced_calls"] == 0 and d["cost_usd"] == 0


def test_the_litellm_wrapper_passes_litellms_own_figure_and_says_when_proxied():
    from scilink.wrappers import litellm_wrapper
    before = tracing.llm_counters()
    litellm_wrapper._record_trace(BEDROCK, [], _response(response_cost=0.4321), 1.0)        # LiteLLM's figure
    litellm_wrapper._record_trace(BEDROCK, [], _response(), 1.0)                            # none: from the tokens
    litellm_wrapper._record_trace(BEDROCK, [], _response(response_cost=0.4321), 1.0, proxied=True)
    litellm_wrapper._record_trace(UNPRICED, [], _response(response_cost=0.4321), 1.0)      # not a priced route
    d = tracing.counters_delta(before)
    assert d["calls"] == 4 and d["priced_calls"] == 2
    assert d["cost_usd"] == pytest.approx(0.4321 + pricing.estimate_cost(BEDROCK, 10_000, 1_000))
    assert pricing.response_cost(BEDROCK, _response(response_cost=0)) is None                # zero is no figure
    assert pricing.response_cost(BEDROCK, _response(response_cost=True)) is None


def test_a_call_with_no_figure_from_litellm_is_priced_with_its_cached_input_apart():
    """With no figure on the response the estimate is made from the token
    counts, and the cached part of the prompt is read from the response in
    either spelling a provider reports it in."""
    from types import SimpleNamespace
    from scilink.wrappers import litellm_wrapper
    aware = pricing.estimate_cost(BEDROCK, 10_000, 1_000, cache_read_tokens=4_000, cache_write_tokens=1_000)
    plain = pricing.estimate_cost(BEDROCK, 10_000, 1_000)
    assert aware != pytest.approx(plain)
    details = SimpleNamespace(cached_tokens=4_000, cache_creation_tokens=1_000)
    for usage in ({"prompt_tokens_details": details},
                  {"cache_read_input_tokens": 4_000, "cache_creation_input_tokens": 1_000}):
        before = tracing.llm_counters()
        litellm_wrapper._record_trace(BEDROCK, [], _response(**usage), 1.0)
        d = tracing.counters_delta(before)
        assert d["priced_calls"] == 1 and d["cost_usd"] == pytest.approx(aware)
    before = tracing.llm_counters()
    tracing.note_llm_call(1.0, 10_000, 1_000, model=BEDROCK, cache_read_tokens=4_000, cache_write_tokens=1_000)
    assert tracing.counters_delta(before)["cost_usd"] == pytest.approx(aware)


def test_a_model_the_table_does_not_hold_is_looked_up_once(monkeypatch):
    import litellm
    pricing._in_table.cache_clear()
    calls = []
    real = litellm.cost_per_token

    def counting(*a, **kw):
        calls.append(kw.get("model"))
        return real(*a, **kw)

    monkeypatch.setattr(litellm, "cost_per_token", counting)
    for _ in range(5):
        assert pricing.estimate_cost("bedrock/no-such-model", 1000, 100) is None
    assert calls == ["bedrock/no-such-model"]
    pricing._in_table.cache_clear()


def test_the_words_are_the_shared_vocabularys():
    from scilink.ui import vocabulary as V
    assert V.as_json()["cost_words"] == V.COST_WORDS and set(V.COST_WORDS) == {"figure", "partial", "note"}
    assert pricing.cost_text(1.0, 2, 2) == V.COST_WORDS["figure"].format(amount="$1.00")
    assert pricing.cost_text(1.0, 3, 2) == V.COST_WORDS["partial"].format(figure="≈ $1.00", priced=2, calls=3)


# ── the ledger behind the web interface ──────────────────────────────────

def test_the_ledger_records_sums_and_reloads_the_estimate(tmp_path):
    led = UsageLedger(tmp_path / "usage.jsonl")
    tracing.set_usage_sink(led.record)
    tracing.bind_session("s1")
    tracing.note_llm_call(1.0, 10_000, 1_000, model=BEDROCK, cost_usd=0.04)
    tracing.note_llm_call(1.0, 10_000, 1_000, model=BEDROCK)
    tracing.note_llm_call(1.0, 500, 50, model=UNPRICED)
    expected = 0.04 + pricing.estimate_cost(BEDROCK, 10_000, 1_000)

    def check(summary):
        assert summary["calls"] == 3 and summary["priced_calls"] == 2
        assert summary["cost_usd"] == pytest.approx(expected, abs=2e-6)
        assert summary["cost_text"] == pricing.cost_text(expected, 3, 2) and "of 3 calls" in summary["cost_text"]
        assert summary["by_model"][BEDROCK]["cost_text"] == pricing.cost_text(expected, 2, 2)
        assert summary["by_model"][UNPRICED]["cost_text"] == "" and summary["by_model"][UNPRICED]["priced_calls"] == 0
        assert summary["by_session"]["s1"]["priced_calls"] == 2

    check(led.summary())
    records = [json.loads(line) for line in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert records[0]["cost_usd"] == pytest.approx(0.04)
    assert "cost_usd" in records[1] and "cost_usd" not in records[2]     # no price, no figure
    check(UsageLedger(tmp_path / "usage.jsonl").summary())                # a restart keeps it


def test_records_written_before_the_estimate_are_counted_as_unpriced(tmp_path):
    path = tmp_path / "usage.jsonl"
    path.write_text(json.dumps({"ts": 1.0, "model": BEDROCK, "prompt_tokens": 10_000,
                                "completion_tokens": 1_000, "latency_s": 1.0, "session": "old"}) + "\n")
    led = UsageLedger(path)
    led.record(BEDROCK, 10_000, 1_000, 1.0, "new", cost_usd=0.05)
    s = led.summary()
    assert s["calls"] == 2 and s["priced_calls"] == 1 and s["cost_text"] == "≈ $0.05 for 1 of 2 calls"
    assert s["by_session"]["old"]["cost_text"] == "" and s["by_session"]["new"]["cost_text"] == "≈ $0.05"


def test_a_new_period_starts_the_estimate_again(tmp_path):
    led = UsageLedger(tmp_path / "usage.jsonl")
    led.record(BEDROCK, 10_000, 1_000, 1.0, "s", cost_usd=0.05)
    assert led.summary()["cost_text"] == "≈ $0.05"
    s = led.new_period()
    assert s["cost_usd"] == 0 and s["priced_calls"] == 0 and s["cost_text"] == ""


def test_a_sessions_figure_is_its_whole_life_across_billing_periods(tmp_path):
    """The sidebar read the session's row of the CURRENT
    period, so a period reset mid-session dropped what it had cost."""
    led = UsageLedger(tmp_path / "usage.jsonl")
    led.record(BEDROCK, 10_000, 1_000, 1.0, "s1", cost_usd=0.20)
    led.record(BEDROCK, 10_000, 1_000, 1.0, "s2", cost_usd=0.05)
    led.new_period()
    led.record(BEDROCK, 10_000, 1_000, 1.0, "s1", cost_usd=0.10)
    led.record(UNPRICED, 10_000, 1_000, 1.0, "s1")

    def check(ledger):
        assert ledger.summary()["by_session"]["s1"]["cost_text"] == "≈ $0.10 for 1 of 2 calls"   # this period
        s1 = ledger.session_cost("s1")
        assert s1["calls"] == 3 and s1["priced_calls"] == 2 and s1["cost_usd"] == pytest.approx(0.30)
        assert s1["cost_text"] == "≈ $0.30 for 2 of 3 calls"
        assert ledger.session_cost("s2")["cost_text"] == "≈ $0.05"
        assert ledger.session_cost("nobody") == {"session": "nobody", "calls": 0, "priced_calls": 0,
                                                 "cost_usd": 0.0, "cost_text": ""}

    check(led)
    check(UsageLedger(tmp_path / "usage.jsonl"))                     # a restart reads it back


def test_the_session_cost_route_serves_one_row(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    from scilink.server.app import create_app
    from scilink.server.session_manager import WebSession
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SCILINK_USAGE_FILE", raising=False)
    monkeypatch.delenv("SCILINK_TOKEN_BUDGET", raising=False)
    app = create_app(tmp_path, serve_frontend=False)
    c = TestClient(app)
    sdir = tmp_path / "meta_session_20260101_000000"; sdir.mkdir()
    s = WebSession(id=sdir.name, session_dir=str(sdir), mode="meta", model=BEDROCK,
                   autonomy="autonomous", agent=SimpleNamespace())
    app.state.manager_for_user("default")._sessions[s.id] = s
    assert c.get(f"/api/v1/sessions/{s.id}/cost").json()["cost_text"] == ""
    tracing.bind_session(s.id)
    tracing.note_llm_call(1.0, 1_000_000, 100_000, model=BEDROCK)        # the process sink
    row = c.get(f"/api/v1/sessions/{s.id}/cost").json()
    assert row["cost_text"] == pricing.cost_text(pricing.estimate_cost(BEDROCK, 1_000_000, 100_000), 1, 1)
    assert set(row) == {"session", "calls", "priced_calls", "cost_usd", "cost_text"}
    assert c.post("/api/v1/usage/period").json()["cost_text"] == ""       # a new period...
    assert c.get(f"/api/v1/sessions/{s.id}/cost").json()["cost_text"] == row["cost_text"]   # ...keeps the session's
    assert c.get("/api/v1/sessions/no_such_session/cost").status_code == 404


def test_the_usage_route_serves_the_figure(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from scilink.server.app import create_app
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SCILINK_USAGE_FILE", raising=False)
    monkeypatch.delenv("SCILINK_TOKEN_BUDGET", raising=False)
    c = TestClient(create_app(tmp_path, serve_frontend=False))
    assert c.get("/api/v1/usage").json()["cost_text"] == ""
    tracing.note_llm_call(1.0, 1_000_000, 100_000, model=BEDROCK)       # the process sink
    u = c.get("/api/v1/usage").json()
    assert u["cost_text"] == pricing.cost_text(pricing.estimate_cost(BEDROCK, 1_000_000, 100_000), 1, 1)
    assert u["cost_text"].startswith("≈ $") and u["by_model"][BEDROCK]["cost_text"] == u["cost_text"]


# ── the terminal ─────────────────────────────────────────────────────────

def test_the_turn_line_shows_the_estimate_only_when_a_call_had_a_price():
    from scilink.cli.shell.render import Renderer
    priced = {"calls": 3, "prompt_tokens": 12_000, "completion_tokens": 900, "cost_usd": 0.0731, "priced_calls": 3}
    assert Renderer.accounting_line(priced, 12.0) == "· 3 LLM calls · 12,000 tokens in · 900 out · ≈ $0.07 · 12s"
    partial = dict(priced, priced_calls=2)
    assert "≈ $0.07 for 2 of 3 calls" in Renderer.accounting_line(partial, 12.0)
    unpriced = {"calls": 3, "prompt_tokens": 12_000, "completion_tokens": 900, "cost_usd": 0.0, "priced_calls": 0}
    assert Renderer.accounting_line(unpriced, 12.0) == "· 3 LLM calls · 12,000 tokens in · 900 out · 12s"
    assert Renderer.accounting_line({"calls": 3, "prompt_tokens": 1, "completion_tokens": 1}, 1.0).count("$") == 0
    assert Renderer.accounting_line(None, 2.0) == "· 2s"


def test_cost_command_ends_with_the_estimate_when_a_call_had_a_price():
    from scilink.cli.shell.commands import cost_line
    totals = {"calls": 4, "seconds": 20.0, "prompt_tokens": 50_000, "completion_tokens": 2_000,
              "cost_usd": 0.31, "priced_calls": 4}
    assert cost_line(totals) == "This session: 4 LLM calls · 50,000 tokens in · 2,000 out · 20s in model calls · ≈ $0.31"
    unpriced = {"calls": 1, "seconds": 20.0, "prompt_tokens": 50_000, "completion_tokens": 2_000}
    assert cost_line(unpriced) == "This session: 1 LLM call · 50,000 tokens in · 2,000 out · 20s in model calls"
    assert "$" not in cost_line({"calls": 0})


def test_the_shell_totals_cover_calls_made_outside_a_turn(tmp_path, monkeypatch):
    """Live: naming the session is a call after the turn (31,158 tokens, more
    than the cached turn), and ``/cost`` left it out of the session's total."""
    import io
    from contextlib import ExitStack
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from rich.console import Console
    from scilink.cli.shell.shell import Shell
    from shell_fakes import FakeAdapter, make_args
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    adapter = FakeAdapter(ask=False)
    with ExitStack() as stack:
        pipe = stack.enter_context(create_pipe_input())
        shell = Shell(adapter, make_args(adapter, ["--yes"]), pt_input=pipe, pt_output=DummyOutput(),
                      console=Console(file=io.StringIO(), force_terminal=False, width=100))
        tracing.note_llm_call(1.0, 5_000, 100, model=BEDROCK)                 # before: not this session's
        shell._usage_start = tracing.llm_counters()
        assert shell.totals["calls"] == 0 and shell.totals["cost_usd"] == 0
        tracing.note_llm_call(2.0, 10_000, 1_000, model=BEDROCK)              # a turn's call
        tracing.note_llm_call(1.0, 30_000, 20, model=BEDROCK)                 # the naming call, after the turn
        t = shell.totals
        # /resume: "this session" is the resumed one, not what the last one spent
        monkeypatch.setattr("scilink.cli.shell.shell.resolve_session", lambda target, mode: tmp_path / "other")
        monkeypatch.setattr("scilink.cli.shell.shell.register_session", lambda path, mode: None)
        shell.creds = None
        shell.resume("other")
        after_resume = shell.totals
    assert after_resume["calls"] == 0 and after_resume["cost_usd"] == 0
    assert t["calls"] == 2 and t["priced_calls"] == 2 and t["prompt_tokens"] == 40_000
    assert t["cost_usd"] == pytest.approx(pricing.estimate_cost(BEDROCK, 10_000, 1_000)
                                          + pricing.estimate_cost(BEDROCK, 30_000, 20))
