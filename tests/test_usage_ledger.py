"""LLM usage is metered per workspace: every call reaches one ledger, sessions
are tagged on their threads, a budget refuses new turns once spent, and a
period marker lets a control plane start the count again without losing
the records."""
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from scilink import tracing
from scilink.usage import UsageLedger, ledger_for


@pytest.fixture(autouse=True)
def _no_sink():
    tracing.set_usage_sink(None)
    tracing.bind_session(None)
    yield
    tracing.set_usage_sink(None)
    tracing.bind_session(None)


def test_ledger_counts_persists_and_reloads(tmp_path):
    led = UsageLedger(tmp_path / "usage.jsonl", budget_tokens=1000)
    led.record("bedrock/opus", 300, 50, 1.2, "s1")
    led.record("bedrock/opus", 200, 20, 0.8, None)
    s = led.summary()
    assert (s["calls"], s["prompt_tokens"], s["completion_tokens"], s["total_tokens"]) == (2, 500, 70, 570)
    assert s["by_model"]["bedrock/opus"]["calls"] == 2
    assert s["by_session"]["s1"]["prompt_tokens"] == 300 and s["by_session"]["unattributed"]["calls"] == 1
    assert s["remaining_tokens"] == 430 and not s["over_budget"]
    again = UsageLedger(tmp_path / "usage.jsonl", budget_tokens=1000)   # a redeploy
    assert again.summary()["total_tokens"] == 570


def test_budget_and_new_period(tmp_path):
    led = UsageLedger(tmp_path / "u.jsonl", budget_tokens=100)
    led.record("m", 90, 10, 0.1, "s")
    assert led.over_budget()
    s = led.new_period()
    assert s["total_tokens"] == 0 and not s["over_budget"]
    lines = [json.loads(l) for l in (tmp_path / "u.jsonl").read_text().splitlines()]
    assert lines[-1]["kind"] == "period" and lines[0]["prompt_tokens"] == 90
    assert UsageLedger(tmp_path / "u.jsonl", 100).summary()["total_tokens"] == 0   # reload honours the marker


def test_ledger_for_reads_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SCILINK_TOKEN_BUDGET", "2e6")
    monkeypatch.delenv("SCILINK_USAGE_FILE", raising=False)
    led = ledger_for(tmp_path)
    assert led.budget == 2_000_000 and led.path == tmp_path / "usage.jsonl"
    monkeypatch.setenv("SCILINK_USAGE_FILE", str(tmp_path / "elsewhere.jsonl"))
    assert ledger_for(tmp_path).path == tmp_path / "elsewhere.jsonl"


def test_sink_receives_every_call_with_its_thread_session(tmp_path):
    led = UsageLedger(tmp_path / "u.jsonl")
    tracing.set_usage_sink(led.record)
    tracing.bind_session("main-session")
    tracing.note_llm_call(latency_s=0.5, prompt_tokens=10, completion_tokens=5, model="m1")
    seen = {}

    def worker():
        seen["before"] = tracing.current_session()          # threads start untagged
        tracing.bind_session("live-run")
        tracing.note_llm_call(prompt_tokens=1, completion_tokens=1, model="m2")
    t = threading.Thread(target=worker); t.start(); t.join()
    with tracing.off_path():                                # off-path calls are still billed
        tracing.note_llm_call(prompt_tokens=2, completion_tokens=2, model="m1")
    s = led.summary()
    assert seen["before"] is None
    assert s["by_session"] == {"main-session": {"calls": 2, "prompt_tokens": 12, "completion_tokens": 7},
                               "live-run": {"calls": 1, "prompt_tokens": 1, "completion_tokens": 1}}
    assert set(s["by_model"]) == {"m1", "m2"}


def test_both_wrappers_report_the_model():
    from scilink.wrappers import litellm_wrapper, openai_wrapper
    got = []
    tracing.set_usage_sink(lambda model, p, c, s, sess: got.append((model, p, c)))
    resp = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10),
                           choices=[])
    litellm_wrapper._record_trace("bedrock/x", [], resp, 0.1)
    openai_wrapper._record_trace("gpt-x", [], resp, 0.1)
    assert got == [("bedrock/x", 7, 3), ("gpt-x", 7, 3)]


def test_server_routes_and_budget_gate(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from scilink.server.app import create_app
    from scilink.server.session_manager import WebSession
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SCILINK_TOKEN_BUDGET", "100")
    monkeypatch.delenv("SCILINK_USAGE_FILE", raising=False)
    app = create_app(tmp_path, serve_frontend=False)
    c = TestClient(app)
    assert c.get("/api/v1/usage").json()["budget_tokens"] == 100
    mgr = app.state.manager_for_user("default")
    sdir = tmp_path / "analysis_session_20260101_000000"; sdir.mkdir()
    s = WebSession(id=sdir.name, session_dir=str(sdir), mode="analyze", model="m",
                   autonomy="autonomous", agent=SimpleNamespace())
    mgr._sessions[s.id] = s
    tracing.note_llm_call(prompt_tokens=80, completion_tokens=30, model="m")   # the process sink
    u = c.get("/api/v1/usage").json()
    assert u["total_tokens"] == 110 and u["over_budget"] and (tmp_path / "usage.jsonl").exists()
    r = c.post(f"/api/v1/sessions/{s.id}/messages", json={"content": "hi"})
    assert r.status_code == 429 and "budget" in r.json()["detail"]
    assert c.post("/api/v1/usage/period").json()["total_tokens"] == 0
    assert c.post(f"/api/v1/sessions/{s.id}/messages", json={"content": "hi"}).status_code != 429
