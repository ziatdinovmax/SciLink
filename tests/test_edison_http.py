"""Edison Scientific's platform is called over HTTP by SciLink's own client
(``lit_agents/_edison_http``): the literature tools import no Edison
package, so the LiteLLM version those packages pin (and a version skew
inside that chain, which once took the whole UI down on Windows) no longer
reaches SciLink.

No network: Edison is an ``httpx.MockTransport`` that speaks the protocol
read from ``edison-client``.
"""
import json
import subprocess
import sys

import httpx
import pytest

from scilink.agents.lit_agents import _edison_http as E

KEY = "test-key-not-a-real-one"


def _task_payload(status="success", **answer):
    return {
        "id": "11111111-2222-3333-4444-555555555555", "status": status, "task": "Has anyone done X?",
        "crow": "job-futurehouse-paperqa3", "user": "someone", "created_at": "2026-11-23T10:00:00Z",
        "share_status": "private", "metadata": {"environment_name": "env", "agent_name": "agent"},
        "environment_frame": {"state": {"state": {
            "response": {"answer": answer or {"answer": "Yes.", "formatted_answer": "Yes [1].",
                                                "answer_reasoning": "because", "has_successful_answer": True}},
            "info": {"usage": {"total_cost": 0.12, "total_queries": 3}}}}},
    }


class FakeEdison:
    """Edison's side of the protocol, with a record of what it was sent."""

    def __init__(self, payloads=None, refuse_key=False):
        self.requests, self.logins, self.valid_token = [], 0, None
        self.payloads = list(payloads or [_task_payload()])
        self.refuse_key = refuse_key

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, dict(request.url.params), body,
                              request.headers.get("authorization")))
        if request.url.path == "/auth/login":
            if self.refuse_key or body != {"api_key": KEY}:
                return httpx.Response(403, json={"detail": "Forbidden"})
            self.logins += 1
            self.valid_token = f"token-{self.logins}"
            return httpx.Response(200, json={"access_token": self.valid_token, "expires_in": 300})
        if request.headers.get("authorization") != f"Bearer {self.valid_token}":
            return httpx.Response(401, json={"detail": "bad token"})
        if request.method == "POST" and request.url.path == "/v0.1/crows":
            return httpx.Response(200, json={"trajectory_id": "11111111-2222-3333-4444-555555555555"})
        if request.method == "GET" and request.url.path.startswith("/v0.1/trajectories/"):
            payload = self.payloads[0] if len(self.payloads) == 1 else self.payloads.pop(0)
            return httpx.Response(200, json=payload)
        return httpx.Response(404, json={"detail": "no such route"})


def _client(fake):
    return E.EdisonClient(api_key=KEY, transport=httpx.MockTransport(fake))


# ── the two calls ────────────────────────────────────────────────────────

def test_it_logs_in_with_the_key_and_sends_the_token_after():
    fake = FakeEdison()
    client = _client(fake)
    assert fake.requests[0][:2] == ("POST", "/auth/login") and fake.requests[0][3] == {"api_key": KEY}
    task_id = client.create_task({"name": E.JobNames.LITERATURE, "query": "Has anyone done X?"})
    assert task_id == "11111111-2222-3333-4444-555555555555"
    method, path, _, body, auth = fake.requests[-1]
    assert (method, path, auth) == ("POST", "/v0.1/crows", "Bearer token-1")
    # the task as edison-client's TaskRequest serialises it
    assert body == {"id": None, "project_id": None, "name": "job-futurehouse-paperqa3",
                    "query": "Has anyone done X?", "runtime_config": None}
    assert KEY not in json.dumps([r[3] for r in fake.requests[1:]])


def test_a_job_may_be_named_by_its_own_string():
    fake = FakeEdison()
    _client(fake).create_task({"name": "job-futurehouse-paperqa3-precedent", "query": "q"})
    assert fake.requests[-1][3]["name"] == "job-futurehouse-paperqa3-precedent"
    assert E.JobNames.PRECEDENT.value == "job-futurehouse-paperqa3-precedent"


@pytest.mark.parametrize("bad", [{"name": E.JobNames.LITERATURE, "query": ""}, {"query": "q"},
                                 {"name": E.JobNames.LITERATURE, "query": 3}])
def test_a_task_without_a_name_or_a_query_is_not_sent(bad):
    fake = FakeEdison()
    client = _client(fake)
    with pytest.raises(ValueError):
        client.create_task(bad)
    assert [r[1] for r in fake.requests] == ["/auth/login"]


def test_a_fetched_task_is_read_as_the_agents_read_it():
    fake = FakeEdison()
    task = _client(fake).get_task("11111111-2222-3333-4444-555555555555")
    assert fake.requests[-1][1] == "/v0.1/trajectories/11111111-2222-3333-4444-555555555555"
    assert fake.requests[-1][2] == {"history": "false", "lite": "false"}
    assert task.status == "success" and task.formatted_answer == "Yes [1]." and task.answer == "Yes."
    assert task.has_successful_answer is True and task.answer_reasoning == "because"
    assert task.query == "Has anyone done X?" and task.job_name == "job-futurehouse-paperqa3"
    assert task.total_cost == 0.12 and task.total_queries == 3
    assert task.environment_name == "env" and task.agent_name == "agent"
    # what the agents ask for with a default, and the record they keep
    assert getattr(task, "search_results", []) == [] and getattr(task, "sources", []) == []
    dumped = json.loads(task.model_dump_json())
    assert dumped["formatted_answer"] == "Yes [1]." and dumped["status"] == "success"
    assert set(dumped) == set(E.TaskStatus._FIELDS)


def test_a_task_still_running_has_a_status_and_no_answer():
    running = {"id": "t", "status": "in progress", "task": "q", "crow": "job-futurehouse-paperqa3"}
    task = _client(FakeEdison([running])).get_task("t")
    assert task.status == "in progress" and task.formatted_answer is None and task.answer is None
    assert task.has_successful_answer is None and task.total_cost is None
    # a reply with nothing usable is a status of "", never an exception
    assert E.TaskStatus({}).status == "" and E.TaskStatus(None).status == ""
    assert E.TaskStatus({"environment_frame": {"state": None}}).answer is None


def test_an_answer_at_the_top_of_the_state_is_read_when_there_is_no_other():
    payload = {"id": "t", "status": "success", "task": "q", "crow": "job",
               "environment_frame": {"state": {"state": {"answer": "CCO"}}}}
    task = E.TaskStatus(payload)
    # the one text is also what the literature callers read
    assert task.answer == "CCO" and task.formatted_answer == "CCO"
    assert E.TaskStatus(_task_payload()).answer == "Yes."        # the paper-search shape wins


# ── what can go wrong ────────────────────────────────────────────────────

def test_a_refused_key_is_reported_at_construction_without_the_key():
    with pytest.raises(PermissionError) as err:
        _client(FakeEdison(refuse_key=True))
    assert KEY not in str(err.value) and "API key" in str(err.value)
    with pytest.raises(ValueError):
        E.EdisonClient(api_key="")


def test_a_stale_token_is_renewed_once_and_the_request_repeated():
    fake = FakeEdison()
    client = _client(fake)
    fake.valid_token = "expired-on-the-server"                    # the token the client holds went stale
    task = client.get_task("t")
    assert task.status == "success" and fake.logins == 2
    assert [r[1] for r in fake.requests] == ["/auth/login", "/v0.1/trajectories/t",
                                             "/auth/login", "/v0.1/trajectories/t"]
    assert [r[4] for r in fake.requests][1::2] == ["Bearer token-1", "Bearer token-2"]


def test_a_request_denied_after_a_fresh_login_is_a_permission_error():
    def always_denied(request):
        if request.url.path == "/auth/login":
            return httpx.Response(200, json={"access_token": "t"})
        return httpx.Response(403, json={"detail": "Forbidden"})

    client = E.EdisonClient(api_key=KEY, transport=httpx.MockTransport(always_denied))
    with pytest.raises(PermissionError):
        client.create_task({"name": E.JobNames.MOLECULES, "query": "q"})


def test_an_error_status_carries_edisons_reason():
    def broken(request):
        if request.url.path == "/auth/login":
            return httpx.Response(200, json={"access_token": "t"})
        return httpx.Response(422, json={"detail": "unknown job"})

    client = E.EdisonClient(api_key=KEY, transport=httpx.MockTransport(broken))
    with pytest.raises(E.EdisonError) as err:
        client.create_task({"name": "job-nobody-knows", "query": "q"})
    assert "422" in str(err.value) and "unknown job" in str(err.value) and KEY not in str(err.value)


def test_a_request_that_does_not_get_through_is_tried_again(monkeypatch):
    monkeypatch.setattr(E.time, "sleep", lambda s: None)
    state = {"failures": 2, "seen": 0}

    def flaky(request):
        if request.url.path == "/auth/login":
            return httpx.Response(200, json={"access_token": "t"})
        state["seen"] += 1
        if state["failures"]:
            state["failures"] -= 1
            raise httpx.ConnectError("connection reset", request=request)
        return httpx.Response(200, json=_task_payload())

    client = E.EdisonClient(api_key=KEY, transport=httpx.MockTransport(flaky))
    assert client.get_task("t").status == "success" and state["seen"] == 3

    state.update(failures=E.READ_ATTEMPTS, seen=0)
    with pytest.raises(E.EdisonError) as err:
        client.get_task("t")
    assert state["seen"] == E.READ_ATTEMPTS and KEY not in str(err.value)


def _scripted(script):
    """A platform whose replies to everything but the login are scripted:
    each entry is a status code, an exception class, or a response."""
    seen = []

    def handler(request):
        if request.url.path == "/auth/login":
            return httpx.Response(200, json={"access_token": "t"})
        seen.append((request.method, request.url.path))
        step = script.pop(0) if len(script) > 1 else script[0]
        if isinstance(step, type) and issubclass(step, Exception):
            raise step("scripted", request=request)
        if isinstance(step, httpx.Response):
            return step
        body = {"trajectory_id": "tid"} if request.method == "POST" else _task_payload()
        return httpx.Response(step, json=body if step < 400 else {"detail": "not now"})
    return handler, seen


@pytest.mark.parametrize("lost", [httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteTimeout])
def test_a_task_whose_reply_was_lost_is_never_created_twice(monkeypatch, lost):
    """The platform may have accepted the task, and every
    further POST would start (and bill) another one."""
    monkeypatch.setattr(E.time, "sleep", lambda s: None)
    handler, seen = _scripted([lost])
    client = E.EdisonClient(api_key=KEY, transport=httpx.MockTransport(handler))
    with pytest.raises(E.EdisonError) as err:
        client.create_task({"name": E.JobNames.LITERATURE, "query": "q"})
    assert seen == [("POST", "/v0.1/crows")]
    assert "may have arrived" in str(err.value) and "not sent again" in str(err.value)


@pytest.mark.parametrize("refusal", [429, 500, 502, 503])
def test_a_task_creation_the_platform_answers_with_an_error_is_not_repeated(monkeypatch, refusal):
    monkeypatch.setattr(E.time, "sleep", lambda s: None)
    handler, seen = _scripted([refusal])
    client = E.EdisonClient(api_key=KEY, transport=httpx.MockTransport(handler))
    with pytest.raises(E.EdisonError) as err:
        client.create_task({"name": "job", "query": "q"})
    assert len(seen) == 1 and str(refusal) in str(err.value)


@pytest.mark.parametrize("never_sent", [httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout])
def test_a_task_creation_that_never_left_is_tried_again(monkeypatch, never_sent):
    monkeypatch.setattr(E.time, "sleep", lambda s: None)
    handler, seen = _scripted([never_sent, never_sent, 200])
    client = E.EdisonClient(api_key=KEY, transport=httpx.MockTransport(handler))
    assert client.create_task({"name": "job", "query": "q"}) == "tid" and len(seen) == E.ATTEMPTS
    handler, seen = _scripted([never_sent])
    client = E.EdisonClient(api_key=KEY, transport=httpx.MockTransport(handler))
    with pytest.raises(E.EdisonError) as err:
        client.create_task({"name": "job", "query": "q"})
    assert len(seen) == E.ATTEMPTS and "Could not reach Edison" in str(err.value)
    assert "may have arrived" not in str(err.value)


@pytest.mark.parametrize("blip", [502, 500, 503, 429, httpx.ReadTimeout, httpx.RemoteProtocolError])
def test_one_bad_reply_does_not_end_a_search_that_is_being_polled(monkeypatch, blip):
    """A single 502 on a poll raised, the polling loop gave
    up, and the job ran on (and was billed) with its answer lost."""
    monkeypatch.setattr(E.time, "sleep", lambda s: None)
    handler, seen = _scripted([blip, blip, 200])
    client = E.EdisonClient(api_key=KEY, transport=httpx.MockTransport(handler))
    task = client.get_task("tid")
    assert task.status == "success" and task.formatted_answer == "Yes [1]." and len(seen) == 3

    handler, seen = _scripted([blip])                      # it never recovers
    client = E.EdisonClient(api_key=KEY, transport=httpx.MockTransport(handler))
    with pytest.raises(E.EdisonError) as err:
        client.get_task("tid")
    assert len(seen) == E.READ_ATTEMPTS and "task tid" in str(err.value)   # the id is not lost with it


def test_the_wait_before_another_try_is_the_platforms_or_a_capped_backoff(monkeypatch):
    waits = []
    monkeypatch.setattr(E, "sleep_unless_cancelled", waits.append)
    handler, seen = _scripted([httpx.Response(429, headers={"Retry-After": "3"}, json={}),
                               httpx.Response(503, headers={"Retry-After": "600"}, json={}), 502, 502, 200])
    client = E.EdisonClient(api_key=KEY, transport=httpx.MockTransport(handler))
    assert client.get_task("tid").status == "success"
    assert waits == [3.0, E.MAX_RETRY_WAIT_S, 4.0, 8.0]


def test_a_stop_reaches_a_client_that_is_waiting(monkeypatch):
    """The wait between attempts was a plain sleep, so a Stop
    or a worker's cancel waited out every attempt of an outage."""
    import threading
    from scilink.ui.output_capture import AgentStoppedError
    from scilink.utils import log_context
    handler, seen = _scripted([502])
    client = E.EdisonClient(api_key=KEY, transport=httpx.MockTransport(handler))
    cancel = threading.Event()
    slept = []

    def sleeping(seconds):                 # the cancel arrives during the first wait
        slept.append(seconds)
        cancel.set()
    monkeypatch.setattr(E.time, "sleep", sleeping)
    log_context.register_cancel(cancel)
    try:
        with pytest.raises(AgentStoppedError):
            client.get_task("tid")
        assert len(seen) == 1 and len(slept) == 1 and slept[0] <= 1.0
        with pytest.raises(AgentStoppedError):       # and no request is sent for a cancelled thread
            client.create_task({"name": "job", "query": "q"})
        assert len(seen) == 1
    finally:
        log_context.unregister_cancel()


def test_an_endpoint_that_cannot_be_reached_is_given_up_on_sooner_than_a_slow_reply():
    client = _client(FakeEdison())
    assert client._http.timeout.connect == E.CONNECT_TIMEOUT_S < client._http.timeout.read == E.REQUEST_TIMEOUT_S


def test_a_client_whose_login_fails_leaves_no_connection_open(monkeypatch):
    made = []
    real = httpx.Client

    def recording(*args, **kwargs):
        made.append(real(*args, **kwargs))
        return made[-1]
    monkeypatch.setattr(E.httpx, "Client", recording)
    with pytest.raises(PermissionError):
        E.EdisonClient(api_key=KEY, transport=httpx.MockTransport(FakeEdison(refuse_key=True)))
    assert len(made) == 1 and made[0].is_closed
    ok = E.EdisonClient(api_key=KEY, transport=httpx.MockTransport(FakeEdison()))
    assert not made[1].is_closed
    ok.close()
    assert made[1].is_closed


# ── the agents on it ─────────────────────────────────────────────────────

def test_the_precedent_agent_runs_a_search_end_to_end(monkeypatch):
    from scilink.agents.lit_agents import literature_agent as L
    running = {"id": "t", "status": "in progress", "task": "q", "crow": "job-futurehouse-paperqa3-precedent"}
    fake = FakeEdison([running, running, _task_payload()])
    monkeypatch.setattr(L, "EdisonClient", lambda api_key: _client(fake))
    monkeypatch.setattr(L, "sleep", lambda s: None)
    agent = L.OwlLiteratureAgent(api_key=KEY, max_wait_time=30)
    result = agent.query_literature("Has anyone done X?")
    assert result["status"] == "success" and result["formatted_answer"] == "Yes [1]."
    assert result["has_successful_answer"] is True and json.loads(result["json"])["status"] == "success"
    created = [r for r in fake.requests if r[1] == "/v0.1/crows"]
    assert len(created) == 1 and created[0][3]["name"] == "job-futurehouse-paperqa3-precedent"


def test_the_precedent_agent_reports_a_dead_job(monkeypatch):
    from scilink.agents.lit_agents import literature_agent as L
    fake = FakeEdison([{"id": "t", "status": "in progress"}, {"id": "t", "status": "cancelled"}])
    monkeypatch.setattr(L, "EdisonClient", lambda api_key: _client(fake))
    monkeypatch.setattr(L, "sleep", lambda s: None)
    result = L.OwlLiteratureAgent(api_key=KEY, max_wait_time=30).query_literature("Has anyone done X?")
    assert result["status"] == "error" and "cancelled" in result["message"]


def test_the_molecules_agent_reads_its_answer(monkeypatch):
    from scilink.agents.lit_agents import molecules_agent as M
    payload = {"id": "t", "status": "success", "task": "q", "crow": "job",
               "environment_frame": {"state": {"state": {"answer": "CCO"}}}}
    fake = FakeEdison([payload])
    monkeypatch.setattr(M, "EdisonClient", lambda api_key: _client(fake))
    agent = M.MoleculesAgent(api_key=KEY)
    method = next(getattr(agent, n) for n in dir(agent)
                  if not n.startswith("_") and callable(getattr(agent, n)) and "quer" in n.lower())
    result = method("a molecule that dissolves in water")
    assert result["status"] == "success" and result["content"] == "CCO"
    assert [r[3]["name"] for r in fake.requests if r[1] == "/v0.1/crows"] == [E.JobNames.MOLECULES.value]


@pytest.mark.parametrize("dead", ["fail", "cancelled", "failed", "error", "timeout"])
def test_the_molecules_agent_reports_a_dead_job_at_once(monkeypatch, dead):
    """Only "failed" and "error" ended the wait; the platform
    reports ``fail`` and ``cancelled``, which were polled for the whole budget."""
    from scilink.agents.lit_agents import molecules_agent as M
    fake = FakeEdison([{"id": "t", "status": "in progress"}, {"id": "t", "status": dead}])
    naps = []
    monkeypatch.setattr(M, "EdisonClient", lambda api_key: _client(fake))
    monkeypatch.setattr(M, "sleep", naps.append)
    result = M.MoleculesAgent(api_key=KEY, max_wait_time=600).query("a molecule that dissolves in water")
    assert result["status"] == "error" and dead in result["message"] and len(naps) == 1


def test_a_literature_answer_in_the_second_shape_reaches_its_caller(monkeypatch):
    from scilink.agents.lit_agents import literature_agent as L
    payload = {"id": "t", "status": "success", "task": "q", "crow": "job",
               "environment_frame": {"state": {"state": {"answer": "Yes, in 2019."}}}}
    monkeypatch.setattr(L, "EdisonClient", lambda api_key: _client(FakeEdison([payload])))
    result = L.OwlLiteratureAgent(api_key=KEY).query_literature("Has anyone done X?")
    assert result["status"] == "success" and result["formatted_answer"] == "Yes, in 2019."


# ── no Edison package is needed, or loaded ───────────────────────────────

_NO_EDISON_PACKAGES = '''
import sys
for name in ("edison_client", "ldp", "lmi", "aviary"):
    sys.modules[name] = None          # importing any of them would raise
import scilink.agents.exp_agents.analysis_orchestrator_tools  # noqa: F401
from scilink.agents.lit_agents import OwlLiteratureAgent, MoleculesAgent  # noqa: F401
from scilink.agents.lit_agents.literature_agent import EdisonClient, JobNames
assert EdisonClient.__module__ == "scilink.agents.lit_agents._edison_http", EdisonClient.__module__
assert JobNames.LITERATURE.value == "job-futurehouse-paperqa3"
print("OK")
'''


def test_the_literature_tools_need_no_edison_package():
    r = subprocess.run([sys.executable, "-c", _NO_EDISON_PACKAGES],
                       capture_output=True, text=True, timeout=280)
    assert r.returncode == 0, f"stdout: {r.stdout}\nstderr: {r.stderr}"
    assert "OK" in r.stdout


def test_the_requirements_name_no_edison_package():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for name in ("pyproject.toml", "requirements.txt"):
        lines = [line.split("#")[0] for line in (root / name).read_text().splitlines()]
        text = "\n".join(lines)
        for package in ("edison-client", "fhlmi", "fhaviary", "ldp"):
            assert package not in text, f"{name} still requires {package}"
        assert "httpx" in text, name


# ── held to Edison's own client where it is installed ────────────────────

def test_the_reading_matches_edisons_client_where_it_is_installed():
    """Edison's client, when this environment has a working one, reads the
    same payload to the same fields and knows the same job names."""
    try:
        from edison_client.models.app import JobNames as Theirs, PQATaskResponse
    except Exception:
        pytest.skip("edison-client is not installed here, or does not import")
    payload = _task_payload()
    ours, theirs = E.TaskStatus(payload), PQATaskResponse(**payload)
    for field in ("status", "query", "job_name", "answer", "formatted_answer", "answer_reasoning",
                  "has_successful_answer", "total_cost", "total_queries", "environment_name", "agent_name"):
        assert getattr(ours, field) == getattr(theirs, field), field
    assert str(theirs.task_id) == ours.task_id
    assert E.JobNames.LITERATURE.value == Theirs.LITERATURE.value
    assert E.JobNames.PRECEDENT.value == Theirs.PRECEDENT.value
