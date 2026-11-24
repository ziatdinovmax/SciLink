"""Edison Scientific's platform over HTTP: the two calls the literature
agents make, and nothing else.

The agents create a task (a job name and a query) and fetch it until it is
done. Edison's own Python client (``edison-client``) does that and a great
deal more, and it imports Edison's agent framework (``ldp``, ``fhlmi``) to
do the rest; that framework pins LiteLLM to a narrow range of old versions,
and SciLink's every model call goes through LiteLLM. So the two calls are
made here directly and SciLink depends on none of those packages: a
literature search no longer decides which LiteLLM the rest of the system
may run on, and a version skew inside that chain can no longer take the
literature tools down.

The protocol is the one ``edison-client`` speaks (read from 0.10.0 and
0.19.0, the same in both):

- ``POST /auth/login`` with ``{"api_key": ...}`` returns an access token;
- every other request carries it as a bearer token, and a 401 or 403 is
  answered by logging in again and repeating the request once;
- ``POST /v0.1/crows`` with the task returns its ``trajectory_id``;
- ``GET /v0.1/trajectories/<id>`` returns the task, whose answer sits in
  ``environment_frame.state.state.response.answer``.

It is Edison's protocol, not a contract Edison made with SciLink: if a
search starts failing after a change on their side, this file is where to
look, and ``edison-client``'s ``clients/rest_client.py`` is the reference.
Job names are retired too (the molecules job was ``job-futurehouse-phoenix``
until Edison renamed it): ``GET /v0.1/crows`` lists, under ``key``, the jobs
a key may start, and Edison's refusal of an unknown name says so.

A request is repeated only when repeating it is safe. A read (fetching a
task) and the login are: they are tried again after a lost connection, a
gateway error or a rate limit, so one bad reply does not end a search that
has run for minutes. Creating a task is not: once the request may have
reached the platform it is never sent again, because a task whose reply was
lost may have started and a second one would be paid for. Every wait between
attempts ends on the thread's cancel (a Stop, a worker's budget).

The API key goes to Edison's login endpoint and nowhere else: never into a
log line, an error message or the task record.
"""
from __future__ import annotations

import json
import threading
import time
from enum import Enum
from typing import Any, Dict, Optional

import httpx

from ...utils.log_context import raise_if_cancelled, sleep_unless_cancelled

BASE_URL = "https://api.platform.edisonscientific.com"
REQUEST_TIMEOUT_S = 30.0
# An endpoint that cannot be reached is given up on sooner than a slow reply.
CONNECT_TIMEOUT_S = 10.0
# A request is tried this many times in all, as Edison's client does; a read
# gets more, since it is free to repeat and a search behind it took minutes.
ATTEMPTS = 3
READ_ATTEMPTS = 5
MAX_RETRY_WAIT_S = 10.0

# The request never left: nothing reached the platform, so any request may
# be sent again.
_NEVER_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)


def _try_again_later(status_code: int) -> bool:
    """A reply that is about the moment, not the request: a rate limit or a
    server-side failure."""
    return status_code == 429 or status_code >= 500


def _wait_before(attempt: int, reply: Optional[httpx.Response] = None) -> float:
    """Seconds to wait after ``attempt`` (0-based): what the platform asked
    for (``Retry-After``), else 1, 2, 4 ..., never above the cap."""
    wait = 2.0 ** attempt
    if reply is not None:
        try:
            wait = float(reply.headers.get("retry-after", ""))
        except ValueError:
            pass
    return min(max(wait, 0.0), MAX_RETRY_WAIT_S)


class JobNames(str, Enum):
    """The jobs the literature agents start, by the name Edison's platform
    knows each under (``edison-client``'s ``JobNames``; all three were on
    the platform's own list on 2026-11-23)."""
    LITERATURE = "job-futurehouse-paperqa3"
    PRECEDENT = "job-futurehouse-paperqa3-precedent"
    MOLECULES = "job-futurehouse-data-analysis-molecules"


class EdisonError(RuntimeError):
    """A request Edison's platform refused or could not serve."""


class TaskStatus:
    """A task as fetched: its ``status`` and, once it succeeded, its answer.

    The attribute names are ``edison-client``'s (``PQATaskResponse``), so the
    agents read it as they read that client's response."""

    _FIELDS = ("status", "query", "user", "created_at", "job_name", "share_status",
               "permitted_accessors", "build_owner", "environment_name", "agent_name",
               "task_id", "project_id", "answer", "formatted_answer", "answer_reasoning",
               "has_successful_answer", "total_cost", "total_queries")

    def __init__(self, data: Dict[str, Any], task_id: Optional[str] = None):
        data = data if isinstance(data, dict) else {}
        metadata = data.get("metadata") or {}
        state = (((data.get("environment_frame") or {}).get("state") or {}).get("state") or {})
        answer = ((state.get("response") or {}).get("answer") or {})
        usage = ((state.get("info") or {}).get("usage") or {})
        self.status: str = str(data.get("status") or "")
        self.query = data.get("task")
        self.user = data.get("user")
        self.created_at = data.get("created_at")
        self.job_name = data.get("crow")
        self.share_status = data.get("share_status")
        self.permitted_accessors = data.get("permitted_accessors")
        self.build_owner = data.get("build_owner")
        self.environment_name = metadata.get("environment_name")
        self.agent_name = metadata.get("agent_name")
        self.task_id = data.get("id") or task_id
        self.project_id = data.get("project_id")
        # Edison's client knows a second shape, with the answer at the top of
        # the state (its PhoenixTaskResponse): read there when the
        # paper-search shape holds none
        self.answer = answer.get("answer") if isinstance(answer, dict) else None
        if self.answer is None and isinstance(state.get("answer"), str):
            self.answer = state.get("answer")
        self.formatted_answer = answer.get("formatted_answer") if isinstance(answer, dict) else None
        if not answer and self.answer is not None:
            # that second shape has one text: the literature callers read
            # ``formatted_answer``, and a success must not reach them empty
            self.formatted_answer = self.answer
        self.answer_reasoning = answer.get("answer_reasoning") if isinstance(answer, dict) else None
        self.has_successful_answer = answer.get("has_successful_answer") if isinstance(answer, dict) else None
        self.total_cost = usage.get("total_cost")
        self.total_queries = usage.get("total_queries")

    def model_dump(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self._FIELDS}

    def model_dump_json(self) -> str:
        return json.dumps(self.model_dump(), default=str)


class EdisonClient:
    """``create_task`` and ``get_task`` against Edison's platform.

    Logs in when it is built, as Edison's client does, so a key the platform
    refuses is reported where the agent is constructed and not at the first
    search. Safe to share between threads."""

    def __init__(self, api_key: str, base_url: str = BASE_URL,
                 timeout: float = REQUEST_TIMEOUT_S,
                 transport: Optional[httpx.BaseTransport] = None):
        if not api_key:
            raise ValueError("An Edison API key is required.")
        self._api_key = api_key
        self._http = httpx.Client(base_url=base_url, transport=transport,
                                  timeout=httpx.Timeout(timeout, connect=min(timeout, CONNECT_TIMEOUT_S)),
                                  headers={"Content-Type": "application/json"})
        self._lock = threading.Lock()
        self._token: Optional[str] = None
        try:
            self._login()
        except BaseException:
            self._http.close()          # a client that never logged in holds no connection open
            raise

    # ── the two calls ────────────────────────────────────────────────────

    def create_task(self, task_data: Dict[str, Any]) -> str:
        """Start a job; returns the task's id. ``task_data`` is ``{"name":
        <JobNames or the job's own name>, "query": <text>}``."""
        name = task_data.get("name")
        name = name.value if isinstance(name, Enum) else name
        query = task_data.get("query")
        if not name or not isinstance(query, str) or not query:
            raise ValueError("A task needs a job name and a non-empty query.")
        body = {"id": None, "project_id": None, "name": str(name), "query": query,
                "runtime_config": task_data.get("runtime_config")}
        # never repeated once it may have arrived: see the module docstring
        reply = self._request("POST", "/v0.1/crows", what=f"creating a {name} task",
                              repeatable=False, json=body)
        try:
            return str(reply.json()["trajectory_id"])
        except Exception as exc:
            raise EdisonError("Edison accepted the task and returned no task id.") from exc

    def get_task(self, task_id: str) -> TaskStatus:
        """The task as it stands now."""
        reply = self._request("GET", f"/v0.1/trajectories/{task_id}", what=f"fetching task {task_id}",
                              repeatable=True, params={"history": False, "lite": False})
        try:
            data = reply.json()
        except Exception as exc:
            raise EdisonError(f"Edison's reply for task {task_id} could not be read.") from exc
        return TaskStatus(data, task_id=str(task_id))

    def close(self) -> None:
        self._http.close()

    # ── transport ────────────────────────────────────────────────────────

    def _login(self) -> None:
        """Exchange the API key for an access token."""
        try:
            reply = self._send("POST", "/auth/login", repeatable=True, json={"api_key": self._api_key})
        except httpx.HTTPError as exc:
            raise EdisonError(f"Could not reach Edison to log in ({type(exc).__name__}).") from None
        if reply.status_code in (401, 403):
            raise PermissionError(
                "Edison refused the API key (it may be mistyped, expired or revoked).")
        if reply.status_code >= 400:
            raise EdisonError(f"Edison's login failed with status {reply.status_code}.")
        try:
            token = reply.json()["access_token"]
        except Exception:
            raise EdisonError("Edison's login returned no access token.") from None
        with self._lock:
            self._token = token

    def _send(self, method: str, url: str, *, repeatable: bool, **kwargs) -> httpx.Response:
        """One request, tried again when that is safe: always when it never
        left; when it may have arrived (the reply was lost, or says "not
        now") only if it is ``repeatable``. The thread's cancel is looked at
        before every attempt and ends every wait."""
        attempts = READ_ATTEMPTS if method == "GET" else ATTEMPTS
        for attempt in range(attempts):
            raise_if_cancelled()
            last = attempt == attempts - 1
            reply = None
            try:
                reply = self._http.request(method, url, **kwargs)
            except _NEVER_SENT:
                if last:
                    raise
            except httpx.TransportError:
                if last or not repeatable:
                    raise
            else:
                if last or not (repeatable and _try_again_later(reply.status_code)):
                    return reply
            sleep_unless_cancelled(_wait_before(attempt, reply))
        raise AssertionError("unreachable")

    def _request(self, method: str, url: str, what: str, *, repeatable: bool, **kwargs) -> httpx.Response:
        """An authenticated request: a refused token is renewed once."""
        reply = None
        for renewed in (False, True):
            with self._lock:
                headers = {"Authorization": f"Bearer {self._token}"}
            try:
                reply = self._send(method, url, repeatable=repeatable, headers=headers, **kwargs)
            except _NEVER_SENT as exc:
                raise EdisonError(f"Could not reach Edison while {what} ({type(exc).__name__}).") from None
            except httpx.HTTPError as exc:
                note = "" if repeatable else " The request may have arrived, so it was not sent again."
                raise EdisonError(
                    f"Edison did not answer while {what} ({type(exc).__name__}).{note}") from None
            if reply.status_code not in (401, 403) or renewed:
                break
            self._login()
        if reply.status_code in (401, 403):
            raise PermissionError(f"Edison denied permission while {what}.")
        if reply.status_code >= 400:
            detail = ""
            try:
                detail = str(reply.json().get("detail") or "")[:200]
            except Exception:
                pass
            raise EdisonError(f"Edison returned status {reply.status_code} while {what}"
                              + (f": {detail}" if detail else "."))
        return reply
