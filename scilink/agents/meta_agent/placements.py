"""Where a swarm item runs: the worker contract and its local placements.

One interface, so the local scheduler already stamps what a later placement
(an HPC job through ``ClusterExecutor``, an ECS task) will be read by::

    submit(spec) -> handle
    poll(handle) -> {state, result?, peak_rss_bytes?, stop_reason?}
    cancel(handle) -> None          # idempotent

``spec`` is plain data and carries no provider key: the mode, the task, its
context, the item's directory and budgets, and what the host the item runs
for looks like (model, endpoints, file roots, the skills and MCP servers it
shares). A key travels as the NAME of the environment variable that holds
it; the worker inherits the environment and reads that variable. (An MCP
server's ``env`` and ``headers`` do travel in the spec, as the meta holds
them: over stdin, never written to disk.) A state is
one of ``STATES``: ``done`` means the worker returned a result (the result's
own ``status`` says how the work went); ``failed`` that the worker itself
failed with nothing returned; ``out_of_memory`` that it was killed with
nothing returned and nobody asked for the cancel — the memory guard or the
operating system's killer; ``interrupted`` that it was lost on a restart.

Two placements run here:

- ``process`` — a fresh interpreter through ``utils.child_process.run_child``
  (never a spawn pool, #721), waited on by a thread attributed to the item's
  turn: the item's cancel and the turn's Stop reach the child's whole tree
  through the handle's own ``cancel``, the child's console goes to
  ``<item dir>/worker.log`` and is relayed line by line into the turn (the
  web UI and the shell see the item's narration as they do a thread's), and
  the child's memory is measured while it runs, which is what fills the
  table of measured peaks per item class (``peaks.py``). An analysis item
  with data runs here when nobody attends the swarm: its questions take
  their defaults either way, and its process is the one that can be ended
  and measured.
- ``thread`` — today's path, in the coordinator's process: a planning or
  simulation item (bound by model calls, no data of its own), and every item
  of an attended swarm, whose questions need the person's channel. Its peak
  cannot be split from the process's, so it measures nothing.

The item's own thread is the waiter in both placements: it submits, polls
until the handle is terminal and closes the delegation. A placement with no
thread per item would be polled from the coordinator's loop through the
same ``poll``.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

STATES = ("queued", "running", "done", "failed", "cancelled", "out_of_memory", "interrupted")
TERMINAL = frozenset(STATES[2:])
PLACEMENTS = ("thread", "process")
#: ``SCILINK_SWARM_PLACEMENT``: ``auto`` (the rule above), ``thread`` (every
#: item in-process, the offline tests' setting), ``process`` (every analysis
#: item with data in a process, attended or not — its questions then take
#: their defaults).
PLACEMENT_ENV = "SCILINK_SWARM_PLACEMENT"
_RUN_ITEM = "scilink.agents.meta_agent.placements:run_item"
_POLL_S = 0.5
WORKER_LOG = "worker.log"
WORKER_USAGE = "worker_usage.json"


@dataclass
class WorkerHandle:
    """What ``submit`` returns; ``poll`` reads it, ``cancel`` sets it."""
    spec: dict
    placement: str
    state: str = "queued"
    result: Optional[dict] = None
    peak_rss_bytes: Optional[float] = None
    current_rss_bytes: Optional[float] = None
    stop_reason: Optional[str] = None
    usage: Dict[str, dict] = field(default_factory=dict)
    unattended: int = 0
    cancel_event: threading.Event = field(default_factory=threading.Event)
    thread_id: Optional[int] = None

    def snapshot(self) -> dict:
        return {"state": self.state, "result": self.result, "peak_rss_bytes": self.peak_rss_bytes,
                "stop_reason": self.stop_reason}


# ------------------------------------------------------------------ the rule

def placement_for(orch, item: dict, attended: bool) -> Tuple[str, str]:
    """``(placement, reason)`` for one item under the rule in the module
    docstring and ``SCILINK_SWARM_PLACEMENT``."""
    forced = (os.environ.get(PLACEMENT_ENV) or "auto").strip().lower()
    heavy = item.get("mode") == "analysis" and bool(item.get("data_path"))
    if forced == "thread":
        return "thread", f"{PLACEMENT_ENV}=thread"
    if not heavy:
        return "thread", "no data of its own: bound by model calls"
    if attended and forced != "process":
        return "thread", "the swarm is attended: its questions need the person's channel"
    refusal = host_refusal(orch)
    if refusal:
        return "thread", refusal
    return "process", ("an analysis item with data, nobody attending"
                       if forced != "process" else f"{PLACEMENT_ENV}=process")


#: The keys a host may hold, each with the spec field that names the
#: environment variable carrying it.
_KEYS = (("api_key", "api_key_env", "API key"),
         ("embedding_api_key", "embedding_api_key_env", "embedding API key"),
         ("futurehouse_api_key", "futurehouse_api_key_env", "FutureHouse API key"))


def _env_name_of(value: str) -> Optional[str]:
    """The environment variable whose value is ``value`` (the first of them
    in the environment's order), or ``None``."""
    for name, held in os.environ.items():
        if held == value:
            return name
    return None


def host_refusal(orch) -> Optional[str]:
    """Why a fresh process could not stand in for ``orch``: an extension a
    process cannot inherit (a callable tool factory), or a key the meta
    holds that is under no variable of the environment the child inherits —
    the spec carries the variable's NAME, never the key, so a key under no
    name cannot travel. The embedding and FutureHouse keys are held to the
    same rule: a child that silently ran without them would run with
    literature and embeddings off."""
    for ext in getattr(orch, "_shared_extensions", None) or []:
        if ext.get("kind") == "tools":
            return "the meta shares custom tools (callables), which a process cannot inherit"
    for attr, _field, name in _KEYS:
        value = getattr(orch, attr, None)
        if value and _env_name_of(value) is None:
            return f"the meta's {name} is not in the environment a worker process inherits"
    return None


def host_spec(orch) -> dict:
    """What a worker process needs to build a child as ``workers.build_child``
    would for ``orch``: plain data, no key."""
    fence = getattr(orch, "path_fence", None)
    exts = []
    for ext in getattr(orch, "_shared_extensions", None) or []:
        if ext.get("kind") == "skill":
            exts.append({"kind": "skill", "skill_path": str(ext["skill_path"])})
        elif ext.get("kind") == "mcp":
            exts.append({k: v for k, v in ext.items()
                         if k in ("kind", "server_name", "command", "url", "env", "transport", "headers")})
    spec = {
        "model_name": getattr(orch, "model_name", None),
        "base_url": getattr(orch, "base_url", None),
        "embedding_model": getattr(orch, "embedding_model", None),
        "embedding_base_url": getattr(orch, "embedding_base_url", None),
        "knowledge_dir": (str(orch.knowledge_dir) if getattr(orch, "knowledge_dir", None) else None),
        "file_roots": [str(r) for r in fence.roots] if fence is not None else None,
        "extensions": exts,
    }
    for attr, field_, _name in _KEYS:
        value = getattr(orch, attr, None)
        spec[field_] = _env_name_of(value) if value else None      # the name, never the key
    return spec


class _Host:
    """The ``orch`` a worker process builds its child against."""

    class _Fence:
        def __init__(self, roots: Optional[List[str]]):
            self.roots = [Path(r) for r in roots] if roots else []

    def __init__(self, spec: dict):
        self.model_name = spec.get("model_name")
        self.base_url = spec.get("base_url")
        self.embedding_model = spec.get("embedding_model")
        self.embedding_base_url = spec.get("embedding_base_url")
        # Each key is read from the variable the spec NAMES (the meta held it
        # under that name); a spec that names none leaves the constructor to
        # its own resolution from the conventional variables.
        for attr, field_, _name in _KEYS:
            name = spec.get(field_)
            setattr(self, attr, os.environ.get(name) if name else None)
        self.knowledge_dir = spec.get("knowledge_dir")
        self.path_fence = self._Fence(spec.get("file_roots")) if spec.get("file_roots") is not None else None
        self._extensions = list(spec.get("extensions") or [])

    def _propagate_extensions_to_child(self, child) -> None:
        import logging
        for ext in self._extensions:
            try:
                if ext.get("kind") == "skill":
                    child.register_skill(ext["skill_path"])
                elif ext.get("kind") == "mcp":
                    child.connect_mcp_server(ext["server_name"], command=ext.get("command"),
                                             url=ext.get("url"), env=ext.get("env"),
                                             transport=ext.get("transport"), headers=ext.get("headers"))
            except Exception as exc:  # noqa: BLE001 - as the meta: logged, the child stays usable
                logging.getLogger(__name__).warning(
                    f"Could not apply {ext.get('kind')} extension in the worker process: {exc}")


def item_spec(orch, item: dict, task: str, base_dir: Path, autonomy: str) -> dict:
    """The spec a process worker runs: the item's own fields, the host, and
    whether the sandbox was approved in this session (a child cannot answer
    the prompt, so the approval travels)."""
    from ... import executors
    return {
        "mode": item["mode"], "task": task, "context": item.get("context"),
        "label": item["label"], "base_dir": str(base_dir), "autonomy": autonomy,
        "budget_s": item.get("_budget_s"), "host": host_spec(orch),
        "sandbox_approved": bool(getattr(executors, "_GLOBAL_SANDBOX_APPROVED", False)
                                 or (os.environ.get("UNSAFE_EXECUTION_OK") or "").lower() == "true"),
    }


# ------------------------------------------------------------- the child's side

class _Defaults:
    """The worker process's channel: every question gets its default, is
    counted, and is marked as unanswered (``hitl.mark_timed_out``), so a
    gate that treats an empty answer as approval records no human decision
    — by construction, whatever item is ever placed here."""

    def __init__(self):
        self.n = 0

    def ask(self, req) -> str:
        from ... import hitl
        self.n += 1
        hitl.mark_timed_out()
        return req.default or ""


def _plain(value: Any) -> Any:
    """JSON-shaped: what crosses the process boundary must pickle, and a
    result that holds an agent's object must not travel with it."""
    return json.loads(json.dumps(value, default=str))


def _write_atomic(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)


def run_item(spec: dict) -> dict:
    """The worker process's side: build the child, run the task, return
    ``{result, usage, unattended}``. Never raises for the work's failure (it
    becomes an error result); what escapes is the process's failure. The
    usage by model is also written to ``<base_dir>/worker_usage.json`` on
    every call, so a child that is killed is still charged for what it spent;
    the child's tool calls go to its own ``events.jsonl``."""
    from ... import executors, hitl, tracing
    from ...session_events import set_thread_event_log
    from .workers import autonomy_for, build_child, release_child
    if spec.get("sandbox_approved"):
        executors._GLOBAL_SANDBOX_APPROVED = True
        os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")
    channel = _Defaults()
    hitl.set_thread_channel(channel)
    base_dir = Path(spec["base_dir"])
    base_dir.mkdir(parents=True, exist_ok=True)
    set_thread_event_log(str(base_dir / "events.jsonl"))
    usage: Dict[str, dict] = {}
    usage_path = base_dir / WORKER_USAGE

    def sink(model, prompt_tokens, completion_tokens, latency_s, session, worker=None):
        row = usage.setdefault(str(model), {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                           "seconds": 0.0})
        row["calls"] += 1
        row["prompt_tokens"] += int(prompt_tokens or 0)
        row["completion_tokens"] += int(completion_tokens or 0)
        row["seconds"] += float(latency_s or 0.0)
        try:
            _write_atomic(usage_path, usage)
        except OSError:
            pass
    tracing.set_usage_sink(sink)
    host = _Host(spec["host"])
    child = None
    try:
        child = build_child(host, spec["mode"], base_dir, label=f"Swarm: {spec['label']}")
        result = child.run_task(spec["task"], context=spec.get("context"),
                                autonomy=autonomy_for(spec["mode"], spec["autonomy"]))
    except Exception as exc:  # noqa: BLE001 - the work's failure is a result
        result = {"status": "error", "error": str(exc), "summary": "", "key_findings": [],
                  "files_produced": [], "suggested_followups": [], "warnings": []}
    finally:
        if child is not None:
            release_child(child)
        tracing.set_usage_sink(None)
        hitl.set_thread_channel(None)
        set_thread_event_log(None)
    return {"result": _plain(result), "usage": usage, "unattended": channel.n}


# ----------------------------------------------------------------- placements

class _LogRelay:
    """Tails the worker's log and prints each line on a thread attributed to
    the item's turn, so the child's narration reaches the turn (the routed
    capture, the shell's status row) as a thread item's prints do."""

    def __init__(self, path: Path, interval_s: float = 0.3):
        self.path = path
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        from ...utils.log_context import attributed_to_current
        self._thread = threading.Thread(target=attributed_to_current(self._run, "worker-log"),
                                        name="swarm-worker-log", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        pos, buf = 0, b""
        while True:
            try:
                with open(self.path, "rb") as f:
                    f.seek(pos)
                    chunk = f.read()
                pos += len(chunk)
                buf += chunk
            except FileNotFoundError:
                pass
            lines = buf.split(b"\n")
            buf = lines.pop()
            try:
                for line in lines:
                    print(line.decode("utf-8", "replace"))
            except BaseException:  # noqa: BLE001 - the turn was stopped: nothing to relay to
                return
            if self._stop.is_set():
                if buf:
                    try:
                        print(buf.decode("utf-8", "replace"))
                    except BaseException:  # noqa: BLE001
                        pass
                return
            self._stop.wait(self._interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)


class LocalProcess:
    """The ``process`` placement: ``run_item`` in a fresh interpreter."""

    def __init__(self, target: str = _RUN_ITEM):
        self.target = target

    def submit(self, spec: dict) -> WorkerHandle:
        from ...utils.log_context import attributed_to_current
        handle = WorkerHandle(spec, "process")
        base_dir = Path(spec["base_dir"])
        base_dir.mkdir(parents=True, exist_ok=True)
        log_path = base_dir / WORKER_LOG
        relay = _LogRelay(log_path)

        def watch(current: float, peak: float) -> None:
            handle.current_rss_bytes, handle.peak_rss_bytes = current, peak

        def usage_from_file() -> None:
            """What the child had charged before it was lost."""
            try:
                data = json.loads((base_dir / WORKER_USAGE).read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    handle.usage = data
            except (OSError, ValueError):
                pass

        def run() -> None:
            from ...ui.output_capture import AgentStoppedError
            from ...utils.child_process import ChildLost, run_child
            handle.thread_id = threading.get_ident()
            handle.state = "running"
            relay.start()
            try:
                outcome = run_child(self.target, spec, watch=watch, log_path=str(log_path))
            except ChildLost as exc:
                usage_from_file()
                if handle.cancel_event.is_set():
                    # The kill was asked for: the reason is the cancel's, not
                    # the signal's ("commonly the out-of-memory killer").
                    handle.stop_reason = handle.stop_reason or "cancelled"
                    handle.state = "cancelled"
                elif exc.killed:
                    handle.stop_reason = exc.reason
                    handle.state = "out_of_memory"
                else:
                    handle.stop_reason = exc.reason
                    handle.state = "failed"
                return
            except AgentStoppedError as exc:          # the thread's cancel, after the kill
                usage_from_file()
                handle.stop_reason = handle.stop_reason or str(exc) or "cancelled"
                handle.state = "cancelled"
                return
            except BaseException as exc:  # noqa: BLE001 - the placement's own failure
                usage_from_file()
                handle.stop_reason = f"{type(exc).__name__}: {exc}"
                handle.state = "failed"
                return
            finally:
                relay.stop()
            report = outcome.value if isinstance(outcome.value, dict) else {"result": outcome.value}
            handle.result = report.get("result")
            handle.usage = dict(report.get("usage") or {})
            handle.unattended = int(report.get("unattended") or 0)
            if outcome.peak_rss_bytes:
                handle.peak_rss_bytes = max(handle.peak_rss_bytes or 0.0, outcome.peak_rss_bytes)
            handle.state = "done"

        # Attributed to the item's TURN (an attributed thread resolves to the
        # turn's root), so the item's cancel reaches the waiter and the
        # child's lines are the turn's. The child itself is ended through
        # ``cancel`` — the handle's own — never through a kill for the item's
        # thread id, which an attributed thread does not answer to.
        t = threading.Thread(target=attributed_to_current(run, "process"),
                             name=f"swarm-process-{spec.get('label', '')[:24]}", daemon=True)
        handle._thread = t  # type: ignore[attr-defined]
        t.start()
        return handle

    def poll(self, handle: WorkerHandle) -> dict:
        return handle.snapshot()

    def cancel(self, handle: WorkerHandle) -> None:
        if handle.cancel_event.is_set():
            return
        handle.cancel_event.set()
        if handle.thread_id is not None:
            from ...executors import kill_subprocesses_for_thread
            kill_subprocesses_for_thread(handle.thread_id)

    def wait(self, handle: WorkerHandle, poll_s: float = _POLL_S) -> dict:
        """Poll until the handle is terminal; the calling thread's own cancel
        (the item's budget, the guard, the coordinator's Stop) cancels the
        worker. Returns the final ``poll``."""
        from ...utils.log_context import cancel_requested
        while handle.state not in TERMINAL:
            if cancel_requested() and not handle.cancel_event.is_set():
                handle.stop_reason = handle.stop_reason or "cancelled"
                self.cancel(handle)
            time.sleep(poll_s)
        t = getattr(handle, "_thread", None)
        if t is not None:
            t.join(timeout=5.0)
        return self.poll(handle)
