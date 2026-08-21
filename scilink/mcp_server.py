"""MCP server that exposes SciLink's analysis and planning tools.

Allows any MCP client (Claude Desktop, Cursor, etc.) to use SciLink as a
tool provider.  Start with::

    scilink serve --model gemini-3.1-pro-preview

Supports three autonomy modes:
- **autonomous** (default) — tools execute without human approval.
- **autopilot** — tools run but return key decisions for review.
- **co-pilot** — tools that need approval return a ``needs_input``
  response; the MCP client must call ``scilink_respond`` to continue.

The ``mcp`` package is installed by default with SciLink.
"""

import asyncio
import os
import concurrent.futures
import contextlib
import io
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server
    from mcp import types
    HAS_MCP = True
except ImportError:
    HAS_MCP = False


def _require_mcp():
    if not HAS_MCP:
        raise ImportError(
            "MCP server support requires the 'mcp' package. "
            "Install with: pip install scilink"
        )


# Tools that require user approval in co-pilot / autopilot modes.
# In co-pilot mode ALL of these pause; in autopilot mode only the
# high-impact subset (run_analysis, run_optimization) pauses.
_COPILOT_APPROVAL_TOOLS = {
    "run_analysis", "select_agent", "assess_novelty",
    "get_recommendations", "run_optimization", "generate_initial_plan",
    "generate_implementation_code", "run_economic_analysis",
    "discard_plan",
    # Meta-mode granular delegations (each one launches a full specialist
    # workflow — co-pilot gates them; in autopilot they run and the
    # children's own questions surface via awaiting_input instead).
    "delegate_to_analysis", "delegate_to_planning", "delegate_to_simulation",
}
_AUTOPILOT_APPROVAL_TOOLS = {
    "run_analysis", "run_optimization", "discard_plan",
}

# Tools that support optional background execution via ``background=true``.
# These are the long-running tools where a blocking call can realistically
# exceed the ~4 minute tool-call timeout in clients like Claude Desktop:
#   - run_analysis / run_optimization: full agent analysis or BO loop
#   - assess_novelty: FutureHouse literature search per claim (max_wait 600s each)
#   - get_recommendations: LLM synthesis over a full analysis record
#   - generate_initial_plan / generate_implementation_code: RAG + LLM generation
#     over potentially large docs/code knowledge bases
#   - run_economic_analysis: TEA with knowledge retrieval + LLM synthesis
#   - orchestrate_analysis / orchestrate_planning: full orchestrator chat loop
_BACKGROUND_CAPABLE_TOOLS = {
    "run_analysis",
    "run_optimization",
    "assess_novelty",
    "get_recommendations",
    "generate_initial_plan",
    "generate_implementation_code",
    "run_economic_analysis",
    "orchestrate_analysis",
    "orchestrate_planning",
    "orchestrate_meta",
    "delegate_to_analysis",
    "delegate_to_planning",
    "delegate_to_simulation",
}


# ── Schema conversion ───────────────────────────────────────────────────

def _openai_to_mcp_tool(schema: dict, prefix: str = "scilink") -> types.Tool:
    """Convert an OpenAI function-calling schema to an MCP Tool object."""
    fn = schema.get("function", schema)
    name = fn.get("name", "")
    input_schema = fn.get("parameters", {"type": "object", "properties": {}})

    # Add optional ``background`` parameter for long-running tools.
    if name in _BACKGROUND_CAPABLE_TOOLS:
        schema_copy = json.loads(json.dumps(input_schema))
        schema_copy.setdefault("properties", {})["background"] = {
            "type": "boolean",
            "description": (
                "If true, run in the background and return a job_id "
                "immediately. Use scilink_job_status and scilink_job_result "
                "to poll and retrieve results. Default: false (blocking)."
            ),
        }
        input_schema = schema_copy

    return types.Tool(
        name=f"{prefix}_{name}" if prefix else name,
        description=fn.get("description", ""),
        inputSchema=input_schema,
    )


# ── Human-in-the-loop channel (agent prompts over MCP) ─────────────────

# ── Job / pending-question durability (issue #469) ─────────────────────
# The jobs table and parked questions lived only in server memory: after a
# restart a supervising client got "Unknown job_id" and its questions
# evaporated. Every transition is now mirrored to <session_dir>/mcp_jobs.json
# (atomic replace, never fatal); on start, unfinished jobs come back as
# status "interrupted" with a structured re-entry hint, and finished ones
# keep answering job_status/job_result with their stored result.

_JOBS_STORE_NAME = "mcp_jobs.json"
_INTERRUPTED_HINT = (
    "The server restarted while this job was in flight; its Python stack "
    "cannot be resumed, but the campaign/session state is checkpointed. "
    "Re-issue the orchestrate call to continue from where the checkpoints "
    "left off. The narration tail from before the restart is attached."
)


def _jobs_store_path(state: dict):
    sd = (state.get("config") or {}).get("session_dir")
    return (Path(sd) / _JOBS_STORE_NAME) if sd else None


def _persist_jobs(state: dict) -> None:
    """Mirror the jobs table + parked questions to the session dir.

    Called on every transition (create / finish / question parked or
    answered). Best-effort and atomic; a persistence failure must never
    break the tool call it rides on."""
    path = _jobs_store_path(state)
    if path is None:
        return
    try:
        with state.get("_persist_lock") or contextlib.nullcontext():
            jobs = {}
            for jid, job in state.get("jobs", {}).items():
                jobs[jid] = {
                    "tool": job.get("tool"),
                    "started_at": job.get("started_at"),
                    "status": job.get("status"),
                    "result": job.get("result"),
                    "log_tail": list(job.get("log_lines") or [])[-100:],
                }
            questions = []
            for q in _pending_questions(state):
                questions.append({**q, "asked_at": datetime.now().isoformat()})
            payload = {"job_counter": state.get("job_counter", 0),
                       "jobs": jobs, "pending_questions": questions}
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=1, default=str),
                           encoding="utf-8")
            tmp.replace(path)
    except Exception as exc:  # noqa: BLE001 - durability is best-effort
        logging.warning(f"Could not persist MCP job state: {exc}")


def _finalize_job(state: dict, job_id: str) -> None:
    """Done-callback on a job's future: capture status/result immediately
    (not just when a client happens to poll) and persist, so a job that
    finishes moments before a crash is recoverable."""
    job = state.get("jobs", {}).get(job_id)
    if job is None or job.get("future") is None:
        return
    try:
        job["result"] = job["future"].result()
        job["status"] = "completed"
    except Exception as exc:  # noqa: BLE001 - failure is a terminal state too
        job["result"] = json.dumps({"status": "error", "message": str(exc)})
        job["status"] = "failed"
    _persist_jobs(state)


def _register_job(state: dict, job_id: str, job: dict) -> None:
    """Insert a job and wire durability (done-callback + persist)."""
    state["jobs"][job_id] = job
    fut = job.get("future")
    if fut is not None:
        fut.add_done_callback(lambda _f, _j=job_id: _finalize_job(state, _j))
    _persist_jobs(state)


def _restore_jobs(state: dict) -> None:
    """Load the persisted jobs table on server start.

    Finished jobs keep answering job_status/job_result with their stored
    result; unfinished ones become status "interrupted" with a re-entry
    hint. Parked questions cannot survive (their asking thread is gone) —
    they are surfaced once via state["interrupted_questions"] so a
    scilink_respond against them explains what happened instead of
    erroring as unknown."""
    path = _jobs_store_path(state)
    if path is None or not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logging.warning(f"Could not read persisted MCP job state: {exc}")
        return
    n_restored = n_interrupted = 0
    for jid, rec in (payload.get("jobs") or {}).items():
        if jid in state["jobs"]:
            continue
        job = {"future": None, "tool": rec.get("tool"),
               "started_at": rec.get("started_at"),
               "status": rec.get("status"), "result": rec.get("result"),
               "log_lines": list(rec.get("log_tail") or [])}
        if job["status"] in ("running", "awaiting_input"):
            job["status"] = "interrupted"
            job["result"] = json.dumps({
                "status": "interrupted", "job_id": jid,
                "message": _INTERRUPTED_HINT,
                "log_tail": "\n".join(job["log_lines"][-25:])})
            n_interrupted += 1
        state["jobs"][jid] = job
        n_restored += 1
    state["interrupted_questions"] = {
        q.get("request_id"): q for q in payload.get("pending_questions") or []}
    state["job_counter"] = max(state.get("job_counter", 0),
                               int(payload.get("job_counter", 0)))
    if n_restored:
        logging.info(
            f"Restored {n_restored} background job record(s) from "
            f"{path.name} ({n_interrupted} interrupted by the restart)")
    _persist_jobs(state)


class _MCPChannel:
    """hitl.FeedbackChannel that parks agent prompts in the server's
    pending queue for a client to answer via ``scilink_respond``.

    The asking worker thread blocks on an event with a mandatory timeout;
    on timeout the request's declared default answer is returned so an
    unattended run degrades to accept-as-is instead of hanging a tool
    call forever. ``job_id`` ties questions to their background job so
    ``scilink_job_status`` can report ``awaiting_input``.
    """

    def __init__(self, state: dict, job_id: str = None,
                 timeout_s: float = 1800.0) -> None:
        self._state = state
        self._job_id = job_id
        self._timeout_s = timeout_s

    def ask(self, req) -> str:
        import threading

        event = threading.Event()
        holder: dict = {}
        self._state["pending"][req.id] = {
            "type": "question", "req": req, "event": event,
            "holder": holder, "job_id": self._job_id,
        }
        _persist_jobs(self._state)
        try:
            answered = event.wait(self._timeout_s)
        finally:
            self._state["pending"].pop(req.id, None)
            _persist_jobs(self._state)
        if not answered:
            logging.warning(
                f"[hitl] question {req.id} ({req.kind}) unanswered after "
                f"{self._timeout_s}s — using its default answer")
            return req.default
        return holder.get("answer", req.default)


def _pending_questions(state: dict, job_id: str = None) -> list:
    """Structured list of parked agent questions (optionally per job)."""
    out = []
    for rid, item in list(state.get("pending", {}).items()):
        if item.get("type") != "question":
            continue
        if job_id is not None and item.get("job_id") != job_id:
            continue
        req = item["req"]
        out.append({"request_id": rid, "kind": req.kind,
                    "prompt": req.prompt, "options": req.options,
                    "origin": req.origin, "job_id": item.get("job_id")})
    return out


# ── Per-job narration buffer (real-time log_tail for job_status) ───────

class _JobLogHandler(logging.Handler):
    """Streams this job's log records into its bounded line buffer.

    Thread-scoped via ``log_context.effective_thread`` (same mapping the
    UI panel uses), so narration from worker threads the job spawns
    (best-of-N candidates, fan-out branches) lands in the right job's
    buffer while concurrent jobs stay separated.
    """

    def __init__(self, lines, root_thread_id: int) -> None:
        super().__init__(level=logging.INFO)
        self._lines = lines
        self._root_tid = root_thread_id

    def emit(self, record: logging.LogRecord) -> None:
        try:
            from scilink.utils.log_context import effective_thread
            if effective_thread(record.thread) != self._root_tid:
                return
            self._lines.append(record.getMessage())
        except Exception:  # noqa: BLE001 — never break the run over the tail
            pass


class _TeeIO(io.StringIO):
    """StringIO that also appends complete lines to the job's buffer."""

    def __init__(self, lines) -> None:
        super().__init__()
        self._lines = lines
        self._partial = ""

    def write(self, s: str) -> int:
        try:
            self._partial += s
            while "\n" in self._partial:
                line, self._partial = self._partial.split("\n", 1)
                if line.strip():
                    self._lines.append(line)
        except Exception:  # noqa: BLE001
            pass
        return super().write(s)


class _job_narration:
    """Context manager: route this thread's narration into ``lines``."""

    def __init__(self, lines) -> None:
        self._lines = lines
        self._handler = None

    def __enter__(self):
        if self._lines is not None:
            import threading

            self._handler = _JobLogHandler(self._lines,
                                           threading.get_ident())
            logging.getLogger().addHandler(self._handler)
        return self

    def __exit__(self, *exc):
        if self._handler is not None:
            logging.getLogger().removeHandler(self._handler)
        return False


def _new_job_lines():
    from collections import deque

    return deque(maxlen=400)


# ── Stdout capture ──────────────────────────────────────────────────────

def _execute_tool_captured(tools, tool_name: str, kwargs: dict,
                           state: dict = None, job_id: str = None,
                           log_lines=None) -> str:
    """Execute a tool while capturing stdout so it doesn't corrupt stdio MCP transport."""
    from scilink.hitl import set_thread_channel

    if state is not None:
        set_thread_channel(_MCPChannel(
            state, job_id=job_id,
            timeout_s=state["config"].get("hitl_timeout_s", 1800.0)))
    captured = _TeeIO(log_lines) if log_lines is not None else io.StringIO()
    try:
        with _job_narration(log_lines), contextlib.redirect_stdout(captured):
            result = tools.execute_tool(tool_name, **kwargs)
            _checkpoint_after_tool(tools, tool_name)
    finally:
        if state is not None:
            set_thread_channel(None)

    log_output = captured.getvalue().strip()
    if log_output:
        logging.info(f"[tool:{tool_name}] {log_output}")

    return result


def _checkpoint_after_tool(tools, tool_name: str) -> None:
    """Persist the owning orchestrator's state after a granular tool call.

    The chat loops checkpoint after every turn; granular MCP calls bypassed
    that, so a server restarted on the same --session-dir came back with the
    data files but none of the state (schema, selected agent, results
    ledger) — the client's campaign silently lost its footing. Cheap JSON
    write; never fatal."""
    orch = getattr(tools, "orch", None)
    ckpt = getattr(orch, "_auto_checkpoint", None)
    if ckpt is None:
        return
    try:
        ckpt()
    except Exception as exc:  # noqa: BLE001 - durability must not break the call
        logging.warning(f"[tool:{tool_name}] checkpoint after call failed: {exc}")


# ── Pending action support (co-pilot / autopilot) ──────────────────────

class PendingAction:
    """Holds a pending tool call that needs user approval before execution."""

    def __init__(self, tool_name: str, kwargs: dict, prompt: str, context: dict = None):
        self.tool_name = tool_name
        self.kwargs = kwargs
        self.prompt = prompt
        self.context = context or {}


def _needs_approval(tool_name: str, mode: str) -> bool:
    """Check whether a tool requires user approval given the autonomy mode."""
    if mode == "autonomous":
        return False
    if mode in ("co-pilot", "co_pilot", "copilot"):
        return tool_name in _COPILOT_APPROVAL_TOOLS
    if mode == "autopilot":
        return tool_name in _AUTOPILOT_APPROVAL_TOOLS
    return False


def _build_approval_prompt(tool_name: str, kwargs: dict) -> str:
    """Build a human-readable description of the pending action."""
    parts = [f"SciLink wants to execute: {tool_name}"]
    if kwargs:
        for k, v in kwargs.items():
            val = str(v)
            if len(val) > 100:
                val = val[:100] + "..."
            parts.append(f"  {k}: {val}")
    parts.append("\nCall scilink_respond with 'yes' to approve or 'no' to cancel.")
    return "\n".join(parts)


# ── Server factory ──────────────────────────────────────────────────────

_HEADLESS_BACKENDS = {"agg", "pdf", "ps", "svg", "template", "cairo"}


def _ensure_headless_matplotlib() -> None:
    """An MCP server has no display and executes tools off the main thread;
    GUI matplotlib backends (macOS Cocoa in particular) refuse that and kill
    the tool call. Default MPLBACKEND to Agg and force-switch away from a GUI
    backend if one was already resolved — operators should not need to know
    this (it was integration rule 2 of every MCP client)."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib
        backend = str(matplotlib.get_backend())
        if (backend.lower() not in _HEADLESS_BACKENDS
                and not backend.lower().startswith("module://")):
            matplotlib.use("Agg", force=True)
            logging.info(f"matplotlib backend {backend} → Agg (headless MCP server)")
    except Exception as exc:  # noqa: BLE001 - never block serving over a plot backend
        logging.warning(f"Could not enforce headless matplotlib backend: {exc}")


def _load_shared_credentials() -> list:
    """Load ``~/.scilink/credentials.env`` (KEY=VALUE lines) into the process
    environment with setdefault semantics — an explicitly set variable always
    wins. Lets every MCP client config stay secret-free: put the LLM
    credentials in ONE file instead of into each client's .mcp.json."""
    path = Path.home() / ".scilink" / "credentials.env"
    loaded = []
    if not path.exists():
        return loaded
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v and k not in os.environ:
                os.environ[k] = v
                loaded.append(k)
        if loaded:
            logging.info(f"Loaded credentials from {path}: {loaded}")
    except Exception as exc:  # noqa: BLE001 - never block serving
        logging.warning(f"Could not read {path}: {exc}")
    return loaded


# ── Foreground narration (log-message notifications) ────────────────────
# Foreground tool calls used to be silent until the final JSON (stdout must
# stay clean for the stdio transport). The captured narration is now also
# streamed to the client as MCP `notifications/message` entries while the
# call runs, so any client that renders log notifications shows live
# progress with no background-job dance. Disable: SCILINK_MCP_NARRATION=0.
import contextvars as _contextvars

_MCP_SESSION: "_contextvars.ContextVar" = _contextvars.ContextVar(
    "scilink_mcp_session", default=None)

_NARRATION_POLL_S = 1.0
_NARRATION_MAX_PER_POLL = 50


def _narration_enabled() -> bool:
    return os.environ.get("SCILINK_MCP_NARRATION", "1").lower() not in (
        "0", "false", "no", "off")


async def _to_thread_streaming(fn, *args) -> str:
    """Run a captured executor (signature ``fn(*args, log_lines)``) in a
    worker thread; while it runs, forward newly captured narration lines to
    the client as info-level log notifications. Notification failures stop
    the streaming but never affect the call."""
    if not _narration_enabled():
        return await asyncio.to_thread(fn, *args, None)
    lines: list = []          # plain list: append-only, no maxlen truncation
    task = asyncio.ensure_future(asyncio.to_thread(fn, *args, lines))
    session = _MCP_SESSION.get()
    sent = 0
    while True:
        done = task.done()
        if session is not None and len(lines) > sent:
            batch = lines[sent:sent + _NARRATION_MAX_PER_POLL]
            sent += len(batch)
            for line in batch:
                if not str(line).strip():
                    continue
                try:
                    await session.send_log_message(
                        "info", str(line), logger="scilink")
                except Exception:  # noqa: BLE001 - client gone / no support
                    session = None
                    break
        if done:
            break
        await asyncio.sleep(_NARRATION_POLL_S)
    return task.result()


def create_server(
    *,
    api_key: str = None,
    model_name: str = "gemini-3.1-pro-preview",
    base_url: str = None,
    mode: str = "both",
    session_dir: str = None,
    analysis_mode: str = "autonomous",
    futurehouse_api_key: str = None,
    hitl_timeout_s: float = 1800.0,
) -> "Server":
    """Create and return a configured MCP Server instance.

    Args:
        api_key: LLM API key.
        model_name: Model identifier.
        base_url: Optional OpenAI-compatible endpoint.
        mode: ``"analyze"``, ``"plan"``, or ``"both"``.
        session_dir: Directory for session outputs.
        analysis_mode: ``"autonomous"``, ``"autopilot"``, or ``"co-pilot"``.
        futurehouse_api_key: Optional FutureHouse/Edison API key.

    Returns:
        A configured ``mcp.server.lowlevel.Server``.
    """
    _require_mcp()

    server = Server("scilink")

    _ensure_headless_matplotlib()
    _load_shared_credentials()
    import threading as _threading

    # ── State (initialized lazily on first tool list) ────────────────
    # Thread pool for background jobs
    _executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    state: Dict[str, Any] = {
        "analysis_orch": None,
        "planning_orch": None,
        "meta_orch": None,
        # Pending items awaiting a client's scilink_respond, keyed by
        # request id. Two shapes: {"type": "tool_approval", "action":
        # PendingAction} for gated tool calls, and {"type": "question",
        # "req": FeedbackRequest, "event": ..., "holder": ..., "job_id":
        # ...} for agent prompts parked by _MCPChannel. A queue, not a
        # slot — concurrent questions never overwrite each other.
        "pending": {},
        "initialized": False,
        "jobs": {},
        "job_counter": 0,
        "interrupted_questions": {},
        "_persist_lock": _threading.Lock(),
        "config": {
            "api_key": api_key,
            "model_name": model_name,
            "base_url": base_url,
            "mode": mode,
            "session_dir": session_dir,
            "analysis_mode": analysis_mode,
            "futurehouse_api_key": futurehouse_api_key,
            "hitl_timeout_s": hitl_timeout_s,
        },
    }

    # Tool name → (orchestrator_key, original_name) mapping
    tool_map: Dict[str, tuple] = {}

    def _ensure_initialized():
        if state["initialized"]:
            return

        config = state["config"]
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            _init_orchestrators(state, config)

        log = captured.getvalue().strip()
        if log:
            logging.info(f"[init] {log}")

        # Build tool map — analysis tools first, then planning with
        # collision avoidance (prefix colliding names with "plan_").
        if state["analysis_orch"]:
            for schema in state["analysis_orch"].tools.openai_schemas:
                fn_name = schema.get("function", {}).get("name", "")
                if fn_name:
                    tool_map[f"scilink_{fn_name}"] = ("analysis_orch", fn_name)

        if state["planning_orch"]:
            for schema in state["planning_orch"].tools.openai_schemas:
                fn_name = schema.get("function", {}).get("name", "")
                if fn_name:
                    mcp_name = f"scilink_{fn_name}"
                    if mcp_name in tool_map:
                        mcp_name = f"scilink_plan_{fn_name}"
                    tool_map[mcp_name] = ("planning_orch", fn_name)

        if state["meta_orch"]:
            for schema in state["meta_orch"].tools.openai_schemas:
                fn_name = schema.get("function", {}).get("name", "")
                if fn_name:
                    tool_map[f"scilink_{fn_name}"] = ("meta_orch", fn_name)

        state["initialized"] = True

    # ── Eager init (call before starting transport) ────────────────

    def _eager_init():
        _ensure_initialized()

    server.eager_init = _eager_init

    # ── tools/list ───────────────────────────────────────────────────

    @server.list_tools()
    async def list_tools() -> List[types.Tool]:
        _ensure_initialized()
        tools = []

        if state["analysis_orch"]:
            for schema in state["analysis_orch"].tools.openai_schemas:
                tools.append(_openai_to_mcp_tool(schema, prefix="scilink"))

        if state["planning_orch"]:
            for schema in state["planning_orch"].tools.openai_schemas:
                fn_name = schema.get("function", {}).get("name", "")
                mcp_name = f"scilink_{fn_name}"
                if mcp_name in tool_map and tool_map[mcp_name][0] != "planning_orch":
                    prefix = "scilink_plan"
                else:
                    prefix = "scilink"
                tools.append(_openai_to_mcp_tool(schema, prefix=prefix))

        if state["meta_orch"]:
            for schema in state["meta_orch"].tools.openai_schemas:
                tools.append(_openai_to_mcp_tool(schema, prefix="scilink"))

        # Always include scilink_respond so it's available if the user
        # switches to co-pilot/autopilot mode mid-session via
        # scilink_set_autonomy (MCP clients only call tools/list once).
        tools.append(types.Tool(
            name="scilink_respond",
            description=(
                "Answer a pending SciLink request. Two cases: (1) a tool "
                "approval from co-pilot/autopilot mode ('needs_input' "
                "status) — respond 'yes'/'approve' to proceed or anything "
                "else to cancel; (2) an agent question surfaced while a "
                "job runs ('awaiting_input' in scilink_job_status) — the "
                "response text is delivered verbatim as the answer (empty "
                "string usually means accept as-is)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": (
                            "The answer. For tool approvals: 'yes'/'approve' "
                            "to proceed, anything else cancels. For agent "
                            "questions: delivered verbatim ('' = accept)."
                        ),
                    },
                    "request_id": {
                        "type": "string",
                        "description": (
                            "Which pending request to answer (from the "
                            "'needs_input' payload or scilink_job_status). "
                            "Optional when exactly one request is pending."
                        ),
                    },
                },
                "required": ["response"],
            },
        ))

        # Background job management tools
        tools.append(types.Tool(
            name="scilink_job_status",
            description=(
                "Check the status of a background job started with "
                "background=true. Returns 'running', 'awaiting_input', "
                "'completed', or 'failed', plus a log_tail of the job's "
                "recent narration so progress is visible in real time."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job ID returned by the original tool call.",
                    },
                    "tail_lines": {
                        "type": "integer",
                        "description": (
                            "How many recent narration lines to include as "
                            "log_tail (default 25, max 200, 0 to omit)."
                        ),
                    },
                },
                "required": ["job_id"],
            },
        ))
        tools.append(types.Tool(
            name="scilink_job_result",
            description=(
                "Retrieve the full result of a completed background job. "
                "Returns an error if the job is still running."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job ID returned by the original tool call.",
                    },
                },
                "required": ["job_id"],
            },
        ))

        # Always add the set_autonomy tool
        tools.append(types.Tool(
            name="scilink_set_autonomy",
            description=(
                "Change the autonomy mode at runtime. In 'autonomous' mode "
                "all tools execute immediately. In 'autopilot' mode high-impact "
                "tools (run_analysis, run_optimization) pause for approval. "
                "In 'co-pilot' mode most action tools pause for approval. "
                "Returns the new mode and whether scilink_respond is now needed."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["autonomous", "autopilot", "co-pilot"],
                        "description": "The autonomy mode to switch to.",
                    },
                },
                "required": ["mode"],
            },
        ))

        # Orchestrator-level tools — delegate entire workflows instead
        # of calling individual tools one by one.
        if state["analysis_orch"]:
            tools.append(types.Tool(
                name="scilink_orchestrate_analysis",
                description=(
                    "Delegate a complete analysis workflow to SciLink's analysis "
                    "orchestrator. Instead of calling individual tools (examine_data, "
                    "select_agent, run_analysis, etc.) one by one, send a natural-"
                    "language prompt and the orchestrator handles the entire flow: "
                    "data examination, metadata handling, agent selection, analysis "
                    "execution, and results compilation. Best for complex multi-step "
                    "analyses where SciLink's domain expertise should drive the "
                    "workflow. Use background=true for non-trivial requests, as the "
                    "orchestrator may take several minutes."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": (
                                "Natural-language instruction for the analysis "
                                "orchestrator, e.g. 'Analyze the XPS data at "
                                "/path/to/data.csv using the xps skill and assess "
                                "novelty of the findings.'"
                            ),
                        },
                        "background": {
                            "type": "boolean",
                            "description": (
                                "If true, run in the background and return a job_id "
                                "immediately. Recommended for most requests. Use "
                                "scilink_job_status and scilink_job_result to poll "
                                "and retrieve results. Default: false."
                            ),
                        },
                    },
                    "required": ["prompt"],
                },
            ))

        if state["planning_orch"]:
            tools.append(types.Tool(
                name="scilink_orchestrate_planning",
                description=(
                    "Delegate a complete planning workflow to SciLink's planning "
                    "orchestrator. Send a natural-language prompt and the "
                    "orchestrator handles the entire flow: objective understanding, "
                    "knowledge base retrieval, experimental plan generation, "
                    "implementation code, and Bayesian optimization. Best for "
                    "complex experimental design tasks where SciLink's domain "
                    "expertise should drive the workflow. Use background=true for "
                    "non-trivial requests."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": (
                                "Natural-language instruction for the planning "
                                "orchestrator, e.g. 'Generate an experimental plan "
                                "for optimizing lithium extraction from brine using "
                                "papers in ./literature/ as the knowledge base.'"
                            ),
                        },
                        "background": {
                            "type": "boolean",
                            "description": (
                                "If true, run in the background and return a job_id "
                                "immediately. Recommended for most requests. Use "
                                "scilink_job_status and scilink_job_result to poll "
                                "and retrieve results. Default: false."
                            ),
                        },
                    },
                    "required": ["prompt"],
                },
            ))

        if state["meta_orch"]:
            tools.append(types.Tool(
                name="scilink_orchestrate_meta",
                description=(
                    "Delegate a complete multi-step scientific investigation to "
                    "SciLink's META orchestrator — its orchestrator-of-"
                    "orchestrators. The meta routes work across the analysis and "
                    "planning specialists, runs complementary datasets as a "
                    "PARALLEL fan-out with cross-modal fusion, and bridges "
                    "findings between modes (e.g. analysis results feeding an "
                    "optimization campaign). Send one natural-language "
                    "instruction describing the whole investigation. Use "
                    "background=true (recommended): poll scilink_job_status for "
                    "progress (log_tail) and answer its questions "
                    "(awaiting_input -> scilink_respond)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": (
                                "Natural-language instruction for the meta "
                                "orchestrator, e.g. 'Analyze A.csv and B.csv in "
                                "parallel (they are complementary modalities of "
                                "one sample), fuse the findings, then design a "
                                "follow-up experiment.'"
                            ),
                        },
                        "background": {
                            "type": "boolean",
                            "description": (
                                "If true, run in the background and return a "
                                "job_id immediately. Recommended. Poll with "
                                "scilink_job_status; fetch with "
                                "scilink_job_result. Default: false."
                            ),
                        },
                    },
                    "required": ["prompt"],
                },
            ))

        return tools

    # ── tools/call ───────────────────────────────────────────────────

    @server.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> List[types.TextContent]:
        _ensure_initialized()
        # Session handle for foreground narration (log-message
        # notifications). Best effort: absent a request context we just
        # run silently, exactly as before.
        try:
            _MCP_SESSION.set(server.request_context.session)
        except Exception:  # noqa: BLE001
            _MCP_SESSION.set(None)

        # Handle the respond tool
        if name == "scilink_respond":
            return await _handle_respond(state, arguments)

        # Handle autonomy mode switch
        if name == "scilink_set_autonomy":
            return _handle_set_autonomy(state, arguments)

        # Handle background job status/result
        if name == "scilink_job_status":
            return _handle_job_status(state, arguments)
        if name == "scilink_job_result":
            return _handle_job_result(state, arguments)

        # Handle orchestrator-level chat tools
        if name in ("scilink_orchestrate_analysis", "scilink_orchestrate_planning"):
            return await _handle_orchestrate(
                state, name, arguments, _executor
            )

        if name == "scilink_orchestrate_meta":
            return await _handle_orchestrate_meta(state, arguments, _executor)

        # Look up which orchestrator owns this tool
        if name not in tool_map:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "message": f"Unknown tool: {name}",
                }),
            )]

        orch_key, original_name = tool_map[name]
        orch = state[orch_key]
        autonomy = state["config"]["analysis_mode"]

        # Co-pilot / autopilot: intercept tools that need approval
        if _needs_approval(original_name, autonomy):
            prompt = _build_approval_prompt(original_name, arguments)
            state["job_counter"] += 1
            request_id = f"act_{state['job_counter']:04d}"
            state["pending"][request_id] = {
                "type": "tool_approval",
                "action": PendingAction(
                    tool_name=original_name,
                    kwargs=arguments,
                    prompt=prompt,
                    context={"orch_key": orch_key},
                ),
            }
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "needs_input",
                    "request_id": request_id,
                    "message": prompt,
                    "tool": original_name,
                    "arguments": arguments,
                }),
            )]

        # Background execution: if background=true and tool supports it,
        # submit to thread pool and return job_id immediately.
        run_in_background = arguments.pop("background", False)
        if run_in_background and original_name in _BACKGROUND_CAPABLE_TOOLS:
            state["job_counter"] += 1
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_id = f"job_{ts}_{state['job_counter']:03d}"

            log_lines = _new_job_lines()
            future = _executor.submit(
                _execute_tool_captured, orch.tools, original_name, arguments,
                state, job_id, log_lines,
            )
            _register_job(state, job_id, {
                "future": future,
                "tool": original_name,
                "started_at": ts,
                "status": "running",
                "result": None,
                "log_lines": log_lines,
            })

            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "started",
                    "job_id": job_id,
                    "tool": original_name,
                    "message": (
                        f"Analysis running in background (job {job_id}). "
                        "Use scilink_job_status to check progress, "
                        "then scilink_job_result to retrieve the result."
                    ),
                }),
            )]

        result = await _to_thread_streaming(
            _execute_tool_captured, orch.tools, original_name, arguments,
            state, None,
        )

        return [types.TextContent(type="text", text=result)]

    # ── resources/list ───────────────────────────────────────────────

    @server.list_resources()
    async def list_resources() -> List[types.Resource]:
        _ensure_initialized()
        resources = []

        if state["analysis_orch"]:
            resources.extend([
                types.Resource(
                    uri="scilink://analysis/status",
                    name="Analysis Session Status",
                    description="Current analysis session state",
                    mimeType="application/json",
                ),
                types.Resource(
                    uri="scilink://analysis/metadata",
                    name="Current Metadata",
                    description="Loaded sample/experiment metadata",
                    mimeType="application/json",
                ),
                types.Resource(
                    uri="scilink://analysis/agents",
                    name="Available Agents",
                    description="Registered analysis agents",
                    mimeType="application/json",
                ),
            ])

        if state["planning_orch"]:
            resources.extend([
                types.Resource(
                    uri="scilink://planning/status",
                    name="Planning Session Status",
                    description="Current planning session state",
                    mimeType="application/json",
                ),
                types.Resource(
                    uri="scilink://planning/plan",
                    name="Current Plan",
                    description="Active experimental plan",
                    mimeType="application/json",
                ),
            ])

        return resources

    # ── resources/read ───────────────────────────────────────────────

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        _ensure_initialized()

        # Analysis resources
        a_orch = state.get("analysis_orch")
        if uri == "scilink://analysis/status" and a_orch:
            return json.dumps({
                "current_data_path": getattr(a_orch, "current_data_path", None),
                "current_data_type": getattr(a_orch, "current_data_type", None),
                "selected_agent_id": getattr(a_orch, "selected_agent_id", None),
                "analysis_count": len(getattr(a_orch, "analysis_results", [])),
                "message_count": getattr(a_orch, "message_count", 0),
                "autonomy_mode": state["config"]["analysis_mode"],
            }, indent=2)

        if uri == "scilink://analysis/metadata" and a_orch:
            return json.dumps(
                getattr(a_orch, "current_metadata", None) or {}, indent=2
            )

        if uri == "scilink://analysis/agents" and a_orch:
            registry = getattr(a_orch, "_agent_registry", {})
            agents = {}
            for aid, entry in registry.items():
                agents[str(aid)] = {
                    "name": entry.get("name", ""),
                    "description": entry.get("description", ""),
                    "short_name": entry.get("short_name", ""),
                }
            return json.dumps(agents, indent=2)

        # Planning resources
        p_orch = state.get("planning_orch")
        if uri == "scilink://planning/status" and p_orch:
            return json.dumps({
                "has_plan": getattr(p_orch, "current_plan", None) is not None,
                "iteration": getattr(p_orch, "current_iteration", 0),
                "message_count": len(getattr(p_orch, "messages", [])),
                "autonomy_level": str(getattr(p_orch, "autonomy_level", "unknown")),
            }, indent=2)

        if uri == "scilink://planning/plan" and p_orch:
            plan = getattr(p_orch, "current_plan", None)
            if plan and hasattr(plan, "to_dict"):
                return json.dumps(plan.to_dict(), indent=2)
            return json.dumps(plan or {}, indent=2)

        # Backward compatibility with Phase 1 URIs
        if uri == "scilink://session/status" and a_orch:
            return await read_resource("scilink://analysis/status")
        if uri == "scilink://session/metadata" and a_orch:
            return await read_resource("scilink://analysis/metadata")
        if uri == "scilink://session/agents" and a_orch:
            return await read_resource("scilink://analysis/agents")

        return json.dumps({"error": f"Unknown resource: {uri}"})

    return server


# ── Orchestrator initialization ──────────────────────────────────────────

def _init_orchestrators(state: dict, config: dict) -> None:
    """Initialize orchestrator(s) based on config."""
    import os
    from datetime import datetime
    from pathlib import Path

    # MCP server manages trust via client — skip interactive sandbox prompt
    os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

    session_dir = config["session_dir"]
    if not session_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = str(Path.home() / "scilink_mcp_sessions" / f"session_{ts}")
    Path(session_dir).mkdir(parents=True, exist_ok=True)
    # Persisted job state (issue #469) lives here; record the resolved dir
    # so persistence works even when the caller left session_dir unset.
    config["session_dir"] = session_dir
    _restore_jobs(state)

    api_key = config["api_key"]
    model_name = config["model_name"]
    base_url = config["base_url"]
    fh_key = config["futurehouse_api_key"] or os.environ.get("FUTUREHOUSE_API_KEY")

    # Restart durability. Each orchestrator gets its own base dir (in
    # --mode both they nest as <session>/analysis and <session>/planning,
    # the meta's own layout — sharing one dir made their checkpoint.json
    # files clobber each other) and restores its checkpoint when one exists,
    # so a server restarted on the same --session-dir resumes the campaign
    # instead of starting a half-state next to the old files.
    def _base_for(kind: str) -> str:
        if config["mode"] == "both":
            d = Path(session_dir) / kind
            d.mkdir(parents=True, exist_ok=True)
            return str(d)
        return session_dir

    def _restore(base: str) -> bool:
        has = (Path(base) / "checkpoint.json").exists()
        if has:
            logging.info(f"Restoring session state from {base}/checkpoint.json")
        return has

    if config["mode"] in ("analyze", "both"):
        from scilink.agents.exp_agents.analysis_orchestrator import (
            AnalysisOrchestratorAgent, AnalysisMode,
        )
        mode_map = {
            "co-pilot": AnalysisMode.CO_PILOT,
            "co_pilot": AnalysisMode.CO_PILOT,
            "copilot": AnalysisMode.CO_PILOT,
            "autopilot": AnalysisMode.AUTOPILOT,
            "autonomous": AnalysisMode.AUTONOMOUS,
        }
        analysis_mode = mode_map.get(
            config["analysis_mode"].lower(), AnalysisMode.AUTONOMOUS
        )
        _abase = _base_for("analysis")
        state["analysis_orch"] = AnalysisOrchestratorAgent(
            base_dir=_abase,
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            analysis_mode=analysis_mode,
            futurehouse_api_key=fh_key,
            restore_checkpoint=_restore(_abase),
        )

    if config["mode"] in ("plan", "both"):
        try:
            from scilink.agents.planning_agents.planning_orchestrator import (
                PlanningOrchestratorAgent, AutonomyLevel,
            )
            autonomy_map = {
                "co-pilot": AutonomyLevel.CO_PILOT,
                "co_pilot": AutonomyLevel.CO_PILOT,
                "copilot": AutonomyLevel.CO_PILOT,
                "autopilot": AutonomyLevel.AUTOPILOT,
                "autonomous": AutonomyLevel.AUTONOMOUS,
            }
            autonomy_level = autonomy_map.get(
                config["analysis_mode"].lower(), AutonomyLevel.AUTONOMOUS
            )
            # Planning orchestrator needs data_dir and knowledge_dir
            # to avoid creating directories with relative paths
            # (fails when Claude Desktop runs from /).
            _pbase = _base_for("planning")
            data_dir = str(Path(_pbase) / "data")
            knowledge_dir = str(Path(_pbase) / "kb_storage")
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            Path(knowledge_dir).mkdir(parents=True, exist_ok=True)
            state["planning_orch"] = PlanningOrchestratorAgent(
                base_dir=_pbase,
                api_key=api_key,
                model_name=model_name,
                base_url=base_url,
                futurehouse_api_key=fh_key,
                autonomy_level=autonomy_level,
                data_dir=data_dir,
                knowledge_dir=knowledge_dir,
                restore_checkpoint=_restore(_pbase),
            )
        except Exception as exc:
            logging.warning(f"Planning orchestrator not available: {exc}")

    if config["mode"] == "meta":
        # The meta is its own surface (it constructs and routes to its
        # specialist children internally) — exposed INSTEAD of the
        # analysis/planning tool sets, mirroring how bare `scilink`
        # launches mission control. MetaMode has two levels: autonomous
        # serving maps to AUTONOMOUS, everything else to AUTOPILOT (the
        # meta has no co-pilot; children's prompts surface either way).
        from scilink.agents.meta_agent.meta_orchestrator import (
            MetaOrchestratorAgent, MetaMode,
        )
        meta_mode = (MetaMode.AUTONOMOUS
                     if config["analysis_mode"].lower() == "autonomous"
                     else MetaMode.AUTOPILOT)
        state["meta_orch"] = MetaOrchestratorAgent(
            base_dir=session_dir,
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            meta_mode=meta_mode,
            futurehouse_api_key=fh_key,
            restore_checkpoint=_restore(session_dir),
        )


# ── Background job handlers ──────────────────────────────────────────────

def _handle_job_status(
    state: dict, arguments: dict
) -> List["types.TextContent"]:
    """Check the status of a background job."""
    job_id = arguments.get("job_id", "")
    job = state["jobs"].get(job_id)
    if job is None:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "error",
                "message": f"Unknown job_id: {job_id}",
            }),
        )]

    future = job["future"]
    if future is not None and future.done():
        try:
            result = future.result()
            job["status"] = "completed"
            job["result"] = result
        except Exception as exc:
            job["status"] = "failed"
            job["result"] = json.dumps({
                "status": "error", "message": str(exc),
            })

    payload = {
        "job_id": job_id,
        "status": job["status"],
        "tool": job["tool"],
        "started_at": job["started_at"],
    }
    # Real-time narration tail: the job's recent log/print lines, so any
    # MCP client (including remote transports with no stderr access) can
    # watch progress in-band while the job runs.
    try:
        n = int(arguments.get("tail_lines", 25))
    except (TypeError, ValueError):
        n = 25
    n = max(0, min(200, n))
    lines = job.get("log_lines")
    if n and lines:
        payload["log_tail"] = "\n".join(list(lines)[-n:])
    if job["status"] == "interrupted":
        payload["message"] = _INTERRUPTED_HINT
    # A running job blocked on a human question reports awaiting_input
    # with the parked question(s) so the client can scilink_respond.
    if job["status"] == "running":
        questions = _pending_questions(state, job_id=job_id)
        if questions:
            payload["status"] = "awaiting_input"
            payload["questions"] = questions
            payload["message"] = (
                "The job is paused on a question. Answer with "
                "scilink_respond(request_id=..., response=...) — empty "
                "response usually means accept as-is.")

    # Heartbeat persistence: the narration buffer fills between transitions,
    # and a supervising client polls job_status regularly — mirror the tail
    # to disk here so a crash loses at most one polling interval of narration.
    if job.get("future") is not None:
        _persist_jobs(state)

    return [types.TextContent(type="text", text=json.dumps(payload))]


def _handle_job_result(
    state: dict, arguments: dict
) -> List["types.TextContent"]:
    """Retrieve the result of a completed background job."""
    job_id = arguments.get("job_id", "")
    job = state["jobs"].get(job_id)
    if job is None:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "error",
                "message": f"Unknown job_id: {job_id}",
            }),
        )]

    future = job["future"]
    if future is None:
        # Restored from a previous server process: the stored result is the
        # answer (completed/failed), or the interrupted re-entry hint.
        return [types.TextContent(type="text", text=job["result"] or json.dumps(
            {"status": job["status"], "job_id": job_id}))]
    if not future.done():
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "running",
                "job_id": job_id,
                "message": "Job is still running. Check again later with scilink_job_status.",
            }),
        )]

    # Ensure result is captured
    if job["result"] is None:
        try:
            job["result"] = future.result()
            job["status"] = "completed"
        except Exception as exc:
            job["result"] = json.dumps({
                "status": "error", "message": str(exc),
            })
            job["status"] = "failed"

    return [types.TextContent(type="text", text=job["result"])]


# ── Orchestrator chat handlers ──────────────────────────────────────────

def _execute_chat_captured(orch, prompt: str) -> str:
    """Execute orch.chat() while capturing stdout."""
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        result = orch.chat(prompt)

    log_output = captured.getvalue().strip()
    if log_output:
        logging.info(f"[orchestrate] {log_output}")

    return result


def _execute_run_task_captured(orch, prompt: str, autonomy_str: str,
                               state: dict = None,
                               job_id: str = None,
                               log_lines=None) -> str:
    """Execute orch.run_task() under the server's autonomy, stdout captured.

    Returns the structured run_task contract as JSON, with the legacy chat
    text preserved in the ``response`` field (run_task's ``summary`` IS the
    chat return string) so clients that read the text keep working. The
    per-call ``autonomy`` override means co-pilot/autopilot serving works
    without mutating the orchestrator's resting mode, and prompts raised
    mid-run park in the server's pending queue via the thread channel.
    """
    from scilink.hitl import set_thread_channel

    level_name = {"co-pilot": "CO_PILOT", "co_pilot": "CO_PILOT",
                  "copilot": "CO_PILOT", "autopilot": "AUTOPILOT",
                  "autonomous": "AUTONOMOUS"}.get(
                      (autonomy_str or "autonomous").lower(), "AUTONOMOUS")
    mode_enum = type(orch).__module__  # resolved below per orchestrator
    if "planning_orchestrator" in mode_enum:
        from scilink.agents.planning_agents.planning_orchestrator import (
            AutonomyLevel as _Level,
        )
    else:
        from scilink.agents.exp_agents.analysis_orchestrator import (
            AnalysisMode as _Level,
        )
    autonomy = _Level[level_name]

    if state is not None:
        set_thread_channel(_MCPChannel(
            state, job_id=job_id,
            timeout_s=state["config"].get("hitl_timeout_s", 1800.0)))
    captured = _TeeIO(log_lines) if log_lines is not None else io.StringIO()
    try:
        with _job_narration(log_lines), contextlib.redirect_stdout(captured):
            res = orch.run_task(prompt, autonomy=autonomy)
    finally:
        if state is not None:
            set_thread_channel(None)

    log_output = captured.getvalue().strip()
    if log_output:
        logging.info(f"[orchestrate] {log_output}")

    out = dict(res)
    out["response"] = res.get("summary", "")
    return json.dumps(out, default=str)


def _execute_meta_chat_captured(orch, prompt: str, autonomy_str: str,
                                state: dict = None, job_id: str = None,
                                log_lines=None) -> str:
    """Run one meta-orchestrator chat turn under the server's autonomy.

    The meta has no run_task (it IS the top-level caller), so this runs
    ``chat()`` with the mode set for the call and restored after —
    AUTONOMOUS serving maps to MetaMode.AUTONOMOUS, autopilot/co-pilot to
    MetaMode.AUTOPILOT (the meta's only human-attached level). Prompts
    raised anywhere in the tree — the meta's own confirms, delegated
    children's plan reviews — park in the server's pending queue via the
    thread channel and surface as awaiting_input.
    """
    from scilink.agents.meta_agent.meta_orchestrator import MetaMode
    from scilink.hitl import set_thread_channel

    run_mode = (MetaMode.AUTONOMOUS
                if (autonomy_str or "autonomous").lower() == "autonomous"
                else MetaMode.AUTOPILOT)
    original_mode = orch.meta_mode

    if state is not None:
        set_thread_channel(_MCPChannel(
            state, job_id=job_id,
            timeout_s=state["config"].get("hitl_timeout_s", 1800.0)))
    captured = _TeeIO(log_lines) if log_lines is not None else io.StringIO()
    try:
        orch.set_meta_mode(run_mode)
        with _job_narration(log_lines), contextlib.redirect_stdout(captured):
            response = orch.chat(prompt)
    finally:
        orch.set_meta_mode(original_mode)
        if state is not None:
            set_thread_channel(None)

    log_output = captured.getvalue().strip()
    if log_output:
        logging.info(f"[orchestrate_meta] {log_output}")

    is_error = response.strip().startswith("❌")
    return json.dumps({
        "status": "error" if is_error else "success",
        "response": response,
    }, default=str)


async def _handle_orchestrate_meta(
    state: dict, arguments: dict, executor
) -> List["types.TextContent"]:
    """Handle the meta-orchestrator chat tool."""
    from datetime import datetime as _dt

    orch = state.get("meta_orch")
    if orch is None:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "error",
                "message": ("Meta orchestrator is not available. Start the "
                            "server with --mode meta."),
            }),
        )]

    prompt = arguments.get("prompt", "")
    if not prompt.strip():
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "error",
                "message": "Empty prompt. Provide a natural-language instruction.",
            }),
        )]

    run_in_background = arguments.pop("background", False)
    autonomy_str = state["config"]["analysis_mode"]

    if run_in_background:
        state["job_counter"] += 1
        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        job_id = f"job_{ts}_{state['job_counter']:03d}"

        log_lines = _new_job_lines()
        future = executor.submit(_execute_meta_chat_captured, orch, prompt,
                                 autonomy_str, state, job_id, log_lines)
        _register_job(state, job_id, {
            "future": future,
            "tool": "orchestrate_meta",
            "started_at": ts,
            "status": "running",
            "result": None,
            "log_lines": log_lines,
        })

        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "started",
                "job_id": job_id,
                "tool": "orchestrate_meta",
                "message": (
                    f"Meta orchestrator running in background (job {job_id}). "
                    "Use scilink_job_status to watch progress (log_tail) and "
                    "answer its questions; scilink_job_result for the result."
                ),
            }),
        )]

    result = await _to_thread_streaming(_execute_meta_chat_captured, orch,
                                        prompt, autonomy_str, state, None)
    return [types.TextContent(type="text", text=result)]


async def _handle_orchestrate(
    state: dict, name: str, arguments: dict, executor
) -> List["types.TextContent"]:
    """Handle orchestrator-level chat tools."""
    from datetime import datetime as _dt

    orch_key = (
        "analysis_orch" if name == "scilink_orchestrate_analysis"
        else "planning_orch"
    )
    orch = state.get(orch_key)
    if orch is None:
        mode_label = "analysis" if "analysis" in name else "planning"
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "error",
                "message": (
                    f"{mode_label.title()} orchestrator is not available. "
                    f"Start the server with --mode {mode_label} or --mode both."
                ),
            }),
        )]

    prompt = arguments.get("prompt", "")
    if not prompt.strip():
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "error",
                "message": "Empty prompt. Provide a natural-language instruction.",
            }),
        )]

    run_in_background = arguments.pop("background", False)
    tool_label = name.replace("scilink_", "")

    autonomy_str = state["config"]["analysis_mode"]

    if run_in_background:
        state["job_counter"] += 1
        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        job_id = f"job_{ts}_{state['job_counter']:03d}"

        log_lines = _new_job_lines()
        future = executor.submit(_execute_run_task_captured, orch, prompt,
                                 autonomy_str, state, job_id, log_lines)
        _register_job(state, job_id, {
            "future": future,
            "tool": tool_label,
            "started_at": ts,
            "status": "running",
            "result": None,
            "log_lines": log_lines,
        })

        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "started",
                "job_id": job_id,
                "tool": tool_label,
                "message": (
                    f"Orchestrator running in background (job {job_id}). "
                    "Use scilink_job_status to check progress, "
                    "then scilink_job_result to retrieve the result."
                ),
            }),
        )]

    result = await _to_thread_streaming(_execute_run_task_captured, orch,
                                        prompt, autonomy_str, state, None)
    return [types.TextContent(type="text", text=result)]


# ── Autonomy mode switch ─────────────────────────────────────────────────

def _handle_set_autonomy(
    state: dict, arguments: dict
) -> List["types.TextContent"]:
    """Switch autonomy mode at runtime."""
    new_mode = arguments.get("mode", "").strip().lower()
    valid = {"autonomous", "autopilot", "co-pilot"}
    if new_mode not in valid:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "error",
                "message": f"Invalid mode '{new_mode}'. Use: {', '.join(sorted(valid))}",
            }),
        )]

    old_mode = state["config"]["analysis_mode"]
    state["config"]["analysis_mode"] = new_mode

    # Propagate to the LIVE orchestrators too — the server-side gate and
    # the agents' own human-feedback behavior must agree (previously this
    # only mutated the config dict, leaving the orchestrators in their
    # construction-time mode).
    _level_name = {"co-pilot": "CO_PILOT", "autopilot": "AUTOPILOT",
                   "autonomous": "AUTONOMOUS"}[new_mode]
    if state.get("analysis_orch") is not None:
        try:
            from scilink.agents.exp_agents.analysis_orchestrator import (
                AnalysisMode,
            )
            state["analysis_orch"].set_analysis_mode(
                AnalysisMode[_level_name])
        except Exception as exc:  # noqa: BLE001
            logging.warning(f"set_autonomy: analysis orch not updated: {exc}")
    if state.get("planning_orch") is not None:
        try:
            from scilink.agents.planning_agents.planning_orchestrator import (
                AutonomyLevel,
            )
            state["planning_orch"].set_autonomy_level(
                AutonomyLevel[_level_name])
        except Exception as exc:  # noqa: BLE001
            logging.warning(f"set_autonomy: planning orch not updated: {exc}")
    if state.get("meta_orch") is not None:
        try:
            from scilink.agents.meta_agent.meta_orchestrator import MetaMode
            state["meta_orch"].set_meta_mode(
                MetaMode.AUTONOMOUS if new_mode == "autonomous"
                else MetaMode.AUTOPILOT)
        except Exception as exc:  # noqa: BLE001
            logging.warning(f"set_autonomy: meta orch not updated: {exc}")

    # Clear pending tool approvals from the previous mode; parked agent
    # questions belong to running jobs and stay answerable.
    for rid in [r for r, it in state.get("pending", {}).items()
                if it.get("type") == "tool_approval"]:
        state["pending"].pop(rid, None)

    return [types.TextContent(
        type="text",
        text=json.dumps({
            "status": "success",
            "previous_mode": old_mode,
            "current_mode": new_mode,
            "approval_required_for": sorted(
                _COPILOT_APPROVAL_TOOLS if new_mode == "co-pilot"
                else _AUTOPILOT_APPROVAL_TOOLS if new_mode == "autopilot"
                else set()
            ),
            "scilink_respond_needed": new_mode != "autonomous",
        }),
    )]


# ── Respond handler (co-pilot / autopilot) ──────────────────────────────

async def _handle_respond(
    state: dict, arguments: dict
) -> List["types.TextContent"]:
    """Answer a pending tool approval or a parked agent question."""
    pending = state.get("pending", {})
    request_id = (arguments.get("request_id") or "").strip()

    if request_id and request_id not in pending:
        dead = (state.get("interrupted_questions") or {}).get(request_id)
        if dead:
            return [types.TextContent(type="text", text=json.dumps({
                "status": "interrupted",
                "request_id": request_id,
                "message": (
                    "This question belonged to a job interrupted by a server "
                    "restart; its asking thread is gone, so the answer cannot "
                    "be delivered. " + _INTERRUPTED_HINT),
                "question": dead,
            }))]
    if request_id:
        item = pending.get(request_id)
        if item is None:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "message": f"Unknown request_id: {request_id}",
                    "pending": _pending_questions(state) + [
                        {"request_id": rid, "kind": "tool_approval",
                         "tool": it["action"].tool_name}
                        for rid, it in pending.items()
                        if it.get("type") == "tool_approval"],
                }),
            )]
    elif len(pending) == 1:
        request_id, item = next(iter(pending.items()))
    elif not pending:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "error",
                "message": "No pending request to respond to.",
            }),
        )]
    else:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "error",
                "message": ("Multiple requests pending — pass request_id "
                            "to pick one."),
                "pending": _pending_questions(state) + [
                    {"request_id": rid, "kind": "tool_approval",
                     "tool": it["action"].tool_name}
                    for rid, it in pending.items()
                    if it.get("type") == "tool_approval"],
            }),
        )]

    raw_response = arguments.get("response", "")

    # Agent question: deliver the response VERBATIM — free text is the
    # answer (the asking code interprets it), never an implicit cancel.
    if item.get("type") == "question":
        item["holder"]["answer"] = raw_response
        item["event"].set()
        # The asking thread pops the entry itself on wake-up.
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "answered",
                "request_id": request_id,
                "kind": item["req"].kind,
                "message": ("Answer delivered; the run resumes. Poll "
                            "scilink_job_status for progress if this was "
                            "a background job."),
            }),
        )]

    # Tool approval: keyword gate (unchanged semantics), but a cancel now
    # echoes the feedback instead of silently discarding it.
    action = item["action"]
    state["pending"].pop(request_id, None)
    response = raw_response.strip().lower()

    if response in ("yes", "y", "approve", "ok", "proceed"):
        orch_key = action.context.get("orch_key", "analysis_orch")
        orch = state.get(orch_key)
        if orch is None:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "message": "Orchestrator not found.",
                }),
            )]

        result = await _to_thread_streaming(
            _execute_tool_captured, orch.tools, action.tool_name,
            action.kwargs, state, None,
        )
        return [types.TextContent(type="text", text=result)]

    return [types.TextContent(
        type="text",
        text=json.dumps({
            "status": "cancelled",
            "message": f"Action cancelled by user: {raw_response}",
            "tool": action.tool_name,
            "note": ("Free-text feedback on a tool approval cancels the "
                     "call — re-issue the tool call with adjusted "
                     "arguments to apply the feedback."),
        }),
    )]


# ── Server runner ────────────────────────────────────────────────────────

async def run_stdio(server: Server, real_stdout=None) -> None:
    """Run the MCP server over stdio transport.

    Args:
        server: The configured MCP Server.
        real_stdout: The original ``sys.stdout`` before redirection.
            Needed because the CLI redirects ``sys.stdout`` to stderr
            to protect the JSON-RPC stream from stray ``print()`` calls.
    """
    _require_mcp()
    import anyio
    from io import TextIOWrapper

    stdout_arg = None
    if real_stdout is not None:
        stdout_arg = anyio.wrap_file(
            TextIOWrapper(real_stdout.buffer, encoding="utf-8")
        )

    async with stdio_server(stdout=stdout_arg) as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream,
            server.create_initialization_options(),
        )


def run_sse(server: Server, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the MCP server over SSE (Server-Sent Events) transport.

    Starts an HTTP server with two endpoints:

    - ``GET /sse`` — SSE stream for client connections
    - ``POST /messages/`` — message submission endpoint

    Args:
        server: The configured MCP Server.
        host: Bind address (default: ``127.0.0.1``).
        port: Bind port (default: ``8000``).
    """
    _require_mcp()

    try:
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Route, Mount
        from starlette.responses import Response
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            f"SSE transport requires additional packages: {exc}\n"
            "Install with: pip install uvicorn starlette sse-starlette"
        ) from exc

    sse_transport = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.run(
                streams[0], streams[1],
                server.create_initialization_options(),
            )
        return Response()

    from starlette.middleware import Middleware
    from starlette.middleware.cors import CORSMiddleware

    app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse, methods=["GET"]),
            Mount("/messages/", app=sse_transport.handle_post_message),
        ],
        middleware=[
            Middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["*"],
                allow_headers=["*"],
            ),
        ],
    )

    logging.info(f"SciLink MCP server (SSE) at http://{host}:{port}/sse")
    uvicorn.run(app, host=host, port=port, log_level="warning")
