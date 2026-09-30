"""Human-in-the-loop chokepoint.

Every agent-level human-feedback prompt in SciLink routes through
``request_human_feedback`` instead of calling ``input()`` directly. The
default channel simply calls ``builtins.input`` (resolved at call time, so
front-ends that intercept ``builtins.input`` keep working), which makes the
console behavior byte-identical to the pre-chokepoint code. Front-ends can
install richer channels — process-wide via ``set_default_channel`` or for
the current thread via ``set_thread_channel`` — to serve prompts through a
UI, a queue, or a remote transport.

Out of scope by design: CLI session-bootstrap/REPL prompts, destructive-op
confirmations in ``scilink memory`` / ``scilink kb``, and the sandbox
consent prompt in ``executors.py`` (TTY-gated security surface). Those are
not agent decision points and keep calling ``input()`` directly.
"""

from __future__ import annotations

import builtins
import itertools
import queue as _queue_mod
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

__all__ = [
    "FeedbackRequest",
    "SUBJECT_BLOCKS",
    "subject_block",
    "make_subject",
    "FeedbackChannel",
    "ConsoleChannel",
    "QueueChannel",
    "QuestionServer",
    "WorkerChannel",
    "question_timeout_s",
    "last_question_timed_out",
    "mark_timed_out",
    "unattended_questions",
    "request_human_feedback",
    "get_channel",
    "set_default_channel",
    "set_thread_channel",
    "use_channel",
    "set_thread_feedback_log",
    "get_thread_feedback_log",
    "use_feedback_log",
]

_counter = itertools.count(1)


def _next_id() -> str:
    return f"q_{int(time.time())}_{next(_counter):04d}"


# ── the subject block vocabulary ─────────────────────────────────
# What a question shows, as data the front-ends render. Fixed on purpose,
# like the skill section vocabulary: a gate authors against these shapes
# and both surfaces know what to do with each. Every block may carry a
# ``label`` — its section heading, shown in an aligned label column (keep
# the console's emoji on it: "🔍 Observations"). Payload keys per type:
#   text        markdown
#   fields      items: [{label, value, unit?, flag? (ok|warn|bad)}]
#   chips       label, items: [str]
#   steps       label, items: [str]
#   table       columns: [str], rows: [[cell, ...]], caption?
#   figure      path (absolute at the gate; presented relative to the
#               session), caption?
#   candidates  items: [{idx, name, metric?, value?, approved?, figure?,
#               judge_comment?, body? (markdown), report? (a file to open:
#               the candidate's full plan)}], pick (None when no
#               candidate is preferred), reasoning?, caveats?: [str],
#               free_text?: {input, submit} — a typed reply the gate also
#               takes (a model name, 'more'), shown as a box under the picker
#   compare     left: {label, blocks}, right: {label, blocks}
#   notice      title, lines: [str], tone? (info|warn)
SUBJECT_BLOCKS = ("text", "fields", "chips", "steps", "table", "figure",
                  "candidates", "compare", "notice")


def subject_block(type_: str, **payload: Any) -> Dict[str, Any]:
    """One subject block; ``type_`` must be in ``SUBJECT_BLOCKS``."""
    if type_ not in SUBJECT_BLOCKS:
        raise ValueError(f"unknown subject block {type_!r}; one of "
                         f"{', '.join(SUBJECT_BLOCKS)}")
    return {"type": type_, **payload}


def make_subject(title: str, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A question's subject: a title and its blocks (empty blocks dropped)."""
    return {"title": title, "blocks": [b for b in blocks if b]}


@dataclass
class FeedbackRequest:
    """A structured human-feedback question.

    ``kind`` identifies the decision type so channels can render the right
    surface (a radio for ``bestofn_select``, buttons for ``confirm``, a text
    box for ``free_text``) without sniffing the prompt string. ``default``
    is the answer that means "accept as-is" — for almost every SciLink
    prompt that is the empty string (press Enter). ``origin`` carries
    routing metadata (agent label, pipeline stage, fan-out branch thread
    id) and must stay JSON-serializable.

    ``subject`` is WHAT is under review, as data: ``{"title": str,
    "blocks": [block, ...]}`` with each block one of ``SUBJECT_BLOCKS``
    (see ``subject_block``). A gate that sets it still prints what it
    always printed — the console, the verbose log and the record — but the
    web UI and the terminal shell render the blocks instead of the captured
    console text, and the decision widget comes from ``kind``. Gates
    without a subject are presented from their printed text as before.
    """

    prompt: str
    kind: str = "free_text"
    options: Optional[List[str]] = None
    default: str = ""
    context: str = ""
    origin: Dict[str, Any] = field(default_factory=dict)
    subject: Optional[Dict[str, Any]] = None
    id: str = field(default_factory=_next_id)
    created_at: float = field(default_factory=time.time)


@runtime_checkable
class FeedbackChannel(Protocol):
    def ask(self, req: FeedbackRequest) -> str:  # pragma: no cover - protocol
        ...


class ConsoleChannel:
    """Blocking console prompt — the process default.

    Calls ``builtins.input`` dynamically so a front-end that monkeypatches
    ``builtins.input`` (the Streamlit UI does, until it installs a channel
    of its own) still intercepts every prompt.
    """

    def ask(self, req: FeedbackRequest) -> str:
        return builtins.input(req.prompt)


#: How long a worker waits for a person before it goes on with the gate's own
#: default. Long enough for someone at the screen to read and answer; short
#: enough that an unattended run is not held (and its budget spent) forever.
QUESTION_TIMEOUT_S = 1800.0


#: Longest timeout honoured (a week): ``Event.wait`` overflows on larger values.
_QUESTION_TIMEOUT_MAX_S = 7 * 24 * 3600.0
#: How often a parked worker looks up from its wait to check for a cancel.
_WAIT_SLICE_S = 1.0


def question_timeout_s() -> Optional[float]:
    """The worker question timeout: ``SCILINK_QUESTION_TIMEOUT_S`` seconds,
    else :data:`QUESTION_TIMEOUT_S`. ``0`` (or ``none``) waits without limit
    — which removes the only bound on a parked worker; anything unreadable
    or negative is the default, and a week is the most honoured."""
    import math
    import os
    raw = (os.environ.get("SCILINK_QUESTION_TIMEOUT_S") or "").strip().lower()
    if not raw:
        return QUESTION_TIMEOUT_S
    if raw in ("0", "none", "off"):
        return None
    try:
        value = float(raw)
    except ValueError:
        return QUESTION_TIMEOUT_S
    if math.isnan(value) or value < 0:
        return QUESTION_TIMEOUT_S
    if value == 0:
        return None
    return min(value, _QUESTION_TIMEOUT_MAX_S)


class QueueChannel:
    """Parks requests from concurrent worker threads for serial serving.

    Worker threads (fan-out branches, swarm workers) install this, usually
    behind a :class:`WorkerChannel`, as their thread channel; ``ask`` parks
    the request and blocks until a coordinator thread — the one that owns
    the human — answers it via ``serve_pending``, which relays each parked
    request through the coordinator's own active channel (console prompt, UI
    modal, ...) one at a time. ``pending()`` says who is waiting, and on what.

    ``timeout_s`` bounds how long a worker waits: at most that long in the
    queue behind other workers, and at most that long again once its question
    is in front of the person (the clock restarts when serving begins, so a
    question being read is not pulled away because others were read first).
    On timeout the request's ``default`` answer is returned, the question is
    withdrawn (a person is never asked something nobody waits for any more)
    and the worker's feedback log records ``timed_out`` — the gate that asked
    can tell through ``last_question_timed_out()`` and must not record the
    default as a human's decision. A worker whose item is cancelled while it
    waits (its stop event, or the turn's Stop) withdraws its question and
    raises, like a print would.
    """

    def __init__(self, timeout_s: Optional[float] = None) -> None:
        self._items: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self.timeout_s = timeout_s
        self.closed: Optional[str] = None      # why nobody will answer any more

    def close(self, reason: str = "closed") -> None:
        """Nobody will serve this queue any more (the person's channel raised,
        the coordinator returned): every waiting worker gets its default now,
        as unattended, and later asks do too instead of waiting out a timeout."""
        with self._lock:
            self.closed = reason
            waiting = [it for it in self._items if it["state"] in ("waiting", "serving")]
        for it in waiting:
            self._finish(it, it["req"].default, unattended=True)

    def ask(self, req: FeedbackRequest) -> str:
        from .utils.log_context import raise_if_cancelled
        item: Dict[str, Any] = {"req": req, "event": threading.Event(),
                                "state": "waiting", "answer": None, "served_at": None}
        with self._lock:
            if self.closed:
                item["state"], item["unattended"] = "answered", True
                item["event"].set()
            else:
                self._items.append(item)
        timeout = self.timeout_s
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            try:
                raise_if_cancelled()
            except BaseException:
                self._withdraw(item, "cancelled")
                raise
            slice_s = (_WAIT_SLICE_S if deadline is None
                       else max(0.0, min(_WAIT_SLICE_S, deadline - time.monotonic())))
            if item["event"].wait(slice_s):
                if item.get("unattended"):
                    # The serving channel (or the queue's closing) gave the
                    # default because nobody answered: not a decision.
                    mark_timed_out()
                    log = get_thread_feedback_log()
                    if log:
                        _append_record(log, {"id": req.id, "event": "timed_out",
                                             "default": req.default, "by": "channel",
                                             "shown": item["served_at"] is not None})
                    return req.default
                return item["answer"] if item["answer"] is not None else req.default
            if deadline is None:
                continue
            now = time.monotonic()
            if now < deadline:
                continue
            with self._lock:
                if item["event"].is_set():
                    continue
                served_at = item["served_at"]
                if served_at is not None and served_at + timeout > now:
                    deadline = served_at + timeout      # a full clock from when it was shown
                    continue
            if self._withdraw(item, "timed_out"):
                mark_timed_out()
                log = get_thread_feedback_log()
                if log:
                    _append_record(log, {"id": req.id, "event": "timed_out",
                                         "default": req.default, "after_s": timeout,
                                         "shown": item["served_at"] is not None})
                return req.default

    def _withdraw(self, item: Dict[str, Any], state: str) -> bool:
        """Take the question back. False when it was answered meanwhile."""
        with self._lock:
            if item["event"].is_set():
                return False
            item["state"] = state
            if item in self._items:
                self._items.remove(item)
            return True

    def pending(self) -> List[Dict[str, Any]]:
        """The questions waiting now, oldest first: who asks, about what."""
        with self._lock:
            items = list(self._items)
        return [{"id": it["req"].id, "worker": it["req"].origin.get("branch_label"),
                 "subject": it["req"].origin.get("work_subject"),
                 "kind": it["req"].kind, "asked_at": it["req"].created_at,
                 "being_answered": it["state"] == "serving"} for it in items]

    def serve_pending(self, through: Optional[FeedbackChannel] = None) -> int:
        """Serve every parked request now; returns how many were answered.

        Called periodically from the coordinator's wait loop. The asker
        (``origin['branch_label']``, and the subject it works on) is prefixed
        onto the prompt so the human knows who is asking. If the serving
        channel raises (EOF, stop, interrupt), the waiting worker is unblocked
        with the request's default answer before the exception propagates.
        """
        served = 0
        while True:
            with self._lock:
                item = next((it for it in self._items if it["state"] == "waiting"), None)
                if item is None:
                    return served
                item["state"], item["served_at"] = "serving", time.monotonic()
            req = item["req"]
            to_serve = replace(req, prompt=_asker_prefix(req.origin) + req.prompt)
            _thread_local.last_timed_out = False
            try:
                answer = (through or get_channel()).ask(to_serve)
            except BaseException:
                # The person's channel is gone (EOF, Stop): the worker gets
                # its default, but as unattended, never as an acceptance.
                self._finish(item, req.default, unattended=True)
                raise
            # A channel with a timeout of its own (the MCP server's) says so
            # through mark_timed_out(); that travels to the asking worker.
            if not self._finish(item, answer, unattended=last_question_timed_out()):
                print("  ⏱  that worker stopped waiting and went on without an answer; "
                      "this one was not used.")
            served += 1

    def _finish(self, item: Dict[str, Any], answer: str, unattended: bool = False) -> bool:
        """Hand ``answer`` to the waiting worker. False when it already gave up."""
        with self._lock:
            if item in self._items:
                self._items.remove(item)
            if item["state"] in ("timed_out", "cancelled") or item["event"].is_set():
                return False
            item["state"], item["answer"] = "answered", answer
            if unattended:
                item["unattended"] = True
            item["event"].set()
            return True


class QuestionServer:
    """Serves a :class:`QueueChannel` to the person on a thread of its own.

    The coordinator that owns the human must keep polling its workers (memory
    guard, budgets, completions); a question shown to the person blocks until
    they answer, so it is shown from here, not from the poll loop. The serving
    channel is captured on the coordinator's thread (a thread-local override
    would not be seen from the server thread) and the thread is attributed to
    the coordinator's session so its prompts reach the same screen.

    Used as a context manager around the coordinator's wait loop. On exit the
    server stops taking new questions; a question already in front of the
    person stays until they answer it (the thread is a daemon), and the
    worker that asked has usually gone on with its default by then.
    """

    def __init__(self, queue: QueueChannel, through: Optional[FeedbackChannel] = None,
                 poll_s: float = 0.25) -> None:
        self._queue, self._through, self._poll_s = queue, through or get_channel(), poll_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.served = 0
        self.error: Optional[BaseException] = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.served += self._queue.serve_pending(through=self._through)
            except BaseException as exc:  # noqa: BLE001 - the person's channel raised (EOF, Stop)
                self.error = exc
                # Nobody will answer from here on: the workers get their
                # defaults now, as unattended, instead of waiting out a timeout.
                self._queue.close(f"the person's channel raised {type(exc).__name__}")
                return
            self._stop.wait(self._poll_s)

    def __enter__(self) -> "QuestionServer":
        from .utils.log_context import start_attributed_thread
        self._thread = start_attributed_thread(self._run, name="question-server")
        return self

    def __exit__(self, *exc) -> bool:
        self._stop.set()
        self._queue.close("the coordinator returned")
        return False


def mark_timed_out() -> None:
    """A channel that gave a question's default because nobody answered in
    time calls this (on the asking thread) before returning it, so the gate
    can tell (``last_question_timed_out``) and never records the default as a
    human's decision."""
    _thread_local.last_timed_out = True


def unattended_questions() -> int:
    """How many questions asked on this thread so far got their default
    because nobody answered in time. A worker reports the change over its
    run as a warning on its result."""
    return int(getattr(_thread_local, "unattended", 0) or 0)


def last_question_timed_out() -> bool:
    """Whether the most recent question asked on this thread got its default
    because nobody answered in time. A gate that treats an empty answer as
    approval checks this before recording a human decision."""
    return bool(getattr(_thread_local, "last_timed_out", False))


def _asker_prefix(origin: Dict[str, Any]) -> str:
    label = origin.get("branch_label")
    if not label:
        return ""
    subject = origin.get("work_subject")
    return f"\n[{origin.get('worker_kind', 'branch')}: {label}{f' · {subject}' if subject else ''}]"


class WorkerChannel:
    """A worker thread's channel: tags each request with the worker asking
    (and the subject it works on), then parks it on the coordinator's queue.

    ``kind`` names what the worker is on the prompt ("branch" for a fan-out
    branch, "worker" for a swarm item). ``on_wait(True)`` / ``on_wait(False)``
    bracket every wait for the person, so a coordinator can leave that time
    out of the worker's wall-clock budget."""

    def __init__(self, queue_channel: QueueChannel, label: str, *,
                 subject: Optional[str] = None, kind: str = "branch",
                 on_wait: Optional[Any] = None) -> None:
        self._qch, self._label, self._subject, self._kind = queue_channel, label, subject, kind
        self._on_wait = on_wait

    def ask(self, req: FeedbackRequest) -> str:
        req.origin.setdefault("branch_label", self._label)
        req.origin.setdefault("worker_kind", self._kind)
        if self._subject:
            req.origin.setdefault("work_subject", self._subject)
        if self._on_wait is not None:
            self._on_wait(True)
        try:
            return self._qch.ask(req)
        finally:
            if self._on_wait is not None:
                self._on_wait(False)


_default_channel: FeedbackChannel = ConsoleChannel()
_thread_local = threading.local()


def set_default_channel(channel: Optional[FeedbackChannel]) -> None:
    """Install the process-wide channel (``None`` restores the console)."""
    global _default_channel
    _default_channel = channel if channel is not None else ConsoleChannel()


def set_thread_channel(channel: Optional[FeedbackChannel]) -> None:
    """Install (or with ``None`` clear) the channel for the current thread.

    Thread-local overrides win over the process default; fan-out branches
    and UI worker threads use this so concurrent prompts route to their
    own owner instead of colliding on one console.
    """
    _thread_local.channel = channel


@contextmanager
def use_channel(channel: FeedbackChannel):
    """Scoped thread-local channel override."""
    previous = getattr(_thread_local, "channel", None)
    _thread_local.channel = channel
    try:
        yield channel
    finally:
        _thread_local.channel = previous


def get_channel() -> FeedbackChannel:
    channel = getattr(_thread_local, "channel", None)
    return channel if channel is not None else _default_channel


# --------------------------------------------------------------- durability

def set_thread_feedback_log(path) -> None:
    """Bind this thread's prompts to a session feedback log.

    ``path`` is the JSONL file (conventionally
    ``<session>/feedback_log.jsonl``); every request asked on this thread
    is appended at ask time and again at answer time, and a
    ``pending_question.json`` sidecar next to the log marks the question
    currently awaiting an answer (removed once answered — if a run dies
    while blocked, the sidecar says what it was waiting for). ``None``
    unbinds. Orchestrators bind their session file around each chat turn.
    """
    _thread_local.feedback_log = path


def get_thread_feedback_log():
    return getattr(_thread_local, "feedback_log", None)


@contextmanager
def use_feedback_log(path):
    """Scoped thread-local feedback-log binding (re-entrant safe)."""
    previous = get_thread_feedback_log()
    _thread_local.feedback_log = path
    try:
        yield
    finally:
        _thread_local.feedback_log = previous


def _append_record(path, record: Dict[str, Any]) -> None:
    """Best-effort append — the log must never break the prompt itself."""
    import json

    try:
        p = __import__("pathlib").Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:  # noqa: BLE001
        pass


def _write_pending(path, req: FeedbackRequest) -> None:
    import json

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"id": req.id, "kind": req.kind, "prompt": req.prompt,
                       "origin": req.origin, "subject": req.subject,
                       "asked_at": req.created_at},
                      f, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        pass


def _clear_pending(path) -> None:
    try:
        __import__("pathlib").Path(path).unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


def request_human_feedback(
    prompt: str,
    *,
    kind: str = "free_text",
    options: Optional[List[str]] = None,
    default: str = "",
    context: str = "",
    origin: Optional[Dict[str, Any]] = None,
    subject: Optional[Dict[str, Any]] = None,
) -> str:
    """Ask the human a question through the active feedback channel.

    Returns the raw answer string exactly as ``input()`` would (no
    stripping — call sites keep their own normalization so behavior stays
    identical). Exceptions from the channel (``EOFError``,
    ``KeyboardInterrupt``, stop signals) propagate to the call site, which
    keeps its existing fallback handling.
    """
    req = FeedbackRequest(
        prompt=prompt,
        kind=kind,
        options=options,
        default=default,
        context=context,
        origin=origin or {},
        subject=subject,
    )
    log = get_thread_feedback_log()
    pending = None
    if log:
        from pathlib import Path as _Path

        _append_record(log, {"id": req.id, "event": "asked",
                             "kind": req.kind, "prompt": req.prompt,
                             "options": req.options, "default": req.default,
                             "origin": req.origin, "subject": req.subject,
                             "t": req.created_at})
        pending = _Path(log).parent / "pending_question.json"
        _write_pending(pending, req)
    _thread_local.last_timed_out = False
    try:
        answer = get_channel().ask(req)
        if last_question_timed_out():
            _thread_local.unattended = unattended_questions() + 1
    except BaseException as exc:
        if log:
            _append_record(log, {
                "id": req.id, "event": "interrupted",
                "error": type(exc).__name__,
                "elapsed_s": round(time.time() - req.created_at, 3)})
        raise
    finally:
        if pending is not None:
            _clear_pending(pending)
    if log:
        _append_record(log, {
            "id": req.id, "event": "answered", "answer": answer,
            "elapsed_s": round(time.time() - req.created_at, 3),
            **({"unattended": True} if last_question_timed_out() else {})})
    return answer
