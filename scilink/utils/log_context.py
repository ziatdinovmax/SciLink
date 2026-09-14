"""Worker-thread log attribution for best-of-N candidate fan-outs.

Two problems when anchor attempts run in ThreadPoolExecutor workers:

1. The Streamlit UI's verbose panel filters log records by thread id (the
   background chat thread), so candidate attempts' detailed narration
   (plan-deviation justifications, verification recommendations, refinement
   reasoning) silently disappears from the panel.
2. In the CLI console all workers' lines interleave without attribution —
   two candidates print identical "Verification 2/7" lines.

A fan-out worker registers itself with its parent (chat) thread id and a
candidate tag. ``effective_thread`` lets the UI filter treat worker records
as belonging to the parent thread, and a root-logger filter prefixes each
registered worker's messages with its candidate tag so the interleave is
attributable everywhere (UI, CLI, log files).

The same mapping is the ONLY way a background thread's ``print()`` reaches
the web app: ``scilink.server.stdout_router`` routes each write by the
calling thread's *effective* thread, so an unregistered helper thread's
output lands on the server console and nowhere else (issue #627 — the
``search_literature`` heartbeat read as a 10-minute hang in the browser).
Any thread that prints or logs on a session's behalf must therefore inherit
the session's route: use ``start_attributed_thread`` for a fire-and-forget
helper and ``attributed_to_current`` for a callable handed to a thread pool,
rather than a bare ``threading.Thread`` / ``pool.submit``.
"""

from __future__ import annotations

import functools
import logging
import threading
from typing import Callable, Dict, Optional, Tuple

# worker thread id -> (parent thread id, candidate tag, prefix messages?)
_WORKERS: Dict[int, Tuple[int, str, bool]] = {}
_LOCK = threading.Lock()
_FILTER_INSTALLED = False


def register_worker(parent_thread_id: int, tag: str, prefix: bool = True) -> None:
    """Register the CURRENT thread as a fan-out worker of ``parent_thread_id``.

    ``prefix`` controls ONLY the cosmetic ``[tag]`` log prefix. Thread routing
    (``effective_thread``, which is how the UI verbose panel keeps a worker's
    narration visible by attributing it to the parent chat thread) is ALWAYS
    active — so a single candidate's logs stay visible even with ``prefix=False``;
    only the redundant tag is dropped.
    """
    _install_prefix_filter()
    with _LOCK:
        _WORKERS[threading.get_ident()] = (parent_thread_id, tag, prefix)


def unregister_worker() -> None:
    with _LOCK:
        _WORKERS.pop(threading.get_ident(), None)


# Longest worker chain resolved by ``effective_thread`` — a bound, not a
# design limit: chat -> fan-out branch -> best-of-N candidate -> helper is
# four hops, and a stale (thread-id reuse) entry must never spin forever.
_MAX_WORKER_DEPTH = 8


def effective_thread(thread_id: int) -> int:
    """Root (chat) thread id for registered workers; identity otherwise.

    Follows the chain, so a worker spawned BY a worker (a best-of-N
    candidate inside a fan-out branch, or a heartbeat inside a candidate)
    still resolves to the chat thread that owns the session route.
    """
    for _ in range(_MAX_WORKER_DEPTH):
        entry = _WORKERS.get(thread_id)
        if not entry or entry[0] == thread_id:
            break
        thread_id = entry[0]
    return thread_id


def attributed_to_current(fn: Callable, tag: str = "",
                          prefix: bool = False) -> Callable:
    """Wrap ``fn`` so that, wherever it later runs, that thread is attributed
    to the CALLING thread's session for the duration of the call.

    Call this on the owning (chat / worker) thread and hand the result to a
    ``ThreadPoolExecutor`` or ``threading.Thread``. Registration is scoped to
    the call — a pool thread that outlives the call is unregistered again,
    so no stale mapping survives thread-id reuse. Exceptions propagate
    unchanged (a pool future must still see them).
    """
    parent = effective_thread(threading.get_ident())

    @functools.wraps(fn)
    def _attributed(*args, **kwargs):
        register_worker(parent, tag, prefix=prefix)
        try:
            return fn(*args, **kwargs)
        finally:
            unregister_worker()

    return _attributed


def start_attributed_thread(target: Callable, *, args: tuple = (),
                            kwargs: Optional[dict] = None,
                            name: Optional[str] = None,
                            daemon: bool = True, tag: str = "",
                            prefix: bool = False) -> threading.Thread:
    """Start a background thread whose prints and log records belong to the
    calling thread's session (drop-in for ``threading.Thread(...).start()``).

    A session stop reaches the helper through the routed stream exactly as
    it reaches the chat thread (``AgentStoppedError`` on its next write); the
    helper then simply ends — a stopped session's progress ticker has nothing
    left to report, and a traceback for it would only be console noise.
    """
    from scilink.ui.output_capture import AgentStoppedError

    attributed = attributed_to_current(target, tag=tag, prefix=prefix)

    def _run():
        try:
            attributed(*args, **(kwargs or {}))
        except AgentStoppedError:
            pass

    t = threading.Thread(target=_run, name=name, daemon=daemon)
    t.start()
    return t


def is_concise_fanout_worker(thread_id: int) -> bool:
    """True if the thread is a fan-out worker running CONCURRENTLY with
    siblings (registered with ``prefix=True`` — the ``[cand_NN]`` tag).

    ``prefix`` is set only when more than one candidate runs at once, so it
    doubles as the "concurrent fan-out" signal. Anything that displays like a
    single run stays fully verbose: a lone escalation probe (attempt 0,
    ``prefix=False``) and a plain single analysis (``n==1``, never registered).
    The UI uses this to trim only the concurrent candidates' per-attempt
    narration to milestones.
    """
    entry = _WORKERS.get(thread_id)
    return bool(entry and entry[2])


# Milestone substrings that survive UI trimming for a concurrent fan-out
# worker (matched after the ``[cand_NN]`` prefix the record factory adds).
# Everything else a candidate worker logs at INFO is per-attempt detail
# (plan-deviation justifications, sub-scores, verification issues, refinement
# narration) and is dropped from the panel. Warnings/errors always pass.
_FANOUT_PANEL_MILESTONES = (
    "Executing Python script",
    "Verification ",
    "Analysis approved",
    "Quality score =",
)


def keep_in_fanout_panel(thread_id: int, message: str, levelno: int) -> bool:
    """UI verbose-panel filter for fan-out worker records.

    Returns True if the record should appear in the Streamlit panel. A
    concurrent fan-out worker is trimmed to milestone lines (plus any hard
    ERROR — a candidate crash); all other records pass unchanged. Consulted
    ONLY by the UI handler — the CLI console and log files keep full detail.

    The threshold is ERROR, not WARNING, on purpose: a candidate's per-attempt
    churn (plan-conformance issues, retry "failed"/"missing outputs") is logged
    at WARNING and is exactly the noise to trim. Candidate-level OUTCOMES are
    surfaced on the parent (chat) thread (``Candidate N finished`` /
    ``Candidate N raised``), which is never a concise worker, so trimming
    worker WARNINGs does not hide whether a candidate succeeded or died.
    """
    if levelno >= logging.ERROR:
        return True
    if not is_concise_fanout_worker(thread_id):
        return True
    return any(m in message for m in _FANOUT_PANEL_MILESTONES)


def _install_prefix_filter() -> None:
    """Install a log-record factory that prefixes worker records (idempotent).

    A record factory (rather than a logger/handler filter) is the one hook
    that sees every record regardless of which named logger emitted it and
    which handlers are attached later.
    """
    global _FILTER_INSTALLED
    if _FILTER_INSTALLED:
        return
    with _LOCK:
        if _FILTER_INSTALLED:
            return
        previous_factory = logging.getLogRecordFactory()

        def factory(*args, **kwargs):
            record = previous_factory(*args, **kwargs)
            entry = _WORKERS.get(record.thread)
            if entry and entry[2]:          # entry[2] = prefix?  (routing is separate)
                record.msg = f"[{entry[1]}] {record.msg}"
            return record

        logging.setLogRecordFactory(factory)
        _FILTER_INSTALLED = True
