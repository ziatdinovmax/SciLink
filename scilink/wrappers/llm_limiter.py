"""How many LLM calls one process has in flight, per model.

Every worker of a fan-out or a swarm, and every best-of-N candidate inside
one, calls the provider from its own thread. Nothing bounded how many did so
at once, so a swarm of N workers met one rate limit together and retried
together. ``llm_slot(model)`` is held for the duration of one provider call
(never across a retry's backoff sleep): at most ``llm_max_inflight()`` calls
to one model run at a time in this process, and the rest wait their turn.

The default sits above what today's largest in-process parallelism reaches
(four fan-out branches, each with a best-of-N of three), so ordinary runs
never wait; it bounds what a swarm can put on one provider at once.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Deque, Dict, Iterator, Optional, Tuple

LLM_MAX_INFLIGHT = 16
_WAIT_LOG_S = 10.0
_WAIT_SLICE_S = 1.0
#: The circuit breaker: this many retryable provider failures (rate limits,
#: overloads, timeouts) across every worker of the process within the window
#: trip it, and a tripped breaker pauses ADMISSION of new work for the hold
#: — the work already running keeps its own retries. A success closes it.
BREAKER_FAILURES = 6
BREAKER_WINDOW_S = 60.0
BREAKER_HOLD_S = 30.0

_logger = logging.getLogger(__name__)
_slots: Dict[Tuple[str, int], threading.BoundedSemaphore] = {}
_slots_lock = threading.Lock()
_breaker_lock = threading.Lock()
_failures: Deque[float] = deque()
_tripped_until = 0.0


def note_provider_failure(model: Optional[str] = None) -> None:
    """One retryable failure from the provider (called where the retry
    policy decides to retry). Trips the breaker at ``BREAKER_FAILURES``
    within ``BREAKER_WINDOW_S``."""
    global _tripped_until
    now = time.monotonic()
    with _breaker_lock:
        _failures.append(now)
        while _failures and now - _failures[0] > BREAKER_WINDOW_S:
            _failures.popleft()
        if len(_failures) >= BREAKER_FAILURES and now >= _tripped_until:
            _tripped_until = now + BREAKER_HOLD_S
            _logger.warning(
                f"Provider circuit breaker tripped: {len(_failures)} retryable failures"
                f"{f' ({model})' if model else ''} within {BREAKER_WINDOW_S:.0f} s; "
                f"new work is held for {BREAKER_HOLD_S:.0f} s.")


def note_provider_ok() -> None:
    """A call that succeeded: the provider answers again, the breaker closes."""
    global _tripped_until
    with _breaker_lock:
        _failures.clear()
        _tripped_until = 0.0


def provider_tripped() -> Optional[float]:
    """Seconds the breaker stays open, or ``None`` when it is closed."""
    left = _tripped_until - time.monotonic()
    return left if left > 0 else None


def reset_breaker() -> None:
    note_provider_ok()


def llm_max_inflight() -> Optional[int]:
    """Calls to one model this process may have in flight at once
    (``SCILINK_LLM_MAX_INFLIGHT`` overrides; ``0`` removes the cap)."""
    raw = (os.environ.get("SCILINK_LLM_MAX_INFLIGHT") or "").strip()
    if not raw:
        return LLM_MAX_INFLIGHT
    try:
        value = int(raw)
    except ValueError:
        return LLM_MAX_INFLIGHT
    return value if value > 0 else None


def _semaphore(model: str, cap: int) -> threading.BoundedSemaphore:
    key = (str(model), cap)
    with _slots_lock:
        sem = _slots.get(key)
        if sem is None:
            sem = _slots[key] = threading.BoundedSemaphore(cap)
        return sem


@contextmanager
def llm_slot(model: Optional[str]) -> Iterator[None]:
    """Hold one of the model's in-flight slots for the block.

    A caller that has to wait says so in the log, and again every few
    seconds; while it waits it checks for a cancel of its worker (a budget or
    memory cancel, the turn's Stop) once a second, so a cancelled worker
    stops here instead of taking the slot and making one more call.
    """
    from ..utils.log_context import raise_if_cancelled
    cap = llm_max_inflight()
    if cap is None:
        yield
        return
    sem = _semaphore(model or "unknown", cap)
    if not sem.acquire(blocking=False):
        _logger.warning(f"Waiting for an LLM slot: {cap} calls to {model} are already in flight.")
        waited = 0.0
        while True:
            raise_if_cancelled()
            if sem.acquire(timeout=_WAIT_SLICE_S):
                break
            waited += _WAIT_SLICE_S
            if waited % _WAIT_LOG_S < _WAIT_SLICE_S:
                _logger.info(f"Still waiting for an LLM slot for {model}.")
    try:
        yield
    finally:
        sem.release()
