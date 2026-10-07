"""How many LLM calls one process has in flight, per model — and when the
provider is failing, how new work is held.

Every worker of a fan-out or a swarm, and every best-of-N candidate inside
one, calls the provider from its own thread. Nothing bounded how many did so
at once, so a swarm of N workers met one rate limit together and retried
together. ``llm_slot(model)`` is held for the duration of one provider call
(never across a retry's backoff sleep): at most ``llm_max_inflight()`` calls
to one model run at a time in this process, and the rest wait their turn.

The default sits above what today's largest in-process parallelism reaches
(four fan-out branches, each with a best-of-N of three), so ordinary runs
never wait; it bounds what a swarm can put on one provider at once.

The circuit breaker (stage 4) is per model: the retry policy reports every
provider call's outcome (``note_provider_failure`` for a retryable failure,
``note_provider_ok`` for a success), and a model is TRIPPED when its last
``BREAKER_WINDOW_S`` hold at least ``BREAKER_FAILURES`` failures AND the
failures are at least half of its calls in that window. A tripped model holds
ADMISSION of new work (``fanout._admit_branch``) for ``BREAKER_HOLD_S``;
running work keeps its own retries. A brown-out is the case: the running
work keeps succeeding now and then, so a success must not close the breaker
on its own — the hold runs its course, and the ratio decides whether the
next window trips again. One model's throttling never holds another's items.
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
BREAKER_FAILURES = 6
BREAKER_WINDOW_S = 60.0
BREAKER_HOLD_S = 30.0
_UNKNOWN = "unknown"

_logger = logging.getLogger(__name__)
_slots: Dict[Tuple[str, int], threading.BoundedSemaphore] = {}
_slots_lock = threading.Lock()
_breaker_lock = threading.Lock()
#: Per model: the window's outcomes as ``(time, ok)``, and when its hold ends.
_outcomes: Dict[str, Deque[Tuple[float, bool]]] = {}
_tripped_until: Dict[str, float] = {}


def _key(model: Optional[str]) -> str:
    """The breaker's key for a model: the wrapper's PREFIXED name, so the
    name an admission asks with (the meta's ``claude-opus-4-6``) and the
    name the retry policy records under (``anthropic/claude-opus-4-6``) are
    one key."""
    if not model:
        return _UNKNOWN
    try:
        from .litellm_wrapper import _normalize_model_name
        return str(_normalize_model_name(str(model)) or model)
    except Exception:  # noqa: BLE001 - the wrapper is not importable here: the bare name
        return str(model)


def _note(model: Optional[str], ok: bool) -> None:
    key = _key(model)
    now = time.monotonic()
    with _breaker_lock:
        window = _outcomes.setdefault(key, deque())
        window.append((now, ok))
        while window and now - window[0][0] > BREAKER_WINDOW_S:
            window.popleft()
        if ok:
            return
        failures = sum(1 for _, good in window if not good)
        if (failures >= BREAKER_FAILURES and 2 * failures >= len(window)
                and now >= _tripped_until.get(key, 0.0)):
            _tripped_until[key] = now + BREAKER_HOLD_S
            _logger.warning(
                f"Provider circuit breaker tripped for {key}: {failures} retryable failures in "
                f"{len(window)} calls within {BREAKER_WINDOW_S:.0f} s; new work on it is held for "
                f"{BREAKER_HOLD_S:.0f} s.")


def note_provider_failure(model: Optional[str] = None) -> None:
    """One retryable failure from the provider (rate limit, overload, a
    server fault, a timeout), as the retry policy sees it."""
    _note(model, False)


def note_provider_ok(model: Optional[str] = None) -> None:
    """One call that succeeded. It counts toward the window's ratio; it does
    not end a hold early."""
    _note(model, True)


def provider_tripped(model: Optional[str] = None) -> Optional[float]:
    """Seconds the model's hold has left, or ``None`` when it is closed. With
    no model, the longest hold of any model (an admission that does not know
    which model its item will call)."""
    now = time.monotonic()
    with _breaker_lock:
        if model is not None:
            left = _tripped_until.get(_key(model), 0.0) - now
            return left if left > 0 else None
        left = max((until - now for until in _tripped_until.values()), default=0.0)
        return left if left > 0 else None


def reset_breaker() -> None:
    with _breaker_lock:
        _outcomes.clear()
        _tripped_until.clear()


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
