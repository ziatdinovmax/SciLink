"""Transparent, generalizable disk memoization for expensive, deterministic
analysis tools.

Motivation: the analysis agents run each generated script in a fresh subprocess
and, on every verification/correction attempt, re-execute the WHOLE script —
re-paying the cost of stable, expensive steps (e.g. a cold DCNN ensemble
inference, ~300 s) even when only a cheap downstream step changed. On a fixed
``executor_timeout`` that can make corrections unwinnable: the agent trims a
cheap step but still times out because the dominant cost is recomputed every
time, with no per-step signal telling it where the time went.

This decorator content-addresses a tool's result by its inputs and stores it on
disk, so the next attempt with IDENTICAL inputs loads instantly instead of
recomputing. It is tool-agnostic — decorate any pure, expensive function
(detection, GPA, Fourier mapping, decomposition, ...). Correct by construction:
the key includes every argument (ndarray bytes + shape + dtype, plus scalars /
strings / nested lists+dicts) and the function's own source, so changing a
parameter, the input, or the tool's code auto-invalidates and recomputes — only
genuinely-unchanged steps are reused.

Scope/safety: caching is active ONLY when the ``SCILINK_TOOL_CACHE_DIR``
environment variable is set (the script executor points it at the analysis
working directory, which is shared across that run's correction attempts). When
it is unset — direct library use, tests, notebooks — the decorator is a
transparent no-op, so those call paths are byte-for-byte unchanged. Any cache
read/write error falls back to computing normally; caching never changes a
result or breaks a tool. Set ``SCILINK_TOOL_CACHE_DISABLE=1`` to force-disable.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import logging
import os
import pickle

import numpy as np

logger = logging.getLogger(__name__)

_KEY_VERSION = "1"


def _update_hash(h, obj):
    """Fold ``obj`` into hash ``h``; raise TypeError if not safely hashable."""
    if isinstance(obj, np.ndarray):
        h.update(b"\x00ndarray")
        h.update(str(obj.shape).encode())
        h.update(str(obj.dtype).encode())
        h.update(np.ascontiguousarray(obj).tobytes())
    elif isinstance(obj, (str, bytes, bool, int, float, type(None))):
        h.update(b"\x01scalar")
        h.update(repr(obj).encode())
    elif isinstance(obj, (list, tuple)):
        h.update(b"\x02seq")
        for item in obj:
            _update_hash(h, item)
    elif isinstance(obj, dict):
        h.update(b"\x03dict")
        for k in sorted(obj, key=repr):
            h.update(repr(k).encode())
            _update_hash(h, obj[k])
    else:
        # Opaque / unhashable argument (model handle, open file, ...): refuse to
        # cache rather than risk a wrong key.
        raise TypeError(f"uncacheable argument of type {type(obj)!r}")


def disk_memoized(func):
    """Content-addressed disk memoization, active only under SCILINK_TOOL_CACHE_DIR.

    Decorate an expensive, deterministic function. The wrapper is a no-op unless
    the cache directory env var is set, so non-agent call paths are unaffected.
    """
    try:
        # Computed once at decoration time so a code change to the tool body
        # invalidates its on-disk cache.
        _src_hash = hashlib.blake2b(
            inspect.getsource(func).encode(), digest_size=8
        ).hexdigest()
    except Exception:
        _src_hash = "nosrc"

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_dir = os.environ.get("SCILINK_TOOL_CACHE_DIR")
        if not cache_dir or os.environ.get("SCILINK_TOOL_CACHE_DISABLE") == "1":
            return func(*args, **kwargs)
        try:
            h = hashlib.blake2b(digest_size=16)
            h.update(
                f"{func.__module__}.{func.__qualname__}:{_src_hash}:v{_KEY_VERSION}".encode()
            )
            for a in args:
                _update_hash(h, a)
            for k in sorted(kwargs):
                h.update(("\x04kw:" + k).encode())
                _update_hash(h, kwargs[k])
            key = h.hexdigest()
        except Exception as exc:
            logger.debug("[tool_cache] %s not cacheable (%s); computing", func.__name__, exc)
            return func(*args, **kwargs)

        path = os.path.join(cache_dir, f"{func.__name__}_{key}.pkl")
        try:
            if os.path.exists(path):
                with open(path, "rb") as fh:
                    result = pickle.load(fh)
                logger.info("[tool_cache] HIT %s -> %s", func.__name__, os.path.basename(path))
                return result
        except Exception as exc:
            logger.debug("[tool_cache] load failed for %s (%s); recomputing", path, exc)

        result = func(*args, **kwargs)
        try:
            os.makedirs(cache_dir, exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "wb") as fh:
                pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
            logger.info("[tool_cache] STORE %s -> %s", func.__name__, os.path.basename(path))
        except Exception as exc:
            logger.debug("[tool_cache] store failed for %s (%s)", path, exc)
        return result

    wrapper.__wrapped__ = func
    return wrapper
