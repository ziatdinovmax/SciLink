"""How many workers a pool may run, from one knob.

Three pools each read their own variable (``SCILINK_FANOUT_MAX_WORKERS``,
``SCILINK_HS_SERIES_WORKERS``, ``SCILINK_CURVE_FIT_WORKERS``) with unrelated
defaults, so a container sized for two vCPUs had no single place to say so.
``SCILINK_MAX_WORKERS`` is that place: a ceiling every pool honours
(``auto`` means the CPU count the process sees). Precedence for a pool:
an explicit value > its own variable > the ceiling > its default; the
ceiling then caps whatever was chosen. Values below 1 are clamped to 1.
"""
from __future__ import annotations

import os
from typing import Optional


def max_workers() -> Optional[int]:
    """The process-wide ceiling, or ``None`` when ``SCILINK_MAX_WORKERS`` is unset."""
    raw = (os.environ.get("SCILINK_MAX_WORKERS") or "").strip().lower()
    if not raw:
        return None
    if raw == "auto":
        return max(1, os.cpu_count() or 1)
    try:
        return max(1, int(float(raw)))
    except ValueError:
        return None


def resolve_workers(explicit: Optional[int], env_var: str, default: int) -> int:
    """A pool's worker count (see the module docstring)."""
    ceiling = max_workers()
    value: Optional[int] = None
    if explicit is not None:
        value = int(explicit)
    else:
        raw = os.environ.get(env_var)
        if raw:
            try:
                value = int(float(raw))
            except ValueError:
                value = None
        if value is None:
            value = ceiling if ceiling is not None else int(default)
    value = max(int(value), 1)
    if ceiling is not None:
        value = min(value, ceiling)
    return value
