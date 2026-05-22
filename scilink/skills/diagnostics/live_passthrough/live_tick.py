"""Diagnostics live_tick — counts peaks in 2-column data, verdicts by count.

Pure plumbing-test, not a scientific skill. Lets the live-monitoring
infrastructure be exercised end-to-end without depending on any
domain-specific skill.

See the sibling ``live_passthrough.md`` for the user-facing
description. The frontmatter wires this function as the
``live_tick.tick_fn`` for the ``diagnostics/live_passthrough`` skill.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from scilink.agents.exp_agents.live_data_sources import LatestData
from scilink.agents.exp_agents.live_types import LiveTickResult

_logger = logging.getLogger(__name__)


def _parse_two_column(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse CSV / whitespace-delimited two-column text → (x, y) arrays.

    Tolerates header lines and `#` comments. Mirrors the parser the
    structure_matching/xrd live_tick uses, kept inline so this skill
    has no cross-skill imports.
    """
    if not text:
        return np.empty(0), np.empty(0)
    xs: list[float] = []
    ys: list[float] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in (line.split(",") if "," in line else line.split())]
        if len(parts) < 2:
            continue
        try:
            xs.append(float(parts[0]))
            ys.append(float(parts[1]))
        except ValueError:
            continue
    return np.asarray(xs), np.asarray(ys)


def _count_peaks(y: np.ndarray) -> list[int]:
    """Return indices of detected peaks in ``y``. Falls back to a trivial
    'local-max above threshold' check when scipy isn't importable."""
    if y.size < 5:
        return []
    try:
        from scipy.signal import find_peaks
    except ImportError:
        # Trivial fallback: any sample greater than both neighbours and
        # above the 80th-percentile threshold counts as a peak.
        thresh = float(np.quantile(y, 0.8))
        idxs: list[int] = []
        for i in range(1, len(y) - 1):
            if y[i] > y[i - 1] and y[i] > y[i + 1] and y[i] >= thresh:
                idxs.append(i)
        return idxs
    # Prominence as 5% of the dynamic range works well for both clean
    # synthetic patterns and noisy real data.
    rng = float(y.max() - y.min())
    prominence = max(0.05 * rng, 1.0)
    peaks, _ = find_peaks(y, prominence=prominence)
    return list(peaks)


def _verdict_for_count(n: int) -> str:
    if n == 0:
        return "unknown"
    if n == 1:
        return "reject"
    if n < 4:
        return "marginal"
    return "accept"


def passthrough_tick(latest_data: LatestData, session_state: dict,
                      skill_state: dict) -> LiveTickResult:
    """The tick function. See module docstring."""
    text = latest_data.text or ""
    x, y = _parse_two_column(text)
    if x.size < 5:
        return LiveTickResult(
            timestamp=time.time(),
            primary_metric=0.0,
            metric_name="peak_count",
            verdict="unknown",
            detected_features=[],
            notes=f"only {x.size} parseable data points so far",
        )
    peak_idxs = _count_peaks(y)
    n_peaks = len(peak_idxs)
    features = [
        {"position": float(x[i]), "intensity": float(y[i])}
        for i in peak_idxs
    ]
    return LiveTickResult(
        timestamp=time.time(),
        primary_metric=float(n_peaks),
        metric_name="peak_count",
        verdict=_verdict_for_count(n_peaks),
        detected_features=features,
        notes=f"{n_peaks} peaks detected across {x.size} points",
        raw={
            "n_points": int(x.size),
            "y_max": float(y.max()),
            "y_min": float(y.min()),
            "source_kind": latest_data.source_kind,
        },
    )


__all__ = ["passthrough_tick"]
