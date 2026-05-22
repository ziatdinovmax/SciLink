"""Live-mode tick function for XRD phase identification.

Plugs the structure_matching/xrd skill into the live-monitoring
infrastructure. Each tick:

  1. Parse the latest data text (CSV: 2-theta, intensity).
  2. Extract peaks via :func:`extract_peaks`.
  3. For each pre-computed candidate (cached in ``session_state``
     on first tick), run :func:`score_xrd_match_fast` and keep the
     best by Pearson correlation.
  4. Return a :class:`LiveTickResult` with the best correlation as
     ``primary_metric``, the candidate's verdict, and the experimental
     peak list as ``detected_features``.

Candidates are loaded once per session: on the first tick the function
reads ``session_state['live_candidates']`` (set by the orchestrator's
session-init flow); if absent it falls back to querying
:func:`search_structures` with whatever chemistry hint sits in the
skill state. Subsequent ticks reuse the cached simulations — only the
fast scorer runs on the hot path.

Wavelength defaults to CuKa; override via the skill state's metadata
(``skill_state['live_tick'].get('wavelength', 'CuKa')``) or by setting
``session_state['wavelength']`` at session start.
"""

from __future__ import annotations

import io
import logging
import time
from typing import Any

import numpy as np

from ....agents.exp_agents.live_data_sources import LatestData
from ....agents.exp_agents.live_types import LiveTickResult
from .extract_peaks import extract_peaks
from .score_match_fast import score_xrd_match_fast
from .search_structures import search_structures
from .simulate_xrd import simulate_xrd_pattern

_logger = logging.getLogger(__name__)


# Verdict thresholds on Pearson correlation (mirror score_match_fast's
# defaults but exposed here for documentation).
_ACCEPT = 0.85
_MARGINAL = 0.60


def xrd_live_tick(
    latest_data: LatestData,
    session_state: dict,
    skill_state: dict,
) -> LiveTickResult:
    """Tick function for the XRD structure_matching skill."""
    text = latest_data.text or ""
    x, y = _parse_two_column_csv(text)
    if x.size < 16:
        # Not enough points yet — instrument is just starting
        return LiveTickResult(
            timestamp=time.time(),
            primary_metric=0.0,
            metric_name="correlation",
            verdict="unknown",
            detected_features=[],
            notes=f"only {x.size} points so far",
        )

    # Lazy-load candidates on the first tick that produces enough data
    candidates = session_state.get("live_candidates")
    if candidates is None:
        candidates = _load_candidates(session_state, skill_state)
        session_state["live_candidates"] = candidates
        if not candidates:
            return LiveTickResult(
                timestamp=time.time(),
                primary_metric=0.0,
                metric_name="correlation",
                verdict="unknown",
                detected_features=[],
                notes="No candidates available — check MP_API_KEY or SCILINK_LOCAL_CIF_DIR",
            )

    # Extract experimental peaks for the detected_features field
    try:
        peaks = extract_peaks(
            x.tolist(), y.tolist(),
            prominence_frac=0.05, max_peaks=12, refine=False,
        )
    except Exception as e:
        _logger.debug("extract_peaks failed: %s", e)
        peaks = {"positions": [], "intensities": []}

    wavelength = session_state.get("wavelength") or _wavelength_from_skill(skill_state) or "CuKa"
    two_theta_range = (float(x.min()), float(x.max()))

    # Score every candidate, keep the best
    best = None
    for cand in candidates:
        sim = cand.get("_sim")
        if sim is None:
            try:
                sim = simulate_xrd_pattern(
                    cand["structure_path"],
                    wavelength=wavelength,
                    two_theta_range=two_theta_range,
                )
            except Exception as e:
                _logger.debug("simulate failed for %s: %s", cand.get("id"), e)
                continue
            cand["_sim"] = sim
        try:
            score = score_xrd_match_fast(
                exp_two_theta=x.tolist(),
                exp_intensity=y.tolist(),
                sim_two_theta=sim["two_theta"],
                sim_intensity=sim["intensities"],
            )
        except Exception as e:
            _logger.debug("score failed for %s: %s", cand.get("id"), e)
            continue
        if best is None or score["correlation"] > best["score"]["correlation"]:
            best = {"candidate": cand, "score": score}

    if best is None:
        return LiveTickResult(
            timestamp=time.time(),
            primary_metric=0.0,
            metric_name="correlation",
            verdict="unknown",
            detected_features=[
                {"position": p, "intensity": i}
                for p, i in zip(peaks["positions"], peaks["intensities"])
            ],
            notes="No candidate could be scored",
        )

    score = best["score"]
    cand = best["candidate"]
    corr = float(score["correlation"])
    verdict = "accept" if corr >= _ACCEPT else ("marginal" if corr >= _MARGINAL else "reject")

    return LiveTickResult(
        timestamp=time.time(),
        primary_metric=corr,
        metric_name="correlation",
        verdict=verdict,
        detected_features=[
            {"position": p, "intensity": i}
            for p, i in zip(peaks["positions"], peaks["intensities"])
        ],
        notes=(
            f"best match {cand.get('source')}:{cand.get('id')} "
            f"({cand.get('formula')}, {cand.get('space_group')}) — "
            f"shift {score['fitted_shift']:+.3f}°, scale {score['fitted_scale']:.4f}"
        ),
        raw={
            "best_match": {
                "id": cand.get("id"),
                "source": cand.get("source"),
                "formula": cand.get("formula"),
                "space_group": cand.get("space_group"),
            },
            "fitted_shift": score["fitted_shift"],
            "fitted_scale": score["fitted_scale"],
            "n_data_points": int(x.size),
            "n_experimental_peaks": len(peaks["positions"]),
            "wavelength": wavelength,
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_two_column_csv(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a CSV / whitespace-delimited two-column text into x, y arrays.

    Tolerates header lines (skipped if they can't parse as float) and
    extra columns (keeps only the first two).
    """
    if not text:
        return np.empty(0), np.empty(0)
    xs: list[float] = []
    ys: list[float] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Split on comma or whitespace
        if "," in line:
            parts = [p.strip() for p in line.split(",")]
        else:
            parts = line.split()
        if len(parts) < 2:
            continue
        try:
            xs.append(float(parts[0]))
            ys.append(float(parts[1]))
        except ValueError:
            # Header / non-numeric — skip
            continue
    return np.asarray(xs), np.asarray(ys)


def _wavelength_from_skill(skill_state: dict) -> str | None:
    block = (skill_state or {}).get("meta", {}).get("live_tick") or {}
    return block.get("wavelength")


def _load_candidates(session_state: dict, skill_state: dict) -> list[dict]:
    """Query search_structures for candidates using whatever chemistry hint
    is available; cache the materialized CIFs in session_state."""
    chemistry = (
        session_state.get("chemistry_hint")
        or _chemistry_from_skill(skill_state)
        or [["Si"], ["C"], ["Ge"], ["Ti", "O"]]  # post-fit broad default
    )
    top_n = int(session_state.get("top_n", 5))
    output_dir = session_state.get("candidates_dir") or "./live_candidates"
    try:
        hits = search_structures(
            query={"chemistry": chemistry, "top_n": top_n},
            output_dir=output_dir,
        )
    except Exception as e:
        _logger.warning("search_structures failed at session start: %s", e)
        return []
    return list(hits.get("candidates", []))


def _chemistry_from_skill(skill_state: dict) -> list[str] | list[list[str]] | None:
    block = (skill_state or {}).get("meta", {}).get("live_tick") or {}
    return block.get("chemistry")


__all__ = ["xrd_live_tick"]
