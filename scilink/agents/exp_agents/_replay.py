"""The policies the three analysis agents share, in one place (#712).

The curve, image and hyperspectral agents each judge a replay of a locked
recipe, each say what "verified" means, and each choose among a series'
regime recipes on a reuse — three copies that drifted. This module holds the
POLICIES, by composition: each agent keeps its pipeline, prompts and
controllers, and routes these decisions through here. It is not a base
class (CLAUDE.md), and it is the shape ``CodegenQCEngine`` set.

- **A verdict record** (``verdict_record``) is one dict shape every agent
  stamps where it decides — a unit at fit time, a replay at its gate, a
  single run or a cube when its result is final — and ``analysis_verdict``
  and the swarm board only READ it. It says whether the result is
  verified, why, and WHO decided (``decided_by``): the agent's own QC gate,
  a replay gate, the recipe a follower replayed, or nothing.
- **A replay gate** turns a replayed result plus the anchor's reference
  into ``{"verdict": good | poor | failed, "score", "threshold", "reasons"}``.
  Three implementations, today's behaviours as they were: a score against
  the accept gate (the curve's R², the image's vision score), the image's
  feature-health gate for a strict replay, and the hyperspectral map gate
  against the anchor's statistics.
- **``select_recipe``** chooses among a series' regime recipes on a reuse:
  today's rule — in lock order, the first the gate calls good, else the
  first that executed — as the default strategy.

Nothing here calls a model.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------- the record
#: Who decided a verdict record.
DECIDERS = ("qc_gate", "replay_gate", "recipe", "excluded", "none")


def verdict_record(*, verified: bool, reason: str, decided_by: str, regime: Optional[str] = None,
                   recipe_of: Optional[str] = None, score: Optional[float] = None,
                   threshold: Optional[float] = None, interpretation_checked: bool = False,
                   **extra: Any) -> Dict[str, Any]:
    """One verdict, one shape. ``own_gate`` is kept as a derived key for the
    readers of the stage-2 stamp (a QC gate or a replay gate decided)."""
    if decided_by not in DECIDERS:
        raise ValueError(f"decided_by must be one of {DECIDERS}, not {decided_by!r}")
    rec: Dict[str, Any] = {"verified": bool(verified), "reason": str(reason), "decided_by": decided_by,
                           "interpretation_checked": bool(interpretation_checked)}
    if regime is not None:
        rec["regime"] = regime
    if recipe_of is not None:
        rec["recipe_of"] = recipe_of
    if score is not None:
        rec["score"] = score
    if threshold is not None:
        rec["threshold"] = threshold
    if decided_by in ("qc_gate", "replay_gate"):
        rec["own_gate"] = True
    rec.update(extra)
    return rec


def is_verdict_record(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("verified"), bool) and "reason" in value


# ------------------------------------------------------------ replay gates
def replay_verdict(verdict: str, *, score: Optional[float] = None, threshold: Optional[float] = None,
                   reasons: Optional[Sequence[str]] = None, gate: str = "") -> Dict[str, Any]:
    if verdict not in ("good", "poor", "failed"):
        raise ValueError(f"a replay verdict is good, poor or failed, not {verdict!r}")
    return {"verdict": verdict, "score": score, "threshold": threshold,
            "reasons": [r for r in (reasons or []) if r], "gate": gate}


class ScoreReplayGate:
    """A replayed result judged by one score against the agent's accept gate:
    the curve agent's R² (``fit_quality.r_squared``), the image agent's
    vision-review score. ``accept`` is the agent's ``is_accept`` (the soft
    band included, as it always was)."""

    def __init__(self, accept: Callable[[float], bool], threshold: float, metric: str):
        self.accept, self.threshold, self.metric = accept, threshold, metric

    def judge(self, score: Any) -> Dict[str, Any]:
        value = float(score or 0.0)
        ok = bool(self.accept(value))
        return replay_verdict("good" if ok else "poor", score=value, threshold=self.threshold,
                              reasons=[] if ok else [f"{self.metric} {value:.4f} below the acceptance "
                                                     f"threshold {self.threshold:.3f}"],
                              gate=self.metric)


def feature_health(features: Any, reference: Any) -> Tuple[bool, str]:
    """Evidence-only acceptance of an image analysed by a LOCKED-SCRIPT replay.

    A strict replay (a live frame) has no model to look at the overlay, and an
    image analysis has no R²: what it has is the numbers the approved script
    reports, and what that script reported on its reference. The gate judges
    METHOD HEALTH from them and nothing else:

    - every numeric quantity the reference run reported is reported again, and
      is finite (a script that silently stops measuring something is broken);
    - the analysis still finds SOMETHING: if every quantity that was non-zero on
      the reference is zero now (no particles, no mask, no lattice), the method
      collapsed on this image or there is nothing in it, and either way the
      frame is not one to track quietly.

    It does not ask whether the values are plausible: in a stream they are
    expected to move, and that is the live loop's range gate (which flags a
    jump and adopts a value that persists) and its audits. What no gate here
    can see is a segmentation that runs, reports finite numbers and is wrong;
    the independent audit is the check for that. Returns ``(ok, reason)``."""
    ref = {k: v for k, v in (reference or {}).items()
           if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}
    got = features if isinstance(features, dict) else {}
    if not ref:
        numeric = [v for v in got.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if not numeric or not all(math.isfinite(v) for v in numeric):
            return False, "the replay reported no finite quantity"
        return True, ""
    missing = [k for k in ref if not isinstance(got.get(k), (int, float)) or isinstance(got.get(k), bool)]
    if missing:
        return False, ("the locked script no longer reports "
                       + ", ".join(sorted(missing)[:4]) + " (it did on the reference)")
    broken = [k for k in ref if not math.isfinite(float(got[k]))]
    if broken:
        return False, "not finite: " + ", ".join(sorted(broken)[:4])
    alive = [k for k, v in ref.items() if v != 0]
    if alive and all(float(got[k]) == 0.0 for k in alive):
        return False, ("every quantity that was non-zero on the reference is zero here "
                       "(nothing was found: the method collapsed on this image, or it is empty)")
    return True, ""


class FeatureHealthGate:
    """The image agent's strict-replay gate (``feature_health``) as a gate."""

    def judge(self, features: Any, reference: Any) -> Dict[str, Any]:
        ok, reason = feature_health(features, reference)
        return replay_verdict("good" if ok else "poor", reasons=[reason], gate="feature_health")


def map_health(result_map, fit_mask, reference: Optional[dict], required: bool) -> Tuple[bool, str]:
    """Deterministic acceptance of a map produced by a LOCKED-SCRIPT replay.

    A verbatim replay re-runs a method the reviewer already approved on the
    anchor; re-judging every map with the LLM only adds cost and judge
    variance (observed live: the map approved on the anchor rejected on
    replays, punching holes in the series schema). So replays are gated on
    evidence alone: valid coverage (within the fit mask when scoped), a
    non-collapsed value distribution, and — for a required output with the
    anchor's stats as ``reference`` — a median inside the locked method's
    plausible range (the anchor's [min, max] widened by one span on each
    side, at least 0.5 % of the magnitude so a near-constant anchor map does
    not reject trivial drift). Coverage is judged against the anchor's own
    coverage (``reference["coverage"]``) when known. A map outside that range is the
    method breaking down on this dataset (e.g. the peak left the fit
    window), which the series driver answers with a fresh-code refit.
    """
    m = np.asarray(result_map, dtype=float)
    if fit_mask is not None:
        try:
            m = m[np.asarray(fit_mask, dtype=bool)]
        except Exception:  # noqa: BLE001 - shape mismatch: judge the full frame
            pass
    m = m.ravel()
    finite = np.isfinite(m)
    cov = float(finite.mean()) if m.size else 0.0
    # Coverage floor: half of what the SAME method achieved on the anchor when
    # that is known (a dilated fit mask over-covers its emitter, so the
    # converged fraction is legitimately well below 1 — observed live at 43 %
    # on a mask-scoped follower, which a fixed 50 % floor wrongly rejected);
    # otherwise a lenient absolute floor.
    # Without a reference there is NO coverage floor: a small emitter fitted
    # full-frame legitimately covers ~1 % of the frame (observed live on a
    # legacy replay), and only the anchor's own coverage can say what this
    # method should reach. All-NaN maps are excluded before the gate.
    ref_cov = (reference or {}).get("coverage") if isinstance(reference, dict) else None
    if isinstance(ref_cov, (int, float)) and 0 < ref_cov <= 1:
        min_cov = 0.25 * float(ref_cov)
        if cov < min_cov:
            return False, (f"valid coverage {cov:.0%} < {min_cov:.0%} (a quarter of the "
                           f"anchor's {float(ref_cov):.0%}) — the locked method did not "
                           "converge here")
    vals = m[finite]
    # A constant map is a collapse only if the SAME method varied on the
    # anchor: a synthetic emitter with one exact centre, or a channel-
    # quantized position, is legitimately constant (observed live on a
    # mask-scoped follower the LLM review used to accept).
    if vals.size > 8 and float(np.ptp(vals)) == 0.0 and isinstance(reference, dict):
        try:
            ref_spread = float(reference.get("max")) - float(reference.get("min"))
        except (TypeError, ValueError):
            ref_spread = None
        if ref_spread is not None and ref_spread > 0:
            return False, ("map is constant across the frame while the anchor's varied "
                           f"over [{float(reference['min']):.4g}, {float(reference['max']):.4g}] "
                           "(fit collapsed to a bound)")
    # The range rule is for SIBLING datasets, where a required output far from
    # the anchor's means the method broke. In a stream the tracked quantity is
    # expected to move (a resonance shifting through a ramp, another part of a
    # sample), so a live loop sets ``values_may_move``: method health is judged
    # here (coverage, collapse), plausibility by the loop's own range gate, which
    # flags a jump and then adopts a value that keeps saying the same thing.
    # Observed live on tiles of one real EELS field: four of eleven tiles were
    # withheld for a plasmon 35 to 60 meV below the first tile's range.
    if (required and isinstance(reference, dict) and not reference.get("values_may_move")
            and all(isinstance(reference.get(k), (int, float)) for k in ("min", "max"))):
        lo, hi = float(reference["min"]), float(reference["max"])
        mean = float(reference.get("mean", (lo + hi) / 2.0))
        span = max(hi - lo, 0.005 * abs(mean), 1e-9)
        med = float(np.median(vals))
        if not (lo - span <= med <= hi + span):
            return False, (f"median {med:.4g} outside the locked method's plausible "
                           f"range [{lo - span:.4g}, {hi + span:.4g}] (anchor "
                           f"[{lo:.4g}, {hi:.4g}]) — method breakdown on this dataset")
    return True, ""


class MapReplayGate:
    """The hyperspectral agent's per-map replay gate (``map_health``) as a gate."""

    def judge(self, result_map, fit_mask, reference: Optional[dict], required: bool) -> Dict[str, Any]:
        ok, reason = map_health(result_map, fit_mask, reference, required)
        return replay_verdict("good" if ok else "poor", reasons=[reason], gate="map_health")


# ------------------------------------------------------------ select_recipe
def select_recipe(candidates: Sequence[Tuple[str, Optional[str]]],
                  run: Callable[[int, str, Optional[str]], Dict[str, Any]],
                  judge: Callable[[Dict[str, Any]], Dict[str, Any]],
                  *, strategy: str = "first_good") -> Dict[str, Any]:
    """Choose among a series' regime recipes on a reuse.

    ``candidates`` are ``(script, source)`` in lock order; ``run(n, script,
    source)`` replays one (the agent's own fit, in its own folder) and
    returns the result; ``judge(result)`` is the replay gate's verdict for a
    result that executed. The default strategy is today's rule: the FIRST
    candidate the gate calls good is chosen; when none is good, the first
    that executed (poor); when none executed, nothing. Returns
    ``{"chosen": n | None, "result", "verdict", "tried": [...]}`` with one
    entry per candidate tried (its index, source, executed, verdict)."""
    if strategy != "first_good":
        raise ValueError(f"unknown strategy {strategy!r}")
    tried: List[Dict[str, Any]] = []
    kept: Optional[Dict[str, Any]] = None
    last_failed: Optional[Dict[str, Any]] = None
    for n, (script, source) in enumerate(candidates, 1):
        result = run(n, script, source)
        if result.get("success"):
            verdict = judge(result)
            tried.append({"n": n, "source": source, "executed": True, "verdict": verdict["verdict"],
                          "score": verdict.get("score")})
            if verdict["verdict"] == "good":
                return {"chosen": n, "result": result, "verdict": verdict, "source": source, "tried": tried}
            if kept is None:
                kept = {"chosen": n, "result": result, "verdict": verdict, "source": source}
        else:
            last_failed = {"result": result, "source": source}
            tried.append({"n": n, "source": source, "executed": False, "verdict": "failed",
                          "error": result.get("error")})
    if kept is not None:
        return {**kept, "tried": tried}
    return {"chosen": None, "result": None, "verdict": None, "source": None, "tried": tried,
            "last_failed": last_failed}
