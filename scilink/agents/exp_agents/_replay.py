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
                           "interpretation_checked": bool(interpretation_checked), "regime": regime}
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


# ------------------------------------------------------------ same state
#: A measurement whose structure the regime's own curves cannot describe beyond
#: this share is not the regime's state (``DriftMonitor.judge()["fraction"]``:
#: 0 = nothing new, 1 = nothing in common; the live loop's material-change bar
#: is 0.10, a different phase reads 0.7 and more).
SAME_STATE_BAR = 0.25
#: Two regimes are not told apart by the data when the second-nearest is
#: within this ratio of the nearest, or both are under the material bar.
AMBIGUITY_RATIO = 2.0


def state_monitor(curves: Sequence[Tuple[Any, Any]] = (), *, state: Optional[Dict[str, Any]] = None):
    """A ``live/drift.py`` ``DriftMonitor`` seeded with a regime's curves
    (its units' data), or restored from a stamped ``drift_state`` (the
    anchor's curve on the monitor's grid, recorded when the recipe was
    locked). Model-free; the measure the live loop replaced its
    peak-counting fingerprint with, for exactly this question."""
    from ...live.drift import DriftMonitor
    m = DriftMonitor()
    if curves:
        m.seed(list(curves))
    elif state:
        m.load_state(state)
    return m


STAMP_MAX_POINTS = 32 * 256         # the longest x a stamp keeps as is (a same-instrument curve then
                                    # lands on the stamp's own grid, with no interpolation at all)
STAMP_MAX_CURVES = 12               # the most curves a regime's stamp carries


def _block_mean(x: np.ndarray, y: np.ndarray, block: int):
    """``x`` and ``y`` reduced by the means of consecutive blocks of ``block``
    points — the monitor's own reduction (``DriftMonitor._to_grid``), never a
    point interpolation, which samples the noise instead of averaging it and
    was measured to inflate a curve's distance to its own stamp 2–4×."""
    n = x.size // block
    m = block * n
    return x[:m].reshape(n, block).mean(axis=1), y[:m].reshape(n, block).mean(axis=1)


def drift_state_of(x: Any, y: Any) -> Optional[Dict[str, Any]]:
    """What a regime's recipe records about ONE curve: the monitor's state
    seeded with it (``DriftMonitor.to_state``). See ``drift_state_of_curves``
    for the regime's record."""
    return drift_state_of_curves([(x, y)])


def drift_state_of_curves(curves: Sequence[Tuple[Any, Any]]) -> Optional[Dict[str, Any]]:
    """What a regime's recipe records about its DATA: the monitor's state
    seeded with the regime's units' curves (at most ``STAMP_MAX_CURVES``,
    evenly spaced along the regime), so a later measurement is held to the
    regime's spread, not to its anchor alone — an anchor-only stamp failed
    half of a regime's own units (median distance 0.25–0.31). The curves
    go in at their own x (one longer than ``STAMP_MAX_POINTS`` reduced by
    block means), on the grid the monitor builds itself."""
    try:
        from ...live.drift import N_GRID
        prepared = []
        for cx, cy in curves:
            xa, ya = np.asarray(cx, dtype=float).ravel(), np.asarray(cy, dtype=float).ravel()
            n = min(xa.size, ya.size)
            xa, ya = xa[:n], ya[:n]
            if n < 16:
                continue
            order = np.argsort(xa, kind="stable")
            xa, ya = xa[order], ya[order]
            if n > STAMP_MAX_POINTS:
                xa, ya = _block_mean(xa, ya, -(-n // STAMP_MAX_POINTS))
            prepared.append((xa, ya))
        if not prepared:
            return None
        if len(prepared) > STAMP_MAX_CURVES:
            keep = np.unique(np.round(np.linspace(0, len(prepared) - 1, STAMP_MAX_CURVES)).astype(int))
            prepared = [prepared[int(i)] for i in keep]
        m = state_monitor(prepared)
        st = m.to_state()
        return st if st.get("x") is not None and st.get("seed") else None
    except Exception:  # noqa: BLE001 - a stamp is a side note on a lock
        return None


def state_distance(monitor, x: Any, y: Any) -> Optional[float]:
    """How much of this curve the regime's curves cannot describe (0..1)."""
    try:
        out = monitor.judge(x, y)
    except Exception:  # noqa: BLE001
        return None
    if not out.get("available"):
        return None
    return float(out["fraction"])


# ------------------------------------------------------------ identity
#: Without a spread (one reference sample) a numeric feature this far from it,
#: as a fraction of its magnitude, is flagged — never silently verified.
IDENTITY_TOLERANCE = 0.05
#: A position within a component is one of these; the component's strength one
#: of _STRENGTH_WORDS. Matched on the parameter's own name, not its component's.
_POSITION_WORDS = ("center", "centre", "position", "pos", "mu", "x0", "shift", "spacing", "d_spacing", "dspacing",
                   "energy", "wavenumber", "two_theta", "2theta", "theta", "angle", "loc")
_NOT_POSITION = ("amplitude", "height", "intensity", "area", "width", "fwhm", "sigma", "gamma", "fraction", "ratio",
                 "count", "error", "err", "std", "unc", "r2", "r_squared", "chi")
_STRENGTH_WORDS = ("amplitude", "height", "intensity", "area", "integrated", "weight")
#: A recipe that IDENTIFIES (a phase-search XRD recipe) reports its identity
#: as names, not positions: these keys are categorical identity. A database
#: id is not: the same phase has several entries.
_IDENTITY_WORDS = ("phase", "space_group", "spacegroup", "symmetry", "polymorph", "structure_type", "assignment")
#: Positions weaker than this share of the strongest are noise-level features
#: an auto-detect recipe finds on some units and not others: not identity.
STRONG_SHARE = 0.10
#: The floor of a position's tolerance, as a share of the data's x-range,
#: so a near-zero position does not get a near-zero tolerance.
POSITION_FLOOR_SHARE = 0.01


def _is_position(key: str) -> bool:
    k = str(key).lower()
    return any(w in k for w in _POSITION_WORDS) and not any(w in k for w in _NOT_POSITION)


def _norm_label(value: Any) -> str:
    """A name as identity: lower case, a trailing space-group SETTING
    dropped (``I 41/a m d :2`` and ``I41/amd`` are one group), every run of
    non-alphanumerics one space — so the TOKENS survive (``TiO2 (anatase)``
    is {tio2, anatase}, not "tio2": a stripped qualifier erased the phase).
    ``names_match`` compares two of these. A number against a symbol (141
    vs I41/amd) still differs: that needs a table, not a rule."""
    import re as _re
    text = str(value).strip()
    text = _re.sub(r"\s*:\s*[A-Za-z0-9]{1,2}\s*$", "", text)  # a trailing ":2" / ":H" setting
    return " ".join(_re.split(r"[^0-9a-z]+", text.lower())).strip()


def names_match(a: str, b: str) -> bool:
    """Two normalised names are one identity when they are the same string
    with the spaces removed (``i 41 a m d`` vs ``i41 amd``), or when one's
    tokens are a subset of the other's (``tio2 anatase`` ⊇ ``anatase``);
    ``tio2 anatase`` against ``tio2 rutile`` is neither."""
    if not a or not b:
        return False
    if a.replace(" ", "") == b.replace(" ", ""):
        return True
    ta, tb = set(a.split()), set(b.split())
    return bool(ta and tb) and (ta <= tb or tb <= ta)


def identity_features(parameters: Any) -> Dict[str, Any]:
    """What a curve fit says it found, as identity: ``names`` (a phase, a
    space group — normalised), and ``positions`` as ``(position, strength)``
    pairs per component (nested ``{peak_1: {center, amplitude}}`` or flat
    ``peak1_center`` / ``peak1_amplitude`` layouts). Amplitudes and widths are
    strengths and shapes, not identity; component COUNT is not identity (an
    auto-detect recipe finds noise peaks on some units and not others)."""
    out: Dict[str, Any] = {"names": {}, "positions": []}
    if not isinstance(parameters, dict):
        return out
    groups: Dict[str, Dict[str, Any]] = {}
    flat: set = set()
    for k, v in parameters.items():
        key = str(k)
        if isinstance(v, dict):
            groups.setdefault(key, {}).update({str(kk): vv for kk, vv in v.items()})
        elif isinstance(v, str) and v.strip() and any(w in key.lower() for w in _IDENTITY_WORDS):
            out["names"][key] = _norm_label(v)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            # a flat layout: peak1_center / peak1_amplitude → component "peak1";
            # bare center / amplitude / sigma → the one component of a single fit
            stem, _, leaf = key.rpartition("_") if "_" in key else ("", "", key)
            if _is_position(leaf) or any(w in leaf.lower() for w in _STRENGTH_WORDS):
                groups.setdefault(stem, {})[leaf] = v
                flat.add(stem)
    for comp, fields in groups.items():
        if comp in flat and not ("peak" in comp.lower() or any(
                any(w in str(kk).lower() for w in _STRENGTH_WORDS) for kk in fields)):
            # a lone flat number with a position-like name (activation_energy,
            # mu_shift, fitted_zero_shift) is not a component's position
            continue
        pos = next((float(vv) for kk, vv in fields.items()
                    if isinstance(vv, (int, float)) and not isinstance(vv, bool) and _is_position(kk)), None)
        if pos is None or not math.isfinite(pos):
            continue
        strength = next((abs(float(vv)) for kk, vv in fields.items()
                         if isinstance(vv, (int, float)) and not isinstance(vv, bool)
                         and any(w in str(kk).lower() for w in _STRENGTH_WORDS)), None)
        out["positions"].append((pos, strength))
    return out


def strong_positions(feats: Dict[str, Any]) -> List[float]:
    pts = [(p, s) for p, s in feats.get("positions") or [] if p is not None]
    if not pts:
        return []
    known = [s for _, s in pts if s is not None]
    if not known:
        return sorted(p for p, _ in pts)
    top = max(known)
    return sorted(p for p, s in pts if s is None or (top > 0 and s >= STRONG_SHARE * top))


def identity_reference(samples: Sequence[Dict[str, Any]], *, x_range: Optional[float] = None) -> Dict[str, Any]:
    """The reference an identity check compares against, from the regime's
    units' identity features: per name, the set of values the units reported
    (``n`` units); the units' strong positions clustered by nearest
    neighbour, each cluster the range the units put it in; and the tolerance
    floor (``POSITION_FLOOR_SHARE`` of the x-range)."""
    samples = [s for s in samples if isinstance(s, dict)]
    names: Dict[str, Dict[str, Any]] = {}
    for s in samples:
        for k, v in (s.get("names") or {}).items():
            names.setdefault(k, {"values": set(), "n": 0})
            names[k]["values"].add(v)
            names[k]["n"] += 1
    floor = POSITION_FLOOR_SHARE * float(x_range) if x_range else None
    per_unit = [strong_positions(s) for s in samples]
    allpos = sorted((p, i) for i, ps in enumerate(per_unit) for p in ps)
    if floor is None and allpos:
        span = allpos[-1][0] - allpos[0][0]
        floor = 0.01 * span if span > 0 else max(abs(allpos[0][0]) * 0.01, 1e-6)
    clusters: List[Dict[str, Any]] = []
    for p, i in allpos:
        if clusters and p - clusters[-1]["max"] <= 2 * (floor or 0):
            clusters[-1]["max"] = p
            clusters[-1]["units"].add(i)
        else:
            clusters.append({"min": p, "max": p, "units": {i}})
    return {"names": {k: {"values": sorted(v["values"]), "n": v["n"]} for k, v in names.items()},
            "clusters": [{"min": c["min"], "max": c["max"], "n_units": len(c["units"])} for c in clusters],
            "n_units": len(samples), "floor": floor}


def identity_check(feats: Dict[str, Any], reference: Dict[str, Any]) -> Dict[str, Any]:
    """Is the replayed result the SAME KIND of thing the regime's units found?

    Names: a value outside the set the units reported is drift (several units
    agreeing is a spread of its own). Positions: each of the replay's STRONG
    positions must fall in some cluster the units' strong positions formed
    (widened by the cluster's own span, at least the floor), and each cluster
    EVERY unit had must have a strong position near it — a new strong
    feature, or a strong feature gone, is drift; a weak peak coming and going
    is not. Returns ``{"checked", "spread_known", "within", "drifted",
    "compared"}``; only a check against two or more units is a spread."""
    drifted: List[Dict[str, Any]] = []
    compared = 0
    n_units = int(reference.get("n_units") or 0)
    for name, ref in (reference.get("names") or {}).items():
        got = (feats.get("names") or {}).get(name)
        if got is None:
            continue
        compared += 1
        if not any(names_match(got, v) for v in (ref.get("values") or [])):
            drifted.append({"name": name, "value": got, "reference": list(ref.get("values") or [])})
    clusters = reference.get("clusters") or []
    floor = float(reference.get("floor") or 0.0)
    mine = strong_positions(feats)
    if clusters and mine:
        compared += 1
        for p in mine:
            if not any(c["min"] - max(c["max"] - c["min"], floor) <= p <= c["max"] + max(c["max"] - c["min"], floor)
                       for c in clusters):
                drifted.append({"name": "position", "value": p, "reference": "no strong feature of the regime near it"})
        for c in clusters:
            if n_units and c["n_units"] >= n_units:          # a feature every unit had
                tol = max(c["max"] - c["min"], floor)
                if not any(c["min"] - tol <= p <= c["max"] + tol for p in mine):
                    drifted.append({"name": "position", "value": None, "reference": [c["min"], c["max"]],
                                    "missing": True})
    out = {"checked": compared > 0, "spread_known": n_units >= 2, "within": not drifted,
           "drifted": drifted, "compared": compared}
    if compared == 0:
        # said, never silent: a layout the reader finds no identity in (a
        # flat XPS / EPR parameter set with no component label) is not
        # checked, and the record says why
        out["reason"] = ("the fitted parameters carry no names and no strong positions the reader recognises"
                         if not (feats.get("names") or feats.get("positions"))
                         else "the regime's units carry nothing comparable to what the recipe reports")
    return out


# ------------------------------------------------------------ select_recipe
def select_recipe(candidates: Sequence[Tuple[str, Optional[str]]],
                  run: Callable[[int, str, Optional[str]], Dict[str, Any]],
                  judge: Callable[..., Dict[str, Any]],
                  *, strategy: str = "first_good",
                  distances: Optional[Sequence[Optional[float]]] = None,
                  ambiguity_ratio: float = AMBIGUITY_RATIO, bar: float = 0.10) -> Dict[str, Any]:
    """Choose among a series' regime recipes on a reuse.

    ``candidates`` are ``(script, source)`` in lock order; ``run(n, script,
    source)`` replays one (the agent's own fit, in its own folder) and
    returns the result; ``judge(n, result)`` is the verdict on a result that
    executed — the replay gate, and whatever else the caller holds a
    candidate to (the same-state and identity checks of #711), so a
    candidate that fits but is not the regime's falls through to the next.
    ``first_good``: the FIRST candidate judged good is chosen; when none is
    good, the first that executed (poor); when none executed, nothing.
    ``nearest_first`` (#710): the candidates are tried in order of
    ``distances`` (how much of the new data the regime's own curves cannot
    describe; unknown ones last, lock order among ties), then the same rule;
    the choice is ``ambiguous`` when the second-nearest is within
    ``ambiguity_ratio`` of the nearest or both are under ``bar``. Returns
    ``{"chosen": n | None, "result", "verdict", "source", "tried", "order",
    "ambiguous", "margin"}``; ``n`` indexes ``candidates``."""
    if strategy not in ("first_good", "nearest_first"):
        raise ValueError(f"unknown strategy {strategy!r}")
    order = list(range(1, len(candidates) + 1))
    ambiguous, margin = False, None
    if strategy == "nearest_first":
        dists = list(distances or [])
        if len(dists) != len(candidates):
            raise ValueError("nearest_first needs one distance (or None) per candidate")
        known = [(d, n) for n, d in zip(order, dists) if isinstance(d, (int, float))]
        order = ([n for _, n in sorted(known, key=lambda dn: (dn[0], dn[1]))]
                 + [n for n, d in zip(order, dists) if not isinstance(d, (int, float))])
        if len(known) >= 2:
            best, second = sorted(d for d, _ in known)[:2]
            margin = round(second - best, 4)
            ambiguous = bool(second <= ambiguity_ratio * max(best, 1e-9) or (best <= bar and second <= bar))
    tried: List[Dict[str, Any]] = []
    kept: Optional[Dict[str, Any]] = None
    last_failed: Optional[Dict[str, Any]] = None
    for n in order:
        script, source = candidates[n - 1]
        result = run(n, script, source)
        if result.get("success"):
            verdict = judge(n, result)
            tried.append({"n": n, "source": source, "executed": True, "verdict": verdict["verdict"],
                          "score": verdict.get("score")})
            if verdict["verdict"] == "good":
                return {"chosen": n, "result": result, "verdict": verdict, "source": source, "tried": tried,
                        "order": order, "ambiguous": ambiguous, "margin": margin}
            if kept is None:
                kept = {"chosen": n, "result": result, "verdict": verdict, "source": source}
        else:
            last_failed = {"result": result, "source": source}
            tried.append({"n": n, "source": source, "executed": False, "verdict": "failed",
                          "error": result.get("error")})
    if kept is not None:
        return {**kept, "tried": tried, "order": order, "ambiguous": ambiguous, "margin": margin}
    return {"chosen": None, "result": None, "verdict": None, "source": None, "tried": tried,
            "last_failed": last_failed, "order": order, "ambiguous": ambiguous, "margin": margin}
