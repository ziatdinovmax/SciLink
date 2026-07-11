"""Script bank — episodic memory of successful analysis scripts.

Every approved analysis already produces a working script; without the bank
that history is inert unless the user hand-points ``prior_analysis_paths`` at
a specific run, or a rare hot success goes through the skill-graduation
ceremony. The bank makes all of it retrievable: on every approved result the
agents append a record here, and at analysis time the bank can be searched so
a proven script is *adapted* instead of re-implemented from scratch.

This is episodic memory alongside graduation's semantic memory: every
success, zero ceremony (no distillation, no review gate). A record carries
three tiers of matching signal, all computed deterministically at write time
(no LLM):

1. **Measurement context** — instrument / technique / sample / conditions,
   trimmed from the run's metadata.
2. **Data fingerprint** — numeric summary of the data the script actually
   solved (axis range, peaks, SNR, …), so retrieval can say "this NEW
   spectrum looks like the one this script solved" even when sample names
   differ.
3. **Outcome** — the verbatim script, model/pipeline type, gate metric,
   plan summary, session provenance.

Records live at ``scilink_home()/script_bank/<domain>/<id>.json`` — a sibling
of ``graduated_skills/`` and ``distill_staging/``, honoring ``$SCILINK_HOME``.
Re-banking the same script (hash match) updates the existing record's usage
stats instead of duplicating it; those stats (how often a script keeps
succeeding across sessions) are the intended evidence-based promotion signal
for skill graduation.

The module is package-neutral: stdlib + numpy/scipy (both hard deps), no
``ase``, no agent imports.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..loader import scilink_home, memory_enabled, _TRUTHY, _FALSY
from ._graduation import safe_path_component, warn_if_ephemeral_store


def bank_dir() -> Path:
    """Root of the script bank (honors ``$SCILINK_HOME``)."""
    return scilink_home() / "script_bank"


def bank_enabled() -> bool:
    """Whether the bank write hooks are active.

    ``SCILINK_SCRIPT_BANK`` overrides in both directions (so the bank can run
    without the full persistent-memory feature, e.g. for real-time reuse, or
    be switched off while memory stays on); otherwise it follows the
    persistent-memory master switch.
    """
    flag = os.environ.get("SCILINK_SCRIPT_BANK", "").strip().lower()
    if flag in _FALSY:
        return False
    if flag in _TRUTHY:
        return True
    return memory_enabled()


def _domain_dir(domain: str, *, root: Optional[Path] = None) -> Path:
    # domain is a filesystem component — sanitize to prevent path traversal.
    return (root or bank_dir()) / safe_path_component(domain, fallback="unknown_domain")


def script_hash(script: str) -> str:
    """Content hash identifying a script up to trailing whitespace."""
    normalized = "\n".join(line.rstrip() for line in (script or "").strip().splitlines())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


# ──────────────────────────────────────────────────────────────
# CRUD
# ──────────────────────────────────────────────────────────────

# A locked-recipe series banks once per run; sessions accumulate across runs.
_MAX_SESSIONS = 20


def add_record(domain: str, record: Dict[str, Any], *, root: Optional[Path] = None) -> Dict[str, Any]:
    """Bank one successful script; return ``{"id", "action"}``.

    ``record`` must carry ``working_script``; everything else (context,
    fingerprint, outcome fields) is stored as given. If a record with the
    same script hash already exists in the domain, its usage stats are
    updated (``n_successes``, ``sessions``, best metric) instead of writing a
    duplicate — call this once per distinct script per run so ``n_successes``
    counts runs, not items in a series.
    """
    script = (record.get("working_script") or "").strip()
    if not script:
        return {"id": None, "action": "skipped_no_script"}
    h = script_hash(script)

    existing = _find_by_hash(domain, h, root=root)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    session = (record.get("provenance") or {}).get("session")
    metric = (record.get("outcome") or {}).get("metric")

    if existing is not None:
        rec, path = existing
        # Backfill matching tiers an earlier write couldn't compute.
        for key in ("data_fingerprint", "measurement_context", "technique_signals"):
            if not rec.get(key) and record.get(key):
                rec[key] = record[key]
        stats = rec.setdefault("stats", {"n_successes": 1, "n_retrievals": 0})
        stats["n_successes"] = int(stats.get("n_successes", 1)) + 1
        sessions = rec.setdefault("sessions", [])
        if session and session not in sessions:
            sessions.append(session)
            del sessions[:-_MAX_SESSIONS]
        if metric is not None:
            rec.setdefault("outcome", {})["last_metric"] = metric
            best = rec["outcome"].get("best_metric") or rec["outcome"].get("metric")
            if _metric_value(metric) is not None and (
                _metric_value(best) is None or _metric_value(metric) > _metric_value(best)
            ):
                rec["outcome"]["best_metric"] = metric
        rec["updated_at"] = now
        path.write_text(json.dumps(rec, indent=2, default=str))
        return {"id": rec["id"], "action": "updated"}

    warn_if_ephemeral_store()
    d = _domain_dir(domain, root=root)
    d.mkdir(parents=True, exist_ok=True)
    rid = uuid.uuid4().hex[:8]
    payload = {
        "id": rid,
        "domain": domain,
        "script_hash": h,
        "created_at": now,
        "sessions": [session] if session else [],
        "stats": {"n_successes": 1, "n_retrievals": 0},
        **record,
    }
    (d / f"{rid}.json").write_text(json.dumps(payload, indent=2, default=str))
    return {"id": rid, "action": "created"}


def _metric_value(metric: Any) -> Optional[float]:
    if isinstance(metric, dict):
        metric = metric.get("value")
    try:
        v = float(metric)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _find_by_hash(domain: str, h: str, *, root: Optional[Path] = None):
    d = _domain_dir(domain, root=root)
    if not d.is_dir():
        return None
    for f in sorted(d.glob("*.json")):
        try:
            rec = json.loads(f.read_text())
        except Exception:
            continue
        if rec.get("script_hash") == h:
            return rec, f
    return None


def list_records(domain: Optional[str] = None, *, root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return bank records, optionally filtered by domain."""
    base = root or bank_dir()
    out: List[Dict[str, Any]] = []
    if not base.is_dir():
        return out
    domains = [base / domain] if domain else [
        p for p in sorted(base.iterdir())
        if p.is_dir() and not p.name.startswith((".", "_"))
    ]
    for dd in domains:
        if not dd.is_dir():
            continue
        for f in sorted(dd.glob("*.json")):
            try:
                rec = json.loads(f.read_text())
            except Exception:
                continue
            rec.setdefault("domain", dd.name)
            out.append(rec)
    return out


def get_record(domain: str, rid: str, *, root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    f = _domain_dir(domain, root=root) / f"{rid}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


def remove_records(domain: str, ids: List[str], *, root: Optional[Path] = None) -> int:
    """Delete bank records by id; return count removed."""
    d = _domain_dir(domain, root=root)
    n = 0
    for rid in ids:
        f = d / f"{rid}.json"
        if f.exists():
            f.unlink()
            n += 1
    return n


# ──────────────────────────────────────────────────────────────
# Tier 1 — measurement context (trimmed metadata)
# ──────────────────────────────────────────────────────────────

_CTX_MAX_FIELDS = 40
_CTX_MAX_CHARS = 300


def measurement_context(system_info: Any) -> Dict[str, Any]:
    """Trim run metadata into a compact, JSON-safe matching record.

    Keeps scalar fields (and short lists) up to a size cap; nested dicts are
    flattened one level with dotted keys. Free-form — whatever the user's
    metadata names (instrument, technique, sample, conditions) is what
    retrieval will soft-match on.
    """
    out: Dict[str, Any] = {}
    if isinstance(system_info, str):
        return {"description": system_info[:_CTX_MAX_CHARS * 4]}
    if not isinstance(system_info, dict):
        return out

    def _clip(v: Any) -> Any:
        if isinstance(v, str):
            return v[:_CTX_MAX_CHARS]
        if isinstance(v, (int, float, bool)) or v is None:
            return v
        if isinstance(v, (list, tuple)) and len(v) <= 12:
            return [_clip(x) for x in v]
        return str(v)[:_CTX_MAX_CHARS]

    for key, value in system_info.items():
        if len(out) >= _CTX_MAX_FIELDS:
            break
        if isinstance(value, dict):
            for k2, v2 in value.items():
                if len(out) >= _CTX_MAX_FIELDS:
                    break
                out[f"{key}.{k2}"] = _clip(v2)
        else:
            out[str(key)] = _clip(value)
    return out


_X_UNIT_KEYS = ("x_units", "x_unit", "axis_units", "x_axis_units",
                "xlabel", "x_label", "units")


def guess_x_units(system_info: Any) -> Optional[str]:
    """Best-effort x-axis units from free-form metadata (retrieval hard-filter)."""
    if isinstance(system_info, dict):
        for key in _X_UNIT_KEYS:
            v = system_info.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()[:40]
    return None


# ──────────────────────────────────────────────────────────────
# Tier 2 — data fingerprints (deterministic, numpy/scipy only)
# ──────────────────────────────────────────────────────────────

def _r(v: Any, nd: int = 4) -> Optional[float]:
    try:
        f = float(v)
        return round(f, nd) if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _peak_summary(x: np.ndarray, y: np.ndarray, k: int = 5) -> Dict[str, Any]:
    """Robust peak census of a 1D signal: count + top-k positions/widths.

    The signal is lightly smoothed and the prominence threshold adapts to its
    own point-to-point noise (reduced by the smoothing window), so the census
    is stable across intensity scales and noise levels; positions/widths are
    in x-axis units.
    """
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks, peak_widths

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 8:
        return {"count": 0, "top": []}

    lo, hi = np.percentile(y, [0.5, 99.5])
    rng = float(hi - lo)
    if rng <= 0:
        return {"count": 0, "top": []}
    yn = (y - lo) / rng
    # Per-point noise from first differences (signal varies slowly; noise doesn't).
    noise = float(np.median(np.abs(np.diff(yn)))) / np.sqrt(2)
    window = max(3, y.size // 400)
    ys = uniform_filter1d(yn, window)
    # 12σ post-smoothing threshold: exact peak counts across noise levels on
    # synthetic benchmarks (3-peak clean+noisy, 12-peak crowded, weak shoulder).
    prominence = max(0.05, 12.0 * noise / np.sqrt(window))
    distance = max(2, y.size // 200)
    peaks, props = find_peaks(ys, prominence=prominence, distance=distance)
    if peaks.size == 0:
        return {"count": 0, "top": []}

    widths_samples = peak_widths(ys, peaks, rel_height=0.5)[0]
    dx = float(np.median(np.abs(np.diff(x)))) if x.size > 1 else 1.0
    order = np.argsort(props["prominences"])[::-1][:k]
    top = [
        {
            "position": _r(x[peaks[i]]),
            "fwhm": _r(widths_samples[i] * dx),
            "prominence": _r(props["prominences"][i], 3),
        }
        for i in order
    ]
    return {"count": int(peaks.size), "top": top}


def _snr_estimate(y: np.ndarray) -> Optional[float]:
    """(p99 − p50) over first-difference noise, capped — scale-free SNR."""
    y = np.asarray(y, dtype=float).ravel()
    y = y[np.isfinite(y)]
    if y.size < 8:
        return None
    noise = float(np.median(np.abs(np.diff(y)))) / np.sqrt(2)
    if noise <= 0:
        return 1000.0
    p50, p99 = np.percentile(y, [50, 99])
    return _r(min(float(p99 - p50) / noise, 1000.0), 1)


def curve_fingerprint(x: Any, y: Any, x_units: Optional[str] = None) -> Dict[str, Any]:
    """Fingerprint of a 1D curve: what a banked script actually solved."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    fp: Dict[str, Any] = {"kind": "curve", "n_points": int(n)}
    if n == 0:
        return fp
    fp["x_units"] = x_units
    fp["x_range"] = [_r(np.nanmin(x)), _r(np.nanmax(x))]
    fp["snr"] = _snr_estimate(y)
    fp["peaks"] = _peak_summary(x, y)
    # Baseline character: net drift + where the median sits in the dynamic
    # range (≈0 for peaks-on-flat-floor, ≈0.5 for oscillatory/step data).
    yf = y[np.isfinite(y)]
    if yf.size >= 10:
        lo, hi = np.percentile(yf, [0.5, 99.5])
        rng = float(hi - lo)
        if rng > 0:
            tail = max(1, yf.size // 10)
            drift = (np.median(yf[-tail:]) - np.median(yf[:tail])) / rng
            fp["baseline"] = {
                "drift": _r(drift, 3),
                "median_level": _r((np.median(yf) - lo) / rng, 3),
            }
    return fp


def image_fingerprint(image: Any, pixel_size_nm: Optional[float] = None) -> Dict[str, Any]:
    """Fingerprint of a 2D image: scale, contrast, edges, periodicity."""
    img = np.asarray(image, dtype=float)
    if img.ndim == 3:  # collapse channels
        img = img.mean(axis=-1)
    fp: Dict[str, Any] = {"kind": "image", "shape": [int(s) for s in img.shape]}
    if img.ndim != 2 or img.size == 0:
        return fp
    fp["pixel_size_nm"] = _r(pixel_size_nm) if pixel_size_nm else None

    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return fp
    p1, p50, p99 = np.percentile(finite, [1, 50, 99])
    rng = float(p99 - p1)
    fp["intensity"] = {
        "p1": _r(p1), "p50": _r(p50), "p99": _r(p99),
        "contrast": _r(float(np.std(finite)) / rng, 3) if rng > 0 else None,
    }

    # Downsample deterministically to bound cost on large frames.
    step = max(1, max(img.shape) // 512)
    small = np.nan_to_num(img[::step, ::step], nan=float(p50))
    if rng > 0 and min(small.shape) >= 16:
        gy, gx = np.gradient(small)
        fp["edge_density"] = _r(float(np.mean(np.hypot(gx, gy))) / rng, 4)
        # Periodicity: strongest non-DC ring of the radial power spectrum vs
        # its median — high for lattices/gratings, ~1 for texture-free noise.
        f = np.abs(np.fft.rfft2(small - small.mean())) ** 2
        f[0, 0] = 0.0
        ky = np.fft.fftfreq(small.shape[0])[:, None]
        kx = np.fft.rfftfreq(small.shape[1])[None, :]
        kr = np.hypot(ky, kx)
        nbins = 32
        bins = np.minimum((kr / (kr.max() or 1.0) * nbins).astype(int), nbins - 1)
        ring = np.array([f[bins == b].mean() if np.any(bins == b) else 0.0
                         for b in range(1, nbins)])
        med = float(np.median(ring[ring > 0])) if np.any(ring > 0) else 0.0
        fp["fft_periodicity"] = _r(float(ring.max()) / med, 2) if med > 0 else None
    return fp


# ──────────────────────────────────────────────────────────────
# Retrieval — adapt-mode exemplar lookup (deterministic v1)
#
# Ranking = fingerprint similarity (dominant) + context token overlap +
# a small cross-session usage bonus. A single best match above the score
# floor is offered to the FIRST codegen attempt as an exemplar to adapt;
# the verification loop backstops a wrong pick and the annealing hot
# script-drop keeps a bad exemplar from trapping the loop.
# ──────────────────────────────────────────────────────────────

_MIN_EXEMPLAR_SCORE = 0.45


def _interval_overlap(a: Any, b: Any) -> float:
    """Jaccard overlap of two [lo, hi] intervals (0 when unknown)."""
    try:
        a0, a1 = sorted(float(v) for v in a)
        b0, b1 = sorted(float(v) for v in b)
    except (TypeError, ValueError):
        return 0.0
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union <= 0:
        return 1.0 if a0 == b0 else 0.0
    return inter / union


def _peak_positions(fp: Dict[str, Any]):
    return [(p.get("position"), p.get("fwhm"))
            for p in (fp.get("peaks") or {}).get("top", [])
            if p.get("position") is not None]


def _peak_match(fp_a: Dict[str, Any], fp_b: Dict[str, Any]) -> float:
    """Symmetric fraction of top peaks matched within a width-aware tolerance."""
    pa, pb = _peak_positions(fp_a), _peak_positions(fp_b)
    ca = (fp_a.get("peaks") or {}).get("count", len(pa)) or 0
    cb = (fp_b.get("peaks") or {}).get("count", len(pb)) or 0
    if ca == 0 and cb == 0:
        return 1.0  # two featureless signals match
    if not pa or not pb:
        return 0.0
    try:
        span = abs(float(fp_a["x_range"][1]) - float(fp_a["x_range"][0]))
    except (TypeError, ValueError, KeyError, IndexError):
        span = abs(max(p for p, _ in pa) - min(p for p, _ in pa)) or 1.0

    def _frac(src, dst):
        hit = 0
        for pos, fwhm in src:
            tol = max(fwhm or 0.0, 0.01 * span)
            if any(abs(pos - q) <= max(tol, qw or 0.0) for q, qw in dst):
                hit += 1
        return hit / len(src)

    return 0.5 * (_frac(pa, pb) + _frac(pb, pa))


def _count_similarity(na: Any, nb: Any) -> float:
    try:
        na, nb = int(na), int(nb)
    except (TypeError, ValueError):
        return 0.0
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    return min(na, nb) / max(na, nb)


def _curve_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    r = _interval_overlap(a.get("x_range"), b.get("x_range"))
    pk = _peak_match(a, b)
    cs = _count_similarity((a.get("peaks") or {}).get("count"),
                           (b.get("peaks") or {}).get("count"))
    return 0.35 * r + 0.45 * pk + 0.20 * cs


def _log_ratio_sim(a: Any, b: Any, scale: float = 1.0) -> Optional[float]:
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return None
    if a <= 0 or b <= 0:
        return 1.0 if a == b else 0.0
    return float(np.exp(-abs(np.log(a / b)) / scale))


def _image_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    sims = []
    ia, ib = a.get("intensity") or {}, b.get("intensity") or {}
    if ia.get("contrast") is not None and ib.get("contrast") is not None:
        sims.append(max(0.0, 1.0 - abs(ia["contrast"] - ib["contrast"]) / 0.5))
    if a.get("edge_density") is not None and b.get("edge_density") is not None:
        sims.append(max(0.0, 1.0 - abs(a["edge_density"] - b["edge_density"]) / 0.2))
    s = _log_ratio_sim(a.get("fft_periodicity"), b.get("fft_periodicity"), scale=2.0)
    if s is not None:
        sims.append(s)
    s = _log_ratio_sim(a.get("pixel_size_nm"), b.get("pixel_size_nm"), scale=1.0)
    if s is not None:
        sims.append(s)
    return float(np.mean(sims)) if sims else 0.0


def _hs_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ax_a, ax_b = a.get("axis") or {}, b.get("axis") or {}
    r = _interval_overlap((ax_a.get("start"), ax_a.get("end")),
                          (ax_b.get("start"), ax_b.get("end")))
    cos = 0.0
    ba = [v if v is not None else 0.0 for v in (a.get("band_means") or [])]
    bb = [v if v is not None else 0.0 for v in (b.get("band_means") or [])]
    if ba and bb and len(ba) == len(bb):
        va, vb = np.asarray(ba, float), np.asarray(bb, float)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom > 0:
            cos = float(np.dot(va, vb) / denom)
    # Reuse the curve peak matcher over the mean-spectrum census; it reads
    # x_range for its tolerance span, so alias the axis range.
    pk = _peak_match({**a, "x_range": [ax_a.get("start"), ax_a.get("end")]}, b)
    # Damp by range agreement: band_means are range-relative, so an identical
    # profile in a DISJOINT axis window (a different measurement) would
    # otherwise score high on cosine alone.
    return (0.30 * r + 0.40 * cos + 0.30 * pk) * (0.4 + 0.6 * r)


_SIMILARITY_FNS = {
    "curve": _curve_similarity,
    "image": _image_similarity,
    "hyperspectral": _hs_similarity,
}


def _context_tokens(*sources: Any) -> set:
    import re
    words = set()
    for src in sources:
        if isinstance(src, dict):
            text = " ".join(str(v) for v in src.values())
        else:
            text = str(src or "")
        words.update(w for w in re.findall(r"[a-z0-9]+", text.lower())
                     if len(w) >= 3)
    return words


def _context_similarity(query: set, rec: Dict[str, Any]) -> float:
    tokens = _context_tokens(rec.get("measurement_context"),
                             rec.get("technique_signals"))
    if not query or not tokens:
        return 0.0
    return len(query & tokens) / len(query | tokens)


def _units_of(fp: Dict[str, Any]) -> Optional[str]:
    u = fp.get("x_units") or (fp.get("axis") or {}).get("units")
    return u.strip().lower() if isinstance(u, str) and u.strip() else None


def find_exemplar(domain: str, fingerprint: Optional[Dict[str, Any]],
                  context: Any = None, *, k: int = 1,
                  min_score: float = _MIN_EXEMPLAR_SCORE,
                  root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Rank bank records against a new dataset; return the top match(es).

    Hard filters: fingerprint kind must match; axis units must agree when
    both sides declare them. Score = 0.8 × fingerprint similarity + 0.2 ×
    context token overlap + a small usage bonus (capped at 0.05) for scripts
    that keep succeeding across sessions. Returns ``[{"record", "score",
    "fingerprint_score"}, ...]`` above ``min_score`` — empty when nothing in
    the bank resembles this data (offering a poor exemplar is worse than
    generating from scratch).
    """
    if not fingerprint or not fingerprint.get("kind"):
        return []
    kind = fingerprint["kind"]
    simfn = _SIMILARITY_FNS.get(kind)
    if simfn is None:
        return []
    query_units = _units_of(fingerprint)
    query_tokens = _context_tokens(context)

    scored = []
    for rec in list_records(domain, root=root):
        fp = rec.get("data_fingerprint") or {}
        if fp.get("kind") != kind or not (rec.get("working_script") or "").strip():
            continue
        rec_units = _units_of(fp)
        if query_units and rec_units and query_units != rec_units:
            continue
        s_fp = simfn(fingerprint, fp)
        s_ctx = _context_similarity(query_tokens, rec)
        n_succ = (rec.get("stats") or {}).get("n_successes", 1) or 1
        usage = min(0.05, 0.01 * (int(n_succ) - 1))
        score = 0.8 * s_fp + 0.2 * s_ctx + usage
        if score >= min_score:
            scored.append((score, s_fp, rec))
    scored.sort(key=lambda t: (-t[0], t[2].get("id", "")))
    return [{"record": rec, "score": round(sc, 3), "fingerprint_score": round(sfp, 3)}
            for sc, sfp, rec in scored[:k]]


def mark_retrieved(domain: str, rid: str, *, root: Optional[Path] = None) -> None:
    """Increment a record's retrieval counter (usage stat; never raises)."""
    try:
        f = _domain_dir(domain, root=root) / f"{rid}.json"
        rec = json.loads(f.read_text())
        stats = rec.setdefault("stats", {})
        stats["n_retrievals"] = int(stats.get("n_retrievals", 0)) + 1
        f.write_text(json.dumps(rec, indent=2, default=str))
    except Exception:
        pass


def record_success(domain: str, rid: str, session: Optional[str] = None,
                   *, root: Optional[Path] = None) -> None:
    """Bump a record's cross-session success stats without re-banking.

    For verbatim cold-start wins (#346 step 4): the banked script passed the
    arithmetic gate on NEW data — exactly the "keeps passing on new data"
    evidence the graduation signal counts — but the run itself is a realtime
    frame and does not go through the write hook. One call per campaign.
    Never raises.
    """
    try:
        f = _domain_dir(domain, root=root) / f"{rid}.json"
        rec = json.loads(f.read_text())
        stats = rec.setdefault("stats", {})
        stats["n_successes"] = int(stats.get("n_successes", 1)) + 1
        if session:
            sessions = rec.setdefault("sessions", [])
            if session not in sessions:
                sessions.append(session)
                del sessions[:-_MAX_SESSIONS]
        rec["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        f.write_text(json.dumps(rec, indent=2, default=str))
    except Exception:
        pass


def render_exemplar_block(match: Dict[str, Any]) -> str:
    """LLM-facing prompt block offering a retrieved script as an exemplar.

    Shared framing across the agents: the exemplar is a starting point to
    adapt, dataset-specific values must be re-derived, and the locked plan
    stays authoritative.
    """
    rec = match["record"]
    outcome = rec.get("outcome") or {}
    signals = rec.get("technique_signals") or {}
    what = (signals.get("model_type") or signals.get("analysis_type")
            or signals.get("analysis_target") or "previous analysis")
    metric = outcome.get("best_metric") or outcome.get("metric")
    metric_txt = (f"{metric.get('name')} = {metric.get('value')}"
                  if isinstance(metric, dict) else "quality gate passed")
    n_succ = (rec.get("stats") or {}).get("n_successes", 1)
    return f"""## Reference: proven script from a previous successful analysis (script bank)
A previous run solved data closely matching this dataset (match score {match['score']}).
What it did: {str(what)[:300]}
Result: {metric_txt}; succeeded in {n_succ} session(s).

Use it as a STARTING POINT to adapt: keep its overall structure and vetted
approach where they fit THIS data, and change whatever does not. Dataset-specific
values (peak positions/counts, ranges, thresholds) must be re-derived from the
data at hand — do not copy them. The analysis plan above remains authoritative;
where the exemplar conflicts with the plan, follow the plan.

```python
{(rec.get("working_script") or "").strip()}
```
"""


def hyperspectral_fingerprint(cube: Any, axis: Any = None,
                              axis_units: Optional[str] = None,
                              n_bands: int = 16) -> Dict[str, Any]:
    """Fingerprint of a 3D datacube via its field-mean spectrum."""
    data = np.asarray(cube, dtype=float)
    fp: Dict[str, Any] = {"kind": "hyperspectral",
                          "shape": [int(s) for s in data.shape]}
    if data.ndim != 3 or data.size == 0:
        return fp
    e = data.shape[-1]
    if axis is None:
        axis = np.arange(e)
        axis_units = axis_units or "channels"
    axis = np.asarray(axis, dtype=float).ravel()[:e]
    fp["axis"] = {"units": axis_units,
                  "start": _r(axis[0]), "end": _r(axis[-1]), "n_channels": int(e)}

    mean_spec = np.nanmean(data.reshape(-1, e), axis=0)
    peak = float(np.nanmax(mean_spec))
    if peak > 0:
        edges = np.linspace(0, e, n_bands + 1).astype(int)
        fp["band_means"] = [
            _r(float(np.nanmean(mean_spec[a:b])) / peak, 3) if b > a else None
            for a, b in zip(edges[:-1], edges[1:])
        ]
    fp["snr"] = _snr_estimate(mean_spec)
    fp["peaks"] = _peak_summary(axis, mean_spec)
    return fp
