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

from ...utils.text_io import atomic_write_text
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


def verify_record(rec: Optional[Dict[str, Any]]) -> bool:
    """Whether a record's script still matches the hash it was banked under.

    The bank stores code that the verbatim path executes with no LLM review,
    so a record edited on disk — or one that never carried a hash — is not
    retrievable. Bookkeeping writes (stats, evidence) never touch the script,
    so they keep a record valid.
    """
    if not isinstance(rec, dict):
        return False
    stored = rec.get("script_hash")
    script = rec.get("working_script")
    return bool(stored) and bool((script or "").strip()) \
        and script_hash(script) == stored


#: Bumped whenever a fingerprint function changes what it computes. Records
#: keep the version they were fingerprinted under; similarity between
#: different versions is meaningless, so retrieval skips a mismatch rather
#: than compare apples with oranges. Records from before versioning are v1.
FINGERPRINT_VERSION = 1

_MAX_EVIDENCE = 50
ARCHIVE_DIRNAME = "_archive"


def _fp_version(fp: Optional[Dict[str, Any]]) -> int:
    try:
        return int((fp or {}).get("v") or 1)
    except (TypeError, ValueError):
        return 1


def data_key(fingerprint: Optional[Dict[str, Any]]) -> Optional[str]:
    """Short digest identifying WHICH data a success was earned on.

    Two runs on the same file produce the same key; a new measurement of even
    the same sample does not (noise moves the peak census). That is the
    distinction "independent evidence" needs.
    """
    if not fingerprint or not fingerprint.get("kind"):
        return None
    blob = json.dumps(fingerprint, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def _add_evidence(rec: Dict[str, Any], fingerprint: Optional[Dict[str, Any]],
                  session: Optional[str], *, adapted: bool = False) -> None:
    """Note one success's evidence key: the data digest when the data is
    known, else the session (one campaign = one piece of evidence).

    Two ledgers, because two different claims ride on them. ``evidence``
    counts every dataset the record helped solve — including through an
    edit-adaptation, which may have changed the model (observed live: a
    two-peak script that FAILED verbatim on three-peak data was adapted to
    three peaks, passed, and the original was credited). That is honest
    evidence that the record is a good starting point — the graduation
    signal. ``verbatim_evidence`` counts only datasets the script solved
    UNCHANGED, which is the only evidence that may license running it with
    no LLM review.
    """
    key = data_key(fingerprint) or (f"session:{session}" if session else None)
    if not key:
        return
    for ledger in (("evidence",) if adapted else ("evidence", "verbatim_evidence")):
        ev = rec.setdefault(ledger, [])
        if key not in ev:
            ev.append(key)
            del ev[:-_MAX_EVIDENCE]


def independent_successes(rec: Dict[str, Any], *, verbatim_only: bool = False) -> int:
    """How many INDEPENDENT successes back a record.

    ``stats.n_successes`` counts runs, so re-running one file three times
    used to make a record "proven". Evidence keys count distinct datasets
    instead; ``verbatim_only`` counts just those the script solved unchanged
    (see :func:`_add_evidence`). Records written before evidence tracking
    fall back to their distinct sessions — weaker, but never more than the
    runs they recorded.
    """
    ev = rec.get("verbatim_evidence" if verbatim_only else "evidence")
    if isinstance(ev, list) and ev:
        return len(set(ev))
    if verbatim_only and isinstance(rec.get("evidence"), list) and rec["evidence"]:
        return 1  # evidence-tracked record with no verbatim ledger: its own data
    n_succ = int((rec.get("stats") or {}).get("n_successes", 1) or 1)
    sessions = {s for s in (rec.get("sessions") or []) if s}
    return max(1, min(n_succ, len(sessions) or 1))


def is_proven(rec: Dict[str, Any]) -> bool:
    """Graduation signal: keeps helping solve new data (any route)."""
    return independent_successes(rec) >= proven_n()


def is_verbatim_proven(rec: Dict[str, Any]) -> bool:
    """Trust signal: keeps solving new data UNCHANGED — the bar for running
    a record with no LLM review where the arithmetic gate alone is weak."""
    return independent_successes(rec, verbatim_only=True) >= proven_n()


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
    action = "updated"
    if existing is None:
        # A script that was archived for disuse and has now succeeded again
        # earned its place back — restore it rather than bank a duplicate.
        archived = _find_by_hash(domain, h, root=root, archived=True)
        if archived is not None:
            rec, apath = archived
            path = _domain_dir(domain, root=root) / apath.name
            apath.replace(path)
            rec.pop("archived", None)
            existing, action = (rec, path), "restored"

    if existing is not None:
        rec, path = existing
        _add_evidence(rec, record.get("data_fingerprint"), session)
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
        atomic_write_text(path, json.dumps(rec, indent=2, default=str))
        return {"id": rec["id"], "action": action}

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
        "stats": {"n_successes": 1, "n_retrievals": 0, "n_failures": 0},
        **record,
    }
    _add_evidence(payload, record.get("data_fingerprint"), session)
    atomic_write_text(d / f"{rid}.json", json.dumps(payload, indent=2, default=str))
    # A growing bank is the moment to retire what nobody uses (at most one
    # sweep a day per domain; failure-isolated).
    auto_archive(domain, root=root)
    return {"id": rid, "action": "created"}


def _metric_value(metric: Any) -> Optional[float]:
    if isinstance(metric, dict):
        metric = metric.get("value")
    try:
        v = float(metric)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _find_by_hash(domain: str, h: str, *, root: Optional[Path] = None,
                  archived: bool = False):
    d = _domain_dir(domain, root=root)
    if archived:
        d = d / ARCHIVE_DIRNAME
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
    fp: Dict[str, Any] = {"kind": "curve", "v": FINGERPRINT_VERSION,
                          "n_points": int(n)}
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
    fp: Dict[str, Any] = {"kind": "image", "v": FINGERPRINT_VERSION,
                          "shape": [int(s) for s in img.shape]}
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


_IMAGE_TERMS_FOR_FULL_SCORE = 3


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
    if not sims:
        return 0.0
    # Averaging only the terms both sides carry made matching MORE lenient as
    # metadata went missing: one agreeing term (contrast alone) scored a
    # perfect 1.0. Agreement on fewer than three of the four terms is weak
    # evidence, so it is scaled down rather than trusted.
    return float(np.mean(sims)) * min(1.0, len(sims) / _IMAGE_TERMS_FOR_FULL_SCORE)


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


def _skills_of(rec: Dict[str, Any]) -> set:
    raw = (rec.get("technique_signals") or {}).get("active_skills") or []
    if isinstance(raw, str):
        raw = [raw]
    return {str(s).strip().lower() for s in raw if str(s).strip()}


def find_exemplar(domain: str, fingerprint: Optional[Dict[str, Any]],
                  context: Any = None, *, k: int = 1,
                  min_score: float = _MIN_EXEMPLAR_SCORE,
                  active_skills: Optional[List[str]] = None,
                  root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Rank bank records against a new dataset; return the top match(es).

    Hard filters: the record's script must still match its stored hash
    (:func:`verify_record`); fingerprint kind and version must match; axis
    units must agree when both sides declare them; and when both the run and
    the record name technique skills, they must share one — a script written
    under the Raman skill's mandatory rules is not a candidate for an FTIR
    run, whatever the axis looks like. Score = 0.8 × fingerprint similarity +
    0.2 × context token overlap + a small usage bonus (capped at 0.05) for
    scripts backed by INDEPENDENT successes, minus a reliability penalty
    (capped at 0.15) for scripts that were retrieved and then failed. Returns ``[{"record", "score",
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
    query_skills = {str(s).strip().lower() for s in (active_skills or [])
                    if str(s).strip()}
    query_version = _fp_version(fingerprint)

    scored = []
    for rec in list_records(domain, root=root):
        fp = rec.get("data_fingerprint") or {}
        if fp.get("kind") != kind or not (rec.get("working_script") or "").strip():
            continue
        if _fp_version(fp) != query_version or not verify_record(rec):
            continue
        rec_skills = _skills_of(rec)
        if query_skills and rec_skills and not (query_skills & rec_skills):
            continue
        rec_units = _units_of(fp)
        if query_units and rec_units and query_units != rec_units:
            continue
        s_fp = simfn(fingerprint, fp)
        s_ctx = _context_similarity(query_tokens, rec)
        n_ind = independent_successes(rec)
        usage = min(0.05, 0.01 * (n_ind - 1))
        # Reliability: a record retrieved and then failing (gate miss,
        # adaptation replaced by a refit) is an attractive nuisance. The
        # penalty grows with the failure count and with the failure SHARE,
        # so one miss on a well-proven script costs almost nothing.
        n_fail = int((rec.get("stats") or {}).get("n_failures", 0) or 0)
        penalty = (min(0.15, 0.05 * n_fail) * n_fail / (n_fail + n_ind)
                   if n_fail else 0.0)
        # Quality tie-breaker: a small term from the record's gate metric so
        # near-ties (same-system variants with equal fingerprints and usage)
        # resolve toward the better-quality script instead of record-id order.
        # Deliberately subordinate: one cross-session success (+0.01 usage)
        # outweighs any realistic metric gap (max term 0.02) — "proven
        # repeatedly" still beats "better once".
        outcome = rec.get("outcome") or {}
        mv = _metric_value(outcome.get("best_metric") or outcome.get("metric"))
        quality = 0.02 * min(max(mv, 0.0), 1.0) if mv is not None else 0.0
        score = 0.8 * s_fp + 0.2 * s_ctx + usage + quality - penalty
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
        rec["last_retrieved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        atomic_write_text(f, json.dumps(rec, indent=2, default=str))
    except Exception:
        pass


def record_success(domain: str, rid: str, session: Optional[str] = None,
                   *, fingerprint: Optional[Dict[str, Any]] = None,
                   adapted: bool = False,
                   root: Optional[Path] = None) -> None:
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
        # Evidence: the new data's digest when the caller has it, else one
        # key per session (so a campaign counts once).
        _add_evidence(rec, fingerprint, session, adapted=adapted)
        if session:
            sessions = rec.setdefault("sessions", [])
            if session not in sessions:
                sessions.append(session)
                del sessions[:-_MAX_SESSIONS]
        rec["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        atomic_write_text(f, json.dumps(rec, indent=2, default=str))
    except Exception:
        pass


def record_failure(domain: str, rid: str, reason: str,
                   session: Optional[str] = None,
                   *, root: Optional[Path] = None) -> None:
    """Note that a retrieved record did NOT deliver on new data.

    The counterpart of :func:`record_success`: a verbatim audition that
    missed the gate, or an edit-adaptation that a verification refit had to
    replace. Without it the bank only ever learned good news, and a record
    could be retrieved forever without paying for its misses. Feeds the
    reliability penalty in :func:`find_exemplar` and the ``never_succeeds``
    aging rule. Never raises.
    """
    try:
        f = _domain_dir(domain, root=root) / f"{rid}.json"
        rec = json.loads(f.read_text())
        stats = rec.setdefault("stats", {})
        stats["n_failures"] = int(stats.get("n_failures", 0) or 0) + 1
        rec["last_failure"] = {
            "reason": str(reason)[:120], "session": session,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        atomic_write_text(f, json.dumps(rec, indent=2, default=str))
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────
# Aging — retire what nobody uses
#
# Every approved analysis banks a script, so a bank only grows, and most of
# what it holds is never retrieved again: the listing (CLI, UI) fills with
# one-off scripts and the records worth looking at get buried. Aging moves
# such records to ``<domain>/_archive/`` — out of listings and retrieval, but
# intact on disk. An archive is reversible (``restore_records``; re-banking
# the same script restores it automatically), which is why it may run
# unattended where a delete may not.
#
# A record is stale when it is
#   never_used      never retrieved, succeeded only on its original data, and
#                   older than the idle window;
#   never_succeeds  retrieved and failed at least ``_STALE_FAILURES`` times
#                   with no independent success — age is irrelevant, it is an
#                   attractive nuisance today;
#   superseded      an unproven member of a variant group (same system) that
#                   contains a proven member, idle past the window.
# Proven records and records awaiting review in staging are never stale.
# ──────────────────────────────────────────────────────────────

_DEFAULT_STALE_DAYS = 60
_STALE_FAILURES = 3
_SWEEP_STAMP = ".last_sweep"


def stale_days() -> int:
    raw = os.environ.get("SCILINK_BANK_STALE_DAYS", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return _DEFAULT_STALE_DAYS


def _age_days(stamp: Any, now: datetime) -> Optional[float]:
    try:
        then = datetime.fromisoformat(str(stamp))
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        return (now - then).total_seconds() / 86400.0
    except (TypeError, ValueError):
        return None


def stale_records(domain: Optional[str] = None, *,
                  idle_days: Optional[int] = None,
                  now: Optional[datetime] = None,
                  root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Records the aging rules would archive: ``[{domain, id, reason, label,
    idle_days}]``. Pure — nothing is moved."""
    now = now or datetime.now(timezone.utc)
    window = idle_days if idle_days is not None else stale_days()
    recs = list_records(domain, root=root)

    # Variant groups holding a proven member, per domain.
    superseded_by: Dict[tuple, str] = {}
    by_id = {(r.get("domain"), r.get("id")): r for r in recs}
    for dom in sorted({r.get("domain") for r in recs if r.get("domain")}):
        try:
            groups = find_variant_groups(dom, root=root)
        except Exception:
            groups = []
        for g in groups:
            members = [by_id.get((dom, i)) for i in g.get("ids") or []]
            proven = [m for m in members if m and is_proven(m)]
            if not proven:
                continue
            for m in members:
                if m and not is_proven(m):
                    superseded_by[(dom, m["id"])] = proven[0]["id"]

    out: List[Dict[str, Any]] = []
    for rec in recs:
        if is_proven(rec) or rec.get("promoted_to_staging"):
            continue
        stats = rec.get("stats") or {}
        n_ret = int(stats.get("n_retrievals", 0) or 0)
        n_fail = int(stats.get("n_failures", 0) or 0)
        n_ind = independent_successes(rec)
        idle = _age_days(rec.get("last_retrieved_at") or rec.get("updated_at")
                         or rec.get("created_at"), now)
        reason = None
        if n_fail >= _STALE_FAILURES and n_ind <= 1:
            reason = "never_succeeds"
        elif idle is not None and idle > window:
            key = (rec.get("domain"), rec.get("id"))
            if n_ret == 0 and n_ind <= 1:
                reason = "never_used"
            elif key in superseded_by:
                reason = "superseded"
        if reason:
            row = {"domain": rec.get("domain"), "id": rec.get("id"),
                   "reason": reason, "label": record_label(rec)[:100],
                   "idle_days": round(idle, 1) if idle is not None else None}
            if reason == "superseded":
                row["superseded_by"] = superseded_by[(rec.get("domain"), rec.get("id"))]
            out.append(row)
    return out


def archive_records(domain: str, ids: List[str], *, reason: Optional[str] = None,
                    root: Optional[Path] = None) -> int:
    """Move records into ``<domain>/_archive/``; return the count moved."""
    d = _domain_dir(domain, root=root)
    n = 0
    for rid in ids:
        f = d / f"{safe_path_component(rid, fallback='x')}.json"
        if not f.exists():
            continue
        try:
            rec = json.loads(f.read_text())
            rec["archived"] = {
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "reason": reason}
            (d / ARCHIVE_DIRNAME).mkdir(parents=True, exist_ok=True)
            atomic_write_text(d / ARCHIVE_DIRNAME / f.name,
                              json.dumps(rec, indent=2, default=str))
            f.unlink()
            n += 1
        except Exception:
            continue
    return n


def list_archived(domain: Optional[str] = None, *,
                  root: Optional[Path] = None) -> List[Dict[str, Any]]:
    base = root or bank_dir()
    if not base.is_dir():
        return []
    domains = [base / domain] if domain else [
        p for p in sorted(base.iterdir())
        if p.is_dir() and not p.name.startswith((".", "_"))]
    out: List[Dict[str, Any]] = []
    for dd in domains:
        for f in sorted((dd / ARCHIVE_DIRNAME).glob("*.json")):
            try:
                rec = json.loads(f.read_text())
            except Exception:
                continue
            rec.setdefault("domain", dd.name)
            out.append(rec)
    return out


def restore_records(domain: str, ids: List[str], *,
                    root: Optional[Path] = None) -> int:
    """Bring archived records back into the live bank."""
    d = _domain_dir(domain, root=root)
    n = 0
    for rid in ids:
        f = d / ARCHIVE_DIRNAME / f"{safe_path_component(rid, fallback='x')}.json"
        if not f.exists():
            continue
        try:
            rec = json.loads(f.read_text())
            rec.pop("archived", None)
            # A restore is a vote of confidence: restart the idle clock so
            # the next sweep does not take it straight back.
            rec["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            atomic_write_text(d / f.name, json.dumps(rec, indent=2, default=str))
            f.unlink()
            n += 1
        except Exception:
            continue
    return n


def sweep(domain: Optional[str] = None, *, idle_days: Optional[int] = None,
          dry_run: bool = False,
          root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Archive every stale record (or just report them with ``dry_run``)."""
    stale = stale_records(domain, idle_days=idle_days, root=root)
    if not dry_run:
        for row in stale:
            archive_records(row["domain"], [row["id"]], reason=row["reason"],
                            root=root)
    return stale


def auto_archive(domain: str, *, root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """The unattended sweep: at most once a day per domain, on a bank write.

    ``SCILINK_BANK_AUTO_ARCHIVE=0`` turns it off (the manual ``bank-sweep``
    still works). Never raises.
    """
    try:
        if (os.environ.get("SCILINK_BANK_AUTO_ARCHIVE", "").strip().lower()
                in _FALSY):
            return []
        d = _domain_dir(domain, root=root)
        stamp = d / ARCHIVE_DIRNAME / _SWEEP_STAMP
        now = datetime.now(timezone.utc)
        if stamp.exists():
            age = _age_days(stamp.read_text().strip(), now)
            if age is not None and age < 1.0:
                return []
        stamp.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(stamp, now.isoformat(timespec="seconds"))
        return sweep(domain, root=root)
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────
# Assist log — did the bank actually help?
#
# Retrieval thresholds were hand-set from a single corpus. Whether a banked
# script shortens a run is an empirical question, so every QC-loop item
# appends one event here: which mode served it (none / exemplar / edit_adapt /
# verbatim), the record and its score, how many verification iterations the
# item then needed, and whether it was approved. ``mode == "none"`` events are
# the baseline the assisted ones are compared against.
#
# One append-only JSONL file for the whole bank (single short writes, so
# concurrent runs interleave whole lines). Bookkeeping only — never raises.
# ──────────────────────────────────────────────────────────────

ASSIST_LOG_NAME = "assist_log.jsonl"
ASSIST_MODES = ("none", "exemplar", "edit_adapt", "verbatim")
_SCORE_BUCKETS = ((0.0, 0.45), (0.45, 0.55), (0.55, 0.70), (0.70, 1.01))


def assist_log_path(*, root: Optional[Path] = None) -> Path:
    return (root or bank_dir()) / ASSIST_LOG_NAME


def log_assist(event: Dict[str, Any], *, root: Optional[Path] = None) -> None:
    """Append one assist event. No-op when the bank is disabled."""
    try:
        if not bank_enabled():
            return
        path = assist_log_path(root=root)
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               **event}
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        pass


def read_assist_log(domain: Optional[str] = None, *,
                    root: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = assist_log_path(root=root)
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue  # a torn line from a killed run
        if domain is None or ev.get("domain") == domain:
            events.append(ev)
    return events


def _assist_rollup(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(events)
    iters = [e["iterations"] for e in events
             if isinstance(e.get("iterations"), (int, float))]
    secs = [e["seconds"] for e in events
            if isinstance(e.get("seconds"), (int, float))]
    return {
        "n": n,
        "approved_rate": (round(sum(1 for e in events if e.get("approved")) / n, 3)
                          if n else None),
        "mean_iterations": round(float(np.mean(iters)), 2) if iters else None,
        "mean_seconds": round(float(np.mean(secs)), 1) if secs else None,
    }


def assist_stats(domain: Optional[str] = None, *,
                 root: Optional[Path] = None) -> Dict[str, Any]:
    """Summarise the assist log per domain → mode, and per score bucket.

    ``survived_rate`` (edit_adapt only) is the fraction of adaptations whose
    script was still the accepted one at the end — the rest were replaced by
    a verification-loop refit, i.e. the adaptation cost a call and bought
    nothing.
    """
    out: Dict[str, Any] = {}
    events = read_assist_log(domain, root=root)
    for dom in sorted({e.get("domain") or "unknown" for e in events}):
        evs = [e for e in events if (e.get("domain") or "unknown") == dom]
        modes: Dict[str, Any] = {}
        for mode in ASSIST_MODES:
            sel = [e for e in evs if e.get("mode") == mode]
            if not sel:
                continue
            roll = _assist_rollup(sel)
            if mode == "edit_adapt":
                roll["survived_rate"] = round(
                    sum(1 for e in sel if e.get("survived")) / len(sel), 3)
            modes[mode] = roll
        buckets: Dict[str, Any] = {}
        assisted = [e for e in evs if e.get("mode") not in (None, "none")
                    and isinstance(e.get("score"), (int, float))]
        for lo, hi in _SCORE_BUCKETS:
            sel = [e for e in assisted if lo <= e["score"] < hi]
            if sel:
                buckets[f"{lo:.2f}-{min(hi, 1.0):.2f}"] = _assist_rollup(sel)
        out[dom] = {"n_events": len(evs), "by_mode": modes,
                    "by_score": buckets}
    return out


# ──────────────────────────────────────────────────────────────
# Management surface + graduation signal
#
# The bank is episodic memory; skill graduation is semantic memory. The
# bridge: records that keep succeeding across sessions ("proven") are the
# evidence-based graduation candidates — promotion copies one into the
# distill-staging buffer, where the EXISTING review-gated upgrade /
# consolidate ceremony applies. The bank record itself is kept (episodic
# history is not consumed by distillation).
# ──────────────────────────────────────────────────────────────

#: Cross-session successes at which a record counts as "proven" — the
#: evidence-based replacement for "climbed to hot once" as a graduation
#: signal. Overridable via $SCILINK_BANK_PROVEN_N.
_DEFAULT_PROVEN_N = 3


def proven_n() -> int:
    raw = os.environ.get("SCILINK_BANK_PROVEN_N", "").strip()
    if raw:
        try:
            return max(2, int(raw))
        except ValueError:
            pass
    return _DEFAULT_PROVEN_N


def record_label(rec: Dict[str, Any]) -> str:
    """One-line 'what this script does' for lists (CLI / UI)."""
    signals = rec.get("technique_signals") or {}
    outcome = rec.get("outcome") or {}
    what = (signals.get("model_type") or signals.get("analysis_type")
            or signals.get("analysis_target") or outcome.get("model_type")
            or outcome.get("analysis_type") or "(unlabeled script)")
    return str(what)


def bank_summary(domain: Optional[str] = None, *,
                 root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Compact display rows for the management surfaces, proven-first.

    ``promoted_to_staging`` is reported only while the staged copy still
    exists (awaiting review) — once it was consumed by upgrade/consolidate
    or pruned, the record shows as promotable again, matching
    :func:`promote_to_staging`'s refusal rule.
    """
    from . import _staging

    threshold = proven_n()
    rows = []
    for rec in list_records(domain, root=root):
        stats = rec.get("stats") or {}
        metric = (rec.get("outcome") or {}).get("best_metric") \
            or (rec.get("outcome") or {}).get("metric")
        sid = rec.get("promoted_to_staging")
        if sid and _staging.get_staged(rec.get("domain") or "", sid) is None:
            sid = None  # dangling — the staged copy was consumed or pruned
        rows.append({
            "domain": rec.get("domain"),
            "id": rec.get("id"),
            "label": record_label(rec)[:120],
            "n_successes": int(stats.get("n_successes", 1) or 1),
            "n_independent": independent_successes(rec),
            "n_verbatim": independent_successes(rec, verbatim_only=True),
            "n_failures": int(stats.get("n_failures", 0) or 0),
            "n_retrievals": int(stats.get("n_retrievals", 0) or 0),
            "sessions": rec.get("sessions") or [],
            "metric": metric,
            "created_at": rec.get("created_at"),
            "proven": independent_successes(rec) >= threshold,
            "promoted_to_staging": sid,
        })
    rows.sort(key=lambda r: (-r["n_independent"], -r["n_successes"], -r["n_retrievals"],
                             -(_metric_value(r["metric"]) or 0.0), r["id"] or ""))
    return rows


def _derive_technique_label(rec: Dict[str, Any]) -> str:
    import re
    label = re.sub(r"[^a-z0-9]+", "_", record_label(rec).lower()).strip("_")
    if len(label) > 48:  # cut at a word boundary, not mid-word
        label = label[:48].rsplit("_", 1)[0]
    return label or "uncategorized"


def find_by_script(domain: str, script: str, *,
                   root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """The bank record holding this exact script (whitespace-insensitive)."""
    hit = _find_by_hash(domain, script_hash(script or ""), root=root)
    return hit[0] if hit else None


def promote_to_staging(domain: str, rid: str, technique: Optional[str] = None,
                       *, root: Optional[Path] = None,
                       staging_root: Optional[Path] = None,
                       provenance: str = "bank_proven",
                       extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Copy a bank record into the distill-staging buffer (graduation path).

    The staged record enters the existing review-gated ceremony (upgrade an
    existing skill, or consolidate N of a technique into a new one) with
    provenance ``bank_proven`` and the cross-session evidence attached. The
    bank record is kept and marked ``promoted_to_staging`` so surfaces show
    it and repeat promotions are flagged. ``technique`` defaults to a
    deterministic label derived from the record (no LLM).
    """
    from . import _staging

    rec = get_record(domain, rid, root=root)
    if rec is None:
        return {"status": "error", "message": f"No bank record {domain}/{rid}."}
    prior_sid = rec.get("promoted_to_staging")
    if prior_sid:
        # Refuse only while the staged copy is still awaiting review. Once it
        # was consumed (upgrade/consolidate remove staged records) or pruned,
        # the mark is dangling and re-promotion is legitimate.
        if _staging.get_staged(domain, prior_sid, root=staging_root) is not None:
            return {"status": "error",
                    "message": (f"Record {rid} is already staged for review "
                                f"(staged id {prior_sid}). Review it with "
                                f"`scilink memory staged`.")}
    script = (rec.get("working_script") or "").strip()
    if not script:
        return {"status": "error", "message": f"Record {rid} has no script."}

    stats = rec.get("stats") or {}
    outcome = rec.get("outcome") or {}
    metric = outcome.get("best_metric") or outcome.get("metric")
    # The default provenance must not overstate the evidence: "proven"
    # is reserved for records that actually earned the star (>= proven_n
    # cross-session successes); an ordinary nomination is just that.
    if provenance == "bank_proven" and not is_proven(rec):
        provenance = "bank_nominated"
    staged_record: Dict[str, Any] = {
        "provenance": provenance,
        "model": record_label(rec),
        "deviation_from_plan": (
            f"{'Proven in' if provenance == 'bank_proven' else 'Nominated from'} "
            f"the script bank: succeeded in "
            f"{stats.get('n_successes', 1)} session(s), retrieved "
            f"{stats.get('n_retrievals', 0)} time(s) "
            f"(sessions: {', '.join(rec.get('sessions') or [])[:200]})."
        ),
        "plan_summary": outcome.get("plan_summary"),
        "measurement_context": rec.get("measurement_context"),
        "working_script": script,
        "session": (rec.get("sessions") or ["?"])[-1],
        "bank_id": rid,
    }
    # Nomination-specific fields (e.g. a T=2 hot win's contrastive
    # planned-vs-final story) ride along into the staged record.
    for k, v in (extra or {}).items():
        if v is not None:
            staged_record[k] = v
    if isinstance(metric, dict) and metric.get("name") == "r_squared":
        staged_record["r_squared"] = metric.get("value")
    elif isinstance(metric, dict) and metric.get("value") is not None:
        staged_record["quality_score"] = metric.get("value")

    label = technique or _derive_technique_label(rec)
    sid = _staging.stage_solution(domain, label, staged_record,
                                  root=staging_root)

    rec["promoted_to_staging"] = sid
    rec["promoted_reason"] = provenance
    rec["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    atomic_write_text(_domain_dir(domain, root=root) / f"{rid}.json",
                      json.dumps(rec, indent=2, default=str))
    return {"status": "success", "staged_id": sid, "technique": label,
            "domain": domain, "bank_id": rid}


#: Minimum pairwise fingerprint similarity for two records to count as
#: variants of the same system. Calibrated on real data: same-system pairs
#: score ~1.0, a phase transition drops to ~0.79 (in-situ XRD), transition
#: onset ~0.90 — 0.85 groups true variants without merging changed systems.
VARIANT_GROUP_THRESHOLD = 0.85


def find_variant_groups(domain: str, *, min_size: int = 2,
                        threshold: float = VARIANT_GROUP_THRESHOLD,
                        root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Cluster a domain's records into same-system variant groups.

    Single-linkage over pairwise fingerprint similarity (same kind only).
    These groups are the natural units for skill consolidation: N different
    successful treatments of one system are exactly what the distillation
    LLM needs to generalize from. Returns, per group of >= ``min_size``:
    ``ids`` (proven-first), ``min_similarity`` (worst pair inside the
    group), ``suggested_technique`` (derived from the best member), and
    ``n_unpromoted`` (members whose staged copy does not currently exist).
    """
    recs = [r for r in list_records(domain, root=root)
            if (r.get("data_fingerprint") or {}).get("kind")
            and (r.get("working_script") or "").strip()]
    if len(recs) < min_size:
        return []
    sims: Dict[tuple, float] = {}
    parent = list(range(len(recs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(recs)):
        fp_i = recs[i]["data_fingerprint"]
        simfn = _SIMILARITY_FNS.get(fp_i.get("kind"))
        if simfn is None:
            continue
        for j in range(i + 1, len(recs)):
            fp_j = recs[j]["data_fingerprint"]
            if fp_j.get("kind") != fp_i.get("kind"):
                continue
            s = simfn(fp_i, fp_j)
            sims[(i, j)] = s
            if s >= threshold:
                parent[find(i)] = find(j)

    clusters: Dict[int, list] = {}
    for i in range(len(recs)):
        clusters.setdefault(find(i), []).append(i)

    summary_by_id = {r["id"]: r for r in bank_summary(domain, root=root)}
    groups = []
    for members in clusters.values():
        if len(members) < min_size:
            continue
        rows = sorted(
            (summary_by_id[recs[i]["id"]] for i in members
             if recs[i]["id"] in summary_by_id),
            key=lambda r: (-r["n_successes"],
                           -(_metric_value(r["metric"]) or 0.0), r["id"] or ""))
        if len(rows) < min_size:
            continue
        pair_sims = [sims.get((min(i, j), max(i, j)), 1.0)
                     for a, i in enumerate(members) for j in members[a + 1:]]
        best = get_record(domain, rows[0]["id"], root=root) or {}
        groups.append({
            "domain": domain,
            "ids": [r["id"] for r in rows],
            "records": rows,
            "min_similarity": round(min(pair_sims), 3) if pair_sims else 1.0,
            "suggested_technique": _derive_technique_label(best),
            "n_unpromoted": sum(1 for r in rows if not r["promoted_to_staging"]),
        })
    groups.sort(key=lambda g: (-len(g["ids"]), g["ids"][0]))
    return groups


def promote_group_to_staging(domain: str, ids: List[str],
                             technique: Optional[str] = None, *,
                             root: Optional[Path] = None,
                             staging_root: Optional[Path] = None) -> Dict[str, Any]:
    """Promote several same-system records under ONE shared technique label.

    The shared label is what lets ``consolidate`` receive the variants
    together — per-record derived labels would scatter them across
    techniques and they would never be reviewed jointly. Records whose
    staged copy already exists are skipped (not an error). Label defaults
    to the best member's derived label.
    """
    from . import _staging

    recs = [(rid, get_record(domain, rid, root=root)) for rid in ids]
    missing = [rid for rid, rec in recs if rec is None]
    if missing:
        return {"status": "error",
                "message": f"No bank record(s): {', '.join(missing)}."}
    if technique is None:
        best = max((rec for _, rec in recs),
                   key=lambda r: ((r.get("stats") or {}).get("n_successes", 1),
                                  _metric_value((r.get("outcome") or {}).get("best_metric")
                                                or (r.get("outcome") or {}).get("metric")) or 0.0))
        technique = _derive_technique_label(best)

    staged_ids, skipped = [], []
    for rid, _rec in recs:
        out = promote_to_staging(domain, rid, technique=technique,
                                 root=root, staging_root=staging_root)
        if out.get("status") == "success":
            staged_ids.append(out["staged_id"])
        else:
            skipped.append({"id": rid, "reason": out.get("message")})
    if not staged_ids and skipped:
        return {"status": "error", "technique": technique, "skipped": skipped,
                "message": "Nothing promoted: " + skipped[0]["reason"]}

    n_staged_total = len(_staging.group_by_technique(
        domain, root=staging_root).get(technique, []))
    return {"status": "success", "technique": technique,
            "staged_ids": staged_ids, "skipped": skipped,
            "n_staged_total": n_staged_total,
            "ready_to_consolidate": n_staged_total >= _staging.consolidate_min_n()}


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
    fp: Dict[str, Any] = {"kind": "hyperspectral", "v": FINGERPRINT_VERSION,
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
