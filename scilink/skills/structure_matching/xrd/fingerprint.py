"""Fingerprint search-match — whole-library powder identification
(``search_match_pattern`` tool + ``build_fingerprint_library`` builder).

This is the classical Hanawalt-style workflow for identifying an unknown whose
phase is probably KNOWN to science (the overwhelming majority of lab powder
work): match the measured d-spacings + relative intensities against a
PRECOMPUTED library of reference patterns, searched all at once — nothing is
simulated at query time, and intensities participate in the search. The same
route commercial (ICDD PDF / Jade / HighScore) and open (Match!/QualX over COD,
CrystalSleuth over RRUFF) tools take. It complements — and for known phases
replaces — the indexing route (``index_pattern``), which is the discovery path
for genuinely new phases.

The library is built ONCE from a directory of CIFs (e.g. a COD mirror or any
local collection) by ``build_fingerprint_library``: each structure's kinematic
pattern is computed (pymatgen), reduced to its top-K lines as wavelength-free
d-spacings + relative intensities, curated (parse failures skipped, duplicates
by formula+cell collapsed), and stored as parquet. ``search_match_pattern``
then keys on the strongest measured lines (Hanawalt-style, vectorized over the
whole library) and scores the shortlist with the existing Hanawalt
figure-of-merit scorer. Deterministic and offline — no LLM, no network.

Set ``SCILINK_XRD_FINGERPRINT_DB`` to the parquet path (or pass ``library_path``)
to use a prebuilt library.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional, Sequence

import numpy as np

from ..._shared._spec import ToolSpec

try:
    from pymatgen.core import Structure
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    Structure = None  # type: ignore
    XRDCalculator = None  # type: ignore

_logger = logging.getLogger(__name__)

_ENV_DB = "SCILINK_XRD_FINGERPRINT_DB"
_TOP_K = 40          # strongest lines stored per entry
_LIB_CACHE: dict[str, Any] = {}


# --- library construction ------------------------------------------------------

def build_fingerprint_library(
    cif_dir: str,
    out_path: str,
    two_theta_max: float = 90.0,
    top_k: int = _TOP_K,
    min_lines: int = 3,
    max_sites: int = 500,
) -> dict[str, Any]:
    """Build a fingerprint library (parquet) from a directory tree of CIFs.

    One-time batch operation (a COD mirror at ~500k CIFs takes CPU-days —
    parallelize by sharding the directory and concatenating the parquets; a
    domain slice of a few thousand CIFs takes minutes). Wavelength-free:
    patterns are stored as d-spacings, so one library serves any source.

    Curation applied: unparsable CIFs skipped (counted), duplicates collapsed
    by (reduced formula, rounded cell) keeping the first seen, entries with
    fewer than ``min_lines`` lines skipped. Partial occupancies are kept (they
    are physical) but flagged ``disordered``. Entries with more than
    ``max_sites`` sites are skipped (counted as ``n_skipped_large``): kinematic
    pattern cost scales ~ sites x reflections — single macromolecular entries
    can take MINUTES (observed: one COD chunk burned 11+ CPU-hours) — and a
    thousand overlapping weak lines make a useless powder fingerprint. Typical
    phase-ID targets are well under 200 sites; RAISE max_sites only if you
    genuinely need very large-cell organics and can pay the compute.

    Returns a summary dict: {'n_indexed', 'n_skipped', 'n_duplicates', 'path'}.
    """
    import pandas as pd

    if not PYMATGEN_AVAILABLE:
        raise RuntimeError("build_fingerprint_library requires pymatgen "
                           "(install scilink[structure-matching]).")
    calc = XRDCalculator(wavelength="CuKa")  # internal grid only; stored as d
    rows = []
    seen: set = set()
    n_skip = n_dup = n_large = 0
    for root, _dirs, files in os.walk(cif_dir):
        for fn in sorted(files):
            if not fn.lower().endswith(".cif"):
                continue
            path = os.path.join(root, fn)
            try:
                s = Structure.from_file(path)
            except Exception:
                n_skip += 1
                continue
            if len(s) > int(max_sites):
                n_large += 1
                continue
            try:
                pat = calc.get_pattern(s, two_theta_range=(5.0, float(two_theta_max)))
            except Exception:
                n_skip += 1
                continue
            if len(pat.x) < int(min_lines):
                n_skip += 1
                continue
            lat = s.lattice
            key = (s.composition.reduced_formula,
                   tuple(round(v, 2) for v in sorted([lat.a, lat.b, lat.c])),
                   round(lat.volume, 1))
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            order = np.argsort(np.asarray(pat.y))[::-1][:int(top_k)]
            ds = np.asarray(pat.d_hkls, dtype=float)[order]
            ys = np.asarray(pat.y, dtype=float)[order]
            ys = 100.0 * ys / max(float(ys.max()), 1e-9)
            try:
                sg_symbol, sg_number = s.get_space_group_info()
            except Exception:
                sg_symbol, sg_number = None, None
            occs = [sp for site in s for sp in [site.species.num_atoms]]
            disordered = any(abs(o - round(o)) > 1e-3 for o in occs)
            rows.append({
                "source_id": os.path.splitext(fn)[0],
                "cif_path": os.path.abspath(path),
                "formula": s.composition.reduced_formula,
                "space_group": sg_symbol,
                "sg_number": sg_number,
                "a": lat.a, "b": lat.b, "c": lat.c,
                "alpha": lat.alpha, "beta": lat.beta, "gamma": lat.gamma,
                "volume": lat.volume,
                "n_sites": len(s),
                "d1": float(ds[0]), "d2": float(ds[1] if len(ds) > 1 else 0.0),
                "d3": float(ds[2] if len(ds) > 2 else 0.0),
                "ds": [float(v) for v in ds],
                "intensities": [float(v) for v in ys],
                "flags": json.dumps({"disordered": disordered}),
            })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    df.to_parquet(out_path, index=False)
    _LIB_CACHE.pop(os.path.abspath(out_path), None)
    return {"n_indexed": len(rows), "n_skipped": n_skip,
            "n_skipped_large": n_large, "n_duplicates": n_dup, "path": out_path}


# --- library distribution / loading --------------------------------------------
#
# Resolution order: explicit ``library_path`` -> $SCILINK_XRD_FINGERPRINT_DB ->
# the persistent per-user store (~/.scilink/xrd_fingerprints/, survives pip
# upgrades — same convention as the persistent skill store). The prebuilt
# COD-derived artifact is fetched ONCE into that store (a few hundred MB, like
# pulling model weights); nobody rebuilds 100 GB of CIFs to use this.

DEFAULT_LIBRARY_URL = (
    "https://github.com/ziatdinovmax/SciLink/releases/download/"
    "xrd-fplib-v1/cod_fingerprints.parquet"
)
DEFAULT_LIBRARY_SHA256: Optional[str] = (
    "e2d1a8424221daad88d3c99b92fe0584cb93f11ef927ea6bdcf6c1bf31365b48"
)  # xrd-fplib-v1: 440,378 entries, COD snapshot 2026-02-05


def _default_store_path() -> str:
    return os.path.join(os.path.expanduser("~"), ".scilink",
                        "xrd_fingerprints", "cod_fingerprints.parquet")


def fetch_fingerprint_library(
    url: Optional[str] = None,
    dest: Optional[str] = None,
    sha256: Optional[str] = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Download a prebuilt fingerprint library into the per-user store.

    One-time, explicit operation (several hundred MB — deliberately NOT run
    implicitly on first use). Streams to a temp file, verifies the checksum
    when one is supplied, then atomically renames into place.

    Returns {'path', 'n_entries', 'sha256', 'source_url'}.
    """
    import hashlib
    import tempfile
    import urllib.request

    url = url or DEFAULT_LIBRARY_URL
    sha256 = sha256 if sha256 is not None else DEFAULT_LIBRARY_SHA256
    dest = os.path.abspath(dest or _default_store_path())
    if os.path.exists(dest) and not overwrite:
        raise FileExistsError(
            f"{dest} already exists — pass overwrite=True (or delete it) to "
            "replace the installed library."
        )
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    _logger.info("Downloading fingerprint library from %s", url)
    digest = hashlib.sha256()
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest), suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            got = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                got += len(chunk)
                if total and got % (50 << 20) < (1 << 20):
                    _logger.info("  %.0f%% (%d MB)", 100 * got / total, got >> 20)
        actual = digest.hexdigest()
        if sha256 and actual.lower() != sha256.lower():
            raise RuntimeError(
                f"Checksum mismatch for {url}: expected {sha256}, got {actual} "
                "— refusing to install a corrupted/tampered library."
            )
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    import pandas as pd
    n = len(pd.read_parquet(dest, columns=["source_id"]))
    _LIB_CACHE.pop(dest, None)
    return {"path": dest, "n_entries": int(n), "sha256": actual, "source_url": url}


def _load_library(library_path: Optional[str]):
    import pandas as pd
    path = library_path or os.environ.get(_ENV_DB)
    if not path:
        store = _default_store_path()
        if os.path.exists(store):
            path = store
    if not path:
        raise RuntimeError(
            "No fingerprint library found. Three ways to get one:\n"
            f"  1. Fetch the prebuilt COD library (recommended, ~hundreds of MB, "
            f"one-time): run `scilink fetch-xrd-library` or call "
            f"fetch_fingerprint_library() — installs to {_default_store_path()}\n"
            f"  2. Point {_ENV_DB} at an existing parquet (shared/HPC installs)\n"
            "  3. Build your own from any CIF collection: "
            "scripts/build_cod_fingerprints.py (COD mirror) or "
            "build_fingerprint_library(cif_dir, out_path) (local/private CIFs)"
        )
    path = os.path.abspath(path)
    if path not in _LIB_CACHE:
        _LIB_CACHE[path] = pd.read_parquet(path)
    return _LIB_CACHE[path]


# --- search-match ---------------------------------------------------------------

TOOL_SPEC = ToolSpec(
    name="search_match_pattern",
    description=(
        "Identify an unknown powder pattern by fingerprint SEARCH-MATCH against "
        "a precomputed reference library (Hanawalt-style: the strongest measured "
        "d-lines key a whole-library search, intensities included; the shortlist "
        "is scored with the Hanawalt figure of merit). THE first-choice route "
        "for a blind unknown whose phase is probably known to science — "
        "deterministic, offline, no chemistry needed, no simulation at query "
        "time. Use index_pattern/validate_cell_lebail only when this returns no "
        "convincing match (possible new phase). Requires a prebuilt library "
        "(build_fingerprint_library over a CIF collection, e.g. a COD mirror)."
    ),
    import_line="from scilink.skills.structure_matching.xrd.fingerprint import search_match_pattern",
    signature=(
        "search_match_pattern(two_theta, intensity, wavelength='CuKa', "
        "library_path=None, top_n=10, tol_deg=0.3, n_key_lines=3, "
        "n_query_lines=8, max_shortlist=500) -> dict"
    ),
    parameters={
        "two_theta": {"type": "list[float]", "description": "Measured peak positions (2θ°) — extract_peaks' 'positions' (NOT the raw profile)."},
        "intensity": {"type": "list[float]", "description": "Peak intensities aligned with two_theta — extract_peaks' 'intensities'."},
        "wavelength": {"type": "str | float", "description": "Source ('CuKa','MoKa',…) or Å; converts measured 2θ to wavelength-free d-spacings for the search."},
        "library_path": {"type": "str", "description": "Path to the fingerprint parquet (default: $SCILINK_XRD_FINGERPRINT_DB)."},
        "top_n": {"type": "int", "description": "Ranked matches returned (default 10)."},
        "tol_deg": {"type": "float", "description": "Position tolerance in 2θ degrees for both keying and scoring (default 0.3). RAISE for shifted/offset patterns (no internal standard); LOWER for well-calibrated data."},
        "n_key_lines": {"type": "int", "description": "How many of a candidate's strongest lines must be present among the query's strongest lines (default 3, classical Hanawalt). LOWER to 2 for textured/preferred-orientation samples where relative intensities are distorted."},
        "n_query_lines": {"type": "int", "description": "How many of the query's strongest measured lines the key lines are matched against (default 8). RAISE for multi-phase mixtures (each phase's key lines hide among more measured lines)."},
        "max_shortlist": {"type": "int", "description": "Cap on keyed candidates that get full figure-of-merit scoring (default 500)."},
        "materialize_top": {"type": "int", "description": "How many top matches get their full CIF resolved into 'structure_path' (default 3; 0 = identification only). The library stores peak lists, not structures — this is the bridge to simulation confirmation and Rietveld. Fetched by COD ID (cached) unless the library was built locally."},
        "materialize_dir": {"type": "str", "description": "Directory the fetched CIFs are written to (default './fp_matches')."},
        "strong_line_frac": {"type": "float", "description": "A candidate line counts as STRONG (and therefore testable for absence) when its relative intensity is ≥ this fraction of the candidate's max (default 0.25). LOWER to test more lines (stricter impostor rejection, but weak lines may be missing from a capped measured peak list); RAISE to test only the dominant lines."},
        "absent_lines_penalty": {"type": "float", "description": "Ranking penalty weight for a candidate whose strong predicted lines are ABSENT from the measurement: adjusted_score = figure_of_merit × (1 − penalty × frac_absent). DEFAULT 0 (evidence is reported, ranking unchanged): validated on real data the absent-fraction cleanly kills some impostors (FeAgS2 masquerading as fluorite: 0.86 absent vs truth 0.00) but INVERTS for phases with polytypes/polymorphs whose extra weak lines are legitimately missing (ZnS wurtzite-vs-sphalerite entries) — so a blind default is unsafe. Instead READ 'frac_strong_lines_absent' per match: among near-tied FOMs, a candidate with substantially MORE absent strong lines than its rivals is the impostor. RAISE the penalty (e.g. 0.5) only when the sample is untextured and single-polymorph candidates are expected."},
    },
    required=["two_theta", "intensity"],
    returns=(
        "dict with 'matches' RANKED BY 'adjusted_score' (= figure_of_merit "
        "penalized by the fraction of the candidate's strong predicted lines "
        "absent from the measurement — 'frac_strong_lines_absent' and the "
        "'absent_strong_lines' positions are reported per match; a high FOM "
        "with substantial absent strong lines is an isostructural/superlattice "
        "impostor). Each match: {source_id, formula, space_group, "
        "cell {a..gamma, volume}, figure_of_merit, n_matched/n_query peaks, and "
        "— for the top materialize_top hits — 'structure_path': the full CIF, "
        "resolved locally or fetched by COD ID; None when offline}), "
        "'n_library_entries', 'n_shortlisted', 'note'. figure_of_merit ≥ ~0.7 "
        "with most strong lines matched is a solid identification — CONFIRM by "
        "simulating the hit's structure_path (simulate_xrd_pattern) against the "
        "measurement and, for lattice precision, refine_rietveld on it. No "
        "convincing match across the library suggests a phase not in the "
        "library — fall back to the indexing route (index_pattern → "
        "validate_cell_lebail)."
    ),
    when_to_use=(
        "FIRST for any unknown pattern, before indexing: blind identification "
        "when the phase is likely known. Also with partial chemistry knowledge "
        "— filter the returned matches by plausible elements rather than "
        "constraining the search."
    ),
)


def search_match_pattern(
    two_theta: Sequence[float],
    intensity: Sequence[float],
    wavelength: Any = "CuKa",
    library_path: Optional[str] = None,
    top_n: int = 10,
    tol_deg: float = 0.3,
    n_key_lines: int = 3,
    n_query_lines: int = 8,
    max_shortlist: int = 500,
    materialize_top: int = 3,
    materialize_dir: str = "./fp_matches",
    strong_line_frac: float = 0.25,
    absent_lines_penalty: float = 0.0,
) -> dict[str, Any]:
    """Fingerprint search-match of measured peaks against the library.

    See ``TOOL_SPEC``. Keying: a candidate survives when its ``n_key_lines``
    strongest lines each match one of the query's ``n_query_lines`` strongest
    measured lines within ``tol_deg`` (compared in 2θ at the query wavelength).
    Shortlist is then scored with the Hanawalt figure of merit (existing
    ``score_xrd_match_robust``) and ranked. The top ``materialize_top`` matches
    get their full CIF resolved into ``materialize_dir`` (``structure_path``) —
    the bridge from the peak-list identification to structure-level work
    (simulate to confirm, then Rietveld): a locally built library's own CIF is
    used directly; otherwise the entry's COD ID is fetched (cached; graceful
    None when offline)."""
    from .score_match_robust import score_xrd_match_robust
    from .simulate_xrd import _ENGINES  # noqa: F401  (import guard parity)

    lam = _resolve_lam(wavelength)
    tt = np.asarray(two_theta, dtype=float)
    yy = np.asarray(intensity, dtype=float)
    if tt.size < 3:
        raise ValueError("search_match_pattern needs at least 3 measured peaks.")
    df = _load_library(library_path)
    if df is None or len(df) == 0:
        raise RuntimeError("Fingerprint library is empty.")

    # Query's strongest lines, in 2-theta.
    order = np.argsort(yy)[::-1][: int(n_query_lines)]
    q_strong_tt = tt[order]

    # Vectorized Hanawalt keying: candidate's top-N d's -> 2theta at the query
    # wavelength; each must land within tol of one of the query's strong lines.
    def _d_to_tt(d):
        with np.errstate(invalid="ignore"):
            s = lam / (2.0 * np.asarray(d, dtype=float))
        s = np.clip(s, -1.0, 1.0)
        return 2.0 * np.degrees(np.arcsin(s))

    keep = np.ones(len(df), dtype=bool)
    for col in ("d1", "d2", "d3")[: max(1, int(n_key_lines))]:
        cand_tt = _d_to_tt(df[col].to_numpy())
        dist = np.abs(cand_tt[:, None] - q_strong_tt[None, :]).min(axis=1)
        keep &= (dist <= float(tol_deg)) | (df[col].to_numpy() <= 0)
    shortlist = df[keep]
    n_short = len(shortlist)
    if n_short > int(max_shortlist):
        # Keep the most PROMISING keyed candidates, not the first N in library
        # file order. Crowded fingerprint regimes (organics / low-angle
        # windows) can key tens of thousands of candidates — an arbitrary
        # head() there silently drops the true phase before scoring (observed:
        # sodium ibuprofen dihydrate keyed but truncated on a real in-situ
        # frame that keyed 3159 entries). Pre-rank by a cheap vectorized
        # candidate-side coverage: the intensity fraction of each candidate's
        # stored lines that lands within tol of ANY measured line — the
        # complement of the absent-strong-lines evidence, applied to the
        # whole stored pattern.
        L = max(len(v) for v in shortlist["ds"])
        ds_pad = np.zeros((n_short, L))
        ii_pad = np.zeros((n_short, L))
        for k, (dv, iv) in enumerate(zip(shortlist["ds"], shortlist["intensities"])):
            ds_pad[k, :len(dv)] = dv
            ii_pad[k, :len(iv)] = iv
        with np.errstate(divide="ignore", invalid="ignore"):
            tt_pad = _d_to_tt(ds_pad)
        valid = (ds_pad > 0) & np.isfinite(tt_pad)
        # min distance to any measured line, accumulated per measured peak so
        # the working set stays (n_short x L) even for huge shortlists
        dmin = np.full(tt_pad.shape, np.inf)
        for q in tt:
            np.minimum(dmin, np.abs(tt_pad - float(q)), out=dmin)
        hit = dmin <= float(tol_deg)
        w = np.where(valid, ii_pad, 0.0)
        pre = (w * hit).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-9)
        shortlist = shortlist.iloc[np.argsort(-pre)[: int(max_shortlist)]]

    matches = []
    for _, row in shortlist.iterrows():
        cand_tt = _d_to_tt(np.asarray(row["ds"], dtype=float))
        m = np.isfinite(cand_tt)
        try:
            sc = score_xrd_match_robust(
                cand_tt[m].tolist(), np.asarray(row["intensities"], dtype=float)[m].tolist(),
                # exp_peaks path: we already HAVE a peak list (the profile path
                # would re-run peak extraction on it and find nothing)
                exp_peaks={"positions": tt.tolist(), "intensities": yy.tolist()},
                algorithm="hanawalt", tol_deg=float(tol_deg))
        except Exception as exc:
            _logger.debug("scoring failed for %s: %s", row["source_id"], exc)
            continue
        # ABSENT-STRONG-LINES check (classical Fink-era negative evidence): a
        # candidate whose STRONG predicted lines have no measured counterpart is
        # an impostor sharing positions on its other lines — the signature of
        # isostructural / superlattice degeneracy (observed live: ZnS pattern
        # confidently matched to bixbyite ErGdO3, whose extra strong superlattice
        # lines are simply not in the data). Position FOM cannot see this;
        # absence of predicted-strong lines can, and is robust to preferred
        # orientation in one direction (texture suppresses SOME lines, not a
        # whole predicted family). Only candidate lines inside the measured
        # window and above strong_line_frac are testable — weak lines may
        # legitimately be missing from a capped measured peak list.
        cints = np.asarray(row["intensities"], dtype=float)[m]
        ctt = cand_tt[m]
        win = (ctt >= tt.min() - float(tol_deg)) & (ctt <= tt.max() + float(tol_deg))
        strong = win & (cints >= 100.0 * float(strong_line_frac))
        if strong.any():
            dmin_ = np.abs(ctt[strong][:, None] - tt[None, :]).min(axis=1)
            absent_mask = dmin_ > float(tol_deg)
            frac_absent = float(absent_mask.mean())
            absent_lines = [round(float(v), 2) for v in ctt[strong][absent_mask]][:10]
        else:
            frac_absent, absent_lines = 0.0, []
        fom = float(sc.get("figure_of_merit", 0.0))
        adjusted = fom * (1.0 - float(absent_lines_penalty) * frac_absent)

        matches.append({
            "source_id": row["source_id"],
            "formula": row["formula"],
            "space_group": row["space_group"],
            "cell": {k: float(row[k]) for k in ("a", "b", "c", "alpha", "beta",
                                                "gamma", "volume")},
            "figure_of_merit": fom,
            "adjusted_score": round(adjusted, 4),
            "frac_strong_lines_absent": round(frac_absent, 3),
            "absent_strong_lines": absent_lines,
            "n_matched": int(sc.get("n_matched", 0)) if "n_matched" in sc else None,
            "cif_path": row.get("cif_path"),
        })
    matches.sort(key=lambda r: -r["adjusted_score"])

    # Resolve full structures for the top hits — identification stores only
    # peak lists, so structure-level follow-up (simulate confirm, Rietveld)
    # needs the actual CIF. A distributed artifact's 'cif_path' points at the
    # BUILD machine (dangling by construction); the durable key is the COD ID.
    for m in matches[: max(0, int(materialize_top))]:
        cif = m.get("cif_path")
        if cif and os.path.exists(str(cif)):          # locally built library
            m["structure_path"] = str(cif)
            continue
        try:
            from .._backends.cod import fetch_cod_cif
            m["structure_path"] = fetch_cod_cif(m["source_id"], materialize_dir)
        except Exception as exc:                       # offline / non-COD id
            _logger.debug("materialize failed for %s: %s", m["source_id"], exc)
            m["structure_path"] = None

    return {
        "matches": matches[: int(top_n)],
        "n_library_entries": int(len(df)),
        "n_shortlisted": int(n_short),
        "note": (
            "figure_of_merit >= ~0.7 with the strong lines matched is a solid "
            "ID — confirm by simulating the hit (cif_path) against the pattern "
            "and refine_rietveld for the lattice. NO convincing match means the "
            "phase is likely not in the library: widen tol_deg / lower "
            "n_key_lines (texture), or fall back to the indexing route."
        ),
    }


def _resolve_lam(wavelength: Any) -> float:
    # mirror the gsas engine's alias table without importing it (keeps this
    # module usable without the gsas extra)
    if isinstance(wavelength, (int, float)):
        return float(wavelength)
    aliases = {"cuka": 1.5406, "cuka1": 1.54056, "cuka2": 1.54439,
               "moka": 0.71073, "moka1": 0.70930, "coka": 1.78897,
               "feka": 1.93604, "crka": 2.28970, "agka": 0.55941}
    key = str(wavelength).strip().lower().replace(" ", "").replace("-", "")
    if key in aliases:
        return aliases[key]
    raise ValueError(f"Unrecognized wavelength {wavelength!r}")
