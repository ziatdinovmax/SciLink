"""Crystallography Open Database (COD) backend.

Queries a local COD SQLite *metadata* snapshot (e.g. ``cod-YYYYMMDD.db3``, the
single ``data`` table of cell / space-group / formula columns keyed by COD id)
for candidate structures by chemistry, then materializes each structure's CIF —
from a local COD CIF archive (``SCILINK_COD_CIF_DIR``, preferred: offline, fast,
reproducible) or the COD website (fallback). COD is the right source for the
organic / metal-organic / molecular crystals that the inorganic-DFT databases
(Materials Project) do not contain.

CIFs are read from a LOCAL COD archive when available (offline, fast,
reproducible) and otherwise fetched from the COD website. The web fetch is a
SAFE, narrow operation — NOT arbitrary web browsing: the URL is hardcoded to the
vetted crystallography.net domain and the only variable is a numeric COD id
taken from the local db's integer primary key (no injection / no arbitrary
URLs), and each CIF is a ~10 KB text file. It is therefore enabled by default;
set SCILINK_COD_ALLOW_WEB=0 to force fully-offline (local-archive-only) operation.

Configuration (env vars):
  SCILINK_COD_DB        path to the COD SQLite metadata db (required).
  SCILINK_COD_CIF_DIR   local CIF archive root (preferred CIF source). CIFs are
                        read here by COD-hashed layout d1/d2d3/d4d5/<id>.cif,
                        flat <id>.cif, or any <id>.cif under the tree.
  SCILINK_COD_ALLOW_WEB '0'/'false' disables the crystallography.net fallback
                        (default on). Use for air-gapped runs with a local
                        archive.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

from ._base import QuerySkipped, QuerySpec, StructureCandidate
from .local_cif import _lattice_within_ranges

try:
    from pymatgen.core import Structure
    PYMATGEN_AVAILABLE = True
except ImportError:
    Structure = None  # type: ignore
    PYMATGEN_AVAILABLE = False

_logger = logging.getLogger(__name__)

# A COD formula is space-delimited element+count tokens, e.g. "- C6 H10 Au Cl4 N3 -".
# Counts can be fractional (occupancy/Z averaging), e.g. "H145.17", "Na0.67".
_ELEMENT_TOKEN = re.compile(r"^([A-Z][a-z]?)[\d.]*$")
_COD_CIF_URL = "https://www.crystallography.net/cod/{cid}.cif"
# A REST request that broke off (a connection reset, a response cut short on
# a large element query) is tried again after a pause, this many times in all;
# one that still fails raises, and the dispatcher records the failure.
_WEB_ATTEMPTS = 3
_WEB_BACKOFF_S = 2.0
# Row caps. The local db is ranked over EVERY entry with the exact element set
# (a cap on the loose SQL pre-filter would cut the list arbitrarily before the
# exact-set filter — for common oxides that dropped the common phase itself);
# _DB_ROW_CAP bounds only the exact-set rows. The REST path keeps at most
# _WEB_ROW_CAP rows. A query that hits a cap is recorded as truncated.
_DB_ROW_CAP = 200000
# Entry metadata read alongside each row (when the snapshot / REST reply has
# them) to choose the entry that represents a phase (see _order_group).
_META_COLUMNS = ("vol", "year", "method", "celltemp", "cellpressure", "status", "doi",
                 "authors", "title")
_WEB_ROW_CAP = 2000


def _formula_elements(formula: Optional[str]) -> set[str]:
    """Element symbols in a COD formula string (count suffixes stripped)."""
    if not formula:
        return set()
    elements: set[str] = set()
    for tok in formula.replace("-", " ").split():
        m = _ELEMENT_TOKEN.match(tok)
        if m:
            elements.add(m.group(1))
    return elements


def fetch_cod_cif(cod_id, dest_dir: Optional[str] = None) -> Optional[str]:
    """Fetch a single COD entry's CIF by its numeric COD ID.

    The public bridge from an IDENTIFICATION hit back to a full structure: the
    fingerprint library (search_match_pattern) stores only peak lists, so once
    a match is made the matched entry's actual CIF — needed for simulation
    confirmation and Rietveld — is retrieved here (local COD archive when
    configured, COD web otherwise; downloads are cached). Returns the CIF path
    (copied into ``dest_dir`` when given) or None if unavailable."""
    import shutil

    sid = str(cod_id).strip()
    if not sid.isdigit():
        return None
    backend = CODBackend()
    cif = backend._locate_cif(int(sid))
    if cif is None:
        return None
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, f"cod_{sid}.cif")
        if os.path.abspath(str(cif)) != os.path.abspath(dest):
            shutil.copyfile(str(cif), dest)
        return dest
    return str(cif)


def _query_volume_center(ranges: dict) -> Optional[float]:
    """Center of a lattice query's volume window (Å³) for cell-agreement ranking.
    Uses the explicit 'volume' key when given; otherwise approximates from the
    a/b/c range midpoints (orthogonal-cell estimate — good enough for ranking)."""
    vr = ranges.get("volume")
    if vr:
        return 0.5 * (float(vr[0]) + float(vr[1]))
    mids = []
    for k in ("a", "b", "c"):
        r = ranges.get(k)
        if not r:
            return None
        mids.append(0.5 * (float(r[0]) + float(r[1])))
    return mids[0] * mids[1] * mids[2]


def _safe_int(v) -> int:
    """COD id as int for the newest-first tiebreak; unparsable ids sort last."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return -1


# COD's SQL dump writes a missing value as the literal string '\N' (MySQL's
# NULL marker), not SQL NULL: ~3,400 entries carry sgNumber '\N'. Read
# verbatim, such an entry keys its phase by the symbol string instead of the
# number, so one phase splits into two groups (observed: alpha-quartz P3_221 as
# 11 numbered + 19 symbol-only entries, the larger group led by a theoretical
# model with a 1.6 % short c axis).
_MISSING = {"", "\\n", "?", ".", "none", "null"}


def _missing(v) -> bool:
    return v is None or (isinstance(v, str) and v.strip().lower() in _MISSING)


def _num(v) -> Optional[float]:
    if _missing(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


_SG_SYMBOL_CACHE: dict = {}


def _sg_number_from_symbol(sg) -> Optional[int]:
    """Space-group number for a COD H-M symbol ('P 32 2 1', 'R -3 c :H',
    'I 1 2/c 1') via pymatgen; None when it cannot be resolved."""
    if _missing(sg):
        return None
    key = str(sg)
    if key in _SG_SYMBOL_CACHE:
        return _SG_SYMBOL_CACHE[key]
    num = None
    try:
        from pymatgen.symmetry.groups import SpaceGroup
        sym = re.sub(r"\s*:\s*\w+$", "", key).strip()          # origin/axes choice
        compact = sym.replace(" ", "")
        # screw axes: COD writes '32' / '42' / '63' where pymatgen wants '3_2' ...
        screw = " ".join(re.sub(r"^([2346])([1-5])$", r"\1_\2", tok) for tok in sym.split())
        for cand in (sym, compact, screw, screw.replace(" ", "")):
            try:
                num = SpaceGroup(cand).int_number
                break
            except Exception:  # noqa: BLE001 - try the next spelling
                continue
    except Exception:  # noqa: BLE001 - pymatgen unavailable
        num = None
    _SG_SYMBOL_CACHE[key] = num
    return num


def _sg_number(sg, sg_number) -> Optional[int]:
    """The entry's space-group number: the stored one when valid, else
    pymatgen's reading of the H-M symbol."""
    n = _num(sg_number)
    if n is not None and n > 0:
        return int(n)
    return _sg_number_from_symbol(sg)


# --- choosing the entry that represents a phase -----------------------------
# A phase (reduced formula + space group) usually has many COD entries: ambient
# determinations, but also high-pressure and high-temperature series, doped
# samples and theoretical models, which the database rarely flags (its pressure
# and temperature columns are mostly empty). Representing a phase by its
# highest COD id picked such outliers as Rietveld starting models (e.g. a
# theoretical quartz model with a 1.6 % short c axis, which misplaces every
# quartz line and biases its refined fraction). 'consensus' (the default) takes the entry from the densest cluster
# of cell volumes — independent ambient determinations agree to a few tenths of
# a percent, while P/T series and models spread out. Support is counted in
# PUBLICATIONS, not entries: each paper votes once, at the median volume of its
# entries, since one variable-temperature or high-pressure paper can deposit
# twenty entries and would otherwise outvote the ambient cluster (observed for
# calcite and gypsum). Experimental, ambient-condition entries are preferred; within the
# cluster the entry closest to its centre, then the most recent publication
# (modern coordinates and displacement parameters). Without cell data it falls
# back to the newest-first order.
REPRESENTATIVE_CHOICES = ("consensus", "newest")
_CONSENSUS_VOL_TOL = 0.01      # relative volume window of the consensus cluster
_CLOSENESS_BIN = 0.001         # entries this close (relative) to the centre tie on closeness
_AMBIENT_T_K = (273.0, 313.0)  # cell temperature counted as ambient (K)
_AMBIENT_P_KPA = 500.0         # cell pressure above this (kPa) is non-ambient


def _meta(row) -> dict:
    return row[6] if len(row) > 6 and isinstance(row[6], dict) else {}


# Entries imported from the American Mineralogist Crystal Structure Database
# (COD 9xxxxxx) leave the condition columns empty and state the conditions in
# the title instead ('... Sample: P = 2.5 GPa', '... T = 400 C').
_NUMBER = r"(-?\d+(?:\.\d*)?|-?\.\d+)"          # a number, never a bare '.'
_TITLE_P = re.compile(r"\bP\s*=\s*" + _NUMBER + r"\s*(GPa|kbar|MPa|kPa|bar)\b", re.I)
_TITLE_T = re.compile(r"\bT\s*=\s*" + _NUMBER + r"\s*(?:°|deg(?:rees)?\.?\s*)?(C|K)\b")
_TO_KPA = {"gpa": 1e6, "kbar": 1e5, "mpa": 1e3, "kpa": 1.0, "bar": 100.0}


def _conditions(meta: dict) -> tuple:
    """(cell temperature K, cell pressure kPa) from the columns, else the title."""
    t, p = _num(meta.get("celltemp")), _num(meta.get("cellpressure"))
    title = str(meta.get("title") or "")
    if p is None:
        m = _TITLE_P.search(title)
        if m:
            p = float(m.group(1)) * _TO_KPA[m.group(2).lower()]
    if t is None:
        m = _TITLE_T.search(title)
        if m:
            t = float(m.group(1)) + (273.15 if m.group(2) == "C" else 0.0)
    return t, p


def _non_ambient(meta: dict) -> bool:
    t, p = _conditions(meta)
    return bool((t is not None and not _AMBIENT_T_K[0] <= t <= _AMBIENT_T_K[1])
                or (p is not None and p > _AMBIENT_P_KPA))


def _theoretical(meta: dict) -> bool:
    return str(meta.get("method") or "").strip().lower() == "theoretical"


def _order_group(rows: list, representative: str = "consensus") -> list:
    """Order one phase's entries, representative first (see the comment above).
    Rows are ``(extra, natoms, cid, formula, sg, sg_number[, meta])``."""
    def newest(t):
        return -_safe_int(t[2])
    if representative == "newest" or len(rows) < 2:
        return sorted(rows, key=newest)
    vols = {t[2]: _num(_meta(t).get("vol")) for t in rows}
    usable = [t for t in rows if vols[t[2]] and vols[t[2]] > 0
              and not _theoretical(_meta(t)) and not _non_ambient(_meta(t))]
    def year(t):
        y = _num(_meta(t).get("year"))
        return y if y is not None else -1.0

    def publication(t):
        # titles can carry per-entry notes (conditions), so a paper is keyed
        # by its DOI, else its authors and year
        m = _meta(t)
        if m.get("doi"):
            return str(m["doi"]).strip().lower()
        if m.get("authors"):
            return f"{str(m['authors']).strip().lower()}|{m.get('year')}"
        return f"entry:{t[2]}"

    center = None
    by_pub: dict = {}
    for t in usable:
        by_pub.setdefault(publication(t), []).append(t)
    votes = [(float(np.median([vols[t[2]] for t in ts])), max(year(t) for t in ts))
             for ts in by_pub.values()]
    if len(votes) >= 2:
        def window(v):
            return [w for w, _ in votes if abs(w - v) <= _CONSENSUS_VOL_TOL * v]
        best = max(votes, key=lambda vy: (len(window(vy[0])), vy[1]))
        if len(window(best[0])) >= 2:     # one publication is no consensus
            center = float(np.median(window(best[0])))
        elif len(votes) >= 3:             # no two papers agree: the middle one
            center = float(np.median([w for w, _ in votes]))

    def key(t):
        m, v = _meta(t), vols[t[2]]
        flagged = str(m.get("status") or "").strip().lower() in ("retracted", "errors")
        if center is None:
            off, close = False, 0
        elif v is None:
            off, close = True, 0
        else:
            rel = abs(v - center) / center
            off, close = rel > _CONSENSUS_VOL_TOL, int(rel / _CLOSENESS_BIN)
        # fewest elements beyond the request first: a partly deuterated or
        # doped entry can share the reduced formula of the phase
        return (t[0], flagged, _theoretical(m), _non_ambient(m), off, close, -year(t), newest(t))
    return sorted(rows, key=key)


def _formula_natoms(formula: Optional[str]) -> float:
    """Total atom count of a COD (reduced) formula string — a structural-
    simplicity proxy: TiO2 -> 3, Ti9O17 -> 26. Used as a RETRIEVAL tiebreaker so
    the simplest stoichiometry consistent with the requested chemistry (the most
    common phase, by an Occam prior) is offered ahead of complex same-composition
    phases — e.g. anatase/rutile (TiO2) before the Magnéli suboxides (TinO2n-1),
    which COD's id order otherwise buries. The scorer + widen-on-failure recover
    genuinely complex phases. Unknown/empty -> inf so it sorts last."""
    if not formula:
        return float("inf")
    total = 0.0
    for tok in formula.replace("-", " ").split():
        m = _ELEMENT_TOKEN.match(tok)
        if m:
            cnt = tok[len(m.group(1)):]
            total += float(cnt) if cnt else 1.0
    return total if total > 0 else float("inf")


def _stoich_fractional(formula: Optional[str]) -> bool:
    """True when a formula carries non-integer counts (doped, defective or
    partially occupied variants such as 'Ti0.913 O0.7304')."""
    for tok in (formula or "").replace("-", " ").split():
        m = _ELEMENT_TOKEN.match(tok)
        if m:
            cnt = tok[len(m.group(1)):]
            if cnt and abs(float(cnt) - round(float(cnt))) > 1e-6:
                return True
    return False


def _phase_key(formula: Optional[str], sg, sg_number) -> tuple:
    """One key per phase: reduced formula + space-group number (string fallback)."""
    try:
        from pymatgen.core import Composition
        f = Composition(str(formula or "").replace("-", " ").replace(" ", "")).reduced_formula
    except Exception:  # noqa: BLE001 - unparseable formulas key as written
        f = str(formula or "").strip("- ").strip()
    n = _sg_number(sg, sg_number)
    return (f, n if n else str(sg or "").replace(" ", ""))


def _rank_rows(scored: list[tuple], rank_by: str = "reported",
               representative: str = "consensus") -> list[tuple]:
    """Order ``(extra, natoms, cid, formula, sg, sg_number[, meta])`` rows by PHASE.

    Rows are grouped per phase (reduced formula + space group). Groups are
    ordered by fewest elements beyond those requested, then whole-number
    before fractional stoichiometry, then by ``rank_by``:
    'reported' — most entries in the database first (how often the phase has
    been reported), then fewest atoms; 'simplest' — fewest atoms, then most
    reported; 'newest' — most recent deposition. Within a group the representative
    entry comes first (``representative``: 'consensus' — see ``_order_group``
    — or 'newest', the highest COD id). Returns ``(cid, formula, sg,
    sg_number, meta)`` rows group by group, so a caller taking one candidate
    per phase gets the groups in rank order; ``meta`` carries the entry's
    database fields and the group size."""
    groups: dict[tuple, list[tuple]] = {}
    for row in scored:
        groups.setdefault(_phase_key(row[3], row[4], row[5]), []).append(row)
    for k in list(groups):
        groups[k] = _order_group(groups[k], representative)

    def gkey(rows):
        natoms, formula = rows[0][1], rows[0][3]
        extra = min(r[0] for r in rows)
        frac = _stoich_fractional(formula)
        # the group's most recent deposition, whichever entry represents it
        newest = -max(_safe_int(r[2]) for r in rows)
        if rank_by == "simplest":
            return (extra, frac, natoms, -len(rows), newest)
        if rank_by == "newest":
            return (extra, frac, newest)
        return (extra, frac, -len(rows), natoms, newest)

    ordered = sorted(groups.values(), key=gkey)
    out = []
    for rows in ordered:
        for t in rows:
            meta = dict(_meta(t))
            meta["group_entries"] = len(rows)
            out.append((t[2], t[3], t[4], _sg_number(t[4], t[5]) or t[5], meta))
    return out


class CODBackend:
    name = "cod"

    def __init__(self, db_path: Optional[str] = None, cif_dir: Optional[str] = None,
                 allow_web: Optional[bool] = None):
        self.db_path = db_path or os.getenv("SCILINK_COD_DB")
        self.cif_dir = cif_dir or os.getenv("SCILINK_COD_CIF_DIR")
        if allow_web is None:
            # Default ON: a CIF fetch from the vetted, domain-locked COD site by
            # numeric id is a safe small-file download. Opt out for air-gapped.
            allow_web = os.getenv("SCILINK_COD_ALLOW_WEB", "1").lower() not in ("0", "false", "no")
        self.allow_web = allow_web
        self._cache_dir = Path(tempfile.gettempdir()) / "scilink_cod_cifs"

    def _has_local_db(self) -> bool:
        return bool(self.db_path) and Path(self.db_path).is_file()

    def is_available(self) -> bool:
        # Usable with EITHER a local metadata db OR (vetted) web search — a user
        # without the ~1 GB db can still search COD online.
        return PYMATGEN_AVAILABLE and (self._has_local_db() or self.allow_web)

    def query(self, spec: QuerySpec) -> list[StructureCandidate]:
        self.last_report = {"dropped": 0, "drop_reasons": [], "truncated": False}
        if not self.is_available():
            raise QuerySkipped("COD is not available (no local db and web access disabled)")
        wanted = {e for e in spec.chemistry}
        if not wanted and not spec.lattice_param_ranges:
            raise QuerySkipped("COD needs chemistry or a lattice filter")

        # Prefer the local db (fast, offline); fall back to COD's REST search
        # when there is no db. Cell-only (blind) queries ALWAYS use the REST
        # API — the local metadata db carries no cell parameters.
        if self._has_local_db() and wanted:
            rows = self._search_db(wanted, spec)
        else:
            rows = self._search_web(wanted, spec)
        # Cell-only (blind) queries rank by CELL AGREEMENT, not composition: the
        # element-count rank below would put elemental phases first (observed
        # live: a blind agent concluded "Te" because elemental Te topped a
        # volume-window hit list). Collect a wider pool, then sort by volume
        # proximity to the query window's center and trim.
        cell_only = not wanted and bool(spec.lattice_param_ranges)
        pool_cap = max(3 * int(spec.top_n), 30) if cell_only else int(spec.top_n)

        candidates: list[StructureCandidate] = []
        phases_taken: set = set()
        for row in rows:
            cid, formula, sg, sg_number = row[:4]
            row_meta = row[4] if len(row) > 4 and isinstance(row[4], dict) else {}
            # top_n counts distinct PHASES: further entries of a phase that
            # already has a loaded candidate are skipped (search_structures
            # dedupes per phase anyway — they would only eat the budget).
            pkey = _phase_key(formula, sg, sg_number)
            if pkey in phases_taken:
                continue
            struct = self._load_structure(int(cid))
            if struct is None:
                # a match that could not be loaded is not an absence: counted
                self.last_report["dropped"] += 1
                if len(self.last_report["drop_reasons"]) < 5:
                    self.last_report["drop_reasons"].append(f"COD {cid}: CIF not available or unparsable")
                continue
            # Post-load lattice filter (exact, on the parsed structure). The
            # REST cell filter is a coarse pre-filter; the local db has none.
            if spec.lattice_param_ranges and not _lattice_within_ranges(
                    struct.lattice, spec.lattice_param_ranges):
                continue
            extra = len(_formula_elements(formula) - wanted)
            candidates.append(StructureCandidate(
                id=str(cid),
                source="cod",
                formula=(formula or "").strip("- ").strip(),
                space_group=sg,
                metadata={
                    "_structure": struct,
                    # the entry's own CIF (local archive or download cache):
                    # materialized verbatim so the declared space group and the
                    # atomic displacement parameters reach refinement
                    "_source_cif": str(getattr(self, "_located", {}).get(int(cid), "")),
                    "spacegroup_number": sg_number,
                    "extra_elements": extra,
                    # why this entry represents its phase (see _order_group)
                    "cod_entry": {k: row_meta[k] for k in
                                  ("year", "method", "vol", "group_entries")
                                  if k in row_meta and not _missing(row_meta[k])},
                },
                # No formation energy in COD; prefer the tightest composition
                # match (fewest elements beyond those requested).
                rank_score=1.0 / (1.0 + extra),
            ))
            phases_taken.add(pkey)
            if len(candidates) >= pool_cap:
                break

        if cell_only and candidates:
            v_center = _query_volume_center(spec.lattice_param_ranges)
            if v_center:
                candidates.sort(key=lambda c: abs(
                    c.metadata["_structure"].lattice.volume - v_center))
                for rank, cand in enumerate(candidates):
                    cand.rank_score = 1.0 / (1.0 + rank)
            candidates = candidates[: int(spec.top_n)]
        return candidates

    # -- internals ---------------------------------------------------------

    def _search_db(self, wanted: set[str], spec: QuerySpec) -> list[tuple]:
        """SQL pre-filter (substring LIKE per element — loose for short symbols
        like C/N) then Python post-filter on the exact element set, so e.g. a
        request for C does not leak Cl/Ca/Cu entries."""
        con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            cols = {r[1] for r in con.execute("PRAGMA table_info(data)")}
            extra_cols = [c for c in _META_COLUMNS if c in cols]
            where = " AND ".join("formula LIKE ?" for _ in wanted)
            params: list = [f"%{e}%" for e in wanted]
            select = ", ".join(["file", "formula", "sg", "sgNumber", *extra_cols])
            sql = f"SELECT {select} FROM data WHERE {where}"
            if spec.space_group_hints:
                # entries whose number is missing ('\N') are kept here and
                # matched on the number their symbol resolves to below
                sql += (" AND (sgNumber IN (%s) OR typeof(sgNumber) != 'integer')"
                        % ",".join("?" * len(spec.space_group_hints)))
                params += [int(s) for s in spec.space_group_hints]
            # No LIMIT here: the LIKE pre-filter is loose, and capping it before
            # the exact-set filter drops arbitrary (often the most common) phases.
            raw = con.execute(sql, params).fetchall()
        finally:
            con.close()
        hints = {int(h) for h in spec.space_group_hints} if spec.space_group_hints else None
        scored: list[tuple] = []
        for rec in raw:
            cid, formula, sg, sg_number = rec[:4]
            els = _formula_elements(formula)
            if not wanted.issubset(els):
                continue
            if hints is not None and _sg_number(sg, sg_number) not in hints:
                continue
            meta = {c: v for c, v in zip(extra_cols, rec[4:]) if not _missing(v)}
            extra = len(els - wanted)
            scored.append((extra, _formula_natoms(formula), cid, formula, sg, sg_number, meta))
        if len(scored) > _DB_ROW_CAP:
            self.last_report["truncated"] = True
            scored.sort(key=lambda t: (t[0], t[1]))
            scored = scored[:_DB_ROW_CAP]
        # Rank by PHASE (see _rank_rows): tightest composition first, whole-
        # number stoichiometry before doped/defective variants, then by
        # spec.rank_by (default: most often reported). Each phase is
        # represented by its NEWEST entry: COD ids are roughly chronological,
        # and early-1900s entries carry idealized coordinates and no
        # displacement parameters — kinematically fine for identification but
        # catastrophically wrong as Rietveld starting models (observed: a 1911
        # zincite entry collapsed its phase to 0 wt% in a certified mixture
        # where a modern entry refined to within 2 wt% of the weighed truth).
        return _rank_rows(scored, getattr(spec, "rank_by", "reported"),
                          getattr(spec, "representative", "consensus"))

    def _search_web(self, wanted: set[str], spec: QuerySpec) -> list[tuple]:
        """COD REST search (no local db, or cell-only). Same vetted domain;
        returns the same (id, formula, sg, sgNumber) rows as the local-db path.

        Queries by element (el1..elN) when chemistry is given, and/or by unit
        cell when ``spec.lattice_param_ranges`` is set — COD's API supports
        cell-edge windows (amin/amax, bmin/bmax, cmin/cmax in Å) and a volume
        window (vmin/vmax in Å^3, permutation-invariant, key 'volume'). The
        cell-only form is the blind-identification path (cell from
        index_pattern, no chemistry known)."""
        import json
        import urllib.parse
        params = [(f"el{i}", el) for i, el in enumerate(sorted(wanted), 1)]
        if spec.lattice_param_ranges:
            for axis in ("a", "b", "c"):
                if axis in spec.lattice_param_ranges:
                    lo, hi = spec.lattice_param_ranges[axis]
                    params += [(f"{axis}min", f"{float(lo):.4f}"),
                               (f"{axis}max", f"{float(hi):.4f}")]
            if "volume" in spec.lattice_param_ranges:
                lo, hi = spec.lattice_param_ranges["volume"]
                params += [("vmin", f"{float(lo):.2f}"), ("vmax", f"{float(hi):.2f}")]
        params.append(("format", "json"))
        url = "https://www.crystallography.net/cod/result?" + urllib.parse.urlencode(params)
        data = None
        for attempt in range(1, _WEB_ATTEMPTS + 1):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    data = json.load(resp)
                break
            except Exception as exc:
                # a 4xx answer will not change on a retry, except a timeout or a throttle
                client_error = (isinstance(exc, urllib.error.HTTPError) and 400 <= exc.code < 500
                                and exc.code not in (408, 429))
                if client_error:
                    raise RuntimeError(f"COD REST search rejected (HTTP {exc.code}): {exc}") from exc
                if attempt == _WEB_ATTEMPTS:
                    # a failed request is not an empty database: the dispatcher records it
                    raise RuntimeError(
                        f"COD REST search failed after {attempt} attempts: {exc}") from exc
                _logger.debug("COD REST search attempt %d failed (%s); retrying", attempt, exc)
                time.sleep(_WEB_BACKOFF_S * attempt)

        if not isinstance(data, list):
            raise RuntimeError(f"COD REST search returned {type(data).__name__}, not a list of rows")
        scored: list[tuple] = []
        hints = set(int(s) for s in spec.space_group_hints) if spec.space_group_hints else None
        for row in data:
            try:
                cid = int(row["file"])
            except (KeyError, TypeError, ValueError):
                self.last_report["dropped"] += 1
                if "a row without a COD id" not in self.last_report["drop_reasons"]:
                    self.last_report["drop_reasons"].append("a row without a COD id")
                continue
            formula = row.get("formula")
            els = _formula_elements(formula)
            if wanted and not wanted.issubset(els):
                continue
            sg_number = row.get("sgNumber")
            if hints is not None and _sg_number(row.get("sg"), sg_number) not in hints:
                continue
            if len(scored) >= _WEB_ROW_CAP:
                # a matching row past the cap: the rest are never ranked
                self.last_report["truncated"] = True
                break
            meta = {c: row.get(c) for c in _META_COLUMNS if not _missing(row.get(c))}
            scored.append((len(els - wanted), _formula_natoms(formula), cid, formula,
                           row.get("sg"), sg_number, meta))
        # Same phase ranking as the local-db path (see _search_db).
        return _rank_rows(scored, getattr(spec, "rank_by", "reported"),
                          getattr(spec, "representative", "consensus"))

    def _load_structure(self, cid: int):
        cif = self._locate_cif(cid)
        if cif is None:
            return None
        try:
            return Structure.from_file(str(cif))
        except Exception as exc:
            _logger.debug("COD %s CIF unparsable: %s", cid, exc)
            return None

    def _locate_cif(self, cid: int) -> Optional[Path]:
        """Path of COD entry ``cid``'s CIF (local archive, or the download
        cache after fetching it), remembered in ``self._located`` so a query
        can hand the file itself on without locating it twice."""
        path = self._locate_cif_uncached(cid)
        if path is not None:
            if not hasattr(self, "_located"):
                self._located = {}
            self._located[int(cid)] = path
        return path

    def _locate_cif_uncached(self, cid: int) -> Optional[Path]:
        sid = str(cid)
        # 1. Local archive (preferred).
        if self.cif_dir:
            root = Path(self.cif_dir)
            # COD hashed layout: d1/d2d3/d4d5/<id>.cif
            if len(sid) == 7:
                hashed = root / sid[0] / sid[1:3] / sid[3:5] / f"{sid}.cif"
                if hashed.is_file():
                    return hashed
            flat = root / f"{sid}.cif"
            if flat.is_file():
                return flat
            hits = list(root.rglob(f"{sid}.cif"))
            if hits:
                return hits[0]
        # 2. COD web fallback (vetted domain, numeric id, ~10 KB file). On by
        # default; disabled for air-gapped runs.
        if not self.allow_web:
            return None
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cached = self._cache_dir / f"{sid}.cif"
        if cached.is_file():
            return cached
        try:
            # Published whole: two matchers fetching one id never share a file.
            from scilink.utils.download import download_once
            # One lock for the whole cache: a lock file beside every 10 KB
            # CIF would double the cache's file count.
            download_once(_COD_CIF_URL.format(cid=sid), cached, timeout=30,
                          lock=self._cache_dir)
            return cached
        except Exception as exc:
            _logger.debug("COD %s CIF fetch failed: %s", cid, exc)
            return None
