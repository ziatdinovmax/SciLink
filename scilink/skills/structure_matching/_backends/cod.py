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
import urllib.request
from pathlib import Path
from typing import Optional

from ._base import QuerySpec, StructureCandidate
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
        if not self.is_available():
            return []
        wanted = {e for e in spec.chemistry}
        if not wanted and not spec.lattice_param_ranges:
            return []

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
        for cid, formula, sg, sg_number in rows:
            struct = self._load_structure(int(cid))
            if struct is None:
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
                    "spacegroup_number": sg_number,
                    "extra_elements": extra,
                },
                # No formation energy in COD; prefer the tightest composition
                # match (fewest elements beyond those requested).
                rank_score=1.0 / (1.0 + extra),
            ))
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
            where = " AND ".join("formula LIKE ?" for _ in wanted)
            params: list = [f"%{e}%" for e in wanted]
            sql = f"SELECT file, formula, sg, sgNumber FROM data WHERE {where}"
            if spec.space_group_hints:
                sql += " AND sgNumber IN (%s)" % ",".join("?" * len(spec.space_group_hints))
                params += [int(s) for s in spec.space_group_hints]
            # Pull a generous pre-filter set; the exact-set filter prunes it.
            sql += " LIMIT 4000"
            raw = con.execute(sql, params).fetchall()
        finally:
            con.close()

        scored: list[tuple] = []
        for cid, formula, sg, sg_number in raw:
            els = _formula_elements(formula)
            if not wanted.issubset(els):
                continue
            extra = len(els - wanted)
            scored.append((extra, _formula_natoms(formula), cid, formula, sg, sg_number))
        # Tightest composition first (exact match before supersets), then
        # simplest stoichiometry (common phase before complex same-composition),
        # then NEWEST deposition first (id descending). COD ids are roughly
        # chronological, and early-1900s entries carry idealized coordinates
        # and no displacement parameters — kinematically fine for
        # identification but catastrophically wrong as Rietveld starting
        # models (observed: a 1911 zincite entry collapsed its phase to
        # 0 wt% in a certified mixture where a modern entry refined to
        # within 2 wt% of the weighed truth).
        scored.sort(key=lambda t: (t[0], t[1], -_safe_int(t[2])))
        return [(cid, formula, sg, sg_number) for _, _, cid, formula, sg, sg_number in scored]

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
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.load(resp)
        except Exception as exc:
            _logger.debug("COD REST search failed: %s", exc)
            return []

        scored: list[tuple] = []
        hints = set(int(s) for s in spec.space_group_hints) if spec.space_group_hints else None
        for row in data:
            try:
                cid = int(row["file"])
            except (KeyError, TypeError, ValueError):
                continue
            formula = row.get("formula")
            els = _formula_elements(formula)
            if wanted and not wanted.issubset(els):
                continue
            sg_number = row.get("sgNumber")
            if hints is not None:
                try:
                    if int(sg_number) not in hints:
                        continue
                except (TypeError, ValueError):
                    continue
            scored.append((len(els - wanted), _formula_natoms(formula), cid, formula, row.get("sg"), sg_number))
            if len(scored) >= 2000:
                break
        # Tightest composition, then simplest stoichiometry, then newest
        # deposition first (see _search_db for the Rietveld rationale).
        scored.sort(key=lambda t: (t[0], t[1], -_safe_int(t[2])))
        return [(cid, formula, sg, sg_number) for _, _, cid, formula, sg, sg_number in scored]

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
            urllib.request.urlretrieve(_COD_CIF_URL.format(cid=sid), cached)
            return cached
        except Exception as exc:
            _logger.debug("COD %s CIF fetch failed: %s", cid, exc)
            return None
