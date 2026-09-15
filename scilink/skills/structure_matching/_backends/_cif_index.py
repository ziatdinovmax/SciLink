"""Lazy parquet index over a directory of CIF files.

Without an index, :class:`LocalCIFBackend` parses every CIF on every
query (~10 ms per file via pymatgen). For a curated set of a few
hundred files that's fine; for a full COD bulk mirror (~500 k CIFs)
it's an 80-minute query.

The index pre-extracts the columns the dispatcher actually filters on
(reduced formula, element set, space-group number, lattice parameters)
into a parquet file under ``~/.scilink/cache/local_cif_index_<root>.parquet``.
A query then becomes a pandas filter (~ms for 500 k rows) followed by
a small number of full CIF parses (only the top-N candidates that
survived filtering).

Failure modes (pyarrow missing, parquet read failure) fall back to the
caller's direct-walk path so behavior degrades gracefully.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_logger = logging.getLogger(__name__)


try:
    import pandas as pd
    import pyarrow  # noqa: F401 — required for parquet I/O
    _INDEX_AVAILABLE = True
except ImportError:
    pd = None  # type: ignore[assignment]
    _INDEX_AVAILABLE = False


def index_available() -> bool:
    """True when the parquet index can be built (pandas + pyarrow installed)."""
    return _INDEX_AVAILABLE


# Number of CIFs above which the index is worth building. Below this,
# direct-walk-and-parse is roughly the same cost as the index build itself.
_INDEX_THRESHOLD = 200

# Index schema version. Bump when the column set or semantics changes
# in a way that invalidates existing parquet caches.
_INDEX_VERSION = 1

# Columns in the parquet index, in order.
_COLUMNS = [
    "path",
    "reduced_formula",
    "elements",           # sorted element symbols joined with '-'
    "spacegroup_number",
    "spacegroup_symbol",
    "a", "b", "c",
    "alpha", "beta", "gamma",
    "mtime",
    "size",
]


@dataclass
class _Fingerprint:
    """Cheap aggregate signature over the CIF set.

    If any of (count, max_mtime, sum_size) changes between builds, the
    index is stale and we rebuild incrementally (re-parse only the
    added / modified files; carry the rest forward).
    """
    count: int
    max_mtime: float
    sum_size: int

    @classmethod
    def of_set(cls, stats: list[tuple[Path, os.stat_result]]) -> "_Fingerprint":
        return cls(
            count=len(stats),
            max_mtime=max((s.st_mtime for _, s in stats), default=0.0),
            sum_size=sum(s.st_size for _, s in stats),
        )


# ---------------------------------------------------------------------------
# Index construction + query
# ---------------------------------------------------------------------------

def cache_path_for(root: Path) -> Path:
    """Return the parquet path for a given root directory."""
    h = hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()[:16]
    return Path.home() / ".scilink" / "cache" / f"local_cif_index_{h}.parquet"


def _walk_cifs(root: Path) -> list[tuple[Path, os.stat_result]]:
    """Return ``(path, stat_result)`` for every .cif under root, sorted by path."""
    out: list[tuple[Path, os.stat_result]] = []
    for p in root.rglob("*.cif"):
        try:
            st = p.stat()
        except OSError:
            continue
        out.append((p, st))
    out.sort(key=lambda x: str(x[0]))
    return out


def _parse_one(cif_path: str) -> Optional[dict]:
    """Parse one CIF and return the index row, or None on failure."""
    try:
        from pymatgen.core import Structure
    except ImportError:
        return None
    try:
        struct = Structure.from_file(cif_path)
    except Exception:
        return None
    composition = struct.composition
    elements = sorted({str(el) for el in composition.elements})
    try:
        sg_symbol, sg_number = struct.get_space_group_info()
    except Exception:
        sg_symbol, sg_number = None, None
    lattice = struct.lattice
    st = os.stat(cif_path)
    return {
        "path": cif_path,
        "reduced_formula": str(composition.reduced_formula),
        "elements": "-".join(elements),
        "spacegroup_number": int(sg_number) if sg_number is not None else -1,
        "spacegroup_symbol": str(sg_symbol) if sg_symbol else "",
        "a": float(lattice.a),
        "b": float(lattice.b),
        "c": float(lattice.c),
        "alpha": float(lattice.alpha),
        "beta": float(lattice.beta),
        "gamma": float(lattice.gamma),
        "mtime": float(st.st_mtime),
        "size": int(st.st_size),
    }


def _load_existing(cache_path: Path):
    if not _INDEX_AVAILABLE or not cache_path.is_file():
        return None
    try:
        df = pd.read_parquet(cache_path)
    except Exception as e:
        _logger.warning("Failed to read existing index %s: %s — rebuilding", cache_path, e)
        return None
    # Schema sanity
    missing = [c for c in _COLUMNS if c not in df.columns]
    if missing:
        _logger.info(
            "Index at %s missing columns %s — rebuilding (v%d schema)",
            cache_path, missing, _INDEX_VERSION,
        )
        return None
    return df


def _write_index(df, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    df.to_parquet(tmp, engine="pyarrow", index=False)
    tmp.replace(cache_path)


def _build_incremental(
    *,
    root: Path,
    cache_path: Path,
    existing,
    walked: list[tuple[Path, os.stat_result]],
    max_workers: Optional[int],
):
    """Re-parse only added / modified CIFs; carry the rest forward."""
    walked_map = {str(p): st for p, st in walked}
    existing_paths = set()
    keep_indices: list[int] = []

    if existing is not None and not existing.empty:
        existing_paths = set(existing["path"].astype(str).tolist())
        # Rows whose path still exists AND whose mtime matches are kept.
        for i, (path, mtime) in enumerate(zip(existing["path"], existing["mtime"])):
            st = walked_map.get(str(path))
            if st is not None and abs(st.st_mtime - float(mtime)) < 1e-3:
                keep_indices.append(i)

    to_parse = [
        str(p) for p, _ in walked
        if str(p) not in existing_paths or _row_needs_reparse(existing, p, walked_map)
    ]
    # Dedupe in case both branches above selected the same path.
    to_parse = sorted(set(to_parse))

    n_carry = len(keep_indices)
    n_parse = len(to_parse)
    if n_parse:
        _logger.info(
            "Building CIF index for %s: %d new/changed CIFs to parse, "
            "%d carried from cache",
            root, n_parse, n_carry,
        )

    new_rows: list[dict] = []
    if to_parse:
        workers = max_workers or max(1, (os.cpu_count() or 4) // 2)
        # Parsing CIFs is CPU-bound (pymatgen does symmetry analysis), so a
        # ProcessPoolExecutor would be faster — but on macOS spawn-fork
        # interactions with the parent's loaded modules can cause subtle
        # issues. ThreadPoolExecutor is good enough at the 10× scale we
        # care about (build amortizes to seconds, not the full hour).
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for i, row in enumerate(pool.map(_parse_one, to_parse, chunksize=64), 1):
                if row is not None:
                    new_rows.append(row)
                if i % 10000 == 0:
                    _logger.info("  ... indexed %d / %d CIFs", i, n_parse)

    if existing is not None and not existing.empty:
        carry = existing.iloc[keep_indices].reset_index(drop=True) if keep_indices else None
    else:
        carry = None

    if pd is None:
        return None
    if new_rows:
        new_df = pd.DataFrame(new_rows, columns=_COLUMNS)
    else:
        new_df = pd.DataFrame(columns=_COLUMNS)
    if carry is not None and not carry.empty:
        merged = pd.concat([carry, new_df], ignore_index=True)
    else:
        merged = new_df

    # Drop any rows whose path no longer exists on disk
    if not merged.empty:
        merged = merged[merged["path"].isin([str(p) for p, _ in walked])].reset_index(drop=True)

    _write_index(merged, cache_path)
    return merged


def _row_needs_reparse(existing, p: Path, walked_map: dict[str, os.stat_result]) -> bool:
    """True when ``p`` exists in ``existing`` but its mtime has drifted."""
    if existing is None or existing.empty:
        return True
    matches = existing[existing["path"] == str(p)]
    if matches.empty:
        return True
    st = walked_map.get(str(p))
    if st is None:
        return False
    cached_mtime = float(matches.iloc[0]["mtime"])
    return abs(st.st_mtime - cached_mtime) >= 1e-3


def load_or_build_index(
    root: Path,
    *,
    force_rebuild: bool = False,
    max_workers: Optional[int] = None,
):
    """Return the up-to-date index DataFrame for ``root`` (or None when unavailable).

    The build is incremental: existing rows are reused when the CIF's
    mtime hasn't drifted; only added / modified files are re-parsed.
    """
    if not _INDEX_AVAILABLE:
        return None
    cache_path = cache_path_for(root)
    walked = _walk_cifs(root)
    if not walked:
        # Empty directory: write an empty index so we don't repeatedly try
        # to rebuild on every query.
        empty = pd.DataFrame(columns=_COLUMNS)
        _write_index(empty, cache_path)
        return empty
    if len(walked) < _INDEX_THRESHOLD and not cache_path.is_file():
        # Small mirror: skip indexing — the caller's direct-walk path is
        # cheaper than building (and lookups will be fast either way).
        return None

    existing = None if force_rebuild else _load_existing(cache_path)
    if existing is not None:
        fp_existing = _Fingerprint(
            count=len(existing),
            max_mtime=float(existing["mtime"].max() if not existing.empty else 0.0),
            sum_size=int(existing["size"].sum() if not existing.empty else 0),
        )
        fp_current = _Fingerprint.of_set(walked)
        if fp_existing == fp_current:
            return existing

    t0 = time.monotonic()
    df = _build_incremental(
        root=root, cache_path=cache_path, existing=existing,
        walked=walked, max_workers=max_workers,
    )
    _logger.info("CIF index ready: %d rows in %.1fs", len(df) if df is not None else 0, time.monotonic() - t0)
    return df


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def filter_index(
    df,
    *,
    chemistry: list[str],
    space_group_hints: Optional[list[int]] = None,
    lattice_param_ranges: Optional[dict[str, tuple[float, float]]] = None,
    top_n: Optional[int] = None,
) -> list[str]:
    """Return CIF paths matching the spec. Empty list when no rows match.

    ``chemistry`` may be empty for a cell-only (blind) query — then no element
    filter is applied and the lattice / space-group filters do the selection."""
    if df is None or df.empty:
        return []
    if chemistry:
        filtered = df[df["elements"] == "-".join(sorted(chemistry))]
    else:
        filtered = df
    if space_group_hints:
        filtered = filtered[filtered["spacegroup_number"].isin(space_group_hints)]
    if lattice_param_ranges:
        for axis in ("a", "b", "c", "alpha", "beta", "gamma"):
            if axis in lattice_param_ranges:
                lo, hi = lattice_param_ranges[axis]
                filtered = filtered[(filtered[axis] >= lo) & (filtered[axis] <= hi)]
    if filtered.empty:
        return []
    paths = filtered["path"].tolist()
    # Stable ordering: sort by spacegroup_number, then by path so test runs
    # are deterministic.
    filtered_sorted = filtered.sort_values(["spacegroup_number", "path"])
    paths = filtered_sorted["path"].tolist()
    if top_n is not None:
        paths = paths[: top_n * 3]  # headroom for downstream dedup
    return paths
