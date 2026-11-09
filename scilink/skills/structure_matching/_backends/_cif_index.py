"""Parquet pre-filter over a directory of CIF files, for the XRD local backend.

Without an index, :class:`LocalCIFBackend` parses every CIF on every
query (~10 ms per file via pymatgen). For a curated set of a few
hundred files that's fine; for a full COD bulk mirror (~500 k CIFs)
it's an 80-minute query.

The rows come from the shared structure index
(:mod:`scilink.utils.structure_index`, also behind plan mode's
``index_structures``), cached under
``~/.scilink/cache/local_cif_index_<root>.parquet``. A query is a pandas
filter on elements, space group and lattice, followed by full pymatgen
parses of the surviving candidates only (the backend re-checks every
criterion, so the index only has to avoid dropping a match).

``spacegroup_number`` here is pymatgen's computed space group, as this
backend has always filtered on; the declared one is in the shared columns.

Failure modes (pyarrow missing, parquet read failure) fall back to the
caller's direct-walk path so behavior degrades gracefully.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

from scilink.utils import structure_index as _shared

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

# Index schema version: the shared index's (an older cache is rebuilt).
_INDEX_VERSION = _shared.INDEX_VERSION


def cache_path_for(root: Path) -> Path:
    """Return the parquet path for a given root directory."""
    h = hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()[:16]
    return Path.home() / ".scilink" / "cache" / f"local_cif_index_{h}.parquet"


def _xrd_view(df):
    """The shared rows as this backend filters them: ``elements`` and
    ``spacegroup_number`` as pymatgen reads the file (the backend re-checks
    each candidate with pymatgen, so the pre-filter uses the same reading and
    cannot drop a file the check would accept). Where pymatgen could not read
    the file the index's own values stand (the check rejects it anyway)."""
    if df is None or df.empty:
        return df
    # a file neither ASE nor pymatgen could read cannot be a candidate
    df = df[df["elements"].notna() | df["computed_elements"].notna()].copy()
    computed = pd.to_numeric(df["computed_spacegroup_number"], errors="coerce")
    declared = pd.to_numeric(df["spacegroup_number"], errors="coerce")
    df["declared_spacegroup_number"] = declared
    df["spacegroup_number"] = computed.where(computed > 0, declared)
    df["elements"] = df["computed_elements"].where(df["computed_elements"].notna(), df["elements"])
    return df


def load_or_build_index(
    root: Path,
    *,
    force_rebuild: bool = False,
    max_workers: Optional[int] = None,
):
    """Return the up-to-date index DataFrame for ``root`` (or None when unavailable).

    The build is incremental: existing rows are reused when the CIF's
    mtime and size haven't changed; only added / modified files are re-parsed.
    """
    if not _INDEX_AVAILABLE:
        return None
    root = Path(root)
    cache_path = cache_path_for(root)
    walked = _shared.walk_structures(root, formats={"cif"})
    if not walked:
        # Empty directory: write an empty index so we don't repeatedly try
        # to rebuild on every query.
        empty = pd.DataFrame(columns=_shared.COLUMNS)
        _shared._write(empty, cache_path)
        return _xrd_view(empty)
    if len(walked) < _INDEX_THRESHOLD and not cache_path.is_file():
        # Small mirror: skip indexing — the caller's direct-walk path is
        # cheaper than building (and lookups will be fast either way).
        return None

    t0 = time.monotonic()
    result = _shared.build_index(root, cache_path, formats={"cif"}, refresh=force_rebuild,
                                 compute_symmetry=True, workers=max_workers)
    if result.n_parsed:
        _logger.info("CIF index ready: %d rows (%d parsed, %d reused) in %.1fs",
                     len(result.df), result.n_parsed, result.n_reused, time.monotonic() - t0)
    if result.failures:
        _logger.info("CIF index: %d files could not be parsed (e.g. %s)",
                     len(result.failures), result.failures[0][0])
    return _xrd_view(result.df)


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
    # Stable ordering: sort by spacegroup_number, then by path so test runs
    # are deterministic.
    filtered_sorted = filtered.sort_values(["spacegroup_number", "path"])
    paths = filtered_sorted["path"].tolist()
    if top_n is not None:
        paths = paths[: top_n * 3]  # headroom for downstream dedup
    return paths
