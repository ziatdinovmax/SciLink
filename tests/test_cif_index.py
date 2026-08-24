"""Tests for the parquet-based CIF index.

Exercises:
  - Build the index from a fresh fixture directory.
  - Query the index returns the expected paths.
  - Incremental rebuild: modifying / adding / removing CIFs is detected.
  - Filter by space-group hints and lattice param ranges.
  - Below the threshold, index is skipped (returns None).
  - LocalCIFBackend uses the index path when available.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pymatgen = pytest.importorskip("pymatgen")
from pymatgen.core import Lattice, Structure  # noqa: E402

from scilink.skills.structure_matching._backends import _cif_index
from scilink.skills.structure_matching._backends.local_cif import LocalCIFBackend
from scilink.skills.structure_matching._backends import QuerySpec


# --- Fixture helpers ----------------------------------------------------------

def _write(path: Path, struct: Structure) -> None:
    path.write_text(struct.to(fmt="cif"))


def _si():
    return Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])


def _diamond():
    return Structure.from_spacegroup("Fd-3m", Lattice.cubic(3.57), ["C"], [[0, 0, 0]])


def _ge():
    return Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.66), ["Ge"], [[0, 0, 0]])


def _nacl():
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]],
    )


def _populate(root: Path, n_extra_si: int = 0) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write(root / "si.cif", _si())
    _write(root / "diamond.cif", _diamond())
    _write(root / "germanium.cif", _ge())
    _write(root / "nacl.cif", _nacl())
    # Stress with extra silicon CIFs to force the threshold check
    for i in range(n_extra_si):
        _write(root / f"si_dup_{i:04d}.cif", _si())


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Re-route the index cache to a per-test directory."""
    cache_dir = tmp_path / "scilink_cache"
    cache_dir.mkdir()

    def _cache_path_for(root):
        import hashlib
        h = hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()[:16]
        return cache_dir / f"local_cif_index_{h}.parquet"

    monkeypatch.setattr(_cif_index, "cache_path_for", _cache_path_for)
    return cache_dir


# --- Availability + threshold -------------------------------------------------

def test_index_available():
    """In the dev env, pyarrow is installed → index path is on."""
    assert _cif_index.index_available()


def test_below_threshold_skips_build(tmp_path, isolated_cache):
    """A 4-CIF mirror is too small to warrant indexing — return None."""
    _populate(tmp_path)
    result = _cif_index.load_or_build_index(tmp_path)
    assert result is None
    # No parquet written when below threshold
    assert not any(isolated_cache.glob("*.parquet"))


def test_threshold_can_be_overridden(tmp_path, isolated_cache, monkeypatch):
    """Bump threshold artificially low to force the index path on small fixtures."""
    monkeypatch.setattr(_cif_index, "_INDEX_THRESHOLD", 2)
    _populate(tmp_path)
    df = _cif_index.load_or_build_index(tmp_path)
    assert df is not None
    # 4 unique structures
    assert len(df) == 4


# --- Build + query roundtrip --------------------------------------------------

def test_build_index_extracts_metadata(tmp_path, isolated_cache, monkeypatch):
    monkeypatch.setattr(_cif_index, "_INDEX_THRESHOLD", 2)
    _populate(tmp_path)
    df = _cif_index.load_or_build_index(tmp_path)
    assert df is not None
    formulas = set(df["reduced_formula"])
    assert {"Si", "C", "Ge", "NaCl"} == formulas
    # Si row should record a≈5.43
    si_row = df[df["reduced_formula"] == "Si"].iloc[0]
    assert abs(si_row["a"] - 5.43) < 0.01
    assert si_row["spacegroup_number"] == 227  # Fd-3m


def test_filter_by_chemistry(tmp_path, isolated_cache, monkeypatch):
    monkeypatch.setattr(_cif_index, "_INDEX_THRESHOLD", 2)
    _populate(tmp_path)
    df = _cif_index.load_or_build_index(tmp_path)
    paths = _cif_index.filter_index(df, chemistry=["Si"])
    assert len(paths) == 1
    assert "si" in Path(paths[0]).stem.lower()


def test_filter_by_space_group_hints(tmp_path, isolated_cache, monkeypatch):
    monkeypatch.setattr(_cif_index, "_INDEX_THRESHOLD", 2)
    _populate(tmp_path)
    df = _cif_index.load_or_build_index(tmp_path)
    paths = _cif_index.filter_index(df, chemistry=["Na", "Cl"], space_group_hints=[225])
    assert len(paths) == 1
    assert "nacl" in Path(paths[0]).stem.lower()


def test_filter_by_lattice_range(tmp_path, isolated_cache, monkeypatch):
    monkeypatch.setattr(_cif_index, "_INDEX_THRESHOLD", 2)
    _populate(tmp_path)
    df = _cif_index.load_or_build_index(tmp_path)
    # Si has a=5.43; restrict to that ±0.05
    paths = _cif_index.filter_index(
        df, chemistry=["Si"], lattice_param_ranges={"a": (5.40, 5.46)},
    )
    assert len(paths) == 1


def test_empty_directory_writes_empty_index(tmp_path, isolated_cache):
    df = _cif_index.load_or_build_index(tmp_path)
    assert df is not None
    assert df.empty
    # Parquet was written so we don't retry on every query
    parquets = list(isolated_cache.glob("*.parquet"))
    assert parquets


# --- Incremental rebuild ------------------------------------------------------

def test_unchanged_fingerprint_reuses_cache(tmp_path, isolated_cache, monkeypatch):
    """Second build with no changes reads from parquet and re-parses nothing."""
    monkeypatch.setattr(_cif_index, "_INDEX_THRESHOLD", 2)
    _populate(tmp_path)
    df1 = _cif_index.load_or_build_index(tmp_path)

    # Spy on _parse_one to count invocations on the second call
    n_parses = {"count": 0}
    original = _cif_index._parse_one

    def counting_parse(path):
        n_parses["count"] += 1
        return original(path)

    monkeypatch.setattr(_cif_index, "_parse_one", counting_parse)
    df2 = _cif_index.load_or_build_index(tmp_path)
    assert n_parses["count"] == 0  # nothing re-parsed
    assert len(df1) == len(df2)


def test_added_file_triggers_incremental_rebuild(tmp_path, isolated_cache, monkeypatch):
    monkeypatch.setattr(_cif_index, "_INDEX_THRESHOLD", 2)
    _populate(tmp_path)
    df1 = _cif_index.load_or_build_index(tmp_path)
    n0 = len(df1)

    # Add one new CIF
    _write(tmp_path / "extra_ge.cif", _ge())

    df2 = _cif_index.load_or_build_index(tmp_path)
    assert len(df2) == n0 + 1
    assert any("extra_ge" in p for p in df2["path"])


def test_removed_file_triggers_incremental_rebuild(tmp_path, isolated_cache, monkeypatch):
    monkeypatch.setattr(_cif_index, "_INDEX_THRESHOLD", 2)
    _populate(tmp_path)
    df1 = _cif_index.load_or_build_index(tmp_path)
    (tmp_path / "diamond.cif").unlink()
    df2 = _cif_index.load_or_build_index(tmp_path)
    assert len(df2) == len(df1) - 1
    assert not any("diamond" in p for p in df2["path"])


def test_modified_file_triggers_incremental_rebuild(tmp_path, isolated_cache, monkeypatch):
    """Replace one CIF's content; mtime changes; row should re-parse."""
    monkeypatch.setattr(_cif_index, "_INDEX_THRESHOLD", 2)
    _populate(tmp_path)
    df1 = _cif_index.load_or_build_index(tmp_path)

    # Overwrite si.cif with a different structure (Ge)
    import time as _time
    _time.sleep(0.05)  # ensure mtime changes detectably
    _write(tmp_path / "si.cif", _ge())

    df2 = _cif_index.load_or_build_index(tmp_path)
    # Same path, different formula
    row = df2[df2["path"].str.endswith("si.cif")].iloc[0]
    assert row["reduced_formula"] == "Ge"


# --- End-to-end via LocalCIFBackend -------------------------------------------

def test_local_cif_backend_uses_index_path(tmp_path, isolated_cache, monkeypatch):
    """Backend.query() goes through the index when one is built."""
    monkeypatch.setattr(_cif_index, "_INDEX_THRESHOLD", 2)
    _populate(tmp_path)
    # Pre-build the index
    _cif_index.load_or_build_index(tmp_path)

    backend = LocalCIFBackend(root_dir=str(tmp_path))
    assert backend.is_available()

    # Spy: index path should not require any extra parses since results are
    # cached. We only check that the result is correct here.
    out = backend.query(QuerySpec(chemistry=["Si"]))
    assert len(out) == 1
    assert out[0].formula == "Si"
    assert out[0].source == "local"


def test_local_cif_backend_falls_back_below_threshold(tmp_path, isolated_cache):
    """4 CIFs is below the default threshold (200); backend should use the
    direct-walk path. Result must still be correct."""
    _populate(tmp_path)
    backend = LocalCIFBackend(root_dir=str(tmp_path))
    out = backend.query(QuerySpec(chemistry=["Si"]))
    assert len(out) == 1
    assert out[0].formula == "Si"


def test_local_cif_backend_returns_same_results_either_path(
    tmp_path, isolated_cache, monkeypatch,
):
    """Direct-walk and indexed paths must agree on a small fixture."""
    _populate(tmp_path)
    backend = LocalCIFBackend(root_dir=str(tmp_path))
    walk_out = backend._query_via_walk(QuerySpec(chemistry=["Si"]))

    monkeypatch.setattr(_cif_index, "_INDEX_THRESHOLD", 2)
    idx = _cif_index.load_or_build_index(tmp_path)
    indexed_out = backend._query_via_index(idx, QuerySpec(chemistry=["Si"]))

    assert len(walk_out) == len(indexed_out) == 1
    assert walk_out[0].id == indexed_out[0].id
    assert walk_out[0].formula == indexed_out[0].formula


# --- Failure-mode test --------------------------------------------------------

def test_missing_pyarrow_returns_none(tmp_path, isolated_cache, monkeypatch):
    """When pyarrow is unavailable, load_or_build_index returns None
    so the backend's walk path runs unchanged."""
    monkeypatch.setattr(_cif_index, "_INDEX_AVAILABLE", False)
    monkeypatch.setattr(_cif_index, "_INDEX_THRESHOLD", 2)
    _populate(tmp_path)
    assert _cif_index.load_or_build_index(tmp_path) is None
