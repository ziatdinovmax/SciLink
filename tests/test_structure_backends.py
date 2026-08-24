"""Offline tests for the structure-matching backend protocol.

Mocks the Materials Project API; synthesizes fixture CIFs on a tmp path for
LocalCIFBackend. No network and no real MP key required.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from scilink.skills.structure_matching._backends import (
    CODBackend,
    LocalCIFBackend,
    MaterialsProjectBackend,
    QuerySpec,
    StructureBackend,
    StructureCandidate,
)
from scilink.skills.structure_matching._backends.materials_project import (
    _normalize_spacegroup,
    _rank_score_from_e_hull,
)


pymatgen = pytest.importorskip("pymatgen")
from pymatgen.core import Lattice, Structure  # noqa: E402


# --- QuerySpec validation -----------------------------------------------------

def test_query_spec_requires_chemistry():
    with pytest.raises(ValueError, match="chemistry"):
        QuerySpec(chemistry=[])


def test_query_spec_top_n_positive():
    with pytest.raises(ValueError, match="top_n"):
        QuerySpec(chemistry=["Si"], top_n=0)


# --- Protocol conformance -----------------------------------------------------

def test_all_backends_satisfy_protocol():
    assert isinstance(MaterialsProjectBackend(api_key="x"), StructureBackend)
    assert isinstance(LocalCIFBackend(root_dir="/nowhere"), StructureBackend)
    assert isinstance(CODBackend(), StructureBackend)


# --- MaterialsProjectBackend --------------------------------------------------

def _mp_record(material_id, formula, sg_symbol, sg_number, e_hull, structure=None):
    """Build an MPRester-style result record (SimpleNamespace mimics ORM)."""
    sym = SimpleNamespace(symbol=sg_symbol, number=sg_number)
    return SimpleNamespace(
        material_id=material_id,
        formula_pretty=formula,
        symmetry=sym,
        energy_above_hull=e_hull,
        structure=structure,
    )


def test_mp_is_available_false_without_key():
    b = MaterialsProjectBackend(api_key=None)
    # is_available depends also on MP_API_AVAILABLE; we only assert that
    # missing key forces False regardless of import state.
    if b.api_key is None:
        assert b.is_available() is False


def test_mp_query_returns_empty_when_unavailable():
    b = MaterialsProjectBackend(api_key=None)
    if b.api_key:
        pytest.skip("env carries an MP key — cannot test unavailable path here")
    assert b.query(QuerySpec(chemistry=["Si"])) == []


@patch("scilink.skills.structure_matching._backends.materials_project.MPRester")
@patch(
    "scilink.skills.structure_matching._backends.materials_project.MP_API_AVAILABLE",
    True,
)
def test_mp_query_ranks_by_e_above_hull(mock_mprester):
    records = [
        _mp_record("mp-1", "TiO2", "P4_2/mnm", 136, 0.0),   # rutile, stable
        _mp_record("mp-2", "TiO2", "I4_1/amd", 141, 0.04),  # anatase
        _mp_record("mp-3", "TiO2", "Pbca",     61,  0.10),  # brookite
    ]
    mock_mprester.return_value.__enter__.return_value.materials.summary.search.return_value = records

    b = MaterialsProjectBackend(api_key="fake")
    out = b.query(QuerySpec(chemistry=["Ti", "O"], top_n=2))

    assert [c.id for c in out] == ["mp-1", "mp-2"]
    assert out[0].rank_score > out[1].rank_score
    assert out[0].space_group == "P4_2/mnm"


@patch("scilink.skills.structure_matching._backends.materials_project.MPRester")
@patch(
    "scilink.skills.structure_matching._backends.materials_project.MP_API_AVAILABLE",
    True,
)
def test_mp_query_filters_by_space_group_hints(mock_mprester):
    records = [
        _mp_record("mp-1", "TiO2", "P4_2/mnm", 136, 0.0),
        _mp_record("mp-2", "TiO2", "I4_1/amd", 141, 0.04),
        _mp_record("mp-3", "TiO2", "Pbca",     61,  0.10),
    ]
    mock_mprester.return_value.__enter__.return_value.materials.summary.search.return_value = records

    b = MaterialsProjectBackend(api_key="fake")
    out = b.query(QuerySpec(chemistry=["Ti", "O"], space_group_hints=[141], top_n=5))

    assert [c.id for c in out] == ["mp-2"]


@patch("scilink.skills.structure_matching._backends.materials_project.MPRester")
@patch(
    "scilink.skills.structure_matching._backends.materials_project.MP_API_AVAILABLE",
    True,
)
def test_mp_query_caches_results(mock_mprester):
    search_mock = MagicMock(return_value=[_mp_record("mp-1", "Si", "Fd-3m", 227, 0.0)])
    mock_mprester.return_value.__enter__.return_value.materials.summary.search = search_mock

    b = MaterialsProjectBackend(api_key="fake")
    spec = QuerySpec(chemistry=["Si"], top_n=3)
    b.query(spec)
    b.query(spec)

    assert search_mock.call_count == 1


@patch(
    "scilink.skills.structure_matching._backends.materials_project.MPRester",
    side_effect=RuntimeError("network down"),
)
@patch(
    "scilink.skills.structure_matching._backends.materials_project.MP_API_AVAILABLE",
    True,
)
def test_mp_query_swallows_exceptions(_mock_mprester):
    b = MaterialsProjectBackend(api_key="fake")
    assert b.query(QuerySpec(chemistry=["Si"])) == []


def test_normalize_spacegroup_handles_variants():
    assert _normalize_spacegroup(None) is None
    assert _normalize_spacegroup("P-1") == "P-1"
    assert _normalize_spacegroup(SimpleNamespace(symbol="Fm-3m")) == "Fm-3m"
    assert _normalize_spacegroup({"symbol": "Pnma"}) == "Pnma"


def test_rank_score_decreases_with_e_hull():
    assert _rank_score_from_e_hull(0.0) > _rank_score_from_e_hull(0.1)
    assert _rank_score_from_e_hull(1.0) == 0.0
    assert _rank_score_from_e_hull(None) == 0.0


# --- LocalCIFBackend ----------------------------------------------------------

def _write_cif(path: Path, structure: Structure) -> None:
    path.write_text(structure.to(fmt="cif"))


def _silicon() -> Structure:
    a = 5.43
    lattice = Lattice.cubic(a)
    return Structure.from_spacegroup(
        "Fd-3m", lattice, ["Si"], [[0, 0, 0]],
    )


def _diamond_carbon() -> Structure:
    a = 3.57
    lattice = Lattice.cubic(a)
    return Structure.from_spacegroup(
        "Fd-3m", lattice, ["C"], [[0, 0, 0]],
    )


def _rocksalt_nacl() -> Structure:
    a = 5.64
    lattice = Lattice.cubic(a)
    return Structure.from_spacegroup(
        "Fm-3m", lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]],
    )


def test_local_is_available_false_for_missing_dir(tmp_path):
    b = LocalCIFBackend(root_dir=str(tmp_path / "nope"))
    assert b.is_available() is False


def test_local_is_available_true_for_real_dir(tmp_path):
    b = LocalCIFBackend(root_dir=str(tmp_path))
    assert b.is_available() is True


def test_local_query_returns_matching_chemistry(tmp_path):
    _write_cif(tmp_path / "si.cif", _silicon())
    _write_cif(tmp_path / "c_diamond.cif", _diamond_carbon())
    _write_cif(tmp_path / "nacl.cif", _rocksalt_nacl())

    b = LocalCIFBackend(root_dir=str(tmp_path))
    out = b.query(QuerySpec(chemistry=["Si"]))

    assert len(out) == 1
    assert out[0].source == "local"
    assert out[0].formula == "Si"
    assert out[0].space_group is not None
    assert Path(out[0].structure_path).name == "si.cif"


def test_local_query_filters_by_space_group(tmp_path):
    _write_cif(tmp_path / "si.cif", _silicon())          # Fd-3m, no. 227
    _write_cif(tmp_path / "nacl.cif", _rocksalt_nacl())  # Fm-3m, no. 225

    b = LocalCIFBackend(root_dir=str(tmp_path))
    out = b.query(QuerySpec(chemistry=["Na", "Cl"], space_group_hints=[225]))

    assert [c.id for c in out] == ["nacl"]


def test_local_query_filters_by_lattice_param_ranges(tmp_path):
    _write_cif(tmp_path / "si.cif", _silicon())          # a = 5.43
    _write_cif(tmp_path / "nacl.cif", _rocksalt_nacl())  # a = 5.64

    b = LocalCIFBackend(root_dir=str(tmp_path))
    out = b.query(QuerySpec(
        chemistry=["Si"], lattice_param_ranges={"a": (5.4, 5.5)},
    ))
    assert [c.id for c in out] == ["si"]

    out = b.query(QuerySpec(
        chemistry=["Si"], lattice_param_ranges={"a": (6.0, 7.0)},
    ))
    assert out == []


def test_local_query_skips_unparseable_files(tmp_path):
    _write_cif(tmp_path / "si.cif", _silicon())
    (tmp_path / "junk.cif").write_text("not a real CIF file")

    b = LocalCIFBackend(root_dir=str(tmp_path))
    out = b.query(QuerySpec(chemistry=["Si"]))

    assert len(out) == 1
    assert out[0].id == "si"


def test_local_query_walks_subdirectories(tmp_path):
    nested = tmp_path / "deep" / "nest"
    nested.mkdir(parents=True)
    _write_cif(nested / "si.cif", _silicon())

    b = LocalCIFBackend(root_dir=str(tmp_path))
    out = b.query(QuerySpec(chemistry=["Si"]))
    assert len(out) == 1


# --- CODBackend (implemented) -------------------------------------------------

def test_cod_unavailable_without_db_or_web(monkeypatch):
    """No local metadata db and web disabled -> not usable."""
    monkeypatch.delenv("SCILINK_COD_DB", raising=False)
    b = CODBackend(db_path=None, cif_dir=None, allow_web=False)
    assert b.is_available() is False


def test_cod_available_via_web(monkeypatch):
    """No db but web enabled -> usable (REST element search fallback). pymatgen
    is importorskip'd at module top, so the only remaining gate is allow_web."""
    monkeypatch.delenv("SCILINK_COD_DB", raising=False)
    b = CODBackend(db_path=None, cif_dir=None, allow_web=True)
    assert b.is_available() is True


def test_cod_query_returns_empty_when_unavailable(monkeypatch):
    """An unavailable backend returns [] (no raise) — query is a no-op, not an
    error, so search_structures can probe it harmlessly."""
    monkeypatch.delenv("SCILINK_COD_DB", raising=False)
    b = CODBackend(db_path=None, cif_dir=None, allow_web=False)
    assert b.query(QuerySpec(chemistry=["Si"])) == []


def test_cod_formula_elements_handles_fractional_counts():
    """COD formula strings carry fractional occupancy/Z counts (e.g. 'H145.17',
    'Na0.67'); the element parser must strip the numeric suffix including the
    decimal. Regression for the parser fix."""
    from scilink.skills.structure_matching._backends.cod import _formula_elements
    assert _formula_elements("- C6 H145.17 Au Cl4 N3 -") == {"C", "H", "Au", "Cl", "N"}
    assert _formula_elements("Na0.67 Mn O2") == {"Na", "Mn", "O"}
