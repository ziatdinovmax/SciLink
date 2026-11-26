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
    QuerySkipped,
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


def test_mp_query_skips_when_unavailable():
    """An empty list means the database answered with nothing; an unavailable
    backend says it was skipped instead."""
    b = MaterialsProjectBackend(api_key=None)
    if b.api_key:
        pytest.skip("env carries an MP key — cannot test unavailable path here")
    with pytest.raises(QuerySkipped):
        b.query(QuerySpec(chemistry=["Si"]))


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
def test_mp_query_raises_on_failure(_mock_mprester):
    """A failed request is not an empty database: it raises, and the
    dispatcher records the failure."""
    b = MaterialsProjectBackend(api_key="fake")
    with pytest.raises(RuntimeError, match="network down"):
        b.query(QuerySpec(chemistry=["Si"]))


@patch(
    "scilink.skills.structure_matching._backends.materials_project.MP_API_AVAILABLE",
    True,
)
def test_mp_query_skips_cell_only():
    b = MaterialsProjectBackend(api_key="fake")
    with pytest.raises(QuerySkipped):
        b.query(QuerySpec(chemistry=[], lattice_param_ranges={"a": (3.0, 4.0)}))


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
    # a walk cannot know an unparsable file's chemistry: counted apart from drops
    assert b.last_report["unparsed"] == 1 and b.last_report["dropped"] == 0


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


def test_cod_query_skips_when_unavailable(monkeypatch):
    """An unavailable backend says it was skipped; [] would read as a database
    that answered and holds nothing."""
    monkeypatch.delenv("SCILINK_COD_DB", raising=False)
    b = CODBackend(db_path=None, cif_dir=None, allow_web=False)
    with pytest.raises(QuerySkipped):
        b.query(QuerySpec(chemistry=["Si"]))


def test_cod_web_failure_raises(monkeypatch):
    """A failed REST request raises (it used to read as an empty result)."""
    monkeypatch.delenv("SCILINK_COD_DB", raising=False)
    b = CODBackend(db_path=None, cif_dir=None, allow_web=True)

    from scilink.skills.structure_matching._backends import cod as cod_mod
    monkeypatch.setattr(cod_mod, "_WEB_BACKOFF_S", 0.0)
    calls = []

    def _down(*a, **k):
        calls.append(1)
        raise OSError("connection refused")
    monkeypatch.setattr("urllib.request.urlopen", _down)
    with pytest.raises(RuntimeError, match="COD REST search failed after 3 attempts"):
        b.query(QuerySpec(chemistry=["Si"]))
    assert len(calls) == cod_mod._WEB_ATTEMPTS


def test_cod_web_request_cut_short_is_retried(monkeypatch):
    """A response broken off mid-read (seen live on a large element query) is
    tried again, and the retry's answer is used."""
    import io, json as _json
    from http.client import IncompleteRead
    from scilink.skills.structure_matching._backends import cod as cod_mod
    monkeypatch.delenv("SCILINK_COD_DB", raising=False)
    monkeypatch.setattr(cod_mod, "_WEB_BACKOFF_S", 0.0)
    b = CODBackend(db_path=None, cif_dir=None, allow_web=True)
    monkeypatch.setattr(b, "_load_structure", lambda cid: _silicon())
    payload = _json.dumps([{"file": "1000001", "formula": "- Si -", "sg": "F d -3 m", "sgNumber": 227}]).encode()
    state = {"n": 0}

    class R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def flaky(*a, **k):
        state["n"] += 1
        if state["n"] == 1:
            raise IncompleteRead(b"partial")
        return R(payload)
    monkeypatch.setattr("urllib.request.urlopen", flaky)
    out = b.query(QuerySpec(chemistry=["Si"]))
    assert len(out) == 1 and state["n"] == 2


def _cod_web(monkeypatch, urlopen):
    from scilink.skills.structure_matching._backends import cod as cod_mod
    monkeypatch.delenv("SCILINK_COD_DB", raising=False)
    monkeypatch.setattr(cod_mod, "_WEB_BACKOFF_S", 0.0)
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    b = CODBackend(db_path=None, cif_dir=None, allow_web=True)
    monkeypatch.setattr(b, "_load_structure", lambda cid: _silicon())
    return b


def _cod_reply(obj):
    import io, json as _json

    class R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False
    return R(_json.dumps(obj).encode())


def test_cod_web_client_error_is_not_retried(monkeypatch):
    """A 4xx answer will not change on a retry: it raises at once."""
    import urllib.error
    calls = []

    def bad_request(*a, **k):
        calls.append(1)
        raise urllib.error.HTTPError("u", 400, "Bad Request", {}, None)
    b = _cod_web(monkeypatch, bad_request)
    with pytest.raises(RuntimeError, match=r"rejected \(HTTP 400\)"):
        b.query(QuerySpec(chemistry=["Si"]))
    assert len(calls) == 1


def test_cod_web_throttle_is_retried(monkeypatch):
    import urllib.error
    calls = []

    def throttled(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError("u", 429, "Too Many Requests", {}, None)
        return _cod_reply([{"file": "1000001", "formula": "- Si -", "sg": "F d -3 m", "sgNumber": 227}])
    b = _cod_web(monkeypatch, throttled)
    assert len(b.query(QuerySpec(chemistry=["Si"]))) == 1 and len(calls) == 2


def test_cod_web_payload_that_is_not_rows_raises(monkeypatch):
    """A reply that is not a list of rows is a failed search, not an empty one."""
    b = _cod_web(monkeypatch, lambda *a, **k: _cod_reply({"error": "busy"}))
    with pytest.raises(RuntimeError, match="not a list of rows"):
        b.query(QuerySpec(chemistry=["Si"]))


def test_cod_web_row_without_an_id_is_a_drop(monkeypatch):
    rows = [{"formula": "- Si -", "sg": "F d -3 m", "sgNumber": 227},
            {"file": "1000001", "formula": "- Si -", "sg": "F d -3 m", "sgNumber": 227}]
    b = _cod_web(monkeypatch, lambda *a, **k: _cod_reply(rows))
    assert len(b.query(QuerySpec(chemistry=["Si"]))) == 1
    assert b.last_report["dropped"] == 1 and "a row without a COD id" in b.last_report["drop_reasons"]


def test_cod_counts_matches_it_cannot_load(monkeypatch):
    """A row that matched but whose CIF cannot be loaded is a drop, reported."""
    monkeypatch.delenv("SCILINK_COD_DB", raising=False)
    b = CODBackend(db_path=None, cif_dir=None, allow_web=True)
    monkeypatch.setattr(b, "_search_web", lambda wanted, spec: [(1000001, "- Si -", "F d -3 m", 227)])
    monkeypatch.setattr(b, "_load_structure", lambda cid: None)
    assert b.query(QuerySpec(chemistry=["Si"])) == []
    assert b.last_report["dropped"] == 1 and b.last_report["drop_reasons"]


def test_cod_formula_elements_handles_fractional_counts():
    """COD formula strings carry fractional occupancy/Z counts (e.g. 'H145.17',
    'Na0.67'); the element parser must strip the numeric suffix including the
    decimal. Regression for the parser fix."""
    from scilink.skills.structure_matching._backends.cod import _formula_elements
    assert _formula_elements("- C6 H145.17 Au Cl4 N3 -") == {"C", "H", "Au", "Cl", "N"}
    assert _formula_elements("Na0.67 Mn O2") == {"Na", "Mn", "O"}


def test_cod_web_cap_is_reported_only_when_rows_are_cut(monkeypatch):
    import io, json as _json
    from scilink.skills.structure_matching._backends import cod as cod_mod
    monkeypatch.delenv("SCILINK_COD_DB", raising=False)
    monkeypatch.setattr(cod_mod, "_WEB_ROW_CAP", 2)

    def rows(n):
        payload = [{"file": str(1000000 + i), "formula": "- Si -", "sg": "F d -3 m", "sgNumber": 227}
                   for i in range(n)]
        class R(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False
        return lambda *a, **k: R(_json.dumps(payload).encode())

    b = CODBackend(db_path=None, cif_dir=None, allow_web=True)
    monkeypatch.setattr(b, "_load_structure", lambda cid: _silicon())
    monkeypatch.setattr("urllib.request.urlopen", rows(2))
    b.query(QuerySpec(chemistry=["Si"]))
    assert not b.last_report["truncated"]
    monkeypatch.setattr("urllib.request.urlopen", rows(3))
    b.query(QuerySpec(chemistry=["Si"]))
    assert b.last_report["truncated"] is True


def test_cod_db_cap_is_reported(monkeypatch, tmp_path):
    import sqlite3
    from scilink.skills.structure_matching._backends import cod as cod_mod
    db = tmp_path / "cod.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE data (file INTEGER, formula TEXT, sg TEXT, sgNumber INTEGER)")
    con.executemany("INSERT INTO data VALUES (?,?,?,?)",
                    [(1000000 + i, "- Si -", "F d -3 m", 227) for i in range(3)])
    con.commit(); con.close()
    monkeypatch.setattr(cod_mod, "_DB_ROW_CAP", 3)
    b = CODBackend(db_path=str(db), cif_dir=None, allow_web=False)
    monkeypatch.setattr(b, "_load_structure", lambda cid: _silicon())
    b.query(QuerySpec(chemistry=["Si"]))
    assert b.last_report["truncated"] is True
