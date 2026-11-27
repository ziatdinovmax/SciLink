"""Offline tests for composition-free identification aids in the fingerprint
route: per-match prevalence (``n_library_reports``), the ``organic_like`` flag,
``exclude_organic`` and the prevalence near-tie rule."""

from __future__ import annotations

import pytest

pymatgen = pytest.importorskip("pymatgen.core")
pd = pytest.importorskip("pandas")
from pymatgen.core import Lattice, Structure  # noqa: E402

from scilink.skills.structure_matching.xrd import fingerprint as fp  # noqa: E402
from scilink.skills.structure_matching.xrd.identify_mixture import (  # noqa: E402
    TOOL_SPEC as IM_SPEC)

SI_TT = [28.442, 47.303, 56.121, 69.130, 76.377, 88.032]
SI_II = [100.0, 55.0, 30.0, 6.0, 11.0, 12.0]


@pytest.mark.parametrize("formula,expected", [
    ("CaCO3", False), ("CaMg(CO3)2", False), ("Cu2CO3(OH)2", False),
    ("Mg5(CO3)4(OH)2(H2O)4", False), ("Li2CO3", False),
    ("Cr23C6", False), ("V6C5", False), ("Al4C3", False), ("SiC", False),
    ("Fe3C", False), ("C", False), ("KCN", False), ("LaO3", False),
    ("C10H8", True), ("CaC2O4(H2O)", True), ("B(HC3)3", True),
    ("TmC33N6O7", True), ("Co3C10O9", True),   # organometallics without H
])
def test_organic_like(formula, expected):
    assert fp._organic_like(formula) is expected


def test_organic_like_unparsable_is_none():
    assert fp._organic_like("not a formula!!") is None


def test_prevalence_tiebreak_orders_within_band_only():
    m = [
        {"id": "rare", "adjusted_score": 0.93, "n_library_reports": 1},
        {"id": "common", "adjusted_score": 0.90, "n_library_reports": 12},
        {"id": "far", "adjusted_score": 0.80, "n_library_reports": 50},
    ]
    out = [x["id"] for x in fp._prevalence_tiebreak(list(m), 0.05)]
    assert out == ["common", "rare", "far"]          # 'far' is outside the band
    assert [x["id"] for x in fp._prevalence_tiebreak(list(m), 0.0)] == \
        ["rare", "common", "far"]                    # band 0 = pure score order
    out2 = [x["id"] for x in fp._prevalence_tiebreak(list(m), 0.01)]
    assert out2 == ["rare", "common", "far"]         # common not within 0.01


@pytest.fixture()
def lib_with_twins(tmp_path):
    cifs = tmp_path / "cifs"
    cifs.mkdir()
    (cifs / "si.cif").write_text(Structure.from_spacegroup(
        "Fd-3m", Lattice.cubic(5.43088), ["Si"], [[0, 0, 0]]).to(fmt="cif"))
    (cifs / "nacl.cif").write_text(Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.6402), ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]]).to(fmt="cif"))
    out = tmp_path / "lib.parquet"
    fp.build_fingerprint_library(str(cifs), str(out))
    df = pd.read_parquet(out)
    si = dict(df[df["formula"] == "Si"].iloc[0])
    si2 = dict(si, source_id="si-second-report", a=5.4310)     # second Si report
    exotic = dict(si, source_id="exotic", formula="Zn2Cu3Sn2S8", sg_number=216)
    organic = dict(si, source_id="organic", formula="C10H8", sg_number=14)
    lib = tmp_path / "twins.parquet"
    pd.concat([df, pd.DataFrame([si2, exotic, organic])],
              ignore_index=True).to_parquet(lib)
    return str(lib)


def test_search_reports_prevalence_and_organic_flag(lib_with_twins):
    r = fp.search_match_pattern(SI_TT, SI_II, library_path=lib_with_twins,
                                materialize_top=0)
    by_id = {m["source_id"]: m for m in r["matches"]}
    assert by_id["exotic"]["n_library_reports"] == 1
    assert by_id["organic"]["organic_like"] is True
    assert by_id["exotic"]["organic_like"] is False
    si_hits = [m for m in r["matches"] if m["formula"] == "Si"]
    assert si_hits and all(m["n_library_reports"] == 2 for m in si_hits)
    # identical lines tie on score: the default near-tie rule puts the
    # multiply-reported phase first
    assert r["matches"][0]["formula"] == "Si"
    assert r["n_excluded_organic"] == 0


def test_exclude_organic_drops_flagged_entries(lib_with_twins):
    r = fp.search_match_pattern(SI_TT, SI_II, library_path=lib_with_twins,
                                materialize_top=0, exclude_organic=True)
    assert "organic" not in {m["source_id"] for m in r["matches"]}
    assert r["n_excluded_organic"] == 1
    assert r["n_library_entries"] == 5               # counts the full library


def test_new_knobs_documented():
    for k in ("exclude_organic", "prevalence_tiebreak_band"):
        assert k in fp.TOOL_SPEC.parameters
        assert k in IM_SPEC.parameters
        assert k in fp.TOOL_SPEC.signature
        assert k in IM_SPEC.signature
