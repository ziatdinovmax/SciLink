"""Offline tests: COD representative-entry selection, instrument presets and
geometry, bounded displacement parameters, and the guarded March-Dollase
preferred-orientation stage."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from scilink.skills.structure_matching._backends import cod as C
from scilink.skills.structure_matching._backends._base import QuerySpec
from scilink.skills.structure_matching.xrd import _gsas_engine as E
from scilink.skills.structure_matching.xrd.wavelength import resolve_geometry

_COLS = ("file INTEGER, formula TEXT, sg TEXT, sgNumber, vol REAL, year INTEGER, method TEXT, "
         "celltemp, cellpressure, status TEXT, doi TEXT, authors TEXT, title TEXT")


def _db(tmp_path, rows):
    db = tmp_path / "cod.db"
    con = sqlite3.connect(db)
    con.execute(f"CREATE TABLE data ({_COLS})")
    con.executemany(f"INSERT INTO data VALUES ({','.join('?' * 13)})", rows)
    con.commit(); con.close()
    return str(db)


def _row(cid, vol, year, sgn=154, method="\\N", authors="A", title="t", doi="\\N",
         celltemp="\\N", cellpressure="\\N", formula="- O2 Si -", sg="P 32 2 1"):
    return (cid, formula, sg, sgn, vol, year, method, celltemp, cellpressure, "\\N", doi, authors, title)


# Quartz-like group: ambient determinations from several papers, a theoretical
# model with the highest id, a high-pressure series from one paper (its own
# notes in the title) and entries whose sgNumber is COD's '\N' marker.
_QUARTZ = [
    _row(1011097, 113.0, 1935, authors="Wei"),
    _row(1538064, 112.98, 1974, authors="Le Page"),
    _row(2100188, 113.06, 2005, authors="Caspi", sgn="\\N"),
    _row(9013321, 113.007, 2008, authors="Antao", sgn="\\N"),
    _row(9000775, 113.13, 1980, authors="Levien", title="quartz at pressure Sample: P = 0.0001 GPa"),
    _row(9000776, 108.2, 1980, authors="Levien", title="quartz at pressure Sample: P = 2.0 GPa"),
    _row(9000777, 104.6, 1980, authors="Levien", title="quartz at pressure Sample: P = 4.0 GPa"),
    _row(9000778, 102.7, 1980, authors="Levien", title="quartz at pressure Sample: P = 5.0 GPa"),
    _row(9015022, 111.912, 1988, method="theoretical", authors="Model", sgn="\\N"),
]


def _search(tmp_path, rows, **spec):
    b = C.CODBackend(db_path=_db(tmp_path, rows), cif_dir=None, allow_web=False)
    b.last_report = {"truncated": False, "dropped": 0, "drop_reasons": []}
    return b._search_db({"Si", "O"}, QuerySpec(chemistry=["Si", "O"], **spec))


def test_missing_marker_does_not_split_a_phase(tmp_path):
    rows = _search(tmp_path, _QUARTZ)
    assert {r[3] for r in rows} == {154}                 # one phase, numbered
    assert rows[0][4]["group_entries"] == len(_QUARTZ)


def test_consensus_skips_theoretical_and_pressure_series(tmp_path):
    rows = _search(tmp_path, _QUARTZ)
    assert rows[0][0] == 9013321                       # ambient, newest of the cluster
    order = [r[0] for r in rows]
    assert order.index(9015022) > order.index(1011097)  # theoretical model ranked behind ambient


def test_newest_representative_is_still_available(tmp_path):
    rows = _search(tmp_path, _QUARTZ, representative="newest")
    assert rows[0][0] == 9015022


def test_series_counts_as_one_publication(tmp_path):
    # many entries of one variable-temperature paper must not outvote the
    # ambient cluster agreed by several independent papers
    rows = [_row(100 + i, 367.9 + 0.05 * i, 2000 + i, sgn=167, authors=f"P{i}",
                 formula="- C Ca O3 -", sg="R -3 c :H") for i in range(4)]
    rows += [_row(900 + i, 369.3 + 0.1 * i, 2010, sgn=167, authors="Series",
                  formula="- C Ca O3 -", sg="R -3 c :H") for i in range(10)]
    b = C.CODBackend(db_path=_db(tmp_path, rows), cif_dir=None, allow_web=False)
    b.last_report = {"truncated": False, "dropped": 0, "drop_reasons": []}
    got = b._search_db({"Ca", "C", "O"}, QuerySpec(chemistry=["Ca", "C", "O"]))
    assert got[0][0] in range(100, 104)


def test_space_group_hint_keeps_entries_with_missing_number(tmp_path):
    rows = _search(tmp_path, _QUARTZ, space_group_hints=[154])
    assert 9013321 in [r[0] for r in rows]


def test_title_conditions_and_symbol_numbers():
    assert C._non_ambient({"title": "x Sample: P = 2.5 GPa"})
    assert not C._non_ambient({"title": "x Sample: P = 0.0001 GPa"})
    assert C._non_ambient({"title": "thermal expansion T = 600 C"})
    assert not C._non_ambient({"celltemp": 295.0})
    assert C._sg_number_from_symbol("P 32 2 1") == 154
    assert C._sg_number_from_symbol("R -3 c :H") == 167
    assert C._sg_number_from_symbol("I 1 2/c 1") == 15
    assert C._sg_number("P 63 m c", "\\N") == 186


def test_group_rank_uses_its_best_composition_match():
    rows = [(0, 5.0, 10, "- H2 Mg O2 -", "P -3 m 1", 164, {"vol": 40.7, "year": 1967}),
            (1, 5.0, 11, "- D H Mg O2 -", "P -3 m 1", 164, {"vol": 40.8, "year": 2020}),
            (0, 5.0, 12, "- H2 Mg O2 -", "P -3 m 1", 164, {"vol": 40.6, "year": 1990}),
            (0, 7.0, 20, "- H2 Mg2 O3 -", "P -3 m 1", 164, {"vol": 60.0, "year": 2000})]
    got = C._rank_rows(rows)
    assert got[0][1] == "- H2 Mg O2 -"            # not demoted by its deuterated entry


# ------------------------------------------------------------------ engine
def test_instrument_presets_and_geometry():
    assert E._is_lab_line("CuKa") and E._is_lab_line(1.54059) and E._is_lab_line("MoKa1")
    assert not E._is_lab_line(0.49591)
    assert E._resolve_instrument_preset(0.49591) == "synchrotron"
    assert E._resolve_instrument_preset("CuKa") == "lab"
    assert E._resolve_geometry(1.54059) == "bragg-brentano"
    assert E._resolve_geometry(0.49591) == "debye-scherrer"
    assert E._resolve_geometry(1.54, "capillary") == "debye-scherrer"
    with pytest.raises(ValueError):
        E._resolve_geometry(1.54, "sideways")
    syn = E._instprm(0.49591, None, "debye-scherrer")
    assert "Polariz.:0.99" in syn and "W:0.063" in syn and "Diff-type:Debye-Scherrer" in syn
    lab = E._instprm("CuKa", None, "bragg-brentano")
    assert "Lam1:" in lab and "W:5.0" in lab and "Diff-type:Bragg-Brentano" in lab
    assert "Diff-type" not in E._instprm("CuKa")          # unchanged when not asked


def test_resolve_geometry_from_metadata():
    assert resolve_geometry({"instrument": "Bragg-Brentano reflection geometry, spinning sample"}) \
        == "bragg-brentano"
    assert resolve_geometry({"instrument": "synchrotron, Debye-Scherrer transmission, 0.7 mm "
                                           "glass capillary"}) == "debye-scherrer"
    assert resolve_geometry({"instrument": "PANalytical X'Pert, reflection (theta/2-theta)"}) \
        == "bragg-brentano"
    assert resolve_geometry({"instrument": "D8 Advance"}) is None
    assert resolve_geometry(None) is None


class _Phase:
    def __init__(self, atoms):
        # AtomPtrs (cx, ct, cs, cia): type at 1, U flag at 9, Uiso at 10
        self.data = {"General": {"AtomPtrs": (3, 1, 7, 9)},
                     "Atoms": [[f"{el}{i}", el, "", 0, 0, 0, 1.0, "1", 0, "I", 0.01]
                               for i, el in enumerate(atoms)]}


_CIF = """data_x
_cell_length_a 4.99
_cell_length_b 4.99
_cell_length_c 17.06
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 120
_symmetry_space_group_name_H-M 'R -3 c'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_U_iso_or_equiv
Ca1 Ca 0 0 0 0.0345
C1 C 0 0 0.25 0.0060
O1 O 0.257 0 0.25 0.0452
"""


def test_cif_displacements_are_bounded(tmp_path):
    cif = tmp_path / "x.cif"
    cif.write_text(_CIF)
    ph, replaced = _Phase(["Ca", "C", "O"]), {}
    used = E._apply_displacements(ph, str(cif), "cif", replaced)
    assert used == {"C": 0.006} and set(replaced) == {"Ca", "O"}
    assert [a[10] for a in ph.data["Atoms"]] == pytest.approx([0.01, 0.006, 0.01])
    raw = E._apply_displacements(_Phase(["Ca", "C", "O"]), str(cif), "cif_raw")
    assert raw["Ca"] == pytest.approx(0.0345)
    with pytest.raises(ValueError):
        E._apply_displacements(_Phase(["Ca"]), str(cif), "cifx")


# ------------------------------------------------- preferred orientation
class _PoPhase:
    def __init__(self, name):
        self.name = name
        self.data = {"Histograms": {"h": {"Scale": [1.0, True],
                                          "Pref.Ori.": ["MD", 1.0, False, [0, 0, 1], 0, {}, [], 0.1]}},
                     "General": {"Cell": [False, 5.0, 5.0, 5.0, 90, 90, 90, 125.0]}}

    def getHAPvalues(self, hist):
        return self.data["Histograms"][hist.name]


class _PoHist:
    name = "h"

    def __init__(self):
        self.data = {"Sample Parameters": {"Scale": [1.0, True]}, "Background": [], "Instrument Parameters": [{}],
                     "Reflection Lists": {}}
        self.residuals = {"wR": 30.0}


class _PoProject:
    """do_refinements: Rwp by the refined PO axis of phase 'a'. Axis [0,1,0]
    fits (MD 0.6); axis [1,0,0] DIVERGES and wrecks the phase scale."""

    def __init__(self, phases, hist):
        self.phases, self.hist = phases, hist

    def do_refinements(self, _):
        po = self.phases[0].getHAPvalues(self.hist)["Pref.Ori."]
        if po[2] and po[3] == [0, 1, 0]:
            po[1] = 0.6; self.hist.residuals = {"wR": 20.0}
        elif po[2] and po[3] == [1, 0, 0]:
            po[1] = 1.0; self.hist.residuals = {"wR": 100.0}
            self.phases[0].getHAPvalues(self.hist)["Scale"][0] = 1e22
        else:
            self.hist.residuals = {"wR": 30.0 if self.phases[0].getHAPvalues(self.hist)["Scale"][0] < 1e3
                                   else 100.0}


def test_preferred_orientation_auto_keeps_best_and_survives_divergence():
    a, b = _PoPhase("a"), _PoPhase("b")
    h = _PoHist()
    gpx = _PoProject([a, b], h)
    rwp = lambda: float(h.residuals["wR"])  # noqa: E731
    trace = []
    out = E._preferred_orientation_stage(gpx, [a, b], h, "auto", rwp, trace,
                                         fractions={"a": 0.6, "b": 0.4})
    assert out["a"]["axis"] == [0, 1, 0] and out["a"]["MD"] == pytest.approx(0.6)
    assert a.getHAPvalues(h)["Scale"][0] == 1.0          # the diverged trial was undone
    assert any(t.get("Rwp") == 100.0 for t in out["a"]["tried"])
    assert out["b"]["axis"] is None                      # no gain on b: nothing kept
    assert b.getHAPvalues(h)["Pref.Ori."][2] is False
    assert rwp() == 20.0


def test_tool_specs_document_new_parameters():
    from scilink.skills.structure_matching.xrd.refine_multiphase import _SPEC_MULTI, _SPEC_SERIES
    from scilink.skills.structure_matching.xrd.refine_rietveld import TOOL_SPEC
    for spec in (_SPEC_MULTI, _SPEC_SERIES, TOOL_SPEC):
        for k in ("geometry", "instrument_preset", "sample_displacement", "preferred_orientation"):
            assert k in spec.parameters and k in spec.signature
    from scilink.skills.structure_matching.xrd import wavelength
    assert any(s.name == "resolve_geometry" for s in wavelength.TOOL_SPECS)
