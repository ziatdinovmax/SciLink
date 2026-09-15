"""Offline tests for identify_mixture (sequential-subtraction mixture ID).

Deterministic, no LLM, no network: a tiny library is built from local CIFs and
synthetic mixture peak lists are composed from the library's OWN stored lines
at known intensity fractions — a machinery test (loop, subtraction, shared-line
guard, confirmation), not a discrimination benchmark (that is the RRUFF eval).

Physics note baked into the fixtures: NaCl(111) at 27.37 and rutile(110) at
27.45 2θ (CuKa) genuinely coincide within the default 0.3 tolerance — the
NaCl+rutile mix therefore exercises the shared-line survival path for real.
"""

from __future__ import annotations

import numpy as np
import pytest

pymatgen = pytest.importorskip("pymatgen.core")
from pymatgen.core import Lattice, Structure  # noqa: E402

from scilink.skills.structure_matching.xrd.fingerprint import (  # noqa: E402
    build_fingerprint_library)
from scilink.skills.structure_matching.xrd.identify_mixture import (  # noqa: E402
    identify_mixture, TOOL_SPEC as IM_SPEC)

LAM = 1.5406


def _write(structure, path):
    path.write_text(structure.to(fmt="cif"))


@pytest.fixture()
def tiny_library(tmp_path):
    cifs = tmp_path / "cifs"
    cifs.mkdir()
    _write(Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43088), ["Si"], [[0, 0, 0]]),
           cifs / "si.cif")
    _write(Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.6402), ["Na", "Cl"],
                                     [[0, 0, 0], [0.5, 0.5, 0.5]]), cifs / "nacl.cif")
    _write(Structure.from_spacegroup("P4_2/mnm", Lattice.tetragonal(4.594, 2.959),
                                     ["Ti", "O"], [[0, 0, 0], [0.305, 0.305, 0]]),
           cifs / "rutile.cif")
    out = tmp_path / "lib.parquet"
    summary = build_fingerprint_library(str(cifs), str(out))
    assert summary["n_indexed"] == 3
    return str(out)


def _entry_lines(library_path, formula):
    import pandas as pd
    row = pd.read_parquet(library_path).set_index("formula").loc[formula]
    d = np.asarray(row["ds"], dtype=float)
    tt = 2.0 * np.degrees(np.arcsin(np.clip(LAM / (2.0 * d), -1, 1)))
    return tt, np.asarray(row["intensities"], dtype=float)


def _mix(library_path, fractions, merge_tol=0.1):
    """Compose a mixture peak list: each phase's lines scaled by its fraction,
    coincident lines (within merge_tol) merged by intensity sum — what a real
    detector reports for overlapping reflections."""
    tt_all, ii_all = [], []
    for formula, frac in fractions.items():
        tt, ii = _entry_lines(library_path, formula)
        tt_all.extend(tt.tolist())
        ii_all.extend((frac * ii).tolist())
    order = np.argsort(tt_all)
    tt_s = np.asarray(tt_all)[order]
    ii_s = np.asarray(ii_all)[order]
    m_tt, m_ii = [tt_s[0]], [ii_s[0]]
    for t, i in zip(tt_s[1:], ii_s[1:]):
        if t - m_tt[-1] <= merge_tol:
            m_ii[-1] += i          # coincident reflections superpose
        else:
            m_tt.append(t)
            m_ii.append(i)
    return list(m_tt), list(m_ii)


def test_tool_registered_with_knobs():
    from scilink.skills._shared._registry import get_tools_for
    names = {t.name for t in get_tools_for("structure_matching", active_skills=["xrd"])}
    assert "identify_mixture" in names
    for knob in ("max_phases", "fom_threshold", "tol_deg", "min_residual_peaks",
                 "shared_line_floor", "confirm"):
        assert knob in IM_SPEC.parameters, f"knob {knob} undocumented"


def test_single_phase_yields_one_phase(tiny_library):
    tt, ii = _mix(tiny_library, {"Si": 1.0})
    r = identify_mixture(tt, ii, library_path=tiny_library, confirm=False)
    assert r["n_phases"] == 1
    assert r["phases"][0]["formula"] == "Si"
    assert "no convincing match" in r["stopped_because"] \
        or "residual peaks" in r["stopped_because"]
    assert r["residual_intensity_frac"] < 0.15


def test_two_phase_mix_finds_both_dominant_first(tiny_library, tmp_path):
    tt, ii = _mix(tiny_library, {"Si": 0.7, "NaCl": 0.3})
    r = identify_mixture(tt, ii, library_path=tiny_library,
                         materialize_dir=str(tmp_path / "m"))
    forms = [p["formula"] for p in r["phases"]]
    assert "Si" in forms and "NaCl" in forms
    assert forms[0] == "Si"                          # dominant discovered first
    shares = {p["formula"]: p["intensity_share"] for p in r["phases"]}
    assert shares["Si"] > shares["NaCl"]             # proxy ordering holds
    # structure bridge: locally built library resolves its own CIFs
    assert all(p["structure_path"] for p in r["phases"])
    # joint confirmation keeps both phases active
    conf = r["multiphase_confirmation"]
    if conf is not None:                             # pulp present in dev env
        active = {p["formula"] for p in conf["active_phases"]}
        assert {"Si", "NaCl"} <= active
        assert conf["verdict"] in ("accept", "marginal")
        assert all(p["confirmed"] for p in r["phases"])


def test_shared_line_survives_subtraction(tiny_library, tmp_path):
    # NaCl(111)@27.37 and rutile(110)@27.45 merge into ONE measured peak; hard
    # removal after accepting the first phase would delete the second phase's
    # strongest evidence. The shared-line floor keeps the unexplained
    # remainder, so both phases must be found.
    tt, ii = _mix(tiny_library, {"NaCl": 0.55, "TiO2": 0.45})
    assert any(27.3 < t < 27.5 for t in tt)          # the merged shared line exists
    r = identify_mixture(tt, ii, library_path=tiny_library,
                         materialize_dir=str(tmp_path / "m"), confirm=False)
    forms = [p["formula"] for p in r["phases"]]
    assert "NaCl" in forms and "TiO2" in forms


def test_three_phase_mix(tiny_library, tmp_path):
    tt, ii = _mix(tiny_library, {"Si": 0.45, "NaCl": 0.35, "TiO2": 0.20})
    r = identify_mixture(tt, ii, library_path=tiny_library,
                         materialize_dir=str(tmp_path / "m"), confirm=False)
    forms = {p["formula"] for p in r["phases"]}
    assert forms == {"Si", "NaCl", "TiO2"}
    assert r["residual_intensity_frac"] < 0.25


def test_fom_threshold_knob_stops_loop(tiny_library):
    tt, ii = _mix(tiny_library, {"Si": 0.7, "NaCl": 0.3})
    strict = identify_mixture(tt, ii, library_path=tiny_library,
                              fom_threshold=0.99, confirm=False)
    lax = identify_mixture(tt, ii, library_path=tiny_library, confirm=False)
    assert strict["n_phases"] < lax["n_phases"]      # knob actually gates
    assert "threshold" in strict["stopped_because"]


def test_missing_phase_leaves_residual(tiny_library, tmp_path):
    # Si + a phase NOT in the library (synthetic lines): Si is identified, the
    # foreign lines survive as residual with substantial intensity — the
    # signal to fall back to indexing.
    tt_si, ii_si = _entry_lines(tiny_library, "Si")
    foreign_tt = [21.5, 34.2, 43.9, 62.7]
    foreign_ii = [40.0, 30.0, 20.0, 15.0]
    tt = tt_si.tolist() + foreign_tt
    ii = (0.7 * ii_si).tolist() + foreign_ii
    r = identify_mixture(tt, ii, library_path=tiny_library, confirm=False,
                         materialize_dir=str(tmp_path / "m"))
    forms = [p["formula"] for p in r["phases"]]
    assert forms[0] == "Si"
    left = r["residual_peaks"]["positions"]
    assert all(any(abs(f - p) < 0.15 for p in left) for f in foreign_tt)
    assert r["residual_intensity_frac"] > 0.2


def test_multiphase_scale_fit_keeps_minority_phase(tiny_library):
    # Regression lock for the per-phase lattice-scale pre-fit: exp-intensity
    # weighting let a 70/30 mixture's MINORITY phase (NaCl) be stretched 3.8%
    # onto the majority phase's strong lines, deactivating it in the joint
    # MILP despite 9 exact matches at scale 1.0. The mixture pre-fit now
    # weights by the candidate's OWN line intensities (fraction-independent).
    pytest.importorskip("pulp")
    from scilink.skills.structure_matching.xrd.score_match_robust import (
        score_xrd_match_multiphase)
    tt, ii = _mix(tiny_library, {"Si": 0.7, "NaCl": 0.3})
    cands = []
    for f in ("Si", "NaCl"):
        ctt, cii = _entry_lines(tiny_library, f)
        cands.append({"id": f, "formula": f, "sim_two_theta": ctt.tolist(),
                      "sim_intensity": cii.tolist()})
    r = score_xrd_match_multiphase(
        exp_peaks={"positions": tt, "intensities": ii}, candidates=cands)
    active = {p["formula"]: p for p in r["active_phases"]}
    assert set(active) == {"Si", "NaCl"}
    assert abs(active["NaCl"]["lattice_scale"] - 1.0) < 1e-6   # not stretched
    assert r["verdict"] == "accept"


def test_input_guards(tiny_library):
    with pytest.raises(ValueError):
        identify_mixture([28.4, 47.3], [100.0, 55.0], library_path=tiny_library)
    with pytest.raises(ValueError):
        identify_mixture([28.4, 47.3, 56.1], [100.0, 55.0],
                         library_path=tiny_library)
