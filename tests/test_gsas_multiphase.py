"""Tests for multi-phase / sequential Rietveld. Registry + spec assertions run
without GSAS-II; the physics round-trips are gated on ``gsas_available()``:
forward-simulate a two-phase (and 3-frame series) pattern with SET phase
fractions through GSAS itself, then recover fractions/cells with a fresh
refinement — ground truth by construction in GSAS's own weight-fraction
convention."""

from __future__ import annotations

import os

import numpy as np
import pytest

from scilink.skills.structure_matching.xrd import _gsas_engine as ge


def _pymatgen_available():
    try:
        import pymatgen.core  # noqa: F401
        return True
    except Exception:
        return False


needs_gsas = pytest.mark.skipif(
    not (ge.gsas_available() and _pymatgen_available()),
    reason="GSAS-II (gsas extra) and pymatgen required")


def test_tools_registered_with_knobs():
    from scilink.skills._shared._registry import get_tools_for
    names = {t.name for t in get_tools_for("structure_matching", active_skills=["xrd"])}
    assert "refine_rietveld_multiphase" in names
    assert "refine_rietveld_series" in names
    from scilink.skills.structure_matching.xrd.refine_multiphase import TOOL_SPECS
    multi, series = TOOL_SPECS
    for knob in ("refine_cell", "refine_profile", "n_background_terms",
                 "two_theta_range"):
        assert knob in multi.parameters
    assert "warm_start" in series.parameters


def test_canonicalize_strips_oxidation_states(tmp_path):
    # A CIF declaring ionic species (Ca2+/F-, ubiquitous in COD) sends GSAS-II
    # down a charged-scattering-factor path with a catastrophically wrong
    # intensity model (pure-fluorite Rietveld corr 0.73 vs 0.97; mixture
    # fractions collapse to 0). The canonicalizer must emit neutral species.
    pytest.importorskip("pymatgen.core")
    from pymatgen.core import Lattice, Structure
    s = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.4631),
                                  ["Ca2+", "F-"], [[0, 0, 0], [0.25, 0.25, 0.25]])
    src = tmp_path / "charged.cif"
    src.write_text(s.to(fmt="cif"))
    assert "Ca2+" in src.read_text()               # fixture really is charged
    out = ge._canonicalize_cif(str(src), str(tmp_path), symprec=0.1)
    txt = open(out).read()
    assert "oxidation_number" not in txt
    assert "Ca2+" not in txt and "F-" not in txt


def test_input_guards():
    with pytest.raises((ValueError, RuntimeError)):
        ge.rietveld_refine_multiphase(["one.cif"], [10, 11], [1, 2])   # <2 phases
    with pytest.raises((ValueError, RuntimeError)):
        ge.rietveld_refine_series([], [{"two_theta": [1], "intensity": [1]}])


@pytest.fixture(scope="module")
def two_phase_cifs(tmp_path_factory):
    from pymatgen.core import Lattice, Structure
    d = tmp_path_factory.mktemp("cifs")
    si = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43088), ["Si"], [[0, 0, 0]])
    na = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.6402), ["Na", "Cl"],
                                   [[0, 0, 0], [0.5, 0.5, 0.5]])
    sp = d / "si.cif"; sp.write_text(si.to(fmt="cif"))
    np_ = d / "nacl.cif"; np_.write_text(na.to(fmt="cif"))
    return str(sp), str(np_)


def _forward_two_phase(si_cif, na_cif, frac_na, workdir, a_si=None, seed=0):
    """Simulate a two-phase pattern with SET HAP fractions; return x, y(noisy),
    and the ground-truth weight fractions."""
    from pymatgen.core import Lattice, Structure
    G2sc = ge._load_g2sc()
    if a_si is not None:
        si = Structure.from_spacegroup("Fd-3m", Lattice.cubic(a_si), ["Si"], [[0, 0, 0]])
        si_cif = os.path.join(workdir, f"si_{a_si}.cif")
        open(si_cif, "w").write(si.to(fmt="cif"))
    instf = os.path.join(workdir, "auto.instprm")
    open(instf, "w").write(ge._INSTPRM_TEMPLATE.format(lam=1.5406))
    d1 = os.path.join(workdir, "p1"); d2 = os.path.join(workdir, "p2")
    os.makedirs(d1, exist_ok=True); os.makedirs(d2, exist_ok=True)
    gpx = G2sc.G2Project(newgpx=os.path.join(workdir, "f.gpx"))
    ps = gpx.add_phase(ge._canonicalize_cif(si_cif, d1, symprec=0.1), phasename="si")
    pn = gpx.add_phase(ge._canonicalize_cif(na_cif, d2, symprec=0.1), phasename="nacl")
    hist = gpx.add_simulated_powder_histogram(histname="sim", iparams=instf,
                                              Tmin=15.0, Tmax=80.0, Tstep=0.02,
                                              phases=[ps, pn])
    for ph in (ps, pn):
        gpx.link_histogram_phase(hist, ph)
        ph.setSampleProfile(hist, "size", "isotropic", 1.0)
    ps.getHAPvalues(hist)["Scale"][0] = 1.0 - frac_na
    pn.getHAPvalues(hist)["Scale"][0] = frac_na
    gpx.do_refinements([{}])
    x = np.asarray(hist.getdata("X"), float)
    y = np.asarray(hist.getdata("Ycalc"), float)
    y = y * (5e4 / max(y.max(), 1e-9)) + 60.0
    rng = np.random.default_rng(seed)
    y = rng.poisson(np.maximum(y, 1.0)).astype(float)
    m_si = ge._phase_cell_mass(ps)
    m_na = ge._phase_cell_mass(pn)
    tot = (1.0 - frac_na) * m_si + frac_na * m_na
    truth = {"si": (1.0 - frac_na) * m_si / tot, "nacl": frac_na * m_na / tot}
    return x, y, truth


@needs_gsas
def test_multiphase_roundtrip(two_phase_cifs, tmp_path):
    si_cif, na_cif = two_phase_cifs
    x, y, truth = _forward_two_phase(si_cif, na_cif, 0.55, str(tmp_path))
    r = ge.rietveld_refine_multiphase([si_cif, na_cif], x, y)
    assert r["converged"]
    wf = list(r["weight_fractions"].values())
    # phase order follows structure_paths: si first
    assert abs(wf[0] - truth["si"]) < 0.02
    assert abs(wf[1] - truth["nacl"]) < 0.02
    a_si = r["phases"][0]["lattice"]["length_a"]
    a_na = r["phases"][1]["lattice"]["length_a"]
    assert abs(a_si - 5.43088) < 5e-4
    assert abs(a_na - 5.6402) < 5e-4
    assert r["profile_corr"] > 0.98


@needs_gsas
def test_series_roundtrip_fractions_and_expansion(two_phase_cifs, tmp_path):
    si_cif, na_cif = two_phase_cifs
    FRACS = [0.10, 0.45, 0.80]
    A_SI = [5.4309, 5.4340, 5.4370]
    frames, truths = [], []
    for k, (f, a) in enumerate(zip(FRACS, A_SI)):
        d = tmp_path / f"fwd{k}"; d.mkdir()
        x, y, truth = _forward_two_phase(si_cif, na_cif, f, str(d), a_si=a, seed=k)
        frames.append({"two_theta": x.tolist(), "intensity": y.tolist(),
                       "label": 300 + 10 * k})
        truths.append(truth)
    r = ge.rietveld_refine_series([si_cif, na_cif], frames)
    assert len(r["frames"]) == 3
    names = r["phase_names"]
    for k, fr in enumerate(r["frames"]):
        assert fr["converged"], f"frame {k} did not converge"
        assert abs(fr["weight_fractions"][names[1]] - truths[k]["nacl"]) < 0.03
        assert fr["profile_corr"] > 0.95
        # per-frame refined Si cell tracks the programmed thermal expansion
        # (2e-3 Å = 0.04%: the warm-started local refinement lands within a
        # milliangstrom-scale of truth; the deliverable is the trend)
        a_eff = fr["cells"][names[0]]["length_a"]
        assert abs(a_eff - A_SI[k]) < 2e-3
    # expansion is monotone across frames
    a_series = [fr["cells"][names[0]]["length_a"] for fr in r["frames"]]
    assert a_series[0] < a_series[1] < a_series[2]
