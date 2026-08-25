"""Tests for quantify_phases_nnls — NNLS phase quantification over a shortlist.

Builds a synthetic two-phase mixture from two structurally DISTINCT phases
(rock-salt NaCl + wurtzite ZnO — different symmetry, non-overlapping peaks), so
the planted fractions are known and a missing phase leaves clear residual.
Checks: (1) NNLS recovers the mixture fractions, (2) it ABSTAINS
(reliable=False) when a phase is missing from the candidates, (3) the
min_fraction knob prunes a minor phase.

Requires pymatgen + scipy; skipped otherwise.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pymatgen")
pytest.importorskip("scipy")


@pytest.fixture(scope="module")
def cifs(tmp_path_factory):
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.cif import CifWriter
    d = tmp_path_factory.mktemp("cifs")
    nacl = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.64),
                                     ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    zno = Structure.from_spacegroup(  # wurtzite — hexagonal, distinct peaks
        "P6_3mc", Lattice.hexagonal(3.25, 5.207),
        ["Zn", "O"], [[1/3, 2/3, 0.0], [1/3, 2/3, 0.382]])
    pa, pb = d / "NaCl.cif", d / "ZnO.cif"
    CifWriter(nacl).write_file(str(pa)); CifWriter(zno).write_file(str(pb))
    return str(pa), str(pb)


def _mix(cifs, cA, cB, grid, fwhm=0.15, eta=0.5, noise=2.0, bg=40.0):
    from scilink.skills.structure_matching.xrd.quantify_nnls import _profile_column
    colA = _profile_column(cifs[0], grid, "CuKa", fwhm, eta)
    colB = _profile_column(cifs[1], grid, "CuKa", fwhm, eta)
    rng = np.random.default_rng(0)
    y = cA * colA + cB * colB + bg + 0.2 * (grid - grid[0])   # sloping background
    y = y + rng.normal(0, noise, grid.size)
    # expected INTENSITY fractions = integrated-contribution shares
    trapz = getattr(np, "trapezoid", None) or np.trapz
    iA, iB = cA * trapz(colA, grid), cB * trapz(colB, grid)
    return y, iA / (iA + iB)


def test_recovers_two_phase_fractions(cifs):
    grid = np.arange(15.0, 80.0, 0.02)
    y, exp_fracA = _mix(cifs, 0.7, 0.3, grid)
    from scilink.skills.structure_matching.xrd.quantify_nnls import quantify_phases_nnls
    r = quantify_phases_nnls(list(grid), list(y), list(cifs), fwhm_deg=0.15)
    assert r["reliable"], r
    assert len(r["phases"]) == 2
    by = {p["structure_path"]: p["intensity_fraction"] for p in r["phases"]}
    got_A = by[cifs[0]]
    assert abs(got_A - exp_fracA) < 0.06, (got_A, exp_fracA)   # recovered the mix
    assert abs(sum(p["intensity_fraction"] for p in r["phases"]) - 1.0) < 1e-3


def test_abstains_when_a_phase_is_missing(cifs):
    # two-phase data, but only ONE candidate offered -> residual >> noise
    grid = np.arange(15.0, 80.0, 0.02)
    y, _ = _mix(cifs, 0.5, 0.5, grid)
    from scilink.skills.structure_matching.xrd.quantify_nnls import quantify_phases_nnls
    r = quantify_phases_nnls(list(grid), list(y), [cifs[0]], fwhm_deg=0.15)
    assert r["reliable"] is False
    assert "missing" in r["note"].lower() or "off-database" in r["note"].lower()
    assert r["residual_over_noise"] > 6.0


def test_min_fraction_prunes_trace_phase(cifs):
    grid = np.arange(15.0, 80.0, 0.02)
    y, _ = _mix(cifs, 0.97, 0.03, grid)      # CaF2 a minor phase
    from scilink.skills.structure_matching.xrd.quantify_nnls import quantify_phases_nnls
    lax = quantify_phases_nnls(list(grid), list(y), list(cifs), min_fraction=0.001)
    assert len(lax["phases"]) == 2                 # keeps the minor phase
    minor = min(p["intensity_fraction"] for p in lax["phases"])
    strict = quantify_phases_nnls(list(grid), list(y), list(cifs),
                                  min_fraction=minor + 0.03)   # threshold just above it
    assert len(strict["phases"]) == 1              # prunes it
    assert strict["phases"][0]["structure_path"] == cifs[0]


def test_tool_spec_registered():
    from scilink.skills.structure_matching.xrd.quantify_nnls import TOOL_SPEC
    assert TOOL_SPEC.name == "quantify_phases_nnls"
    # every meaningful knob is exposed to the LLM
    for knob in ("fwhm_deg", "background", "min_fraction", "residual_over_noise_max"):
        assert knob in TOOL_SPEC.parameters
