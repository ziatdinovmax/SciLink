"""Offline tests: weight-fraction esds propagated from the refinement
covariance, and the record metadata reaching the script-generation prompt."""

from __future__ import annotations

import numpy as np
import pytest

from scilink.skills.structure_matching.xrd import _gsas_engine as E
from scilink.skills.structure_matching.xrd.refine_multiphase import _SPEC_MULTI


class _Ph:
    def __init__(self, pid, name, f, mass):
        self.id, self.name, self._f, self.mass = pid, name, f, mass

    def getHAPvalues(self, hist):
        return {"Scale": [self._f, True]}


class _Hist:
    id = 0


def _wt(f, M):
    return f * M / np.sum(f * M)


@pytest.fixture()
def three_phases(monkeypatch):
    monkeypatch.setattr(E, "_phase_cell_mass", lambda ph: ph.mass)
    phases = [_Ph(0, "a", 1.0, 120.0), _Ph(1, "b", 0.4, 300.0), _Ph(2, "c", 2.5, 80.0)]
    # phase 0 fixed (absorbed by the histogram scale); 1 and 2 refined
    vary = [":0:Scale", "1:0:Scale", "2:0:Scale", ":0:Back;0"]
    cov = np.array([[4e-4, 1e-5, 2e-5, 0.0],
                    [1e-5, 9e-4, 3e-4, 0.0],
                    [2e-5, 3e-4, 1.6e-3, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])
    gpx = {"Covariance": {"data": {"varyList": vary, "covMatrix": cov}}}
    return gpx, phases, cov


def test_esd_matches_finite_difference_propagation(three_phases):
    gpx, phases, cov = three_phases
    esd, note = E._wt_fraction_esds(gpx, phases, _Hist())
    f = np.array([p._f for p in phases]); M = np.array([p.mass for p in phases])
    # numerical Jacobian of w wrt the refined fractions (phases 1, 2)
    J = np.zeros((3, 2)); h = 1e-7
    for c, k in enumerate((1, 2)):
        fp = f.copy(); fp[k] += h
        fm = f.copy(); fm[k] -= h
        J[:, c] = (_wt(fp, M) - _wt(fm, M)) / (2 * h)
    sub = cov[1:3, 1:3]
    expect = np.sqrt(np.einsum("ij,jk,ik->i", J, sub, J))
    got = np.array([esd[n] for n in ("a", "b", "c")])
    assert np.allclose(got, expect, rtol=1e-5)
    assert "statistical" in note


def test_collapsed_phase_gives_no_esd(three_phases):
    gpx, phases, _ = three_phases
    phases[2]._f = 0.0
    esd, note = E._wt_fraction_esds(gpx, phases, _Hist())
    assert all(v is None for v in esd.values())
    assert "collapsed" in note


def test_floored_phase_with_zero_variance_gives_no_esd(three_phases):
    # GSAS-II pins a vanishing fraction at 1e-12 with zero variance
    gpx, phases, cov = three_phases
    phases[1]._f = 1e-12
    cov[1, :] = 0.0; cov[:, 1] = 0.0
    esd, note = E._wt_fraction_esds(gpx, phases, _Hist())
    assert all(v is None for v in esd.values()) and "'b'" in note


def test_no_covariance_gives_no_esd(three_phases):
    _, phases, _ = three_phases
    esd, note = E._wt_fraction_esds({}, phases, _Hist())
    assert all(v is None for v in esd.values()) and "covariance" in note


def test_scales_not_varied_gives_no_esd(three_phases):
    gpx, phases, cov = three_phases
    gpx["Covariance"]["data"]["varyList"] = [":0:Scale", "x", "y", ":0:Back;0"]
    esd, note = E._wt_fraction_esds(gpx, phases, _Hist())
    assert all(v is None for v in esd.values()) and "not varied" in note


def test_tool_spec_documents_esds():
    assert "weight_fraction_esds" in _SPEC_MULTI.returns
    assert "STATISTICAL" in _SPEC_MULTI.returns


def test_system_info_block_reaches_codegen():
    from scilink.agents.exp_agents.controllers import curve_fitting_controllers as C
    si = {"wavelength_angstrom": 0.49591, "instrument": "synchrotron capillary"}
    block = C._system_info_block({"system_info": si})
    assert "0.49591" in block and "system_info" in block and "None" in block
    assert C._system_info_block({}) == ""
    big = C._system_info_block({"system_info": {"notes": "x" * 20000}})
    assert "truncated" in big and len(big) < C._SYSTEM_INFO_CODEGEN_CHARS + 1000
    import inspect
    for fn in (C.UnifiedSeriesProcessingController._generate_fitting_script,
               C.UnifiedSeriesProcessingController._correct_script):
        assert "_system_info_block(state)" in inspect.getsource(fn)
