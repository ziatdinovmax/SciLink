"""Ground-truth regression tests for the crystalline_deformation Burgers tool.

Pattern (from the PR #393 review): numeric skill tools are validated against
analytic fields with known answers, not against themselves. The displacement
gradient of an isotropic edge dislocation is known in closed form, so
burgers_from_gpa must recover |b| exactly; the horizontal-edge signs in
_line_integral are exactly what this test pins down (a sign flip there gives a
Poisson-ratio-dependent error, |b| ~ 0.455 nm for nu = 0.3).
"""
import numpy as np
import pytest

from scilink.skills.image_analysis.crystalline_deformation.gpa_tools import (
    burgers_from_gpa,
)

H = W = 512
CALIB = 0.1   # nm/px
NU = 0.3      # Poisson ratio
B_TRUE = 1.0  # nm, along +x


def _edge_dislocation_beta():
    """Analytic displacement-gradient components of an edge dislocation
    (isotropic elasticity), b = B_TRUE along +x, core at the field centre."""
    b_px = B_TRUE / CALIB
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    x = xx - W / 2 + 0.5
    y = yy - H / 2 + 0.5
    r2 = x * x + y * y
    cc = 1 / (2 * (1 - NU))
    dd = (1 - 2 * NU) / (4 * (1 - NU))
    ee = 1 / (4 * (1 - NU))
    pref = b_px / (2 * np.pi)
    bxx = pref * (-y / r2 + cc * y * (y * y - x * x) / r2**2)      # dux/dx
    bxy = pref * (x / r2 + cc * x * (x * x - y * y) / r2**2)       # dux/dy
    byx = -pref * (2 * dd * x / r2 + 4 * ee * x * y * y / r2**2)   # duy/dx
    byy = -pref * (2 * dd * y / r2 - 4 * ee * y * x * x / r2**2)   # duy/dy
    e_xy = 0.5 * (bxy + byx)
    w_z = 0.5 * (byx - bxy)   # standard omega_z (= gpa_strain_map's wxy, + sign)
    return bxx, byy, e_xy, w_z


def test_edge_dislocation_magnitude_exact():
    bxx, byy, e_xy, w_z = _edge_dislocation_beta()
    out = burgers_from_gpa(bxx, byy, e_xy, w_z, pixel_size=CALIB,
                           dislocation_positions=[(H // 2, W // 2)],
                           circuit_half_width=60, n_shrink=0)
    b = out["burgers_vectors"][0]
    assert b["b_magnitude"] == pytest.approx(B_TRUE, abs=5e-3)


def test_edge_dislocation_direction_along_x():
    bxx, byy, e_xy, w_z = _edge_dislocation_beta()
    out = burgers_from_gpa(bxx, byy, e_xy, w_z, pixel_size=CALIB,
                           dislocation_positions=[(H // 2, W // 2)],
                           circuit_half_width=60, n_shrink=0)
    bx, by = out["burgers_vectors"][0]["b_vector"]
    assert abs(bx) == pytest.approx(B_TRUE, abs=5e-3)
    assert abs(by) < 5e-2 * B_TRUE


def test_magnitude_stable_across_circuit_sizes():
    """|b| must not depend on the circuit size (the closed-loop invariant a
    wrong edge sign destroys)."""
    bxx, byy, e_xy, w_z = _edge_dislocation_beta()
    mags = []
    for hw in (30, 60, 100):
        out = burgers_from_gpa(bxx, byy, e_xy, w_z, pixel_size=CALIB,
                               dislocation_positions=[(H // 2, W // 2)],
                               circuit_half_width=hw, n_shrink=0)
        mags.append(out["burgers_vectors"][0]["b_magnitude"])
    assert np.ptp(mags) < 2e-2 * B_TRUE
    assert np.mean(mags) == pytest.approx(B_TRUE, abs=1e-2)
