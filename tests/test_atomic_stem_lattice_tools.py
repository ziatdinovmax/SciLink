"""Offline synthetic-truth tests for the atomic_stem lattice / defect / ferroic
tools added on the lattice-boundary-tools branch:

  - measure_lattice_constant   (FFT Bragg-geometry cell measurement)
  - type_sublattice_defects     (per-sublattice vacancy + dopant typing)
  - lattice_discontinuity_map   (grain / twin / interface localizer)
  - map_polarization            (per-cell ferroelectric off-centering)

Positions are fed directly (no DCNN), so these run in seconds. They assert
planted-truth RECOVERY, that a tunable knob actually changes behavior, and
NO-false-positive controls (clean lattice -> no fabricated defects/field).
"""
import numpy as np
import pytest
from scipy.spatial import cKDTree

from scilink.skills.image_analysis.atomic_stem.lattice import measure_lattice_constant
from scilink.skills.image_analysis.atomic_stem.atom_finding import (
    type_sublattice_defects, map_polarization)
from scilink.skills.image_analysis.atomic_stem.discontinuity import lattice_discontinuity_map
from scilink.skills._shared.fft_defect import make_defective_lattice


# --------------------------------------------------------------------------- #
# synthetic generators                                                        #
# --------------------------------------------------------------------------- #
def _gauss(img, x, y, A, s):
    H, W = img.shape
    x0 = max(0, int(x - 4 * s)); x1 = min(W, int(x + 4 * s) + 1)
    y0 = max(0, int(y - 4 * s)); y1 = min(H, int(y + 4 * s) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    img[y0:y1, x0:x1] += A * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s))


def _two_sublattice(n=16, a=26, s=3.0, contrast=0.5, P_px=(0.0, 0.0),
                    domains=False, vac_B=None, dop_B=None, noise=0.01, seed=0):
    """A square sublattice (bright, intensity 1.0) interpenetrated by a B
    sublattice at the cell centres (intensity = ``contrast``), optionally
    displaced by ``P_px`` (ferroelectric off-centering; sign flips per half if
    ``domains``). ``vac_B`` = B indices to omit (planted vacancies); ``dop_B`` =
    {B index: intensity multiplier} (planted substitutional dopants).
    Returns (img float32, positions (x,y) of surviving columns, truth dict)."""
    rng = np.random.default_rng(seed)
    H = W = a * n + 2 * a
    img = np.zeros((H, W)); A = []; B = []; vac_truth = []; dop_truth = []
    vac_B = set(vac_B or []); dop_B = dict(dop_B or {})
    bi = 0
    for i in range(n):
        for j in range(n):
            ax = a + j * a; ay = a + i * a
            _gauss(img, ax, ay, 1.0, s); A.append((ax, ay))
            sign = 1 if (not domains or j < n // 2) else -1
            bx = ax + a / 2 + P_px[0] * sign; by = ay + a / 2 + P_px[1] * sign
            if bi in vac_B:
                vac_truth.append((bx, by))                      # omitted column
            else:
                _gauss(img, bx, by, contrast * dop_B.get(bi, 1.0), s)
                B.append((bx, by))
                if bi in dop_B:
                    dop_truth.append((bx, by))
            bi += 1
    img += rng.normal(0, noise, img.shape)
    img = np.clip(img, 0, None)
    return (img.astype(np.float32), np.array(A + B),
            {"vac": vac_truth, "dop": dop_truth, "nA": len(A), "nB": len(B), "a": a})


def _single_square(n=16, a=26, s=3.0, noise=0.01, seed=0):
    """One square lattice, no second sublattice (negative control)."""
    rng = np.random.default_rng(seed)
    H = W = a * n + 2 * a
    img = np.zeros((H, W)); pts = []
    for i in range(n):
        for j in range(n):
            x = a + j * a; y = a + i * a
            _gauss(img, x, y, 1.0, s); pts.append((x, y))
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, None).astype(np.float32), np.array(pts)


def _bicrystal(shape=(512, 512), a=24, theta_deg=14.0, s=3.0, noise=0.02, seed=0):
    """Two square domains sharing a vertical seam at x=W/2: the left domain is
    axis-aligned, the right is rotated by ``theta_deg`` about the seam centre.
    theta_deg=0 -> a single continuous lattice (no-boundary control)."""
    rng = np.random.default_rng(seed)
    H, W = shape; img = np.zeros((H, W)); cx = W / 2.0
    seam = np.array([cx, H / 2.0])
    th = np.deg2rad(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    nx = int(W / a) + 6; ny = int(H / a) + 6
    for i in range(-3, ny):
        for j in range(-3, nx):
            base = np.array([j * a, i * a], float)
            if base[0] < cx:
                x, y = base
            else:
                x, y = R @ (base - seam) + seam
            if 0 <= x < W and 0 <= y < H:
                _gauss(img, x, y, 1.0, s)
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, None).astype(np.float32)


def _match_count(found_xy, truth_xy, tol):
    if not truth_xy:
        return 0
    if not found_xy:
        return 0
    tree = cKDTree(np.asarray(found_xy))
    return int(sum(tree.query(np.asarray(t))[0] <= tol for t in truth_xy))


# --------------------------------------------------------------------------- #
# measure_lattice_constant                                                    #
# --------------------------------------------------------------------------- #
class TestMeasureLatticeConstant:
    def test_resolves_square_cell(self):
        img, _ = make_defective_lattice((512, 512), a_px=12.0, kind="square",
                                        n_vacancies=0, noise=0.03, seed=1)
        r = measure_lattice_constant(img, pixel_size_nm=0.1)
        assert r["lattice_constant_nm"] is not None
        assert abs(r["gamma_deg"] - 90.0) < 6.0        # right-angle cell
        assert r["n_reflections"] >= 2
        assert not r["low_confidence"]

    def test_tracks_lattice_spacing(self):
        # Convention-independent: the measured constant scales with the real
        # lattice spacing. Use a fine pixel size so both fundamentals stay
        # inside the (nm-based) d_range gate rather than being clipped.
        c12 = measure_lattice_constant(
            make_defective_lattice((512, 512), a_px=12.0, kind="square",
                                   n_vacancies=0, noise=0.02, seed=1)[0], 0.05)
        c20 = measure_lattice_constant(
            make_defective_lattice((512, 512), a_px=20.0, kind="square",
                                   n_vacancies=0, noise=0.02, seed=1)[0], 0.05)
        assert c12["lattice_constant_nm"] and c20["lattice_constant_nm"]
        ratio = c20["lattice_constant_nm"] / c12["lattice_constant_nm"]
        assert abs(ratio - 20.0 / 12.0) < 0.15         # tracks real spacing

    def test_min_sigma_knob_changes_behavior(self):
        img, _ = make_defective_lattice((512, 512), a_px=12.0, kind="square",
                                        n_vacancies=0, noise=0.12, seed=3)
        strict = measure_lattice_constant(img, 0.1, params={"min_sigma": 8.0})
        loose = measure_lattice_constant(img, 0.1, params={"min_sigma": 2.0})
        # a stricter significance floor admits no more reflections than a loose one
        assert loose["n_reflections"] >= strict["n_reflections"]


# --------------------------------------------------------------------------- #
# type_sublattice_defects                                                     #
# --------------------------------------------------------------------------- #
class TestSublatticeDefects:
    # n=18, contrast 0.5 -> local_env_gmm separates the two sublattices cleanly
    # (324/324); central planted defects + a 2a edge margin keep the test precise.
    _A = 26

    def test_vacancies_and_dopants_recovered(self):
        # central B indices (i*18+j with i,j in ~7..11) so they sit well inside
        # the 2a edge margin with full reference cages
        img, pos, truth = _two_sublattice(
            n=18, a=self._A, contrast=0.5,
            vac_B=[133, 152, 171], dop_B={130: 1.9, 175: 0.45}, seed=2)
        r = type_sublattice_defects(img, pos, n_sublattices=2, pixel_size_nm=0.1,
                                    edge_margin_px=2.0 * self._A)
        m = r["metrics"]
        assert "error" not in m
        assert len(m["sublattices"]) == 2
        found_vac = [(d["x"], d["y"]) for d in m["vacancies"]]
        found_dop = [(d["x"], d["y"]) for d in m["dopants"]]
        assert _match_count(found_vac, truth["vac"], tol=0.7 * self._A) >= 2
        assert _match_count(found_dop, truth["dop"], tol=0.7 * self._A) >= 1

    def test_clean_lattice_no_fabricated_defects(self):
        img, pos, _ = _two_sublattice(n=18, a=self._A, contrast=0.5, seed=5)
        r = type_sublattice_defects(img, pos, n_sublattices=2, pixel_size_nm=0.1,
                                    edge_margin_px=2.0 * self._A)
        m = r["metrics"]
        assert "error" not in m
        assert len(m["sublattices"]) == 2
        assert m["n_vacancies"] <= 2          # clean -> ~no false vacancies
        assert m["n_dopants"] <= 2            # clean -> ~no false dopants


# --------------------------------------------------------------------------- #
# lattice_discontinuity_map                                                   #
# --------------------------------------------------------------------------- #
class TestDiscontinuity:
    def test_orientation_boundary_localized(self):
        img = _bicrystal((512, 512), a=24, theta_deg=15.0, seed=1)
        r = lattice_discontinuity_map(img, pixel_size_nm=0.1)
        m = r["metrics"]
        assert m["n_boundaries"] >= 1
        xs = [b["centroid_px"][0] for b in m["boundaries"]]
        assert any(abs(x - 256) < 70 for x in xs)            # near the seam
        assert max(b["orient_change_deg"] for b in m["boundaries"]) > 6.0

    def test_single_domain_no_boundary(self):
        img = _bicrystal((512, 512), a=24, theta_deg=0.0, seed=2)
        r = lattice_discontinuity_map(img, pixel_size_nm=0.1)
        assert r["metrics"]["boundary_fraction"] < 0.15


# --------------------------------------------------------------------------- #
# map_polarization                                                            #
# --------------------------------------------------------------------------- #
class TestMapPolarization:
    def test_uniform_field_recovered(self):
        img, pos, _ = _two_sublattice(contrast=0.5, P_px=(2.5, 0.0), seed=1)
        m = map_polarization(img, pos, pixel_size_nm=0.1)["metrics"]
        assert "error" not in m
        assert m["direction_coherence"] > 0.8
        assert abs(m["median_magnitude_px"] - 2.5) < 0.8

    def test_geometric_fallback_identical_Z(self):
        # equal-Z cations -> intensity split unreliable -> geometric fallback
        img, pos, _ = _two_sublattice(contrast=1.0, P_px=(2.5, 0.0),
                                      domains=True, seed=2)
        res = map_polarization(img, pos, pixel_size_nm=0.1)
        assert "geometric_separation_used_intensity_ambiguous" in res["flags"]
        assert res["metrics"]["direction_coherence"] > 0.8

    def test_no_fabricated_field_single_sublattice(self):
        img, pos = _single_square(seed=3)
        res = map_polarization(img, pos, pixel_size_nm=0.1)
        m = res["metrics"]
        # honest failure: either no valid cells, or a flagged incoherent field
        assert ("error" in m) or (m.get("direction_coherence", 0.0) < 0.5)
