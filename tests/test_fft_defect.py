"""Offline tests for scilink.skills._shared.fft_defect (synthetic lattices)."""
import numpy as np
import pytest

from scilink.skills._shared.fft_defect import fft_defect_map, make_defective_lattice
from scilink.skills._shared.strain import make_strained_lattice


def _matched(found, truth_pts, tol):
    """Truth points that have a found candidate within tol px."""
    hits = []
    for ty, tx in truth_pts:
        if any(np.hypot(d["y"] - ty, d["x"] - tx) <= tol for d in found):
            hits.append((ty, tx))
    return hits


def _false_positives(found, truth_all, tol):
    return [d for d in found
            if all(np.hypot(d["y"] - ty, d["x"] - tx) > tol
                   for ty, tx in truth_all)]


class TestVacancies:
    def test_hex_vacancies_recovered(self):
        img, truth = make_defective_lattice(
            (512, 512), a_px=14.0, kind="hex", n_vacancies=6, noise=0.05, seed=1)
        res = fft_defect_map(img)
        assert res["periodic"]
        assert abs(res["pattern_period_px"] - 14.0 * np.sqrt(3) / 2) < 2.5
        vac = truth["vacancies"]
        assert len(_matched(res["defects"], vac, tol=14.0)) == len(vac)
        # vacancies are deficits and disrupt lattice coherence
        for ty, tx in vac:
            d = min(res["defects"],
                    key=lambda c: np.hypot(c["y"] - ty, c["x"] - tx))
            assert d["sign"] == "deficit"
        assert len(_false_positives(res["defects"], vac, tol=14.0)) <= 1

    def test_square_vacancies_recovered(self):
        img, truth = make_defective_lattice(
            (512, 512), a_px=12.0, kind="square", n_vacancies=5,
            noise=0.05, seed=2)
        res = fft_defect_map(img)
        assert res["periodic"]
        vac = truth["vacancies"]
        assert len(_matched(res["defects"], vac, tol=12.0)) == len(vac)
        assert len(_false_positives(res["defects"], vac, tol=12.0)) <= 1

    def test_mesoscale_particle_array(self):
        # missing "particles" in a coarse array — pattern-agnostic claim
        img, truth = make_defective_lattice(
            (768, 768), a_px=40.0, kind="hex", n_vacancies=4,
            atom_sigma_frac=0.25, noise=0.06, seed=3)
        res = fft_defect_map(img)
        assert res["periodic"]
        vac = truth["vacancies"]
        assert len(_matched(res["defects"], vac, tol=40.0)) == len(vac)
        assert len(_false_positives(res["defects"], vac, tol=40.0)) <= 1


class TestRobustness:
    def test_rotated_lattice(self):
        from scipy import ndimage as ndi
        img, truth = make_defective_lattice(
            (640, 640), a_px=14.0, kind="hex", n_vacancies=4, noise=0.0,
            seed=10)
        rot = ndi.rotate(img, 17.0, reshape=False, order=3, mode="reflect")
        # rotate truth coordinates around the image centre (ndi.rotate spins
        # the image content by -angle in (y, x) frame)
        th = np.deg2rad(17.0)
        c = (np.array(rot.shape) - 1) / 2.0
        rmat = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        vac = [tuple(rmat @ (np.array(v) - c) + c) for v in truth["vacancies"]]
        crop = rot[80:-80, 80:-80]
        vac = [(y - 80, x - 80) for y, x in vac
               if 80 + 30 < y < 640 - 110 and 80 + 30 < x < 640 - 110]
        assert len(vac) >= 2  # most planted defects survive the crop
        res = fft_defect_map(crop + np.random.default_rng(0).normal(0, 0.05, crop.shape))
        assert res["periodic"]
        assert len(_matched(res["defects"], vac, tol=14.0)) == len(vac)
        assert len(_false_positives(res["defects"], vac, tol=14.0)) <= 1

    def test_heavy_noise(self):
        img, truth = make_defective_lattice(
            (512, 512), a_px=14.0, kind="hex", n_vacancies=5, noise=0.15,
            seed=11)
        res = fft_defect_map(img)
        assert res["periodic"]
        vac = truth["vacancies"]
        # heavy noise: most vacancies still recovered, few false alarms
        assert len(_matched(res["defects"], vac, tol=14.0)) >= 4
        assert len(_false_positives(res["defects"], vac, tol=14.0)) <= 2


class TestOtherDefectTypes:
    def test_dopants_are_excess(self):
        img, truth = make_defective_lattice(
            (512, 512), a_px=14.0, n_vacancies=0, n_dopants=4,
            dopant_contrast=2.0, noise=0.05, seed=4)
        res = fft_defect_map(img)
        dop = truth["dopants"]
        assert len(_matched(res["defects"], dop, tol=14.0)) == len(dop)
        for ty, tx in dop:
            d = min(res["defects"],
                    key=lambda c: np.hypot(c["y"] - ty, c["x"] - tx))
            assert d["sign"] == "excess"

    def test_interstitials_found(self):
        img, truth = make_defective_lattice(
            (512, 512), a_px=14.0, n_vacancies=0, n_interstitials=4,
            noise=0.05, seed=5)
        res = fft_defect_map(img)
        inter = truth["interstitials"]
        assert len(_matched(res["defects"], inter, tol=14.0)) >= len(inter) - 1


class TestHonestNegatives:
    def test_pure_noise_not_periodic(self):
        rng = np.random.default_rng(0)
        res = fft_defect_map(rng.normal(0, 1, (512, 512)))
        assert res["periodic"] is False
        assert "defects" not in res

    def test_pristine_lattice_near_zero_defects(self):
        img, _ = make_defective_lattice(
            (512, 512), a_px=14.0, n_vacancies=0, noise=0.05, seed=6)
        res = fft_defect_map(img)
        assert res["periodic"]
        assert res["n_defects"] <= 1

    def test_smooth_strain_not_flagged(self):
        # smoothly strained but defect-free lattice: GPA territory, and the
        # residual map must not light up
        img, _ = make_strained_lattice((512, 512), a_px=10.0, kind="hex")
        res = fft_defect_map(img)
        assert res["periodic"]
        assert res["n_defects"] <= 2


class TestBothSigns:
    def test_protrusions_and_vacancies_together(self):
        # a lattice carrying BOTH missing columns and bright adatoms must
        # return both deficit and excess candidates
        img, truth = make_defective_lattice(
            (512, 512), a_px=14.0, kind="hex", n_vacancies=4, n_dopants=4,
            dopant_contrast=2.0, noise=0.05, seed=20)
        res = fft_defect_map(img)
        signs = {d["sign"] for d in res["defects"]}
        assert "deficit" in signs and "excess" in signs
        assert len(_matched(res["defects"], truth["vacancies"], 14.0)) >= 3
        assert len(_matched(res["defects"], truth["dopants"], 14.0)) >= 3

    def test_caller_must_resample_anisotropic(self):
        # the tool assumes square pixels; an externally-resampled square copy
        # recovers the lattice geometry (caller's job, not the tool's)
        from scipy import ndimage as ndi
        img, _ = make_defective_lattice((256, 512), a_px=16.0, kind="hex",
                                        n_vacancies=3, noise=0.05, seed=21)
        compressed = ndi.zoom(img, (0.5, 1.0), order=3)   # 2:1 anisotropic
        sq = ndi.zoom(compressed, (2.0, 1.0), order=3)    # caller resamples
        res = fft_defect_map(sq)
        assert res["periodic"]


class TestContract:
    def test_calibrated_fields(self):
        img, _ = make_defective_lattice((256, 256), a_px=12.0, n_vacancies=2,
                                        seed=7)
        res = fft_defect_map(img, pixel_size_nm=0.05)
        assert "pattern_period_nm" in res
        for d in res["defects"]:
            assert "y_nm" in d and "x_nm" in d

    def test_maps_returned(self):
        img, _ = make_defective_lattice((256, 256), a_px=12.0, n_vacancies=2,
                                        seed=8)
        res = fft_defect_map(img)
        assert res["residual_sigma_map"].shape == img.shape
        assert res["lattice_amplitude_map"].shape == img.shape
        assert res["valid_mask"].dtype == bool

    def test_rgb_input(self):
        img, _ = make_defective_lattice((256, 256), a_px=12.0, n_vacancies=2,
                                        seed=9)
        rgb = np.stack([img] * 3, axis=-1)
        res = fft_defect_map(rgb)
        assert res["periodic"]

    def test_registry_discovery(self):
        from scilink.skills._shared._registry import get_tools_for
        names = [s.name for s in get_tools_for("image_analysis")]
        assert "fft_defect_map" in names
