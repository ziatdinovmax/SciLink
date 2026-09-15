"""Offline tests for the GPA illumination-envelope guard.

Regression for the benchmark HAADF cases where the Bragg amplitude tracked the
illumination/thickness envelope, so the 'valid' strain field followed the
detector gradient rather than the lattice. The guard flags amplitude/intensity
correlation and withholds `answerable`.
"""
import numpy as np

from scilink.skills._shared.strain import gpa_strain_map, make_strained_lattice


def _clean_lattice(shape=(256, 256)):
    img, g = make_strained_lattice(shape, a_px=10.0, displacement=None,
                                   kind="square", noise=0.02, seed=0)
    return img, g


def test_clean_lattice_not_flagged():
    img, g = _clean_lattice()
    r = gpa_strain_map(img, reflections=g, reference_roi="auto")
    assert r["flags"]["amplitude_tracks_intensity"] is False
    assert abs(r["amp_intensity_corr"]) < 0.6
    assert r["answerable"] is True


def test_illumination_envelope_is_flagged():
    """Multiply the lattice by a strong linear illumination ramp: the Bragg
    amplitude now scales with the raw intensity envelope -> guard must fire and
    withhold answerable."""
    img, g = _clean_lattice()
    H, W = img.shape
    ramp = np.linspace(0.05, 1.0, W)[None, :] * np.ones((H, 1))
    img_env = img * ramp
    r = gpa_strain_map(img_env, reflections=g, reference_roi="auto")
    assert r["amp_intensity_corr"] > 0.6, r["amp_intensity_corr"]
    assert r["flags"]["amplitude_tracks_intensity"] is True
    assert r["answerable"] is False


def test_strain_recovery_still_works():
    """The guard must not break the core self-test: a graded strain (contrast
    relative to the reference) is still recovered on a uniformly-illuminated
    lattice, and the result stays answerable."""
    W = 256
    def disp(yy, xx):
        eps = 0.03
        return np.zeros_like(yy), eps * xx ** 2 / (2.0 * W)  # exx = eps*x/W (graded)
    img, g = make_strained_lattice((256, W), a_px=10.0, displacement=disp,
                                   kind="square", noise=0.01, seed=1)
    r = gpa_strain_map(img, reflections=g, reference_roi="auto", detrend=False)
    assert r["flags"]["amplitude_tracks_intensity"] is False  # uniform illumination
    assert r["answerable"] is True
    assert r["stats"]["exx"]["n"] > 100
    # a graded strain leaves real spread in exx (not zeroed by the reference)
    assert r["stats"]["exx"]["p99_abs"] > 0.005


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}"); passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
