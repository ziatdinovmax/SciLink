"""Offline tests for the grain twin-proxy reliability gate.

Regression for the benchmark false-positive where a 0.83-straight EBSD network
was reported as an 0.83 twin fraction. The twin proxy is only reliable when a
curved-boundary baseline exists (curved_fraction >= min_curved_fraction).
"""
import numpy as np

from scilink.skills._shared.grain_analysis import grain_analysis, make_grain_map


def test_polygonal_map_is_not_a_twin_false_positive():
    """A synthetic Voronoi map has straight boundaries and NO twins; the proxy
    must gate itself off and report twin fraction 0, not the raw straight frac."""
    img, _ = make_grain_map(shape=(400, 600), n_seeds=40, kind="ipf", seed=0)
    r = grain_analysis(img, mode="ipf")
    assert r["straight_boundary_fraction"] > 0.5, "polygonal map should read mostly-straight"
    assert r["twin_proxy_reliable"] is False
    assert r["twin_boundary_fraction"] == 0.0
    assert r["straight_segments"] == []
    assert r["mean_twin_segments_per_grain"] == 0.0


def test_curved_baseline_is_reported_and_consistent():
    """curved_fraction == 1 - straight_fraction and drives the gate."""
    img, _ = make_grain_map(shape=(400, 600), n_seeds=40, kind="ipf", seed=1)
    r = grain_analysis(img, mode="ipf")
    sf, cf = r["straight_boundary_fraction"], r["curved_boundary_fraction"]
    assert abs((sf + cf) - 1.0) < 1e-6
    assert r["twin_proxy_reliable"] == bool(cf >= r["min_curved_fraction"])


def test_threshold_is_tunable():
    """Lowering min_curved_fraction can re-enable the proxy; raising it disables."""
    img, _ = make_grain_map(shape=(400, 600), n_seeds=40, kind="ipf", seed=0)
    cf = grain_analysis(img, mode="ipf")["curved_boundary_fraction"]
    # threshold just below the actual curved fraction -> reliable
    r_lo = grain_analysis(img, mode="ipf", min_curved_fraction=max(cf - 0.01, 0.0))
    assert r_lo["twin_proxy_reliable"] is True
    # threshold above it -> not reliable
    r_hi = grain_analysis(img, mode="ipf", min_curved_fraction=min(cf + 0.01, 1.0))
    assert r_hi["twin_proxy_reliable"] is False


def test_old_threshold_would_have_false_positived():
    """Document the regression: a 0.7-0.9 straight fraction (curved 0.1-0.3) was
    blessed by the old `frac < 0.9` rule but is correctly rejected now."""
    img, _ = make_grain_map(shape=(400, 600), n_seeds=60, kind="ipf", seed=3)
    r = grain_analysis(img, mode="ipf")
    sf = r["straight_boundary_fraction"]
    if 0.7 <= sf < 0.9:  # the exact regime that used to misfire
        assert r["twin_proxy_reliable"] is False
        assert r["twin_boundary_fraction"] == 0.0


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
