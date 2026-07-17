"""Offline tests for the measure_edge_step shared tool.

Synthetic datacubes with a PLANTED edge step (no spekpy, no benchmark files)
so the checks are deterministic and CI-portable. They pin the behaviour the
X-ray coupon suite exposed: improvised per-pixel code under-measured the edge
jump ~2x with tight/smoothed windows; this tool recovers the true step, is
edge-specific (a fake edge energy yields ~0), gates on flux, and exposes every
knob to the LLM (no locked defaults).

  conda run -n scilink python -m pytest tests/test_edge_step.py -q
"""
import inspect

import numpy as np

from scilink.skills._shared.edge_step import measure_edge_step, TOOL_SPEC


def _make_cube(
    H=24, W=24, edge=80.8, step=0.5, coupon_frac=0.5,
    i0_level=5000.0, i0_decay=None, cont_slope=0.005, cont_curv=0.0,
    seed=0, noise=False,
):
    """A (H,W,E) transmission cube with a known edge step over `coupon_frac`
    of the pixels. OD(E) = smooth continuum + step*H(E-edge); the tool should
    recover `step` regardless of the (smooth) continuum. `i0_decay` makes the
    incident flux fall with energy (to drive the post-edge flux gate)."""
    rng = np.random.default_rng(seed)
    E = np.arange(600, 1001) * 0.1          # 60.0 .. 100.0 keV, 401 channels
    ne = E.size
    cont = 0.2 + cont_slope * (E - 80.0) + cont_curv * (E - 80.0) ** 2
    coupon_od = cont + step * (E >= edge).astype(float)

    I0_spec = (np.full(ne, i0_level) if i0_decay is None
               else i0_level * np.exp(-(E - E[0]) / i0_decay))
    npix = H * W
    I0 = np.broadcast_to(I0_spec, (npix, ne)).copy()

    od = np.zeros((npix, ne))
    od[: int(npix * coupon_frac)] = coupon_od           # coupon = first block
    I = I0 * np.exp(-od)
    if noise:
        I = rng.poisson(np.clip(I, 0, None)).astype(float)
    return I.reshape(H, W, ne), I0.reshape(H, W, ne), E


def _coupon_empty(edge_step_map, coupon_frac=0.5):
    es = edge_step_map.ravel()
    cut = int(es.size * coupon_frac)
    return es[:cut], es[cut:]


def test_recovers_planted_step():
    """The headline fix: recover the true jump (not ~2x low) on the coupon,
    ~0 on the empty field."""
    data, i0, E = _make_cube(step=0.5)
    r = measure_edge_step(data, i0, E, 80.8)
    assert r["measurable"]
    coup, empty = _coupon_empty(r["edge_step"])
    assert abs(np.median(coup) - 0.5) < 0.05      # true step, not 0.25
    assert abs(np.median(empty)) < 0.05           # empty stays empty


def test_recovers_step_with_noise():
    data, i0, E = _make_cube(step=0.6, noise=True, seed=3)
    r = measure_edge_step(data, i0, E, 80.8)
    coup, _ = _coupon_empty(r["edge_step"])
    assert abs(np.median(coup) - 0.6) < 0.06


def test_edge_specific_fake_edge_gives_zero():
    """Edge-specificity: measuring at an energy with NO edge yields ~0, so a
    large step means a real discontinuity, not continuum-slope fitting."""
    data, i0, E = _make_cube(step=0.5, edge=80.8)
    r = measure_edge_step(data, i0, E, 90.0)      # no edge planted at 90 keV
    assert abs(r["field_step"]) < 0.05


def test_auto_center_snaps_to_offset_edge():
    """A mis-specified edge energy is corrected toward the real edge."""
    data, i0, E = _make_cube(step=0.6, edge=80.8)
    r = measure_edge_step(data, i0, E, 80.0, auto_center=True, search_tol_kev=2.0)
    assert abs(r["edge_kev_used"] - 80.8) < 0.3


def test_flux_gate_rejects_starved_edge():
    """Post-edge photon starvation -> measurable=False (refuse, don't fabricate)."""
    data, i0, E = _make_cube(step=0.5, i0_level=4000.0, i0_decay=4.0)
    r = measure_edge_step(data, i0, E, 80.8, flux_floor_counts=20.0)
    assert not r["measurable"]
    assert "flux" in r["reason"].lower()


def test_win_width_knob_changes_output():
    """A knob the LLM can turn must actually move the result (curved continuum
    makes the window width matter)."""
    data, i0, E = _make_cube(step=0.5, cont_curv=0.02)
    narrow = measure_edge_step(data, i0, E, 80.8, win_width=2.0)
    wide = measure_edge_step(data, i0, E, 80.8, win_width=6.0)
    c_n, _ = _coupon_empty(narrow["edge_step"])
    c_w, _ = _coupon_empty(wide["edge_step"])
    assert abs(np.median(c_n) - np.median(c_w)) > 0.02


def test_t_floor_knob_changes_opaque_result():
    """t_floor is the saturation guard: on an opaque coupon (T below the floor
    above the edge) it demonstrably changes the measured step."""
    data, i0, E = _make_cube(step=14.0, cont_slope=0.0)   # post-edge OD ~14 -> T<1e-6
    lo = measure_edge_step(data, i0, E, 80.8, t_floor=1e-6)
    hi = measure_edge_step(data, i0, E, 80.8, t_floor=1e-3)
    s_lo, _ = _coupon_empty(lo["edge_step"])
    s_hi, _ = _coupon_empty(hi["edge_step"])
    assert abs(np.median(s_lo) - np.median(s_hi)) > 1.0


def test_all_knobs_exposed_to_llm():
    """Anti-overfit contract: every function knob is surfaced in TOOL_SPEC (the
    only surface the LLM sees) — no locked defaults."""
    knobs = [p for p in inspect.signature(measure_edge_step).parameters]
    for p in knobs:
        assert p in TOOL_SPEC.parameters, f"{p} is hidden from the LLM"


def test_registry_discovers_tool():
    """The tool must be visible to hyperspectral codegen via the registry."""
    from scilink.skills._shared._registry import get_tools_for
    names = {t.name for t in get_tools_for("hyperspectral", active_skills=[])}
    assert "measure_edge_step" in names
