"""Offline tests for the vetted map-over-pixels harness (#356).

Covers the mechanical layer end to end: declarative spec normalization,
serial-vs-parallel map equivalence, planted-truth accuracy with sigma err
maps, mask scoping, the soft time budget's partial-return contract,
dead/NaN-pixel guards, knob-changes-behavior, thread-caller safety (fan-out
branches run in threads), and TOOL_SPEC registration.

  conda run -n scilink python tests/test_parallel_pixel_fit.py
"""
import os
import time

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

import numpy as np

from scilink.skills._shared.parallel_pixel_fit import (
    _normalize_model_spec, _build_lmfit_model, fit_per_pixel)

results = {}
RNG = np.random.default_rng(3)
WL = np.linspace(400, 900, 90)


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def _cube(h=20, w=20, drift=True, noise=0.3):
    centers = 600 + (40 * (np.arange(w) / (w - 1)) if drift else 0.0)
    centers = np.broadcast_to(np.atleast_1d(centers), (w,))
    spec = 4.0 * np.exp(-0.5 * ((WL[None, :] - centers[:, None]) / 18) ** 2)
    cube = (5.0 + np.broadcast_to(spec[None], (h, w, WL.size))
            + RNG.normal(0, noise, (h, w, WL.size)))
    truth = np.broadcast_to(centers[None], (h, w))
    return cube, truth


MODEL = [{"type": "gaussian", "window": (570, 670)}, "constant"]


def main():
    # ------------------------------------------------------------------
    print("1) declarative spec normalization:")
    s = _normalize_model_spec("gaussian")
    check("type string -> one-component spec",
          len(s) == 1 and s[0]["type"] == "gaussian"
          and s[0]["prefix"] == "gaus1_")
    s = _normalize_model_spec(["gaussian", "gaussian", "linear"])
    check("repeated types get distinct prefixes",
          [c["prefix"] for c in s] == ["gaus1_", "gaus2_", "line1_"])
    s = _normalize_model_spec({"type": "sigmoid"})
    check("sigmoid aliases to step_logistic", s[0]["type"] == "step_logistic")
    try:
        _normalize_model_spec("gaussain")
        check("unknown type fails loudly in the parent", False)
    except ValueError as e:
        check("unknown type fails loudly in the parent", "Registry" in str(e))
    try:
        _normalize_model_spec([])
        check("empty model rejected", False)
    except ValueError:
        check("empty model rejected", True)

    m, p = _build_lmfit_model(_normalize_model_spec(
        [{"type": "gaussian", "center": 620, "center_min": 600,
          "center_max": 640, "sigma": 15},
         {"type": "constant", "fix": {"c": 5.0}}]))
    check("flat init + _min/_max keys land on lmfit params",
          p["gaus1_center"].value == 620 and p["gaus1_center"].min == 600
          and p["gaus1_center"].max == 640 and p["gaus1_sigma"].value == 15)
    check("fix freezes the parameter",
          p["cons1_c"].value == 5.0 and not p["cons1_c"].vary)
    m, p = _build_lmfit_model(_normalize_model_spec(
        [{"type": "gaussian", "window": (580, 660)}]))
    check("window sets peak-center bounds",
          p["gaus1_center"].min == 580 and p["gaus1_center"].max == 660)
    check("peak widths floored positive", p["gaus1_sigma"].min > 0)

    # ------------------------------------------------------------------
    print("2) planted-truth accuracy + uncertainty maps:")
    cube, truth = _cube()
    r = fit_per_pixel(cube, WL, MODEL, n_jobs=1)
    c = r["maps"]["gaus1_center"]
    check("center map recovers the planted drift (<1 unit mean error)",
          np.nanmean(np.abs(c - truth)) < 1.0)
    check("full coverage on a clean cube", r["coverage"].all())
    check("sigma err maps populated from covariance",
          np.isfinite(r["err_maps"]["gaus1_center"]).mean() > 0.9
          and np.nanmedian(r["err_maps"]["gaus1_center"]) < 2.0)
    check("R2 map present and high", np.nanmean(r["maps"]["R2"]) > 0.8)
    check("derived fwhm exported", any("fwhm" in k for k in r["maps"]))
    check("stats block coherent",
          r["stats"]["n_fit"] == r["stats"]["n_selected"] == truth.size)

    print("3) serial == parallel:")
    r2 = fit_per_pixel(cube, WL, MODEL, n_jobs=2, chunk_size=64)
    check("maps bit-identical across n_jobs (same math, chunked)",
          np.array_equal(c, r2["maps"]["gaus1_center"], equal_nan=True))
    check("err maps identical too",
          np.array_equal(r["err_maps"]["gaus1_center"],
                         r2["err_maps"]["gaus1_center"], equal_nan=True))

    # ------------------------------------------------------------------
    print("4) mask scoping:")
    mask = np.zeros(truth.shape, bool)
    mask[5:10, 5:10] = True
    rm = fit_per_pixel(cube, WL, MODEL, mask=mask, n_jobs=1)
    check("only masked pixels fit",
          rm["coverage"].sum() == mask.sum()
          and (rm["coverage"] == mask).all())
    check("outside-mask pixels are NaN",
          np.isnan(rm["maps"]["gaus1_center"][~mask]).all())
    check("stats reflect the mask", rm["stats"]["n_selected"] == 25)
    r0 = fit_per_pixel(cube, WL, MODEL, mask=np.zeros(truth.shape, bool))
    check("empty mask degrades gracefully with a note",
          r0["stats"]["n_fit"] == 0 and r0["notes"]
          and not r0["coverage"].any())
    try:
        fit_per_pixel(cube, WL, MODEL, mask=np.ones((3, 3), bool))
        check("wrong-shape mask rejected", False)
    except ValueError:
        check("wrong-shape mask rejected", True)

    # ------------------------------------------------------------------
    print("5) time budget partial-return contract:")
    big, _ = _cube(h=60, w=60)
    rb = fit_per_pixel(big, WL, MODEL, n_jobs=1, chunk_size=256,
                       time_budget_s=1.0)
    s = rb["stats"]
    check("budget stops dispatch with work remaining",
          s["n_skipped_budget"] > 0 and s["n_fit"] > 0)
    check("budget note surfaced", any("budget" in n for n in rb["notes"]))
    cov = rb["coverage"].ravel()
    cm = rb["maps"]["gaus1_center"].ravel()
    check("uncovered pixels are NaN in the maps", np.isnan(cm[~cov]).all())
    check("covered pixels carry values", np.isfinite(cm[cov]).all())

    # ------------------------------------------------------------------
    print("6) dead / NaN / saturated pixel guards:")
    dirty, truth_d = _cube(h=12, w=12)
    dirty[0, 0] = np.nan                      # all-NaN pixel
    dirty[0, 1] = 7.0                         # saturated/constant pixel
    dirty[0, 2, ::2] = np.nan                 # half-NaN pixel (still fittable)
    rd = fit_per_pixel(dirty, WL, MODEL, n_jobs=1)
    check("all-NaN pixel skipped (NaN + coverage False)",
          not rd["coverage"][0, 0]
          and np.isnan(rd["maps"]["gaus1_center"][0, 0]))
    check("constant pixel skipped", not rd["coverage"][0, 1])
    check("partial-NaN pixel still fit on its finite channels",
          rd["coverage"][0, 2]
          and abs(rd["maps"]["gaus1_center"][0, 2] - truth_d[0, 2]) < 3.0)
    check("dirty pixels don't poison neighbors",
          rd["coverage"][1:].all())

    # ------------------------------------------------------------------
    print("7) knobs change behavior:")
    rt = fit_per_pixel(cube, WL, MODEL, n_jobs=1,
                       bounds={"center": (610, 612)})
    ct = rt["maps"]["gaus1_center"]
    check("tightened bounds clamp the map",
          np.nanmax(ct) <= 612 + 1e-6 and np.nanmin(ct) >= 610 - 1e-6)
    shifted, truth_s = _cube(drift=True)
    r_auto = fit_per_pixel(shifted, WL,
                           [{"type": "gaussian", "window": (570, 670),
                             "center": 605}, "constant"],
                           init="auto", n_jobs=1)
    r_spec = fit_per_pixel(shifted, WL,
                           [{"type": "gaussian", "window": (570, 670),
                             "center": 605}, "constant"],
                           init="spec", n_jobs=1)
    e_auto = np.nanmean(np.abs(r_auto["maps"]["gaus1_center"] - truth_s))
    e_spec = np.nanmean(np.abs(r_spec["maps"]["gaus1_center"] - truth_s))
    check("init knob changes behavior (auto locks the drift at least as "
          f"well as spec: {e_auto:.2f} vs {e_spec:.2f})", e_auto <= e_spec)
    try:
        fit_per_pixel(cube, WL, MODEL, init="bogus")
        check("bad init rejected", False)
    except ValueError:
        check("bad init rejected", True)

    # ------------------------------------------------------------------
    print("8) thread-caller safety (fan-out branches are threads):")
    from concurrent.futures import ThreadPoolExecutor
    small, truth_t = _cube(h=10, w=10)

    def _in_thread():
        return fit_per_pixel(small, WL, MODEL, n_jobs=2, chunk_size=32)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1, f2 = pool.submit(_in_thread), pool.submit(_in_thread)
        out1, out2 = f1.result(timeout=300), f2.result(timeout=300)
    check("parallel fit inside worker threads completes (no fork deadlock)",
          out1["coverage"].all() and out2["coverage"].all())
    check("concurrent thread callers agree",
          np.array_equal(out1["maps"]["gaus1_center"],
                         out2["maps"]["gaus1_center"], equal_nan=True))

    # ------------------------------------------------------------------
    print("9) input shapes + speedup smoke:")
    flat = small.reshape(-1, WL.size)
    rf = fit_per_pixel(flat, WL, MODEL, n_jobs=1)
    check("(pixels, E) input returns 1-D maps",
          rf["maps"]["gaus1_center"].shape == (100,)
          and rf["coverage"].all())
    try:
        fit_per_pixel(small, WL[:-5], MODEL)
        check("axis-length mismatch rejected", False)
    except ValueError:
        check("axis-length mismatch rejected", True)
    # Big enough that ~1-2s of loky spawn warmup can't mask the win
    # (the real target regime is 10^5+ pixels; see the live test).
    perf, _ = _cube(h=64, w=64)
    t0 = time.monotonic()
    rs = fit_per_pixel(perf, WL, MODEL, n_jobs=1)
    t_ser = time.monotonic() - t0
    t0 = time.monotonic()
    rp = fit_per_pixel(perf, WL, MODEL, n_jobs=4, chunk_size=512)
    t_par = time.monotonic() - t0
    print(f"    serial {t_ser:.1f}s vs parallel(4) {t_par:.1f}s")
    check("parallel beats serial on 4096 pixels", t_par < t_ser)

    # ------------------------------------------------------------------
    print("10) TOOL_SPEC registration:")
    from scilink.skills._shared._registry import get_tools_for
    specs = {s.name: s for s in get_tools_for("hyperspectral")}
    check("fit_per_pixel registered for the hyperspectral agent",
          "fit_per_pixel" in specs)
    sp = specs.get("fit_per_pixel")
    check("every knob documented with tuning guidance",
          sp is not None and all(
              k in sp.parameters and len(str(sp.parameters[k])) > 40
              for k in ("data", "axis", "model", "mask", "init", "bounds",
                        "n_jobs", "chunk_size", "time_budget_s")))
    check("example wires the gate's fit_mask",
          sp is not None and "fit_mask" in sp.example)
    from scilink.agents.exp_agents.controllers.hyperspectral_controllers import (
        build_code_generation_prompt)
    prompt = build_code_generation_prompt(
        "t", 96, 96, 90, "nm", 400.0, 900.0, "raw")
    check("SIZE BUDGET block routes big fits to fit_per_pixel",
          "fit_per_pixel" in prompt and "SIZE BUDGET" in prompt)

    print("\n" + "=" * 50)
    npass = sum(results.values())
    print(f"PARALLEL PIXEL FIT: {npass}/{len(results)} checks passed")
    for k, v in results.items():
        if not v:
            print("  FAILED:", k)
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
