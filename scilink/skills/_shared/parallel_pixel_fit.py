"""Vetted map-over-pixels harness for hyperspectral per-pixel fitting (#356).

A full-frame per-pixel map is embarrassingly parallel — hundreds of thousands
of independent, few-ms lineshape fits — but code generated into the in-process
``exec()`` sandbox cannot parallelize it: exec-defined functions don't pickle,
so nothing written there can cross a process boundary. This module is the
package-side escape hatch: a deterministic, testable harness whose chunk
workers are module-level (picklable everywhere, including Windows spawn) and
whose model is a *declarative spec* built from registry primitives rather than
a callable, so the description — not code — crosses to the workers.

Engineering constraints this encodes (all portability-driven):

- **loky** executor (via joblib): spawn-like semantics on macOS/Linux/Windows
  and safe to invoke from worker THREADS — fan-out branches run in threads,
  where a fork-based pool is a deadlock hazard.
- **Explicit memmap handoff**: the (pixels, energy) array is dumped once to a
  temp memmap and workers receive the file-backed view, so a 0.25–1 GB cube
  is not copied (or re-pickled) per task. Cleanup retries around Windows
  file-locking quirks.
- **BLAS pinning in workers** (threadpoolctl): without it n_jobs × BLAS-threads
  oversubscription makes "parallel" slower than serial.
- **Soft time budget**: chunks are dispatched in waves; at the deadline the
  harness stops dispatching and returns partial maps plus a coverage mask —
  degrade, never die at the sandbox's exec cap.
- **σ as a first-class output**: per-parameter uncertainties from the fit
  covariance are returned as ``err_maps`` alongside the value maps.

Scope: independent per-pixel fits ONLY. Coupled/global fits (parameters shared
across pixels, spatial regularization, MCR-ALS-style constraints) are not
embarrassingly parallel and are out of scope. lmfit remains the per-fit engine
inside the workers.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Declarative model registry
# ---------------------------------------------------------------------------
# Model specs are data, not callables (exec-defined functions cannot cross a
# process boundary). Each component names a registry primitive; the lmfit
# composite is rebuilt inside each worker from the spec.

_PEAK_TYPES = ("gaussian", "lorentzian", "voigt", "pseudo_voigt")
_MODEL_REGISTRY = {
    # peaks
    "gaussian": ("GaussianModel", {}),
    "lorentzian": ("LorentzianModel", {}),
    "voigt": ("VoigtModel", {}),
    "pseudo_voigt": ("PseudoVoigtModel", {}),
    # backgrounds
    "constant": ("ConstantModel", {}),
    "linear": ("LinearModel", {}),
    "quadratic": ("QuadraticModel", {}),
    "power_law": ("PowerLawModel", {}),
    # decays / onsets
    "exponential": ("ExponentialModel", {}),
    "step_logistic": ("StepModel", {"form": "logistic"}),
    "step_erf": ("StepModel", {"form": "erf"}),
}
_SPEC_META_KEYS = {"type", "prefix", "window", "bounds", "fix"}


def _normalize_model_spec(model) -> list[dict]:
    """Accept str | dict | list of either; return a list of component dicts
    with explicit prefixes. Raises ValueError on unknown component types so a
    typo fails loudly in the parent, not cryptically in a worker."""
    if isinstance(model, (str, dict)):
        model = [model]
    if not isinstance(model, (list, tuple)) or not model:
        raise ValueError(
            "model must be a component type string, a component dict, or a "
            f"non-empty list of those; got {type(model).__name__}")
    out = []
    counts: dict[str, int] = {}
    for comp in model:
        if isinstance(comp, str):
            comp = {"type": comp}
        if not isinstance(comp, dict) or "type" not in comp:
            raise ValueError(f"model component must be a type string or a "
                             f"dict with a 'type' key; got {comp!r}")
        ctype = str(comp["type"]).lower().replace("-", "_")
        if ctype == "sigmoid":
            ctype = "step_logistic"
        if ctype not in _MODEL_REGISTRY:
            raise ValueError(
                f"unknown model component '{comp['type']}'. Registry: "
                f"{sorted(_MODEL_REGISTRY)}")
        comp = dict(comp)
        comp["type"] = ctype
        counts[ctype] = counts.get(ctype, 0) + 1
        comp.setdefault("prefix", f"{ctype[:4]}{counts[ctype]}_")
        out.append(comp)
    return out


def _build_lmfit_model(spec: list[dict]):
    """Composite lmfit model + Parameters from a normalized spec."""
    import lmfit.models as lm

    model = None
    for comp in spec:
        cls_name, kwargs = _MODEL_REGISTRY[comp["type"]]
        sub = getattr(lm, cls_name)(prefix=comp["prefix"], **kwargs)
        model = sub if model is None else model + sub
    params = model.make_params()

    for comp in spec:
        pfx = comp["prefix"]
        bounds = dict(comp.get("bounds") or {})
        # Flat convenience keys: center=620, center_min=600, center_max=640.
        for key, val in comp.items():
            if key in _SPEC_META_KEYS:
                continue
            if key.endswith("_min"):
                lo, hi = bounds.get(key[:-4], (None, None))
                bounds[key[:-4]] = (val, hi)
            elif key.endswith("_max"):
                lo, hi = bounds.get(key[:-4], (None, None))
                bounds[key[:-4]] = (lo, val)
            elif pfx + key in params:
                params[pfx + key].set(value=float(val))
        win = comp.get("window")
        if win is not None and comp["type"] in _PEAK_TYPES:
            bounds.setdefault("center", (float(win[0]), float(win[1])))
        for pname, (lo, hi) in bounds.items():
            if pfx + pname in params:
                params[pfx + pname].set(
                    min=lo if lo is not None else -np.inf,
                    max=hi if hi is not None else np.inf)
        for pname, val in (comp.get("fix") or {}).items():
            if pfx + pname in params:
                params[pfx + pname].set(value=float(val), vary=False)
        # Keep peak widths physical: positive sigma unless the author says so.
        for pname in ("sigma", "gamma"):
            if pfx + pname in params and params[pfx + pname].min <= 0:
                params[pfx + pname].set(min=1e-12)
    return model, params


def _moment_init(params, spec: list[dict], y: np.ndarray, x: np.ndarray,
                 keep_widths: bool = False):
    """Per-pixel heuristic starting values for peak components: center from
    the argmax inside the component's window (or bounds), amplitude from the
    baseline-subtracted maximum, sigma from a span-scaled default.
    keep_widths=True preserves widths already seeded (author spec or the
    global reference fit) instead of re-deriving them."""
    base = float(np.nanmedian(y))
    for comp in spec:
        if comp["type"] not in _PEAK_TYPES:
            if comp["type"] == "constant" and comp["prefix"] + "c" in params \
                    and params[comp["prefix"] + "c"].vary:
                params[comp["prefix"] + "c"].set(value=base)
            continue
        pfx = comp["prefix"]
        cpar = params.get(pfx + "center")
        if cpar is None:
            continue
        lo = cpar.min if np.isfinite(cpar.min) else x[0]
        hi = cpar.max if np.isfinite(cpar.max) else x[-1]
        sel = (x >= lo) & (x <= hi)
        if not sel.any():
            sel = np.ones_like(x, bool)
        yi = y[sel]
        xi = x[sel]
        if cpar.vary:
            cpar.set(value=float(xi[int(np.nanargmax(yi))]))
        spar = params.get(pfx + "sigma")
        if spar is not None and spar.vary and (
                (not keep_widths and "sigma" not in comp)
                or not np.isfinite(spar.value) or spar.value <= 0):
            # No author/reference width: span-scaled default beats lmfit's
            # unitless 1.0 on physical axes (nm, eV, cm^-1).
            spar.set(value=max((xi[-1] - xi[0]) / 10.0, 1e-6))
        apar = params.get(pfx + "amplitude")
        if apar is not None and apar.vary:
            height = float(np.nanmax(yi) - base)
            sig = float(spar.value) if spar is not None else \
                (xi[-1] - xi[0]) / 10.0
            # lmfit peak amplitude is the AREA; height ≈ A / (sigma*sqrt(2π)).
            apar.set(value=max(height, 1e-12) * sig * np.sqrt(2 * np.pi))
    return params


# ---------------------------------------------------------------------------
# Chunk worker — module-level, picklable, BLAS-pinned
# ---------------------------------------------------------------------------

def _fit_chunk(spectra, axis, spec, init_mode, ref_values, start, stop,
               param_names):
    """Fit pixels [start:stop) of a (pixels, energy) array. Returns
    (start, values, errors, r2, ok) as plain float arrays. Runs inside a loky
    worker (or inline for the serial path) — must stay importable and must
    not touch any parent-process state."""
    try:
        from threadpoolctl import threadpool_limits
        _ctl = threadpool_limits(limits=1)
    except Exception:  # pragma: no cover - threadpoolctl always ships w/ sklearn
        _ctl = None
    try:
        model, params = _build_lmfit_model(spec)
        if ref_values:
            for name, val in ref_values.items():
                if name in params and params[name].vary:
                    params[name].set(value=val)
        n = stop - start
        x = np.asarray(axis, float)
        values = np.full((n, len(param_names)), np.nan)
        errors = np.full((n, len(param_names)), np.nan)
        r2 = np.full(n, np.nan)
        ok = np.zeros(n, bool)
        for j in range(n):
            y = np.asarray(spectra[start + j], float)
            finite = np.isfinite(y)
            if finite.sum() < max(8, 2 * len(params)) or \
                    np.nanstd(y) < 1e-30:
                continue  # dead / saturated / mostly-NaN pixel
            xi, yi = x[finite], y[finite]
            p = params.copy()
            if init_mode == "moments":
                p = _moment_init(p, spec, yi, xi)
            elif init_mode == "auto":
                # Global reference values are already in `params` (widths,
                # backgrounds); snap each peak's center to this pixel's
                # local maximum and re-derive its amplitude, so a field-wide
                # drift (the common case) doesn't start every fit off-peak.
                p = _moment_init(p, spec, yi, xi,
                                 keep_widths=bool(ref_values))
            try:
                res = model.fit(yi, p, x=xi)
            except Exception:
                continue
            if not res.success:
                continue
            for k, pname in enumerate(param_names):
                par = res.params.get(pname)
                if par is None:
                    continue
                values[j, k] = par.value
                if par.stderr is not None:
                    errors[j, k] = par.stderr
            ss_res = float(np.sum(res.residual ** 2))
            ss_tot = float(np.sum((yi - yi.mean()) ** 2))
            r2[j] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            ok[j] = True
        return start, values, errors, r2, ok
    finally:
        if _ctl is not None:
            _ctl.unregister()


def _chunk_ranges(n_pixels: int, chunk_size: int):
    return [(s, min(s + chunk_size, n_pixels))
            for s in range(0, n_pixels, chunk_size)]


def _rmtree_with_retry(path: str, attempts: int = 5) -> None:
    """Windows memmap files can stay locked briefly after worker exit."""
    for i in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except OSError:
            time.sleep(0.2 * (i + 1))
    shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fit_per_pixel(data, axis, model, mask=None, init="auto", bounds=None,
                  n_jobs=-1, chunk_size=2048, time_budget_s=None) -> dict:
    """Parallel independent per-pixel lineshape fitting over a datacube.

    See TOOL_SPEC below for the knob-by-knob documentation. Returns a dict:
    ``maps`` (param -> (H,W) value map, plus ``R2``), ``err_maps`` (param ->
    (H,W) sigma-from-covariance map), ``coverage`` ((H,W) bool: fit succeeded),
    ``stats`` (counters + timing), ``notes`` (list of strings — budget stops,
    fallbacks, skip reasons).
    """
    t0 = time.monotonic()
    notes: list[str] = []

    data = np.asarray(data)
    if data.ndim == 2:            # (pixels, energy) -> treat as (N, 1, E)
        shape2d = (data.shape[0],)
        spectra_all = data.astype(np.float64, copy=False)
    elif data.ndim == 3:
        shape2d = data.shape[:2]
        spectra_all = data.reshape(-1, data.shape[2]).astype(
            np.float64, copy=False)
    else:
        raise ValueError(f"data must be (H, W, E) or (pixels, E); "
                         f"got shape {data.shape}")
    axis = np.asarray(axis, float)
    if axis.ndim != 1 or axis.size != spectra_all.shape[1]:
        raise ValueError(
            f"axis must be 1-D with length {spectra_all.shape[1]} "
            f"(the spectral dimension); got shape {axis.shape}")

    spec = _normalize_model_spec(model)
    if bounds:
        # Top-level bounds addressed as "<prefix><param>" or bare "<param>"
        # (bare names apply to every component that has that param).
        for comp in spec:
            b = dict(comp.get("bounds") or {})
            for key, rng in bounds.items():
                if key.startswith(comp["prefix"]):
                    b[key[len(comp["prefix"]):]] = tuple(rng)
                elif "_" not in key or not any(
                        key.startswith(c["prefix"]) for c in spec):
                    b.setdefault(key, tuple(rng))
            comp["bounds"] = b
    if init not in ("auto", "moments", "spec"):
        raise ValueError("init must be 'auto', 'moments', or 'spec'")

    # Fail loudly in the PARENT if the spec can't build a model at all.
    _probe_model, _probe_params = _build_lmfit_model(spec)
    param_names = [p for p in _probe_params
                   if _probe_params[p].vary or _probe_params[p].expr]

    # --- pixel selection: caller mask ∩ finite/live pixels -----------------
    n_pix = spectra_all.shape[0]
    if mask is not None:
        mask = np.asarray(mask, bool)
        if mask.shape != shape2d:
            raise ValueError(f"mask shape {mask.shape} != spatial shape "
                             f"{shape2d}")
        keep = mask.reshape(-1).copy()
    else:
        keep = np.ones(n_pix, bool)
    kept_idx = np.flatnonzero(keep)
    if kept_idx.size == 0:
        notes.append("mask selects 0 pixels — nothing to fit")
        empty = {p: np.full(shape2d, np.nan) for p in param_names}
        empty["R2"] = np.full(shape2d, np.nan)
        return {"maps": empty, "err_maps":
                {p: np.full(shape2d, np.nan) for p in param_names},
                "coverage": np.zeros(shape2d, bool),
                "stats": {"n_selected": 0, "n_fit": 0, "n_failed": 0,
                          "n_skipped_budget": 0,
                          "elapsed_s": time.monotonic() - t0},
                "notes": notes}
    spectra = np.ascontiguousarray(spectra_all[kept_idx])

    # --- global reference fit (init='auto'): one serial fit of the mean ----
    ref_values: dict[str, float] = {}
    if init == "auto":
        mean_spec = np.nanmean(spectra, axis=0)
        try:
            m, p = _build_lmfit_model(spec)
            p = _moment_init(p, spec, mean_spec, axis)
            ref = m.fit(mean_spec, p, x=axis)
            if ref.success:
                ref_values = {name: par.value for name, par in
                              ref.params.items() if par.vary}
            else:
                notes.append("global reference fit did not converge; "
                             "falling back to per-pixel moment init")
        except Exception as e:
            notes.append(f"global reference fit failed ({e}); falling back "
                         "to per-pixel moment init")

    # --- worker count / backend --------------------------------------------
    n_cores = os.cpu_count() or 1
    if n_jobs in (None, 0):
        n_jobs = 1
    elif n_jobs < 0:
        n_jobs = max(1, min(n_cores + 1 + n_jobs, 16))
    n_jobs = int(min(n_jobs, 16))
    chunk_size = max(64, int(chunk_size))
    chunks = _chunk_ranges(len(kept_idx), chunk_size)
    n_jobs = min(n_jobs, len(chunks))

    parallel_cls = None
    if n_jobs > 1:
        try:
            from joblib import Parallel, delayed  # noqa: F401
            parallel_cls = Parallel
        except Exception as e:  # pragma: no cover - joblib ships w/ sklearn
            notes.append(f"joblib unavailable ({e}); running serially")
            n_jobs = 1

    # --- memmap handoff for large cubes -------------------------------------
    # Workers receive a file-backed view (path + offset cross the process
    # boundary, not the array), so the cube is materialized exactly once.
    mm_dir = None
    worker_view = spectra
    if n_jobs > 1 and spectra.nbytes > 50e6:
        mm_dir = tempfile.mkdtemp(prefix="scilink_ppf_")
        mm_path = os.path.join(mm_dir, "spectra.mmap")
        mm = np.memmap(mm_path, dtype=np.float64, mode="w+",
                       shape=spectra.shape)
        mm[:] = spectra
        mm.flush()
        worker_view = np.memmap(mm_path, dtype=np.float64, mode="r",
                                shape=spectra.shape)

    values = np.full((len(kept_idx), len(param_names)), np.nan)
    errors = np.full((len(kept_idx), len(param_names)), np.nan)
    r2 = np.full(len(kept_idx), np.nan)
    ok = np.zeros(len(kept_idx), bool)
    n_dispatched = 0
    deadline = (t0 + float(time_budget_s)) if time_budget_s else None

    def _absorb(results):
        nonlocal n_dispatched
        for start, v, e, r, o in results:
            n = len(o)
            values[start:start + n] = v
            errors[start:start + n] = e
            r2[start:start + n] = r
            ok[start:start + n] = o
            n_dispatched += n

    try:
        if n_jobs == 1:
            for start, stop in chunks:
                if deadline and time.monotonic() > deadline:
                    notes.append(
                        f"time budget ({time_budget_s:.0f}s) reached after "
                        f"{n_dispatched}/{len(kept_idx)} pixels — returning "
                        "partial maps (see coverage)")
                    break
                _absorb([_fit_chunk(worker_view, axis, spec, init,
                                    ref_values, start, stop, param_names)])
        else:
            from joblib import Parallel, delayed
            # Waves of n_jobs chunks: the deadline is checked between waves,
            # so at most one wave of work lands after the budget expires.
            for w in range(0, len(chunks), n_jobs):
                if deadline and time.monotonic() > deadline:
                    notes.append(
                        f"time budget ({time_budget_s:.0f}s) reached after "
                        f"{n_dispatched}/{len(kept_idx)} pixels — returning "
                        "partial maps (see coverage)")
                    break
                wave = chunks[w:w + n_jobs]
                _absorb(Parallel(n_jobs=n_jobs, backend="loky")(
                    delayed(_fit_chunk)(worker_view, axis, spec, init,
                                        ref_values, start, stop, param_names)
                    for start, stop in wave))
    finally:
        if mm_dir is not None:
            del worker_view
            _rmtree_with_retry(mm_dir)

    # --- assemble (H, W) maps ----------------------------------------------
    maps: dict[str, np.ndarray] = {}
    err_maps: dict[str, np.ndarray] = {}
    for k, pname in enumerate(param_names):
        vm = np.full(n_pix, np.nan)
        em = np.full(n_pix, np.nan)
        vm[kept_idx] = values[:, k]
        em[kept_idx] = errors[:, k]
        maps[pname] = vm.reshape(shape2d)
        err_maps[pname] = em.reshape(shape2d)
    r2_full = np.full(n_pix, np.nan)
    r2_full[kept_idx] = r2
    maps["R2"] = r2_full.reshape(shape2d)
    cov = np.zeros(n_pix, bool)
    cov[kept_idx] = ok
    coverage = cov.reshape(shape2d)

    elapsed = time.monotonic() - t0
    n_fit = int(ok.sum())
    n_failed = int(n_dispatched - ok[:n_dispatched].sum()) \
        if n_dispatched else 0
    stats = {"n_selected": int(len(kept_idx)), "n_fit": n_fit,
             "n_failed": n_failed,
             "n_skipped_budget": int(len(kept_idx) - n_dispatched),
             "n_jobs": n_jobs, "elapsed_s": round(elapsed, 3),
             "fit_rate_hz": round(n_dispatched / elapsed, 1)
             if elapsed > 0 else None}
    if n_dispatched and n_failed / max(n_dispatched, 1) > 0.5:
        notes.append(
            f"{n_failed}/{n_dispatched} fits failed or were skipped as "
            "dead/saturated — check the model spec (component types, "
            "windows) against the data before trusting the sparse maps")
    return {"maps": maps, "err_maps": err_maps, "coverage": coverage,
            "stats": stats, "notes": notes}


# ---------------------------------------------------------------------------
# Tool spec
# ---------------------------------------------------------------------------

from ._spec import ToolSpec  # noqa: E402

TOOL_SPEC = ToolSpec(
    name="fit_per_pixel",
    description=(
        "Parallel map-over-pixels harness for INDEPENDENT per-pixel lineshape "
        "fits over a (H,W,E) cube — the vetted way to fit a full frame when "
        "the science genuinely needs single-pixel resolution. Describes the "
        "model declaratively (registry primitives: gaussian / lorentzian / "
        "voigt / pseudo_voigt peaks; constant / linear / quadratic / "
        "power_law backgrounds; exponential decay; step_logistic / step_erf "
        "onsets) and fans the fits out over processes with memmapped input "
        "and BLAS pinning — a hand-written pixel loop cannot parallelize in "
        "this sandbox (exec-defined functions don't pickle). Returns value "
        "maps, sigma err maps from the fit covariance, an R2 map, and a "
        "coverage mask; a soft time budget returns partial maps instead of "
        "dying at the execution cap. Coupled/global fits (shared parameters, "
        "spatial regularization) are out of scope."
    ),
    signature=(
        "fit_per_pixel(data, axis, model, mask=None, init='auto', "
        "bounds=None, n_jobs=-1, chunk_size=2048, time_budget_s=None) -> dict"
    ),
    import_line=(
        "from scilink.skills._shared.parallel_pixel_fit import fit_per_pixel"),
    parameters={
        "data": "Datacube (H,W,E) float array (or (pixels,E)); energy/"
                "wavelength trailing. Passed raw — do your despiking first "
                "if the data needs it.",
        "axis": "1-D spectral axis of length E (same units the model hints "
                "use).",
        "model": "Declarative composite: a type string ('gaussian'), a "
                 "component dict, or a list summed left to right, e.g. "
                 "[{'type':'gaussian','window':(600,640),'sigma':15}, "
                 "'constant']. Dict keys: 'type' (registry name), optional "
                 "'window' (lo,hi) = peak-center search range + bounds, any "
                 "param init value by name ('center', 'sigma', 'amplitude' "
                 "— amplitude is AREA, lmfit convention), '<param>_min'/"
                 "'<param>_max' bounds, 'fix' {param: value} to freeze, "
                 "'prefix' to name the component ('p1_'). ADD components "
                 "for multi-peak spectra; NARROW a peak's window if it "
                 "locks onto a neighbor.",
        "mask": "(H,W) bool — fit ONLY these pixels (others return NaN, "
                "coverage False). Pass the gate's fit_mask / your own "
                "bright-region mask; this is the single biggest cost lever.",
        "init": "'auto' (default: one serial fit of the field-mean spectrum "
                "seeds every pixel, then each peak center snaps to the "
                "pixel's local maximum — robust for maps with spatial "
                "drift); 'moments' (pure per-pixel heuristics — use when "
                "the field mean is unrepresentative, e.g. a few bright "
                "pixels in a dark frame); 'spec' (use exactly the values "
                "in the model spec — use when you already fitted a "
                "reference spectrum yourself).",
        "bounds": "Extra {param: (lo,hi)} applied on top of the spec; bare "
                  "names ('sigma') apply to every component that has the "
                  "param, prefixed names ('gaus1_sigma') to one. TIGHTEN to "
                  "stop parameter runaway on weak pixels; LOOSEN if maps "
                  "saturate at a bound (visible as flat plateaus).",
        "n_jobs": "Worker processes. -1 (default) = all cores capped at 16; "
                  "1 = serial in-process (debugging / tiny masks — skips "
                  "process + memmap overhead). LOWER on a shared node; "
                  "RAISING beyond physical cores never helps.",
        "chunk_size": "Pixels per dispatched task (default 2048). RAISE for "
                      "very fast fits (amortize IPC further); LOWER (e.g. "
                      "512) for slow multi-component fits so the time-budget "
                      "waves stay responsive.",
        "time_budget_s": "Soft wall-clock cap. Dispatching stops at the "
                         "deadline and PARTIAL maps return with coverage "
                         "marking what was fit — set it a couple of minutes "
                         "below the sandbox execution cap so you return "
                         "data instead of timing out. None (default) = "
                         "fit everything.",
    },
    required=["data", "axis", "model"],
    agents=["hyperspectral"],
    when_to_use=(
        "Full-frame (or large-mask) per-pixel iterative fitting that a "
        "Python pixel loop would make unacceptably slow — e.g. mapping peak "
        "position/width/ratio at native resolution over 10^5+ pixels. Use "
        "INSTEAD of writing a per-pixel lmfit/curve_fit loop whenever the "
        "model is expressible from the registry primitives; keep hand-rolled "
        "loops for models the registry cannot express (then mask + bin per "
        "the size-budget rules)."
    ),
    returns=(
        "dict: maps {param -> (H,W) float map, incl. derived fwhm/height "
        "and an R2 map}, err_maps {param -> (H,W) sigma from covariance}, "
        "coverage ((H,W) bool: fit succeeded), stats (n_selected/n_fit/"
        "n_failed/n_skipped_budget/n_jobs/elapsed_s/fit_rate_hz), notes "
        "(budget stops, fallbacks — surface these in your description)."
    ),
    example=(
        "r = fit_per_pixel(hspy_data, axis,\n"
        "                  [{'type': 'gaussian', 'window': (600, 640)},\n"
        "                   'constant'],\n"
        "                  mask=fit_mask, time_budget_s=240)\n"
        "peak_pos = r['maps']['gaus1_center']\n"
        "peak_pos_err = r['err_maps']['gaus1_center']\n"
        "maps = {'Peak_Position': peak_pos, 'Peak_Position_err': peak_pos_err,\n"
        "        'R2': r['maps']['R2']}"
    ),
)
