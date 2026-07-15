"""Cheap unsupervised reduction of a measurement series (fan-out steering).

The #296 phase-(d) steering payload: answer only "WHERE along the control
variable does this series change, and how sharply?" via mean-centered SVD —
a change DETECTOR, not a re-derivation (no fitting, no peak model, no phase
assumptions). Available at branch-launch time, when the companion branch's
own proper analysis does not exist yet.

The known artifacts are OWNED here (computed flags, not prompt prose):

- Shifting peaks (thermal expansion, band shifts) make SVD produce
  derivative-shaped components. Detected by correlating loading 1 against
  d(mean spectrum)/dx and returned as ``shift_dominated``. A shift-dominated
  component still LOCATES the change correctly (the score curve remains a
  valid transition marker) but its loadings must NOT be read as component /
  species spectra — the ``caution`` field states this.
- Overall intensity drift dominating component 1: flagged as
  ``intensity_drift`` (score 1 tracks total intensity).
- Component sign ambiguity: fixed by convention (score 1 ends higher than
  it starts).
- Non-uniform / differing x-grids: resampled to a common grid and said so.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_CONTROL_KEY_RE = re.compile(
    r"temperature|temp\b|_temp|time|pressure|voltage|potential|field|"
    r"concentration|dose|current", re.IGNORECASE)
_N_GRID = 500
_SHIFT_R_THRESHOLD = 0.7
_DRIFT_R_THRESHOLD = 0.9

_CAUTION_SHIFT = (
    "shift-dominated component: the score curve still LOCATES the change "
    "correctly, but the loadings are translation artifacts and must NOT be "
    "interpreted as component/species spectra.")


def _load_curve(path: str):
    """Load one series file as (x, y). .npy or text; 1-D data gets an index
    x-axis; multi-column data uses the first two columns."""
    p = str(path)
    if p.lower().endswith(".npy"):
        arr = np.load(p)
    else:
        arr = np.loadtxt(p, comments=["#", "%", "//"])
    arr = np.atleast_1d(np.asarray(arr, dtype=float))
    if arr.ndim == 1:
        return np.arange(arr.size, dtype=float), arr
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T
    if arr.shape[1] == 1:
        return np.arange(arr.shape[0], dtype=float), arr[:, 0]
    return arr[:, 0], arr[:, 1]


def _control_value(path: str) -> tuple:
    """Best-effort control-variable value for one file: stem-matched sidecar
    JSON (a numeric field whose key looks like a control variable), else the
    last number in the filename, else None. Returns (value, source)."""
    p = Path(path)
    sidecar = p.with_suffix(".json")
    if sidecar.exists():
        try:
            with open(sidecar, "r", errors="replace") as fh:
                meta = json.load(fh)
            if isinstance(meta, dict):
                for k, v in meta.items():
                    if _CONTROL_KEY_RE.search(str(k)) and isinstance(
                            v, (int, float)) and not isinstance(v, bool):
                        return float(v), f"sidecar:{k}"
        except Exception:  # noqa: BLE001 - fall through to filename
            pass
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", p.stem.replace("_", " "))
    if nums:
        return float(nums[-1]), "filename"
    return None, None


def reduce_series(files: List[str], out_dir: Optional[str] = None,
                  label: str = "companion", n_grid: int = _N_GRID) -> dict:
    """Reduce a file series to score-vs-control + a change-point estimate.

    Returns a dict with ``status``; on success: ``change_point``,
    ``change_sharpness`` (steepest single step as a fraction of the score
    range), ``control_variable`` (name/source/range), ``variance_explained``,
    ``flags`` (shift_dominated / intensity_drift / resampled, with their
    correlations), ``caution`` (set when loadings must not be interpreted),
    and — when ``out_dir`` is given — ``score_curve_path`` /
    ``reduction_json_path`` artifacts. Never raises."""
    try:
        files = [str(f) for f in (files or [])]
        if len(files) < 4:
            return {"status": "error",
                    "error": f"need >= 4 series points, got {len(files)}"}
        curves, controls, sources = [], [], []
        for f in files:
            x, y = _load_curve(f)
            if x.size < 8:
                return {"status": "error", "error": f"{f}: too few channels"}
            curves.append((x, y))
            cv, src = _control_value(f)
            controls.append(cv)
            sources.append(src)
        if any(c is None for c in controls):
            controls = list(range(len(files)))
            source = "index"
        else:
            source = next(s for s in sources if s)
        controls = np.asarray(controls, dtype=float)

        # Common grid: intersection of x-ranges (flags a resample if the
        # grids differ at all).
        lo = max(float(x.min()) for x, _ in curves)
        hi = min(float(x.max()) for x, _ in curves)
        if not hi > lo:
            return {"status": "error", "error": "series x-ranges do not overlap"}
        grid = np.linspace(lo, hi, int(n_grid))
        same_grid = all(x.size == curves[0][0].size
                        and np.allclose(x, curves[0][0]) for x, _ in curves)
        mat = np.vstack([np.interp(grid, x, y) for x, y in curves])

        order = np.argsort(controls)
        controls, mat = controls[order], mat[order]

        total_intensity = mat.sum(axis=1)
        mean_spec = mat.mean(axis=0)
        centered = mat - mean_spec
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        var = s ** 2
        var_explained = (var / var.sum())[:2].tolist() if var.sum() > 0 else [0.0, 0.0]
        score1 = u[:, 0] * s[0]
        loading1 = vt[0]
        # Sign convention: score 1 ends higher than it starts.
        if score1[-1] < score1[0]:
            score1, loading1 = -score1, -loading1
        score2 = (u[:, 1] * s[1]) if s.size > 1 else np.zeros_like(score1)

        # Change point: steepest step of score 1 per unit control variable.
        dc = np.diff(controls)
        dc[dc == 0] = np.finfo(float).eps
        rate = np.abs(np.diff(score1)) / dc
        i = int(np.argmax(rate))
        change_point = float((controls[i] + controls[i + 1]) / 2.0)
        rng = float(score1.max() - score1.min())
        sharpness = float(abs(score1[i + 1] - score1[i]) / rng) if rng > 0 else 0.0

        def _abs_corr(a, b):
            if np.std(a) == 0 or np.std(b) == 0:
                return 0.0
            return float(abs(np.corrcoef(a, b)[0, 1]))

        shift_r = _abs_corr(loading1, np.gradient(mean_spec))
        drift_r = _abs_corr(score1, total_intensity)
        flags = {
            "shift_dominated": shift_r >= _SHIFT_R_THRESHOLD,
            "loading1_vs_dmean_r": round(shift_r, 3),
            "intensity_drift": drift_r >= _DRIFT_R_THRESHOLD,
            "score1_vs_total_intensity_r": round(drift_r, 3),
            "resampled_to_common_grid": not same_grid,
        }

        out = {
            "status": "success",
            "label": label,
            "n_points": int(len(files)),
            "n_channels": int(grid.size),
            "control_variable": {
                "source": source,
                "min": float(controls.min()), "max": float(controls.max()),
            },
            "change_point": change_point,
            "change_sharpness": round(sharpness, 3),
            "variance_explained": [round(v, 3) for v in var_explained],
            "flags": flags,
        }
        if flags["shift_dominated"]:
            out["caution"] = _CAUTION_SHIFT

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            jpath = os.path.join(out_dir, "reduction.json")
            with open(jpath, "w") as fh:
                json.dump({**out, "score1": score1.tolist(),
                           "controls": controls.tolist()}, fh, indent=2)
            out["reduction_json_path"] = jpath
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(controls, score1, "o-", label="score 1")
                if s.size > 1 and var_explained[1] > 0.02:
                    ax.plot(controls, score2, ".--", alpha=0.4, label="score 2")
                ax.axvline(change_point, ls=":", color="tab:red",
                           label=f"change point ≈ {change_point:g}")
                ax.set_xlabel(f"control variable ({source})")
                ax.set_ylabel("SVD score")
                ax.set_title(f"{label}: unsupervised change detection "
                             f"(PC1 {var_explained[0]:.0%} of variance)")
                ax.legend(fontsize=8)
                fig.tight_layout()
                fpath = os.path.join(out_dir, "score_curve.png")
                fig.savefig(fpath, dpi=110)
                plt.close(fig)
                out["score_curve_path"] = fpath
            except Exception as e:  # noqa: BLE001 - figure is best-effort
                logger.warning(f"series reduction: could not plot: {e}")
        return out
    except Exception as e:  # noqa: BLE001 - steering must never break a fan-out
        logger.warning(f"series reduction failed for {label}: {e}")
        return {"status": "error", "error": str(e)}
