"""Offline unit tests for the fan-out steering reduction (#296 phase d).

Synthetic series with planted ground truth: a composition change at a known
control value, a pure peak-shift series (must flag shift_dominated), an
intensity-drift series (must flag intensity_drift), sidecar/filename/index
control-variable extraction, and the failure paths.

  conda run -n scilink python tests/test_series_reduction.py
"""
import json
import os
import tempfile

import numpy as np

from scilink.skills._shared.series_reduction import reduce_series

RNG = np.random.default_rng(1)
results = {}


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def _g(x, c, s):
    return np.exp(-0.5 * ((x - c) / s) ** 2)


def _write_series(d, maker, temps=range(30, 165, 10), sidecar=True,
                  stem="s_{T:03d}C"):
    files = []
    for T in temps:
        x = np.linspace(400, 1800, 700)
        y = maker(x, T) + RNG.normal(0, 0.004, x.size)
        f = os.path.join(d, stem.format(T=T) + ".txt")
        np.savetxt(f, np.column_stack([x, y]))
        if sidecar:
            json.dump({"temperature_C": float(T)},
                      open(f.replace(".txt", ".json"), "w"))
        files.append(f)
    return files


def main():
    # 1) Composition change planted at T0 = 85.
    d = tempfile.mkdtemp()
    sig = lambda T: 1 / (1 + np.exp(-(T - 85.0) / 4.0))
    files = _write_series(d, lambda x, T: ((1 - sig(T)) * _g(x, 900, 40)
                                           + sig(T) * _g(x, 1300, 40)))
    out = reduce_series(files, out_dir=os.path.join(d, "red"), label="test")
    print("1) planted composition change at 85:")
    check("success", out["status"] == "success")
    check("change point within half a step of 85",
          abs(out["change_point"] - 85.0) <= 5.0)
    check("not flagged shift-dominated", not out["flags"]["shift_dominated"])
    check("control variable from sidecar",
          out["control_variable"]["source"].startswith("sidecar"))
    check("figure written", os.path.exists(out.get("score_curve_path", "")))
    check("json written", os.path.exists(out.get("reduction_json_path", "")))
    saved = json.load(open(out["reduction_json_path"]))
    check("json carries score curve + controls",
          len(saved.get("score1", [])) == len(files)
          and saved["controls"] == sorted(saved["controls"]))

    # 2) Pure peak shift (no composition change) -> shift_dominated flag,
    #    caution set, loadings fenced off.
    d = tempfile.mkdtemp()
    files = _write_series(d, lambda x, T: _g(x, 900 + 1.2 * (T - 30), 40))
    out = reduce_series(files, label="shift")
    print("2) pure peak shift:")
    check("shift: success", out["status"] == "success")
    check("shift_dominated flagged", out["flags"]["shift_dominated"])
    check("caution names loadings",
          "loadings" in out.get("caution", "").lower())

    # 3) Global intensity drift -> intensity_drift flag.
    d = tempfile.mkdtemp()
    files = _write_series(d, lambda x, T: (1 + 0.01 * (T - 30)) * _g(x, 900, 40))
    out = reduce_series(files, label="drift")
    print("3) intensity drift:")
    check("drift: success", out["status"] == "success")
    check("intensity_drift flagged", out["flags"]["intensity_drift"])

    # 4) No sidecars -> filename fallback; neither -> index.
    d = tempfile.mkdtemp()
    files = _write_series(d, lambda x, T: ((1 - sig(T)) * _g(x, 900, 40)
                                           + sig(T) * _g(x, 1300, 40)),
                          sidecar=False)
    out = reduce_series(files, label="fname")
    print("4) control-variable fallbacks:")
    check("filename fallback", out["status"] == "success"
          and out["control_variable"]["source"] == "filename"
          and abs(out["change_point"] - 85.0) <= 5.0)
    d = tempfile.mkdtemp()
    files = _write_series(d, lambda x, T: ((1 - sig(T)) * _g(x, 900, 40)
                                           + sig(T) * _g(x, 1300, 40)),
                          sidecar=False, stem="frame_run{T:03d}x")
    # filenames still carry a number; strip to force index by renaming
    files2 = []
    for i, f in enumerate(sorted(files)):
        nf = os.path.join(os.path.dirname(f), f"frame_{chr(97 + i)}.txt")
        os.rename(f, nf); files2.append(nf)
    out = reduce_series(files2, label="idx")
    check("index fallback", out["status"] == "success"
          and out["control_variable"]["source"] == "index")

    # 5) Failure paths: too few points; disjoint x-ranges.
    print("5) failure paths:")
    out = reduce_series(files2[:3])
    check("too few points -> error", out["status"] == "error")
    d = tempfile.mkdtemp()
    fs = []
    for i in range(5):
        x = np.linspace(100 + 1000 * i, 600 + 1000 * i, 100)
        f = os.path.join(d, f"p{i}.txt")
        np.savetxt(f, np.column_stack([x, np.ones_like(x)]))
        fs.append(f)
    out = reduce_series(fs)
    check("disjoint ranges -> error", out["status"] == "error")

    # 6) Sign convention: score 1 ends higher than it starts.
    d = tempfile.mkdtemp()
    files = _write_series(d, lambda x, T: (1 - sig(T)) * _g(x, 900, 40))
    out = reduce_series(files, out_dir=os.path.join(d, "red"))
    saved = json.load(open(out["reduction_json_path"]))
    print("6) sign convention:")
    check("score1 increases end-over-start",
          saved["score1"][-1] >= saved["score1"][0])

    print("\n" + "=" * 50)
    npass = sum(results.values())
    print(f"SERIES REDUCTION: {npass}/{len(results)} checks passed")
    for k, v in results.items():
        if not v:
            print("  FAILED:", k)
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
