"""User-provided EELS core-loss edge analysis — the kind of script a
scientist attaches in chat and asks the agent to use (adapting it to the
session's data as needed).

Pipeline
--------
1. Load an (N, 2) array of energy loss (eV) vs intensity.
2. For each expected edge, fit a power-law background A * E^-r in a
   pre-edge window and subtract it.
3. Find the edge onset as the first energy where the background-subtracted
   signal exceeds 5x the pre-edge residual noise.
4. Fit Gaussians to the near-edge white lines / peaks within a window
   after the onset and report their centres, widths and areas.
5. Write results as JSON and a figure.

Usage:  python user_edge_fit.py spectrum.npy [out_dir]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

# Expected core-loss edges in the example (eV): pre-edge fit window, onset
# search window, and how far past the onset to look for near-edge peaks.
EDGES = {
    "Ti_L23": {"pre": (400, 445), "onset": (445, 475), "post_ev": 20, "n_peaks": 2},
    "O_K":    {"pre": (490, 525), "onset": (525, 545), "post_ev": 25, "n_peaks": 2},
}
NOISE_SIGMA_THRESHOLD = 5.0


def power_law(E, A, r):
    return A * np.power(E, -r)


def gaussians(E, *p):
    y = np.zeros_like(E, dtype=float)
    for i in range(0, len(p), 3):
        a, mu, sig = p[i:i + 3]
        y += a * np.exp(-0.5 * ((E - mu) / sig) ** 2)
    return y


def analyze_edge(E, I, name, cfg):
    lo, hi = cfg["pre"]
    m = (E >= lo) & (E <= hi)
    (A, r), _ = curve_fit(power_law, E[m], I[m], p0=(I[m][0] * E[m][0] ** 3, 3.0), maxfev=20000)
    bg = power_law(E, A, r)
    sig = I - bg
    noise = float(np.std(sig[m]))

    olo, ohi = cfg["onset"]
    w = (E >= olo) & (E <= ohi)
    above = np.where(w & (sig > NOISE_SIGMA_THRESHOLD * noise))[0]
    if len(above) == 0:
        return {"edge": name, "detected": False, "background": {"A": A, "r": r}}
    onset = float(E[above[0]])

    pw = (E >= onset) & (E <= onset + cfg["post_ev"])
    Ew, Sw = E[pw], sig[pw]
    # seed the peaks at the n largest local maxima in the window
    order = np.argsort(Sw)[::-1]
    seeds = []
    for idx in order:
        if all(abs(Ew[idx] - s) > 2.0 for s in seeds):
            seeds.append(Ew[idx])
        if len(seeds) == cfg["n_peaks"]:
            break
    p0 = []
    for mu in sorted(seeds):
        p0 += [float(Sw.max()), float(mu), 1.5]
    try:
        popt, _ = curve_fit(gaussians, Ew, Sw, p0=p0, maxfev=20000)
        peaks = [{"center_eV": float(popt[i + 1]), "fwhm_eV": float(2.3548 * abs(popt[i + 2])),
                  "area": float(abs(popt[i]) * abs(popt[i + 2]) * np.sqrt(2 * np.pi))}
                 for i in range(0, len(popt), 3)]
        peaks.sort(key=lambda p: p["center_eV"])
    except RuntimeError:
        peaks = []
    return {"edge": name, "detected": True, "onset_eV": onset,
            "background": {"A": float(A), "r": float(r)}, "peaks": peaks}


def main(path, out_dir="."):
    data = np.load(path)
    E, I = data[:, 0], data[:, 1]
    results = {name: analyze_edge(E, I, name, cfg) for name, cfg in EDGES.items()}
    ti = results["Ti_L23"].get("peaks", [])
    if len(ti) == 2:
        results["Ti_L3_L2_separation_eV"] = ti[1]["center_eV"] - ti[0]["center_eV"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "edge_fit_results.json").write_text(json.dumps(results, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(E, I, lw=0.8, label="spectrum")
    for name, res in results.items():
        if isinstance(res, dict) and res.get("detected"):
            ax.axvline(res["onset_eV"], ls="--", lw=0.8, label=f"{name} onset {res['onset_eV']:.1f} eV")
            for p in res.get("peaks", []):
                ax.axvline(p["center_eV"], color="gray", lw=0.5)
    ax.set_xlabel("Energy loss (eV)")
    ax.set_ylabel("Intensity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "edge_fit.png", dpi=150)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ".")
