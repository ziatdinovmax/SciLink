"""Live STRESS battery for the meta fan-out gate + a same-modality full run.

Covers data-type variety the other live tests don't: same-modality
spectroscopy pairs, thermal curves with a shared axis, temporal before/after
(directionality probe), survey/detail, a mis-routed results table, two-region,
plus unrelated/redundant sanity. Then a DSC+TGA end-to-end (two CurveFitting
branches with a shared-temperature operand).

  export AWS_BEARER_TOKEN_BEDROCK=...  AWS_REGION_NAME=us-east-1
  conda run -n scilink python tests/test_meta_fanout_stress_live.py          # gate battery
  UNSAFE_EXECUTION_OK=true conda run -n scilink python ... --full            # + DSC/TGA run
"""
import argparse, json, os, sys, tempfile
import numpy as np

MODEL = "bedrock/us.anthropic.claude-opus-4-8"
RNG = np.random.RandomState(3)


# ---- 1D helpers ----
def _g(x, c, w, a):       # gaussian
    return a * np.exp(-0.5 * ((x - c) / w) ** 2)

def _step(x, c, w, h):    # sigmoid mass-loss step (drops by h)
    return -h / (1.0 + np.exp(-(x - c) / w))

def _curve(x, y):
    return np.vstack([x, y]).astype(np.float32).T


def _save(d, name, arr, meta):
    p = os.path.join(d, name + ".npy"); np.save(p, arr)
    mp = os.path.join(d, name + ".json"); json.dump(meta, open(mp, "w"), indent=2)
    return {"path": p, "metadata": mp}


def _img_two_phase(n=64, frac=0.6, seed=0):
    from scipy.ndimage import gaussian_filter
    r = np.random.RandomState(seed)
    f = gaussian_filter(r.randn(n, n), 3.0)
    ph = (f > np.percentile(f, 100 * (1 - frac))).astype(np.float32)
    return gaussian_filter(0.6 * ph + 0.3 * (1 - ph) + r.normal(0, 0.03, (n, n)), 0.6).astype(np.float32)


def make_fixtures(d):
    F = {}
    # --- thermal: DSC + TGA on a shared temperature axis (30..600 C) ---
    T = np.linspace(30, 600, 600)
    tga = 100.0 + _step(T, 110, 12, 6) + _step(T, 410, 18, 40) + RNG.normal(0, 0.2, T.size)
    dsc = (_g(T, 110, 12, -3.0)      # endotherm: dehydration (coincides w/ TGA step)
           + _g(T, 250, 15, 2.0)     # exotherm: crystallization (NO mass change)
           + _g(T, 410, 18, -5.0)    # endotherm: decomposition (coincides w/ TGA step)
           + 0.01 * (T - 30) + RNG.normal(0, 0.05, T.size))
    F["tga"] = _save(d, "tga_curve", _curve(T, tga), {
        "technique": "Thermogravimetric analysis (TGA)", "sample": "hydrated metal carbonate",
        "x_axis": "temperature (C)", "y_axis": "mass (%)", "heating_rate": "10 C/min",
        "description": "Mass vs temperature; dehydration + decomposition steps."})
    F["dsc"] = _save(d, "dsc_curve", _curve(T, dsc), {
        "technique": "Differential scanning calorimetry (DSC)", "sample": "hydrated metal carbonate",
        "x_axis": "temperature (C)", "y_axis": "heat flow (mW, endo down)", "heating_rate": "10 C/min",
        "description": "Heat flow vs temperature on the SAME temperature axis as the TGA."})

    # --- vibrational: Raman + FTIR of the same carbonate (different ranges) ---
    xr = np.linspace(100, 1600, 700)
    raman = _g(xr, 1086, 6, 1.0) + _g(xr, 712, 6, 0.3) + _g(xr, 1432, 9, 0.2) + 0.02 + RNG.normal(0, 0.01, xr.size)
    xi = np.linspace(400, 2000, 800)
    ftir = _g(xi, 1420, 30, 0.9) + _g(xi, 875, 12, 0.6) + _g(xi, 712, 10, 0.4) + 0.05 + RNG.normal(0, 0.01, xi.size)
    F["raman"] = _save(d, "raman", _curve(xr, raman), {
        "technique": "Raman spectroscopy", "sample": "calcite CaCO3",
        "x_axis": "Raman shift (cm-1)", "description": "Raman-active CO3 modes."})
    F["ftir"] = _save(d, "ftir", _curve(xi, ftir), {
        "technique": "FTIR spectroscopy", "sample": "calcite CaCO3",
        "x_axis": "wavenumber (cm-1)", "description": "IR-active CO3 modes of the SAME sample."})

    # --- electronic: PL + UV-Vis of the same semiconductor ---
    wl = np.linspace(350, 800, 600)
    absorb = 1.0 / (1.0 + np.exp((wl - 520) / 8)) + 0.02 + RNG.normal(0, 0.01, wl.size)   # abs edge ~520nm
    pl = _g(wl, 560, 18, 1.0) + 0.01 + RNG.normal(0, 0.01, wl.size)                       # emission ~560nm
    F["uvvis"] = _save(d, "uvvis", _curve(wl, absorb), {
        "technique": "UV-Vis absorption", "sample": "CdSe quantum dots",
        "x_axis": "wavelength (nm)", "description": "Absorption edge."})
    F["pl"] = _save(d, "pl", _curve(wl, pl), {
        "technique": "Photoluminescence (PL)", "sample": "CdSe quantum dots",
        "x_axis": "wavelength (nm)", "description": "Emission of the SAME quantum dots."})

    # --- temporal before/after images (same region, two time points) ---
    F["before"] = _save(d, "anneal_t0", _img_two_phase(frac=0.30, seed=1), {
        "technique": "SEM imaging", "sample": "alloy coupon", "region_id": "R1",
        "time_point": "t=0 (as-deposited)", "description": "Microstructure BEFORE annealing."})
    F["after"] = _save(d, "anneal_t1", _img_two_phase(frac=0.55, seed=2), {
        "technique": "SEM imaging", "sample": "alloy coupon", "region_id": "R1",
        "time_point": "t=2h at 600C (annealed)", "description": "SAME region AFTER annealing."})

    # --- survey + detail (same sample, different magnification) ---
    F["survey"] = _save(d, "survey_lowmag", _img_two_phase(n=96, seed=3), {
        "technique": "SEM imaging", "sample": "catalyst pellet", "magnification": "500x",
        "field_of_view_um": 200.0, "description": "Wide overview."})
    F["detail"] = _save(d, "detail_highmag", _img_two_phase(n=64, seed=4), {
        "technique": "SEM imaging", "sample": "catalyst pellet", "magnification": "20000x",
        "field_of_view_um": 5.0, "description": "High-mag detail of one region of the survey."})

    # --- mis-routed: a RESULTS TABLE (planning data, not analysis) + an image ---
    import csv
    tbl = os.path.join(d, "results_table.csv")
    with open(tbl, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["sample", "temp_C", "pressure_bar", "yield_pct"])
        for i in range(12):
            w.writerow([f"S{i}", 200 + 10 * i, 1 + 0.5 * i, round(40 + 3 * i + RNG.rand() * 5, 1)])
    json.dump({"description": "process results table: rows=runs, cols=conditions+yield"},
              open(os.path.join(d, "results_table.json"), "w"))
    F["table"] = {"path": tbl, "metadata": os.path.join(d, "results_table.json")}
    F["img_for_table"] = _save(d, "micro_img", _img_two_phase(seed=5), {
        "technique": "optical microscopy", "sample": "catalyst pellet",
        "description": "Micrograph (unrelated to the process table)."})

    # --- two XPS from DIFFERENT regions of one sample (spatial heterogeneity) ---
    be = np.linspace(450, 470, 400)
    F["xps_core"] = _save(d, "xps_core", _curve(be, _g(be, 454, 0.9, 0.9) + _g(be, 458.6, 1.1, 0.3) + 0.05 + RNG.normal(0, 0.01, be.size)), {
        "technique": "XPS Ti 2p", "sample": "oxidized coupon", "region_id": "core (metal-rich)",
        "x_axis": "binding energy (eV)", "description": "Spot 1: mostly metallic Ti."})
    F["xps_edge"] = _save(d, "xps_edge", _curve(be, _g(be, 454, 0.9, 0.3) + _g(be, 458.6, 1.1, 0.9) + 0.05 + RNG.normal(0, 0.01, be.size)), {
        "technique": "XPS Ti 2p", "sample": "oxidized coupon", "region_id": "edge (oxide-rich)",
        "x_axis": "binding energy (eV)", "description": "Spot 2 (different location): mostly TiO2."})

    # --- unrelated sanity: Raman of polymer + DSC of a metal alloy ---
    F["raman_poly"] = _save(d, "raman_polymer", _curve(xr, _g(xr, 1450, 10, 1.0) + 0.02 + RNG.normal(0, 0.01, xr.size)), {
        "technique": "Raman spectroscopy", "sample": "polyethylene film", "x_axis": "Raman shift (cm-1)"})
    # (DSC of metal alloy is F['dsc'] of a different sample) — reuse a fresh one:
    F["dsc_metal"] = _save(d, "dsc_metal", _curve(T, _g(T, 480, 10, -4.0) + 0.01 * (T - 30) + RNG.normal(0, 0.05, T.size)), {
        "technique": "DSC", "sample": "Al-Si casting alloy", "x_axis": "temperature (C)"})

    # --- redundant sanity: two FTIR of the same sample, same technique ---
    F["ftir_dup"] = _save(d, "ftir_rep2", _curve(xi, _g(xi, 1420, 30, 0.9) + _g(xi, 875, 12, 0.6) + _g(xi, 712, 10, 0.4) + 0.05 + RNG.normal(0, 0.01, xi.size)), {
        "technique": "FTIR spectroscopy", "sample": "calcite CaCO3", "x_axis": "wavenumber (cm-1)",
        "description": "Repeat FTIR acquisition of the same sample."})
    return F


def _agent(base_dir):
    from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent, MetaMode
    return MetaOrchestratorAgent(base_dir=base_dir, api_key=None, base_url=None,
                                 model_name=MODEL, meta_mode=MetaMode.AUTONOMOUS)


def _ds(F, *keys):
    return [{"path": F[k]["path"], "metadata": F[k]["metadata"]} for k in keys]


def gate_battery(F, base_dir):
    ag = _agent(base_dir)
    pos = lambda v: v["verdict"] in ("complementary", "partially_complementary") and len(v["fanout_set"]) >= 2
    neg = lambda v: v["verdict"] in ("unrelated", "redundant", "uncertain") and len(v["fanout_set"]) < 2
    obs = lambda v: True   # observational (genuinely ambiguous — record, don't grade)
    cases = [
        ("DSC + TGA (thermal, shared T axis)", _ds(F, "dsc", "tga"), pos, "complementary"),
        ("Raman + FTIR (same calcite)", _ds(F, "raman", "ftir"), pos, "complementary"),
        ("PL + UV-Vis (same QDs)", _ds(F, "pl", "uvvis"), pos, "complementary"),
        ("before/after SEM (same region, temporal)", _ds(F, "before", "after"), obs, "observe: complementary? directional?"),
        ("survey + detail (same sample, diff mag)", _ds(F, "survey", "detail"), obs, "observe"),
        ("image + results-table (mis-routed)", _ds(F, "img_for_table", "table"), neg, "decline (table is planning data)"),
        ("two XPS, different regions (spatial)", _ds(F, "xps_core", "xps_edge"), obs, "observe: complementary-spatial vs redundant"),
        ("Raman(polymer) + DSC(metal) — UNRELATED", _ds(F, "raman_poly", "dsc_metal"), neg, "decline"),
        ("two FTIR same sample — REDUNDANT", _ds(F, "ftir", "ftir_dup"), neg, "decline (redundant)"),
    ]
    results = {}
    for name, datasets, pred, expect in cases:
        v = json.loads(ag._assess_complementarity(datasets)); ag._complementarity_cache.clear()
        ok = False
        try: ok = bool(pred(v))
        except Exception: ok = False
        graded = pred is not obs
        tag = ("PASS" if ok else "FAIL") if graded else "OBSERVE"
        results[name] = (tag != "FAIL")
        print(f"\n### {name}\n  expect: {expect}")
        print(f"  -> verdict={v.get('verdict')} conf={v.get('confidence')} "
              f"join={str(v.get('join_axis'))[:70]!r} fanout={len(v.get('fanout_set',[]))}")
        if v.get("excluded_notes"): print(f"     notes: {str(v['excluded_notes'])[:160]}")
        print(f"  [{tag}]")
    graded = [k for k,(name) in zip(results, [c[2] for c in cases]) ]
    npass = sum(1 for c in cases if c[2] is not obs and results[c[0]])
    ngraded = sum(1 for c in cases if c[2] is not obs)
    print(f"\nGATE BATTERY: {npass}/{ngraded} graded passed ({len(cases)-ngraded} observational)")
    return npass == ngraded


def dsc_tga_full(F, base_dir):
    ag = _agent(base_dir)
    dsc, tga = F["dsc"], F["tga"]
    print("\n### FULL: DSC + TGA (two CurveFitting branches, shared T operand) ###")
    out = json.loads(ag._run_fanout([
        {"data_path": dsc["path"], "metadata": dsc["metadata"], "label": "DSC curve",
         "task": f"Fit/characterize the DSC heat-flow curve at {dsc['path']}: identify thermal "
                 "events (endo/exothermic) and their onset temperatures."},
        {"data_path": tga["path"], "metadata": tga["metadata"], "label": "TGA curve",
         "task": f"Characterize the TGA mass-loss curve at {tga['path']}: identify mass-loss steps, "
                 "their onset temperatures and magnitudes."},
    ]))
    print("FANOUT:", json.dumps(out, indent=2)[:1200])
    if out.get("status") != "success" or out.get("branches_with_output", 0) < 2:
        print("DSC/TGA: branches did not both produce output"); return False
    idxs = [r["delegation_index"] for r in out["results"] if r["produced_output"]]
    fout = json.loads(ag._fuse_delegations(
        idxs, focus="Which DSC events coincide with TGA mass loss (decomposition/dehydration) "
                    "vs which are mass-neutral (phase transitions)?"))
    print("FUSION:", json.dumps(fout, indent=2)[:1700])
    ok = fout.get("status") == "success" and bool(fout.get("detailed_analysis"))
    print(f"DSC/TGA: fused={ok}")
    return ok


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        print("Set AWS_BEARER_TOKEN_BEDROCK (+ AWS_REGION_NAME)."); sys.exit(2)
    d = tempfile.mkdtemp(prefix="fanout_stress_"); F = make_fixtures(d)
    res = {}
    with tempfile.TemporaryDirectory() as bd:
        res["gate_battery"] = gate_battery(F, os.path.join(bd, "g"))
    if args.full:
        with tempfile.TemporaryDirectory() as bd:
            res["dsc_tga_full"] = dsc_tga_full(F, os.path.join(bd, "f"))
    print("\n" + "=" * 60); print("SUMMARY:", res)
    sys.exit(0 if all(res.values()) else 1)


if __name__ == "__main__":
    main()
