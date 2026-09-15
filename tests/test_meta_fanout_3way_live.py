"""Live test: 3-WAY fan-out over image + datacube + spectrum of one sample.

The strongest heterogeneous-concurrency case: HAADF image ->ImageAnalysisAgent,
EELS datacube ->HyperspectralAnalysisAgent, XPS spectrum ->CurveFittingAgent,
all three concurrently, full-mesh aux, then a 3-dataset fusion.

  export AWS_BEARER_TOKEN_BEDROCK=...  AWS_REGION_NAME=us-east-1
  UNSAFE_EXECUTION_OK=true conda run -n scilink python tests/test_meta_fanout_3way_live.py --full
"""
import argparse, json, os, sys, tempfile
import numpy as np

MODEL = "bedrock/us.anthropic.claude-opus-4-8"


def make(d, n=32, nE=100):
    from scipy.ndimage import gaussian_filter
    r = np.random.RandomState(13)
    field = gaussian_filter(r.randn(n, n), 3.0)
    phase = (field > np.percentile(field, 45)).astype(np.float32)   # 1 = metal

    haadf = gaussian_filter(0.62 * phase + 0.32 * (1 - phase) + r.normal(0, 0.03, (n, n)), 0.6).astype(np.float32)

    E = np.arange(nE)
    g = lambda c, w, a: a * np.exp(-0.5 * ((E - c) / w) ** 2)
    s_metal, s_oxide = g(30, 4, 1.0) + g(48, 5, 0.4), g(34, 4, 1.0) + g(52, 5, 0.8)
    cube = np.empty((n, n, nE), np.float32)
    for j in range(n):
        for i in range(n):
            cube[j, i] = (s_metal if phase[j, i] > 0.5 else s_oxide) + 0.05 + r.normal(0, 0.02, nE)

    be = np.linspace(450, 470, 400)
    gx = lambda c, w, a: a * np.exp(-0.5 * ((be - c) / w) ** 2)
    xps = (gx(454, 0.9, 0.62) + gx(460.1, 1.0, 0.3) + gx(458.6, 1.1, 0.38) + gx(464.4, 1.2, 0.18)
           + 0.05 + r.normal(0, 0.012, be.size))

    def save(name, arr, meta):
        p = os.path.join(d, name + ".npy"); np.save(p, arr)
        mp = os.path.join(d, name + ".json"); json.dump(meta, open(mp, "w"), indent=2)
        return {"data_path": p, "metadata": mp}

    img = save("haadf", haadf.astype(np.float32), {
        "technique": "STEM-HAADF imaging", "sample": "TiOx thin film", "region_id": "R1",
        "field_of_view_nm": 8.0, "pixels": f"{n}x{n}",
        "description": f"Z-contrast image of region R1, pixel-aligned with the EELS datacube."})
    cubep = save("eels", cube, {
        "technique": "STEM-EELS spectrum image", "sample": "TiOx thin film", "region_id": "R1",
        "field_of_view_nm": 8.0, "shape": f"{n}x{n}x{nE}",
        "energy_range": {"start": 450.0, "end": 450.0 + nE, "units": "eV"},
        "description": f"EELS datacube co-registered to the SAME {n}x{n} R1 grid as the HAADF."})
    xpsp = save("xps", np.vstack([be, xps]).astype(np.float32).T, {
        "technique": "XPS Ti 2p core level", "sample": "TiOx thin film",
        "x_axis": "binding energy (eV)",
        "description": "Bulk-averaged Ti 2p chemical state (metallic Ti + TiO2) of the SAME film."})
    return img, cubep, xpsp


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        print("Set AWS_BEARER_TOKEN_BEDROCK (+ AWS_REGION_NAME)."); sys.exit(2)
    from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent, MetaMode
    d = tempfile.mkdtemp(prefix="fanout_3way_"); img, cube, xps = make(d)

    with tempfile.TemporaryDirectory() as bd:
        ag = MetaOrchestratorAgent(base_dir=bd, api_key=None, base_url=None,
                                   model_name=MODEL, meta_mode=MetaMode.AUTONOMOUS)
        print("### GATE: 3-way (HAADF image + EELS datacube + XPS spectrum) ###")
        v = json.loads(ag._assess_complementarity([
            {"path": img["data_path"], "metadata": img["metadata"]},
            {"path": cube["data_path"], "metadata": cube["metadata"]},
            {"path": xps["data_path"], "metadata": xps["metadata"]}]))
        print(f"  verdict={v.get('verdict')} conf={v.get('confidence')} "
              f"join={str(v.get('join_axis'))[:80]!r} fanout={len(v.get('fanout_set',[]))}")
        gate_ok = v.get("verdict") in ("complementary", "partially_complementary") and len(v.get("fanout_set", [])) == 3
        print(f"  gate -> {'PASS' if gate_ok else 'CHECK'}")

        res = {"gate_3way": gate_ok}
        if args.full:
            ag._complementarity_cache.clear()
            print("\n### FULL: 3-way delegate_to_analyses ###")
            out = json.loads(ag._run_fanout([
                {"data_path": img["data_path"], "metadata": img["metadata"], "label": "HAADF image",
                 "task": f"Analyze the HAADF-STEM image at {img['data_path']}: segment the contrast phases and report area fractions."},
                {"data_path": cube["data_path"], "metadata": cube["metadata"], "label": "EELS datacube",
                 "task": f"Analyze the EELS datacube at {cube['data_path']}: decompose into spectral components and map them spatially."},
                {"data_path": xps["data_path"], "metadata": xps["metadata"], "label": "XPS Ti2p",
                 "task": f"Fit the XPS Ti 2p spectrum at {xps['data_path']}: identify metallic Ti vs TiO2 and report relative areas."},
            ]))
            print("FANOUT:", json.dumps(out, indent=2)[:1400])
            res["branches_with_output"] = out.get("branches_with_output")
            if out.get("status") == "success" and out.get("branches_with_output", 0) >= 2:
                idxs = [r["delegation_index"] for r in out["results"] if r["produced_output"]]
                fout = json.loads(ag._fuse_delegations(
                    idxs, focus="Reconcile the HAADF structure, the EELS spatial chemistry, and the bulk XPS chemical state."))
                print("FUSION:", json.dumps(fout, indent=2)[:1700])
                res["full_3way"] = fout.get("status") == "success" and bool(fout.get("detailed_analysis"))
            else:
                res["full_3way"] = False
        print("\nSUMMARY:", res)
        sys.exit(0 if all(v for v in res.values() if isinstance(v, bool)) else 1)


if __name__ == "__main__":
    main()
