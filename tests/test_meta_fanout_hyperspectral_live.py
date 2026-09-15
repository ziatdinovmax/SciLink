"""Live test: meta fan-out over an IMAGE + a co-registered HYPERSPECTRAL
datacube (Bedrock opus-4-8). The canonical STEM-HAADF + EELS-SI multimodal
pair — genuine pixel co-registration, so the two branches route to different
specialists (ImageAnalysisAgent vs HyperspectralAnalysisAgent).

  export AWS_BEARER_TOKEN_BEDROCK=...  AWS_REGION_NAME=us-east-1
  conda run -n scilink python tests/test_meta_fanout_hyperspectral_live.py          # gate only
  UNSAFE_EXECUTION_OK=true conda run -n scilink python ... --full                    # + end-to-end
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np

MODEL = "bedrock/us.anthropic.claude-opus-4-8"


def _make_datasets(d, n=32, nE=100):
    from scipy.ndimage import gaussian_filter
    rng = np.random.RandomState(11)
    # Two-phase spatial map (smoothed random field), shared by BOTH modalities.
    field = gaussian_filter(rng.randn(n, n), sigma=3.0)
    phase = (field > np.percentile(field, 45)).astype(np.float32)   # 1=metal

    # Co-registered HAADF image: Z-contrast tracks the phase map.
    haadf = (0.62 * phase + 0.32 * (1 - phase))
    haadf = gaussian_filter(haadf + rng.normal(0, 0.03, haadf.shape), 0.6).astype(np.float32)

    # EELS datacube: each pixel carries a spectrum set by its phase.
    E = np.arange(nE)
    def g(c, w, a):
        return a * np.exp(-0.5 * ((E - c) / w) ** 2)
    spec_metal = g(30, 4, 1.0) + g(48, 5, 0.4)            # metal L-edge
    spec_oxide = g(34, 4, 1.0) + g(52, 5, 0.8)            # chemically shifted + higher white-line
    cube = np.empty((n, n, nE), np.float32)
    for j in range(n):
        for i in range(n):
            base = spec_metal if phase[j, i] > 0.5 else spec_oxide
            cube[j, i] = base + 0.05 + rng.normal(0, 0.02, nE)
    cube = cube.astype(np.float32)

    def save(name, arr, meta):
        p = os.path.join(d, f"{name}.npy"); np.save(p, arr)
        mp = os.path.join(d, f"{name}.json"); json.dump(meta, open(mp, "w"), indent=2)
        return p, mp

    img = save("haadf_image", haadf, {
        "technique": "STEM-HAADF imaging", "sample": "TiOx thin film",
        "region_id": "R1", "field_of_view_nm": 8.0, "pixels": f"{n}x{n}",
        "description": f"Z-contrast image of region R1, {n}x{n}, pixel-aligned "
                       "with the EELS spectrum image below.",
    })
    cubep = save("eels_si", cube, {
        "technique": "STEM-EELS spectrum image (SI)", "sample": "TiOx thin film",
        "region_id": "R1", "field_of_view_nm": 8.0,
        "shape": f"{n}x{n}x{nE}",
        # EELS skill requires a numeric energy axis (start/end) for axis_2.
        "energy_range": {"start": 450.0, "end": 450.0 + nE, "units": "eV"},
        "energy_axis": "eV loss (Ti L-edge region)",
        "description": f"EELS datacube co-registered to the SAME {n}x{n} region "
                       "R1 grid as the HAADF image (one spectrum per pixel).",
    })
    return {"img": img, "cube": cubep}


def _agent(base_dir):
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent, MetaMode)
    return MetaOrchestratorAgent(base_dir=base_dir, api_key=None, base_url=None,
                                 model_name=MODEL, meta_mode=MetaMode.AUTONOMOUS)


def test_gate(P, base_dir):
    ag = _agent(base_dir)
    img, imgm = P["img"]; cube, cubem = P["cube"]
    v = json.loads(ag._assess_complementarity(
        [{"path": img, "metadata": imgm}, {"path": cube, "metadata": cubem}]))
    print("### GATE: HAADF image + co-registered EELS datacube")
    print(f"  verdict={v.get('verdict')} conf={v.get('confidence')} join={v.get('join_axis')!r}")
    print(f"  fanout_set={[os.path.basename(p) for p in v.get('fanout_set', [])]}")
    ok = v.get("verdict") in ("complementary", "partially_complementary") and len(v.get("fanout_set", [])) == 2
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_full(P, base_dir):
    ag = _agent(base_dir)
    img, imgm = P["img"]; cube, cubem = P["cube"]
    print("### FULL: delegate_to_analyses(HAADF image + EELS datacube) ###")
    out = json.loads(ag._run_fanout([
        {"data_path": img, "metadata": imgm, "label": "HAADF image",
         "task": f"Analyze the HAADF-STEM image at {img}: segment the two "
                 "contrast phases and report their area fractions."},
        {"data_path": cube, "metadata": cubem, "label": "EELS datacube",
         "task": f"Analyze the EELS spectrum-image datacube at {cube}: decompose "
                 "it into spectral components and map their spatial distribution."},
    ]))
    print("FANOUT:", json.dumps(out, indent=2)[:1400])
    if out.get("status") != "success" or out.get("branches_with_output", 0) < 2:
        print("FULL: branches did not both produce output"); return False
    idxs = [r["delegation_index"] for r in out["results"] if r["produced_output"]]
    fout = json.loads(ag._fuse_delegations(
        idxs, focus="Do the HAADF contrast phases spatially coincide with the "
                    "EELS spectral components, and what oxidation/phase picture results?"))
    print("FUSION:", json.dumps(fout, indent=2)[:1700])
    ok = fout.get("status") == "success" and bool(fout.get("detailed_analysis"))
    print(f"FULL: fused={ok}")
    return ok


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        print("Set AWS_BEARER_TOKEN_BEDROCK (+ AWS_REGION_NAME)."); sys.exit(2)
    d = tempfile.mkdtemp(prefix="fanout_hs_"); P = _make_datasets(d)
    res = {}
    with tempfile.TemporaryDirectory() as bd:
        res["gate"] = test_gate(P, os.path.join(bd, "g"))
    if args.full:
        with tempfile.TemporaryDirectory() as bd:
            res["full"] = test_full(P, os.path.join(bd, "f"))
    print("\nSUMMARY:", res)
    sys.exit(0 if all(res.values()) else 1)


if __name__ == "__main__":
    main()
