"""Live test for the meta-agent fan-out primitive (Bedrock opus-4-8).

Exercises, against a real LLM:
  - the complementarity GATE on same-modality and CROSS-MODALITY scenarios
    (complementary / redundant / unrelated) — the core new judgment;
  - optionally (--full) end-to-end delegate_to_analyses -> fuse_delegations,
    including a CROSS-MODAL run (image + spectrum) that routes the two
    branches to different specialist agents and fuses across modalities.

Run (real code execution needs a sandbox; UNSAFE_EXECUTION_OK=true bypasses):
  export AWS_BEARER_TOKEN_BEDROCK=...   AWS_REGION_NAME=us-east-1
  conda run -n scilink python tests/test_meta_fanout_live.py            # gates only
  UNSAFE_EXECUTION_OK=true conda run -n scilink python tests/test_meta_fanout_live.py --full
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np

MODEL = "bedrock/us.anthropic.claude-opus-4-8"


# ----------------------------------------------------------------------
# Realistic fixtures (so the analysis agents' validity gates don't reject
# them as synthetic test patterns).
# ----------------------------------------------------------------------

def _two_phase_image(rng, n=96, bright_frac=0.62):
    """A natural-looking two-phase microstructure (smoothed random field
    thresholded), NOT a periodic test pattern."""
    from scipy.ndimage import gaussian_filter
    field = gaussian_filter(rng.randn(n, n), sigma=4.0)
    thr = np.percentile(field, 100 * (1 - bright_frac))
    phase = (field > thr).astype(np.float32)          # 1 = bright (metal)
    img = 0.66 * phase + 0.30 * (1 - phase)
    img = gaussian_filter(img + rng.normal(0, 0.04, img.shape), 0.6)
    return img.astype(np.float32), float(phase.mean())


def _two_component_spectrum(rng, metal_area_frac=0.62):
    """An XPS-like Ti 2p doublet: a metallic and an oxide component on a
    sloped background with noise — exactly what curve fitting handles."""
    x = np.linspace(450.0, 470.0, 450)               # binding energy (eV)

    def g(a, c, w):
        return a * np.exp(-0.5 * ((x - c) / w) ** 2)
    # amplitudes chosen so metal:oxide AREA ~ metal_area_frac
    metal = g(0.95, 454.0, 0.9) + g(0.48, 460.1, 1.0)   # 2p3/2 + 2p1/2
    oxide = g(0.58, 458.6, 1.1) + g(0.29, 464.4, 1.2)
    bg = 0.05 + 0.0015 * (x - 450.0)
    y = metal + oxide + bg + rng.normal(0, 0.012, x.size)
    return np.vstack([x, y]).T.astype(np.float32)


def _save(d, name, arr, meta):
    p = os.path.join(d, f"{name}.npy")
    np.save(p, arr)
    mp = os.path.join(d, f"{name}.json")
    with open(mp, "w") as fh:
        json.dump(meta, fh, indent=2)
    return p, mp


def _make_datasets(d):
    rng = np.random.RandomState(7)
    img, bf = _two_phase_image(rng)
    spec = _two_component_spectrum(rng)
    # An UNRELATED spectrum: a single sharp Raman line of a different material.
    xr = np.linspace(100, 1200, 600)
    raman = (np.exp(-0.5 * ((xr - 520) / 4) ** 2)
             + 0.02 * rng.rand(xr.size))               # Si 520 cm^-1
    raman_arr = np.vstack([xr, raman]).T.astype(np.float32)
    # A REDUNDANT second XPS spectrum of the SAME sample (same technique).
    spec2 = _two_component_spectrum(np.random.RandomState(8))

    P = {}
    P["img"] = _save(d, "bse_image", img, {
        "technique": "BSE-SEM imaging", "sample": "oxidized Ti-6Al-4V coupon",
        "region_id": "R1", "field_of_view_um": 5.0,
        "description": f"Backscatter image of region R1; two-phase contrast "
                       f"(bright metallic ~{bf:.0%}, dark oxide).",
    })
    P["spec"] = _save(d, "xps_ti2p", spec, {
        "technique": "XPS Ti 2p core level", "sample": "oxidized Ti-6Al-4V coupon",
        "region_id": "R1", "x_axis": "binding energy (eV)",
        "description": "Ti 2p spectrum of the SAME coupon/region; metallic Ti "
                       "+ TiO2 oxide components.",
    })
    P["raman_other"] = _save(d, "si_raman", raman_arr, {
        "technique": "Raman spectroscopy", "sample": "silicon wafer (unrelated)",
        "x_axis": "Raman shift (cm-1)",
        "description": "Si 520 cm-1 line of an unrelated silicon reference.",
    })
    P["spec_dup"] = _save(d, "xps_ti2p_rep2", spec2, {
        "technique": "XPS Ti 2p core level", "sample": "oxidized Ti-6Al-4V coupon",
        "region_id": "R1", "x_axis": "binding energy (eV)",
        "description": "Repeat XPS Ti 2p acquisition of the SAME coupon/region.",
    })
    return P


def _new_agent(base_dir, mode):
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent, MetaMode)
    return MetaOrchestratorAgent(
        base_dir=base_dir, api_key=None, base_url=None, model_name=MODEL,
        meta_mode=MetaMode[mode])


def _ds(P, *keys):
    return [{"path": P[k][0], "metadata": P[k][1]} for k in keys]


# ----------------------------------------------------------------------
# Gate battery (cheap — one LLM call each)
# ----------------------------------------------------------------------

def test_gate(P, base_dir):
    agent = _new_agent(base_dir, "AUTONOMOUS")
    cases = [
        # name, datasets, predicate(verdict)->ok, expectation
        ("cross-modal complementary (image + XPS, same coupon)",
         _ds(P, "img", "spec"),
         lambda v: v["verdict"] == "complementary" and len(v["fanout_set"]) == 2,
         "complementary, run both"),
        ("cross-modal unrelated (image of Ti + Raman of Si)",
         _ds(P, "img", "raman_other"),
         lambda v: v["verdict"] in ("unrelated", "uncertain") and len(v["fanout_set"]) < 2,
         "decline"),
        ("redundant (two XPS Ti 2p of the same coupon)",
         _ds(P, "spec", "spec_dup"),
         lambda v: v["verdict"] in ("redundant", "uncertain") and len(v["fanout_set"]) < 2,
         "decline (redundant)"),
        ("3-way cross-modal (image + XPS same coupon + unrelated Si Raman)",
         _ds(P, "img", "spec", "raman_other"),
         lambda v: (len(v["fanout_set"]) == 2
                    and P["raman_other"][0] not in v["fanout_set"]
                    and P["img"][0] in v["fanout_set"]
                    and P["spec"][0] in v["fanout_set"]),
         "prune Si Raman, fan out image+XPS"),
    ]
    results = []
    for name, datasets, ok_fn, expect in cases:
        v = json.loads(agent._assess_complementarity(datasets))
        agent._complementarity_cache.clear()  # independent judgments
        try:
            ok = bool(ok_fn(v))
        except Exception:
            ok = False
        print(f"\n### GATE: {name}")
        print(f"  expect: {expect}")
        print(f"  verdict={v.get('verdict')} conf={v.get('confidence')} "
              f"join={v.get('join_axis')!r}")
        print(f"  fanout_set={[os.path.basename(p) for p in v.get('fanout_set', [])]} "
              f"redundant={[[os.path.basename(p) for p in c] for c in v.get('redundant_clusters', [])]} "
              f"unrelated={[os.path.basename(p) for p in v.get('unrelated', [])]}")
        print(f"  -> {'PASS' if ok else 'FAIL'}")
        results.append(ok)
    passed = all(results)
    print(f"\nGATE BATTERY: {sum(results)}/{len(results)} passed")
    return passed


# ----------------------------------------------------------------------
# Full cross-modal end-to-end (slow — real codegen on two agents + fusion)
# ----------------------------------------------------------------------

def test_full_crossmodal(P, base_dir):
    agent = _new_agent(base_dir, "AUTONOMOUS")
    img, imgm = P["img"]; spec, specm = P["spec"]
    print("\n### FULL CROSS-MODAL: delegate_to_analyses(image + XPS spectrum) ###")
    out = json.loads(agent._run_fanout([
        {"data_path": img, "metadata": imgm, "label": "BSE image",
         "task": f"Analyze the BSE-SEM image at {img}: segment the phases and "
                 "report the area fraction of each phase."},
        {"data_path": spec, "metadata": specm, "label": "XPS Ti 2p",
         "task": f"Fit the XPS Ti 2p spectrum at {spec}: identify the chemical "
                 "components (metallic Ti vs TiO2) and report their relative areas."},
    ]))
    print("FANOUT:", json.dumps(out, indent=2)[:1400])
    if out.get("status") != "success" or out.get("branches_with_output", 0) < 2:
        print("FULL CROSS-MODAL: branches did not both produce output")
        return False
    idxs = [r["delegation_index"] for r in out["results"] if r["produced_output"]]
    print("\n### FULL CROSS-MODAL: fuse_delegations ###")
    fout = json.loads(agent._fuse_delegations(
        idxs, focus="Do the two image phases correspond to the two XPS chemical "
                    "components (metal vs oxide), and are the fractions consistent?"))
    print("FUSION:", json.dumps(fout, indent=2)[:1800])
    ok = fout.get("status") == "success" and bool(fout.get("detailed_analysis"))
    print(f"\nFULL CROSS-MODAL: fused={ok}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also run the end-to-end cross-modal fan-out + fuse (slow)")
    args = ap.parse_args()
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        print("Set AWS_BEARER_TOKEN_BEDROCK (+ AWS_REGION_NAME).")
        sys.exit(2)

    d = tempfile.mkdtemp(prefix="fanout_live_")
    P = _make_datasets(d)
    results = {}
    with tempfile.TemporaryDirectory() as bd:
        results["gate_battery"] = test_gate(P, os.path.join(bd, "gate"))
    if args.full:
        with tempfile.TemporaryDirectory() as bd:
            results["full_crossmodal"] = test_full_crossmodal(
                P, os.path.join(bd, "xmodal"))

    print("\n" + "=" * 60)
    print("SUMMARY:", results)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
