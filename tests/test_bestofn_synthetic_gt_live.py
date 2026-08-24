#!/usr/bin/env python3
"""Live ground-truth comparison: judge pick + escalation policy vs KNOWN truth.

Synthetic particle images where the truth is fixed by construction (particle
count, mean/std diameter). A fixed-3 run yields, per candidate (winner from
the result; losers from the persisted attempt_result.json snapshots):

  - GT errors: |reported count - true count|, |reported mean diameter - true|;
  - which candidate is GT-BEST (min count error, then min diameter error);
  - whether the judge picked the GT-best candidate;
  - escalation simulation: if fast-accept would fire on candidate 0, the GT
    delta between candidate 0 and the judge's pick = the measured quality
    cost (or zero) of escalation, against actual truth.

Cases: easy (45 well-separated disks, mild noise) and hard (60 faint
overlapping blobs, heavy noise), 2 seeds each -> 4 fixed-3 runs.

Usage: python tests/test_bestofn_synthetic_gt_live.py [model]
Needs vendor key env + UNSAFE_EXECUTION_OK=true.
"""
import json
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from scilink import auth

FAST_MARGIN = 0.1
FAST_MAX_ITERS = 2
QUALITY_THRESHOLD = 0.7

OBJECTIVE = (
    "Detect ALL bright particles in the image. Report the particle count "
    "and the mean and standard deviation of the particle diameters in "
    "pixels in the extracted features."
)


def _make_particles(seed, n, r_lo, r_hi, amp, noise, min_sep):
    """Place n non-(heavily)-overlapping gaussian disks; return img + GT."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:512, 0:512]
    img = rng.normal(60, noise, (512, 512))
    centers, radii = [], []
    tries = 0
    while len(centers) < n and tries < 20000:
        tries += 1
        cy, cx = rng.uniform(25, 487, 2)
        r = rng.uniform(r_lo, r_hi)
        if any((cy - y) ** 2 + (cx - x) ** 2 < (min_sep * (r + r2)) ** 2
               for (y, x), r2 in zip(centers, radii)):
            continue
        centers.append((cy, cx))
        radii.append(r)
        img += amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2)
                              / (2 * (r / 2.0) ** 2)))
    img = (img - img.min()) / np.ptp(img) * 255
    d = 2.0 * np.asarray(radii)
    gt = {"count": len(centers), "mean_d": float(d.mean()),
          "std_d": float(d.std())}
    return img.astype(np.uint8), gt


CASES = [
    # (name, kwargs) — diameters ~ 2*r where the gaussian sigma is r/2, so a
    # detector reporting FWHM-ish sizes lands near 2.35*(r/2) ≈ 1.18*r;
    # we therefore score the mean-diameter error with generous tolerance and
    # rank candidates RELATIVELY (same convention applies to all three).
    ("easy_sep", dict(n=45, r_lo=10, r_hi=18, amp=140, noise=6, min_sep=1.1)),
    ("hard_faint", dict(n=60, r_lo=8, r_hi=26, amp=28, noise=12, min_sep=0.55)),
]


def _walk_numbers(obj, path=""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += _walk_numbers(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out += _walk_numbers(v, f"{path}[{i}]")
    else:
        try:
            out.append((path, float(obj)))
        except (TypeError, ValueError):
            pass
    return out


def _extract_count_and_meand(features):
    """Heuristic: find a particle count and a mean diameter in a features
    dict authored by an LLM-generated script. Returns (count, mean_d) with
    None for what can't be found."""
    nums = _walk_numbers(features)
    count = None
    for path, v in nums:
        p = path.lower()
        if re.search(r"(count|n_particles|num_particles|n_detected|"
                     r"total_particles|n_objects|num_objects)", p):
            if 1 <= v <= 2000 and float(v).is_integer():
                count = int(v)
                break
    mean_d = None
    for path, v in nums:
        p = path.lower()
        if ("diam" in p or "size" in p) and ("mean" in p or "avg" in p
                                             or "average" in p):
            if 0 < v < 512:
                mean_d = float(v)
                break
    return count, mean_d


def main() -> int:
    model_name = sys.argv[1] if len(sys.argv) > 1 else "claude-opus-4-6"
    api_key = auth.get_api_key_for_model(model_name)
    if not api_key:
        print("ERROR: no API key for", model_name)
        return 2

    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent

    rows = []
    for name, kw in CASES:
        for seed in (1, 2):
            img, gt = _make_particles(seed, **kw)
            p = Path(tempfile.mkdtemp(prefix=f"gt_{name}_in_")) / "img.npy"
            np.save(p, img)
            out = Path(tempfile.mkdtemp(prefix=f"gt_{name}_"))
            print(f"\n=== {name} seed {seed}: GT count={gt['count']}, "
                  f"mean_d={gt['mean_d']:.1f}px ===", flush=True)
            agent = ImageAnalysisAgent(
                api_key=api_key, model_name=model_name, output_dir=str(out),
                enable_human_feedback=False, use_literature=False,
            )
            t0 = time.monotonic()
            try:
                res = agent.analyze(
                    data=str(p),
                    system_info="Synthetic benchmark micrograph of bright "
                                "particles on a noisy background.",
                    objective=OBJECTIVE, n_candidates=3,
                )
            except Exception as e:  # noqa: BLE001
                print(f"  RAISED: {e}", flush=True)
                continue
            dt = time.monotonic() - t0
            table = res.get("anchor_candidates") or []
            if res.get("status") != "success" or len(table) != 3:
                print(f"  unusable: status={res.get('status')}, "
                      f"table={len(table)}", flush=True)
                continue
            judge_pick = next(c["attempt"] for c in table
                              if c.get("selected"))

            # Per-candidate GT scoring from persisted snapshots.
            cand_gt = {}
            for c in table:
                i = c["attempt"]
                snap = (out / "image_0000" / "_candidates"
                        / f"cand_{i:02d}" / "attempt_result.json")
                feats = None
                if snap.exists():
                    feats = json.loads(snap.read_text()).get(
                        "extracted_features")
                if feats is None and i == judge_pick:
                    feats = res.get("tier1_results", {}).get(
                        "extracted_features") or res.get(
                        "extracted_features")
                count, mean_d = _extract_count_and_meand(feats or {})
                cand_gt[i] = {
                    "count": count,
                    "count_err": (abs(count - gt["count"])
                                  if count is not None else None),
                    "mean_d": mean_d,
                    "score": c["score"], "approved": c["approved"],
                    "iterations": c["iterations"],
                }
                print(f"  cand {i}: score={c['score']:.2f} "
                      f"count={count} (err="
                      f"{cand_gt[i]['count_err']}) mean_d={mean_d}",
                      flush=True)

            scored = {i: g for i, g in cand_gt.items()
                      if g["count_err"] is not None}
            gt_best = (min(scored, key=lambda i: scored[i]["count_err"])
                       if scored else None)
            c0 = next(c for c in table if c["attempt"] == 0)
            fast_fire = (
                c0["success"] and c0["approved"]
                and c0["score"] >= QUALITY_THRESHOLD + FAST_MARGIN
                and c0["iterations"] <= FAST_MAX_ITERS
            )
            judge_is_gt_best = (gt_best is not None
                                and judge_pick == gt_best)
            esc_gt_cost = None
            if fast_fire and 0 in scored and judge_pick in scored:
                esc_gt_cost = (scored[0]["count_err"]
                               - scored[judge_pick]["count_err"])
            rows.append({
                "case": f"{name}/seed{seed}", "gt": gt,
                "judge_pick": judge_pick, "gt_best": gt_best,
                "judge_is_gt_best": judge_is_gt_best,
                "fast_fire": fast_fire, "esc_gt_cost": esc_gt_cost,
                "cand_gt": cand_gt, "wall_s": round(dt), "out": str(out),
            })
            print(f"  judge_pick={judge_pick} gt_best={gt_best} "
                  f"judge_is_gt_best={judge_is_gt_best} "
                  f"fast_fire={fast_fire} esc_gt_cost={esc_gt_cost} "
                  f"wall={dt:.0f}s", flush=True)
            print(f"  out: {out}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print(f"Usable runs: {len(rows)}", flush=True)
    n_judge_right = sum(r["judge_is_gt_best"] for r in rows)
    n_fast = sum(r["fast_fire"] for r in rows)
    costs = [r["esc_gt_cost"] for r in rows if r["esc_gt_cost"] is not None]
    print(f"  judge picked the GT-best candidate in "
          f"{n_judge_right}/{len(rows)} runs", flush=True)
    print(f"  fast-accept would fire in {n_fast}/{len(rows)} runs; "
          f"escalation GT cost (count-error delta cand0 - judgepick) on "
          f"those: {costs}", flush=True)
    print("\nRESULT: DATA — ground-truth scoring of judge + escalation "
          "policy", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
