#!/usr/bin/env python3
"""Live comparison: pure fixed-3 best-of-N vs the 1+2 escalation policy,
on REAL benchmark images (scilink-benchmarking/ImageAnalysisAgent).

Design: escalation's attempt 0 is statistically identical to a fixed-3 run's
candidate 0, so each FIXED-3 run doubles as its own escalation simulation.
Per run we record:

  - fast_accept_would_fire: candidate 0 passes the fast-accept predicate
    (approved, score >= threshold + margin, iterations <= 2);
  - judge_pick: which candidate the full 3-way judge selected;
  - missed_improvement: fast_accept_would_fire AND judge_pick != 0 — the
    quality cost of escalation (escalation would have locked candidate 0
    where the judge, seeing all three, preferred another);
  - score spreads/iterations, for threshold tuning.

Stratified sample: easy/medium/hard across TEM, OM, AFM with each image's
real metadata sidecar and benchmark objective.

Usage: python tests/test_escalation_vs_fixed3_live.py [model]
Needs vendor key env + UNSAFE_EXECUTION_OK=true.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

from scilink import auth

BENCH = Path("/Users/maxim.ziatdinov/Code/scilink-benchmarking/ImageAnalysisAgent")

# Must mirror UnifiedImageProcessingController's constants.
FAST_MARGIN = 0.1
FAST_MAX_ITERS = 2
QUALITY_THRESHOLD = 0.7

# (modality, stem, objective from the benchmark's objectives.txt)
CASES = [
    ("TEM", "(easy) AuNP",
     "Get statistics of the diameters and circularities of the AuNPs in the "
     "TEM image for synthesis yield quantification."),
    ("TEM", "(medium) triangular AuNP",
     "Get statistics of the edge length of the triangular prism AuNPs in "
     "the TEM image for synthesis yield quantification."),
    ("TEM", "(hard) dry ferrtin",
     "Calculate the radial distribution function to characterize the "
     "self-assembly structure of those low-contrast and near-focus ferrtin "
     "particles in the TEM image."),
    ("OM", "(medium) RAC", None),   # objective read from objectives.txt note
    ("OM", "(hard) NMC", None),
    ("AFM", "(easy) Au", None),
    ("AFM", "(medium) c98RhuA", None),
    ("AFM", "(hard) MoSBP3", None),
]


def _find_objective(modality: str, stem: str) -> str:
    """Pull the objective line for a stem from the modality's objectives.txt."""
    obj_file = BENCH / modality / "objectives.txt"
    name = stem.split(") ", 1)[1] if ") " in stem else stem
    for line in obj_file.read_text().splitlines():
        if name.lower() in line.lower():
            return line.split(":", 1)[1].strip() if ":" in line else line.strip()
    return f"Analyze the {name} image and extract its key quantitative features."


def _image_and_meta(modality: str, stem: str):
    d = BENCH / modality
    img = next((p for p in d.iterdir()
                if p.stem == stem and p.suffix.lower() in
                (".tif", ".tiff", ".png", ".jpg", ".npy")), None)
    meta_file = d / f"{stem}.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else None
    return img, meta


def main() -> int:
    model_name = sys.argv[1] if len(sys.argv) > 1 else "claude-opus-4-6"
    api_key = auth.get_api_key_for_model(model_name)
    if not api_key:
        print("ERROR: no API key for", model_name)
        return 2
    if not BENCH.is_dir():
        print("ERROR: benchmark dir not found:", BENCH)
        return 2

    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent

    rows = []
    for modality, stem, objective in CASES:
        img, meta = _image_and_meta(modality, stem)
        if img is None:
            print(f"\n=== {modality}/{stem}: image not found, skipping ===",
                  flush=True)
            continue
        objective = objective or _find_objective(modality, stem)
        print(f"\n=== {modality}/{stem} (fixed-3) ===", flush=True)
        print(f"  objective: {objective[:120]}", flush=True)
        out = tempfile.mkdtemp(prefix=f"cmp_{modality}_")
        agent = ImageAnalysisAgent(
            api_key=api_key, model_name=model_name, output_dir=out,
            enable_human_feedback=False, use_literature=False,
        )
        t0 = time.monotonic()
        try:
            res = agent.analyze(data=str(img), system_info=meta,
                                objective=objective, n_candidates=3)
        except Exception as e:  # noqa: BLE001
            print(f"  RAISED: {e}", flush=True)
            continue
        dt = time.monotonic() - t0
        table = res.get("anchor_candidates") or []
        if res.get("status") != "success" or len(table) != 3:
            print(f"  unusable run: status={res.get('status')}, "
                  f"table={len(table)}", flush=True)
            continue
        by_attempt = {c["attempt"]: c for c in table}
        c0 = by_attempt[0]
        judge_pick = next(c["attempt"] for c in table if c.get("selected"))
        fast_fire = (
            c0["success"] and c0["approved"]
            and c0["score"] >= QUALITY_THRESHOLD + FAST_MARGIN
            and c0["iterations"] <= FAST_MAX_ITERS
        )
        missed = fast_fire and judge_pick != 0
        scores = [round(by_attempt[i]["score"], 2) for i in range(3)]
        iters = [by_attempt[i]["iterations"] for i in range(3)]
        rows.append({
            "case": f"{modality}/{stem}", "scores": scores, "iters": iters,
            "judge_pick": judge_pick, "fast_fire": fast_fire,
            "missed_improvement": missed, "wall_s": round(dt), "out": out,
            "judge_fallback": (res.get("anchor_judge") or {}).get("fallback"),
        })
        print(f"  scores={scores} iters={iters} judge_pick={judge_pick} "
              f"fast_accept_would_fire={fast_fire} "
              f"MISSED_IMPROVEMENT={missed} wall={dt:.0f}s", flush=True)
        print(f"  out: {out}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print(f"Usable runs: {len(rows)}", flush=True)
    n_fast = sum(r["fast_fire"] for r in rows)
    n_missed = sum(r["missed_improvement"] for r in rows)
    for r in rows:
        print(f"  {r['case']}: scores={r['scores']} pick={r['judge_pick']} "
              f"fast={r['fast_fire']} missed={r['missed_improvement']} "
              f"({r['wall_s']}s)", flush=True)
    print(f"\nOverall: fast-accept would fire in {n_fast}/{len(rows)} runs "
          f"(escalation saves ~2 attempts each); the 3-way judge preferred "
          f"a different candidate in {n_missed} of those (quality cost).",
          flush=True)
    for r in rows:
        if r["missed_improvement"]:
            print(f"  MISSED: {r['case']}: scores={r['scores']}, judge "
                  f"picked {r['judge_pick']} over fast-accepted 0 -> "
                  f"inspect {r['out']}", flush=True)
    print("\nRESULT: DATA — tunes the fast-accept thresholds "
          "(no hard pass/fail)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
