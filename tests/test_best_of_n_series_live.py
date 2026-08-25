#!/usr/bin/env python3
"""Live check: per-regime anchor fan-out on a 2-regime image series.

Synthesizes a 4-image temperature series with a morphological transition
(images 0-1: space-filling polycrystalline grains; images 2-3: isolated
bright particles) and runs ImageAnalysisAgent.analyze(n_candidates=3).

Checks:
  - the series plan contains >= 2 regimes;
  - each regime ANCHOR fanned out (has image_NNNN/_candidates/cand_00..02 and
    an anchor_candidates table with exactly one selected entry);
  - non-anchor images did NOT fan out (no _candidates dir);
  - result["anchor_candidates"] carries one table per anchor.

Needs a vendor key (e.g. AWS_BEARER_TOKEN_BEDROCK + AWS_REGION_NAME) and
UNSAFE_EXECUTION_OK=true.

Usage: python tests/test_best_of_n_series_live.py [model_name]
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from scilink import auth

GRAINS = str(
    Path(__file__).resolve().parents[1]
    / "examples/polycrystalline_grains_demo/image.npy"
)


def _make_series(tmp: Path) -> list:
    rng = np.random.default_rng(0)
    grains = np.load(GRAINS).astype(np.float64)

    def _save(idx, arr):
        a = arr.astype(np.float64)
        a = (a - a.min()) / max(np.ptp(a), 1e-9) * 255
        p = tmp / f"frame_{idx}.npy"
        np.save(p, a.astype(np.uint8))
        return str(p)

    paths = []
    # Regime 1: polycrystalline grains (two frames of the same phase)
    paths.append(_save(0, grains + rng.normal(0, 4, grains.shape)))
    paths.append(_save(1, np.roll(grains, (37, -22), axis=(0, 1))
                       + rng.normal(0, 4, grains.shape)))

    # Regime 2: isolated bright particles on dark background
    yy, xx = np.mgrid[0:512, 0:512]
    for idx in (2, 3):
        img = rng.normal(20, 4, (512, 512))
        for _ in range(35):
            cy, cx = rng.uniform(20, 492, 2)
            r = rng.uniform(6, 16)
            img += 200 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2)
                                  / (2 * r ** 2)))
        paths.append(_save(idx, img))
    return paths


def main() -> int:
    model_name = sys.argv[1] if len(sys.argv) > 1 else "claude-opus-4-6"
    api_key = auth.get_api_key_for_model(model_name)
    if not api_key:
        print("ERROR: no API key in environment for", model_name)
        return 2

    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent

    tmp = Path(tempfile.mkdtemp(prefix="bestofn_series_"))
    paths = _make_series(tmp)
    out = Path(tempfile.mkdtemp(prefix="bestofn_series_out_"))

    agent = ImageAnalysisAgent(
        api_key=api_key, model_name=model_name, output_dir=str(out),
        enable_human_feedback=False, use_literature=False,
    )
    result = agent.analyze(
        data=paths,
        system_info=(
            "SEM image series of a thin film during in-situ annealing. The "
            "microstructure undergoes a morphological transition partway "
            "through the series: space-filling polycrystalline grains "
            "transform into isolated particles."
        ),
        objective=(
            "Characterize the microstructure in each morphological regime "
            "with an appropriate method: grain size statistics for the "
            "polycrystalline regime, particle size/count statistics for the "
            "particle regime."
        ),
        series_metadata={"variable": "temperature",
                         "values": [300, 400, 500, 600], "unit": "K"},
        n_candidates=3,
    )

    print("status:", result.get("status"))
    srj = json.loads((out / "series_analysis_results.json").read_text())
    regimes = (srj.get("series_analysis_plan") or {}).get("regimes") or []
    print(f"regimes in plan: {len(regimes)}")
    for r in regimes:
        print(f"  - {r.get('name')}: indices {r.get('image_indices')}")

    anchor_tables = result.get("anchor_candidates") or {}
    print(f"anchor tables in result: {len(anchor_tables)} "
          f"(anchor indices: {sorted(anchor_tables)})")

    checks = []
    checks.append(("plan has >=2 regimes", len(regimes) >= 2))
    checks.append(("one anchor table per regime",
                   len(anchor_tables) == max(len(regimes), 1)))

    fanned, clean = [], []
    for idx in range(len(paths)):
        cdir = out / f"image_{idx:04d}" / "_candidates"
        n_cands = len(list(cdir.glob("cand_*"))) if cdir.is_dir() else 0
        (fanned if n_cands else clean).append(idx)
        print(f"image_{idx:04d}: {n_cands} candidate dirs")

    anchor_idx = {int(k) for k in anchor_tables}
    checks.append(("fan-out only at anchors", set(fanned) == anchor_idx))
    checks.append(("non-anchors untouched",
                   all(i not in anchor_idx for i in clean)))

    for k, table in sorted(anchor_tables.items()):
        cands = table["candidates"]
        sel = [c for c in cands if c.get("selected")]
        print(f"anchor {k}: {len(cands)} candidates, "
              f"{len(sel)} selected, judge fallback="
              f"{table.get('judge', {}).get('fallback')}")
        checks.append((f"anchor {k} table well-formed",
                       len(cands) == 3 and len(sel) == 1))

    print()
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'CHECK'}] {name}")
    ok = all(c for _, c in checks) and result.get("status") == "success"
    print("\nRESULT:", "PASS — per-regime anchor fan-out works"
          if ok else "CHECK — see above")
    print("output dir:", out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
