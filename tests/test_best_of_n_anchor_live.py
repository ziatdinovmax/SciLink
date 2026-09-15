#!/usr/bin/env python3
"""Live test for parallel best-of-N anchor analysis.

Runs ImageAnalysisAgent.analyze on a real image and checks:

  A. n_candidates=3 -> three attempt dirs under image_0000/_candidates/,
     winner artifacts promoted to image_0000/, anchor_candidates table
     (with exactly one selected=True) in the result and in
     series_analysis_results.json, judge/milestone log lines, and
     wall clock well under 3x a serial run.
  B. n_candidates=1 (default) -> no _candidates/ dir, no anchor_candidates
     field; baseline behaviour unchanged.

Ad-hoc live test — needs a vendor API key (e.g. ANTHROPIC_API_KEY) and,
because image analysis executes generated code, UNSAFE_EXECUTION_OK=true.

Usage:
    UNSAFE_EXECUTION_OK=true python tests/test_best_of_n_anchor_live.py \
        [model_name] [image_path]
"""
import io
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

from scilink import auth

DEFAULT_IMAGE = str(
    Path(__file__).resolve().parents[1]
    / "examples/polycrystalline_grains_demo/image.npy"
)
SYS_INFO = "A microscopy image of a polycrystalline material."
OBJECTIVE = "Segment the grains and report grain size statistics."


def _run(model_name, api_key, image, *, n_candidates):
    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent

    out = tempfile.mkdtemp(prefix=f"bestofn_{n_candidates}_")
    agent = ImageAnalysisAgent(
        api_key=api_key, model_name=model_name, output_dir=out,
        enable_human_feedback=False, use_literature=False,
    )

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    root.addHandler(handler)

    raised = None
    result = None
    t0 = time.monotonic()
    try:
        result = agent.analyze(
            data=image, system_info=SYS_INFO, objective=OBJECTIVE,
            n_candidates=n_candidates,
        )
    except Exception as e:  # noqa: BLE001
        raised = e
    finally:
        root.removeHandler(handler)
    return result, buf.getvalue(), raised, Path(out), time.monotonic() - t0


def main() -> int:
    model_name = sys.argv[1] if len(sys.argv) > 1 else "claude-opus-4-6"
    image = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_IMAGE
    api_key = auth.get_api_key_for_model(model_name)
    if not api_key:
        print("ERROR: no API key in environment for", model_name)
        return 2

    checks = []

    # --- A: best-of-3 -------------------------------------------------
    print("=== A: n_candidates=3 ===")
    res, log, raised, out, dt = _run(model_name, api_key, image, n_candidates=3)
    if raised is not None:
        print(f"  RAISED: {type(raised).__name__}: {raised}")
        checks.append(False)
    else:
        cand_dirs = sorted(
            p.name for p in (out / "image_0000" / "_candidates").glob("cand_*")
        ) if (out / "image_0000" / "_candidates").is_dir() else []
        promoted_viz = (out / "image_0000" / "visualization.png").exists()
        table = (res or {}).get("anchor_candidates") or []
        selected = [c for c in table if c.get("selected")]
        judge = (res or {}).get("anchor_judge") or {}
        milestones = log.count("finished (")
        srj = out / "series_analysis_results.json"
        table_persisted = False
        if srj.exists():
            persisted = json.loads(srj.read_text())
            table_persisted = any(
                r.get("anchor_candidates")
                for r in persisted.get("results", [])
                if isinstance(r, dict)
            )
        print(f"  status             : {res.get('status')!r}")
        print(f"  wall clock         : {dt:.0f}s")
        print(f"  candidate dirs     : {cand_dirs}")
        print(f"  promoted winner viz: {promoted_viz}")
        print(f"  candidate table    : {len(table)} entries, "
              f"{len(selected)} selected")
        for c in table:
            print(f"    - attempt {c['attempt']}: score={c['score']}, "
                  f"approved={c['approved']}, selected={c['selected']}")
        print(f"  judge reasoning    : {judge.get('reasoning', '')[:160]!r}")
        print(f"  judge fallback     : {judge.get('fallback')}")
        print(f"  milestone log lines: {milestones}")
        print(f"  table persisted    : {table_persisted}")
        ok_a = (
            res.get("status") == "success"
            and len(cand_dirs) == 3
            and promoted_viz
            and len(table) == 3
            and len(selected) == 1
            and milestones == 3
            and table_persisted
        )
        checks.append(ok_a)
        print(f"  -> {'PASS' if ok_a else 'CHECK'}")
        print(f"  output dir: {out}")

    # --- B: n=1 baseline ----------------------------------------------
    print("\n=== B: n_candidates=1 (baseline parity) ===")
    res, log, raised, out, dt = _run(model_name, api_key, image, n_candidates=1)
    if raised is not None:
        print(f"  RAISED: {type(raised).__name__}: {raised}")
        checks.append(False)
    else:
        no_cand_dir = not (out / "image_0000" / "_candidates").exists()
        no_table = "anchor_candidates" not in (res or {})
        no_bestofn_log = "Best-of-" not in log
        print(f"  status             : {res.get('status')!r}")
        print(f"  wall clock         : {dt:.0f}s")
        print(f"  no _candidates dir : {no_cand_dir}")
        print(f"  no candidate table : {no_table}")
        print(f"  no best-of-N logs  : {no_bestofn_log}")
        ok_b = (
            res.get("status") == "success"
            and no_cand_dir and no_table and no_bestofn_log
        )
        checks.append(ok_b)
        print(f"  -> {'PASS' if ok_b else 'CHECK'}")
        print(f"  output dir: {out}")

    ok = all(checks) and len(checks) == 2
    print()
    print("RESULT:", "PASS — best-of-N fan-out + judge work; baseline "
          "unaffected" if ok else "CHECK — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
