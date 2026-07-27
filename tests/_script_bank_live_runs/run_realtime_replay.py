"""Realtime-profile live replay (#346 step 3) — the plan's validation case.

Replays the 38-frame in-situ XRD dehydration series frame-by-frame:
anchor = camp_leg1 (frame 0001, analyzed thoroughly in the campaign run);
frames 0002-0038 run under profile="realtime" against it. Reports per-frame
wall-clock (target: seconds vs ~5 min thorough), gate verdict, and the
fingerprint drift channel — the known 43-58 C transition frames SHOULD flag
drift="suspected" while the R2 gate stays good (the two-channel design).

Zero LLM calls per frame: no Bedrock key is needed for the frames at all —
the model is never called on the happy path.

Usage: python run_realtime_replay.py
"""

import json
import os
import time
from pathlib import Path

OUT = Path(__file__).parent
os.environ.setdefault("AWS_REGION_NAME", "us-east-1")
os.environ["UNSAFE_EXECUTION_OK"] = "true"
os.environ["SCILINK_HOME"] = str(OUT / "scilink_home")
os.environ["SCILINK_MEMORY"] = "0"
os.environ["SCILINK_SCRIPT_BANK"] = "1"

MODEL = "bedrock/us.anthropic.claude-opus-4-8"
XRD = Path.home() / "Code" / "benchmarking_for_paper2" / "xrd_exp"
ANCHOR = OUT / "camp_leg1"
META = {"technique": "powder XRD",
        "description": ("In-situ synchrotron XRD during a dehydration ramp; "
                        "x is 2-theta in degrees.")}


def main():
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent

    frames = sorted(XRD.glob("IBD_Insitu_Dehydration_01_*.txt"))[1:]  # skip anchor
    rows = []
    for f in frames:
        tag = f.stem.split("_01_")[1]
        t0 = time.time()
        agent = CurveFittingAgent(
            model_name=MODEL, output_dir=str(OUT / "rt_frames" / tag),
            enable_human_feedback=False, use_literature=False,
        )
        try:
            res = agent.analyze(
                str(f), system_info=META,
                profile="realtime",
                prior_analysis_paths=[str(ANCHOR)],
                reuse_locked_script=True,
            )
            dt = time.time() - t0
            rv = res.get("reuse_validity") or {}
            row = {
                "frame": tag,
                "seconds": round(dt, 1),
                "status": res.get("status"),
                "r2": (res.get("fit_quality") or {}).get("r_squared"),
                "verdict": rv.get("verdict"),
                "similarity": rv.get("fingerprint_similarity"),
                "drift": rv.get("drift"),
                "profile": res.get("profile"),
            }
        except Exception as e:  # noqa: BLE001
            row = {"frame": tag, "seconds": round(time.time() - t0, 1),
                   "status": f"EXCEPTION: {e}"}
        rows.append(row)
        print(json.dumps(row), flush=True)

    (OUT / "result_realtime_replay.json").write_text(
        json.dumps(rows, indent=1))
    ok = [r for r in rows if r.get("status") == "success"]
    drifted = [r["frame"] for r in ok if r.get("drift") == "suspected"]
    print(f"\nREPLAY SUMMARY: {len(ok)}/{len(rows)} frames ok, "
          f"median {sorted(r['seconds'] for r in ok)[len(ok)//2]}s/frame, "
          f"drift flagged on {len(drifted)} frames "
          f"(first: {drifted[0] if drifted else 'none'})", flush=True)


if __name__ == "__main__":
    main()
