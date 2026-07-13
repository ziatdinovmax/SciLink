"""Realtime replay on a SECOND physical XRD series — FAIR_SL016 IMAuCl4.

In-situ lab XRD of imidazolium-AuCl4 precursors on a 31->120 C ramp
(1 K/min, 53 frames, 10-45 deg 2theta, 0.01 step, cps). Two questions the
ibuprofen-dehydration runs could not answer:

1. Cross-compound cold start (phase A): the live-run bank holds ibuprofen
   XRD veterans. Auditioning them verbatim against AuCl4 frame 1 tests the
   negative case on real data — either the audition gate rejects them and
   demotes loudly to a thorough anchor, or the auto-detecting global fitter
   passes on foreign data (leg-3 lesson: R2 gate is blind to identity) and
   the drift channel must carry the discrimination. Either outcome's run
   is the anchor for phase B.
2. Different pattern evolution (phase B): frames 2-53 replay under
   profile="realtime"; the precursor transformation should trip the
   fingerprint-drift channel somewhere inside the ramp.

Usage: python run_imau_realtime.py   (needs AWS_BEARER_TOKEN_BEDROCK)
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
XRD = Path.home() / "Code" / "FAIR_SL016_IMAuCl4"
META = {"technique": "powder XRD",
        "description": ("In-situ powder XRD during a thermal ramp "
                        "(31->120 C, 1 K/min) of imidazolium tetrachloro"
                        "aurate precursors; x is 2-theta in degrees, "
                        "intensity in cps.")}


def _frame(n):
    return str(XRD / f"FAIR_SL016_IMAuCl4_precursors_31C_120C_1Kmin_{n}.xy")


def _temp(n):
    return round(31 + (n - 1) * 89 / 52, 1)


def _agent(out_name):
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    return CurveFittingAgent(model_name=MODEL, output_dir=str(OUT / out_name),
                             enable_human_feedback=False, use_literature=False)


def main():
    rows = []

    # Phase A — cross-compound cold start on frame 1 (the only step that
    # may spend LLM calls: audition rejects -> thorough anchor).
    t0 = time.time()
    res = _agent("imau_anchor").analyze(
        _frame(1), system_info=META, profile="realtime")
    rv = res.get("reuse_validity") or {}
    anchor_row = {
        "frame": 1, "temp_C": _temp(1), "seconds": round(time.time() - t0, 1),
        "status": res.get("status"), "profile": res.get("profile"),
        "r2": (res.get("fit_quality") or {}).get("r_squared"),
        "cold_start": res.get("cold_start"),
        "source": rv.get("source"), "verdict": rv.get("verdict"),
    }
    rows.append(anchor_row)
    print(json.dumps(anchor_row), flush=True)

    # Phase B — realtime replay of frames 2..53 against the phase-A anchor.
    for n in range(2, 54):
        t0 = time.time()
        try:
            res = _agent(f"imau_rt/{n:04d}").analyze(
                _frame(n), system_info=META, profile="realtime",
                prior_analysis_paths=[str(OUT / "imau_anchor")],
                reuse_locked_script=True)
            rv = res.get("reuse_validity") or {}
            row = {
                "frame": n, "temp_C": _temp(n),
                "seconds": round(time.time() - t0, 1),
                "status": res.get("status"),
                "r2": (res.get("fit_quality") or {}).get("r_squared"),
                "verdict": rv.get("verdict"),
                "similarity": rv.get("fingerprint_similarity"),
                "drift": rv.get("drift"),
                "profile": res.get("profile"),
            }
        except Exception as e:  # noqa: BLE001
            row = {"frame": n, "temp_C": _temp(n),
                   "seconds": round(time.time() - t0, 1),
                   "status": f"EXCEPTION: {e}"}
        rows.append(row)
        print(json.dumps(row), flush=True)

    (OUT / "result_imau_realtime.json").write_text(json.dumps(rows, indent=1))
    ok = [r for r in rows[1:] if r.get("status") == "success"]
    drifted = [r["frame"] for r in ok if r.get("drift") == "suspected"]
    secs = sorted(r["seconds"] for r in ok)
    print(f"\nIMAU REPLAY SUMMARY: anchor={anchor_row['status']} "
          f"(cold_start={anchor_row['cold_start']}, "
          f"profile={anchor_row['profile']}); "
          f"{len(ok)}/{len(rows) - 1} frames ok, "
          f"median {secs[len(secs) // 2] if secs else '?'}s/frame, "
          f"drift on {len(drifted)} frames "
          f"(first: {drifted[0] if drifted else 'none'})", flush=True)


if __name__ == "__main__":
    main()
