"""Round-2 scenario matrix (#346, full stack) — bedrock opus-4-8.

Imitates the messy parts of a real in-situ campaign:

  1. glitch      — detector-glitch frames under realtime vs the camp_leg1
                   anchor: all-zeros frame, NaN-laced frame, 10-point runt
                   frame. Honest expectation: keep schema, verdict poor
                   and/or drift suspected — never a silent 'good'.
  2. reanchor    — the canonical drift workflow: frame 0010 flagged drift →
                   THOROUGH re-analysis (agent-judged, old anchor as
                   reference) → frames 0011-0012 realtime against the NEW
                   anchor → drift clears (same post-transition phase).
  3. multiframe  — one analyze() call with THREE frames + profile=realtime +
                   NO prior: cold-start audition locks the recipe on the
                   anchor, non-anchor frames run the locked script — all
                   zero-LLM.
  4. orch        — the orchestrator LLM must choose profile='realtime' +
                   prior + reuse from the task wording alone (tests the
                   LLM-facing tool description).

Usage: python run_round2_live.py [names...]   (needs AWS_BEARER_TOKEN_BEDROCK)
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent
os.environ.setdefault("AWS_REGION_NAME", "us-east-1")
os.environ["UNSAFE_EXECUTION_OK"] = "true"
os.environ["SCILINK_HOME"] = str(OUT / "scilink_home")
os.environ["SCILINK_MEMORY"] = "0"
os.environ["SCILINK_SCRIPT_BANK"] = "1"

MODEL = "bedrock/us.anthropic.claude-opus-4-8"
XRD = Path.home() / "Code" / "benchmarking_for_paper2" / "xrd_exp"
META = {"technique": "powder XRD",
        "description": ("In-situ synchrotron XRD during a dehydration ramp; "
                        "x is 2-theta in degrees.")}
ANCHOR = OUT / "camp_leg1"


def _frame(n):
    return str(sorted(XRD.glob(f"IBD_Insitu_Dehydration_01_{n:04d}_*.txt"))[0])


def _agent(out_name):
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    return CurveFittingAgent(model_name=MODEL, output_dir=str(OUT / out_name),
                             enable_human_feedback=False, use_literature=False)


def _report(name, res, t0):
    rv = res.get("reuse_validity") or {}
    print(f"\n[{name}] {time.time()-t0:.1f}s status={res.get('status')} "
          f"profile={res.get('profile')} verdict={rv.get('verdict')} "
          f"r2={(res.get('fit_quality') or {}).get('r_squared')} "
          f"drift={rv.get('drift')} sim={rv.get('fingerprint_similarity')} "
          f"warn={str(res.get('quality_warning'))[:80]}", flush=True)
    return res


def glitch():
    real = np.loadtxt(_frame(2), comments="#")
    cases = {}
    z = real.copy(); z[:, 1] = 0.0
    cases["zeros"] = z
    nn = real.copy(); nn[::3, 1] = np.nan
    cases["nans"] = nn
    cases["runt"] = real[:10]
    last = None
    for tag, arr in cases.items():
        f = OUT / f"glitch_{tag}.csv"
        np.savetxt(f, arr, delimiter=",", header="two_theta,intensity", comments="")
        t0 = time.time()
        try:
            last = _report(f"glitch_{tag}", _agent(f"glitch_{tag}").analyze(
                str(f), system_info=META, profile="realtime",
                prior_analysis_paths=[str(ANCHOR)], reuse_locked_script=True), t0)
        except Exception as e:  # noqa: BLE001
            print(f"\n[glitch_{tag}] EXCEPTION after {time.time()-t0:.1f}s: {e}",
                  flush=True)
    return last or {"status": "error"}


def reanchor():
    # Leg A — thorough re-analysis of the drifted frame (old anchor as
    # agent-judged reference; bank retrieval must stay silent — precedence).
    t0 = time.time()
    res_a = _report("reanchor_thorough", _agent("reanchor_thorough").analyze(
        _frame(10), system_info=META,
        prior_analysis_paths=[str(ANCHOR)]), t0)
    if res_a.get("status") != "success":
        return res_a
    # Leg B — resume realtime against the NEW anchor; drift should clear.
    out = res_a
    for n in (11, 12):
        t0 = time.time()
        out = _report(f"reanchor_rt_{n:04d}", _agent(f"reanchor_rt_{n:04d}").analyze(
            _frame(n), system_info=META, profile="realtime",
            prior_analysis_paths=[str(OUT / "reanchor_thorough")],
            reuse_locked_script=True), t0)
    return out


def multiframe():
    t0 = time.time()
    res = _agent("mf_coldstart").analyze(
        [_frame(21), _frame(22), _frame(23)], system_info=META,
        series_metadata={"variable": "temperature",
                         "values": [78.1, 80.5, 83.0], "unit": "C"},
        profile="realtime")
    rv = res.get("reuse_validity") or {}
    ok = [r for r in (res.get("series_results") or []) if r.get("success")]
    print(f"\n[multiframe] {time.time()-t0:.1f}s status={res.get('status')} "
          f"frames_ok={len(ok)}/3 cold_start={res.get('cold_start')} "
          f"anchor_drift={rv.get('drift')}", flush=True)
    return res


def orch():
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisMode, AnalysisOrchestratorAgent,
    )
    orch_agent = AnalysisOrchestratorAgent(
        base_dir=str(OUT / "orch_rt_session"), model_name=MODEL,
        analysis_mode=AnalysisMode.AUTONOMOUS,
    )
    out = orch_agent.run_task(
        f"Frame 6 of an in-situ XRD dehydration series just arrived at "
        f"{_frame(6)} (powder XRD, x = 2-theta in degrees). The series anchor "
        f"was already analyzed thoroughly at {ANCHOR}. This is a high-cadence "
        f"measurement stream: analyze this frame in REALTIME mode, reusing "
        f"the anchor's locked script — do not re-derive the model. Report "
        f"the gate verdict and whether drift is suspected."
    )
    print(f"\n[orch] status={out.get('status')}")
    print("summary:", str(out.get("summary"))[:400], flush=True)
    return out


ALL = {"glitch": glitch, "reanchor": reanchor, "multiframe": multiframe,
       "orch": orch}

if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    results = {}
    for name in names:
        print(f"\n{'='*70}\nLIVE {name.upper()} START\n{'='*70}", flush=True)
        try:
            results[name] = ALL[name]().get("status")
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results[name] = f"EXCEPTION: {e}"
    print("\nSUMMARY:", json.dumps(results, indent=2))
