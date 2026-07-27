"""In-situ XRD dehydration campaign replay (#346 step 2 scenario test).

Imitates a real multi-session campaign on the 38-frame dehydration series
(known phase transition ~43-58 C: dominant peak 3.7 -> 4.0 deg, new 11.9 deg
reflection):

  leg1_anchor  — frame 0001 fresh: bank must offer the XRD record b88d813a
                 (dry-run 0.823) and adapt it into this run's locked recipe.
  leg2_series  — frames 0002+0003 as a 2-frame series with
                 prior_analysis_paths=leg1 + reuse_locked_script: precedence
                 (0 exemplar offers), reuse verdicts, once-per-run banking.
  leg3_drift   — frame 0013 (58.2 C, POST-transition) under the same reuse:
                 the locked pre-transition recipe meets a new phase — record
                 the reuse verdict (drift detector behavior, either outcome
                 is informative for a global profile fit).
  leg4_newphase — frame 0013 fresh (no priors): exemplar at ~0.653 must be
                 ADAPTED to the new phase (fit must include the 11.9 deg
                 reflection).

Usage: python run_campaign_live.py [names...]   (needs AWS_BEARER_TOKEN_BEDROCK)
"""

import json
import os
import sys
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
META = {"technique": "powder XRD",
        "description": ("In-situ synchrotron XRD during a dehydration ramp; "
                        "x is 2-theta in degrees.")}


def _frame(n):
    hits = sorted(XRD.glob(f"IBD_Insitu_Dehydration_01_{n:04d}_*.txt"))
    assert hits, f"frame {n} not found"
    return str(hits[0])


def _dump(name, res):
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if k != "visualization_bytes"}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return "<bytes>" if isinstance(o, bytes) else o
    (OUT / f"result_{name}.json").write_text(json.dumps(clean(res), indent=2, default=str))
    rv = res.get("reuse_validity") or {}
    print(f"\n[{name}] status={res.get('status')} "
          f"r2={(res.get('fit_quality') or {}).get('r_squared')} "
          f"reuse_verdict={rv.get('verdict')} banked={res.get('banked_scripts')}",
          flush=True)


def _agent(out_name):
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    return CurveFittingAgent(model_name=MODEL, output_dir=str(OUT / out_name),
                             enable_human_feedback=False, use_literature=False)


def leg1_anchor():
    res = _agent("camp_leg1").analyze(_frame(1), system_info=META)
    _dump("camp_leg1", res)
    return res


def leg2_series():
    res = _agent("camp_leg2").analyze(
        [_frame(2), _frame(3)], system_info=META,
        series_metadata={"variable": "temperature", "values": [33.4, 35.9],
                         "unit": "C"},
        prior_analysis_paths=[str(OUT / "camp_leg1")],
        reuse_locked_script=True)
    _dump("camp_leg2", res)
    for r in (res.get("series_results") or []):
        rv = r.get("reuse_validity") or {}
        print(f"   frame {r.get('name')}: verdict={rv.get('verdict')} "
              f"r2={(r.get('fit_quality') or {}).get('r_squared')}", flush=True)
    return res


def leg3_drift():
    res = _agent("camp_leg3").analyze(
        _frame(13), system_info=META,
        prior_analysis_paths=[str(OUT / "camp_leg1")],
        reuse_locked_script=True)
    _dump("camp_leg3", res)
    return res


def leg4_newphase():
    res = _agent("camp_leg4").analyze(_frame(13), system_info=META)
    _dump("camp_leg4", res)
    return res


ALL = {"leg1_anchor": leg1_anchor, "leg2_series": leg2_series,
       "leg3_drift": leg3_drift, "leg4_newphase": leg4_newphase}

if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    results = {}
    for name in names:
        t0 = time.time()
        print(f"\n{'='*70}\nLIVE {name.upper()} START\n{'='*70}", flush=True)
        try:
            results[name] = ALL[name]().get("status")
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results[name] = f"EXCEPTION: {e}"
        print(f"LIVE {name.upper()} DONE in {time.time()-t0:.0f}s "
              f"-> {results[name]}", flush=True)
    print("\nSUMMARY:", json.dumps(results, indent=2))
