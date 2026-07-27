"""Verbatim cold-start live matrix (#346 step 4) — bedrock opus-4-8.

Bank state going in: XRD records b88d813a (frame-1 script), c4fd4e8d
(campaign veteran, n_successes=3), af8c9476 (post-transition frame-13
script), plus Raman/others.

  1. cs_pre    — frame 0002 (pre-transition), realtime, NO prior: audition
                 should lock an XRD record in seconds, zero LLM calls,
                 source=script_bank:<id>, drift=none.
  2. cs_chain  — frames 0003-0005 realtime with prior=[cs_pre dir]: the
                 cold-start run is itself a valid anchor.
  3. cs_post   — frame 0020 (post-transition), realtime, NO prior: should
                 lock the record whose ORIGIN matches this phase (frame-13
                 script) with drift=none relative to it — origin-aware drift.
  4. cs_miss   — synthetic exponential-decay curve (matches nothing at the
                 0.55 verbatim floor): loud demotion to a thorough anchor
                 (LLM pipeline runs and completes).

Usage: python run_coldstart_live.py [names...]   (needs AWS_BEARER_TOKEN_BEDROCK)
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


def _frame(n):
    return str(sorted(XRD.glob(f"IBD_Insitu_Dehydration_01_{n:04d}_*.txt"))[0])


def _agent(out_name):
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    return CurveFittingAgent(model_name=MODEL, output_dir=str(OUT / out_name),
                             enable_human_feedback=False, use_literature=False)


def _report(name, res, t0):
    rv = res.get("reuse_validity") or {}
    print(f"\n[{name}] {time.time()-t0:.1f}s status={res.get('status')} "
          f"profile={res.get('profile')} "
          f"source={rv.get('verdict') and rv.get('source')} "
          f"r2={(res.get('fit_quality') or {}).get('r_squared')} "
          f"drift={rv.get('drift')} sim={rv.get('fingerprint_similarity')} "
          f"cold_start={res.get('cold_start')}", flush=True)
    return res


def cs_pre():
    t0 = time.time()
    return _report("cs_pre", _agent("cs_pre").analyze(
        _frame(2), system_info=META, profile="realtime"), t0)


def cs_chain():
    out = []
    for n in (3, 4, 5):
        t0 = time.time()
        out.append(_report(f"cs_chain_{n:04d}", _agent(f"cs_chain_{n:04d}").analyze(
            _frame(n), system_info=META, profile="realtime",
            prior_analysis_paths=[str(OUT / "cs_pre")],
            reuse_locked_script=True), t0))
    return out[-1]


def cs_post():
    t0 = time.time()
    return _report("cs_post", _agent("cs_post").analyze(
        _frame(20), system_info=META, profile="realtime"), t0)


def cs_miss():
    x = np.linspace(0, 500, 1200)
    y = (8.0 * np.exp(-x / 60) + 2.0 * np.exp(-x / 300)
         + np.random.RandomState(3).normal(0, 0.03, x.size))
    f = OUT / "trpl_decay.csv"
    np.savetxt(f, np.column_stack([x, y]), delimiter=",",
               header="time_ns,counts", comments="")
    t0 = time.time()
    return _report("cs_miss", _agent("cs_miss").analyze(
        str(f), system_info={"technique": "time-resolved PL",
                             "description": "biexponential decay, x in ns"},
        profile="realtime"), t0)


ALL = {"cs_pre": cs_pre, "cs_chain": cs_chain, "cs_post": cs_post,
       "cs_miss": cs_miss}

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
