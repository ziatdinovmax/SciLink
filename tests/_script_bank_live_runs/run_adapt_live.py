"""Script-bank adapt-mode retrieval live matrix (#346 step 2) — bedrock opus-4-8.

Uses the bank seeded by run_bank_live.py (record 7ff14941 = Raman_1 script).

  1. adapt_self   — Raman_1 again, fresh session, no prior paths: exemplar
                    must be offered (dry-run score 0.815) and adapted; bank
                    stats must show the retrieval + the new success.
  2. adapt_cross  — Raman_14 (different compound, dry-run score 0.625): the
                    exemplar is offered but its peak positions are WRONG for
                    this data — the adapt framing must yield a fit at THIS
                    spectrum's bands (~445/610/235), not the exemplar's.
  3. adapt_miss   — Raman_13 (below the 0.45 floor): no exemplar line in the
                    log; from-scratch generation unchanged.

Usage: python run_adapt_live.py [names...]   (needs AWS_BEARER_TOKEN_BEDROCK)
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
BENCH = Path.home() / "Code" / "benchmarking_for_paper2" / "raman_chris" / "strong_metadata"


def _show_bank(tag):
    from scilink.skills._shared import _script_bank as sb
    print(f"\n----- BANK STATE after {tag} -----")
    for rec in sb.list_records("curve_fitting"):
        print(json.dumps({
            "id": rec["id"], "n_successes": rec["stats"]["n_successes"],
            "n_retrievals": rec["stats"].get("n_retrievals", 0),
            "sessions": rec["sessions"],
            "fp_top_peaks": [p["position"] for p in
                             ((rec.get("data_fingerprint") or {}).get("peaks") or {}).get("top", [])[:3]],
            "metric": rec.get("outcome", {}).get("metric"),
        }))
    print("----- END BANK STATE -----\n", flush=True)


def _run_case(name, case_dir, out_name):
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    case = BENCH / case_dir
    agent = CurveFittingAgent(
        model_name=MODEL, output_dir=str(OUT / out_name),
        enable_human_feedback=False, use_literature=False,
    )
    res = agent.analyze(str(case / "data1.csv"),
                        system_info=str(case / "metadata.json"))
    res.pop("visualization_bytes", None)
    path = OUT / f"result_{name}.json"

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if k != "visualization_bytes"}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return "<bytes>" if isinstance(o, bytes) else o
    path.write_text(json.dumps(clean(res), indent=2, default=str))
    print(f"\n[{name}] status={res.get('status')} "
          f"model={str(res.get('model_type'))[:80]} "
          f"r2={(res.get('fit_quality') or {}).get('r_squared')} "
          f"banked={res.get('banked_scripts')}", flush=True)
    _show_bank(name)
    return res


ALL = {
    "adapt_self": lambda: _run_case("adapt_self", "Raman_1_data_hold", "adapt_self"),
    "adapt_cross": lambda: _run_case("adapt_cross", "Raman_14_data_hold", "adapt_cross"),
    "adapt_miss": lambda: _run_case("adapt_miss", "Raman_13_data_hold", "adapt_miss"),
    # Raman_13 turned into a hit once adapt_cross banked Raman_14 (the bank
    # grows); Raman_9 misses against the grown bank — the true no-exemplar case.
    "adapt_miss2": lambda: _run_case("adapt_miss2", "Raman_9_data_hold", "adapt_miss2"),
}

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
