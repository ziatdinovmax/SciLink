"""Live validation of the median-of-3 reuse-validity vote, 3 image datasets.

  aunp     — reuse the phase-4 prior (baseline: today's SINGLE-vote run of the
             identical reuse flipped to 'poor' at 0.59; phase-4 full QC scored
             the same output 0.81)
  vaterite — fresh prior run, then voted reuse (optical microscopy)
  mosbp    — fresh prior run, then voted reuse (AFM, peptide assembly)

Reports per dataset: the 3 votes, median, verdict.
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("AWS_REGION_NAME", "us-east-1")
os.environ["UNSAFE_EXECUTION_OK"] = "true"

MODEL = "bedrock/us.anthropic.claude-opus-4-8"
BENCH = Path.home() / "Code" / "benchmarking_for_paper2"
OUT = Path(__file__).parent
P4 = OUT.parent / "_qc_phase4_live_runs"

sys.path.insert(0, str(OUT))
from run_phase5_live import _dump  # noqa: E402


def _agent(outdir):
    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
    return ImageAnalysisAgent(model_name=MODEL, output_dir=str(outdir),
                              enable_human_feedback=False, use_literature=False)


def _reuse(name, image, meta, prior, objective=None):
    res = _agent(OUT / f"vote_{name}").analyze(
        str(image), system_info=str(meta), objective=objective,
        prior_analysis_paths=[str(prior)], reuse_locked_script=True,
    )
    _dump(f"vote_{name}", res)
    rv = res.get("reuse_validity") or {}
    print(f"[vote_{name}] votes={rv.get('score_votes')} "
          f"median={rv.get('quality_score')} verdict={rv.get('verdict')}")
    return res


def run_aunp():
    return _reuse("aunp", BENCH / "TEM" / "(easy) AuNP.tif",
                  BENCH / "TEM" / "(easy) AuNP.json", P4 / "image_aunp",
                  objective="Measure the size distribution of the Au nanoparticles.")


def _fresh_pair(name, image, meta, objective=None):
    prior_dir = OUT / f"vote_{name}_prior"
    res1 = _agent(prior_dir).analyze(str(image), system_info=str(meta),
                                     objective=objective)
    print(f"[{name} prior] status={res1.get('status')}")
    return _reuse(name, image, meta, prior_dir, objective=objective)


def run_vaterite():
    return _fresh_pair("vaterite", BENCH / "OM" / "(easy) vaterite.tif",
                       BENCH / "OM" / "(easy) vaterite.json",
                       objective="Count the vaterite particles and measure their size distribution.")


def run_mosbp():
    return _fresh_pair("mosbp", BENCH / "AFM" / "(easy) MoSBP1.tif",
                       BENCH / "AFM" / "(easy) MoSBP1.json")


ALL = {"aunp": run_aunp, "vaterite": run_vaterite, "mosbp": run_mosbp}

if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    results = {}
    for name in names:
        t0 = time.time()
        print(f"\n{'='*70}\nLIVE VOTE_{name.upper()} START\n{'='*70}", flush=True)
        try:
            results[name] = ALL[name]().get("status")
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results[name] = f"EXCEPTION: {e}"
        print(f"LIVE VOTE_{name.upper()} DONE in {time.time()-t0:.0f}s -> {results[name]}", flush=True)
    print("\nSUMMARY:", json.dumps(results, indent=2))
