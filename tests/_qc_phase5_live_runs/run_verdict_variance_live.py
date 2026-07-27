"""Measure the reuse-verdict score distribution across quality regimes.

3 reuse runs x 3 votes = 9 pooled votes per case (within-run AND cross-run
variance):
  aunp_same     — good result, same image (baseline; morning outlier 0.59?)
  vaterite_same — borderline result (0.59-0.63 vs gate 0.7)
  fmica_cross   — the REAL #172 case: the 2 mM anchor script from the
                  phase-4 image_series run applied to the 10 mM image
"""
import json, os, sys, time
from pathlib import Path
os.environ.setdefault("AWS_REGION_NAME", "us-east-1")
os.environ["UNSAFE_EXECUTION_OK"] = "true"
MODEL = "bedrock/us.anthropic.claude-opus-4-8"
BENCH = Path.home() / "Code" / "benchmarking_for_paper2"
OUT = Path(__file__).parent
P4 = OUT.parent / "_qc_phase4_live_runs"

def _reuse_votes(name, image, meta, prior, objective=None):
    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
    agent = ImageAnalysisAgent(model_name=MODEL,
                               output_dir=str(OUT / f"var_{name}"),
                               enable_human_feedback=False, use_literature=False)
    res = agent.analyze(str(image), system_info=str(meta), objective=objective,
                        prior_analysis_paths=[str(prior)],
                        reuse_locked_script=True)
    rv = res.get("reuse_validity") or {}
    return rv.get("score_votes") or [], rv.get("verdict")

CASES = {
    "aunp_same": dict(image=BENCH / "TEM" / "(easy) AuNP.tif",
                      meta=BENCH / "TEM" / "(easy) AuNP.json",
                      prior=P4 / "image_aunp",
                      objective="Measure the size distribution of the Au nanoparticles."),
    "vaterite_same": dict(image=BENCH / "OM" / "(easy) vaterite.tif",
                          meta=BENCH / "OM" / "(easy) vaterite.json",
                          prior=OUT / "vote_vaterite_prior",
                          objective="Count the vaterite particles and measure their size distribution."),
    "fmica_cross": dict(image=BENCH / "AFM" / "(hard) fmica 10mM KCl.tif",
                        meta=BENCH / "AFM" / "(hard) fmica 10mM KCl.json",
                        prior=OUT.parent / "_qc_phase4_live_runs" / "image_series",
                        objective="Characterize the surface structure/adsorbate pattern on mica in KCl."),
}

if __name__ == "__main__":
    pooled = {}
    for name, kw in CASES.items():
        votes_all, verdicts = [], []
        for rep in range(3):
            t0 = time.time()
            try:
                votes, verdict = _reuse_votes(f"{name}_r{rep}", **kw)
            except Exception as e:
                import traceback; traceback.print_exc()
                votes, verdict = [], f"EXC:{e}"
            votes_all += list(votes); verdicts.append(verdict)
            print(f"[{name} rep{rep}] votes={votes} verdict={verdict} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        pooled[name] = {"votes": votes_all, "verdicts": verdicts}
        print(f"POOLED {name}: {sorted(votes_all)}", flush=True)
    (OUT / "result_verdict_variance.json").write_text(json.dumps(pooled, indent=1))
    print("\nFINAL:", json.dumps(pooled, indent=1))
