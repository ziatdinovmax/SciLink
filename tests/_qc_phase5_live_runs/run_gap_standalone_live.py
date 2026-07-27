"""Gap-matrix live legs, STANDALONE agents (sequential).

  reuse_curve    — #172 verbatim reuse of the phase-4 curve_raman locked script
  reuse_image    — #172 reuse of the phase-4 image_aunp locked script
  bon_curve      — best-of-N (n_candidates=2) on weak-metadata Raman
  bon_image      — best-of-N (n_candidates=2) on AuNP
  hf_curve       — human-feedback adjust_threshold live (patched input, live LLM):
                   r2_threshold=0.999 forces the poor-fit prompt; the "human"
                   answers "threshold 0.9"
  staging_curve  — T=2 staging fire attempt: starting_annealing_level=2 with an
                   ISOLATED memory store (SCILINK_HOME=tmp, SCILINK_MEMORY=1);
                   fires only if a hot-produced refit gets approved (stochastic)

Usage: python run_gap_standalone_live.py [names...]
"""

import builtins
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

STRONG = BENCH / "raman_chris" / "strong_metadata" / "Raman_1_data_hold"
WEAK = BENCH / "raman_chris" / "weak_metadata" / "Raman_3_data_hold"


def run_reuse_curve():
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    agent = CurveFittingAgent(model_name=MODEL, output_dir=str(OUT / "reuse_curve"),
                              enable_human_feedback=False, use_literature=False)
    res = agent.analyze(
        str(STRONG / "data1.csv"), system_info=str(STRONG / "metadata.json"),
        prior_analysis_paths=[str(P4 / "curve_raman")],
        reuse_locked_script=True,
    )
    _dump("reuse_curve", res)
    rv = res.get("reuse_validity")
    print(f"[reuse_curve] reuse_validity={rv}")
    return res


def run_reuse_image():
    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
    agent = ImageAnalysisAgent(model_name=MODEL, output_dir=str(OUT / "reuse_image"),
                               enable_human_feedback=False, use_literature=False)
    res = agent.analyze(
        str(BENCH / "TEM" / "(easy) AuNP.tif"),
        system_info=str(BENCH / "TEM" / "(easy) AuNP.json"),
        objective="Measure the size distribution of the Au nanoparticles.",
        prior_analysis_paths=[str(P4 / "image_aunp")],
        reuse_locked_script=True,
    )
    _dump("reuse_image", res)
    print(f"[reuse_image] reuse_validity={res.get('reuse_validity')}")
    return res


def run_bon_curve():
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    agent = CurveFittingAgent(model_name=MODEL, output_dir=str(OUT / "bon_curve"),
                              enable_human_feedback=False, use_literature=False)
    res = agent.analyze(str(WEAK / "data1.csv"),
                        system_info=str(WEAK / "metadata.json"),
                        n_candidates=2)
    _dump("bon_curve", res)
    return res


def run_bon_image():
    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
    agent = ImageAnalysisAgent(model_name=MODEL, output_dir=str(OUT / "bon_image"),
                               enable_human_feedback=False, use_literature=False)
    res = agent.analyze(
        str(BENCH / "TEM" / "(easy) AuNP.tif"),
        system_info=str(BENCH / "TEM" / "(easy) AuNP.json"),
        objective="Measure the size distribution of the Au nanoparticles.",
        n_candidates=2,
    )
    _dump("bon_image", res)
    return res


def run_hf_curve():
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    prompts_seen = []
    real_input = builtins.input

    def fake_input(prompt=""):
        prompts_seen.append(str(prompt))
        ans = "threshold 0.9" if "Your input" in str(prompt) else ""
        print(f"\n[fake-human] prompt≈{str(prompt)[:60]!r} -> {ans!r}", flush=True)
        return ans

    builtins.input = fake_input
    try:
        agent = CurveFittingAgent(model_name=MODEL, output_dir=str(OUT / "hf_curve"),
                                  enable_human_feedback=True, use_literature=False)
        res = agent.analyze(str(WEAK / "data1.csv"),
                            system_info=str(WEAK / "metadata.json"),
                            r2_threshold=0.999,
                            max_verification_iterations=2)
    finally:
        builtins.input = real_input
    _dump("hf_curve", res)
    print(f"[hf_curve] input prompts seen: {len(prompts_seen)}; "
          f"poor-fit prompt fired: {any('Your input' in p for p in prompts_seen)}")
    return res


def run_staging_curve():
    import tempfile
    store = tempfile.mkdtemp(prefix="scilink_store_")
    os.environ["SCILINK_HOME"] = store
    os.environ["SCILINK_MEMORY"] = "1"
    try:
        from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
        agent = CurveFittingAgent(model_name=MODEL, output_dir=str(OUT / "staging_curve"),
                                  enable_human_feedback=False, use_literature=False)
        res = agent.analyze(str(WEAK / "data1.csv"),
                            system_info=str(WEAK / "metadata.json"),
                            starting_annealing_level=2)
        _dump("staging_curve", res)
        staged = [str(p.relative_to(store)) for p in Path(store).rglob("*") if p.is_file()]
        print(f"[staging_curve] store={store}; staged files: {staged}")
        res["_staged_files"] = staged
        return res
    finally:
        os.environ.pop("SCILINK_HOME", None)
        os.environ.pop("SCILINK_MEMORY", None)


ALL = {
    "reuse_curve": run_reuse_curve,
    "reuse_image": run_reuse_image,
    "bon_curve": run_bon_curve,
    "bon_image": run_bon_image,
    "hf_curve": run_hf_curve,
    "staging_curve": run_staging_curve,
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
        print(f"LIVE {name.upper()} DONE in {time.time()-t0:.0f}s -> {results[name]}", flush=True)
    print("\nSUMMARY:", json.dumps(results, indent=2))
