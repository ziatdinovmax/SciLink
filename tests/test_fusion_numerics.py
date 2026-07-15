"""Offline tests for the fusion numerics bundle (issue #296 phase a).

No network — monkeypatches the gate/fusion LLM and the ephemeral child, like
test_meta_fanout_robustness.py. The fake children write REAL result artifacts
(features.csv / analysis_results.json) into their branch dirs so
_branch_numerics exercises actual file reads. Assertions check that the
schema previews reach the fusion prompt and the fusion report, and that a
branch with no numerics degrades to text fusion instead of blocking.

  conda run -n scilink python tests/test_fusion_numerics.py
"""
import csv
import json
import os
import re
import tempfile

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

import numpy as np
import scilink.agents.meta_agent.fanout as fo
from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent, MetaMode


def _agent():
    d = tempfile.mkdtemp()
    A = os.path.join(d, "A.npy"); B = os.path.join(d, "B.npy")
    np.save(A, np.zeros((8, 8))); np.save(B, np.ones((8, 8)))
    ag = MetaOrchestratorAgent(base_dir=d, api_key="sk-dummy",
                               model_name="claude-opus-4-6",
                               meta_mode=MetaMode.AUTONOMOUS)
    return ag, A, B


# Behavior map: primary data_path -> 'series' | 'single' | 'scalar_only' | 'bare'
BEHAVIORS = {}


def _write_series_artifacts(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ft = os.path.join(out_dir, "features.csv")
    with open(ft, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "temperature_C", "peak_1_center", "peak_1_area",
                    "fit_crystallinity_index", "fit_r_squared"])
        for i, t in enumerate((30.0, 50.0, 70.0, 90.0, 110.0)):
            w.writerow([f"idx_{i}", t, 28.4 + 0.01 * i, 1200 - 40 * i,
                        0.9 - 0.1 * i, 0.998])
    ar = os.path.join(out_dir, "analysis_results.json")
    with open(ar, "w") as fh:
        json.dump({"trend_analysis": {
            "tracked": "fit_crystallinity_index vs temperature",
            "breakpoint_C": 90.0}}, fh)
    sf = os.path.join(out_dir, "series_fit_results.json")
    with open(sf, "w") as fh:
        json.dump({"total_spectra": 5}, fh)
    return ft, ar, sf


def _write_single_artifacts(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ft = os.path.join(out_dir, "features.csv")
    with open(ft, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "transition_temperature_C", "fit_r_squared"])
        w.writerow(["data1", 91.2, 0.995])
    ar = os.path.join(out_dir, "analysis_results.json")
    with open(ar, "w") as fh:
        json.dump({"fitting_parameters": {"transition_temperature_C": 91.2},
                   "fit_quality": {"r_squared": 0.995}}, fh)
    return ft, ar


def _write_scalar_only_artifacts(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ar = os.path.join(out_dir, "analysis_results.json")
    with open(ar, "w") as fh:
        json.dump({"fitting_parameters": {"onset_C": 91.0},
                   "fit_quality": {"r_squared": 0.99}}, fh)
    return (ar,)


def _install_fake_child():
    def fake_child(orch, base_dir):
        class C:
            def run_task(self, task, context=None, autonomy=None):
                m = re.search(r"PRIMARY dataset for THIS analysis: (\S+)", task)
                primary = m.group(1) if m else None
                beh = BEHAVIORS.get(primary, "bare")
                out_dir = os.path.join(str(base_dir), "results", "run_001")
                result = {"status": "success", "summary": f"{beh} summary",
                          "key_findings": [f"{beh} finding"],
                          "files_produced": []}
                if beh == "series":
                    ft, ar, sf = _write_series_artifacts(out_dir)
                    result["files_produced"] = [ft, ar, sf]
                    result["feature_tables"] = [ft]
                elif beh == "single":
                    # feature_tables deliberately OMITTED: exercises the
                    # files_produced fallback in _branch_numerics.
                    ft, ar = _write_single_artifacts(out_dir)
                    result["files_produced"] = [ft, ar]
                elif beh == "scalar_only":
                    (ar,) = _write_scalar_only_artifacts(out_dir)
                    result["files_produced"] = [ar]
                return result
        return C()
    fo._make_ephemeral_analysis_child = fake_child


FUSION_PROMPTS = []


def _install_fake_gate(fanout_set, join_axis="sample temperature"):
    def fake(orch, prompt, extra_parts=None):
        if "complementary measurements of ONE system" in prompt:  # HOLISTIC fusion
            FUSION_PROMPTS.append(prompt)
            return {"detailed_analysis": "fused narrative",
                    "scientific_claims": [{"claim": "c", "keywords": ["k"]}]}
        return {"verdict": "complementary", "confidence": 0.9, "rationale": "r",
                "join_axis": join_axis, "fanout_set": list(fanout_set),
                "redundant_clusters": [], "unrelated": [], "excluded_notes": ""}
    fo._llm_json = fake


def _branches(*paths):
    return [{"data_path": p, "task": f"Analyze {p}", "label": os.path.basename(p)}
            for p in paths]


def _run(ag, A, B, behA, behB, join_axis="sample temperature"):
    BEHAVIORS.clear(); BEHAVIORS[A] = behA; BEHAVIORS[B] = behB
    _install_fake_gate([A, B], join_axis=join_axis)
    ag._complementarity_cache.clear()
    out = json.loads(ag._run_fanout(_branches(A, B)))
    idxs = [r["delegation_index"] for r in out["results"] if r["produced_output"]]
    FUSION_PROMPTS.clear()
    fused = json.loads(ag._fuse_delegations(idxs))
    return out, fused, (FUSION_PROMPTS[-1] if FUSION_PROMPTS else "")


results = {}


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def main():
    _install_fake_child()

    # 1) series + single: previews for both reach the prompt + report.
    ag, A, B = _agent()
    out, fused, prompt = _run(ag, A, B, "series", "single")
    print("1) series + single-measurement:")
    check("both branches ran", out["branches_run"] == 2)
    check("entries stamped with join_axis",
          all(e.get("join_axis") == "sample temperature"
              for e in ag._delegation_ledger if e.get("fanout")))
    check("fusion success", fused["status"] == "success")
    check("numerics_branches == 2", fused.get("numerics_branches") == 2)
    check("join_axis in fusion result", fused.get("join_axis") == "sample temperature")
    check("NUMERICS preamble in prompt", "NUMERICS:" in prompt)
    check("join axis named in preamble", "sample temperature" in prompt)
    check("series shape in prompt", "5 rows x 6 cols" in prompt)
    check("join-axis column + range in prompt",
          "temperature_C (range 30 to 110, 5 points)" in prompt)
    check("column names in prompt", "fit_crystallinity_index" in prompt)
    check("trend analysis in prompt",
          "breakpoint_C" in prompt and "trend analysis" in prompt)
    check("single-row values inlined", "single-row values" in prompt
          and "91.2" in prompt)
    check("table paths in prompt (audit)", "features.csv" in prompt)
    report = json.load(open(fused["report_path"]))
    check("report carries branch_numerics for both",
          isinstance(report.get("branch_numerics"), dict)
          and len(report["branch_numerics"]) == 2)
    check("report carries join_axis", report.get("join_axis") == "sample temperature")

    # 2) series + bare (no numeric artifacts): degrade, never block.
    ag, A, B = _agent()
    out, fused, prompt = _run(ag, A, B, "series", "bare")
    print("2) numerics-less branch degrades:")
    check("fusion still success", fused["status"] == "success")
    check("only one branch contributed numerics",
          fused.get("numerics_branches") == 1)
    check("series preview still present", "5 rows x 6 cols" in prompt)

    # 3) scalar-only branch (analysis_results.json, no features.csv):
    #    fitting_parameters fallback reaches the prompt.
    ag, A, B = _agent()
    out, fused, prompt = _run(ag, A, B, "series", "scalar_only")
    print("3) scalar fitting_parameters fallback:")
    check("both branches contributed numerics", fused.get("numerics_branches") == 2)
    check("fitting parameters in prompt",
          "fitting parameters" in prompt and "onset_C" in prompt)
    check("fit quality in prompt", "fit quality" in prompt)

    # 4) join axis that matches no column: reported, not fatal.
    ag, A, B = _agent()
    out, fused, prompt = _run(ag, A, B, "series", "single",
                              join_axis="magnetic field strength")
    print("4) join-axis column miss:")
    check("fusion success on axis miss", fused["status"] == "success")
    check("miss reported in preview",
          "none matched the gate's join axis" in prompt)

    print("\n" + "=" * 50)
    npass = sum(results.values())
    print(f"FUSION NUMERICS: {npass}/{len(results)} checks passed")
    for k, v in results.items():
        if not v:
            print("  FAILED:", k)
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
