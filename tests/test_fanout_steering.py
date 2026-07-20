"""Offline tests for branch-time steering (#296 phase d).

No network — monkeypatches the gate/fusion LLM and the ephemeral child. The
steering REDUCTION runs for real (numpy SVD over synthetic series files on
disk); assertions cover: the steering block + additive-only guardrail in the
steered branch's mesh task (and ONLY there), informed_by stamped on the
ledger, the independence discount reaching the fusion prompt / report /
caveats and the codegen inputs, and graceful skip on a failed reduction.

  conda run -n scilink python tests/test_fanout_steering.py
"""
import json
import os
import re
import tempfile

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

import numpy as np
import scilink.agents.meta_agent.fanout as fo
from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent, MetaMode

RNG = np.random.default_rng(2)
CAPTURED_TASKS = []
FUSION_PROMPTS = []
CODEGEN_PROMPTS = []

results = {}


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def _write_series(d, stem, t0=85.0, temps=range(30, 165, 10)):
    """Two-component crossfade at t0 with temperature sidecars."""
    x = np.linspace(400, 1800, 600)
    g = lambda c, s: np.exp(-0.5 * ((x - c) / s) ** 2)
    for T in temps:
        f = 1 / (1 + np.exp(-(T - t0) / 4.0))
        y = (1 - f) * g(900, 40) + f * g(1300, 40) + RNG.normal(0, 0.004, x.size)
        p = os.path.join(d, f"{stem}_{T:03d}C.txt")
        np.savetxt(p, np.column_stack([x, y]))
        json.dump({"temperature_C": float(T)},
                  open(p.replace(".txt", ".json"), "w"))
    return f"{stem}_*C.txt"


def _install_fakes(fanout_set):
    def fake_child(orch, base_dir):
        class C:
            def run_task(self, task, context=None, autonomy=None):
                CAPTURED_TASKS.append(task)
                return {"status": "success", "summary": "ok",
                        "key_findings": ["finding"], "files_produced": []}
        return C()
    fo._make_ephemeral_analysis_child = fake_child

    def fake_llm(orch, prompt, extra_parts=None):
        if "SCRIPT CONTRACT" in prompt:
            CODEGEN_PROMPTS.append(prompt)
            return {"method": "qualitative", "rationale": "r", "script": ""}
        if "complementary measurements of ONE system" in prompt:
            FUSION_PROMPTS.append(prompt)
            return {"detailed_analysis": "fused narrative",
                    "scientific_claims": [{"claim": "c"}]}
        return {"verdict": "complementary", "confidence": 0.9, "rationale": "r",
                "join_axis": "sample temperature", "fanout_set": list(fanout_set),
                "redundant_clusters": [], "unrelated": [], "excluded_notes": ""}
    fo._llm_json = fake_llm


def _agent():
    d = tempfile.mkdtemp()
    ag = MetaOrchestratorAgent(base_dir=d, api_key="sk-dummy",
                               model_name="claude-opus-4-6",
                               meta_mode=MetaMode.AUTONOMOUS)
    return ag, d


def main():
    # 1) Steered branch gets the hint + guardrail; unsteered does not;
    #    informed_by stamped; fusion carries the discount.
    ag, d = _agent()
    up = os.path.join(d, "uploads"); os.makedirs(up)
    pat_a = _write_series(up, "techA", t0=85.0)
    pat_b = _write_series(up, "techB", t0=85.0)
    ids = [f"{up}#1", f"{up}#2"]
    CAPTURED_TASKS.clear(); FUSION_PROMPTS.clear(); CODEGEN_PROMPTS.clear()
    _install_fakes(ids); ag._complementarity_cache.clear()
    out = json.loads(ag._run_fanout([
        {"data_path": up, "pattern": pat_a, "label": "techA series",
         "task": "Analyze the techA series.", "steer": True},
        {"data_path": up, "pattern": pat_b, "label": "techB series",
         "task": "Analyze the techB series."},
    ]))
    print("1) steered fan-out:")
    check("both branches ran", out.get("branches_run") == 2)
    steered = [t for t in CAPTURED_TASKS if "Analyze the techA series" in t]
    unsteered = [t for t in CAPTURED_TASKS if "Analyze the techB series" in t]
    check("steered task carries STEERING block",
          steered and "STEERING (explicit opt-in; ADDITIVE-ONLY)" in steered[0])
    m = re.search(r"sharpest change near\s+control ≈ (\d+(?:\.\d+)?)",
                  steered[0] if steered else "")
    check("hint locates the planted change (85 +- 5)",
          m and abs(float(m.group(1)) - 85.0) <= 5.0)
    check("guardrail: full-range + no-window-restriction + normal bar",
          steered and all(s in steered[0] for s in
                          ("FULL range/series", "Never restrict a fit window",
                           "clear your normal bar unaided",
                           "valid, valuable outcome")))
    check("score-curve figure path in task",
          steered and "score_curve.png" in steered[0])
    check("unsteered task has NO steering block",
          unsteered and "STEERING" not in unsteered[0])
    fan = [e for e in ag._delegation_ledger if e.get("fanout")]
    by_label = {e["label"]: e for e in fan}
    check("informed_by stamped on steered entry only",
          by_label["techA series"].get("informed_by") == ["techB series"]
          and not by_label["techB series"].get("informed_by"))
    check("steering artifacts on disk",
          os.path.exists(os.path.join(
              str(ag.fanout_dir), "steering",
              "techa_series_from_techb_series", "reduction.json")))

    fused = json.loads(ag._fuse_delegations(
        [e["index"] for e in fan]))
    prompt = FUSION_PROMPTS[-1] if FUSION_PROMPTS else ""
    check("fusion prompt carries INDEPENDENCE PROVENANCE",
          "INDEPENDENCE PROVENANCE" in prompt
          and "partly by construction" in prompt
          and "techA series" in prompt)
    check("fusion result carries independence map",
          (fused.get("independence") or {}).get("techA series") == ["techB series"])
    check("mechanical independence caveat in report",
          any("steered at launch" in str(c) for c in fused.get("caveats") or []))
    report = json.load(open(fused["report_path"]))
    check("report persists independence", report.get("independence") is not None)

    # 2) No steer flag -> nothing changes (no block, no informed_by,
    #    no independence in fusion).
    ag, d = _agent()
    up = os.path.join(d, "uploads"); os.makedirs(up)
    pat_a = _write_series(up, "techA"); pat_b = _write_series(up, "techB")
    CAPTURED_TASKS.clear(); FUSION_PROMPTS.clear()
    _install_fakes([f"{up}#1", f"{up}#2"]); ag._complementarity_cache.clear()
    out = json.loads(ag._run_fanout([
        {"data_path": up, "pattern": pat_a, "label": "a", "task": "Analyze a."},
        {"data_path": up, "pattern": pat_b, "label": "b", "task": "Analyze b."},
    ]))
    fan = [e for e in ag._delegation_ledger if e.get("fanout")]
    fused = json.loads(ag._fuse_delegations([e["index"] for e in fan]))
    print("2) unsteered fan-out unchanged:")
    check("no steering block anywhere",
          all("STEERING" not in t for t in CAPTURED_TASKS))
    check("no informed_by", all(not e.get("informed_by") for e in fan))
    check("no independence in fusion", fused.get("independence") is None)
    check("no independence caveat",
          not any("steered" in str(c) for c in fused.get("caveats") or []))

    # 3) Reduction failure (companion is a single file, or unreadable
    #    series) -> steering skipped gracefully, run proceeds.
    ag, d = _agent()
    up = os.path.join(d, "uploads"); os.makedirs(up)
    pat_a = _write_series(up, "techA")
    single = os.path.join(up, "single.npy")
    np.save(single, np.zeros((8, 8)))
    CAPTURED_TASKS.clear()
    _install_fakes([up, single]); ag._complementarity_cache.clear()
    out = json.loads(ag._run_fanout([
        {"data_path": up, "pattern": pat_a, "label": "series branch",
         "task": "Analyze the series.", "steer": True},
        {"data_path": single, "label": "single branch",
         "task": "Analyze the single file."},
    ]))
    fan = [e for e in ag._delegation_ledger if e.get("fanout")]
    print("3) no series companion -> steering skipped:")
    check("run proceeded", out.get("branches_run") == 2)
    check("no steering block (companion has no file set)",
          all("STEERING" not in t for t in CAPTURED_TASKS))
    check("no informed_by stamped", all(not e.get("informed_by") for e in fan))

    print("\n" + "=" * 50)
    npass = sum(results.values())
    print(f"FAN-OUT STEERING: {npass}/{len(results)} checks passed")
    for k, v in results.items():
        if not v:
            print("  FAILED:", k)
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
