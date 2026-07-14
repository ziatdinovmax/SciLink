"""Offline robustness / edge-case tests for the meta fan-out primitive.

No network — monkeypatches the gate LLM and the ephemeral child so the
orchestration logic (branch errors, empty branches, fusion guards, concurrency,
ledger integrity) is exercised deterministically and fast. Complements the
live test (gate judgment + real codegen) in test_meta_fanout_live.py.

  conda run -n scilink python tests/test_meta_fanout_robustness.py
"""
import json
import os
import tempfile
import threading
import time

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

import numpy as np
import scilink.agents.meta_agent.fanout as fo
from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent, MetaMode


def _agent(mode="AUTONOMOUS"):
    d = tempfile.mkdtemp()
    A = os.path.join(d, "A.npy"); B = os.path.join(d, "B.npy"); C = os.path.join(d, "C.npy")
    np.save(A, np.zeros((8, 8))); np.save(B, np.ones((8, 8))); np.save(C, np.full((8, 8), 2.0))
    ag = MetaOrchestratorAgent(base_dir=d, api_key="sk-dummy",
                               model_name="claude-opus-4-6", meta_mode=MetaMode[mode])
    return ag, A, B, C


# Behavior map: substring-in-task -> 'good' | 'empty' | 'error' | 'slow'
BEHAVIORS = {}
_ACTIVE = {"n": 0, "max": 0, "lock": threading.Lock()}
CAPTURED_TASKS = []   # the mesh task each fake child actually received


def _install_fake_child():
    def fake_child(orch, base_dir):
        class C:
            def run_task(self, task, context=None, autonomy=None):
                CAPTURED_TASKS.append(task)
                with _ACTIVE["lock"]:
                    _ACTIVE["n"] += 1
                    _ACTIVE["max"] = max(_ACTIVE["max"], _ACTIVE["n"])
                try:
                    # Match behavior to the PRIMARY dataset only — full-mesh puts
                    # every path in every task, so substring-match would alias.
                    import re
                    m = re.search(r"PRIMARY dataset for THIS analysis: (\S+)", task)
                    primary = m.group(1) if m else None
                    beh = (BEHAVIORS.get(primary, "good") if primary
                           else next((v for k, v in BEHAVIORS.items() if k in task), "good"))
                    if beh == "slow":
                        time.sleep(0.3); beh = "good"
                    if beh == "error":
                        raise RuntimeError("synthetic branch failure")
                    if beh == "empty":
                        return {"status": "success", "summary": "blocked, no sandbox",
                                "key_findings": [], "files_produced": []}
                    return {"status": "success", "summary": f"ok",
                            "key_findings": ["finding"], "files_produced": ["/x/v.png"]}
                finally:
                    with _ACTIVE["lock"]:
                        _ACTIVE["n"] -= 1
        return C()
    fo._make_ephemeral_analysis_child = fake_child


def _verdict(paths, fanout_set, join_type=None):
    return {"verdict": "complementary", "confidence": 0.9, "rationale": "r",
            "join_axis": "grid", "join_type": join_type,
            "fanout_set": list(fanout_set),
            "redundant_clusters": [], "unrelated": [], "excluded_notes": ""}


GATE_PROMPTS = []   # every prompt the fake gate LLM received (for inspection)


def _install_fake_gate(fanout_set, join_type=None):
    def fake(orch, prompt, extra_parts=None):
        GATE_PROMPTS.append(prompt)
        if "complementary measurements of ONE system" in prompt:   # HOLISTIC fusion prompt
            return {"detailed_analysis": "fused narrative",
                    "scientific_claims": [{"claim": "c", "keywords": ["k"]}]}
        return _verdict(None, fanout_set, join_type=join_type)
    fo._llm_json = fake


def _branches(*paths):
    return [{"data_path": p, "task": f"Analyze {p}", "label": os.path.basename(p)}
            for p in paths]


results = {}


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def main():
    _install_fake_child()

    # 1) One branch errors -> reported, not productive, no fuse recommended.
    ag, A, B, C = _agent()
    BEHAVIORS.clear(); BEHAVIORS[A] = "good"; BEHAVIORS[B] = "error"
    _install_fake_gate([A, B]); ag._complementarity_cache.clear()
    out = json.loads(ag._run_fanout(_branches(A, B)))
    print("1) branch error:")
    check("run still 'success' overall", out["status"] == "success")
    check("branches_run == 2", out["branches_run"] == 2)
    check("branches_with_output == 1", out["branches_with_output"] == 1)
    errored = [r for r in out["results"] if r["status"] == "error"]
    check("errored branch recorded with status=error", len(errored) == 1)
    check("error: warning surfaced", "warning" in out)
    check("error: next_step says do NOT fuse", "do NOT fuse" in out["next_step"])

    # 2) One empty-but-successful branch -> flagged.
    ag, A, B, C = _agent()
    BEHAVIORS.clear(); BEHAVIORS[A] = "good"; BEHAVIORS[B] = "empty"
    _install_fake_gate([A, B]); ag._complementarity_cache.clear()
    out = json.loads(ag._run_fanout(_branches(A, B)))
    print("2) empty-but-successful branch:")
    check("empty: branches_with_output == 1", out["branches_with_output"] == 1)
    check("empty: warning surfaced", "warning" in out)

    # 3) Fuse edge cases.
    ag, A, B, C = _agent()
    BEHAVIORS.clear(); BEHAVIORS[A] = "good"; BEHAVIORS[B] = "good"
    _install_fake_gate([A, B]); ag._complementarity_cache.clear()
    out = json.loads(ag._run_fanout(_branches(A, B)))
    good_idxs = [r["delegation_index"] for r in out["results"] if r["produced_output"]]
    print("3) fuse edge cases:")
    f1 = json.loads(ag._fuse_delegations(good_idxs[:1]))
    check("fuse with <2 indices -> error", f1["status"] == "error")
    f2 = json.loads(ag._fuse_delegations([999, 1000]))
    check("fuse with nonexistent indices -> error", f2["status"] == "error")
    f3 = json.loads(ag._fuse_delegations(good_idxs))
    check("fuse with 2 good -> success", f3["status"] == "success")
    check("fusion wrote a report file", os.path.exists(f3.get("report_path", "")))
    check("fusion recorded as ledger entry mode=fusion",
          any(e.get("mode") == "fusion" for e in ag._delegation_ledger))

    # 4) Concurrency + ledger integrity at N=4 (slow children must overlap).
    ag, A, B, C = _agent()
    Dp = os.path.join(os.path.dirname(A), "D.npy"); np.save(Dp, np.full((8, 8), 3.0))
    BEHAVIORS.clear()
    for p in (A, B, C, Dp): BEHAVIORS[p] = "slow"
    _install_fake_gate([A, B, C, Dp]); ag._complementarity_cache.clear()
    _ACTIVE["max"] = 0
    t0 = time.time()
    out = json.loads(ag._run_fanout(_branches(A, B, C, Dp)))
    dt = time.time() - t0
    print("4) concurrency N=4:")
    check("4 branches ran", out["branches_run"] == 4)
    idxs = [r["delegation_index"] for r in out["results"]]
    check("ledger indices unique", len(set(idxs)) == 4)
    check("indices contiguous & ordered", idxs == sorted(idxs))
    check("ran concurrently (>=2 overlap)", _ACTIVE["max"] >= 2)
    check("wall-clock < serial (4x0.3=1.2s)", dt < 1.0)
    grp = {e.get("parallel_group") for e in ag._delegation_ledger if e.get("fanout")}
    check("all 4 share one parallel_group", len(grp) == 1)

    # 5) Single-branch fan-out rejected.
    ag, A, B, C = _agent()
    _install_fake_gate([A]); ag._complementarity_cache.clear()
    out = json.loads(ag._run_fanout(_branches(A)))
    print("5) single branch:")
    check("single branch -> error", out["status"] == "error")

    # 6) Gate fail-closed on unparseable verdict.
    ag, A, B, C = _agent()
    fo._llm_json = lambda orch, prompt: None
    ag._complementarity_cache.clear()
    out = json.loads(ag._run_fanout(_branches(A, B)))
    print("6) gate fail-closed:")
    check("unparseable gate -> declined", out["status"] == "declined")

    # 7) Size caps: many complementary datasets.
    def _many(n):
        ag, A, B, C = _agent()
        d = os.path.dirname(A)
        paths = [os.path.join(d, f"m{i}.npy") for i in range(n)]
        for p in paths:
            np.save(p, np.zeros((8, 8)))
        BEHAVIORS.clear()
        for p in paths: BEHAVIORS[p] = "good"
        _install_fake_gate(paths); ag._complementarity_cache.clear()
        return ag, paths
    print("7) size caps (autonomous):")
    ag, paths = _many(6)   # > SOFT_CAP (5)
    out = json.loads(ag._run_fanout(_branches(*paths)))
    check("6-way autonomous -> declined (soft cap)", out["status"] == "declined")
    ag, paths = _many(9)   # > HARD_CAP (8)
    out = json.loads(ag._run_fanout(_branches(*paths)))
    check("9-way -> declined (hard cap)", out["status"] == "declined")
    ag, paths = _many(3)   # within caps
    out = json.loads(ag._run_fanout(_branches(*paths)))
    check("3-way autonomous -> runs", out["status"] == "success" and out["branches_run"] == 3)

    # 8) issue #326: two DISTINCT branches sharing ONE directory data_path
    #    (in-situ series in a single upload folder) must both run, and the
    #    gate must see their analysis tasks + per-branch ids.
    ag, A, B, C = _agent()
    up = os.path.join(os.path.dirname(A), "uploads")
    os.makedirs(up, exist_ok=True)
    np.save(os.path.join(up, "ftir_000.npy"), np.zeros(64))
    np.save(os.path.join(up, "xrd_000.npy"), np.ones(64))
    ids = [f"{up}#1", f"{up}#2"]
    BEHAVIORS.clear()
    _install_fake_gate(ids); ag._complementarity_cache.clear()
    out = json.loads(ag._run_fanout([
        {"data_path": up, "task": "Fit the in-situ FTIR series", "label": "ftir_series"},
        {"data_path": up, "task": "Fit the in-situ XRD series", "label": "xrd_series"},
    ]))
    print("8) same-dir distinct branches (issue #326):")
    check("both same-path branches ran", out.get("branches_run") == 2)
    labels = sorted(r["label"] for r in out.get("results", []))
    check("branches distinct, not one duplicated",
          labels == ["ftir_series", "xrd_series"])
    fan_tasks = " || ".join(e["task"] for e in ag._delegation_ledger if e.get("fanout"))
    check("ledger carries both distinct tasks",
          "FTIR" in fan_tasks and "XRD" in fan_tasks)
    gate_prompt = GATE_PROMPTS[-1]
    check("gate descriptor includes analysis tasks",
          "in-situ FTIR series" in gate_prompt and "in-situ XRD series" in gate_prompt)
    check("gate descriptor includes per-branch ids",
          ids[0] in gate_prompt and ids[1] in gate_prompt)

    # 9) True duplicates — identical (data_path, task) — are deduped with the
    #    distinct sibling kept; an all-duplicate call errors out loudly.
    ag, A, B, C = _agent()
    BEHAVIORS.clear()
    _install_fake_gate([A, B]); ag._complementarity_cache.clear()
    dup = {"data_path": A, "task": "Analyze A", "label": "a1"}
    out = json.loads(ag._run_fanout(
        [dup, dict(dup), {"data_path": B, "task": "Analyze B", "label": "b"}]))
    print("9) true-duplicate dedup:")
    check("duplicate dropped, 2 distinct branches ran", out.get("branches_run") == 2)
    out = json.loads(ag._run_fanout([dup, dict(dup)]))
    check("all-duplicates -> error mentioning the drop",
          out.get("status") == "error" and "duplicate" in out.get("message", "").lower())

    # 10) Gate cache is keyed per-branch: a stale path-keyed verdict must NOT
    #     be reused for two same-path branches (it was computed without task
    #     context and would wrongly decline / collapse them).
    ag, A, B, C = _agent()
    up2 = os.path.join(os.path.dirname(A), "uploads2")
    os.makedirs(up2, exist_ok=True)
    np.save(os.path.join(up2, "s.npy"), np.zeros(4))
    ag._complementarity_cache[frozenset({up2})] = {
        "verdict": "redundant", "confidence": 0.9, "rationale": "stale path-keyed",
        "join_axis": None, "fanout_set": [], "redundant_clusters": [], "unrelated": []}
    BEHAVIORS.clear()
    _install_fake_gate([f"{up2}#1", f"{up2}#2"]);
    out = json.loads(ag._run_fanout([
        {"data_path": up2, "task": "Fit the FTIR series", "label": "f"},
        {"data_path": up2, "task": "Fit the XRD series", "label": "x"},
    ]))
    print("10) per-branch gate-cache key:")
    check("stale path-keyed verdict not reused; both branches ran",
          out.get("branches_run") == 2)

    # 11) Long INLINE metadata (a prose description, not a path) must not
    #     crash the gate or mesh-task composition: Path.exists() on a string
    #     longer than the OS filename limit raises OSError.
    ag, A, B, C = _agent()
    BEHAVIORS.clear()
    _install_fake_gate([A, B]); ag._complementarity_cache.clear()
    long_meta = ("in-situ FTIR temperature series, files 'In-situ ibuprofen_"
                 "<T>C.txt' at 35, 40, 50, 60, 70, 80, 90, 100, 110, 120 C; "
                 "each file has a matching sidecar JSON carrying temperature_C."
                 " Two columns: wavenumber (cm-1), absorbance.") * 3
    out = json.loads(ag._run_fanout([
        {"data_path": A, "task": "Analyze A", "label": "a", "metadata": long_meta},
        {"data_path": B, "task": "Analyze B", "label": "b", "metadata": long_meta},
    ]))
    print("11) long inline metadata:")
    check("no crash; both branches ran", out.get("branches_run") == 2)
    check("inline metadata reached the gate descriptors",
          "wavenumber" in GATE_PROMPTS[-1])

    # 12) Fix 3: same-dir branches WITH patterns -> gate probes each branch's
    #     own files, and companions hand over a representative FILE (loadable
    #     as an auxiliary operand), not the shared extension-less directory.
    #     (join_type co_registered: the operand mesh is the one wiring that
    #     still passes companions after the mesh-policy change.)
    ag, A, B, C = _agent()
    up3 = os.path.join(os.path.dirname(A), "uploads3")
    os.makedirs(up3, exist_ok=True)
    for i in range(3):
        np.save(os.path.join(up3, f"ftir_{i:02d}.npy"), np.zeros(32))
    for i in range(2):
        np.save(os.path.join(up3, f"xrd_{i:02d}.npy"), np.ones(64))
    ids3 = [f"{up3}#1", f"{up3}#2"]
    BEHAVIORS.clear()
    _install_fake_gate(ids3, join_type="co_registered")
    ag._complementarity_cache.clear()
    CAPTURED_TASKS.clear()
    out = json.loads(ag._run_fanout([
        {"data_path": up3, "task": "Fit the FTIR series", "label": "ftir",
         "pattern": "ftir_*.npy"},
        {"data_path": up3, "task": "Fit the XRD series", "label": "xrd",
         "pattern": "xrd_*.npy"},
    ]))
    print("12) per-branch file sets (Fix 3):")
    check("both pattern branches ran", out.get("branches_run") == 2)
    gp = GATE_PROMPTS[-1]
    check("gate saw per-branch file counts", '"n_files": 3' in gp and '"n_files": 2' in gp)
    check("gate probed a branch file, not the directory", "ftir_00.npy" in gp)
    mesh = " || ".join(CAPTURED_TASKS)
    check("companions are files, not the bare directory",
          "ftir_00.npy" in mesh and "xrd_00.npy" in mesh
          and f"auxiliary_data: {up3}  " not in mesh)
    # The branch's PRIMARY data_path must be the GLOB (run_analysis expands it
    # to just this branch's files); handing it the bare directory is what made
    # a branch swallow its sibling series.
    check("branch primary data_path is the glob, not the directory",
          f"PRIMARY dataset for THIS analysis: {os.path.join(up3, 'ftir_*.npy')}" in mesh
          and f"PRIMARY dataset for THIS analysis: {os.path.join(up3, 'xrd_*.npy')}" in mesh
          and f"PRIMARY dataset for THIS analysis: {up3} " not in mesh)
    check("mesh task warns against replacing the pattern with its parent dir",
          "do NOT replace it with the parent directory" in mesh)

    # 13) Same (data_path, task) but DIFFERENT patterns are distinct branches,
    #     not duplicates.
    ag, A, B, C = _agent()
    BEHAVIORS.clear()
    _install_fake_gate([f"{up3}#1", f"{up3}#2"]); ag._complementarity_cache.clear()
    out = json.loads(ag._run_fanout([
        {"data_path": up3, "task": "Fit the series", "label": "s1",
         "pattern": "ftir_*.npy"},
        {"data_path": up3, "task": "Fit the series", "label": "s2",
         "pattern": "xrd_*.npy"},
    ]))
    print("13) pattern-aware dedup:")
    check("distinct patterns kept as 2 branches", out.get("branches_run") == 2)

    print("\n" + "=" * 50)
    npass = sum(results.values())
    print(f"ROBUSTNESS: {npass}/{len(results)} checks passed")
    for k, v in results.items():
        if not v:
            print("  FAILED:", k)
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
