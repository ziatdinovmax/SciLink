"""Live validation of the fan-out wall-clock budget (#358).

Three branches: two well-posed synthetic series, plus a PURE-NOISE
hyperspectral cube whose task demands per-pixel two-peak maps — a fit the
QC machinery must keep refusing (there are no peaks), i.e. a branch
engineered to churn. With a tight branch budget the fan-out must complete
bounded: the churning branch abandoned and recorded degraded, fusion
running over the two productive branches, the timed-out branch excluded.

  export AWS_BEARER_TOKEN_BEDROCK=...  AWS_REGION_NAME=us-east-1
  UNSAFE_EXECUTION_OK=true python tests/test_wallclock_budget_live.py
"""
import json
import os
import sys
import tempfile
import time

import numpy as np

MODEL = os.environ.get("SCILINK_TEST_MODEL",
                       "bedrock/us.anthropic.claude-opus-4-8")
BRANCH_BUDGET_S = float(os.environ.get("BRANCH_BUDGET_S", "600"))
RNG = np.random.default_rng(21)

checks = {}


def check(name, cond):
    checks[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def _sig(t, t0, w):
    return 1.0 / (1.0 + np.exp(-(t - t0) / w))


def _g(x, c, s):
    return np.exp(-0.5 * ((x - c) / s) ** 2)


def _series(d, stem, t0, span, centers):
    x = np.linspace(*span, 800)
    for T in range(30, 165, 10):
        f = _sig(T, t0, 4.0)
        y = ((1 - f) * _g(x, centers[0], (span[1] - span[0]) / 30)
             + f * _g(x, centers[1], (span[1] - span[0]) / 30)
             + RNG.normal(0, 0.004, x.size))
        p = os.path.join(d, f"{stem}_{T:03d}C.txt")
        np.savetxt(p, np.column_stack([x, y]))
        json.dump({"temperature_C": float(T)},
                  open(p.replace(".txt", ".json"), "w"))
    return f"{stem}_*C.txt"


def main():
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        print("Set AWS_BEARER_TOKEN_BEDROCK (+ AWS_REGION_NAME)."); sys.exit(2)
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent, MetaMode)
    base = tempfile.mkdtemp(prefix="budget_live_")
    print(f"session: {base} | branch budget: {BRANCH_BUDGET_S:.0f}s")
    ag = MetaOrchestratorAgent(base_dir=base, api_key=None, base_url=None,
                               model_name=MODEL, meta_mode=MetaMode.AUTONOMOUS)

    d = tempfile.mkdtemp(prefix="budget_data_")
    subject = ("ONE synthetic benchmark pellet heated in-situ, measured by "
               "multiple techniques in one experiment")
    pat_a = _series(d, "bA_vib", 85.0, (2500, 4000), (3400, 2900))
    pat_b = _series(d, "bB_xrd", 88.0, (5, 40), (11.0, 11.9))
    # Pure-noise cube: no peaks exist, so per-pixel two-peak required
    # outputs cannot pass QC — the branch is designed to churn.
    cube = RNG.normal(100, 5, (96, 96, 180)).astype(np.float32)
    cp = os.path.join(d, "noise_cube.npy")
    np.save(cp, cube)
    json.dump({"technique": "hyperspectral emission map",
               "wavelengths_nm": list(np.linspace(400, 900, 180)),
               "note": "acquired on the same pellet in the same session"},
              open(cp.replace(".npy", ".json"), "w"))

    t0 = time.monotonic()
    out = json.loads(ag._run_fanout([
        {"data_path": d, "pattern": pat_a, "label": "vibrational series",
         "task": f"Analyze the series in {d} — ONLY files matching "
                 f"'{pat_a}'. Temperature series on {subject}. Characterize "
                 "the transition and report its temperature."},
        {"data_path": d, "pattern": pat_b, "label": "diffraction series",
         "task": f"Analyze the series in {d} — ONLY files matching "
                 f"'{pat_b}'. Temperature series on {subject}. Characterize "
                 "the structural transition and report its temperature."},
        {"data_path": cp, "label": "emission map",
         "task": f"Analyze the hyperspectral image at {cp} (96x96x180, "
                 f"wavelengths in the metadata JSON) of {subject}. Extract "
                 "PER-PIXEL maps of the two emission peaks: you MUST report "
                 "Peak1_Position, Peak1_FWHM, Peak2_Position, Peak2_FWHM "
                 "and the peak ratio at every pixel."},
    ], branch_time_budget_s=BRANCH_BUDGET_S))
    dt = time.monotonic() - t0
    print("FANOUT:", json.dumps({k: out.get(k) for k in (
        "status", "branches_run", "branches_with_output",
        "branches_timed_out")}, indent=2))
    print(f"fan-out wall time: {dt / 60:.1f} min")
    check("fan-out ran to completion", out.get("status") == "success")
    check("fan-out bounded (well under unbounded-churn territory)",
          dt < BRANCH_BUDGET_S + 900)
    fan = [e for e in ag._delegation_ledger if e.get("fanout")]
    noisy = [e for e in fan if e.get("label") == "emission map"]
    productive_n = out.get("branches_with_output")
    if noisy and noisy[0].get("timed_out"):
        print("(churning branch was abandoned on the budget)")
        check("timed-out branch degraded with the budget message",
              noisy[0]["status"] == "error"
              and "wall-clock budget" in (noisy[0].get("error") or ""))
        check("timed-out counted in the result",
              out.get("branches_timed_out") == 1)
    else:
        # The branch may instead FINISH honestly (QC refusing + salvage) —
        # also a valid bounded outcome; the budget just wasn't needed.
        print("(churning branch finished within the budget on its own — "
              "loop budgets/QC honesty bounded it first)")
        check("churning branch outcome is honest (no fabricated peaks)",
              noisy and (noisy[0].get("status") != "success"
                         or "peak" not in " ".join(
                             str(k) for k in
                             (noisy[0].get("key_findings") or [])).lower()
                         or any(w in (noisy[0].get("summary") or "").lower()
                                for w in ("no ", "not ", "noise", "absen",
                                          "fail", "lack", "flat"))))
        check("no timeout needed", "branches_timed_out" not in out)

    check("both good branches productive", productive_n >= 2)
    idxs = [r["delegation_index"] for r in out.get("results", [])
            if r.get("produced_output")]
    if len(idxs) >= 2:
        fused = json.loads(ag._fuse_delegations(
            idxs, focus="Do the productive techniques agree on the "
                        "transition temperature?"))
        check("fusion over the productive branches succeeded",
              fused.get("status") == "success")
        if noisy and noisy[0].get("timed_out"):
            check("abandoned branch excluded from fusion",
                  noisy[0]["index"] not in fused.get("fused_from", []))
    else:
        check("fusion over the productive branches succeeded", False)

    print("\n" + "=" * 60)
    npass = sum(checks.values())
    print(f"WALL-CLOCK BUDGET LIVE: {npass}/{len(checks)} checks passed")
    for k, v in checks.items():
        if not v:
            print("  FAILED:", k)
    sys.exit(0 if npass == len(checks) else 1)


if __name__ == "__main__":
    main()
