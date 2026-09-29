"""
Live tests for the observable-convergence loop (#447).

Scenario A — proactive (LLM only, no LAMMPS):
    A synthetic, completed LAMMPS run whose stress autocorrelation decays far
    slower than the run length, so the Green-Kubo integral cannot plateau.
    The prompt asks for the viscosity and whether it is trustworthy, WITHOUT
    naming check_observable_convergence. Passes if the orchestrator calls the
    tool on its own; the tool's verdict is reported but not asserted.

Scenario B — loop (LLM + LAMMPS, run inside a SLURM allocation):
    A real LJ-argon Green-Kubo run with a deliberately short first production.
    Passes if the orchestrator runs, detects non-convergence, re-runs with more
    sampling, and re-checks — i.e. run_simulation and
    check_observable_convergence are each called at least twice, with a re-run
    following an unconverged verdict.

Run:
    python tests/test_convergence_live.py --scenario proactive
    python tests/test_convergence_live.py --scenario loop      # needs `lmp`
    python tests/test_convergence_live.py --scenario all

Env: SCILINK_API_KEY or ANTHROPIC_API_KEY; optional SCILINK_MODEL,
SCILINK_BASE_URL.
"""

import argparse
import json
import math
import os
import random
import shutil
import sys
import textwrap
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = "claude-opus-4-6"
RUN_ROOT = REPO_ROOT / "tests" / "_simulate_runs" / "convergence_live"


def _make_orch(base_dir: Path):
    from scilink.agents.sim_agents import (
        SimulationOrchestratorAgent, SimulationMode,
    )
    key = os.environ.get("SCILINK_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("Set SCILINK_API_KEY or ANTHROPIC_API_KEY.")
    kwargs = dict(
        base_dir=str(base_dir),
        api_key=key,
        model_name=os.environ.get("SCILINK_MODEL", DEFAULT_MODEL),
        simulation_mode=SimulationMode.AUTONOMOUS,
    )
    if os.environ.get("SCILINK_BASE_URL"):
        kwargs["base_url"] = os.environ["SCILINK_BASE_URL"]
    return SimulationOrchestratorAgent(**kwargs)


def _track(orch, names):
    """Wrap tools so every call is logged in order: [(name, kwargs, result)]."""
    log = []
    for name in names:
        fn = orch.tools.functions_map[name]

        def wrapper(_fn=fn, _name=name, **kwargs):
            t0 = time.time()
            print(f"\n>>> [{_name}] called: {json.dumps(kwargs, default=str)[:400]}")
            result = _fn(**kwargs)
            print(f">>> [{_name}] returned in {time.time() - t0:.0f}s: "
                  f"{str(result)[:600]}")
            parsed = _parse(result)
            for key in ("force_field", "error", "message"):
                if key in parsed and str(parsed.get("status", "")).startswith(
                        ("fail", "error")):
                    print(f">>> [{_name}] {key}: "
                          f"{json.dumps(parsed[key], default=str)[:1500]}")
            log.append((_name, kwargs, result))
            return result

        orch.tools.functions_map[name] = wrapper
    return log


def _parse(result):
    try:
        return json.loads(result)
    except (TypeError, ValueError):
        return {}


# ---------------------------------------------------------------------------
# Scenario A — proactive
# ---------------------------------------------------------------------------

def _write_unconverged_log(run_dir: Path) -> None:
    """LAMMPS `real`-units log whose stress ACF cannot decay within the run.

    Off-diagonal stresses follow an AR(1) process with correlation time ~0.6x
    the series length, so the stress ACF has not decayed within the usable lag
    window. White noise would be wrong here: its ACF decays in one lag and the
    integral plateaus at once.
    """
    random.seed(0)
    run_dir.mkdir(parents=True, exist_ok=True)
    thermo_every, n_steps = 10, 20000
    n = n_steps // thermo_every + 1
    phi = math.exp(-1.0 / (0.6 * n))
    sigma = 300.0
    innov = sigma * math.sqrt(1 - phi * phi)

    comps = {k: random.gauss(0, sigma) for k in ("pxy", "pxz", "pyz", "dxx", "dyy")}
    rows = ["Step Temp Press Pxx Pyy Pzz Pxy Pxz Pyz Volume"]
    for i in range(n):
        for k in comps:
            comps[k] = phi * comps[k] + random.gauss(0, innov)
        p0 = random.gauss(1.0, 50.0)
        pxx, pyy = p0 + comps["dxx"], p0 + comps["dyy"]
        pzz = 3 * p0 - pxx - pyy
        rows.append(
            f"{i * thermo_every} {random.gauss(94.4, 1.5):.2f} {p0:.3f} "
            f"{pxx:.3f} {pyy:.3f} {pzz:.3f} {comps['pxy']:.3f} "
            f"{comps['pxz']:.3f} {comps['pyz']:.3f} 17240.0"
        )

    deck = textwrap.dedent(f"""\
        units real
        atom_style atomic
        read_data argon.data
        pair_style lj/cut 10.0
        pair_coeff 1 1 0.2381 3.405
        timestep 2.0
        fix 1 all nvt temp 94.4 94.4 200.0
        thermo_style custom step temp press pxx pyy pzz pxy pxz pyz vol
        thermo {thermo_every}
        run {n_steps}
    """)
    (run_dir / "in.lammps").write_text(deck)
    (run_dir / "log.lammps").write_text(
        "LAMMPS (2 Aug 2023)\n" + deck + "\n".join(rows)
        + "\nLoop time of 42.0 on 4 procs for 20000 steps with 256 atoms\n"
        + "Total wall time: 0:00:42\n"
    )


def scenario_proactive() -> bool:
    work = RUN_ROOT / "proactive"
    shutil.rmtree(work, ignore_errors=True)
    run_dir = work / "lammps_output"
    _write_unconverged_log(run_dir)

    orch = _make_orch(work / "sim")
    assert "check_observable_convergence" in orch.tools.functions_map
    log = _track(orch, ["analyze_output", "check_observable_convergence",
                        "run_simulation"])
    # Scenario A tests the decision, not a re-run: stub execution out.
    orch.tools.functions_map["run_simulation"] = lambda **kw: json.dumps({
        "status": "error",
        "message": "Execution is disabled in this test session; report "
                   "what you would do instead.",
    })

    response = orch.chat(
        f"I finished an NVT production run of liquid argon (256 atoms, 94.4 K, "
        f"LAMMPS real units) and the outputs are in {run_dir}. Compute the "
        f"shear viscosity from it via Green-Kubo and tell me whether the value "
        f"can be trusted for publication."
    )
    print("\n--- final response (last 1200 chars) ---\n" + response[-1200:])

    conv_calls = [c for c in log if c[0] == "check_observable_convergence"]
    verdicts = [_parse(c[2]) for c in conv_calls]
    reported_unconverged = any(v.get("converged") is False for v in verdicts)
    text = response.lower()
    flags_problem = any(s in text for s in (
        "not converged", "unconverged", "has not converged", "did not converge",
        "no plateau", "not plateau", "plateau_reached", "not trustworthy",
        "cannot be trusted", "not reliable", "extend"))

    print("\n=== Scenario A results ===")
    print(f"  tool call order            : {[c[0] for c in log]}")
    print(f"  convergence tool called    : {bool(conv_calls)}")
    print(f"  tool verdict unconverged   : {reported_unconverged}")
    print(f"  response flags the problem : {flags_problem}")

    # Pass = the orchestrator reached for the tool unprompted. The verdict
    # itself comes from an LLM-written plateau detector on synthetic data, so
    # it is reported, not asserted; scenario B tests acting on a verdict.
    ok = bool(conv_calls)
    if conv_calls and not reported_unconverged:
        print("  NOTE: tool ran but did not report non-convergence — the "
              "analysis script judged the synthetic integral as plateaued.")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Scenario B — loop with real LAMMPS
# ---------------------------------------------------------------------------

def scenario_loop() -> bool:
    if not shutil.which("lmp") and not shutil.which("lmp_serial") \
            and not shutil.which("lmp_mpi"):
        print("SKIP: no LAMMPS binary on PATH (lmp / lmp_serial / lmp_mpi).")
        return False
    if not (os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")):
        print("WARN: not inside a SLURM allocation; run_simulation will only "
              "execute locally if no HPC connection is configured.")

    work = RUN_ROOT / "loop"
    shutil.rmtree(work, ignore_errors=True)
    orch = _make_orch(work / "sim")
    # Two+ runs, each with analysis and checks, exceed the default tool cap.
    orch.max_iterations = max(orch.max_iterations, 60)
    log = _track(orch, ["run_simulation", "analyze_output",
                        "check_observable_convergence"])

    # Water, not LJ argon: the MD pipeline parameterizes liquids through
    # OpenFF, which has no parameters for a bare noble-gas atom.
    response = orch.chat(
        "Compute the shear viscosity of liquid water (TIP3P) at 298 K and "
        "1 atm via Green-Kubo in LAMMPS, using a small box of about 216 "
        "water molecules. To save compute, start with a short NVT "
        "production of only 20 ps after equilibration, logging the "
        "pressure tensor every few steps. I need a converged value for a "
        "paper, so don't report a number you can't stand behind."
    )
    print("\n--- final response (last 1500 chars) ---\n" + response[-1500:])

    order = [c[0] for c in log]
    runs = [i for i, c in enumerate(log) if c[0] == "run_simulation"]
    checks = [(i, _parse(c[2])) for i, c in enumerate(log)
              if c[0] == "check_observable_convergence"]
    unconverged_idx = [i for i, v in checks if v.get("converged") is False]
    rerun_after_unconverged = any(r > u for u in unconverged_idx for r in runs)
    recheck_after_rerun = any(
        ci > r for r in runs for u in unconverged_idx if r > u
        for ci, _ in checks)
    final = checks[-1][1] if checks else {}

    print("\n=== Scenario B results ===")
    print(f"  tool call order                 : {order}")
    print(f"  run_simulation calls            : {len(runs)}")
    print(f"  convergence checks              : {len(checks)}")
    print(f"  saw an unconverged verdict      : {bool(unconverged_idx)}")
    print(f"  re-ran after unconverged verdict: {rerun_after_unconverged}")
    print(f"  re-checked after the re-run     : {recheck_after_rerun}")
    print(f"  final verdict converged         : {final.get('converged')}")
    for i, v in checks:
        for prop, p in (v.get("properties") or {}).items():
            print(f"    check@{i}: {prop} = {p.get('value')} {p.get('units')} "
                  f"(state={p.get('state')})")

    if not unconverged_idx and checks:
        print("  NOTE: the first run already converged — the loop was not "
              "exercised. Shorten the initial production and re-run.")
    ok = bool(checks) and rerun_after_unconverged and recheck_after_rerun
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["proactive", "loop", "all"],
                    default="proactive")
    args = ap.parse_args()

    results = {}
    if args.scenario in ("proactive", "all"):
        results["proactive"] = scenario_proactive()
    if args.scenario in ("loop", "all"):
        results["loop"] = scenario_loop()

    print("\n" + "=" * 60)
    for k, v in results.items():
        print(f"  {k:10s}: {'PASS' if v else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)
