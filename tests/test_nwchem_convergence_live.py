"""Live NWChem basis-convergence test (real NWChem, run in a SLURM allocation).

This is the run that was all-`None` on the pre-fix branch: a real def2 basis
sweep where NWChem completed (rc=0) every rung but the observable extractor read
nothing, so the sweep could not converge. It validates, on real output, that
`read_convergence_observable` now finds the energy in `run_stdout.log` and the
comparator reaches a plateau.

NWChem is MPI-linked and needs its module + basis-library env, so execution goes
through a small wrapper (see the sbatch); the deck is written as `job.nw` to
match it. Env-driven, no cluster specifics in this file:

    export NWCHEM_RUN_CMD="bash --login /path/to/run_nwchem.sh"
    python tests/test_nwchem_convergence_live.py

Passes if every rung yields a readable energy AND the sweep reports a plateau.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RUN_ROOT = REPO_ROOT / "tests" / "_nwchem_convergence_runs"

# Water, B3LYP — matches the DFT basis sweep that failed to extract before. The
# deck MUST be named job.nw (the wrapper runs `nwchem job.nw` in each run dir),
# and the orbital basis `* library <x>` line is what the sweep rewrites.
_DECK = (
    "echo\n"
    "start h2o_basis\n"
    "geometry units angstrom\n"
    "  O  0.000  0.000  0.119\n"
    "  H  0.000  0.757 -0.477\n"
    "  H  0.000 -0.757 -0.477\n"
    "end\n"
    "basis\n"
    "  * library def2-svp\n"
    "end\n"
    "dft\n"
    "  xc b3lyp\n"
    "end\n"
    "task dft energy\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    run_cmd = os.environ.get("NWCHEM_RUN_CMD")
    if not run_cmd:
        raise SystemExit("set NWCHEM_RUN_CMD (e.g. 'bash --login /path/run_nwchem.sh')")

    work = RUN_ROOT
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    base_inputs = {"job.nw": _DECK}

    from scilink.skills.loader import load_skill
    from scilink.skills._shared._registry import get_tool_function
    from scilink.agents.sim_agents.convergence import converge_parameters
    from scilink.agents.sim_agents.refinement import LocalExecutor

    specs = load_skill("nwchem", domain="molecular_qc")["meta"]["convergence"]
    print("convergence specs:", json.dumps(specs), flush=True)

    set_param = get_tool_function("set_convergence_param", active_skills=["nwchem"])
    read_obs = get_tool_function("read_convergence_observable", active_skills=["nwchem"])
    executor = LocalExecutor(timeout=args.timeout)

    def run_ladder(param, members):
        dirs = {}
        root = work / "convergence" / str(param)
        for setting, inputs in members.items():
            rdir = root / str(setting)
            rdir.mkdir(parents=True, exist_ok=True)
            print(f"  [{param}={setting}] NWChem in {rdir} ...", flush=True)
            res = executor.run(inputs, run_cmd, str(rdir))
            print(f"    -> {res.get('status')} rc={res.get('returncode')}", flush=True)
            dirs[setting] = str(rdir)
        return dirs

    pc = converge_parameters(
        base_inputs=base_inputs, specs=specs,
        set_param=lambda i, p, v: set_param(input_files=i, param=p, value=v),
        read_observable=lambda d, o: read_obs(output_dir=d, observable=o),
        run_ladder=run_ladder,
    )

    print("\n" + "=" * 60 + "\nRESULTS\n" + "=" * 60, flush=True)
    print(f"all_converged: {pc.all_converged}")
    all_readable = True
    for s in pc.sweeps:
        print(f"\n{s.param_name}:")
        for setting, value in s.observations:
            print(f"  {setting:>12}: {value}")
            if value is None:
                all_readable = False
        c = s.convergence
        print(f"  -> converged={c.converged} setting={c.setting} value={c.value}"
              f"\n     {c.reason}")

    # The pre-fix failure was every rung reading None despite rc=0. Require both:
    # the extractor finds every energy, and the comparator reaches a plateau.
    ok = all_readable and pc.all_converged
    if not all_readable:
        print("\nFAIL: a rung produced no readable energy (extraction broken)")
    elif not pc.all_converged:
        print("\nFAIL: energies read, but no plateau — widen the ladder/tolerance")
    else:
        print("\nPASS: every rung readable and the basis sweep converged")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
