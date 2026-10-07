"""Live DFT value-ACCURACY test (real VASP, run in a SLURM allocation).

Where ``test_dft_convergence_live.py`` only checks that each ladder rung yields a
readable energy, this goes the whole way the paper's accuracy claim needs: it
converges ENCUT + k-points with the convergence machinery, adopts the converged
settings, runs ONE cell relaxation at those settings, and scores the relaxed
lattice constant against a reference within 2.5% (the `test_dft` band). Passing
shows the converged parameters actually produce a physically correct value — not
just that the sweep plateaus.

Run inside a compute-node allocation (VASP runs in-process via LocalExecutor):

    export VASP_PP_PATH=/path/to/potpaw_PBE     # per-element POTCAR root
    export VASP_RUN_CMD="mpirun vasp_std"       # + module loads (see the sbatch)
    python tests/test_dft_accuracy_live.py --system si

`--system si` (diamond semiconductor, a = 5.431 Å) or `--system cu` (FCC metal,
a = 3.615 Å). References are the experimental cubic lattice constants; PBE sits
well inside 2.5% of each.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RUN_ROOT = REPO_ROOT / "tests" / "_dft_accuracy_runs"

# Experimental conventional-cubic lattice constants (Å) and the scoring band.
# Cu: 3.6149 Å (CODATA/Cu fcc, 298 K); Si: 5.4310 Å (Si diamond, 298 K).
_REFERENCE_A = {"cu": 3.6149, "si": 5.4310}
_TOL_FRAC = 0.025  # within 2.5%, matching test_dft's lattice-constant tolerance

# Per-system smearing for the static (convergence) and relax (production) decks.
_SMEAR = {"cu": "ISMEAR = 1\nSIGMA = 0.2\n", "si": "ISMEAR = 0\nSIGMA = 0.05\n"}

# Static SCF deck for the convergence sweep (energy/atom observable). KSPACING in
# the INCAR, no KPOINTS file, so the k-point ladder can vary it.
def _static_incar(system: str) -> str:
    return (f"SYSTEM = {system} convergence\nPREC = Accurate\nENCUT = 400\n"
            f"{_SMEAR[system]}EDIFF = 1E-6\nLREAL = .FALSE.\n"
            "IBRION = -1\nNSW = 0\nLWAVE = .FALSE.\nLCHARG = .FALSE.\n"
            "KSPACING = 0.3\n")


# Production relaxation deck at the ADOPTED ENCUT/KSPACING (full cell relax).
def _relax_incar(system: str, encut, kspacing) -> str:
    return (f"SYSTEM = {system} accuracy relax\nPREC = Accurate\nENCUT = {encut}\n"
            f"{_SMEAR[system]}EDIFF = 1E-6\nEDIFFG = -0.01\nLREAL = .FALSE.\n"
            "IBRION = 2\nISIF = 3\nNSW = 80\nLWAVE = .FALSE.\nLCHARG = .FALSE.\n"
            f"KSPACING = {kspacing}\n")


def _build_conventional(system: str, poscar_path: Path):
    """Write the CONVENTIONAL cubic cell so lattice_constant_a (= abc[0]) is the
    conventional a, not a primitive-cell edge."""
    from ase.build import bulk
    from ase.io import write
    a0 = _REFERENCE_A[system]
    kind = "fcc" if system == "cu" else "diamond"
    el = "Cu" if system == "cu" else "Si"
    atoms = bulk(el, kind, a=a0, cubic=True)
    write(str(poscar_path), atoms, format="vasp", sort=True, direct=True)


def _species_from_poscar(poscar_path: Path):
    return poscar_path.read_text().splitlines()[5].split()


def _assemble_potcar(species, pp_root: Path) -> str:
    parts = []
    for el in species:
        p = pp_root / el / "POTCAR"
        if not p.is_file():
            raise SystemExit(f"missing POTCAR for {el} at {p} "
                             f"(is VASP_PP_PATH the per-element root?)")
        parts.append(p.read_text())
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["cu", "si"], default="si")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    pp_root = os.environ.get("VASP_PP_PATH")
    run_cmd = os.environ.get("VASP_RUN_CMD", "mpirun vasp_std")
    if not pp_root:
        raise SystemExit("set VASP_PP_PATH to the per-element POTCAR root")
    pp_root = Path(pp_root)

    work = RUN_ROOT / args.system
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    poscar = work / "POSCAR"
    _build_conventional(args.system, poscar)
    species = _species_from_poscar(poscar)
    print(f"system={args.system}  species={species}  run_cmd={run_cmd!r}", flush=True)

    base_inputs = {
        "INCAR": _static_incar(args.system),
        "POSCAR": poscar.read_text(),
        "POTCAR": _assemble_potcar(species, pp_root),
    }

    from scilink.skills.loader import load_skill
    from scilink.skills._shared._registry import get_tool_function
    from scilink.agents.sim_agents.convergence import converge_parameters
    from scilink.agents.sim_agents.refinement import LocalExecutor

    specs = load_skill("vasp", domain="periodic_dft")["meta"]["convergence"]
    print("convergence specs:", json.dumps(specs), flush=True)

    set_param = get_tool_function("set_convergence_param", active_skills=["vasp"])
    get_param = get_tool_function("get_convergence_param", active_skills=["vasp"])
    read_obs = get_tool_function("read_convergence_observable", active_skills=["vasp"])
    executor = LocalExecutor(timeout=args.timeout)

    def run_ladder(param, members):
        dirs = {}
        root = work / "convergence" / str(param)
        for setting, inputs in members.items():
            rdir = root / str(setting)
            rdir.mkdir(parents=True, exist_ok=True)
            print(f"  [{param}={setting}] VASP in {rdir} ...", flush=True)
            res = executor.run(inputs, run_cmd, str(rdir))
            print(f"    -> {res.get('status')} rc={res.get('returncode')}", flush=True)
            dirs[setting] = str(rdir)
        return dirs

    # ── Step 1: converge ENCUT + k-points (energy/atom) ──
    pc = converge_parameters(
        base_inputs=base_inputs, specs=specs,
        set_param=lambda i, p, v: set_param(input_files=i, param=p, value=v),
        read_observable=lambda d, o: read_obs(output_dir=d, observable=o),
        run_ladder=run_ladder,
        get_param=lambda i, p: get_param(input_files=i, param=p),
    )
    print("\nconverged:", pc.all_converged,
          "| floored:", json.dumps(pc.floors), flush=True)

    # Adopted settings (fall back to the base deck's values if a sweep didn't
    # plateau — converge_parameters leaves those unchanged).
    adopted_encut = get_param(input_files=pc.final_inputs, param="ENCUT") or 400
    adopted_kspacing = get_param(input_files=pc.final_inputs, param="k-points") or 0.3
    print(f"adopted: ENCUT={adopted_encut} eV  KSPACING={adopted_kspacing} /Å",
          flush=True)

    # ── Step 2: ONE relaxation at the adopted settings ──
    prod = work / "production"
    prod.mkdir(parents=True, exist_ok=True)
    prod_inputs = {
        "INCAR": _relax_incar(args.system, adopted_encut, adopted_kspacing),
        "POSCAR": base_inputs["POSCAR"],
        "POTCAR": base_inputs["POTCAR"],
    }
    print(f"\nrelaxing at adopted settings in {prod} ...", flush=True)
    res = executor.run(prod_inputs, run_cmd, str(prod))
    print(f"  -> {res.get('status')} rc={res.get('returncode')}", flush=True)

    # ── Step 3: score the relaxed lattice constant vs reference ──
    a_computed = read_obs(output_dir=str(prod), observable="lattice_constant_a")
    a_ref = _REFERENCE_A[args.system]

    print("\n" + "=" * 60 + "\nVALUE-ACCURACY\n" + "=" * 60, flush=True)
    if a_computed is None:
        print("FAIL: relaxation produced no readable converged lattice constant "
              "(check the production run's ionic convergence)")
        sys.exit(1)
    dev = abs(a_computed - a_ref) / a_ref
    print(f"  computed a = {a_computed:.4f} Å")
    print(f"  reference  = {a_ref:.4f} Å  (experiment)")
    print(f"  deviation  = {dev*100:.2f} %   (band: {_TOL_FRAC*100:.1f} %)")
    ok = dev <= _TOL_FRAC
    print("\n" + ("PASS: within the accuracy band" if ok
                  else "FAIL: outside the accuracy band"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
