"""
MLIPAgent deploy-pretrained example.

Runs the same code path twice (Cu bulk and MoS2) and writes everything
into a timestamped subdirectory so reruns don't clobber each other.

This script gracefully degrades:
  - With MACE installed (cluster): runs the full deploy_pretrained
    pipeline, including model selection and weight download/load.
  - Without MACE (local dev): exercises agent assembly, skill loading,
    and direct LAMMPS input generation with a placeholder model path.

Both paths produce a runnable LAMMPS input file you can inspect.

Usage:
  python examples/mlip/deploy_example.py
  python examples/mlip/deploy_example.py --systems cu mos2
  python examples/mlip/deploy_example.py --temperature 600 --pressure 1.0
"""

import argparse
import datetime
import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk, fcc111, molecule
from ase.io.lammpsdata import write_lammps_data


def cu_bulk_system():
    atoms = bulk("Cu", "fcc", a=3.615, cubic=True)
    return {
        "name": "cu_bulk",
        "atoms": atoms,
        "elements": ["Cu"],
        "research_goal": (
            "Equilibrate FCC Cu at 300 K and report a stable lattice "
            "constant for downstream surface-energy work."
        ),
    }


def mos2_monolayer():
    """A 2H-MoS2 monolayer with vacuum along z."""
    a = 3.16
    c_vac = 20.0
    cell = np.array([
        [a, 0.0, 0.0],
        [-a / 2, a * np.sqrt(3) / 2, 0.0],
        [0.0, 0.0, c_vac],
    ])
    positions = [
        (0.0, 0.0, c_vac / 2),
        (a / 2, a * np.sqrt(3) / 6, c_vac / 2 + 1.59),
        (a / 2, a * np.sqrt(3) / 6, c_vac / 2 - 1.59),
    ]
    atoms = Atoms("MoS2", positions=positions, cell=cell, pbc=[True, True, True])
    return {
        "name": "mos2_monolayer",
        "atoms": atoms,
        "elements": ["Mo", "S"],
        "research_goal": (
            "Verify mace-mp-0 reproduces the MoS2 monolayer lattice "
            "constant within 2% at room temperature."
        ),
    }


def cu_co_slab():
    """Cu(111) 2x2x3 slab with one CO molecule atop a surface Cu atom.

    Classic heterogeneous catalysis benchmark; water-gas-shift relevant.
    Expected behavior: CO stays adsorbed atop, Cu-C ~1.85 A, C-O ~1.15 A.
    """
    slab = fcc111("Cu", size=(2, 2, 3), a=3.615, vacuum=8.0)
    top_z = slab.positions[:, 2].max()
    top_atom = slab.positions[slab.positions[:, 2].argmax()]
    co = Atoms(
        "CO",
        positions=[
            [top_atom[0], top_atom[1], top_z + 1.85],
            [top_atom[0], top_atom[1], top_z + 1.85 + 1.15],
        ],
    )
    slab += co
    return {
        "name": "cu_co_slab",
        "atoms": slab,
        "elements": sorted(set(slab.get_chemical_symbols())),
        "research_goal": (
            "Equilibrate CO on Cu(111) at 300 K and verify atop binding "
            "geometry consistent with water-gas-shift catalysis."
        ),
    }


def licoo2_bulk():
    """LiCoO2 in the R-3m (alpha-NaFeO2) structure -- canonical Li-ion cathode."""
    from ase.spacegroup import crystal
    licoo2 = crystal(
        ["Li", "Co", "O"],
        basis=[(0, 0, 0), (0, 0, 0.5), (0, 0, 0.2604)],
        spacegroup=166,
        cellpar=[2.815, 2.815, 14.05, 90, 90, 120],
    )
    return {
        "name": "licoo2_bulk",
        "atoms": licoo2,
        "elements": sorted(set(licoo2.get_chemical_symbols())),
        "research_goal": (
            "Equilibrate the R-3m LiCoO2 cathode at 300 K and verify the "
            "c/a ratio remains within 1% of the experimental value (~4.99)."
        ),
    }


def mos2_vacancy():
    """2x2 2H-MoS2 monolayer with one S atom removed (single S vacancy)."""
    base = mos2_monolayer()["atoms"]
    super_cell = base * (2, 2, 1)
    s_indices = [i for i, s in enumerate(super_cell.get_chemical_symbols())
                 if s == "S"]
    del super_cell[s_indices[0]]
    return {
        "name": "mos2_vacancy",
        "atoms": super_cell,
        "elements": sorted(set(super_cell.get_chemical_symbols())),
        "research_goal": (
            "Equilibrate a S-vacancy defect in MoS2 monolayer at 300 K "
            "and characterize local distortion around the vacancy site."
        ),
    }


def methanol_cluster():
    """A small cluster of 5 methanol molecules in a cubic box.

    Battery electrolyte / biomass-conversion solvent. All-organic
    composition (C/H/O) routes the agent to mace-off23.
    """
    n_mols = 5
    box = 10.0
    rng = np.random.default_rng(0)
    template = molecule("CH3OH")
    cluster = Atoms(cell=[box, box, box], pbc=[True, True, True])
    for _ in range(n_mols):
        m = template.copy()
        m.translate(rng.uniform(2.0, box - 2.0, size=3) - m.get_center_of_mass())
        cluster += m
    return {
        "name": "methanol_cluster",
        "atoms": cluster,
        "elements": sorted(set(cluster.get_chemical_symbols())),
        "research_goal": (
            "Equilibrate liquid methanol at 300 K and verify hydrogen-bonded "
            "network forms with expected O-H...O distances near 1.9 A."
        ),
    }


def alanine_dipeptide():
    """N-acetyl-L-alanine-N'-methylamide (Ac-Ala-NHMe) -- 22 atoms.

    Textbook Ramachandran / peptide-MD benchmark molecule. All-organic
    (C/H/N/O) routes the agent to mace-off23.

    Coordinates are an approximate alpha-helix-like conformation
    (phi~-60, psi~-45); mace-off23 will relax to the local minimum
    within ~50-100 fs of MD.
    """
    symbols = "CHHHCONHCHCHHHCONHCHHH"
    positions = np.array([
        [-3.00, 1.50,  0.00],   # acetyl methyl C
        [-3.40, 1.00,  0.90],   #   H
        [-3.40, 2.50,  0.10],   #   H
        [-3.40, 1.00, -0.90],   #   H
        [-1.50, 1.50,  0.00],   # acetyl carbonyl C
        [-0.90, 2.50,  0.00],   #   =O
        [-0.70, 0.30,  0.00],   # amide N
        [-1.20,-0.50,  0.00],   #   H
        [ 0.70, 0.30,  0.00],   # alpha C
        [ 1.00, 0.80,  0.90],   #   H
        [ 1.30,-1.10,  0.00],   # methyl side-chain C
        [ 2.40,-1.10,  0.00],   #   H
        [ 1.00,-1.60,  0.90],   #   H
        [ 1.00,-1.60, -0.90],   #   H
        [ 1.20, 1.10, -1.10],   # second carbonyl C
        [ 0.70, 1.30, -2.20],   #   =O
        [ 2.30, 1.90, -0.70],   # second amide N
        [ 2.70, 1.80,  0.20],   #   H
        [ 3.20, 2.90, -1.40],   # NHMe methyl C
        [ 2.70, 3.80, -1.60],   #   H
        [ 4.10, 3.00, -0.90],   #   H
        [ 3.40, 2.60, -2.40],   #   H
    ])
    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.center(vacuum=5.0)
    return {
        "name": "alanine_dipeptide",
        "atoms": atoms,
        "elements": sorted(set(atoms.get_chemical_symbols())),
        "research_goal": (
            "Equilibrate alanine dipeptide at 300 K and characterize "
            "phi/psi backbone dynamics consistent with the conformational "
            "sampling expected for an isolated peptide."
        ),
    }


SYSTEMS = {
    "cu":            cu_bulk_system,
    "mos2":          mos2_monolayer,
    "cu_co":         cu_co_slab,
    "licoo2":        licoo2_bulk,
    "mos2_vac":      mos2_vacancy,
    "meoh":          methanol_cluster,
    "ala":           alanine_dipeptide,
}


def print_section(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def write_data_file(atoms, out_dir):
    data_path = out_dir / "system.data"
    elements = sorted(set(atoms.get_chemical_symbols()))
    write_lammps_data(
        str(data_path), atoms, atom_style="atomic",
        specorder=elements, masses=True,
    )
    return data_path


def report_skill_context(agent):
    print(f"  skill loaded:    {agent.skill_name}")
    if agent.skill_sections is None:
        print("  WARNING: skill_sections is None — agent will get empty LLM context")
        return
    sections = [
        k for k in agent.skill_sections
        if k not in ("name", "meta", "extras") and agent.skill_sections.get(k)
    ]
    print(f"  populated:       {sections}")
    validation = agent.skill_sections.get("validation", "")
    print(f"  validation len:  {len(validation)} chars")


def check_mace_availability():
    from scilink.skills._shared import mlip_tools
    backends = mlip_tools.check_backends()
    mace_info = backends.get("mace", {})
    return mace_info.get("available", False), mace_info


def run_with_mace(
    agent, system_info_dict, research_goal, out_dir, temperature, pressure,
    runner, structure_file, n_steps, device,
    backend=None,
):
    if backend:
        print(f"  Running deploy_pretrained (runner={runner}, "
              f"backend={backend} -- forced, skipping LLM selection).")
    else:
        print(f"  Running deploy_pretrained (runner={runner}, "
              f"backend chosen by LLM).")
    if runner == "lammps":
        sim_params = {
            "timestep": 0.5,            # ps (LAMMPS metal units)
            "temperature": temperature,
            "pressure": pressure,
        }
    else:  # ase
        sim_params = {
            "timestep": 1.0,            # fs (ASE convention)
            "temperature": temperature,
            "pressure": pressure,
            "n_steps": n_steps,
            "device": device,
        }
    # structure_file is required for every runner now — MLIPAgent
    # delegates run generation to the MD agent, which reads it.
    kwargs = dict(runner=runner, structure_file=str(structure_file))
    if backend:
        kwargs["backend"] = backend

    return agent.deploy_pretrained(
        system_info=system_info_dict,
        research_goal=research_goal,
        simulation_params=sim_params,
        **kwargs,
    )


def run_without_mace(
    agent, system, out_dir, temperature, pressure,
    runner, structure_file, n_steps, device,
):
    """
    Demonstrate the orchestration without MACE installed: drive skill
    context lookup, then exercise the engine-neutral runner path with a
    placeholder DeployedPotential. No real calculator is constructed,
    so no MACE needed — but the run-generation path is identical to the
    cluster's (the runners only read the DeployedPotential descriptor).
    """
    print(f"  MACE not installed locally — exercising deterministic")
    print(f"  pieces only (skill context + {runner} run generation).")

    agent._load_backend_skill("mace")
    planning = agent._get_skill_context(section="planning")

    organic_elements = {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"}
    sys_elements = set(system["elements"])
    is_organic_only = sys_elements and sys_elements.issubset(organic_elements)

    if is_organic_only and "mace-off23" in planning:
        chosen = "mace-off23"
        rationale = f"all elements {sorted(sys_elements)} are organic"
    elif "mace-mp-0" in planning:
        chosen = "mace-mp-0"
        rationale = (
            f"system contains inorganic elements ({sorted(sys_elements)}) "
            f"→ mace-mp-0 per MACE skill rules"
        )
    else:
        chosen = "mace-mp-0"
        rationale = "default — MACE backend skill did not load"
    print(f"  pretrained pick: {chosen}")
    print(f"  rationale:       {rationale}")

    # Placeholder potential — same shape mlip_tools.deploy() returns on
    # the cluster, but with a placeholder model path and no calculator
    # constructed. The runners only read this descriptor, so the
    # generated input/script is exactly what the cluster would produce
    # (modulo the model path).
    from scilink.agents.sim_agents._potential import (
        DeployedPotential, ASECalculatorSpec,
    )
    loader = "mace_off" if "off" in chosen else "mace_mp"
    potential = DeployedPotential(
        kind="mlip", backend="mace", model_name=chosen,
        model_file=f"/CLUSTER_PATH/to/{chosen}.model",
        elements=list(system["elements"]),
        ase_calculator=ASECalculatorSpec(
            import_line=f"from mace.calculators import {loader}",
            construct_expr=(
                f"{loader}(model='medium', device=DEVICE, "
                f"default_dtype='float64')"
            ),
            device_env_var="MACE_DEVICE",
        ),
        notes="PLACEHOLDER MODEL PATH — update on the cluster",
    )

    if runner == "lammps":
        from scilink.skills.molecular_dynamics.lammps import lammps
        run_path = lammps.run_with_potential(
            potential,
            structure_file=Path(structure_file).name,
            working_dir=str(out_dir),
            task="md",
            timestep=0.5,
            temperature=temperature,
            pressure=pressure,
        )
    else:  # ase
        from scilink.agents.sim_agents._ase_runner import generate_ase_script
        run_path = generate_ase_script(
            potential,
            working_dir=str(out_dir),
            structure_file=Path(structure_file).name,
            task="md",
            timestep=1.0,
            temperature=temperature,
            pressure=pressure,
            n_steps=n_steps,
            device=device,
        )
    return {
        "backend": "mace", "model_name": chosen,
        "elements": system["elements"], "runner": runner,
        "run_path": run_path,
        "note": "PLACEHOLDER MODEL PATH — update on the cluster",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--systems", nargs="+",
        default=["cu_co", "licoo2", "mos2_vac", "meoh", "ala"],
        choices=list(SYSTEMS.keys()),
        help="Demo lineup defaults to the five PNNL-relevant systems; "
             "first three route to mace-mp-0, last two to mace-off23.",
    )
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--pressure", type=float, default=None,
                        help="If set, NPT; otherwise NVT")
    parser.add_argument(
        "--api-key", default=os.environ.get("SCILINK_API_KEY", ""),
        help="Falls back to SCILINK_API_KEY env var",
    )
    parser.add_argument("--model-name", default="gemini-2.5-pro")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--runner", choices=["lammps", "ase"], default="lammps",
        help="lammps: write a LAMMPS+MACE input file. "
             "ase: write a runnable Python MD script using mace-torch.",
    )
    parser.add_argument(
        "--n-steps", type=int, default=200,
        help="ASE only: number of MD steps in the generated script.",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="ASE only: device the generated script will run on.",
    )
    parser.add_argument(
        "--run", action="store_true",
        help="ASE only: after generation, also execute the script "
             "in-process (requires MACE installed). Useful for "
             "single-command end-to-end demos on a GPU node.",
    )
    parser.add_argument(
        "--backend", choices=["mace", "chgnet"], default=None,
        help="Force a specific MLIP backend, bypassing the agent's "
             "LLM-driven model selection. Useful for benchmarking a "
             "specific engine end-to-end. CHGNet only works with "
             "--runner ase (no LAMMPS pair_style). Default: let the "
             "agent's LLM pick.",
    )
    args = parser.parse_args()

    if args.backend == "chgnet" and args.runner == "lammps":
        print(
            "ERROR: --backend chgnet is incompatible with --runner lammps. "
            "CHGNet has no LAMMPS pair_style; use --runner ase.",
            file=sys.stderr,
        )
        return 2

    mace_available, mace_info = check_mace_availability()
    print_section("ENVIRONMENT")
    print(f"  MACE installed:  {mace_available}")
    if mace_info.get("version"):
        print(f"  MACE version:    {mace_info['version']}")
    if mace_available:
        n_pretrained = len(mace_info.get("pretrained", []))
        print(f"  pretrained avail: {n_pretrained}")
    if not args.api_key:
        if mace_available:
            print("  WARNING: no SCILINK_API_KEY — LLM model selection will fail")
        else:
            print("  no SCILINK_API_KEY (fine for local-only walkthrough)")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = Path(
        args.out_dir or Path(__file__).parent / f"mlip_run_{timestamp}"
    ).resolve()
    base_out.mkdir(parents=True, exist_ok=True)
    print(f"  output dir:      {base_out}")

    from scilink.agents.sim_agents.mlip_agent import MLIPAgent

    summary = {"systems": {}, "mace_available": mace_available}

    for system_key in args.systems:
        system = SYSTEMS[system_key]()
        print_section(f"SYSTEM: {system['name']}  ({len(system['atoms'])} atoms)")

        system_dir = base_out / system["name"]
        system_dir.mkdir(exist_ok=True)
        data_path = write_data_file(system["atoms"], system_dir)
        print(f"  data file:       {data_path}")
        print(f"  elements:        {system['elements']}")
        print(f"  research goal:   {system['research_goal']}")

        agent = MLIPAgent(
            working_dir=str(system_dir),
            api_key=args.api_key or "sk-no-llm-needed",
            model_name=args.model_name,
        )
        report_skill_context(agent)

        if mace_available and args.api_key:
            system_info = {
                "elements": {e: system["atoms"].get_chemical_symbols().count(e)
                             for e in system["elements"]},
                "n_atoms": len(system["atoms"]),
            }
            try:
                result = run_with_mace(
                    agent, system_info, system["research_goal"], system_dir,
                    args.temperature, args.pressure,
                    args.runner, data_path,
                    args.n_steps, args.device,
                    backend=args.backend,
                )
            except Exception as exc:
                print(f"  deploy_pretrained failed: {exc!r}")
                if args.backend:
                    # An explicit --backend means this run is testing a
                    # specific engine. Falling back to run_without_mace
                    # (which is MACE-hardcoded) would mask the failure
                    # and the test would pass deceptively. Fail loudly.
                    print(
                        f"  --backend {args.backend} was forced — NOT "
                        f"falling back to the MACE deterministic path. "
                        f"Re-raising so the failure is visible."
                    )
                    raise
                result = run_without_mace(
                    agent, system, system_dir,
                    args.temperature, args.pressure,
                    args.runner, data_path,
                    args.n_steps, args.device,
                )
        else:
            result = run_without_mace(
                agent, system, system_dir,
                args.temperature, args.pressure,
                args.runner, data_path,
                args.n_steps, args.device,
            )

        output_file = result.get("run_path")
        label = "LAMMPS input" if args.runner == "lammps" else "ASE MD script"
        print()
        print(f"  --- generated {label} (head) ---")
        with open(output_file) as f:
            head = "".join(f.readlines()[:14])
        print(textwrap.indent(head, "    "))

        if args.run and args.runner == "ase" and mace_available:
            print()
            print("  --- executing ASE MD script in-process ---")
            import subprocess
            r = subprocess.run(
                [sys.executable, output_file],
                cwd=str(system_dir),
                env={**os.environ, "MACE_DEVICE": args.device},
            )
            print(f"  script exit code: {r.returncode}")
            for artifact in ("thermo.log", "traj.traj"):
                p = system_dir / artifact
                if p.exists():
                    print(f"  produced: {p} ({p.stat().st_size} bytes)")
        elif args.run and args.runner == "lammps":
            print("  --run is ASE-only; for LAMMPS runs, submit `lmp -in in.lammps` separately.")
        elif args.run and not mace_available:
            print("  --run skipped: MACE not installed.")

        summary["systems"][system["name"]] = {
            "data_file": str(data_path),
            "runner": args.runner,
            "output_file": output_file,
            "backend": result.get("backend"),
            "model_name": result.get("model_name"),
        }

    summary_path = base_out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print_section("DONE")
    print(f"  summary: {summary_path}")
    if not mace_available:
        print()
        print("  Next step: copy this directory to a node with MACE installed")
        print("  and rerun with the same command — the script will detect MACE")
        print("  and run deploy_pretrained end-to-end.")


if __name__ == "__main__":
    main()
