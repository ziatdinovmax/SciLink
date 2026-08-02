#!/usr/bin/env python3
"""
scilink prepare-ff - AMBER Force Field Preparation

Generates LAMMPS-ready data files from PDB structures using the AMBER
force field pipeline (AmberTools + ParmEd).

Produces a self-contained LAMMPS data file with all coefficients and
charges embedded, plus a LAMMPS input header file with the correct
style commands.
"""

import os
import sys
import json
import argparse
import logging
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for 'scilink prepare-ff' command."""

    parser = argparse.ArgumentParser(
        prog="scilink prepare-ff",
        description="Prepare AMBER force field parameters for LAMMPS simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple peptide in vacuum
  scilink prepare-ff --pdb peptide.pdb --goal "Study conformational dynamics"

  # Protein solvated in water with ions
  scilink prepare-ff --pdb protein.pdb --goal "Protein folding" --solvate --neutralize

  # Custom output directory and water model
  scilink prepare-ff --pdb system.pdb --goal "Binding study" \\
      --output-dir ./my_sim --water-model opc --protein-ff ff19SB

  # With non-standard ligand
  scilink prepare-ff --pdb complex.pdb --goal "Drug binding" \\
      --ligand ligand.pdb --ligand-name LIG --ligand-charge -1 --solvate

  # Generate test system (alanine dipeptide)
  scilink prepare-ff --test --goal "Test AMBER pipeline"

  # Generate test system with solvation
  scilink prepare-ff --test --solvate --goal "Test solvated system"

  # Check if AmberTools are available
  scilink prepare-ff --check-tools

Pipeline:
  PDB → [pdb4amber] → [antechamber (if ligands)] → [parmchk2] → [tleap] → [ParmEd] → LAMMPS .data

Output Files:
  system.data           LAMMPS data file (all coefficients + charges)
  ff_params.lammps      LAMMPS input header (styles + read_data)
  preparation.json      Metadata about the preparation
  system.prmtop         AMBER topology (intermediate)
  system.inpcrd         AMBER coordinates (intermediate)

Requirements:
  AmberTools:  conda install -c conda-forge ambertools
  ParmEd:      conda install -c conda-forge parmed
  LLM API key: Set SCILINK_API_KEY, GOOGLE_API_KEY, or use --api-key

Environment Variables:
  SCILINK_API_KEY     API key for internal proxy
  GOOGLE_API_KEY      Google Gemini API key
  OPENAI_API_KEY      OpenAI API key
  ANTHROPIC_API_KEY   Anthropic API key
        """,
    )

    # ── Input ─────────────────────────────────────────────────────
    input_group = parser.add_argument_group("Input")
    input_group.add_argument(
        "--pdb",
        type=str,
        dest="pdb_file",
        help="Path to PDB structure file",
    )
    input_group.add_argument(
        "--goal",
        type=str,
        dest="research_goal",
        default="Molecular dynamics simulation",
        help="Research goal (guides force field selection)",
    )
    input_group.add_argument(
        "--test",
        action="store_true",
        help="Generate alanine dipeptide (ACE-ALA-NME) test system",
    )

    # ── Output ────────────────────────────────────────────────────
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir", "-o",
        type=str,
        dest="output_dir",
        default="./ff_output",
        help="Output directory (default: ./ff_output)",
    )

    # ── Solvation ─────────────────────────────────────────────────
    solv_group = parser.add_argument_group("Solvation")
    solv_group.add_argument(
        "--solvate",
        action="store_true",
        help="Add explicit water box via tleap",
    )
    solv_group.add_argument(
        "--box-buffer",
        type=float,
        default=10.0,
        help="Water box padding in Angstroms (default: 10.0)",
    )
    solv_group.add_argument(
        "--neutralize",
        action="store_true",
        help="Add counter-ions for charge neutrality",
    )
    solv_group.add_argument(
        "--water-model",
        type=str,
        default="tip3p",
        choices=["tip3p", "spce", "opc", "opc3", "tip4pew"],
        help="Water model (default: tip3p)",
    )

    # ── Force Field ───────────────────────────────────────────────
    ff_group = parser.add_argument_group("Force Field")
    ff_group.add_argument(
        "--protein-ff",
        type=str,
        default=None,
        help="Protein force field (e.g., ff19SB, ff14SB). Auto-selected if not specified.",
    )
    ff_group.add_argument(
        "--gaff-version",
        type=str,
        default="gaff2",
        choices=["gaff", "gaff2"],
        help="GAFF version for small molecules (default: gaff2)",
    )

    # ── Ligands ───────────────────────────────────────────────────
    lig_group = parser.add_argument_group("Ligands (non-standard residues)")
    lig_group.add_argument(
        "--ligand",
        type=str,
        action="append",
        dest="ligand_files",
        metavar="PDB",
        help="Path to ligand PDB file (can be repeated)",
    )
    lig_group.add_argument(
        "--ligand-name",
        type=str,
        action="append",
        dest="ligand_names",
        metavar="NAME",
        help="Residue name for ligand (same order as --ligand)",
    )
    lig_group.add_argument(
        "--ligand-charge",
        type=int,
        action="append",
        dest="ligand_charges",
        metavar="CHARGE",
        help="Net charge for ligand (same order as --ligand)",
    )

    # ── Model / API ───────────────────────────────────────────────
    api_group = parser.add_argument_group("Model / API")
    api_group.add_argument(
        "--model",
        type=str,
        default="gemini-3-pro-preview",
        help="LLM model name (default: gemini-3-pro-preview)",
    )
    api_group.add_argument(
        "--base-url",
        type=str,
        dest="base_url",
        default=None,
        help="Base URL for OpenAI-compatible endpoint",
    )
    api_group.add_argument(
        "--api-key",
        type=str,
        dest="api_key",
        default=None,
        help="API key (overrides environment variables)",
    )

    # ── Utilities ─────────────────────────────────────────────────
    util_group = parser.add_argument_group("Utilities")
    util_group.add_argument(
        "--check-tools",
        action="store_true",
        help="Check AmberTools availability and exit",
    )
    util_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Check tools mode ──────────────────────────────────────────
    if args.check_tools:
        return _check_tools()

    # ── Validate inputs ───────────────────────────────────────────
    if not args.pdb_file and not args.test:
        parser.error("Provide --pdb <file> or --test")

    if args.pdb_file and not Path(args.pdb_file).exists():
        parser.error(f"PDB file not found: {args.pdb_file}")

    # Build small_molecule_info from --ligand flags
    small_molecule_info = None
    if args.ligand_files:
        small_molecule_info = []
        names = args.ligand_names or []
        charges = args.ligand_charges or []
        for i, lig_file in enumerate(args.ligand_files):
            if not Path(lig_file).exists():
                parser.error(f"Ligand file not found: {lig_file}")
            small_molecule_info.append({
                "pdb": lig_file,
                "name": names[i] if i < len(names) else f"LIG{i+1}",
                "charge": charges[i] if i < len(charges) else 0,
            })

    # ── Resolve API key ───────────────────────────────────────────
    api_key = args.api_key or os.environ.get("SCILINK_API_KEY") \
              or os.environ.get("GOOGLE_API_KEY") \
              or os.environ.get("OPENAI_API_KEY") \
              or os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        parser.error(
            "No API key found. Set SCILINK_API_KEY, GOOGLE_API_KEY, "
            "or use --api-key"
        )

    # ── Run ───────────────────────────────────────────────────────
    return _run_preparation(
        pdb_file=args.pdb_file,
        research_goal=args.research_goal,
        output_dir=args.output_dir,
        use_test_system=args.test,
        solvate=args.solvate,
        box_buffer=args.box_buffer,
        neutralize=args.neutralize,
        water_model=args.water_model,
        protein_ff=args.protein_ff,
        small_molecule_info=small_molecule_info,
        api_key=api_key,
        model_name=args.model,
        base_url=args.base_url,
    )


# ─── Implementation ──────────────────────────────────────────────

def _check_tools() -> int:
    """Check AmberTools availability and print status."""
    from scilink.tools.amber_tools import check_amber_tools

    print("\n🔍 Checking AmberTools availability...\n")
    tools = check_amber_tools()

    for name, info in tools["tools"].items():
        status = "✅" if info["found"] else "❌"
        path = info["path"] or "not found"
        print(f"  {status} {name:15s} {path}")

    parmed = tools.get("parmed", {})
    if parmed.get("available"):
        print(f"  ✅ {'parmed':15s} v{parmed.get('version', '?')}")
    else:
        print(f"  ❌ {'parmed':15s} not installed")

    print()
    if tools["available"]:
        print("✅ All required tools available. Ready for AMBER pipeline.")
    else:
        print(f"❌ Missing: {', '.join(tools['missing'])}")
        print("\nInstall via:")
        print("  conda install -c conda-forge ambertools parmed")

    return 0 if tools["available"] else 1


def _generate_test_pdb(output_dir: Path) -> str:
    """Generate alanine dipeptide PDB using tleap."""
    pdb_path = output_dir / "alanine_dipeptide.pdb"

    tleap_script = f"""
source leaprc.protein.ff19SB
x = sequence {{ ACE ALA NME }}
savepdb x {pdb_path.resolve()}
quit
"""
    script_path = output_dir / "_build_test.in"
    script_path.write_text(tleap_script)

    result = subprocess.run(
        ["tleap", "-f", str(script_path)],
        capture_output=True, text=True, timeout=60,
    )

    # Clean up
    for f in [script_path, Path("leap.log")]:
        if f.exists():
            f.unlink()

    if pdb_path.exists() and pdb_path.stat().st_size > 0:
        print(f"✅ Generated test PDB: {pdb_path}")
        return str(pdb_path)

    raise RuntimeError(
        f"Failed to generate test PDB.\n"
        f"tleap stdout: {result.stdout[-300:]}\n"
        f"tleap stderr: {result.stderr[-300:]}"
    )


def _run_preparation(
    pdb_file,
    research_goal,
    output_dir,
    use_test_system,
    solvate,
    box_buffer,
    neutralize,
    water_model,
    protein_ff,
    small_molecule_info,
    api_key,
    model_name,
    base_url,
) -> int:
    """Run the AMBER preparation pipeline."""

    from scilink.tools.amber_tools import check_amber_tools
    from scilink.agents.sim_agents.force_field_agent import ForceFieldAgent

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Check tools ───────────────────────────────────────────────
    tools = check_amber_tools()
    if not tools["available"]:
        print(f"❌ Missing AmberTools: {tools['missing']}")
        print("Install: conda install -c conda-forge ambertools parmed")
        return 1

    # ── Generate or resolve PDB ───────────────────────────────────
    if use_test_system:
        print("\n🧪 Generating alanine dipeptide test system...")
        pdb_file = _generate_test_pdb(output_path)
    else:
        pdb_file = str(Path(pdb_file).resolve())

    # ── Print config ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"🧪 AMBER Force Field Preparation")
    print(f"{'='*60}")
    print(f"  PDB:          {pdb_file}")
    print(f"  Output:       {output_path}")
    print(f"  Goal:         {research_goal[:60]}...")
    print(f"  Solvate:      {solvate}")
    if solvate:
        print(f"  Water model:  {water_model}")
        print(f"  Box buffer:   {box_buffer} Å")
        print(f"  Neutralize:   {neutralize}")
    if protein_ff:
        print(f"  Protein FF:   {protein_ff}")
    if small_molecule_info:
        for sm in small_molecule_info:
            print(f"  Ligand:       {sm['name']} (charge {sm['charge']}) — {sm['pdb']}")
    print(f"  Model:        {model_name}")
    print(f"{'='*60}\n")

    # ── Initialize agent ──────────────────────────────────────────
    agent = ForceFieldAgent(
        working_dir=str(output_path),
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
        skill="amber",
    )

    # ── Step 1: Select force field ────────────────────────────────
    print("[1/3] Selecting force field...")
    selection = agent.select_force_field(
        pdb_file=pdb_file,
        research_goal=research_goal,
    )
    ff = selection["force_field"]
    print(f"  Force field:  {ff.get('force_field', '?')}")
    print(f"  Water model:  {ff.get('compatible_water_model', '?')}")
    print(f"  Skill used:   {selection.get('skill_used', '?')}")

    # ── Step 2: Run AMBER pipeline ────────────────────────────────
    print("\n[2/3] Running AMBER pipeline...")
    params = agent.acquire_parameters(
        selection_info=selection,
        pdb_file=pdb_file,
        small_molecule_info=small_molecule_info,
        solvate=solvate,
        box_buffer=box_buffer,
        neutralize=neutralize,
    )

    pipeline = params.get("pipeline", "unknown")
    print(f"  Pipeline:     {pipeline}")

    if pipeline != "amber":
        print(f"  ⚠️  AMBER pipeline was not used (fell back to: {params.get('source', '?')})")

    # ── Step 3: Generate LAMMPS files ─────────────────────────────
    print("\n[3/3] Generating LAMMPS files...")
    param_files = agent.generate_lammps_parameters(
        parameter_info=params,
        data_file=params.get("data_file", ""),
    )

    data_file = params.get("data_file")
    validation = params.get("validation", {})

    # ── Save metadata ─────────────────────────────────────────────
    result = {
        "status": "success" if validation.get("valid", False) else "warning",
        "data_file": data_file,
        "param_files": {k: str(v) for k, v in param_files.items()},
        "force_field": ff,
        "pipeline": pipeline,
        "validation": validation,
        "research_goal": research_goal,
        "solvated": solvate,
        "water_model": water_model if solvate else None,
        "pdb_file": pdb_file,
    }

    prep_file = output_path / "preparation.json"
    with open(prep_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # ── Print summary ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"✅ LAMMPS files ready")
    print(f"{'='*60}")
    print(f"  Data file:    {data_file}")
    print(f"  Param file:   {param_files.get('main', 'N/A')}")
    print(f"  Atoms:        {validation.get('n_atoms', '?')}")
    print(f"  Net charge:   {validation.get('total_charge', '?')}")
    print(f"  Pipeline:     {pipeline}")
    print(f"  Metadata:     {prep_file}")
    print(f"{'='*60}")

    # Show data file head
    if data_file and os.path.exists(data_file):
        print(f"\n--- {Path(data_file).name} (first 20 lines) ---")
        with open(data_file) as f:
            for i, line in enumerate(f):
                if i >= 20:
                    print("  ...")
                    break
                print(f"  {line.rstrip()}")

    # Show param file
    main_param = param_files.get("main")
    if main_param and os.path.exists(main_param):
        print(f"\n--- {Path(main_param).name} ---")
        with open(main_param) as f:
            print(f.read())

    if validation.get("valid"):
        print("\n✅ Ready for LAMMPS simulation!")
        print(f"   Use the data file and param file in your LAMMPS input script,")
        print(f"   or pass them to the LAMMPSOrchestrator.")
    else:
        print("\n⚠️  Validation warnings detected. Review the output above.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
