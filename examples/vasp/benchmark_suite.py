"""Benchmark suite for the VASP agentic workflow.

Generates VASP inputs (POSCAR / INCAR / KPOINTS) for each registered
case via run_complete_workflow, writes a per-case SLURM submit
script, and emits a top-level submit_all.sh + manifest.

Each case lives in its own subdirectory under --output. The agent
calls happen locally (LLM); VASP runs on the cluster after you rsync
the output dir over.

Usage (locally, where SCILINK_API_KEY is set):

    # Generate inputs for the default case set:
    python examples/vasp/benchmark_suite.py

    # Custom output dir + only run a subset:
    python examples/vasp/benchmark_suite.py --output bench_run_2/ --only si_bulk mgo

    # Different model:
    python examples/vasp/benchmark_suite.py --model gemini-2.5-pro
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import List, Optional


# ══════════════════════════════════════════════════════════════
# Case registry
# ══════════════════════════════════════════════════════════════

@dataclasses.dataclass
class BenchmarkCase:
    label: str
    description: str
    notes: str = ""
    walltime: str = "00:30:00"
    nodes: int = 1
    ntasks_per_node: int = 16


DEFAULT_CASES: List[BenchmarkCase] = [
    BenchmarkCase(
        label="si_bulk",
        description=(
            "Bulk silicon in the diamond structure, 2-atom primitive cell. "
            "Compute the ground-state SCF energy with PBE."
        ),
        notes="Canonical baseline. Should converge first-try.",
    ),
    BenchmarkCase(
        label="mgo",
        description=(
            "Bulk MgO in the rocksalt structure (B1), 2-atom primitive cell. "
            "Compute the ground-state SCF energy with PBE."
        ),
        notes="Binary, ionic. Tests POTCAR assembly with two elements.",
    ),
    BenchmarkCase(
        label="bcc_fe",
        description=(
            "Bulk iron in the BCC structure, 1-atom primitive cell. "
            "This is a ferromagnetic system: use spin-polarized DFT "
            "(ISPIN = 2) with an initial MAGMOM of about 5 muB on iron. "
            "Compute the ground-state SCF energy with PBE."
        ),
        notes="Tests the agent's awareness of ISPIN=2 + MAGMOM for magnetic systems.",
    ),
    BenchmarkCase(
        label="si_vacancy",
        description=(
            "2x2x2 supercell of diamond silicon with one silicon vacancy. "
            "Relax the ionic positions (fix the cell shape and volume) and "
            "compute the total energy with PBE."
        ),
        walltime="01:00:00",
        notes="Defect, larger cell, ionic relaxation. Slower.",
    ),
]


# ══════════════════════════════════════════════════════════════
# SLURM submit-script generation
# ══════════════════════════════════════════════════════════════

# EDIT THESE when running on a cluster other than deception:
SLURM_DEFAULT_PSEUDO_DIR = "/path/to/potpaw_PBE.54"  # replace with cluster-specific path
SLURM_DEFAULT_MODULES = """\
module purge
module load intel/2022.1.0
module load mkl/2023.0.0
module load openmpi/5.0.7"""
SLURM_DEFAULT_VASP_CMD = "mpirun vasp_std"


def build_submit_script(
    case: BenchmarkCase,
    *,
    pseudo_dir: str = SLURM_DEFAULT_PSEUDO_DIR,
    modules: str = SLURM_DEFAULT_MODULES,
    vasp_command: str = SLURM_DEFAULT_VASP_CMD,
) -> str:
    """One submit.sh per case. Pseudo-dir + module loads are settings the
    user must adapt to their cluster — defaults are placeholders.

    Note: do not use textwrap.dedent here. Multi-line substitutions
    (e.g. `{modules}`) only get indented on the first line, which
    breaks dedent's "common leading whitespace" inference and leaves
    the script's `#!/bin/bash` on column 8. sbatch then rejects it.
    Plain f-string with everything anchored at column 0 sidesteps
    the whole class of bug.
    """
    job_name = f"scilink_{case.label}"
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={case.walltime}
#SBATCH --nodes={case.nodes}
#SBATCH --ntasks-per-node={case.ntasks_per_node}
#SBATCH --output={job_name}_%j.out
#SBATCH --error={job_name}_%j.err

set -e
cd "$SLURM_SUBMIT_DIR"

# ---- Module environment (edit to match your cluster) ----
{modules}

# ---- Assemble POTCAR from pseudo dir ----
# EDIT THIS: path to potpaw_PBE on your cluster
PSEUDO_DIR="{pseudo_dir}"

ELEMENTS=($(sed -n '6p' POSCAR))
if [ ${{#ELEMENTS[@]}} -eq 0 ]; then
    echo "No elements parsed from POSCAR; aborting." >&2
    exit 1
fi
: > POTCAR
for e in "${{ELEMENTS[@]}}"; do
    if [ ! -f "$PSEUDO_DIR/$e/POTCAR" ]; then
        echo "Missing POTCAR for element $e at $PSEUDO_DIR/$e/POTCAR" >&2
        exit 1
    fi
    cat "$PSEUDO_DIR/$e/POTCAR" >> POTCAR
done

# ---- Run VASP ----
{vasp_command}
"""


def build_submit_all_script(case_labels: List[str]) -> str:
    """One driver script that submits every case from the cluster side.

    Plain f-string (no textwrap.dedent) for the same reason as
    build_submit_script -- the multi-line `{sbatch_lines}` substitution
    breaks dedent's whitespace inference.
    """
    sbatch_lines = "\n".join(
        f'( cd "$BASE/{label}" && sbatch submit.sh )' for label in case_labels
    )
    return f"""#!/bin/bash
# Submit every case in this benchmark dir. Run from the parent of
# the per-case subdirs (i.e. the dir that contains all the labels).
set -e
BASE="$(cd "$(dirname "$0")" && pwd)"
echo "Submitting all cases under $BASE"

{sbatch_lines}

echo "All cases submitted. Use 'squeue -u $USER' to monitor."
"""


# ══════════════════════════════════════════════════════════════
# Main driver
# ══════════════════════════════════════════════════════════════

def resolve_cases(filter_labels: Optional[List[str]]) -> List[BenchmarkCase]:
    if not filter_labels:
        return DEFAULT_CASES
    by_label = {c.label: c for c in DEFAULT_CASES}
    unknown = [lbl for lbl in filter_labels if lbl not in by_label]
    if unknown:
        raise SystemExit(
            f"Unknown case labels: {unknown}. "
            f"Available: {sorted(by_label.keys())}"
        )
    return [by_label[lbl] for lbl in filter_labels]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=f"benchmark_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Parent output dir (default: benchmark_suite_<timestamp>/)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Run only these case labels (e.g. --only si_bulk mgo). "
             "Default: all DEFAULT_CASES.",
    )
    parser.add_argument(
        "--model",
        default="claude-opus-4-6",
        help="LLM model for the DFT pipeline. Default matches the wizard.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Explicit LLM API key. If unset, the pipeline auto-discovers "
             "based on --model.",
    )
    parser.add_argument(
        "--method",
        choices=["llm", "atomate2"],
        default="llm",
        help="Input-generation method for the DFT pipeline.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=4,
        help="Structure-refinement cycle cap for the DFT pipeline.",
    )
    parser.add_argument(
        "--pseudo-dir",
        default=None,
        help="Override the PSEUDO_DIR baked into each submit.sh.",
    )
    args = parser.parse_args()

    cases = resolve_cases(args.only)
    parent = Path(args.output).resolve()
    parent.mkdir(parents=True, exist_ok=True)

    print(f"Output dir: {parent}")
    print(f"Cases: {[c.label for c in cases]}")
    print(f"Model: {args.model}")
    print()

    # Lazy import so --help works without dependencies
    from scilink.agents.sim_agents.simulation_pipeline import run_complete_workflow

    manifest: dict = {
        "started_at": datetime.now().isoformat(),
        "model": args.model,
        "method": args.method,
        "cases": [],
    }

    for case in cases:
        case_dir = parent / case.label
        case_dir.mkdir(parents=True, exist_ok=True)
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"Case: {case.label}")
        print(f"  {case.description}")
        print()

        case_record: dict = {
            "label": case.label,
            "description": case.description,
            "notes": case.notes,
            "case_dir": str(case_dir),
        }

        try:
            result = run_complete_workflow(
                case.description,
                scale="periodic_dft",
                software="vasp",
                method=args.method,
                api_key=args.api_key,
                model_name=args.model,
                output_dir=str(case_dir),
                max_refinement_cycles=args.max_cycles,
            )
            # run_complete_workflow reports status under "final_status"
            # ("success" / "failed_structure_generation" /
            # "failed_input_generation"). Generated input filenames are the
            # keys of result["input_generation"]["input_files"].
            case_record["status"] = result.get("final_status", "unknown")
            case_record["steps_completed"] = result.get("steps_completed", [])
            input_files = (result.get("input_generation") or {}).get("input_files") or {}
            if input_files:
                case_record["final_files"] = sorted(input_files.keys())
                case_record["ready_for_vasp"] = result.get("final_status") == "success"

            # Per-case submit.sh
            pseudo_dir = args.pseudo_dir or SLURM_DEFAULT_PSEUDO_DIR
            (case_dir / "submit.sh").write_text(
                build_submit_script(case, pseudo_dir=pseudo_dir)
            )
            (case_dir / "submit.sh").chmod(0o755)
        except Exception as exc:
            case_record["status"] = "input_generation_failed"
            case_record["error"] = str(exc)
            print(f"  ERROR generating inputs: {exc}")

        manifest["cases"].append(case_record)
        print()

    # Top-level submit_all.sh + manifest
    (parent / "submit_all.sh").write_text(
        build_submit_all_script([c.label for c in cases])
    )
    (parent / "submit_all.sh").chmod(0o755)
    (parent / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Wrote benchmark suite to: {parent}")
    print()
    print("Per-case status:")
    for record in manifest["cases"]:
        # ✓ only when the orchestrator reported success end-to-end.
        # Anything else (failed_*, error, unknown, exception) → ✗.
        is_ok = record.get("status") == "success"
        marker = "✓" if is_ok else "✗"
        ready = ""
        if record.get("ready_for_vasp"):
            ready = " (inputs ready for VASP)"
        print(f"  {marker} {record['label']:14s}  {record['status']}{ready}")
    print()
    print("Next steps:")
    print(f"  1. Edit each */submit.sh: set PSEUDO_DIR + module loads to match your cluster.")
    print(f"     (Or pass --pseudo-dir on the next run to bake the right path in up-front.)")
    print(f"  2. rsync -av {parent}/ alle927@deception.pnl.gov:/people/alle927/{parent.name}/")
    print(f"  3. ssh alle927@deception.pnl.gov 'cd /people/alle927/{parent.name} && bash submit_all.sh'")
    print(f"  4. Monitor with 'squeue -u alle927'. After completion, scp the *.out files back.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
