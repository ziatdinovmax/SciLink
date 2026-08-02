"""Source 1 fixture generator — run the production generators N times per
prompt and emit stub label entries for hand-labeling.

Per the experiment design (Section 4.2): "Generate N=20 INCARs per prompt
with the unmodified PeriodicDFTAgent. Label each by hand for true_issues.
Yields ~60 labeled fixtures with realistic (not synthetic) error
distribution."

Same shape for LAMMPS — `MDSimulationAgent.generate_simulation` runs the
production prompt path N times and dumps the LAMMPS scripts.

Usage
-----
    # generate VASP Source 1 (3 prompts × 20 trials = 60 INCARs)
    python -m benchmark.critic_experiment.generate_source1 --engine vasp

    # generate LAMMPS Source 1
    python -m benchmark.critic_experiment.generate_source1 --engine lammps

    # both, smaller n for smoke
    python -m benchmark.critic_experiment.generate_source1 \\
        --engine both --n-trials 2

    # different model
    python -m benchmark.critic_experiment.generate_source1 \\
        --engine vasp --model claude-opus-4-7

After generation, the fixture files live under
``fixtures/<engine>/from_generator/`` and stub label rows
(with ``true_issues=[]``) are appended to ``fixtures/labels.jsonl``.
Filling in ``true_issues`` is the labeling pass — done by hand by the
author with physics expertise, NOT the variant implementer.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional

from ._schema import FixtureLabel, append_labels
from .prompts import VASP_PROMPTS, LAMMPS_PROMPTS
from .prompts.vasp import Prompt


EXPERIMENT_DIR = Path(__file__).resolve().parent

logger = logging.getLogger("critic_experiment.generate_source1")


# ──────────────────────────────────────────────────────────────────
#  Per-engine generation
# ──────────────────────────────────────────────────────────────────

def _build_structure(prompt: Prompt) -> "tuple[Path, str]":
    """Materialize the prompt's system to disk; return (path, format)."""
    from benchmark.systems import get_system
    from benchmark._vasp import write_poscar

    system = get_system(prompt.system_name)
    if system.fragments:
        atoms = system.fragments["full"]()
    else:
        atoms = system.build()

    # Write POSCAR for VASP; ASE-supported xyz/data for LAMMPS would
    # work but POSCAR is the lingua-franca on this codebase and the MD
    # agent reads it via ASE.
    tmpdir = Path(tempfile.mkdtemp(prefix=f"critic_exp_{prompt.id}_"))
    path = tmpdir / "POSCAR"
    write_poscar(atoms, str(path))
    return path, "poscar"


def _generate_one_vasp(prompt: Prompt, model_name: str,
                       api_key: Optional[str], base_url: Optional[str]) -> str:
    """One call to PeriodicDFTAgent.generate_inputs → returns INCAR text.

    Raises if the call fails so the caller can record the trial as
    failed (we still want labeled records of "the generator crashed
    on this prompt").
    """
    from scilink.agents.sim_agents.periodic_dft_agent import PeriodicDFTAgent

    structure_file, _ = _build_structure(prompt)
    agent = PeriodicDFTAgent(
        api_key=api_key, base_url=base_url, model_name=model_name,
    )
    result = agent.generate_inputs(
        structure_file=str(structure_file),
        request=prompt.request,
        software="vasp",
    )
    if result.get("status") != "success":
        raise RuntimeError(f"generate_inputs failed: {result.get('message')}")

    incar_text = result.get("input_files", {}).get("INCAR", "")
    if not incar_text:
        raise RuntimeError("generate_inputs returned no INCAR")
    return incar_text


def _generate_one_lammps(prompt: Prompt, model_name: str,
                         api_key: Optional[str], base_url: Optional[str]) -> str:
    """One call to MDSimulationAgent.generate_simulation → returns LAMMPS
    input script text.

    The return shape on this codebase puts the generated script either
    under ``result["script_content"]`` or as a path under
    ``result["script_path"]``. We probe both.
    """
    from scilink.agents.sim_agents.md_simulation_agent import MDSimulationAgent

    structure_file, _ = _build_structure(prompt)
    agent = MDSimulationAgent(
        working_dir=str(structure_file.parent),
        api_key=api_key, base_url=base_url, model_name=model_name,
    )
    result = agent.generate_simulation(
        structure_file=str(structure_file),
        research_goal=prompt.request,
        runner="lammps",
    )
    if not isinstance(result, dict):
        raise RuntimeError(f"generate_simulation returned non-dict: {type(result)}")
    if result.get("status") not in (None, "success"):
        raise RuntimeError(f"generate_simulation failed: {result.get('message')}")

    # Probe return shape: prefer inline content; fall back to file path.
    script_text = result.get("script_content") or result.get("script")
    if not script_text:
        sp = result.get("script_path") or result.get("input_script_path")
        if sp and Path(sp).exists():
            script_text = Path(sp).read_text(encoding="utf-8")
    if not script_text:
        raise RuntimeError(
            f"generate_simulation returned no script content; keys were: "
            f"{sorted(result.keys())}"
        )
    return script_text


# ──────────────────────────────────────────────────────────────────
#  Driver
# ──────────────────────────────────────────────────────────────────

def _write_fixture(text: str, engine: str, prompt_id: str, trial: int) -> Path:
    """Drop one generated fixture to disk; return relative path."""
    ext = "incar" if engine == "vasp" else "lammps"
    rel = f"fixtures/{engine if engine == 'lammps' else 'incar'}/from_generator/{engine}_{prompt_id}_{trial:03d}.{ext}"
    full = EXPERIMENT_DIR / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(text, encoding="utf-8")
    return Path(rel)


def _run_engine(engine: str, prompts, n_trials: int, model_name: str,
                api_key: Optional[str], base_url: Optional[str]) -> List[FixtureLabel]:
    """Run all prompts × n_trials for one engine; return stub labels."""
    gen = _generate_one_vasp if engine == "vasp" else _generate_one_lammps
    labels: List[FixtureLabel] = []

    for prompt in prompts:
        print(f"\n--- {engine} :: {prompt.id} ({n_trials} trials) ---")
        for trial in range(1, n_trials + 1):
            try:
                text = gen(prompt, model_name, api_key, base_url)
            except Exception as e:
                print(f"  trial {trial:03d}: FAILED — {e}")
                logger.debug("trace", exc_info=True)
                continue

            rel_path = _write_fixture(text, engine, prompt.id, trial)
            label = FixtureLabel(
                id=f"{engine}_{prompt.id}_{trial:03d}",
                engine=engine,
                source="from_generator",
                prompt_id=prompt.id,
                prompt_text=prompt.request,
                system_name=prompt.system_name,
                fixture_path=str(rel_path),
                true_issues=[],  # ← labeled by hand later
            )
            labels.append(label)
            print(f"  trial {trial:03d}: wrote {rel_path}")

    return labels


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine", choices=["vasp", "lammps", "both"], required=True,
        help="Which engine(s) to generate fixtures for.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=20,
        help="Trials per prompt. Design doc default = 20 (yields ~60/engine).",
    )
    parser.add_argument(
        "--model", default=os.environ.get("SCILINK_MODEL", "claude-sonnet-4-6"),
        help="LLM model name. Defaults to SCILINK_MODEL env or claude-sonnet-4-6.",
    )
    parser.add_argument(
        "--api-key", default=os.environ.get("SCILINK_API_KEY"),
        help="API key. Defaults to SCILINK_API_KEY env.",
    )
    parser.add_argument(
        "--base-url", default=os.environ.get("SCILINK_BASE_URL"),
        help="Optional base URL for internal proxy. Defaults to SCILINK_BASE_URL env.",
    )
    parser.add_argument(
        "--labels-file", default="fixtures/labels.jsonl",
        help="Path to labels.jsonl (relative to critic_experiment/).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    labels_path = EXPERIMENT_DIR / args.labels_file
    new_labels: List[FixtureLabel] = []

    if args.engine in ("vasp", "both"):
        new_labels.extend(
            _run_engine("vasp", VASP_PROMPTS, args.n_trials,
                        args.model, args.api_key, args.base_url)
        )
    if args.engine in ("lammps", "both"):
        new_labels.extend(
            _run_engine("lammps", LAMMPS_PROMPTS, args.n_trials,
                        args.model, args.api_key, args.base_url)
        )

    if new_labels:
        append_labels(labels_path, new_labels)
        print(f"\n✓ wrote {len(new_labels)} stub labels to {labels_path}")
        print(f"\nNext: fill in `true_issues` per fixture (labeling pass).")
        print(f"See {EXPERIMENT_DIR / 'README.md'} for the schema + workflow.")
    else:
        print("\n⚠ no fixtures generated (all trials failed?)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
