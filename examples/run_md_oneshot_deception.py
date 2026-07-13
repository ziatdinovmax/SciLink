"""End-to-end execution smoke for the engine-neutral one-shot on HPC.

Validates that SciLink can take a natural-language goal and *actually run* a
classical-MD simulation — route (scale, engine), build the structure + force
field, generate the LAMMPS deck, execute it via the engine skill's own run
command, and refine — with no human in the loop.

Two stages, smallest-first so the execution path is de-risked before the heavy
science:

  Stage A (default): drive the simulation orchestrator's ``run_simulation``
    tool directly. One LLM-driven workflow, no chat loop — the tightest test of
    the new execution plumbing.

  Stage B (--via-meta): the same goal through the meta agent in AUTONOMOUS
    mode (bare-``scilink`` experience), delegating to the simulation child.

Environment:
  SCILINK_API_KEY    proxy key (required for the internal proxy path)
  SCILINK_BASE_URL   proxy URL (required with the -project model names)
  SCILINK_MODEL      model name (default: claude-opus-4-8-project)
  SCILINK_RUN_COMMAND  optional run-command override, e.g. 'lmp_mpi -in {script}'
                       (use if the LAMMPS binary the skill finds is wrong)

Usage:
  python examples/run_md_oneshot_deception.py            # Stage A, direct
  python examples/run_md_oneshot_deception.py --via-meta # Stage B, through meta
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# A SMALL, FAST system on purpose: this validates the execution path, not the
# heavy electrolyte science. High-level system + goal only — no pre-chewed
# SMILES, box size, or component counts (SciLink decides those).
GOAL = (
    "Run a short classical molecular dynamics simulation of liquid water at "
    "298 K and 1 atm to confirm the box equilibrates to a sensible density."
)


def _creds():
    api_key = os.environ.get("SCILINK_API_KEY")
    base_url = os.environ.get("SCILINK_BASE_URL")
    model = os.environ.get("SCILINK_MODEL", "claude-opus-4-8-project")
    if not api_key or not base_url:
        sys.exit("Set SCILINK_API_KEY and SCILINK_BASE_URL (proxy) first.")
    return api_key, base_url, model


def stage_a(base_dir: Path) -> dict:
    """Direct run_simulation — the tightest test of the execution plumbing."""
    from scilink.agents.sim_agents.simulation_orchestrator import (
        SimulationOrchestratorAgent, SimulationMode,
    )
    api_key, base_url, model = _creds()
    orch = SimulationOrchestratorAgent(
        base_dir=str(base_dir), api_key=api_key, base_url=base_url,
        model_name=model, simulation_mode=SimulationMode.AUTONOMOUS,
    )
    kwargs = {"description": GOAL}
    run_command = os.environ.get("SCILINK_RUN_COMMAND")
    if run_command:
        kwargs["run_command"] = run_command
    print(f"▶ run_simulation (AUTONOMOUS): {GOAL}")
    out = orch.tools.execute_tool("run_simulation", **kwargs)
    return json.loads(out)


def stage_b(base_dir: Path) -> dict:
    """Through the meta agent in AUTONOMOUS — the bare-scilink experience."""
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent, MetaMode,
    )
    api_key, base_url, model = _creds()
    meta = MetaOrchestratorAgent(
        base_dir=str(base_dir), api_key=api_key, base_url=base_url,
        model_name=model, meta_mode=MetaMode.AUTONOMOUS,
    )
    print(f"▶ meta.chat (AUTONOMOUS) → delegate_to_simulation: {GOAL}")
    reply = meta.chat(
        "Please build and run this simulation end to end, then report the "
        f"equilibrated density: {GOAL}"
    )
    return {"meta_reply": reply}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--via-meta", action="store_true",
                    help="route through the meta agent (Stage B)")
    ap.add_argument("--base-dir", default=None)
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(args.base_dir or f"./md_oneshot_smoke_{stamp}").resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    result = (stage_b if args.via_meta else stage_a)(base_dir)

    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2, default=str))
    print(f"\nSession dir: {base_dir}")


if __name__ == "__main__":
    main()
