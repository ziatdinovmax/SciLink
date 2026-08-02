#!/usr/bin/env python3
"""
Cluster smoke test for the molecular_qc / NWChem PR (#286).

Validates the ONE seam the unit tests mock: ClusterExecutor -> NWChem, plus
snapshot_run (cclib) against a REAL NWChem output.

Run this from a checkout of the `molecular-qc-agent` branch, in an env with
cclib installed, on (or with SSH access to) the cluster.

    pip install cclib
    python cluster_smoketest_molecular_qc.py           # test 1 only
    python cluster_smoketest_molecular_qc.py --full     # test 1 + test 2

Test 1 also writes the returned NWChem .out into tests/fixtures/nwchem/ — that
is the fixture that flips the skipped snapshot test to passing.
"""

import argparse
from pathlib import Path

# ─────────────────────────── FILL THIS IN ───────────────────────────
SSH = dict(
    hostname="LOGIN_NODE_HOSTNAME",     # e.g. "deception.pnl.gov"
    username="YOUR_USERNAME",
    key_path="",                        # e.g. "~/.ssh/id_ed25519"  (or use password="")
    proxy_jump="",                      # e.g. "you@bastion"        (leave "" if none)
    port=22,
)
RESOURCES = dict(
    partition="PARTITION",              # e.g. "normal" / "short"
    account="ALLOCATION",               # e.g. your project/charge code
    nodes=1,
    ntasks=8,                           # cores; must match the mpirun -np below
    time="00:20:00",
)
SETUP = [                               # shell lines run before the command
    "module load nwchem",               # adjust to your site, e.g. "module load nwchem/7.2"
]
RUN_COMMAND = "mpirun -np 8 nwchem {script}"   # or "srun nwchem {script}" on SLURM
# ─────────────────────────────────────────────────────────────────────

REPO = Path(__file__).resolve().parent
FIXTURES = REPO / "tests" / "fixtures" / "nwchem"

# A trivial, fast job: B3LYP/6-31G* single point on water.
WATER_DECK = """start h2o_smoketest
geometry units angstrom
  O   0.0000   0.0000   0.0000
  H   0.0000   0.0000   0.9572
  H   0.9266   0.0000  -0.2400
end
basis
  * library 6-31G*
end
dft
  xc b3lyp
end
task dft energy
"""

# Minimal amine geometry for the full-pipeline test (NWChem will optimize it).
METHYLAMINE_XYZ = """7
methylamine
C   0.051   0.704   0.000
N   0.051  -0.759   0.000
H   1.080   1.056   0.000
H  -0.454   1.086   0.885
H  -0.454   1.086  -0.885
H  -0.399  -1.093   0.813
H  -0.399  -1.093  -0.813
"""


def make_executor(work_dir: Path):
    from scilink.agents.sim_agents.cluster_executor import ClusterExecutor
    return ClusterExecutor.connect(
        hostname=SSH["hostname"], username=SSH["username"],
        key_path=SSH["key_path"], proxy_jump=SSH["proxy_jump"], port=SSH["port"],
        resources=RESOURCES, setup=SETUP,
        timeout=3600, poll_interval=20,
    )


def test1_plumbing():
    """Push a trivial deck through ClusterExecutor and parse it with snapshot_run."""
    from scilink.skills.molecular_qc.nwchem.nwchem_output import snapshot_run
    run_dir = REPO / "_smoketest_water"
    run_dir.mkdir(exist_ok=True)

    print("[test1] connecting + submitting trivial NWChem job ...")
    ex = make_executor(run_dir)
    result = ex.run(
        input_files={"h2o.nw": WATER_DECK},
        run_command=RUN_COMMAND.format(script="h2o.nw"),
        run_dir=str(run_dir),
    )
    print("[test1] executor result:", {k: result.get(k) for k in ("status", "returncode", "run_status")})

    print("[test1] snapshot_run on the downloaded output:")
    snap = snapshot_run(str(run_dir))
    for k in ("status", "files_found", "convergence_status", "scf_energy", "headline"):
        print(f"    {k}: {snap.get(k)}")

    outs = list(run_dir.glob("*.out")) + list(run_dir.glob("*.nwo")) + list(run_dir.glob("*.log"))
    if outs and snap.get("scf_energy") is not None:
        FIXTURES.mkdir(parents=True, exist_ok=True)
        dest = FIXTURES / "h2o_b3lyp_smoketest.out"
        dest.write_text(outs[-1].read_text(errors="replace"))
        print(f"[test1] ✓ captured fixture -> {dest.relative_to(REPO)} "
              f"(unblocks test_snapshot_real_output)")
    else:
        print("[test1] ⚠ no parseable .out — check the executor result and remote logs")


def test2_pipeline():
    """Full PR path: molecular_qc agent generates the deck, NWChem runs it."""
    from scilink.agents.sim_agents.simulation_pipeline import run_complete_workflow
    run_dir = REPO / "_smoketest_methylamine"
    run_dir.mkdir(exist_ok=True)
    (run_dir / "methylamine.xyz").write_text(METHYLAMINE_XYZ)

    print("[test2] running molecular_qc pipeline on methylamine (opt) ...")
    ex = make_executor(run_dir)
    res = run_complete_workflow(
        "Optimize the geometry of this amine with B3LYP/def2-SVP and report the SCF energy.",
        scale="molecular_qc", software="nwchem",
        structure_file=str(run_dir / "methylamine.xyz"),
        output_dir=str(run_dir),
        validate=True,
        executor=ex, run_command=RUN_COMMAND,
        max_run_cycles=2,
        # api_key / base_url: taken from your usual SciLink env config
    )
    print("[test2] final_status:", res.get("final_status"))
    print("[test2] engine:", res.get("engine"), "| steps:", res.get("steps_completed"))
    if "refinement" in res:
        print("[test2] refinement:", res["refinement"].get("status"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also run the full-pipeline test (test 2)")
    args = ap.parse_args()
    if SSH["hostname"] == "LOGIN_NODE_HOSTNAME":
        raise SystemExit("Fill in the SSH / RESOURCES / SETUP / RUN_COMMAND config block first.")
    test1_plumbing()
    if args.full:
        test2_pipeline()


if __name__ == "__main__":
    main()
