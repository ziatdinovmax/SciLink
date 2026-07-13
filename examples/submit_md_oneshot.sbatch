#!/bin/bash
#
# SLURM submission for the engine-neutral MD one-shot execution smoke.
# Runs the whole meta -> simulation -> route -> structure -> deck -> run ->
# refine chain as a batch job, so an SSH disconnect can't kill the run.
#
# Submit (export the proxy creds in the submit shell first — they inherit):
#
#     export SCILINK_API_KEY=<proxy-key>
#     export SCILINK_BASE_URL=https://ai-incubator-api.pnnl.gov/
#     sbatch examples/submit_md_oneshot.sbatch                 # Stage B (via meta)
#     STAGE=direct sbatch examples/submit_md_oneshot.sbatch    # Stage A (run_simulation)
#
# Overridable via env at submit time: REPO, BRANCH, CONDA_ENV, STAGE,
# SCILINK_MODEL, SCILINK_RUN_COMMAND. Output -> md_oneshot_<jobid>.out/.err.
#
#SBATCH --job-name=md_oneshot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=md_oneshot_%j.out
#SBATCH --error=md_oneshot_%j.err

set -euo pipefail

# ── config (all overridable from the submit environment) ──────────────
REPO="${REPO:-/qfs/people/alle927/SciLink/SciLink}"
BRANCH="${BRANCH:-meta-simulation-delegation}"
CONDA_ENV="${CONDA_ENV:-/people/alle927/.conda/envs/scilink_ffmd}"
STAGE="${STAGE:-meta}"   # "meta" -> --via-meta ; "direct" -> run_simulation only

# ── fail fast on missing proxy creds (the -project models need the proxy) ──
: "${SCILINK_API_KEY:?export SCILINK_API_KEY (proxy key) before sbatch}"
: "${SCILINK_BASE_URL:?export SCILINK_BASE_URL (proxy url) before sbatch}"

# ── activate the env by PATH (robust in a non-interactive batch shell; no
#    conda-init needed). LD_LIBRARY_PATH picks up the env's libstdc++. ──
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
PY="$CONDA_ENV/bin/python"

cd "$REPO"
git fetch origin --quiet
git checkout "$BRANCH" --quiet
git pull origin "$BRANCH" --quiet

# ── diagnostics so the .out file is self-describing ───────────────────
echo "=== environment ==="
echo "node        : $(hostname)"
echo "date        : $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "repo        : $REPO"
echo "branch tip  : $(git log --oneline -1)"
echo "python      : $($PY --version)"
echo "lmp on PATH : $(command -v lmp || echo 'NONE')"
echo "stage       : $STAGE"
echo "model       : ${SCILINK_MODEL:-<default>}"
echo "run_command : ${SCILINK_RUN_COMMAND:-<from skill default>}"
echo ""

# ── run ───────────────────────────────────────────────────────────────
FLAG=""; [ "$STAGE" = "meta" ] && FLAG="--via-meta"
echo "=================================================================="
echo "  run_md_oneshot_deception.py $FLAG"
echo "=================================================================="
$PY examples/run_md_oneshot_deception.py $FLAG

echo ""
echo "=================================================================="
echo "  done — job $SLURM_JOB_ID"
echo "=================================================================="
