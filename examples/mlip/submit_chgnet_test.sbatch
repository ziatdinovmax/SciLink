#!/bin/bash
#
# SLURM submission for the CHGNet + MLIPAgent test sequence.
# Submit from anywhere on deception:
#
#     sbatch examples/mlip/submit_chgnet_test.sbatch
#
# Output lands in chgnet_smoke_<jobid>.out / .err in the submit dir.
# Pick a less-busy partition by editing --partition below if a100 is
# queued (dlt and h100 are good alternatives; dlv has older drivers).
#
#SBATCH --job-name=chgnet_smoke
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=chgnet_smoke_%j.out
#SBATCH --error=chgnet_smoke_%j.err

set -euo pipefail

# ── env setup (same flow we settled on for interactive sessions) ──
source /people/alle927/scilink_mace_env.sh

cd /people/alle927/SciLink

# Pull the latest mlip_agents_v2 (skip if you already have it checked out)
git fetch origin --quiet
git checkout mlip_agents_v2 --quiet
git pull origin mlip_agents_v2 --quiet
echo "branch tip: $(git log --oneline -1)"
echo ""

# Ensure chgnet is in the env (pip is a no-op if already installed)
$CONDA_PREFIX/bin/pip install -q chgnet

# ── diagnostics so the .out file is self-describing ──
echo "=== environment ==="
echo "node     : $(hostname)"
echo "date     : $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "python   : $($CONDA_PREFIX/bin/python --version)"
echo "torch    : $($CONDA_PREFIX/bin/python -c 'import torch; print(torch.__version__, "cuda built-in:", torch.version.cuda)')"
echo "cuda     : $($CONDA_PREFIX/bin/python -c 'import torch; print("available:", torch.cuda.is_available())')"
echo "chgnet   : $($CONDA_PREFIX/bin/python -c 'import chgnet; print(chgnet.__version__)')"
nvidia-smi -L
echo ""

# ── STEP 1: chgnet alone (no SciLink agent) ──
# Confirms chgnet + torch + ase are healthy in this env. Fast (<30s).
echo "=================================================================="
echo "  STEP 1: chgnet standalone smoke"
echo "=================================================================="
$CONDA_PREFIX/bin/python examples/mlip/chgnet_smoke.py
echo ""

# ── STEP 2: full MLIPAgent + ASE runner + CHGNet, executing in-process ──
# Forces backend=chgnet (skips LLM selection). Runs 50 steps of NVT
# on Cu(111) + CO. Outputs a self-contained dir with run_md.py,
# thermo.log, traj.traj.
echo "=================================================================="
echo "  STEP 2: MLIPAgent + ASE + CHGNet (cu_co_slab, 50 steps)"
echo "=================================================================="
$CONDA_PREFIX/bin/python examples/mlip/deploy_example.py \
    --systems cu_co \
    --runner ase \
    --backend chgnet \
    --n-steps 50 \
    --device cuda \
    --run

echo ""
echo "=================================================================="
echo "  done"
echo "=================================================================="
