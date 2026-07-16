#!/bin/bash
#SBATCH --job-name=scilink_si_bulk
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --output=scilink_si_bulk_%j.out
#SBATCH --error=scilink_si_bulk_%j.err

set -e
cd "$SLURM_SUBMIT_DIR"

# ---- Module environment (edit to match your cluster) ----
module purge
module load intel/2022.1.0
module load mkl/2023.0.0
module load openmpi/5.0.7

# ---- Assemble POTCAR from pseudo dir ----
# EDIT THIS: path to potpaw_PBE on your cluster
PSEUDO_DIR="/people/alle927/VASP_POT/potpaw_PBE.54"

ELEMENTS=($(sed -n '6p' POSCAR))
if [ ${#ELEMENTS[@]} -eq 0 ]; then
    echo "No elements parsed from POSCAR; aborting." >&2
    exit 1
fi
: > POTCAR
for e in "${ELEMENTS[@]}"; do
    if [ ! -f "$PSEUDO_DIR/$e/POTCAR" ]; then
        echo "Missing POTCAR for element $e at $PSEUDO_DIR/$e/POTCAR" >&2
        exit 1
    fi
    cat "$PSEUDO_DIR/$e/POTCAR" >> POTCAR
done

# ---- Run VASP ----
mpirun vasp_std
