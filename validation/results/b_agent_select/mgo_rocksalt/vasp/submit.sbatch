#!/bin/bash
#SBATCH --job-name=val_mgo_rocksalt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=00:30:00
#SBATCH --output=vasp.out
#SBATCH --error=vasp.err

set -e
cd "$SLURM_SUBMIT_DIR"

# ── MPI / MKL toolchain (edit versions to match your cluster) ──
module purge
module load intel/2022.1.0
module load mkl/2023.0.0
module load openmpi/5.0.7

# ── Assemble POTCAR from the cluster pseudopotential library ──
# $VASP_PP_PATH points at the POTCAR root, e.g. .../potpaw_PBE.54
: "${VASP_PP_PATH:?set VASP_PP_PATH to the POTCAR library root}"
rm -f POTCAR
for el in Mg O; do
    if [ ! -f "$VASP_PP_PATH/$el/POTCAR" ]; then
        echo "Missing POTCAR for element $el at $VASP_PP_PATH/$el/POTCAR" >&2
        exit 1
    fi
    cat "$VASP_PP_PATH/$el/POTCAR" >> POTCAR
done

mpirun vasp_std
