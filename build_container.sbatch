#!/bin/bash
#SBATCH -A ATSCALE
#SBATCH -t "02:00:00"
#SBATCH -N 1
#SBATCH -J build_scilink_container
#SBATCH -o build_container_%j.out
#SBATCH -e build_container_%j.err

echo "================================================"
echo "Building SciLink MD Analysis Container"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Load modules
source /etc/profile.d/modules.sh
module purge
module load apptainer/1.4.5

# Set up Apptainer environment
export APPTAINER_TMPDIR=/scratch/${USER}/apptainer_tmp
export APPTAINER_CACHEDIR=/scratch/${USER}/apptainer_cache

# Clean and recreate directories
rm -rf $APPTAINER_TMPDIR
rm -rf $APPTAINER_CACHEDIR
mkdir -p $APPTAINER_TMPDIR
mkdir -p $APPTAINER_CACHEDIR

echo "Apptainer temp: $APPTAINER_TMPDIR"
echo "Apptainer cache: $APPTAINER_CACHEDIR"
echo ""

# Navigate to SciLink directory
cd /people/alle927/SciLink/SciLink

# Output paths
DOCKERFILE="Dockerfile.scilink_mdanalysis"
DEFFILE="scilink_mdanalysis.def"
SIFFILE="scilink_mdanalysis.sif"

# Verify Dockerfile exists
if [ ! -f "$DOCKERFILE" ]; then
    echo "ERROR: Dockerfile not found: $DOCKERFILE"
    exit 1
fi

echo "Dockerfile: $DOCKERFILE"
echo "Output: $SIFFILE"
echo ""

# Convert Dockerfile to Apptainer definition
echo "Step 1: Converting Dockerfile to Apptainer definition..."
spython recipe --force \
    --parser docker \
    --writer singularity \
    "$DOCKERFILE" \
    "$DEFFILE"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to convert Dockerfile"
    exit 1
fi

echo "✓ Conversion complete: $DEFFILE"
echo ""

# Build the container
echo "Step 2: Building Apptainer container..."
echo "This may take 20-30 minutes..."
echo ""

apptainer build --force --fakeroot "$SIFFILE" "$DEFFILE"

if [ $? -ne 0 ]; then
    echo "ERROR: Container build failed"
    exit 1
fi

echo ""
echo "================================================"
echo "✓ Container built successfully!"
echo "================================================"
echo "Container: $SIFFILE"
echo "Size: $(du -h $SIFFILE | cut -f1)"
echo ""
echo "To use:"
echo "  apptainer exec --bind /people/alle927:/people/alle927 \\"
echo "    $SIFFILE python your_script.py"
echo ""
echo "End time: $(date)"
echo "================================================"
