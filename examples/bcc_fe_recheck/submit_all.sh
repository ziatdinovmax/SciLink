#!/bin/bash
# Submit every case in this benchmark dir. Run from the parent of
# the per-case subdirs (i.e. the dir that contains all the labels).
set -e
BASE="$(cd "$(dirname "$0")" && pwd)"
echo "Submitting all cases under $BASE"

( cd "$BASE/bcc_fe" && sbatch submit.sh )

echo "All cases submitted. Use 'squeue -u $USER' to monitor."
