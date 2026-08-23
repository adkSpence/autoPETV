#!/bin/bash
# Push one fold's results back to the shared bucket.
# Usage: ./04_push_results.sh <fold>
set -euo pipefail
FOLD="${1:?usage: ./04_push_results.sh <fold 0-4>}"
source "$(dirname "${BASH_SOURCE[0]}")/01_setup_env.sh"

OUT="$nnUNet_results/Dataset990_AutoPETCombined/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_$FOLD"
DST="gs://autopetv-data-transfer/nnUNet_results_resencl/fold_$FOLD"

[ -f "$OUT/checkpoint_final.pth" ] || { echo "FATAL: no checkpoint_final.pth — training not finished?"; exit 1; }
[ -f "$OUT/validation/summary.json" ] || { echo "FATAL: no validation/summary.json — validation not finished?"; exit 1; }

gsutil -m cp "$OUT/checkpoint_final.pth" "$OUT/checkpoint_best.pth" "$DST/"
gsutil cp "$OUT"/training_log_*.txt "$DST/"
gsutil cp "$OUT/validation/summary.json" "$DST/validation_summary.json"
echo "Fold $FOLD pushed to $DST"
