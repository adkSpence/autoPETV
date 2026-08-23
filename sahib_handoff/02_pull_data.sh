#!/bin/bash
# Pull the preprocessed ResEncL dataset. Idempotent: gsutil rsync only
# transfers missing/changed files, so re-running after an interruption is safe.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/01_setup_env.sh"

SRC="gs://autopetv-data-transfer/nnUNet_preprocessed_resencl/Dataset990_AutoPETCombined"
DST="$nnUNet_preprocessed/Dataset990_AutoPETCombined"
mkdir -p "$DST"

gsutil -m rsync -r "$SRC" "$DST"

# --- verify the pieces training actually needs ---
for f in dataset.json splits_final.json nnUNetResEncUNetLPlans.json; do
    [ -f "$DST/$f" ] || { echo "FATAL: missing $DST/$f"; exit 1; }
done
[ -d "$DST/nnUNetResEncUNetLPlans_3d_fullres" ] || { echo "FATAL: missing preprocessed data dir"; exit 1; }
[ -d "$DST/gt_segmentations" ] || { echo "FATAL: missing gt_segmentations"; exit 1; }

N_SPLITS=$(python3 -c "import json;print(len(json.load(open('$DST/splits_final.json'))))")
[ "$N_SPLITS" = "5" ] || { echo "FATAL: splits_final.json has $N_SPLITS splits, expected 5"; exit 1; }
N_FILES=$(find "$DST/nnUNetResEncUNetLPlans_3d_fullres" -type f | wc -l)
echo "Data OK: $N_FILES files in preprocessed dir, 5 splits present."
echo "EXPECTED_FILE_COUNT check: compare against the count in EXPECTED_COUNTS.txt"
cat "$(dirname "${BASH_SOURCE[0]}")/EXPECTED_COUNTS.txt" 2>/dev/null || true
