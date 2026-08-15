#!/bin/bash
# Train one ResEncL fold at 1000 epochs (stock nnUNetTrainer default).
# Usage: ./03_train_fold.sh <fold>   (priority queue: 2 -> 3 -> 4 -> 0 -> 1;
#         every completed fold is independently useful, stop anywhere)
# Idempotent: resumes from checkpoint_latest.pth automatically if present.
set -euo pipefail
FOLD="${1:?usage: ./03_train_fold.sh <fold 0-4>}"
source "$(dirname "${BASH_SOURCE[0]}")/01_setup_env.sh"

OUT="$nnUNet_results/Dataset990_AutoPETCombined/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_$FOLD"

RESUME=()
if [ -f "$OUT/checkpoint_latest.pth" ]; then
    echo "Found checkpoint_latest.pth — resuming."
    RESUME=(--c)
fi

# NOTE: no -tr flag on purpose. Stock nnUNetTrainer = 1000 epochs, which is
# exactly what we want. Do not add a custom trainer.
nnUNetv2_train 990 3d_fullres "$FOLD" -p nnUNetResEncUNetLPlans "${RESUME[@]}"

echo "Fold $FOLD complete. Validation summary:"
python3 -c "
import json
s = json.load(open('$OUT/validation/summary.json'))
print('Mean Validation Dice:', s['foreground_mean']['Dice'])
"
