#!/bin/bash
# ALTERNATIVE to 02_pull_data.sh: build the dataset locally from the official
# AutoPET V raw release instead of pulling our 130GB preprocessed copy.
# Everything custom about our dataset is deterministic (fixed seeds), so a
# local build is bit-equivalent to ours -- verified by hash check at the end.
#
# Prerequisite: the official AutoPET V raw release (QIBA-aligned, Apr 2026,
# FDG + PSMA) arranged as flat nnU-Net style files:
#   $RAW_SRC/imagesTr/{case}_0000.nii.gz   (CT)
#   $RAW_SRC/imagesTr/{case}_0001.nii.gz   (PET, SUV)
#   $RAW_SRC/labelsTr/{case}.nii.gz
# If your copy is in a different layout, see modal_deploy/download_*.py in
# the repo root for how we acquired/arranged ours.
#
# Usage: RAW_SRC=/path/to/raw ./02b_build_dataset.sh
set -euo pipefail
: "${RAW_SRC:?set RAW_SRC=/path/to/official/raw (imagesTr+labelsTr)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
source "$HERE/01_setup_env.sh"

DST="$nnUNet_raw/Dataset990_AutoPETCombined"
mkdir -p "$DST"

# 1. Link official CT/PET/labels into the dataset (no copy needed)
ln -sfn "$RAW_SRC/imagesTr" "$DST/imagesTr_src" 2>/dev/null || true
mkdir -p "$DST/imagesTr" "$DST/labelsTr"
for f in "$RAW_SRC"/imagesTr/*_0000.nii.gz "$RAW_SRC"/imagesTr/*_0001.nii.gz; do
    ln -sf "$f" "$DST/imagesTr/$(basename "$f")"
done
for f in "$RAW_SRC"/labelsTr/*.nii.gz; do
    ln -sf "$f" "$DST/labelsTr/$(basename "$f")"
done
N_LABELS=$(ls "$DST/labelsTr" | wc -l)
[ "$N_LABELS" = "1614" ] || { echo "FATAL: $N_LABELS labels, expected 1614 -- wrong/partial raw release?"; exit 1; }

# 2. Generate the FG/BG scribble EDT channels (_0002/_0003) -- deterministic,
#    seed=42, centerline strategy, EDT truncation 10.0. ~1-2h on many cores.
"$HERE/../.venv/bin/python3" 2>/dev/null || true
pip install nibabel scikit-image connected-components-3d scipy -q
SCRIBBLE_LABELS_DIR="$DST/labelsTr" SCRIBBLE_OUT_DIR="$DST/imagesTr" \
    python3 "$REPO/modal_deploy/generate_scribbles_batch.py"

# 3. Verify the generated channels are bit-equivalent to ours -- MANDATORY.
python3 "$HERE/verify_channels.py" --check "$DST/imagesTr"

# 4. dataset.json (shipped) + ResEncL planning + preprocessing
cp "$HERE/dataset.json" "$DST/dataset.json"
nnUNetv2_plan_and_preprocess -d 990 -pl nnUNetPlannerResEncL --no_pp
nnUNetv2_preprocess -d 990 -plans_name nnUNetResEncUNetLPlans -c 3d_fullres -np 12

# 5. Our exact split definitions (shipped) -- MUST be in place before training
cp "$HERE/splits_final.json" "$nnUNet_preprocessed/Dataset990_AutoPETCombined/splits_final.json"

echo "Dataset built and verified. Next: ./03_train_fold.sh 2"
