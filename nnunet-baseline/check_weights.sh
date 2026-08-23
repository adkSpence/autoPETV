#!/bin/bash

# Script to check if the trained model weights are properly downloaded
# via git-lfs (large checkpoint files aren't stored as plain text in git).
# No auto-download fallback: our checkpoint only exists via git-lfs, so a
# missing/pointer file means `git lfs pull` wasn't run -- fail loudly
# rather than silently substituting a different model's weights.

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
PLAIN_DIR="$SCRIPTPATH/nnUNet_results/Dataset990_AutoPETCombined/nnUNetTrainer_500ep_freqsave__nnUNetPlans__3d_fullres"
RESENCL_DIR="$SCRIPTPATH/nnUNet_results/Dataset990_AutoPETCombined/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres"
# process.py ensembles plain fold 0+1 with ResEncL fold 2 (nnUNetv2_ensemble
# across configurations) -- all three checkpoints are required.
CHECKPOINT_PATHS=(
    "$PLAIN_DIR/fold_0/checkpoint_final.pth"
    "$PLAIN_DIR/fold_1/checkpoint_final.pth"
    "$RESENCL_DIR/fold_2/checkpoint_final.pth"
)

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "==============================================="
echo "Checking nnUNet model weights..."
echo "==============================================="

is_git_lfs_pointer() {
    local file="$1"
    if [ ! -f "$file" ]; then
        return 1
    fi
    local filesize=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
    if [ "$filesize" -lt 1000 ]; then
        if head -n 1 "$file" 2>/dev/null | grep -q "version https://git-lfs.github.com"; then
            return 0
        fi
    fi
    return 1
}

for CHECKPOINT_PATH in "${CHECKPOINT_PATHS[@]}"; do
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo -e "${RED}ERROR: Checkpoint file not found: $CHECKPOINT_PATH${NC}"
        echo "Run 'git lfs pull' to fetch the trained model weights."
        exit 1
    fi

    if is_git_lfs_pointer "$CHECKPOINT_PATH"; then
        echo -e "${RED}ERROR: Checkpoint file is a git-lfs pointer (not downloaded): $CHECKPOINT_PATH${NC}"
        echo "Run 'git lfs pull' to fetch the trained model weights."
        exit 1
    fi

    filesize=$(stat -f%z "$CHECKPOINT_PATH" 2>/dev/null || stat -c%s "$CHECKPOINT_PATH" 2>/dev/null)
    filesize_mb=$((filesize / 1024 / 1024))

    if [ "$filesize_mb" -lt 100 ]; then
        echo -e "${RED}ERROR: Checkpoint file seems too small (${filesize_mb}MB). Expected > 100MB: $CHECKPOINT_PATH${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ $(basename "$(dirname "$CHECKPOINT_PATH")")/checkpoint_final.pth exists and appears valid (${filesize_mb}MB)${NC}"
done
echo -e "${GREEN}===============================================${NC}"
echo -e "${GREEN}Weight check complete! Ready to proceed.${NC}"
echo -e "${GREEN}===============================================${NC}"
exit 0
