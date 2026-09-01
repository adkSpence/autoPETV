#!/bin/bash

# Script to check if the trained model weights are properly downloaded
# via git-lfs (large checkpoint files aren't stored as plain text in git).
# No auto-download fallback: our checkpoint only exists via git-lfs, so a
# missing/pointer file means `git lfs pull` wasn't run -- fail loudly
# rather than silently substituting a different model's weights.

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
MODEL_DIR="$SCRIPTPATH/../interactive/nnUNet_results/Dataset994_AutoPETInteractiveFull/nnUNetTrainerResEnc4ChannelRoundSupervisionEDT__nnUNetResEncUNetMPlans__3d_fullres"
CHECKPOINT_PATH="$MODEL_DIR/fold_0/checkpoint_deploy.pth"
EXPECTED_SHA256="3c43905f9472d6ee2bdc1a63101b4b00d338731e44a75fbfe530fa1479bde570"

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

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo -e "${RED}ERROR: Checkpoint file not found: $CHECKPOINT_PATH${NC}"
    echo "Run 'git lfs pull' to fetch the trained model weights."
    exit 1
fi

if is_git_lfs_pointer "$CHECKPOINT_PATH"; then
    echo -e "${RED}ERROR: Checkpoint file is a git-lfs pointer (not downloaded)${NC}"
    echo "Run 'git lfs pull' to fetch the trained model weights."
    exit 1
fi

filesize=$(stat -f%z "$CHECKPOINT_PATH" 2>/dev/null || stat -c%s "$CHECKPOINT_PATH" 2>/dev/null)
filesize_mb=$((filesize / 1024 / 1024))

if [ "$filesize_mb" -lt 100 ]; then
    echo -e "${RED}ERROR: Checkpoint file seems too small (${filesize_mb}MB). Expected > 100MB.${NC}"
    exit 1
fi

actual_sha256=$(sha256sum "$CHECKPOINT_PATH" | cut -d' ' -f1)
if [ "$actual_sha256" != "$EXPECTED_SHA256" ]; then
    echo -e "${RED}ERROR: Deployment checkpoint hash mismatch${NC}"
    echo "Expected: $EXPECTED_SHA256"
    echo "Actual:   $actual_sha256"
    exit 1
fi

echo -e "${GREEN}✓ Checkpoint file exists and appears valid (${filesize_mb}MB)${NC}"
for metadata in dataset.json plans.json; do
    if [ ! -f "$MODEL_DIR/$metadata" ]; then
        echo -e "${RED}ERROR: Missing model metadata: $MODEL_DIR/$metadata${NC}"
        exit 1
    fi
done
echo -e "${GREEN}===============================================${NC}"
echo -e "${GREEN}Weight check complete! Ready to proceed.${NC}"
echo -e "${GREEN}===============================================${NC}"
exit 0
