#!/bin/bash
# Idempotent environment setup. Run once, then `source 01_setup_env.sh` in
# every later shell (it exports the nnU-Net env vars).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

BASE="$PWD/workdir"
mkdir -p "$BASE/nnUNet_raw" "$BASE/nnUNet_preprocessed" "$BASE/nnUNet_results"

if [ ! -d "$BASE/venv" ]; then
    python3 -m venv "$BASE/venv"
    "$BASE/venv/bin/pip" install --upgrade pip
    "$BASE/venv/bin/pip" install nnunetv2==2.6.0
fi

export PATH="$BASE/venv/bin:$PATH"
export nnUNet_raw="$BASE/nnUNet_raw"
export nnUNet_preprocessed="$BASE/nnUNet_preprocessed"
export nnUNet_results="$BASE/nnUNet_results"

# --- sanity checks ---
python3 -c "import nnunetv2" || { echo "FATAL: nnunetv2 not importable"; exit 1; }
python3 -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU visible'" \
    || { echo "FATAL: torch sees no GPU (fine on a login node; required on the training node)"; }
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
if [ "$VRAM_MB" -lt 23000 ]; then
    echo "WARNING: GPU has ${VRAM_MB} MiB VRAM. ResEncL needs >= 24 GB — a 16 GB card WILL OOM."
fi
DISK_GB=$(df -BG --output=avail "$BASE" | tail -1 | tr -dc '0-9')
if [ "$DISK_GB" -lt 200 ]; then
    echo "WARNING: only ${DISK_GB} GB free at $BASE — need ~200 GB."
fi
echo "Env OK. base=$BASE vram=${VRAM_MB}MiB disk_free=${DISK_GB}GB"
