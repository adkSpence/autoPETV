"""
Train fold 1 (a different 20% train/val split from fold 0) on the full
combined dataset, warm-started from fold 0's converged 1000-epoch weights
via nnU-Net's -pretrained_weights flag -- loads only the network's learned
weights (not optimizer state or epoch counter), so this starts at a fresh
epoch 0 with a clean LR schedule, but shouldn't need the full 1000 epochs
a cold random-init run would, since the network already has useful learned
features. Cheaper than a from-scratch fold while still being a genuinely
different, ensemble-worthy model (different held-out data).

Usage:
    modal run --detach modal_deploy/train_combined_fold1_warmstart.py
"""
from pathlib import Path

import modal

DATASET_ID = 990
DATASET_NAME = f"Dataset{DATASET_ID}_AutoPETCombined"
VOLUME_PATH = "/vol"
CONFIGURATION = "3d_fullres"
SOURCE_FOLD = 0
TARGET_FOLD = 1
NUM_EPOCHS = 450
SOURCE_TRAINER_NAME = "nnUNetTrainer_500ep_freqsave"  # fold 0's trainer (holds the 1000-epoch weights)
TARGET_TRAINER_NAME = "nnUNetTrainer_fold1_warmstart"
PLANS_IDENTIFIER = "nnUNetPlans"

CUSTOM_TRAINER_CODE = f'''
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class {TARGET_TRAINER_NAME}(nnUNetTrainer):
    """Fold 1, warm-started from fold 0's converged weights -- expected to
    converge faster than cold random init, so a shorter {NUM_EPOCHS}-epoch
    schedule rather than a full fresh 1000."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = {NUM_EPOCHS}
        self.save_every = 1
'''

app = modal.App("autopetv-train-fold1-warmstart")
volume = modal.Volume.from_name("autopetv-combined-data", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install("nnunetv2==2.6.0", "SimpleITK==2.4.1")


@app.function(
    image=image,
    gpu="A10",
    cpu=4,
    memory=32768,
    timeout=23 * 3600,
    volumes={VOLUME_PATH: volume},
    env={
        "nnUNet_raw": str(Path(VOLUME_PATH) / "nnUNet_raw"),
        "nnUNet_preprocessed": str(Path(VOLUME_PATH) / "nnUNet_preprocessed"),
        "nnUNet_results": str(Path(VOLUME_PATH) / "nnUNet_results"),
    },
)
def train():
    import subprocess

    import nnunetv2

    trainer_dir = Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer" / "variants" / "training_length"
    trainer_dir.mkdir(parents=True, exist_ok=True)
    (trainer_dir / f"{TARGET_TRAINER_NAME}.py").write_text(CUSTOM_TRAINER_CODE)

    source_checkpoint = (
        Path(VOLUME_PATH) / "nnUNet_results" / DATASET_NAME /
        f"{SOURCE_TRAINER_NAME}__{PLANS_IDENTIFIER}__{CONFIGURATION}" / f"fold_{SOURCE_FOLD}" /
        "checkpoint_final.pth"
    )
    if not source_checkpoint.exists():
        raise RuntimeError(f"Source checkpoint not found: {source_checkpoint} -- fold 0 may not be done yet.")
    print(f"Warm-starting from {source_checkpoint}", flush=True)

    output_folder = (
        Path(VOLUME_PATH) / "nnUNet_results" / DATASET_NAME /
        f"{TARGET_TRAINER_NAME}__{PLANS_IDENTIFIER}__{CONFIGURATION}" / f"fold_{TARGET_FOLD}"
    )
    resume = (output_folder / "checkpoint_latest.pth").exists() or (output_folder / "checkpoint_final.pth").exists()
    print(f"Existing fold-{TARGET_FOLD} checkpoint found: {resume}", flush=True)

    cmd = [
        "nnUNetv2_train",
        str(DATASET_ID), CONFIGURATION, str(TARGET_FOLD),
        "-tr", TARGET_TRAINER_NAME,
        "-p", PLANS_IDENTIFIER,
    ]
    if resume:
        cmd.append("--c")
    else:
        cmd += ["-pretrained_weights", str(source_checkpoint)]

    subprocess.run(cmd, check=True)
    print("Fold 1 training complete.", flush=True)


@app.local_entrypoint()
def main():
    call = train.spawn()
    print(f"Spawned detached call: {call.object_id}")
