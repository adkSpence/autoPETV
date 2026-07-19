"""
Train the baseline nnU-Net architecture (3d_fullres, 4-channel
CT/PET/FG-scribble/BG-scribble) on the full combined FDG+PSMA dataset
(1614 cases) -- the actual scribble-capable, submittable model, on an
A10 GPU.

Same design as train_full.py: custom trainer (500 epochs, checkpoint
every epoch instead of the default every-50), auto-resume via --c if a
checkpoint already exists, .spawn() for genuine detachment.

Usage:
    modal run --detach modal_deploy/train_combined.py
"""
from pathlib import Path

import modal

DATASET_ID = 990
DATASET_NAME = f"Dataset{DATASET_ID}_AutoPETCombined"
VOLUME_PATH = "/vol"
CONFIGURATION = "3d_fullres"
FOLD = 0
TRAINER_NAME = "nnUNetTrainer_500ep_freqsave"
PLANS_IDENTIFIER = "nnUNetPlans"

CUSTOM_TRAINER_CODE = '''
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_500ep_freqsave(nnUNetTrainer):
    """500-epoch schedule (its own calibrated LR decay, not a truncated
    1000-epoch run) with checkpointing every epoch instead of the
    default every-50, so an interruption costs at most one epoch."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500
        self.save_every = 1
'''

app = modal.App("autopetv-train-combined")
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
    (trainer_dir / f"{TRAINER_NAME}.py").write_text(CUSTOM_TRAINER_CODE)

    output_folder = (
        Path(VOLUME_PATH) / "nnUNet_results" / DATASET_NAME /
        f"{TRAINER_NAME}__{PLANS_IDENTIFIER}__{CONFIGURATION}" / f"fold_{FOLD}"
    )
    checkpoint_path = output_folder / "checkpoint_latest.pth"
    resume = checkpoint_path.exists()
    print(f"Existing checkpoint found: {resume} ({checkpoint_path})", flush=True)

    cmd = [
        "nnUNetv2_train",
        str(DATASET_ID), CONFIGURATION, str(FOLD),
        "-tr", TRAINER_NAME,
        "-p", PLANS_IDENTIFIER,
    ]
    if resume:
        cmd.append("--c")

    subprocess.run(cmd, check=True)
    print("Training complete.", flush=True)


@app.local_entrypoint()
def main():
    call = train.spawn()
    print(f"Spawned detached call: {call.object_id}")
