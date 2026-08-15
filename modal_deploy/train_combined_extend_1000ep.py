"""
Extend the existing, completed 500-epoch combined-model training run to
1000 epochs, resuming from checkpoint_final.pth (cheaper than a fresh
1000-epoch run -- only ~500 more epochs needed, not 1000 from scratch).

Keeps the exact same trainer class name (nnUNetTrainer_500ep_freqsave) so
the output folder path matches and nnU-Net's --c resume logic finds the
existing checkpoint -- only num_epochs changes internally, to 1000. Always
passes --c (rather than checking for checkpoint_latest.pth, which doesn't
exist for a normally-completed run -- only checkpoint_final.pth does);
nnU-Net's own continue-training resolution already knows to fall back
through final -> latest -> best on its own.

Note: nnU-Net's LR schedule will show a discontinuity at the resume point
(current_epoch=500 against the new num_epochs=1000 target recalculates a
higher LR than where the original 500-epoch schedule ended) -- a legitimate
warm-restart-style extension, not a smooth single continuous decay curve
the whole way from epoch 0.

Usage:
    modal run --detach modal_deploy/train_combined_extend_1000ep.py
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
    """Extended to a 1000-epoch schedule, resuming from the completed
    500-epoch checkpoint. Checkpointing every epoch so an interruption
    costs at most one epoch."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000
        self.save_every = 1
'''

app = modal.App("autopetv-train-combined-extend")
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
    final_exists = (output_folder / "checkpoint_final.pth").exists()
    latest_exists = (output_folder / "checkpoint_latest.pth").exists()
    print(f"checkpoint_final.pth: {final_exists}, checkpoint_latest.pth: {latest_exists}", flush=True)
    if not (final_exists or latest_exists):
        raise RuntimeError("No checkpoint found to extend -- refusing to silently start fresh.")

    cmd = [
        "nnUNetv2_train",
        str(DATASET_ID), CONFIGURATION, str(FOLD),
        "-tr", TRAINER_NAME,
        "-p", PLANS_IDENTIFIER,
        "--c",
    ]

    subprocess.run(cmd, check=True)
    print("Extended training complete.", flush=True)


@app.local_entrypoint()
def main():
    call = train.spawn()
    print(f"Spawned detached call: {call.object_id}")
