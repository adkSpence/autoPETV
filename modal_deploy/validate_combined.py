"""
Run nnU-Net's final validation only (--val) for the combined FDG+PSMA
4-channel model, using the checkpoint already saved by train_combined.py
(training itself completed cleanly; the automatic post-training
validation phase appears to have hung and was killed separately).

Validation split here is larger than the FDG-only run (~320 cases vs
~180), so timeout is sized generously from the start rather than
repeating the earlier too-short-timeout mistake.

Usage:
    modal run --detach modal_deploy/validate_combined.py
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
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500
        self.save_every = 1
'''

app = modal.App("autopetv-validate-combined")
volume = modal.Volume.from_name("autopetv-combined-data", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install("nnunetv2==2.6.0", "SimpleITK==2.4.1")


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=16384,
    timeout=20 * 3600,
    volumes={VOLUME_PATH: volume},
    env={
        "nnUNet_raw": str(Path(VOLUME_PATH) / "nnUNet_raw"),
        "nnUNet_preprocessed": str(Path(VOLUME_PATH) / "nnUNet_preprocessed"),
        "nnUNet_results": str(Path(VOLUME_PATH) / "nnUNet_results"),
    },
)
def validate():
    import subprocess

    import nnunetv2

    trainer_dir = Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer" / "variants" / "training_length"
    trainer_dir.mkdir(parents=True, exist_ok=True)
    (trainer_dir / f"{TRAINER_NAME}.py").write_text(CUSTOM_TRAINER_CODE)

    subprocess.run(
        [
            "nnUNetv2_train",
            str(DATASET_ID), CONFIGURATION, str(FOLD),
            "-tr", TRAINER_NAME,
            "-p", PLANS_IDENTIFIER,
            "--val",
        ],
        check=True,
    )
    volume.commit()
    print("Validation complete.", flush=True)


@app.local_entrypoint()
def main():
    call = validate.spawn()
    print(f"Spawned detached call: {call.object_id}")
