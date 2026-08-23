"""
Ablation: does giving PET the same robust, percentile-clipped normalization
CT already gets (CTNormalization) instead of plain ZScoreNormalization
help, given the data audit found 16% of cases have PET SUV outliers
(>100, likely bladder/injection-site artifacts) that a raw mean/std
normalization is not robust to?

Reuses Dataset992_AblationEdt (EDT encoding, 250-case subset) -- only the
PET channel's normalization scheme changes. CTNormalization only needs the
per-channel intensity stats already computed during fingerprinting for
every channel regardless of which scheme is assigned, so this just edits
the plans file and re-preprocesses under a new plans identifier; no new
raw data or re-fingerprinting needed.

Usage:
    modal run --detach modal_deploy/train_ablation_petnorm.py
"""
from pathlib import Path

import modal

DATASET_ID = 992
DATASET_NAME = "Dataset992_AblationEdt"
VOLUME_PATH = "/vol"
CONFIGURATION = "3d_fullres"
FOLD = 0
NUM_EPOCHS = 90
TRAINER_NAME = f"nnUNetTrainer_ablation{NUM_EPOCHS}ep"
NEW_PLANS_IDENTIFIER = "nnUNetPlansPETClip"

CUSTOM_TRAINER_CODE = f'''
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class {TRAINER_NAME}(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = {NUM_EPOCHS}
        self.save_every = 10
'''

app = modal.App("autopetv-train-ablation-petnorm")
volume = modal.Volume.from_name("autopetv-combined-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install("nnunetv2==2.6.0", "SimpleITK==2.4.1")


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=16384,
    timeout=6 * 3600,
    volumes={VOLUME_PATH: volume},
    env={
        "nnUNet_raw": str(Path(VOLUME_PATH) / "nnUNet_raw"),
        "nnUNet_preprocessed": str(Path(VOLUME_PATH) / "nnUNet_preprocessed"),
        "nnUNet_results": str(Path(VOLUME_PATH) / "nnUNet_results"),
    },
)
def train():
    import json
    import subprocess

    import nnunetv2

    trainer_dir = Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer" / "variants" / "training_length"
    trainer_dir.mkdir(parents=True, exist_ok=True)
    (trainer_dir / f"{TRAINER_NAME}.py").write_text(CUSTOM_TRAINER_CODE)

    preprocessed_dir = Path(VOLUME_PATH) / "nnUNet_preprocessed" / DATASET_NAME
    base_plans_path = preprocessed_dir / "nnUNetPlans.json"
    new_plans_path = preprocessed_dir / f"{NEW_PLANS_IDENTIFIER}.json"

    if not new_plans_path.exists():
        print("=== Building PET-clip plans variant ===", flush=True)
        plans = json.loads(base_plans_path.read_text())
        plans["plans_name"] = NEW_PLANS_IDENTIFIER

        schemes = plans["configurations"][CONFIGURATION]["normalization_schemes"]
        print(f"Original normalization_schemes: {schemes}", flush=True)
        schemes[1] = "CTNormalization"  # channel 1 = PET
        print(f"Modified normalization_schemes: {schemes}", flush=True)

        new_plans_path.write_text(json.dumps(plans, indent=2))

        print("=== Preprocessing with PET-clip plans ===", flush=True)
        subprocess.run(
            ["nnUNetv2_preprocess", "-d", str(DATASET_ID), "-plans_name", NEW_PLANS_IDENTIFIER, "-c", CONFIGURATION],
            check=True,
        )

    output_folder = (
        Path(VOLUME_PATH) / "nnUNet_results" / DATASET_NAME /
        f"{TRAINER_NAME}__{NEW_PLANS_IDENTIFIER}__{CONFIGURATION}" / f"fold_{FOLD}"
    )
    resume = (output_folder / "checkpoint_latest.pth").exists()
    print(f"Existing checkpoint: {resume}", flush=True)

    cmd = [
        "nnUNetv2_train", str(DATASET_ID), CONFIGURATION, str(FOLD),
        "-tr", TRAINER_NAME, "-p", NEW_PLANS_IDENTIFIER,
    ]
    if resume:
        cmd.append("--c")

    print("Starting training...", flush=True)
    subprocess.run(cmd, check=True)
    print("Training complete.", flush=True)


@app.local_entrypoint()
def main():
    call = train.spawn()
    print(f"Spawned detached call: {call.object_id}")
