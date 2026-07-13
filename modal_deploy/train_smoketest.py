"""
Validate the nnU-Net training workflow end to end on Modal, using the tiny
local data subset (sandbox/download_data_subset.py) and nnU-Net's own
built-in nnUNetTrainer_1epoch debug variant.

PS. Sandbox is not tracked

This is a pipeline mechanics check not a real training run to check if preprocessing and training produce a valid checkpoint. Not
enough data/epochs to produce a usable model.

Usage:
    modal run modal_deploy/train_smoketest.py
"""
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

DATASET_ID = 997
DATASET_NAME = f"Dataset{DATASET_ID}_AutoPETSubset"

app = modal.App("autopetv-train-smoketest")
volume = modal.Volume.from_name("autopetv-train-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("nnunetv2==2.6.0", "SimpleITK==2.4.1")
)

VOLUME_PATH = "/vol"


@app.function(image=image, gpu="T4", timeout=3600, volumes={VOLUME_PATH: volume})
def train_smoketest(images: dict[str, bytes], labels: dict[str, bytes]):
    import os
    import subprocess

    raw_dir = Path(VOLUME_PATH) / "nnUNet_raw" / DATASET_NAME
    (raw_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (raw_dir / "labelsTr").mkdir(parents=True, exist_ok=True)

    for rel_path, data in images.items():
        (raw_dir / "imagesTr" / rel_path).write_bytes(data)
    for rel_path, data in labels.items():
        (raw_dir / "labelsTr" / rel_path).write_bytes(data)

    case_ids = sorted({p.name.replace("_0000.nii.gz", "") for p in (raw_dir / "imagesTr").glob("*_0000.nii.gz")})
    dataset_json = {
        "channel_names": {"0": "CT", "1": "PET"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(case_ids),
        "file_ending": ".nii.gz",
    }
    (raw_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    print(f"Wrote dataset.json for {len(case_ids)} cases: {case_ids}")

    env = os.environ.copy()
    env["nnUNet_raw"] = str(Path(VOLUME_PATH) / "nnUNet_raw")
    env["nnUNet_preprocessed"] = str(Path(VOLUME_PATH) / "nnUNet_preprocessed")
    env["nnUNet_results"] = str(Path(VOLUME_PATH) / "nnUNet_results")

    print("=== Running nnUNetv2_plan_and_preprocess ===", flush=True)
    subprocess.run(
        ["nnUNetv2_plan_and_preprocess", "-d", str(DATASET_ID), "--verify_dataset_integrity"],
        env=env, check=True,
    )

    print("=== Running nnUNetv2_train (1 epoch debug trainer) ===", flush=True)
    subprocess.run(
        ["nnUNetv2_train", str(DATASET_ID), "3d_fullres", "0", "-tr", "nnUNetTrainer_1epoch"],
        env=env, check=True,
    )

    results_dir = Path(VOLUME_PATH) / "nnUNet_results" / DATASET_NAME
    checkpoints = list(results_dir.rglob("checkpoint_*.pth"))
    print(f"Checkpoints produced: {[str(c) for c in checkpoints]}")

    volume.commit()
    return [str(c.relative_to(VOLUME_PATH)) for c in checkpoints]


@app.local_entrypoint()
def main():
    images_dir = DATA_DIR / "imagesTr"
    labels_dir = DATA_DIR / "labelsTr"

    images = {p.name: p.read_bytes() for p in images_dir.glob("*.nii.gz")}
    labels = {p.name: p.read_bytes() for p in labels_dir.glob("*.nii.gz")}
    print(f"Uploading {len(images)} image files, {len(labels)} label files...")

    checkpoints = train_smoketest.remote(images, labels)
    print(f"\nDone. Checkpoints on Modal Volume 'autopetv-train-data': {checkpoints}")
