"""
Run nnUNetv2_plan_and_preprocess on the full 900-case dataset already
downloaded to the Modal Volume (see download_full_dataset.py). Uses
multiple CPU cores in parallel (-np) since preprocessing time scales with
case count and would otherwise take 10+ hours serially.

Usage:
    modal run --detach modal_deploy/preprocess_full.py
"""
from pathlib import Path

import modal

DATASET_ID = 996
VOLUME_PATH = "/vol"
NUM_PROCESSES = 8

app = modal.App("autopetv-preprocess-full")
volume = modal.Volume.from_name("autopetv-train-data", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install("nnunetv2==2.6.0", "SimpleITK==2.4.1")


@app.function(
    image=image,
    cpu=NUM_PROCESSES,
    memory=32768,
    timeout=6 * 3600,
    volumes={VOLUME_PATH: volume},
)
def preprocess():
    import os
    import subprocess

    env = os.environ.copy()
    env["nnUNet_raw"] = str(Path(VOLUME_PATH) / "nnUNet_raw")
    env["nnUNet_preprocessed"] = str(Path(VOLUME_PATH) / "nnUNet_preprocessed")
    env["nnUNet_results"] = str(Path(VOLUME_PATH) / "nnUNet_results")

    subprocess.run(
        [
            "nnUNetv2_plan_and_preprocess",
            "-d", str(DATASET_ID),
            "--verify_dataset_integrity",
            "-np", str(NUM_PROCESSES),
        ],
        env=env, check=True,
    )
    volume.commit()
    print("Preprocessing complete.", flush=True)


@app.local_entrypoint()
def main():
    # .spawn() hands the call off to run independently on Modal's side.
    call = preprocess.spawn()
    print(f"Spawned detached call: {call.object_id}")
