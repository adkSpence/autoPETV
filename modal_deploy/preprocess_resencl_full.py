"""
Preprocess the full Dataset990_AutoPETCombined (1614 cases) under ResEncL
plans -- the data Sahib's 1000-epoch scale runs consume. CPU-only.

Measured justification for ResEncL: Dice 0.7506 / lesion-F1 0.8723 vs plain
UNet 0.7102 / 0.8214 on the fixed 250-case ablation (identical protocol).

Timeout is deliberately generous (24h): two prior runs (interactive-refine,
ResEncL ablation) were killed by a 6h ceiling sized for happy-path timing.

Usage:
    modal run --detach modal_deploy/preprocess_resencl_full.py
"""
from pathlib import Path

import modal

VOLUME_PATH = "/vol"
DATASET_ID = 990
PLANS_IDENTIFIER = "nnUNetResEncUNetLPlans"

app = modal.App("autopetv-preprocess-resencl-full")
volume = modal.Volume.from_name("autopetv-combined-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install("nnunetv2==2.6.0", "SimpleITK==2.4.1")


@app.function(
    image=image,
    cpu=16,
    memory=65536,
    timeout=24 * 3600,
    volumes={VOLUME_PATH: volume},
    env={
        "nnUNet_raw": str(Path(VOLUME_PATH) / "nnUNet_raw"),
        "nnUNet_preprocessed": str(Path(VOLUME_PATH) / "nnUNet_preprocessed"),
        "nnUNet_results": str(Path(VOLUME_PATH) / "nnUNet_results"),
    },
)
def preprocess():
    import subprocess

    plans_path = (Path(VOLUME_PATH) / "nnUNet_preprocessed" / "Dataset990_AutoPETCombined" /
                  f"{PLANS_IDENTIFIER}.json")
    if not plans_path.exists():
        print("=== Planning (ResEncL) ===", flush=True)
        subprocess.run(
            ["nnUNetv2_plan_and_preprocess", "-d", str(DATASET_ID), "-pl", "nnUNetPlannerResEncL", "--no_pp"],
            check=True,
        )
    else:
        print("Plans already exist, skipping planning.", flush=True)

    out_dir = (Path(VOLUME_PATH) / "nnUNet_preprocessed" / "Dataset990_AutoPETCombined" /
               f"{PLANS_IDENTIFIER}_3d_fullres")
    n_existing = len(list(out_dir.glob("*.b2nd"))) if out_dir.exists() else 0
    print(f"Existing preprocessed arrays: {n_existing}", flush=True)

    print("=== Preprocessing 1614 cases (ResEncL plans, 3d_fullres) ===", flush=True)
    subprocess.run(
        ["nnUNetv2_preprocess", "-d", str(DATASET_ID), "-plans_name", PLANS_IDENTIFIER,
         "-c", "3d_fullres", "-np", "12"],
        check=True,
    )
    volume.commit()

    n_files = len(list(out_dir.glob("*")))
    print(f"=== Done. {n_files} files in {out_dir.name} ===", flush=True)
    print("Record this count in sahib_handoff/EXPECTED_COUNTS.txt", flush=True)


@app.local_entrypoint()
def main():
    call = preprocess.spawn()
    print(f"Spawned detached call: {call.object_id}")
