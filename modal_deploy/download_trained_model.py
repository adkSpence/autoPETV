"""
Download the trained combined-model checkpoint (and its accompanying
plans/dataset json) from the Modal Volume to local disk, so it can be
placed into nnunet-baseline/nnUNet_results/ for real submission-container
inference.

Usage:
    modal run modal_deploy/download_trained_model.py
"""
from pathlib import Path

import modal

DATASET_ID = 990
DATASET_NAME = f"Dataset{DATASET_ID}_AutoPETCombined"
TRAINER_NAME = "nnUNetTrainer_500ep_freqsave"
PLANS_IDENTIFIER = "nnUNetPlans"
CONFIGURATION = "3d_fullres"
FOLD = 0
VOLUME_PATH = "/vol"

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "trained_model_download"

app = modal.App("autopetv-download-trained-model")
volume = modal.Volume.from_name("autopetv-combined-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=image, volumes={VOLUME_PATH: volume})
def fetch():
    result_dir = (
        Path(VOLUME_PATH) / "nnUNet_results" / DATASET_NAME /
        f"{TRAINER_NAME}__{PLANS_IDENTIFIER}__{CONFIGURATION}"
    )
    fold_dir = result_dir / f"fold_{FOLD}"

    files = {}
    for name in ["dataset.json", "plans.json"]:
        p = result_dir / name
        if p.exists():
            files[name] = p.read_bytes()

    for name in ["checkpoint_final.pth"]:
        p = fold_dir / name
        if p.exists():
            files[f"fold_{FOLD}/{name}"] = p.read_bytes()

    return files


@app.local_entrypoint()
def main():
    files = fetch.remote()
    for rel_path, data in files.items():
        out_path = OUT_DIR / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        print(f"Wrote {out_path} ({len(data)/1e6:.1f} MB)")
