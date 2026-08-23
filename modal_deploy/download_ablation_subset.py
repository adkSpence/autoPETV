"""
Download the 250-case ablation subset (raw CT/PET/labels, EDT scribbles)
from the Modal Volume to local disk, so it can be packaged as a Kaggle
Dataset. Only pulls the fixed subset from ablation_subset.json, not the
full 1614-case dataset.

Usage:
    modal run modal_deploy/download_ablation_subset.py
"""
import json
from pathlib import Path

import modal

VOLUME_PATH = "/vol"
SOURCE_DATASET_NAME = "Dataset990_AutoPETCombined"

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "kaggle_ablation_subset"

app = modal.App("autopetv-download-ablation-subset")
volume = modal.Volume.from_name("autopetv-combined-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=1800)
def fetch_case(case_id: str):
    raw_dir = Path(VOLUME_PATH) / "nnUNet_raw" / SOURCE_DATASET_NAME
    images_dir = raw_dir / "imagesTr"
    labels_dir = raw_dir / "labelsTr"

    files = {}
    for suffix in ["0000", "0001"]:
        p = images_dir / f"{case_id}_{suffix}.nii.gz"
        if p.exists():
            files[f"images/{case_id}_{suffix}.nii.gz"] = p.read_bytes()
    label_p = labels_dir / f"{case_id}.nii.gz"
    if label_p.exists():
        files[f"labels/{case_id}.nii.gz"] = label_p.read_bytes()
    return files


@app.function(image=image, volumes={VOLUME_PATH: volume})
def get_subset_case_ids():
    subset = json.loads((Path(VOLUME_PATH) / "ablation_subset.json").read_text())
    return subset["train"] + subset["val"]


@app.local_entrypoint()
def main():
    case_ids = get_subset_case_ids.remote()
    print(f"Downloading {len(case_ids)} cases to {OUT_DIR}...")

    (OUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labels").mkdir(parents=True, exist_ok=True)

    for i, files in enumerate(fetch_case.map(case_ids)):
        for rel_path, data in files.items():
            out_path = OUT_DIR / rel_path
            out_path.write_bytes(data)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(case_ids)}")

    print(f"Done. Files written to {OUT_DIR}")
