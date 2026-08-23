"""
Download the already-generated EDT-encoded FG/BG scribble channels
(_0002/_0003) for the 250-case ablation subset from Dataset992_AblationEdt
-- the raw CT/PET/labels were already pulled by download_ablation_subset.py;
this fills the gap so the Kaggle-side pipeline doesn't need to reimplement
scribble generation.

Usage:
    modal run modal_deploy/download_ablation_scribbles.py
"""
import json
from pathlib import Path

import modal

VOLUME_PATH = "/vol"
SOURCE_DATASET_NAME = "Dataset992_AblationEdt"

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "kaggle_ablation_subset"

app = modal.App("autopetv-download-ablation-scribbles")
volume = modal.Volume.from_name("autopetv-combined-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=1800)
def fetch_case(case_id: str):
    images_dir = Path(VOLUME_PATH) / "nnUNet_raw" / SOURCE_DATASET_NAME / "imagesTr"
    files = {}
    for suffix in ["0002", "0003"]:
        p = images_dir / f"{case_id}_{suffix}.nii.gz"
        if p.exists():
            files[f"scribbles/{case_id}_{suffix}.nii.gz"] = p.read_bytes()
    return files


@app.function(image=image, volumes={VOLUME_PATH: volume})
def get_subset_case_ids():
    subset = json.loads((Path(VOLUME_PATH) / "ablation_subset.json").read_text())
    return subset["train"] + subset["val"]


@app.local_entrypoint()
def main():
    case_ids = get_subset_case_ids.remote()
    print(f"Downloading EDT scribbles for {len(case_ids)} cases to {OUT_DIR}...")

    (OUT_DIR / "scribbles").mkdir(parents=True, exist_ok=True)

    for i, files in enumerate(fetch_case.map(case_ids)):
        for rel_path, data in files.items():
            out_path = OUT_DIR / rel_path
            out_path.write_bytes(data)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(case_ids)}")

    print(f"Done. Files written to {OUT_DIR}/scribbles")
