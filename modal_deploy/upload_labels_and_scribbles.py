"""
Upload the locally-generated labels + FG/BG scribble heatmaps to the
Modal Volume, alongside the CT/PET images already downloaded there
(download_combined_images.py) -- completes the 4-channel raw dataset.

Uploaded in batches (not one giant payload) since labels+heatmaps
together are ~830MB across 1614 cases.

Usage:
    modal run modal_deploy/upload_labels_and_scribbles.py
"""
from pathlib import Path

import modal

DATASET_ID = 990
DATASET_NAME = f"Dataset{DATASET_ID}_AutoPETCombined"
VOLUME_PATH = "/vol"
BATCH_SIZE = 200

ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = ROOT / "data_combined" / "labelsTr"
IMAGES_DIR = ROOT / "data_combined" / "imagesTr"

app = modal.App("autopetv-upload-labels-scribbles")
volume = modal.Volume.from_name("autopetv-combined-data", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=image, timeout=1800, volumes={VOLUME_PATH: volume})
def upload_batch(labels: dict[str, bytes], heatmaps: dict[str, bytes]):
    raw_dir = Path(VOLUME_PATH) / "nnUNet_raw" / DATASET_NAME
    (raw_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
    (raw_dir / "imagesTr").mkdir(parents=True, exist_ok=True)

    for name, data in labels.items():
        (raw_dir / "labelsTr" / name).write_bytes(data)
    for name, data in heatmaps.items():
        (raw_dir / "imagesTr" / name).write_bytes(data)

    volume.commit()
    return len(labels), len(heatmaps)


@app.local_entrypoint()
def main():
    case_ids = sorted(p.name.replace(".nii.gz", "") for p in LABELS_DIR.glob("*.nii.gz"))
    print(f"{len(case_ids)} cases to upload, in batches of {BATCH_SIZE}")

    for start in range(0, len(case_ids), BATCH_SIZE):
        batch = case_ids[start:start + BATCH_SIZE]
        labels = {f"{cid}.nii.gz": (LABELS_DIR / f"{cid}.nii.gz").read_bytes() for cid in batch}
        heatmaps = {}
        for cid in batch:
            for suffix in ("_0002.nii.gz", "_0003.nii.gz"):
                p = IMAGES_DIR / f"{cid}{suffix}"
                if p.exists():
                    heatmaps[f"{cid}{suffix}"] = p.read_bytes()

        n_labels, n_heatmaps = upload_batch.remote(labels, heatmaps)
        print(f"  batch {start}-{start+len(batch)}: uploaded {n_labels} labels, {n_heatmaps} heatmap files")

    print("Done.")
