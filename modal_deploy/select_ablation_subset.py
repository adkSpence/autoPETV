"""
Select a fixed, balanced subset of the combined FDG+PSMA dataset for fast
ablation screening (architecture + scribble-encoding experiments), mirroring
Sahib's approach: small-but-substantial subset, short training, quick
relative ranking before committing to a full expensive retrain.

125 FDG + 125 PSMA non-empty-GT cases (balanced rather than proportional to
the full dataset's 1014:600 split, so the smaller PSMA class isn't
underrepresented in a screening set this small), split 200 train / 50 val.
Writes the case-ID list to the Volume so every ablation variant trains on
the exact same cases -- a fair comparison, not a confound.

Usage:
    modal run modal_deploy/select_ablation_subset.py
"""
import json
import random

import modal

VOLUME_PATH = "/vol"
SOURCE_DATASET = "Dataset990_AutoPETCombined"
NUM_PER_TRACER = 125
SEED = 42

app = modal.App("autopetv-select-ablation-subset")
volume = modal.Volume.from_name("autopetv-combined-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install("nibabel", "numpy")


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=1800)
def select():
    from pathlib import Path
    import nibabel as nib
    import numpy as np

    labels_dir = Path(VOLUME_PATH) / "nnUNet_raw" / SOURCE_DATASET / "labelsTr"
    all_cases = sorted(p.name.replace(".nii.gz", "") for p in labels_dir.glob("*.nii.gz"))

    fdg_cases, psma_cases = [], []
    for i, case_id in enumerate(all_cases):
        if (i + 1) % 200 == 0:
            print(f"  scanned {i+1}/{len(all_cases)}", flush=True)
        gt = nib.load(labels_dir / f"{case_id}.nii.gz").get_fdata()
        if not np.any(gt):
            continue
        (fdg_cases if case_id.startswith("fdg_") else psma_cases).append(case_id)

    print(f"Non-empty-GT cases available: FDG={len(fdg_cases)}, PSMA={len(psma_cases)}", flush=True)

    random.seed(SEED)
    fdg_sample = random.sample(fdg_cases, min(NUM_PER_TRACER, len(fdg_cases)))
    psma_sample = random.sample(psma_cases, min(NUM_PER_TRACER, len(psma_cases)))

    subset = fdg_sample + psma_sample
    random.shuffle(subset)

    val_size = int(round(0.2 * len(subset)))
    val_cases = subset[:val_size]
    train_cases = subset[val_size:]

    result = {
        "seed": SEED,
        "num_per_tracer": NUM_PER_TRACER,
        "train": sorted(train_cases),
        "val": sorted(val_cases),
    }

    out_path = Path(VOLUME_PATH) / "ablation_subset.json"
    out_path.write_text(json.dumps(result, indent=2))
    volume.commit()

    print(f"\nTrain: {len(train_cases)}, Val: {len(val_cases)}, Total: {len(subset)}", flush=True)
    print(f"Saved to {out_path}", flush=True)


@app.local_entrypoint()
def main():
    select.remote()
