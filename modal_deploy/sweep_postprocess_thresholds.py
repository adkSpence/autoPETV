"""
Tune the postprocessing filter's (min_volume_ml, min_suv) thresholds
against real validation predictions -- nnU-Net's automatic post-training
validation already wrote predictions for all 323 held-out cases, so this
needs no new GPU inference, just a CPU-only grid search.

The current thresholds (0.35mL, SUV 6.0) were set by eyeballing false
positives on a single test case (see nnunet-baseline/postprocess_filters.py's
own docstring) -- this replaces that guess with the combination that
actually maximizes mean Dice across the real held-out set.

Usage:
    modal run modal_deploy/sweep_postprocess_thresholds.py
"""
import modal

DATASET_ID = 990
DATASET_NAME = f"Dataset{DATASET_ID}_AutoPETCombined"
VOLUME_PATH = "/vol"
TRAINER_NAME = "nnUNetTrainer_500ep_freqsave"
PLANS_IDENTIFIER = "nnUNetPlans"
CONFIGURATION = "3d_fullres"
FOLD = 0

VOLUME_THRESHOLDS = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]
SUV_THRESHOLDS = [0.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]

app = modal.App("autopetv-sweep-postprocess-thresholds")
volume = modal.Volume.from_name("autopetv-combined-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "nibabel", "numpy", "connected-components-3d",
)


@app.function(image=image, cpu=8, memory=16384, timeout=3 * 3600, volumes={VOLUME_PATH: volume})
def sweep():
    from pathlib import Path
    import cc3d
    import nibabel as nib
    import numpy as np

    val_dir = (Path(VOLUME_PATH) / "nnUNet_results" / DATASET_NAME /
               f"{TRAINER_NAME}__{PLANS_IDENTIFIER}__{CONFIGURATION}" / f"fold_{FOLD}" / "validation")
    raw_dir = Path(VOLUME_PATH) / "nnUNet_raw" / DATASET_NAME
    labels_dir = raw_dir / "labelsTr"
    images_dir = raw_dir / "imagesTr"

    pred_files = sorted(val_dir.glob("*.nii.gz"))
    print(f"Precomputing per-component stats for {len(pred_files)} validation cases "
          f"(loading full volumes once, discarding immediately after)...", flush=True)

    # Per case: gt_voxel_count, and one row per predicted connected component:
    # (volume_ml, suv_max, n_voxels, intersection_with_gt_voxels). This lets the
    # threshold sweep below run purely on these tiny summaries instead of
    # re-loading/re-filtering full-resolution volumes for every combo -- full
    # PET volumes for all 323 cases held simultaneously would need ~65GB.
    case_stats = []
    for i, pred_path in enumerate(pred_files):
        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(pred_files)}", flush=True)
        case_id = pred_path.name.replace(".nii.gz", "")
        gt_path = labels_dir / f"{case_id}.nii.gz"
        pet_path = images_dir / f"{case_id}_0001.nii.gz"
        if not (gt_path.exists() and pet_path.exists()):
            continue

        pred = nib.load(pred_path).get_fdata().astype(np.uint8)
        gt = nib.load(gt_path).get_fdata().astype(np.uint8)
        if gt.sum() == 0:
            continue  # empty-GT cases would always score 0/undefined Dice regardless of thresholds

        pet_img = nib.load(pet_path)
        pet = pet_img.get_fdata().astype(np.float32)
        spacing = pet_img.header.get_zooms()
        voxel_volume_ml = float(np.prod(spacing)) / 1000

        labeled, num_components = cc3d.connected_components(pred.astype(int), connectivity=18, return_N=True)
        components = []
        for label in range(1, num_components + 1):
            mask = labeled == label
            n_voxels = int(mask.sum())
            volume_ml = n_voxels * voxel_volume_ml
            suv_max = float(pet[mask].max())
            intersection = int((mask & (gt > 0)).sum())
            components.append((volume_ml, suv_max, n_voxels, intersection))

        case_stats.append({"gt_voxels": int(gt.sum()), "components": components})

    print(f"\nSweeping {len(VOLUME_THRESHOLDS)}x{len(SUV_THRESHOLDS)} threshold combos "
          f"over {len(case_stats)} non-empty-GT cases (stats-only, no more volume loading)...", flush=True)

    results = []
    for min_vol in VOLUME_THRESHOLDS:
        for min_suv in SUV_THRESHOLDS:
            dices = []
            for case in case_stats:
                pred_voxels = 0
                intersection_voxels = 0
                for volume_ml, suv_max, n_voxels, intersection in case["components"]:
                    if volume_ml < min_vol and suv_max < min_suv:
                        continue  # filtered out
                    pred_voxels += n_voxels
                    intersection_voxels += intersection
                union = pred_voxels + case["gt_voxels"]
                dice = 2 * intersection_voxels / union if union > 0 else 1.0
                dices.append(dice)
            mean_dice = float(np.mean(dices))
            results.append({"min_volume_ml": min_vol, "min_suv": min_suv, "mean_dice": mean_dice})
            print(f"  min_volume_ml={min_vol}, min_suv={min_suv} -> mean_dice={mean_dice:.4f}", flush=True)

    results.sort(key=lambda r: -r["mean_dice"])
    print("\n=== Top 5 threshold combinations ===", flush=True)
    for r in results[:5]:
        print(f"  {r}", flush=True)

    current = next(r for r in results if r["min_volume_ml"] == 0.35 and r["min_suv"] == 6.0)
    print(f"\nCurrent thresholds (0.35mL, SUV 6.0): mean_dice={current['mean_dice']:.4f}", flush=True)
    print(f"Best found: {results[0]}", flush=True)

    import json
    out_path = Path(VOLUME_PATH) / "postprocess_threshold_sweep.json"
    out_path.write_text(json.dumps(results, indent=2))
    volume.commit()
    print(f"\nSaved full results to {out_path}", flush=True)


@app.local_entrypoint()
def main():
    sweep.remote()
