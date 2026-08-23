"""
Extend the existing (min_volume_ml, min_suv) postprocessing filter with a
third, organ-aware signal: whether a predicted lesion component overlaps a
physiological-uptake organ (bladder, bowel, heart, liver, kidneys, brain --
the classic FDG/PSMA false-positive sources), per TotalSegmentator masks
precomputed locally and uploaded to postprocess_organs/.

A 5-case prototype showed FPs are ~4.5x more likely to overlap a flagged
organ than TPs (15.6% vs 3.5%) -- real signal, not noise, but not enough
to justify a hard exclusion rule (a real 67%-organ-overlap TP existed too).
This sweeps organ-overlap-fraction thresholds *combined* with the existing
volume/SUV filter (component survives if it passes volume+SUV OR fails
organ-overlap check), same conservative AND-based design as the original.

Follows the same precompute-once-then-sweep-in-memory pattern as
sweep_postprocess_thresholds.py to avoid reloading full volumes per combo.

Usage:
    modal run modal_deploy/sweep_postprocess_organ_aware.py
"""
import modal

DATASET_ID = 990
DATASET_NAME = f"Dataset{DATASET_ID}_AutoPETCombined"
VOLUME_PATH = "/vol"
TRAINER_NAME = "nnUNetTrainer_500ep_freqsave"
PLANS_IDENTIFIER = "nnUNetPlans"
CONFIGURATION = "3d_fullres"
FOLD = 0
ORGAN_DIR = "postprocess_organs"

# Existing best-known thresholds from sweep_postprocess_thresholds.py, held
# fixed here while sweeping the new organ-overlap axis on top of them.
BASE_MIN_VOLUME_ML = 0.35
BASE_MIN_SUV = 6.0

# TotalSegmentator (--fast, "total" task) label ids for classic physiological
# FDG/PSMA uptake organs -- see class_map in totalsegmentator.map_to_binary.
FLAGGED_ORGAN_IDS = {1, 2, 3, 5, 6, 18, 19, 20, 21, 51, 90}

ORGAN_OVERLAP_THRESHOLDS = [1.01, 0.75, 0.5, 0.35, 0.2, 0.1, 0.05]  # 1.01 = never trigger (control)

app = modal.App("autopetv-sweep-postprocess-organ-aware")
volume = modal.Volume.from_name("autopetv-combined-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "nibabel", "numpy", "connected-components-3d", "scipy",
)


@app.function(image=image, cpu=8, memory=16384, timeout=3 * 3600, volumes={VOLUME_PATH: volume})
def sweep():
    from pathlib import Path
    import cc3d
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import zoom

    val_dir = (Path(VOLUME_PATH) / "nnUNet_results" / DATASET_NAME /
               f"{TRAINER_NAME}__{PLANS_IDENTIFIER}__{CONFIGURATION}" / f"fold_{FOLD}" / "validation")
    raw_dir = Path(VOLUME_PATH) / "nnUNet_raw" / DATASET_NAME
    labels_dir = raw_dir / "labelsTr"
    images_dir = raw_dir / "imagesTr"
    organ_dir = Path(VOLUME_PATH) / ORGAN_DIR

    pred_files = sorted(val_dir.glob("*.nii.gz"))
    print(f"Precomputing per-component stats (volume/SUV/organ-overlap) for "
          f"{len(pred_files)} validation cases...", flush=True)

    # Per case: gt_voxel_count, and one row per predicted connected component:
    # (volume_ml, suv_max, n_voxels, intersection_with_gt_voxels, organ_overlap_frac)
    case_stats = []
    skipped_no_organ = 0
    for i, pred_path in enumerate(pred_files):
        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(pred_files)}", flush=True)
        case_id = pred_path.name.replace(".nii.gz", "")
        gt_path = labels_dir / f"{case_id}.nii.gz"
        pet_path = images_dir / f"{case_id}_0001.nii.gz"
        organ_path = organ_dir / f"{case_id}.nii.gz"
        if not (gt_path.exists() and pet_path.exists()):
            continue
        if not organ_path.exists():
            skipped_no_organ += 1
            continue

        pred = nib.load(pred_path).get_fdata().astype(np.uint8)
        gt = nib.load(gt_path).get_fdata().astype(np.uint8)
        if gt.sum() == 0:
            continue  # empty-GT cases would always score 0/undefined Dice regardless of thresholds

        pet_img = nib.load(pet_path)
        pet = pet_img.get_fdata().astype(np.float32)
        spacing = pet_img.header.get_zooms()
        voxel_volume_ml = float(np.prod(spacing)) / 1000

        organ = nib.load(organ_path).get_fdata().astype(int)
        if organ.shape != pred.shape:
            factors = [p / o for p, o in zip(pred.shape, organ.shape)]
            organ = zoom(organ, factors, order=0)
        flagged_mask = np.isin(organ, list(FLAGGED_ORGAN_IDS))

        labeled, num_components = cc3d.connected_components(pred.astype(int), connectivity=18, return_N=True)
        components = []
        for label in range(1, num_components + 1):
            mask = labeled == label
            n_voxels = int(mask.sum())
            volume_ml = n_voxels * voxel_volume_ml
            suv_max = float(pet[mask].max())
            intersection = int((mask & (gt > 0)).sum())
            organ_overlap_frac = float((mask & flagged_mask).sum()) / n_voxels if n_voxels > 0 else 0.0
            components.append((volume_ml, suv_max, n_voxels, intersection, organ_overlap_frac))

        case_stats.append({"gt_voxels": int(gt.sum()), "components": components})

    print(f"Stats ready for {len(case_stats)} cases ({skipped_no_organ} skipped, no organ mask).", flush=True)
    print(f"\nSweeping {len(ORGAN_OVERLAP_THRESHOLDS)} organ-overlap thresholds "
          f"(base filter fixed at {BASE_MIN_VOLUME_ML}mL / SUV {BASE_MIN_SUV})...", flush=True)

    results = []
    for organ_thresh in ORGAN_OVERLAP_THRESHOLDS:
        dices = []
        for case in case_stats:
            pred_voxels = 0
            intersection_voxels = 0
            for volume_ml, suv_max, n_voxels, intersection, organ_frac in case["components"]:
                base_pass = not (volume_ml < BASE_MIN_VOLUME_ML and suv_max < BASE_MIN_SUV)
                organ_flagged = organ_frac >= organ_thresh
                if (not base_pass) or organ_flagged:
                    continue  # filtered out
                pred_voxels += n_voxels
                intersection_voxels += intersection
            union = pred_voxels + case["gt_voxels"]
            dice = 2 * intersection_voxels / union if union > 0 else 1.0
            dices.append(dice)
        mean_dice = float(np.mean(dices))
        results.append({"organ_overlap_threshold": organ_thresh, "mean_dice": mean_dice})
        label = "control (no organ filter)" if organ_thresh > 1.0 else f"organ_overlap>={organ_thresh}"
        print(f"  {label} -> mean_dice={mean_dice:.4f}", flush=True)

    results.sort(key=lambda r: -r["mean_dice"])
    print("\n=== Ranked results ===", flush=True)
    for r in results:
        print(f"  {r}", flush=True)

    control = next(r for r in results if r["organ_overlap_threshold"] > 1.0)
    best = results[0]
    print(f"\nControl (existing volume/SUV filter only): mean_dice={control['mean_dice']:.4f}", flush=True)
    print(f"Best organ-aware combo: {best}", flush=True)
    if best["organ_overlap_threshold"] <= 1.0:
        print(f"Improvement over control: {best['mean_dice'] - control['mean_dice']:+.4f}", flush=True)

    import json
    out_path = Path(VOLUME_PATH) / "postprocess_organ_aware_sweep.json"
    out_path.write_text(json.dumps(results, indent=2))
    volume.commit()
    print(f"\nSaved full results to {out_path}", flush=True)


@app.local_entrypoint()
def main():
    sweep.remote()
