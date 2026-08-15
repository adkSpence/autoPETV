"""
Re-score the existing postprocessing filter sweeps (volume/SUV threshold,
and organ-aware overlap threshold) against a lesion-instance-level
precision/recall/F1 metric, not just voxel Dice.

Rationale: our submitted leaderboard score is Dice 0.7242 / F1 0.7002 --
that F1 is almost certainly a lesion-matching metric, not voxel overlap,
and every ablation/sweep so far has only ever scored mean voxel Dice. The
organ-aware filter sweep found every organ-overlap threshold *hurt* Dice --
but removing a small spurious FP lesion barely moves Dice (tiny volume)
while it's a full precision win under lesion-matching. This script checks
whether that filter actually helps under the metric that plausibly matters
more for the real leaderboard.

Matching criterion: a predicted connected component is a "hit" if it
overlaps >=1 voxel of >=1 GT lesion (any-overlap criterion, the simplest
and most common definition for these challenges); a GT lesion is "detected"
if any surviving predicted component overlaps it. Precision/recall/F1 are
computed micro-averaged (pooled TP/FP/FN across the whole validation set,
not case-averaged) since case-level averaging is unstable for cases with 0-1
lesions.

One data pass covers both existing sweeps (reuses the same case loading as
sweep_postprocess_thresholds.py and sweep_postprocess_organ_aware.py, just
adds GT-lesion-level labeling on top) plus scores mean Dice at the same time
for direct before/after comparison.

Usage:
    modal run modal_deploy/sweep_postprocess_lesion_metric.py
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

BASE_MIN_VOLUME_ML = 0.35
BASE_MIN_SUV = 6.0

FLAGGED_ORGAN_IDS = {1, 2, 3, 5, 6, 18, 19, 20, 21, 51, 90}
ORGAN_OVERLAP_THRESHOLDS = [1.01, 0.75, 0.5, 0.35, 0.2, 0.1, 0.05]  # 1.01 = never trigger (control)

app = modal.App("autopetv-sweep-postprocess-lesion-metric")
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
    print(f"Precomputing per-component + per-GT-lesion stats for {len(pred_files)} cases...", flush=True)

    # Per case: n_gt_lesions, and one row per predicted component:
    # (volume_ml, suv_max, n_voxels, intersection_voxels, organ_overlap_frac, gt_lesion_ids_hit)
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
            continue  # no lesions to detect; excluded from both Dice and lesion-metric scoring

        pet_img = nib.load(pet_path)
        pet = pet_img.get_fdata().astype(np.float32)
        spacing = pet_img.header.get_zooms()
        voxel_volume_ml = float(np.prod(spacing)) / 1000

        organ = nib.load(organ_path).get_fdata().astype(int)
        if organ.shape != pred.shape:
            factors = [p / o for p, o in zip(pred.shape, organ.shape)]
            organ = zoom(organ, factors, order=0)
        flagged_mask = np.isin(organ, list(FLAGGED_ORGAN_IDS))

        gt_labeled, n_gt_lesions = cc3d.connected_components(gt.astype(int), connectivity=18, return_N=True)
        pred_labeled, n_pred_components = cc3d.connected_components(pred.astype(int), connectivity=18, return_N=True)

        components = []
        for label in range(1, n_pred_components + 1):
            mask = pred_labeled == label
            n_voxels = int(mask.sum())
            volume_ml = n_voxels * voxel_volume_ml
            suv_max = float(pet[mask].max())
            intersection = int((mask & (gt > 0)).sum())
            organ_overlap_frac = float((mask & flagged_mask).sum()) / n_voxels if n_voxels > 0 else 0.0
            gt_ids_hit = set(int(x) for x in np.unique(gt_labeled[mask]) if x != 0)
            components.append({
                "volume_ml": volume_ml, "suv_max": suv_max, "n_voxels": n_voxels,
                "intersection": intersection, "organ_frac": organ_overlap_frac, "gt_ids_hit": gt_ids_hit,
            })

        case_stats.append({"gt_voxels": int(gt.sum()), "n_gt_lesions": n_gt_lesions, "components": components})

    print(f"Stats ready for {len(case_stats)} cases ({skipped_no_organ} skipped, no organ mask).", flush=True)

    def score(min_volume_ml, min_suv, organ_thresh):
        total_intersection = 0
        total_pred_voxels = 0
        total_gt_voxels = 0
        tp_components = 0
        fp_components = 0
        total_gt_lesions = 0
        detected_gt_lesions = 0
        dices = []

        for case in case_stats:
            total_gt_voxels += case["gt_voxels"]
            total_gt_lesions += case["n_gt_lesions"]
            hit_ids_this_case = set()
            case_pred_voxels = 0
            case_intersection = 0

            for comp in case["components"]:
                base_pass = not (comp["volume_ml"] < min_volume_ml and comp["suv_max"] < min_suv)
                organ_flagged = comp["organ_frac"] >= organ_thresh
                if (not base_pass) or organ_flagged:
                    continue  # filtered out, doesn't count toward precision/recall/dice at all

                case_pred_voxels += comp["n_voxels"]
                case_intersection += comp["intersection"]
                if comp["gt_ids_hit"]:
                    tp_components += 1
                    hit_ids_this_case |= comp["gt_ids_hit"]
                else:
                    fp_components += 1

            detected_gt_lesions += len(hit_ids_this_case)
            total_pred_voxels += case_pred_voxels
            total_intersection += case_intersection
            union = case_pred_voxels + case["gt_voxels"]
            dices.append(2 * case_intersection / union if union > 0 else 1.0)

        precision = tp_components / (tp_components + fp_components) if (tp_components + fp_components) > 0 else 0.0
        recall = detected_gt_lesions / total_gt_lesions if total_gt_lesions > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_dice = float(np.mean(dices))
        return {"mean_dice": mean_dice, "lesion_precision": precision, "lesion_recall": recall, "lesion_f1": f1,
                "total_gt_lesions": total_gt_lesions, "detected_gt_lesions": detected_gt_lesions,
                "tp_components": tp_components, "fp_components": fp_components}

    print(f"\n=== Volume/SUV threshold sweep (organ filter off), scored on both metrics ===", flush=True)
    volume_thresholds = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]
    suv_thresholds = [0.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    vol_suv_results = []
    for mv in volume_thresholds:
        for ms in suv_thresholds:
            r = score(mv, ms, 1.01)
            r["min_volume_ml"] = mv
            r["min_suv"] = ms
            vol_suv_results.append(r)
    vol_suv_results.sort(key=lambda r: -r["lesion_f1"])
    print("Top 5 by lesion F1:", flush=True)
    for r in vol_suv_results[:5]:
        print(f"  vol={r['min_volume_ml']} suv={r['min_suv']} -> dice={r['mean_dice']:.4f} "
              f"F1={r['lesion_f1']:.4f} (P={r['lesion_precision']:.4f} R={r['lesion_recall']:.4f})", flush=True)
    current = next(r for r in vol_suv_results if r["min_volume_ml"] == 0.35 and r["min_suv"] == 6.0)
    print(f"Current thresholds (0.35mL, SUV 6.0): dice={current['mean_dice']:.4f} F1={current['lesion_f1']:.4f} "
          f"(P={current['lesion_precision']:.4f} R={current['lesion_recall']:.4f})", flush=True)

    print(f"\n=== Organ-overlap sweep (base filter fixed at {BASE_MIN_VOLUME_ML}mL/{BASE_MIN_SUV}SUV) ===", flush=True)
    organ_results = []
    for ot in ORGAN_OVERLAP_THRESHOLDS:
        r = score(BASE_MIN_VOLUME_ML, BASE_MIN_SUV, ot)
        r["organ_overlap_threshold"] = ot
        organ_results.append(r)
        label = "control (no organ filter)" if ot > 1.0 else f"organ_overlap>={ot}"
        print(f"  {label} -> dice={r['mean_dice']:.4f} F1={r['lesion_f1']:.4f} "
              f"(P={r['lesion_precision']:.4f} R={r['lesion_recall']:.4f})", flush=True)

    organ_control = next(r for r in organ_results if r["organ_overlap_threshold"] > 1.0)
    organ_best_f1 = max(organ_results, key=lambda r: r["lesion_f1"])
    print(f"\nOrgan-filter control: dice={organ_control['mean_dice']:.4f} F1={organ_control['lesion_f1']:.4f}", flush=True)
    print(f"Best organ-aware combo by F1: threshold={organ_best_f1['organ_overlap_threshold']}, "
          f"dice={organ_best_f1['mean_dice']:.4f} F1={organ_best_f1['lesion_f1']:.4f} "
          f"(delta F1 vs control: {organ_best_f1['lesion_f1'] - organ_control['lesion_f1']:+.4f}, "
          f"delta dice vs control: {organ_best_f1['mean_dice'] - organ_control['mean_dice']:+.4f})", flush=True)

    import json
    out = {"volume_suv_sweep": vol_suv_results, "organ_aware_sweep": organ_results}
    out_path = Path(VOLUME_PATH) / "postprocess_lesion_metric_sweep.json"
    out_path.write_text(json.dumps(out, indent=2))
    volume.commit()
    print(f"\nSaved full results to {out_path}", flush=True)


@app.local_entrypoint()
def main():
    sweep.remote()
