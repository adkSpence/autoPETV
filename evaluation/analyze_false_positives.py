"""
Inspect what distinguishes true-positive vs false-positive predicted lesion
components for our one test case, using volume and PET SUVmax - the two
usual signals for a post-processing false-positive filter.
"""
import sys
from pathlib import Path

import cc3d
import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from metrics import _get_paired_crop, _calc_overlapping_labels, calc_iou
PRED_PATH = ROOT / "test/output/images/tumor-lesion-segmentation/case_psma_ffcaa75377465b37_2018-03-04_0000.mha"
GT_PATH = ROOT / "test/labels/psma_ffcaa75377465b37_2018-03-04.nii.gz"
PET_PATH = ROOT / "test/images/psma_ffcaa75377465b37_2018-03-04_0001.nii.gz"

OVERLAP_THRESHOLD = 0.1
CONNECTIVITY = 18


def main():
    pred_img = sitk.ReadImage(str(PRED_PATH))
    gt_img = sitk.ReadImage(str(GT_PATH))
    pet_img = sitk.ReadImage(str(PET_PATH))

    pred = sitk.GetArrayFromImage(pred_img).astype(np.uint8)
    gt = sitk.GetArrayFromImage(gt_img).astype(np.uint8)
    pet = sitk.GetArrayFromImage(pet_img)
    spacing = gt_img.GetSpacing()
    voxel_volume_ml = np.prod(spacing) / 1000

    crop = _get_paired_crop(pred, np.clip(gt, 0, 1), 2)
    pred_c, gt_c, pet_c = pred[crop], gt[crop], pet[crop]

    gt_multiclass, _ = cc3d.connected_components(gt_c, connectivity=CONNECTIVITY, return_N=True)
    pred_multiclass, num_pred = cc3d.connected_components(pred_c, connectivity=CONNECTIVITY, return_N=True)

    gt_unique = np.unique(gt_multiclass)
    overlaps = _calc_overlapping_labels(pred_multiclass, gt_multiclass, gt_unique)

    matched_pred = set()
    for gt_label, pred_label in overlaps:
        mask_gt = gt_multiclass == gt_label
        mask_pred = pred_multiclass == pred_label
        if calc_iou(mask_pred, mask_gt) >= OVERLAP_THRESHOLD:
            matched_pred.add(pred_label)

    print(f"{'label':>5} {'status':>6} {'volume_mL':>10} {'suv_max':>8} {'suv_mean':>9}")
    for label in range(1, num_pred + 1):
        mask = pred_multiclass == label
        volume_ml = mask.sum() * voxel_volume_ml
        suv_max = pet_c[mask].max()
        suv_mean = pet_c[mask].mean()
        status = "TP" if label in matched_pred else "FP"
        print(f"{label:>5} {status:>6} {volume_ml:>10.3f} {suv_max:>8.2f} {suv_mean:>9.2f}")


if __name__ == "__main__":
    main()
