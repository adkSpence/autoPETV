"""
Apply the low-confidence component filter to our baseline prediction, then
score before vs. after with the same metrics.py logic, to see whether the
post-processing step actually helps.
"""
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from metrics import MetricEvaluator, calc_dice
from postprocessing.lesion_filters import filter_low_confidence_components
PRED_PATH = ROOT / "test/output/images/tumor-lesion-segmentation/case_psma_ffcaa75377465b37_2018-03-04_0000.mha"
GT_PATH = ROOT / "test/labels/psma_ffcaa75377465b37_2018-03-04.nii.gz"
PET_PATH = ROOT / "test/images/psma_ffcaa75377465b37_2018-03-04_0001.nii.gz"


def score(pred: np.ndarray, gt: np.ndarray, spacing) -> dict:
    global_dice = calc_dice(pred, np.clip(gt, 0, 1))
    evaluator = MetricEvaluator()
    result = evaluator(
        prediction=pred,
        ground_truth=gt,
        case_name="psma_ffcaa75377465b37",
        spacing=spacing,
        return_meta=True,
    )
    return {
        "dice": global_dice,
        "f1": result["f1"],
        "tp": result["tp"],
        "fp": result["fp"],
        "fn": result["fn"],
    }


def main():
    pred_img = sitk.ReadImage(str(PRED_PATH))
    gt_img = sitk.ReadImage(str(GT_PATH))
    pet_img = sitk.ReadImage(str(PET_PATH))

    pred = sitk.GetArrayFromImage(pred_img).astype(np.uint8)
    gt = sitk.GetArrayFromImage(gt_img).astype(np.uint8)
    pet = sitk.GetArrayFromImage(pet_img)
    spacing = gt_img.GetSpacing()

    before = score(pred, gt, spacing)

    filtered_pred = filter_low_confidence_components(pred, pet, spacing)
    after = score(filtered_pred, gt, spacing)

    print(f"{'metric':>8} {'before':>10} {'after':>10}")
    for key in ["dice", "f1", "tp", "fp", "fn"]:
        b, a = before[key], after[key]
        fmt = "{:>10.4f}" if isinstance(b, float) else "{:>10d}"
        print(f"{key:>8} {fmt.format(b)} {fmt.format(a)}")


if __name__ == "__main__":
    main()
