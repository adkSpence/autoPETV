"""
Score the baseline's prediction on the single bundled test/ fixture case
against its ground-truth label, using the challenge's own metrics.py.

This is a single-case smoke test, not a real evaluation set

Usage:
    python evaluation/eval_baseline_case.py
"""
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from metrics import MetricEvaluator, calc_dice
PRED_PATH = ROOT / "test/output/images/tumor-lesion-segmentation/case_psma_ffcaa75377465b37_2018-03-04_0000.mha"
GT_PATH = ROOT / "test/labels/psma_ffcaa75377465b37_2018-03-04.nii.gz"


def main():
    pred_img = sitk.ReadImage(str(PRED_PATH))
    gt_img = sitk.ReadImage(str(GT_PATH))

    pred = sitk.GetArrayFromImage(pred_img).astype(np.uint8)
    gt = sitk.GetArrayFromImage(gt_img).astype(np.uint8)

    global_dice = calc_dice(pred, np.clip(gt, 0, 1))

    evaluator = MetricEvaluator()
    result = evaluator(
        prediction=pred,
        ground_truth=gt,
        case_name="psma_ffcaa75377465b37",
        spacing=gt_img.GetSpacing(),
        return_meta=True,
    )

    print(f"Global voxel-wise Dice: {global_dice:.4f}")
    print(f"Instance-level F1:      {result['f1']:.4f}")
    print(f"TP / FP / FN:           {result['tp']} / {result['fp']} / {result['fn']}")
    print(f"GT lesions / predicted: {result['num_gt_instances']} / {result['num_pred_instances']}")
    print(f"FPV (mL):               {result.get('fpv'):.4f}")
    print(f"FNV (mL):               {result.get('fnv'):.4f}")


if __name__ == "__main__":
    main()
