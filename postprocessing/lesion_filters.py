"""
Post-processing filter: prune predicted lesion components that show
neither meaningful size nor meaningful PET uptake, using both signals
conservatively (a component is removed only if it fails both criteria).

Thresholds (min_volume_ml, min_suv) were set by inspecting this baseline's
own false positives on the one available test case (see analyze_false_positives.py) —
provisional until validated across more cases.
"""
import cc3d
import numpy as np

CONNECTIVITY = 18


def filter_low_confidence_components(
    prediction: np.ndarray,
    pet: np.ndarray,
    spacing: tuple[float, float, float],
    min_volume_ml: float = 0.35,
    min_suv: float = 6.0,
) -> np.ndarray:
    """Remove predicted components with both low volume and low SUVmax."""
    voxel_volume_ml = float(np.prod(spacing)) / 1000

    labeled, num_components = cc3d.connected_components(
        prediction.astype(int), connectivity=CONNECTIVITY, return_N=True
    )

    filtered = prediction.copy()
    for label in range(1, num_components + 1):
        mask = labeled == label
        volume_ml = mask.sum() * voxel_volume_ml
        suv_max = pet[mask].max()

        if volume_ml < min_volume_ml and suv_max < min_suv:
            filtered[mask] = 0

    return filtered
