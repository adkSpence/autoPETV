"""
Post-processing filter: prune predicted lesion components that show
neither meaningful size nor meaningful PET uptake, using both signals
conservatively (a component is removed only if it fails both criteria).

Thresholds (min_volume_ml, min_suv) tuned against real validation
predictions on lesion-instance precision/recall/F1 (not just voxel Dice --
see modal_deploy/sweep_postprocess_lesion_metric.py), since the leaderboard
score reports both Dice and F1 and the latter is a lesion-matching metric.
0.2mL/SUV 10.0 sits in the middle of a cluster of near-equally-good
combinations (F1 ~0.809-0.812 vs 0.8054 for the prior 0.35mL/SUV 6.0
thresholds) to avoid overfitting to a single grid cell.
"""
import cc3d
import numpy as np

CONNECTIVITY = 18


def filter_low_confidence_components(
    prediction: np.ndarray,
    pet: np.ndarray,
    spacing: tuple[float, float, float],
    min_volume_ml: float = 0.2,
    min_suv: float = 10.0,
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
