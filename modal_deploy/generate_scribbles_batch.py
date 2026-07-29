"""
Generate FG/BG scribble heatmaps (_0002.nii.gz / _0003.nii.gz) for every
downloaded label, in parallel across local CPU cores. Reuses the exact
logic from interactive/simulate_scribbles.py's __main__ block, just
importable and parallelized instead of one subprocess per case.

Uses EDT (Euclidean Distance Transform) encoding, not Gaussian heatmaps --
Team LesionLocator's AutoPET IV winning submission found EDT encoding
consistently outperforms Gaussian (Dice 68.33 -> 76.19+ in their
ablation): Gaussian's low-intensity, near-delta-function voxel values
are poorly captured by the network, whereas EDT gives a dense, smooth
gradient across the whole volume. See interactive/simulate_scribbles.py
for the encoding functions themselves.

Run from repo root: python modal_deploy/generate_scribbles_batch.py
"""
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.morphology import binary_dilation, ball

sys.path.insert(0, str(Path(__file__).parent.parent / "interactive"))
from simulate_scribbles import (
    get_random_k_components,
    generate_scribbles_for_components,
    save_heatmap_nifti,
    generate_edt_from_scribbles,
)

ROOT = Path(__file__).parent.parent
LABELS_DIR = ROOT / "data_combined" / "labelsTr"
OUT_DIR = ROOT / "data_combined" / "imagesTr"
STRATEGY = "centerline"
SEED = 42
EDT_TRUNCATE_DISTANCE = 10.0


def process_one(label_path_str: str) -> str:
    label_path = Path(label_path_str)
    case_id = label_path.name.replace(".nii.gz", "")
    fg_out = OUT_DIR / f"{case_id}_0002.nii.gz"
    bg_out = OUT_DIR / f"{case_id}_0003.nii.gz"

    if fg_out.exists() and bg_out.exists():
        return f"{case_id}: skip"

    try:
        img = nib.load(label_path)
        data = img.get_fdata().astype(np.uint8)

        if np.sum(data) == 0:
            # No lesion at all: nothing to be near, so every voxel is
            # "maximally far" from a (nonexistent) click -- fill with the
            # truncation ceiling rather than zeros, consistent with what
            # generate_edt_from_scribbles returns for an empty volume.
            empty = np.full_like(data, EDT_TRUNCATE_DISTANCE, dtype=np.float32)
            save_heatmap_nifti(empty, str(label_path), str(fg_out))
            save_heatmap_nifti(empty, str(label_path), str(bg_out))
            return f"{case_id}: empty"

        labels_fg, comp_ids_fg = get_random_k_components(data, k=5)
        scribble_fg = generate_scribbles_for_components(labels_fg, comp_ids_fg, STRATEGY, SEED)
        heatmap_fg = generate_edt_from_scribbles(scribble_fg, truncate_distance=EDT_TRUNCATE_DISTANCE)

        dilated = binary_dilation(data, ball(1))
        dilated = binary_dilation(dilated, ball(1))
        bg_region = ((dilated.astype(np.uint8) - data.astype(np.uint8)) > 0).astype(np.uint8)

        labels_bg, comp_ids_bg = get_random_k_components(bg_region, k=5)
        scribble_bg = generate_scribbles_for_components(labels_bg, comp_ids_bg, STRATEGY, SEED)
        heatmap_bg = generate_edt_from_scribbles(scribble_bg, truncate_distance=EDT_TRUNCATE_DISTANCE)

        save_heatmap_nifti(heatmap_fg, str(label_path), str(fg_out))
        save_heatmap_nifti(heatmap_bg, str(label_path), str(bg_out))
        return f"{case_id}: ok"
    except Exception as e:
        return f"{case_id}: FAILED ({e})"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    label_paths = sorted(str(p) for p in LABELS_DIR.glob("*.nii.gz"))
    print(f"{len(label_paths)} labels to process, using {cpu_count()} cores")

    t0 = time.time()
    num_workers = max(1, cpu_count() - 1)
    with Pool(num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_one, label_paths, chunksize=4)):
            if (i + 1) % 50 == 0 or "FAILED" in result:
                print(f"  [{i+1}/{len(label_paths)}] {result}  ({time.time()-t0:.0f}s elapsed)", flush=True)

    print(f"\nDone in {time.time()-t0:.0f}s")
    fg_count = len(list(OUT_DIR.glob("*_0002.nii.gz")))
    bg_count = len(list(OUT_DIR.glob("*_0003.nii.gz")))
    print(f"FG heatmaps: {fg_count}, BG heatmaps: {bg_count}")


if __name__ == "__main__":
    main()
