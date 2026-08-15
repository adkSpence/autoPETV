"""
Build and preprocess a small 4-channel ablation dataset (250 cases, fixed
subset from select_ablation_subset.py) for one scribble-encoding variant.
Reuses CT/PET/labels directly from the full combined dataset (Dataset990)
-- only the FG/BG channels differ per encoding, generated fresh here from
the ground-truth label (and CT intensity, for the geodesic variant).

Encodings: edt (reference/current), gaussian (real sigma>0, not the
degenerate sigma=0 baseline), disk (fixed-radius binary blob), geodesic
(adaptive geodesic-Gaussian, arXiv 2303.06942).

Usage:
    modal run --detach modal_deploy/preprocess_ablation.py --encoding edt
    modal run --detach modal_deploy/preprocess_ablation.py --encoding gaussian
    modal run --detach modal_deploy/preprocess_ablation.py --encoding disk
    modal run --detach modal_deploy/preprocess_ablation.py --encoding geodesic
"""
import json
import multiprocessing
from pathlib import Path
from time import sleep

import modal
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor

SOURCE_DATASET_ID = 990
SOURCE_DATASET_NAME = f"Dataset{SOURCE_DATASET_ID}_AutoPETCombined"
VOLUME_PATH = "/vol"
NUM_PROCESSES = 4
CONFIGURATION = "3d_fullres"
PLANS_IDENTIFIER = "nnUNetPlans"

DATASET_IDS = {"edt": 992, "gaussian": 993, "disk": 994, "geodesic": 995}

app = modal.App("autopetv-preprocess-ablation")
volume = modal.Volume.from_name("autopetv-combined-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "nnunetv2==2.6.0", "SimpleITK==2.4.1", "nibabel", "scipy", "scikit-image", "connected-components-3d",
)


def _resumable_preprocessor_run(self, dataset_name_or_id, configuration_name, plans_identifier, num_processes):
    from batchgenerators.utilities.file_and_folder_operations import join, isdir, isfile, maybe_mkdir_p, load_json
    from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets
    from tqdm import tqdm

    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    assert isdir(join(nnUNet_raw, dataset_name))

    plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + ".json")
    plans_manager = PlansManager(load_json(plans_file))
    configuration_manager = plans_manager.get_configuration(configuration_name)
    dataset_json = load_json(join(nnUNet_preprocessed, dataset_name, "dataset.json"))

    output_directory = join(nnUNet_preprocessed, dataset_name, configuration_manager.data_identifier)
    maybe_mkdir_p(output_directory)

    dataset = get_filenames_of_train_images_and_targets(join(nnUNet_raw, dataset_name), dataset_json)
    remaining_keys = [k for k in dataset.keys() if not isfile(join(output_directory, k) + ".pkl")]
    print(f"Skipping {len(dataset) - len(remaining_keys)} done, processing {len(remaining_keys)}", flush=True)
    if not remaining_keys:
        return

    r = []
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        remaining = list(range(len(remaining_keys)))
        workers = [j for j in p._pool]
        for k in remaining_keys:
            r.append(p.starmap_async(
                self.run_case_save,
                ((join(output_directory, k), dataset[k]["images"], dataset[k]["label"],
                  plans_manager, configuration_manager, dataset_json),)
            ))
        with tqdm(desc=None, total=len(remaining_keys)) as pbar:
            while len(remaining) > 0:
                if not all(j.is_alive() for j in workers):
                    raise RuntimeError("A background worker died (likely OOM).")
                done = [i for i in remaining if r[i].ready()]
                for i in done:
                    r[i].get()
                    pbar.update()
                remaining = [i for i in remaining if i not in done]
                sleep(0.1)


class ResumableDefaultPreprocessor(DefaultPreprocessor):
    run = _resumable_preprocessor_run


@app.function(
    image=image,
    cpu=NUM_PROCESSES,
    memory=16384,
    timeout=6 * 3600,
    volumes={VOLUME_PATH: volume},
    env={
        "nnUNet_raw": str(Path(VOLUME_PATH) / "nnUNet_raw"),
        "nnUNet_preprocessed": str(Path(VOLUME_PATH) / "nnUNet_preprocessed"),
        "nnUNet_results": str(Path(VOLUME_PATH) / "nnUNet_results"),
    },
)
def preprocess(encoding: str):
    import shutil
    import subprocess

    import nibabel as nib
    import numpy as np

    dataset_id = DATASET_IDS[encoding]
    dataset_name = f"Dataset{dataset_id}_Ablation{encoding.capitalize()}"

    subset = json.loads((Path(VOLUME_PATH) / "ablation_subset.json").read_text())
    case_ids = subset["train"] + subset["val"]

    raw_dir = Path(VOLUME_PATH) / "nnUNet_raw" / dataset_name
    images_dir = raw_dir / "imagesTr"
    labels_dir = raw_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    source_images = Path(VOLUME_PATH) / "nnUNet_raw" / SOURCE_DATASET_NAME / "imagesTr"
    source_labels = Path(VOLUME_PATH) / "nnUNet_raw" / SOURCE_DATASET_NAME / "labelsTr"

    # --- inline encoding functions (avoids importing the full
    # interactive/simulate_scribbles.py dependency graph into this image) ---
    def generate_edt(scribble_vol, truncate_distance=10.0):
        from scipy.ndimage import distance_transform_edt
        if not np.any(scribble_vol):
            return np.full(scribble_vol.shape, truncate_distance, dtype=np.float32)
        return np.clip(distance_transform_edt(scribble_vol == 0), 0, truncate_distance).astype(np.float32)

    def generate_gaussian(scribble_vol, sigma=3.0):
        from scipy.ndimage import gaussian_filter
        if not np.any(scribble_vol):
            return np.zeros(scribble_vol.shape, dtype=np.float32)
        return gaussian_filter(scribble_vol.astype(np.float32), sigma=sigma)

    def generate_disk(scribble_vol, radius=3):
        from skimage.morphology import binary_dilation, ball
        if not np.any(scribble_vol):
            return np.zeros(scribble_vol.shape, dtype=np.float32)
        return binary_dilation(scribble_vol.astype(bool), ball(radius)).astype(np.float32)

    def generate_geodesic_gaussian(scribble_vol, intensity_vol, sigma=5.0, intensity_weight=1.0, num_passes=2):
        if not np.any(scribble_vol):
            return np.zeros(scribble_vol.shape, dtype=np.float32)
        inf = 1e6
        dist = np.where(scribble_vol > 0, 0.0, inf).astype(np.float32)
        intensity = intensity_vol.astype(np.float32)
        offsets = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        for _ in range(num_passes):
            for dx, dy, dz in offsets:
                shifted_dist = np.roll(dist, shift=(dx, dy, dz), axis=(0, 1, 2))
                shifted_intensity = np.roll(intensity, shift=(dx, dy, dz), axis=(0, 1, 2))
                step_cost = 1.0 + intensity_weight * np.abs(intensity - shifted_intensity)
                dist = np.minimum(dist, shifted_dist + step_cost)
        return np.exp(-(dist ** 2) / (2 * sigma ** 2)).astype(np.float32)

    def get_random_k_components(label, k=5, seed=42):
        import cc3d
        import random
        labels = cc3d.connected_components(label, connectivity=26)
        unique = np.unique(labels)
        unique = unique[unique != 0]
        if len(unique) == 0:
            return labels, []
        random.seed(seed)
        return labels, random.sample(list(unique), min(k, len(unique)))

    def generate_scribbles_for_components(labels, component_ids, seed=42):
        from skimage.morphology import skeletonize
        import cc3d
        import networkx as nx
        from scipy.spatial.distance import cdist
        scribble_vol = np.zeros_like(labels, dtype=np.uint8)
        for cid in component_ids:
            comp_mask = (labels == cid).astype(np.uint8)
            slice_sums = comp_mask.sum(axis=(0, 1))
            if slice_sums.max() == 0:
                continue
            best_slice = slice_sums.argmax()
            slice_mask = comp_mask[:, :, best_slice]
            try:
                skeleton = skeletonize(slice_mask).astype(np.uint8)
                skel_cc = cc3d.connected_components(skeleton, connectivity=8)
                unique, counts = np.unique(skel_cc, return_counts=True)
                counts_dict = dict(zip(unique, counts))
                counts_dict.pop(0, None)
                if not counts_dict:
                    scribble_vol[:, :, best_slice] += slice_mask
                    continue
                largest = max(counts_dict, key=counts_dict.get)
                scribble_vol[:, :, best_slice] += (skel_cc == largest).astype(np.uint8)
            except Exception:
                scribble_vol[:, :, best_slice] += slice_mask
        return (scribble_vol > 0).astype(np.uint8)

    def build_fg_bg_scribbles(data):
        from skimage.morphology import binary_dilation, ball
        labels_fg, comp_ids_fg = get_random_k_components(data, k=5)
        scribble_fg = generate_scribbles_for_components(labels_fg, comp_ids_fg)
        dilated = binary_dilation(data, ball(1))
        dilated = binary_dilation(dilated, ball(1))
        bg_region = ((dilated.astype(np.uint8) - data.astype(np.uint8)) > 0).astype(np.uint8)
        labels_bg, comp_ids_bg = get_random_k_components(bg_region, k=5)
        scribble_bg = generate_scribbles_for_components(labels_bg, comp_ids_bg)
        return scribble_fg, scribble_bg

    def encode(scribble_vol, ct_vol, empty_value):
        if encoding == "edt":
            return generate_edt(scribble_vol) if np.any(scribble_vol) else np.full(scribble_vol.shape, 10.0, dtype=np.float32)
        elif encoding == "gaussian":
            return generate_gaussian(scribble_vol)
        elif encoding == "disk":
            return generate_disk(scribble_vol)
        elif encoding == "geodesic":
            return generate_geodesic_gaussian(scribble_vol, ct_vol)
        raise ValueError(encoding)

    print(f"Building raw dataset {dataset_name} ({len(case_ids)} cases, encoding={encoding})", flush=True)
    for i, case_id in enumerate(case_ids):
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(case_ids)}", flush=True)

        ct_out = images_dir / f"{case_id}_0000.nii.gz"
        pet_out = images_dir / f"{case_id}_0001.nii.gz"
        fg_out = images_dir / f"{case_id}_0002.nii.gz"
        bg_out = images_dir / f"{case_id}_0003.nii.gz"
        label_out = labels_dir / f"{case_id}.nii.gz"

        if fg_out.exists() and bg_out.exists():
            continue

        shutil.copy(source_images / f"{case_id}_0000.nii.gz", ct_out)
        shutil.copy(source_images / f"{case_id}_0001.nii.gz", pet_out)
        shutil.copy(source_labels / f"{case_id}.nii.gz", label_out)

        label_img = nib.load(label_out)
        data = label_img.get_fdata().astype(np.uint8)
        ct_vol = nib.load(ct_out).get_fdata() if encoding == "geodesic" else None

        if np.any(data):
            scribble_fg, scribble_bg = build_fg_bg_scribbles(data)
        else:
            scribble_fg = np.zeros_like(data, dtype=np.uint8)
            scribble_bg = np.zeros_like(data, dtype=np.uint8)

        heatmap_fg = encode(scribble_fg, ct_vol, empty_value=True)
        heatmap_bg = encode(scribble_bg, ct_vol, empty_value=True)

        for heatmap, out_path in [(heatmap_fg, fg_out), (heatmap_bg, bg_out)]:
            out_img = nib.Nifti1Image(heatmap.astype(np.float32), label_img.affine, label_img.header)
            nib.save(out_img, out_path)

    dataset_json_path = raw_dir / "dataset.json"
    dataset_json = {
        "channel_names": {"0": "CT", "1": "PET", "2": "FG", "3": "BG"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(case_ids),
        "file_ending": ".nii.gz",
    }
    dataset_json_path.write_text(json.dumps(dataset_json, indent=2))

    splits = [{"train": subset["train"], "val": subset["val"]}] * 5
    preprocessed_dir = Path(VOLUME_PATH) / "nnUNet_preprocessed" / dataset_name
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    (preprocessed_dir / "splits_final.json").write_text(json.dumps(splits, indent=2))

    print("=== Planning (no preprocessing yet) ===", flush=True)
    subprocess.run(
        ["nnUNetv2_plan_and_preprocess", "-d", str(dataset_id), "--verify_dataset_integrity", "--no_pp"],
        check=True,
    )

    print("=== Preprocessing (resumable) ===", flush=True)
    ResumableDefaultPreprocessor(verbose=False).run(dataset_id, CONFIGURATION, PLANS_IDENTIFIER, NUM_PROCESSES)

    gt_dir = preprocessed_dir / "gt_segmentations"
    gt_dir.mkdir(parents=True, exist_ok=True)
    for f in labels_dir.glob("*.nii.gz"):
        shutil.copy(f, gt_dir / f.name)

    volume.commit()
    print(f"Done: {dataset_name}", flush=True)


@app.local_entrypoint()
def main(encoding: str = "edt"):
    assert encoding in DATASET_IDS, f"encoding must be one of {list(DATASET_IDS)}"
    call = preprocess.spawn(encoding)
    print(f"Spawned detached call for encoding={encoding}: {call.object_id}")
