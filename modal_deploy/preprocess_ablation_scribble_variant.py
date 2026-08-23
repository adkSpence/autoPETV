"""
Build and preprocess an additional EDT-scribble variant of the ablation
dataset (same 250-case subset, same CT/PET/labels as Dataset992_AblationEdt)
but with a DIFFERENT random seed for scribble generation -- for the
scribble-resampling ablation (task #21): instead of literally regenerating
scribbles every training iteration (measured too slow: ~19s/case with the
full skeletonize+cc3d pipeline, which would add tens of minutes per epoch),
precompute a small set of distinct scribble draws and have the dataloader
randomly pick among them per case per epoch -- near-zero runtime cost,
real epoch-to-epoch variation, just not infinite variety.

Usage:
    modal run --detach modal_deploy/preprocess_ablation_scribble_variant.py --variant 1
    modal run --detach modal_deploy/preprocess_ablation_scribble_variant.py --variant 2
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

VARIANT_DATASET_IDS = {1: 996, 2: 997}

app = modal.App("autopetv-preprocess-ablation-scribble-variant")
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
def preprocess(variant: int):
    import shutil
    import subprocess

    import nibabel as nib
    import numpy as np

    dataset_id = VARIANT_DATASET_IDS[variant]
    dataset_name = f"Dataset{dataset_id}_AblationEdtVariant{variant}"
    seed = 42 + variant * 1000

    subset = json.loads((Path(VOLUME_PATH) / "ablation_subset.json").read_text())
    case_ids = subset["train"] + subset["val"]

    raw_dir = Path(VOLUME_PATH) / "nnUNet_raw" / dataset_name
    images_dir = raw_dir / "imagesTr"
    labels_dir = raw_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    source_images = Path(VOLUME_PATH) / "nnUNet_raw" / SOURCE_DATASET_NAME / "imagesTr"
    source_labels = Path(VOLUME_PATH) / "nnUNet_raw" / SOURCE_DATASET_NAME / "labelsTr"

    def generate_edt(scribble_vol, truncate_distance=10.0):
        from scipy.ndimage import distance_transform_edt
        if not np.any(scribble_vol):
            return np.full(scribble_vol.shape, truncate_distance, dtype=np.float32)
        return np.clip(distance_transform_edt(scribble_vol == 0), 0, truncate_distance).astype(np.float32)

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

    def build_fg_bg_scribbles(data, seed):
        from skimage.morphology import binary_dilation, ball
        labels_fg, comp_ids_fg = get_random_k_components(data, k=5, seed=seed)
        scribble_fg = generate_scribbles_for_components(labels_fg, comp_ids_fg, seed=seed)
        dilated = binary_dilation(data, ball(1))
        dilated = binary_dilation(dilated, ball(1))
        bg_region = ((dilated.astype(np.uint8) - data.astype(np.uint8)) > 0).astype(np.uint8)
        labels_bg, comp_ids_bg = get_random_k_components(bg_region, k=5, seed=seed + 1)
        scribble_bg = generate_scribbles_for_components(labels_bg, comp_ids_bg, seed=seed + 1)
        return scribble_fg, scribble_bg

    print(f"Building raw dataset {dataset_name} ({len(case_ids)} cases, seed={seed})", flush=True)
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

        if np.any(data):
            scribble_fg, scribble_bg = build_fg_bg_scribbles(data, seed=seed + hash(case_id) % 1000)
        else:
            scribble_fg = np.zeros_like(data, dtype=np.uint8)
            scribble_bg = np.zeros_like(data, dtype=np.uint8)

        heatmap_fg = generate_edt(scribble_fg)
        heatmap_bg = generate_edt(scribble_bg)

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
def main(variant: int = 1):
    assert variant in VARIANT_DATASET_IDS
    call = preprocess.spawn(variant)
    print(f"Spawned detached call for variant={variant}: {call.object_id}")
