"""
Preprocess the full 4-channel (CT/PET/FG-scribble/BG-scribble) combined
FDG+PSMA dataset (1614 cases) -- 3d_fullres only, matching the
baseline's own inference config. Same resumable design as
preprocess_full.py: writes dataset.json first, then uses
ResumableDefaultPreprocessor (skips already-done cases, no rmtree) and
explicitly creates gt_segmentations (needed for final validation,
not created by calling the preprocessor directly).

Usage:
    modal run --detach modal_deploy/preprocess_combined.py
"""
import json
import multiprocessing
from pathlib import Path
from time import sleep

import modal
from batchgenerators.utilities.file_and_folder_operations import join, isdir, isfile, maybe_mkdir_p, load_json
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets
from tqdm import tqdm

DATASET_ID = 990
DATASET_NAME = f"Dataset{DATASET_ID}_AutoPETCombined"
VOLUME_PATH = "/vol"
NUM_PROCESSES = 4
CONFIGURATION = "3d_fullres"
PLANS_IDENTIFIER = "nnUNetPlans"


class ResumableDefaultPreprocessor(DefaultPreprocessor):
    def run(self, dataset_name_or_id, configuration_name, plans_identifier, num_processes):
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
        skipped = len(dataset) - len(remaining_keys)
        print(f"Skipping {skipped} already-preprocessed cases, processing {len(remaining_keys)} remaining", flush=True)

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


app = modal.App("autopetv-preprocess-combined")
volume = modal.Volume.from_name("autopetv-combined-data", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install("nnunetv2==2.6.0", "SimpleITK==2.4.1")


@app.function(
    image=image,
    cpu=NUM_PROCESSES,
    memory=32768,
    timeout=23 * 3600,
    volumes={VOLUME_PATH: volume},
    env={
        "nnUNet_raw": str(Path(VOLUME_PATH) / "nnUNet_raw"),
        "nnUNet_preprocessed": str(Path(VOLUME_PATH) / "nnUNet_preprocessed"),
        "nnUNet_results": str(Path(VOLUME_PATH) / "nnUNet_results"),
    },
)
def preprocess():
    import shutil
    import subprocess

    raw_dir = Path(VOLUME_PATH) / "nnUNet_raw" / DATASET_NAME
    images_dir = raw_dir / "imagesTr"
    labels_dir = raw_dir / "labelsTr"

    num_training = len(list(labels_dir.glob("*.nii.gz")))
    dataset_json_path = raw_dir / "dataset.json"
    if not dataset_json_path.exists():
        dataset_json = {
            "channel_names": {"0": "CT", "1": "PET", "2": "FG", "3": "BG"},
            "labels": {"background": 0, "lesion": 1},
            "numTraining": num_training,
            "file_ending": ".nii.gz",
        }
        dataset_json_path.write_text(json.dumps(dataset_json, indent=2))
        print(f"Wrote dataset.json for {num_training} cases", flush=True)

    print("=== Running nnUNetv2_plan_and_preprocess (planning only) ===", flush=True)
    subprocess.run(
        ["nnUNetv2_plan_and_preprocess", "-d", str(DATASET_ID), "--verify_dataset_integrity", "--no_pp"],
        check=True,
    )

    print("=== Running resumable preprocessing (3d_fullres only) ===", flush=True)
    ResumableDefaultPreprocessor(verbose=False).run(DATASET_ID, CONFIGURATION, PLANS_IDENTIFIER, NUM_PROCESSES)

    gt_dir = Path(VOLUME_PATH) / "nnUNet_preprocessed" / DATASET_NAME / "gt_segmentations"
    gt_dir.mkdir(parents=True, exist_ok=True)
    for f in labels_dir.glob("*.nii.gz"):
        shutil.copy(f, gt_dir / f.name)

    volume.commit()
    print("Preprocessing complete.", flush=True)


@app.local_entrypoint()
def main():
    call = preprocess.spawn()
    print(f"Spawned detached call: {call.object_id}")
