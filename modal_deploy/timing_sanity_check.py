"""
Timing/resource sanity check for the real submission container's
inference path, under the exact AutoPET V resource constraints (single
A10G, 24GB VRAM, 8 CPUs, 30GB memory, ~15-20 min/case). Not the full
6-step eval -- just confirms one real case fits the limits and completes
in time, using the actual EDT-trained checkpoint and the same
nnUNetv2_predict + postprocessing path process.py uses.

Usage:
    modal run modal_deploy/timing_sanity_check.py
"""
import time
from pathlib import Path

import modal

DATASET_ID = 990
DATASET_NAME = f"Dataset{DATASET_ID}_AutoPETCombined"
VOLUME_PATH = "/vol"
TRAINER_NAME = "nnUNetTrainer_500ep_freqsave"
PLANS_IDENTIFIER = "nnUNetPlans"
CONFIGURATION = "3d_fullres"
FOLD = 0

CUSTOM_TRAINER_CODE = '''
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_500ep_freqsave(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500
        self.save_every = 1
'''

app = modal.App("autopetv-timing-sanity-check")
volume = modal.Volume.from_name("autopetv-combined-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "nnunetv2==2.6.0", "SimpleITK==2.4.1", "nibabel", "connected-components-3d",
)


@app.function(
    image=image,
    gpu="T4",
    cpu=8,
    memory=30 * 1024,  # 30GB, matches the grand-challenge algorithm resource cap
    timeout=1800,
    volumes={VOLUME_PATH: volume},
    env={
        "nnUNet_raw": str(Path(VOLUME_PATH) / "nnUNet_raw"),
        "nnUNet_preprocessed": str(Path(VOLUME_PATH) / "nnUNet_preprocessed"),
        "nnUNet_results": str(Path(VOLUME_PATH) / "nnUNet_results"),
    },
)
def timed_predict(case_id: str = None):
    import json
    import subprocess
    import shutil
    import threading

    import cc3d
    import nibabel as nib
    import numpy as np

    # torch.cuda stats in this (parent) process don't see the subprocess's
    # GPU memory -- nnUNetv2_predict runs in its own process. Poll
    # nvidia-smi instead, which reports true device-wide usage.
    peak_vram_mb = [0]
    stop_polling = threading.Event()

    def poll_vram():
        while not stop_polling.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=True,
                ).stdout.strip()
                used_mb = int(out.splitlines()[0])
                peak_vram_mb[0] = max(peak_vram_mb[0], used_mb)
            except Exception:
                pass
            stop_polling.wait(0.5)

    def install_custom_trainer():
        import nnunetv2
        trainer_dir = Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer" / "variants" / "training_length"
        trainer_dir.mkdir(parents=True, exist_ok=True)
        (trainer_dir / f"{TRAINER_NAME}.py").write_text(CUSTOM_TRAINER_CODE)

    def filter_low_confidence_components(prediction, pet, spacing, min_volume_ml=0.2, min_suv=10.0):
        voxel_volume_ml = float(np.prod(spacing)) / 1000
        labeled, num_components = cc3d.connected_components(prediction.astype(int), connectivity=18, return_N=True)
        filtered = prediction.copy()
        for label in range(1, num_components + 1):
            mask = labeled == label
            volume_ml = mask.sum() * voxel_volume_ml
            suv_max = pet[mask].max()
            if volume_ml < min_volume_ml and suv_max < min_suv:
                filtered[mask] = 0
        return filtered

    install_custom_trainer()

    preprocessed_dir = Path(VOLUME_PATH) / "nnUNet_preprocessed" / DATASET_NAME
    raw_dir = Path(VOLUME_PATH) / "nnUNet_raw" / DATASET_NAME
    images_tr = raw_dir / "imagesTr"

    if case_id is None:
        splits = json.loads((preprocessed_dir / "splits_final.json").read_text())
        case_id = splits[FOLD]["val"][0]

    work_dir = Path("/tmp/timing_check")
    in_dir = work_dir / "input"
    out_dir = work_dir / "output"
    in_dir.mkdir(parents=True, exist_ok=True)

    ct_path = images_tr / f"{case_id}_0000.nii.gz"
    pet_path = images_tr / f"{case_id}_0001.nii.gz"
    fg_path = images_tr / f"{case_id}_0002.nii.gz"
    bg_path = images_tr / f"{case_id}_0003.nii.gz"

    for src, name in [(ct_path, "0000"), (pet_path, "0001"), (fg_path, "0002"), (bg_path, "0003")]:
        shutil.copy(src, in_dir / f"{case_id}_{name}.nii.gz")

    print(f"Timing case {case_id}...", flush=True)
    t0 = time.time()

    poll_thread = threading.Thread(target=poll_vram, daemon=True)
    poll_thread.start()

    subprocess.run(
        [
            "nnUNetv2_predict", "-i", str(in_dir), "-o", str(out_dir),
            "-d", str(DATASET_ID), "-c", CONFIGURATION, "-f", str(FOLD),
            "-tr", TRAINER_NAME, "-p", PLANS_IDENTIFIER,
        ],
        check=True,
    )
    t_predict = time.time()

    stop_polling.set()
    poll_thread.join()

    pred_path = out_dir / f"{case_id}.nii.gz"
    pred_img = nib.load(pred_path)
    pred = pred_img.get_fdata().astype(np.uint8)
    pet_img = nib.load(pet_path)
    pet_arr = pet_img.get_fdata().astype(np.float32)
    spacing = pet_img.header.get_zooms()

    filtered = filter_low_confidence_components(pred, pet_arr, spacing)
    t_postprocess = time.time()

    peak_vram_gb = peak_vram_mb[0] / 1024

    print(f"\n=== Timing results for {case_id} (T4) ===", flush=True)
    print(f"Predict time:      {t_predict - t0:.1f}s", flush=True)
    print(f"Postprocess time:  {t_postprocess - t_predict:.1f}s", flush=True)
    print(f"Total time:        {t_postprocess - t0:.1f}s", flush=True)
    print(f"Peak GPU memory:   {peak_vram_gb:.2f} GB (polled via nvidia-smi, T4 has 16GB)", flush=True)
    print(f"Image shape:       {pred.shape}", flush=True)
    print(f"Prediction voxels: {int(pred.sum())} -> {int(filtered.sum())} after postprocessing", flush=True)

    return {
        "case_id": case_id,
        "predict_seconds": t_predict - t0,
        "total_seconds": t_postprocess - t0,
        "peak_vram_gb": peak_vram_gb,
    }


@app.local_entrypoint()
def main(case_id: str = None):
    result = timed_predict.remote(case_id)
    print(result)
