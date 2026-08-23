"""
Short training run for one ablation variant (architecture or scribble
encoding), on the fixed 250-case subset. ~90 epochs -- enough for a
reliable relative ranking between variants, not full convergence (mirrors
Sahib's fast-screening approach: cheap, short, parallel-ish comparison
before committing to one expensive full retrain).

Covers ablations 1-5 (architecture: default vs ResEncL planner; encoding:
edt/gaussian/disk/geodesic). Ablation 6 (per-epoch scribble resampling)
needs a custom dataloader override and is handled separately.

Usage:
    modal run --detach modal_deploy/train_ablation.py --encoding edt --planner default
    modal run --detach modal_deploy/train_ablation.py --encoding edt --planner resencl
    modal run --detach modal_deploy/train_ablation.py --encoding gaussian --planner default
    modal run --detach modal_deploy/train_ablation.py --encoding disk --planner default
    modal run --detach modal_deploy/train_ablation.py --encoding geodesic --planner default
"""
from pathlib import Path

import modal

VOLUME_PATH = "/vol"
CONFIGURATION = "3d_fullres"
FOLD = 0
NUM_EPOCHS = 90
TRAINER_NAME = f"nnUNetTrainer_ablation{NUM_EPOCHS}ep"

DATASET_IDS = {"edt": 992, "gaussian": 993, "disk": 994, "geodesic": 995}

CUSTOM_TRAINER_CODE = f'''
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class {TRAINER_NAME}(nnUNetTrainer):
    """Short-schedule trainer for ablation screening -- its own calibrated
    {NUM_EPOCHS}-epoch LR decay, not a truncated 500-epoch run."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = {NUM_EPOCHS}
        self.save_every = 10
'''

app = modal.App("autopetv-train-ablation")
volume = modal.Volume.from_name("autopetv-combined-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install("nnunetv2==2.6.0", "SimpleITK==2.4.1")


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=16384,
    timeout=6 * 3600,
    volumes={VOLUME_PATH: volume},
    env={
        "nnUNet_raw": str(Path(VOLUME_PATH) / "nnUNet_raw"),
        "nnUNet_preprocessed": str(Path(VOLUME_PATH) / "nnUNet_preprocessed"),
        "nnUNet_results": str(Path(VOLUME_PATH) / "nnUNet_results"),
    },
)
def train(encoding: str, planner: str):
    import subprocess

    import nnunetv2

    dataset_id = DATASET_IDS[encoding]
    dataset_name = f"Dataset{dataset_id}_Ablation{encoding.capitalize()}"

    trainer_dir = Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer" / "variants" / "training_length"
    trainer_dir.mkdir(parents=True, exist_ok=True)
    (trainer_dir / f"{TRAINER_NAME}.py").write_text(CUSTOM_TRAINER_CODE)

    if planner == "resencl":
        plans_identifier = "nnUNetResEncUNetLPlans"
        plans_path = Path(VOLUME_PATH) / "nnUNet_preprocessed" / dataset_name / f"{plans_identifier}.json"
        if not plans_path.exists():
            print("=== Planning with ResEncL planner ===", flush=True)
            subprocess.run(
                ["nnUNetv2_plan_and_preprocess", "-d", str(dataset_id), "-pl", "nnUNetPlannerResEncL", "--no_pp"],
                check=True,
            )
            print("=== Preprocessing (ResEncL plans) ===", flush=True)
            subprocess.run(
                ["nnUNetv2_preprocess", "-d", str(dataset_id), "-plans_name", plans_identifier, "-c", CONFIGURATION],
                check=True,
            )
    else:
        plans_identifier = "nnUNetPlans"

    run_name = f"{encoding}_{planner}"
    output_folder = (
        Path(VOLUME_PATH) / "nnUNet_results" / dataset_name /
        f"{TRAINER_NAME}__{plans_identifier}__{CONFIGURATION}" / f"fold_{FOLD}"
    )
    resume = (output_folder / "checkpoint_latest.pth").exists()
    print(f"[{run_name}] Existing checkpoint: {resume}", flush=True)

    cmd = [
        "nnUNetv2_train", str(dataset_id), CONFIGURATION, str(FOLD),
        "-tr", TRAINER_NAME, "-p", plans_identifier,
    ]
    if resume:
        cmd.append("--c")

    print(f"[{run_name}] Starting training...", flush=True)
    subprocess.run(cmd, check=True)
    print(f"[{run_name}] Training complete.", flush=True)


@app.local_entrypoint()
def main(encoding: str = "edt", planner: str = "default", gpu: str = "T4"):
    assert encoding in DATASET_IDS
    assert planner in ("default", "resencl")
    fn = train if gpu == "T4" else train.with_options(gpu=gpu)
    call = fn.spawn(encoding, planner)
    print(f"Spawned detached call for {encoding}/{planner} on {gpu}: {call.object_id}")
