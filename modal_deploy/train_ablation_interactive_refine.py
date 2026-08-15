"""
Ablation: interaction-aware/corrective-scribble refinement, ported from the
sandbox/interactive_refine_smoketest.py mechanism (validated end-to-end on
MPS locally) to a real nnU-Net trainer on the actual architecture/patch
size, for a genuine measured comparison against the EDT baseline (0.7102
on the 250-case ablation subset).

Same two-round idea, now inside nnU-Net's real training loop via a custom
train_step override (not a custom dataloader -- the model needs to have
already run once per batch to know where its errors are, which the
dataloader has no access to):

  Round 1 (no_grad): forward the batch as-is (scribble channels are the
  usual GT-derived EDT scribbles from preprocessing) -> get a prediction.
  Find where it's wrong: FN voxels (missed lesion) and FP voxels (spurious
  prediction) vs the batch's real GT.

  Round 2 (grad): replace the FG/BG scribble channels (indices 2 and 3 --
  CT=0, PET=1) with EDT-encoded corrective scribbles placed exactly at
  round 1's FN/FP voxels -> forward again -> backprop on this round's loss
  only.

This matches Team LesionLocator's AutoPET IV winning approach in spirit
(distance-transform prompt encoding + simulated clicks during training),
scaled down to what's testable in an ablation run rather than their full
promptable-refinement pipeline.

Cost note: round 1 forces a GPU->CPU sync every training iteration (EDT is
a CPU/scipy operation) -- slower per-epoch than the baseline EDT trainer,
acceptable for a 90-epoch ablation whose purpose is answering "is this
concept worth building out properly," not speed.

Usage:
    modal run --detach modal_deploy/train_ablation_interactive_refine.py
"""
from pathlib import Path

import modal

VOLUME_PATH = "/vol"
DATASET_ID = 992  # Dataset992_AblationEdt -- same EDT-encoded ablation dataset as every other ablation
CONFIGURATION = "3d_fullres"
FOLD = 0
NUM_EPOCHS = 90
TRAINER_NAME = f"nnUNetTrainer_interactive{NUM_EPOCHS}ep"
PLANS_IDENTIFIER = "nnUNetPlans"

app = modal.App("autopetv-train-ablation-interactive-refine")
volume = modal.Volume.from_name("autopetv-combined-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install("nnunetv2==2.6.0", "SimpleITK==2.4.1", "scipy")


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
def train():
    import subprocess
    import nnunetv2

    trainer_code = f'''
import numpy as np
import torch
from torch import autocast
from scipy.ndimage import distance_transform_edt
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context

EDT_TRUNCATE = 10.0
FG_CHANNEL = 2
BG_CHANNEL = 3


def _edt_encode(binary_mask):
    if not np.any(binary_mask):
        return np.full(binary_mask.shape, EDT_TRUNCATE, dtype=np.float32)
    edt = distance_transform_edt(binary_mask == 0)
    return np.clip(edt, 0, EDT_TRUNCATE).astype(np.float32)


class {TRAINER_NAME}(nnUNetTrainer):
    """90-epoch ablation trainer for interaction-aware corrective-scribble
    refinement -- see module docstring in train_ablation_interactive_refine.py
    for the full design rationale."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = {NUM_EPOCHS}
        self.save_every = 10

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # --- round 1: no-grad probe to find this batch's current errors ---
        with torch.no_grad():
            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                output1 = self.network(data)
            logits1 = output1[0] if isinstance(output1, (list, tuple)) else output1
            pred1 = torch.argmax(logits1, dim=1).cpu().numpy()  # (B, D, H, W)

        target0 = target[0] if isinstance(target, list) else target
        gt_np = target0.detach().cpu().numpy()
        if gt_np.ndim == 5:
            gt_np = gt_np[:, 0]  # (B, 1, D, H, W) -> (B, D, H, W)

        data_np = data.detach().cpu().numpy()
        batch_size = data_np.shape[0]
        any_errors = False
        for b in range(batch_size):
            gt_b = gt_np[b] > 0
            pred_b = pred1[b] > 0
            fn_mask = gt_b & (~pred_b)
            fp_mask = (~gt_b) & pred_b
            if not (fn_mask.any() or fp_mask.any()):
                continue  # round 1 already correct on this sample -- leave its scribble channels as-is
            any_errors = True
            data_np[b, FG_CHANNEL] = _edt_encode(fn_mask.astype(np.uint8))
            data_np[b, BG_CHANNEL] = _edt_encode(fp_mask.astype(np.uint8))

        data2 = torch.from_numpy(data_np).to(self.device, non_blocking=True) if any_errors else data

        # --- round 2: real forward+backward, on the corrective-scribble input ---
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output2 = self.network(data2)
            l = self.loss(output2, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {{"loss": l.detach().cpu().numpy()}}
'''

    trainer_dir = Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer" / "variants" / "training_length"
    trainer_dir.mkdir(parents=True, exist_ok=True)
    (trainer_dir / f"{TRAINER_NAME}.py").write_text(trainer_code)

    output_folder = (
        Path(VOLUME_PATH) / "nnUNet_results" / f"Dataset{DATASET_ID}_AblationEdt" /
        f"{TRAINER_NAME}__{PLANS_IDENTIFIER}__{CONFIGURATION}" / f"fold_{FOLD}"
    )
    resume = (output_folder / "checkpoint_latest.pth").exists()
    print(f"Existing checkpoint: {resume}", flush=True)

    cmd = [
        "nnUNetv2_train", str(DATASET_ID), CONFIGURATION, str(FOLD),
        "-tr", TRAINER_NAME, "-p", PLANS_IDENTIFIER,
    ]
    if resume:
        cmd.append("--c")

    print("Starting interactive-refinement ablation training...", flush=True)
    subprocess.run(cmd, check=True)
    print("Training complete.", flush=True)


@app.local_entrypoint()
def main():
    call = train.spawn()
    print(f"Spawned detached call: {call.object_id}")
