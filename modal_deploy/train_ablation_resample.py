"""
Ablation #6: scribble resampling. Instead of a single static scribble draw
baked into preprocessing, the dataloader randomly picks among 3 precomputed
EDT scribble variants (Dataset992/996/997_AblationEdt*, same CT/PET/labels,
different random scribble draws) per case per training iteration -- a
cheap approximation of "online interaction simulation" (LesionLocator,
arXiv 2508.21680) without the per-iteration regeneration cost, which was
measured too slow to run live (~19s/case with the full skeletonize+cc3d
pipeline).

Usage:
    modal run --detach modal_deploy/train_ablation_resample.py
"""
from pathlib import Path

import modal

VOLUME_PATH = "/vol"
CONFIGURATION = "3d_fullres"
FOLD = 0
NUM_EPOCHS = 90
TRAINER_NAME = f"nnUNetTrainer_resample{NUM_EPOCHS}ep"
PLANS_IDENTIFIER = "nnUNetPlans"

# variant 0 = Dataset992_AblationEdt (already exists), variants 1/2 = the
# two additional scribble draws from preprocess_ablation_scribble_variant.py
VARIANT_DATASETS = {
    0: "Dataset992_AblationEdt",
    1: "Dataset996_AblationEdtVariant1",
    2: "Dataset997_AblationEdtVariant2",
}
PRIMARY_DATASET_ID = 992  # the dataset nnUNetv2_train is invoked against (case IDs/splits identical across variants)

app = modal.App("autopetv-train-ablation-resample")
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
def train():
    import subprocess

    import nnunetv2

    # --- custom dataloader: randomly picks among the 3 variant datasets'
    # preprocessed data (channels 2/3 = FG/BG scribbles differ; CT/PET/seg
    # identical) for each case, each time a training batch is assembled ---
    dataloader_code = f'''
import numpy as np
from pathlib import Path
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd

VARIANT_FOLDERS = {{
    variant: str(Path("{VOLUME_PATH}") / "nnUNet_preprocessed" / name / "nnUNetPlans_3d_fullres")
    for variant, name in {VARIANT_DATASETS!r}.items()
}}


class nnUNetDataLoaderResample(nnUNetDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._variant_datasets = {{
            variant: nnUNetDatasetBlosc2(folder, identifiers=self._data.identifiers)
            for variant, folder in VARIANT_FOLDERS.items()
        }}

    def generate_train_batch(self):
        import torch
        from threadpoolctl import threadpool_limits

        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)

        for j, i in enumerate(selected_keys):
            force_fg = self.get_do_oversample(j)
            variant = np.random.choice(list(self._variant_datasets.keys()))
            data, seg, seg_prev, properties = self._variant_datasets[variant].load_case(i)

            shape = data.shape[1:]
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties["class_locations"])
            bbox = [[lb, ub] for lb, ub in zip(bbox_lbs, bbox_ubs)]

            data_all[j] = crop_and_pad_nd(data, bbox, 0)
            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)[None]))
            seg_all[j] = seg_cropped

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images, segs = [], []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{{"image": data_all[b], "segmentation": seg_all[b]}})
                        images.append(tmp["image"])
                        segs.append(tmp["segmentation"])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images
            return {{"data": data_all, "target": seg_all, "keys": selected_keys}}

        return {{"data": data_all, "target": seg_all, "keys": selected_keys}}
'''

    dataloader_module_dir = Path(nnunetv2.__path__[0]) / "training" / "dataloading"
    (dataloader_module_dir / "data_loader_resample.py").write_text(dataloader_code)

    trainer_code = f'''
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.data_loader_resample import nnUNetDataLoaderResample
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA


class {TRAINER_NAME}(nnUNetTrainer):
    """Short-schedule trainer for the scribble-resampling ablation: the
    training dataloader randomly picks among 3 precomputed scribble
    variants per case per batch (see nnUNetDataLoaderResample); validation
    uses the standard single (variant-0) dataloader, matching real
    inference-time evaluation."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = {NUM_EPOCHS}
        self.save_every = 10

    def get_dataloaders(self):
        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        (rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes
         ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales, is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = nnUNetDataLoaderResample(
            dataset_tr, self.batch_size, initial_patch_size, self.configuration_manager.patch_size,
            self.label_manager, oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling)
        dl_val = nnUNetDataLoader(
            dataset_val, self.batch_size, self.configuration_manager.patch_size,
            self.configuration_manager.patch_size, self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                          num_processes=allowed_num_processes,
                                                          num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                          pin_memory=self.device.type == "cuda", wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val, transform=None,
                                                        num_processes=max(1, allowed_num_processes // 2),
                                                        num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                        pin_memory=self.device.type == "cuda", wait_time=0.002)
        return mt_gen_train, mt_gen_val
'''

    trainer_dir = Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer" / "variants" / "training_length"
    trainer_dir.mkdir(parents=True, exist_ok=True)
    (trainer_dir / f"{TRAINER_NAME}.py").write_text(trainer_code)

    output_folder = (
        Path(VOLUME_PATH) / "nnUNet_results" / VARIANT_DATASETS[0] /
        f"{TRAINER_NAME}__{PLANS_IDENTIFIER}__{CONFIGURATION}" / f"fold_{FOLD}"
    )
    resume = (output_folder / "checkpoint_latest.pth").exists()
    print(f"Existing checkpoint: {resume}", flush=True)

    cmd = [
        "nnUNetv2_train", str(PRIMARY_DATASET_ID), CONFIGURATION, str(FOLD),
        "-tr", TRAINER_NAME, "-p", PLANS_IDENTIFIER,
    ]
    if resume:
        cmd.append("--c")

    print("Starting resample-ablation training...", flush=True)
    subprocess.run(cmd, check=True)
    print("Training complete.", flush=True)


@app.local_entrypoint()
def main():
    call = train.spawn()
    print(f"Spawned detached call: {call.object_id}")
