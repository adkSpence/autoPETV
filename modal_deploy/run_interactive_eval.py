"""
Modal-hosted port of interactive/interactive_loop.py's 6-step interactive
correction loop -- the actual AutoPET V scoring protocol (1 initial
no-scribble prediction + up to 5 corrective rounds, scored via AUC-Dice /
AUC-F1 across iterations). The original script assumes a local Docker
container with GPU passthrough (calls `bash nnunet-baseline/test.sh`),
which we don't have; this calls nnUNetv2_predict directly instead, against
a held-out subset of the training split's fold-0 validation cases (never
seen during training), on the EDT-trained combined model.

Self-contained on purpose: everything needed (scribble simulation, EDT
encoding, dice/f1 metrics, postprocess filter) is inlined below rather than
imported from interactive/simulate_scribbles.py, metrics.py,
nnunet-baseline/postprocess_filters.py -- those files aren't available
inside the Modal container, and porting the import graph (cc3d, networkx,
skimage) file-by-file is more fragile than duplicating the ~150 lines
actually used here. Logic is a straight copy from those files.

Usage:
    modal run modal_deploy/run_interactive_eval.py --num-cases 8
"""
import json
from pathlib import Path

import modal

DATASET_ID = 990
DATASET_NAME = f"Dataset{DATASET_ID}_AutoPETCombined"
VOLUME_PATH = "/vol"
TRAINER_NAME = "nnUNetTrainer_500ep_freqsave"
PLANS_IDENTIFIER = "nnUNetPlans"
CONFIGURATION = "3d_fullres"
FOLD = 0
MAX_ITERS = 6  # iteration 0 (no scribbles) + 5 corrective rounds
EDT_TRUNCATE_DISTANCE = 10.0

CUSTOM_TRAINER_CODE = '''
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_500ep_freqsave(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500
        self.save_every = 1
'''

app = modal.App("autopetv-interactive-eval")
volume = modal.Volume.from_name("autopetv-combined-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "nnunetv2==2.6.0",
    "SimpleITK==2.4.1",
    "nibabel",
    "scipy",
    "scikit-image",
    "cc3d",
    "networkx",
)


@app.function(
    image=image,
    gpu="A10",
    cpu=4,
    memory=32768,
    timeout=6 * 3600,
    volumes={VOLUME_PATH: volume},
    env={
        "nnUNet_raw": str(Path(VOLUME_PATH) / "nnUNet_raw"),
        "nnUNet_preprocessed": str(Path(VOLUME_PATH) / "nnUNet_preprocessed"),
        "nnUNet_results": str(Path(VOLUME_PATH) / "nnUNet_results"),
    },
)
def run_eval(num_cases: int = 8, seed: int = 42):
    import random
    import subprocess

    import cc3d
    import nibabel as nib
    import numpy as np
    import networkx as nx
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize
    from skimage.segmentation import find_boundaries
    from scipy.spatial.distance import cdist

    # -------------------------------------------------------------------
    # Inlined from interactive/simulate_scribbles.py
    # -------------------------------------------------------------------
    def scribble_centerline(slice_mask, trunc_fraction=0.1):
        skeleton = skeletonize(slice_mask).astype(np.uint8)
        skel_cc = cc3d.connected_components(skeleton, connectivity=8)
        unique, counts = np.unique(skel_cc, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        counts_dict.pop(0, None)
        if len(counts_dict) == 0:
            return slice_mask.copy()
        largest = max(counts_dict, key=counts_dict.get)
        skeleton = (skel_cc == largest).astype(np.uint8)
        coords = np.argwhere(skeleton)
        if len(coords) < 2:
            return slice_mask.copy()
        G = nx.Graph()
        for y, x in coords:
            G.add_node((y, x))
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx_ = y + dy, x + dx
                    if 0 <= ny < skeleton.shape[0] and 0 <= nx_ < skeleton.shape[1] and skeleton[ny, nx_]:
                        G.add_edge((y, x), (ny, nx_))
        dist_matrix = cdist(coords, coords)
        idx = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)
        p1, p2 = tuple(coords[idx[0]]), tuple(coords[idx[1]])
        try:
            path = nx.shortest_path(G, source=p1, target=p2)
        except Exception:
            return slice_mask.copy()
        path_coords = np.array(path)
        if len(path_coords) > 10:
            n = len(path_coords)
            start, end = int(n * trunc_fraction), int(n * (1 - trunc_fraction))
            path_coords = path_coords[start:end]
        scribble = np.zeros_like(slice_mask)
        for y, x in path_coords:
            if slice_mask[y, x]:
                scribble[y, x] = 1
        return scribble

    def simulate_scribble_from_label(label_array, seed=42):
        """Best-slice centerline scribble on the largest 2D component. Returns
        (coords_xyz list, size)."""
        best_slice, best_component, best_area = None, None, 0
        for z in range(label_array.shape[2]):
            slice_mask = label_array[:, :, z]
            if np.sum(slice_mask) == 0:
                continue
            labels_2d = cc3d.connected_components(slice_mask, connectivity=8)
            unique, counts = np.unique(labels_2d, return_counts=True)
            counts_dict = dict(zip(unique, counts))
            counts_dict.pop(0, None)
            if not counts_dict:
                continue
            largest_label = max(counts_dict, key=counts_dict.get)
            area = counts_dict[largest_label]
            if area > best_area:
                best_area = area
                best_slice = z
                best_component = (labels_2d == largest_label).astype(np.uint8)
        if best_component is None:
            return [], 0
        scribble_slice = scribble_centerline(best_component)
        scribble_vol = np.zeros_like(label_array, dtype=np.uint8)
        scribble_vol[:, :, best_slice] = scribble_slice
        coords = np.argwhere(scribble_vol > 0)
        coords_xyz = [[int(c[0]), int(c[1]), int(c[2])] for c in coords]
        return coords_xyz, int(np.sum(scribble_vol))

    def generate_edt_from_scribbles(scribble_vol, truncate_distance=EDT_TRUNCATE_DISTANCE):
        if not np.any(scribble_vol):
            return np.full(scribble_vol.shape, truncate_distance, dtype=np.float32)
        inverse_mask = scribble_vol == 0
        edt = distance_transform_edt(inverse_mask)
        return np.clip(edt, 0, truncate_distance).astype(np.float32)

    def save_heatmap_nifti(heatmap, reference_nifti_path, output_path):
        ref = nib.load(reference_nifti_path)
        out = nib.Nifti1Image(heatmap.astype(np.float32), ref.affine, ref.header)
        nib.save(out, output_path)

    # -------------------------------------------------------------------
    # Inlined from metrics.py (dice + lesion-level F1)
    # -------------------------------------------------------------------
    def calc_dice(prediction, ground_truth):
        if ground_truth.sum() == 0:
            return float("nan")
        intersection = (ground_truth * prediction).sum()
        union = ground_truth.sum() + prediction.sum()
        return 2 * intersection / union

    def calc_f1(prediction, ground_truth, overlap_threshold=0.1, connectivity=18):
        gt_cc, num_gt = cc3d.connected_components(ground_truth.astype(int), connectivity=connectivity, return_N=True)
        pred_cc, num_pred = cc3d.connected_components(prediction.astype(int), connectivity=connectivity, return_N=True)
        if num_gt == 0:
            return float("nan")
        matched_gt, matched_pred = set(), set()
        gt_labels = np.unique(gt_cc)
        gt_labels = gt_labels[gt_labels != 0]
        for i in gt_labels:
            mask_gt = gt_cc == i
            overlapping_pred = np.unique(pred_cc[mask_gt])
            overlapping_pred = overlapping_pred[overlapping_pred != 0]
            for j in overlapping_pred:
                mask_pred = pred_cc == j
                intersection = (mask_gt & mask_pred).sum()
                union = mask_gt.sum() + mask_pred.sum() - intersection
                iou = intersection / union if union > 0 else 0.0
                if iou >= overlap_threshold:
                    matched_gt.add(i)
                    matched_pred.add(j)
        tp = len(matched_gt)
        fn = num_gt - tp
        fp = num_pred - len(matched_pred)
        if tp == 0:
            return 0.0
        return (2 * tp) / (2 * tp + fp + fn)

    # -------------------------------------------------------------------
    # Inlined from nnunet-baseline/postprocess_filters.py
    # -------------------------------------------------------------------
    def filter_low_confidence_components(prediction, pet, spacing, min_volume_ml=0.35, min_suv=6.0):
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

    # -------------------------------------------------------------------
    # nnU-Net inference helper
    # -------------------------------------------------------------------
    def install_custom_trainer():
        import nnunetv2
        trainer_dir = Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer" / "variants" / "training_length"
        trainer_dir.mkdir(parents=True, exist_ok=True)
        (trainer_dir / f"{TRAINER_NAME}.py").write_text(CUSTOM_TRAINER_CODE)

    def run_predict(in_dir, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "nnUNetv2_predict", "-i", str(in_dir), "-o", str(out_dir),
                "-d", str(DATASET_ID), "-c", CONFIGURATION, "-f", str(FOLD),
                "-tr", TRAINER_NAME, "-p", PLANS_IDENTIFIER, "--disable_tta",
            ],
            check=True,
        )

    # -------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------
    install_custom_trainer()

    preprocessed_dir = Path(VOLUME_PATH) / "nnUNet_preprocessed" / DATASET_NAME
    raw_dir = Path(VOLUME_PATH) / "nnUNet_raw" / DATASET_NAME
    images_tr = raw_dir / "imagesTr"
    labels_tr = raw_dir / "labelsTr"

    splits = json.loads((preprocessed_dir / "splits_final.json").read_text())
    val_cases = splits[FOLD]["val"]
    random.seed(seed)
    non_empty_cases = []
    for case_id in val_cases:
        label_path = labels_tr / f"{case_id}.nii.gz"
        if np.any(nib.load(label_path).get_fdata()):
            non_empty_cases.append(case_id)
        if len(non_empty_cases) >= num_cases:
            break

    print(f"Evaluating {len(non_empty_cases)} held-out (non-empty-GT) cases: {non_empty_cases}", flush=True)

    work_dir = Path("/tmp/eval")
    results = {}

    for case_id in non_empty_cases:
        print(f"\n=== Case {case_id} ===", flush=True)
        case_dir = work_dir / case_id
        in_dir = case_dir / "input"
        out_dir = case_dir / "output"
        in_dir.mkdir(parents=True, exist_ok=True)

        ct_path = images_tr / f"{case_id}_0000.nii.gz"
        pet_path = images_tr / f"{case_id}_0001.nii.gz"
        label_path = labels_tr / f"{case_id}.nii.gz"

        ct_img = nib.load(ct_path)
        pet_img = nib.load(pet_path)
        gt = nib.load(label_path).get_fdata().astype(np.uint8)
        pet_arr = pet_img.get_fdata().astype(np.float32)
        spacing = pet_img.header.get_zooms()
        shape = gt.shape

        # Copy CT/PET into the per-case working input dir (fixed across iters)
        import shutil
        shutil.copy(ct_path, in_dir / f"{case_id}_0000.nii.gz")
        shutil.copy(pet_path, in_dir / f"{case_id}_0001.nii.gz")

        fg_mask = np.zeros(shape, dtype=np.uint8)
        bg_mask = np.zeros(shape, dtype=np.uint8)
        pred = None
        case_records = []

        for it in range(MAX_ITERS):
            if it > 0:
                overseg = ((pred == 1) & (gt == 0)).astype(np.uint8)
                underseg = ((pred == 0) & (gt == 1)).astype(np.uint8)

                bg_coords, fp_size = ([], 0) if not np.any(overseg) else simulate_scribble_from_label(overseg, seed)
                fg_coords, fn_size = ([], 0) if not np.any(underseg) else simulate_scribble_from_label(underseg, seed)

                if fp_size <= fn_size and fg_coords:
                    for x, y, z in fg_coords:
                        fg_mask[x, y, z] = 1
                elif bg_coords:
                    for x, y, z in bg_coords:
                        bg_mask[x, y, z] = 1

            fg_edt = generate_edt_from_scribbles(fg_mask)
            bg_edt = generate_edt_from_scribbles(bg_mask)
            save_heatmap_nifti(fg_edt, str(ct_path), str(in_dir / f"{case_id}_0002.nii.gz"))
            save_heatmap_nifti(bg_edt, str(ct_path), str(in_dir / f"{case_id}_0003.nii.gz"))

            run_predict(in_dir, out_dir)

            pred_path = out_dir / f"{case_id}.nii.gz"
            pred = nib.load(pred_path).get_fdata().astype(np.uint8)
            pred = filter_low_confidence_components(pred, pet_arr, spacing)

            dice = calc_dice(pred, gt)
            f1 = calc_f1(pred, gt)
            case_records.append({"iteration": it, "dice": float(dice), "f1": float(f1)})
            print(f"  iter {it}: dice={dice:.4f} f1={f1:.4f}", flush=True)

        results[case_id] = case_records
        shutil.rmtree(case_dir, ignore_errors=True)

        results_path = work_dir / "interactive_eval_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(results, indent=2))

    # -------------------------------------------------------------------
    # AUC-Dice / AUC-F1 per case + overall average
    # -------------------------------------------------------------------
    auc_results = {}
    for case_id, records in results.items():
        iterations = np.array([r["iteration"] for r in records], dtype=float)
        dice = np.array([r["dice"] for r in records], dtype=float)
        f1 = np.array([r["f1"] for r in records], dtype=float)
        auc_results[case_id] = {
            "auc_dice": float(np.trapz(dice, iterations) / (iterations[-1] - iterations[0])),
            "auc_f1": float(np.trapz(f1, iterations) / (iterations[-1] - iterations[0])),
        }

    mean_auc_dice = float(np.mean([v["auc_dice"] for v in auc_results.values()]))
    mean_auc_f1 = float(np.mean([v["auc_f1"] for v in auc_results.values()]))

    print("\n=== Per-case AUC ===", flush=True)
    for case_id, v in auc_results.items():
        print(f"  {case_id}: auc_dice={v['auc_dice']:.4f} auc_f1={v['auc_f1']:.4f}", flush=True)
    print(f"\nMean AUC-Dice: {mean_auc_dice:.4f}", flush=True)
    print(f"Mean AUC-F1:   {mean_auc_f1:.4f}", flush=True)

    out_path = Path(VOLUME_PATH) / "interactive_eval_results.json"
    out_path.write_text(json.dumps({"per_case_scores": results, "per_case_auc": auc_results,
                                     "mean_auc_dice": mean_auc_dice, "mean_auc_f1": mean_auc_f1}, indent=2))
    volume.commit()
    print(f"Saved to {out_path}", flush=True)

    return {"mean_auc_dice": mean_auc_dice, "mean_auc_f1": mean_auc_f1}


@app.local_entrypoint()
def main(num_cases: int = 8):
    result = run_eval.remote(num_cases=num_cases)
    print(result)
