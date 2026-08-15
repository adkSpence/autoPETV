"""
Cheap, CPU-only data quality audit of the full 1614-case combined dataset:
CT/PET shape and spacing mismatches (a registration-corruption proxy, since
AutoPET's distributed CT/PET are supposed to already share a grid),
degenerate/anomalous volumes, intensity outliers, and the empty-GT ratio.
No GPU, no training -- just reads headers/arrays and reports facts, no
speculation about what it "should" find.

Usage:
    modal run modal_deploy/audit_data_quality.py
"""
import modal

VOLUME_PATH = "/vol"
DATASET_NAME = "Dataset990_AutoPETCombined"

app = modal.App("autopetv-audit-data-quality")
volume = modal.Volume.from_name("autopetv-combined-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install("nibabel", "numpy")


@app.function(image=image, cpu=8, memory=16384, timeout=3 * 3600, volumes={VOLUME_PATH: volume})
def audit():
    from pathlib import Path
    import json
    import nibabel as nib
    import numpy as np

    raw_dir = Path(VOLUME_PATH) / "nnUNet_raw" / DATASET_NAME
    images_dir = raw_dir / "imagesTr"
    labels_dir = raw_dir / "labelsTr"

    case_ids = sorted(p.name.replace(".nii.gz", "") for p in labels_dir.glob("*.nii.gz"))
    print(f"Auditing {len(case_ids)} cases...", flush=True)

    findings = {
        "shape_mismatch_ct_pet": [],
        "shape_mismatch_label_ct": [],
        "spacing_mismatch_ct_pet": [],
        "degenerate_volume": [],       # any dimension <= 2
        "all_zero_ct": [],
        "all_zero_pet": [],
        "extreme_ct_hu": [],           # min < -2000 or max > 5000
        "extreme_pet_suv": [],         # negative values or max > 100
        "empty_gt": [],
        "load_errors": [],
    }

    SPACING_TOL = 0.05  # relative tolerance

    for i, case_id in enumerate(case_ids):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(case_ids)}", flush=True)

        try:
            ct_path = images_dir / f"{case_id}_0000.nii.gz"
            pet_path = images_dir / f"{case_id}_0001.nii.gz"
            label_path = labels_dir / f"{case_id}.nii.gz"

            ct_img = nib.load(ct_path)
            pet_img = nib.load(pet_path)
            label_img = nib.load(label_path)

            ct_shape, pet_shape, label_shape = ct_img.shape, pet_img.shape, label_img.shape
            ct_spacing = np.array(ct_img.header.get_zooms())
            pet_spacing = np.array(pet_img.header.get_zooms())

            if ct_shape != pet_shape:
                findings["shape_mismatch_ct_pet"].append({"case": case_id, "ct": ct_shape, "pet": pet_shape})

            if label_shape != ct_shape:
                findings["shape_mismatch_label_ct"].append({"case": case_id, "label": label_shape, "ct": ct_shape})

            if len(ct_spacing) == len(pet_spacing) == 3:
                rel_diff = np.abs(ct_spacing - pet_spacing) / np.maximum(ct_spacing, 1e-6)
                if np.any(rel_diff > SPACING_TOL):
                    findings["spacing_mismatch_ct_pet"].append({
                        "case": case_id, "ct_spacing": ct_spacing.tolist(), "pet_spacing": pet_spacing.tolist(),
                    })

            if any(d <= 2 for d in ct_shape) or any(d <= 2 for d in pet_shape):
                findings["degenerate_volume"].append({"case": case_id, "ct_shape": ct_shape, "pet_shape": pet_shape})

            ct_data = ct_img.get_fdata()
            pet_data = pet_img.get_fdata()
            gt_data = label_img.get_fdata()

            if not np.any(ct_data):
                findings["all_zero_ct"].append(case_id)
            if not np.any(pet_data):
                findings["all_zero_pet"].append(case_id)

            ct_min, ct_max = float(ct_data.min()), float(ct_data.max())
            if ct_min < -2000 or ct_max > 5000:
                findings["extreme_ct_hu"].append({"case": case_id, "min": ct_min, "max": ct_max})

            pet_min, pet_max = float(pet_data.min()), float(pet_data.max())
            if pet_min < 0 or pet_max > 100:
                findings["extreme_pet_suv"].append({"case": case_id, "min": pet_min, "max": pet_max})

            if not np.any(gt_data):
                findings["empty_gt"].append(case_id)

        except Exception as e:
            findings["load_errors"].append({"case": case_id, "error": str(e)})

    print("\n=== Audit summary (facts only) ===", flush=True)
    for key, items in findings.items():
        print(f"{key}: {len(items)}", flush=True)

    empty_gt_pct = 100 * len(findings["empty_gt"]) / len(case_ids)
    print(f"\nEmpty-GT ratio: {len(findings['empty_gt'])}/{len(case_ids)} ({empty_gt_pct:.1f}%)", flush=True)

    # Show a few concrete examples for anything non-empty, so this isn't just counts
    for key in ["shape_mismatch_ct_pet", "spacing_mismatch_ct_pet", "degenerate_volume",
                "all_zero_ct", "all_zero_pet", "extreme_ct_hu", "extreme_pet_suv", "load_errors"]:
        if findings[key]:
            print(f"\nExamples of {key}:", flush=True)
            for item in findings[key][:5]:
                print(f"  {item}", flush=True)

    out_path = Path(VOLUME_PATH) / "data_quality_audit.json"
    out_path.write_text(json.dumps(findings, indent=2, default=str))
    volume.commit()
    print(f"\nFull results saved to {out_path}", flush=True)


@app.local_entrypoint()
def main():
    audit.remote()
