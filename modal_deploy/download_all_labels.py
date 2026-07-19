"""
Download ALL label files (not images) from the combined FDG+PSMA archive,
via range requests -- labels are small enough to pull locally in full,
unlike the CT/PET images which go straight to a Modal Volume instead
(see download_combined_images.py).

Run from repo root: python modal_deploy/download_all_labels.py
"""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from remote_zip_reader import HttpRangeFile

URL = "https://fdat.uni-tuebingen.de/records/6gjsg-zcg93/files/psma-fdg-pet-ct-lesions.zip?download=1"

ROOT = Path(__file__).parent.parent
OUT_LABELS = ROOT / "data_combined" / "labelsTr"


def clean_case_id(raw_case_id: str) -> str:
    parts = raw_case_id.split("_")
    tracer, patient_hash = parts[0], parts[1]
    date_token = parts[2].split("-NA-")[0] if len(parts) > 2 else ""
    date_compact = date_token.replace("-", "")
    return f"{tracer}_{patient_hash}_{date_compact}" if date_compact else f"{tracer}_{patient_hash}"


def main():
    OUT_LABELS.mkdir(parents=True, exist_ok=True)

    print("Opening remote zip (listing only)...")
    f = HttpRangeFile(URL)
    zf = zipfile.ZipFile(f)
    names = zf.namelist()

    label_entries = sorted(n for n in names if n.startswith("labelsTr/") and n.endswith(".nii.gz"))
    print(f"{len(label_entries)} label files found")

    for i, entry in enumerate(label_entries):
        raw_case_id = Path(entry).name.replace(".nii.gz", "")
        case_id = clean_case_id(raw_case_id)
        out_path = OUT_LABELS / f"{case_id}.nii.gz"

        if out_path.exists():
            continue

        for attempt in range(3):
            try:
                out_path.write_bytes(zf.read(entry))
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  retrying {case_id} after error: {e}")

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(label_entries)}] downloaded so far...")

    total = len(list(OUT_LABELS.glob("*.nii.gz")))
    total_mb = sum(p.stat().st_size for p in OUT_LABELS.glob("*.nii.gz")) / 1e6
    print(f"\nDone. {total} label files, {total_mb:.0f} MB total in {OUT_LABELS}")


if __name__ == "__main__":
    main()
