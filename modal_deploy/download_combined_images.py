"""
Download CT/PET image pairs for all ~1614 cases in the combined
FDG+PSMA archive, directly onto the Modal Volume via range requests --
same technique as download_full_dataset.py, adapted for this archive's
layout (already imagesTr/_0000,_0001 + labelsTr) and clean case-ID
naming matching what we used for local scribble generation.

Only CT/PET here -- labels and scribble heatmaps are uploaded
separately (upload_labels_and_scribbles.py) since those are small
enough to have been generated/kept locally.

Usage:
    modal run --detach modal_deploy/download_combined_images.py
"""
from pathlib import Path

import modal

DATASET_ID = 990
DATASET_NAME = f"Dataset{DATASET_ID}_AutoPETCombined"
URL = "https://fdat.uni-tuebingen.de/records/6gjsg-zcg93/files/psma-fdg-pet-ct-lesions.zip?download=1"
VOLUME_PATH = "/vol"

app = modal.App("autopetv-download-combined-images")
volume = modal.Volume.from_name("autopetv-combined-data", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install("requests")


@app.function(image=image, timeout=6 * 3600, volumes={VOLUME_PATH: volume})
def download_all():
    import requests
    import zipfile

    class HttpRangeFile:
        def __init__(self, url: str):
            self.url = url
            self.session = requests.Session()
            r = self.session.get(url, headers={"Range": "bytes=0-0"})
            r.raise_for_status()
            self._size = int(r.headers["Content-Range"].split("/")[-1])
            self._pos = 0

        def seekable(self):
            return True

        def seek(self, offset, whence=0):
            if whence == 0:
                self._pos = offset
            elif whence == 1:
                self._pos += offset
            elif whence == 2:
                self._pos = self._size + offset
            return self._pos

        def tell(self):
            return self._pos

        def read(self, size=-1):
            end = (self._size - 1) if (size is None or size < 0) else min(self._pos + size, self._size) - 1
            if self._pos > end:
                return b""
            for attempt in range(3):
                try:
                    r = self.session.get(self.url, headers={"Range": f"bytes={self._pos}-{end}"})
                    r.raise_for_status()
                    data = r.content
                    self._pos += len(data)
                    return data
                except Exception:
                    if attempt == 2:
                        raise

    def clean_case_id(raw_case_id: str) -> str:
        parts = raw_case_id.split("_")
        tracer, patient_hash = parts[0], parts[1]
        date_token = parts[2].split("-NA-")[0] if len(parts) > 2 else ""
        date_compact = date_token.replace("-", "")
        return f"{tracer}_{patient_hash}_{date_compact}" if date_compact else f"{tracer}_{patient_hash}"

    raw_dir = Path(VOLUME_PATH) / "nnUNet_raw" / DATASET_NAME
    (raw_dir / "imagesTr").mkdir(parents=True, exist_ok=True)

    print("Listing remote zip contents...", flush=True)
    f = HttpRangeFile(URL)
    zf = zipfile.ZipFile(f)
    names = zf.namelist()

    label_entries = sorted(n for n in names if n.startswith("labelsTr/") and n.endswith(".nii.gz"))
    print(f"{len(label_entries)} cases found", flush=True)

    for i, label_entry in enumerate(label_entries):
        raw_case_id = Path(label_entry).name.replace(".nii.gz", "")
        case_id = clean_case_id(raw_case_id)

        ct_out = raw_dir / "imagesTr" / f"{case_id}_0000.nii.gz"
        pet_out = raw_dir / "imagesTr" / f"{case_id}_0001.nii.gz"

        if ct_out.exists() and pet_out.exists():
            continue

        ct_entry = f"imagesTr/{raw_case_id}_0000.nii.gz"
        pet_entry = f"imagesTr/{raw_case_id}_0001.nii.gz"

        try:
            ct_out.write_bytes(zf.read(ct_entry))
            pet_out.write_bytes(zf.read(pet_entry))
            if (i + 1) % 50 == 0:
                print(f"[{i+1}/{len(label_entries)}] downloaded so far...", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(label_entries)}] {case_id}: FAILED ({e})", flush=True)

        if (i + 1) % 50 == 0:
            volume.commit()

    volume.commit()
    total = len(list((raw_dir / "imagesTr").glob("*_0000.nii.gz")))
    print(f"Done. {total} cases with CT/PET present.", flush=True)


@app.local_entrypoint()
def main():
    call = download_all.spawn()
    print(f"Spawned detached call: {call.object_id}")
