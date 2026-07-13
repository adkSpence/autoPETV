"""
Download all 900 FDG-PET-CT-Lesions cases directly onto the Modal Volume,
using range requests against the remote zip so we never route ~90GB through local disk or local bandwidth.

Usage:
    modal run --detach modal_deploy/download_full_dataset.py
"""
import json
from pathlib import Path

import modal

DATASET_ID = 996
DATASET_NAME = f"Dataset{DATASET_ID}_AutoPETFull"
URL = "https://fdat.uni-tuebingen.de/records/wf9fy-txq84/files/fdg-pet-ct-lesions.zip?download=1"
VOLUME_PATH = "/vol"

app = modal.App("autopetv-download-full")
volume = modal.Volume.from_name("autopetv-train-data", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install("requests")


@app.function(image=image, timeout=6 * 3600, volumes={VOLUME_PATH: volume})
def download_all():
    import requests

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
            r = self.session.get(self.url, headers={"Range": f"bytes={self._pos}-{end}"})
            r.raise_for_status()
            data = r.content
            self._pos += len(data)
            return data

    import zipfile

    raw_dir = Path(VOLUME_PATH) / "nnUNet_raw" / DATASET_NAME
    (raw_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (raw_dir / "labelsTr").mkdir(parents=True, exist_ok=True)

    print("Listing remote zip contents...", flush=True)
    f = HttpRangeFile(URL)
    zf = zipfile.ZipFile(f)
    names = zf.namelist()

    patients = sorted(set(
        n.split("/")[1] for n in names
        if n.startswith("FDG-PET-CT-Lesions/PETCT_") and "/" in n[len("FDG-PET-CT-Lesions/"):]
    ))
    print(f"{len(patients)} patients found", flush=True)

    for i, patient in enumerate(patients):
        case_id = patient.replace("PETCT_", "fdg_")
        ct_out = raw_dir / "imagesTr" / f"{case_id}_0000.nii.gz"
        pet_out = raw_dir / "imagesTr" / f"{case_id}_0001.nii.gz"
        seg_out = raw_dir / "labelsTr" / f"{case_id}.nii.gz"

        if ct_out.exists() and pet_out.exists() and seg_out.exists():
            print(f"[{i+1}/{len(patients)}] {case_id}: already present, skipping", flush=True)
            continue

        study_files = sorted(n for n in names if f"/{patient}/" in n and n.endswith(".nii.gz"))
        if not study_files:
            print(f"[{i+1}/{len(patients)}] {case_id}: no files found, skipping", flush=True)
            continue
        study_prefix = study_files[0].rsplit("/", 1)[0]

        try:
            ct_out.write_bytes(zf.read(f"{study_prefix}/CTres.nii.gz"))
            pet_out.write_bytes(zf.read(f"{study_prefix}/SUV.nii.gz"))
            seg_out.write_bytes(zf.read(f"{study_prefix}/SEG.nii.gz"))
            print(f"[{i+1}/{len(patients)}] {case_id}: downloaded", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(patients)}] {case_id}: FAILED ({e})", flush=True)

        if (i + 1) % 20 == 0:
            volume.commit()

    dataset_json = {
        "channel_names": {"0": "CT", "1": "PET"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(list((raw_dir / "labelsTr").glob("*.nii.gz"))),
        "file_ending": ".nii.gz",
    }
    (raw_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    volume.commit()
    print("Done. dataset.json:", dataset_json, flush=True)


@app.local_entrypoint()
def main():

    call = download_all.spawn()
    print(f"Spawned detached call: {call.object_id}")
