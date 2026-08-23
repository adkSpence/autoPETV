"""
Stream the full preprocessed Dataset990_AutoPETCombined (125GB) from the
Modal Volume to local disk in small batches, immediately re-uploading each
batch to GCS via the already-authenticated local gcloud session, then
deleting the local copy -- bounded local disk usage (not holding 125GB at
once), resumable (skips files already uploaded to GCS).

Usage (plain Python, not `modal run` -- this orchestrates local gsutil
calls too, not just Modal function calls):
    python3 modal_deploy/relay_to_gcs.py
"""
import subprocess
from pathlib import Path

import modal

VOLUME_PATH = "/vol"
DATASET_NAME = "Dataset990_AutoPETCombined"
GCS_BUCKET = "gs://autopetv-data-transfer/nnUNet_preprocessed/Dataset990_AutoPETCombined"

LOCAL_STAGING = Path("/tmp/relay_staging")

app = modal.App("autopetv-relay-to-gcs")
volume = modal.Volume.from_name("autopetv-combined-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=1800)
def list_files():
    from pathlib import Path
    d = Path(VOLUME_PATH) / "nnUNet_preprocessed" / DATASET_NAME
    files = [str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()]
    return files


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=1800)
def fetch_batch(rel_paths: list[str]):
    d = Path(VOLUME_PATH) / "nnUNet_preprocessed" / DATASET_NAME
    result = {}
    for rel_path in rel_paths:
        result[rel_path] = (d / rel_path).read_bytes()
    return result


def list_already_uploaded() -> set[str]:
    """One bucket listing instead of one gsutil round-trip per file --
    checking 6460 files individually would take way too long."""
    result = subprocess.run(
        ["gsutil", "ls", "-r", GCS_BUCKET],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return set()
    prefix = GCS_BUCKET + "/"
    uploaded = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith(prefix) and not line.endswith("/") and line != GCS_BUCKET + ":":
            uploaded.add(line[len(prefix):])
    return uploaded


def main(batch_size: int = 20):
    print("Listing files on Modal Volume...", flush=True)
    with modal.enable_output():
        with app.run():
            all_files = list_files.remote()
    print(f"Total files: {len(all_files)}", flush=True)

    uploaded = list_already_uploaded()
    remaining = [f for f in all_files if f not in uploaded]
    print(f"Already uploaded: {len(all_files) - len(remaining)}, remaining: {len(remaining)}", flush=True)

    LOCAL_STAGING.mkdir(parents=True, exist_ok=True)

    with modal.enable_output():
        with app.run():
            for i in range(0, len(remaining), batch_size):
                batch = remaining[i:i + batch_size]
                data = fetch_batch.remote(batch)

                for rel_path, content in data.items():
                    local_path = LOCAL_STAGING / rel_path
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    local_path.write_bytes(content)

                # One multi-file parallel upload per batch instead of one
                # gsutil subprocess per file -- avoids per-invocation
                # startup overhead adding up across thousands of files.
                subprocess.run(
                    ["gsutil", "-m", "-q", "cp", "-r", str(LOCAL_STAGING) + "/.", GCS_BUCKET + "/"],
                    check=True,
                )
                for rel_path in data.keys():
                    (LOCAL_STAGING / rel_path).unlink()

                done = min(i + batch_size, len(remaining))
                print(f"  {done}/{len(remaining)} uploaded", flush=True)

    print("Relay complete.", flush=True)


if __name__ == "__main__":
    main()
