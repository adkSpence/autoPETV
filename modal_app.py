"""
Modal smoke test for the autoPETV nnU-Net baseline container.

Mirrors nnunet-baseline/test.sh: builds the same Docker image the challenge
submission uses, runs it once against the fixture case in test/input, and
pulls the prediction back down into test/output.

Usage:
    modal run modal_app.py
"""
from pathlib import Path

import modal

ROOT = Path(__file__).parent
BASELINE_DIR = ROOT / "nnunet-baseline"
TEST_INPUT = ROOT / "test" / "input"
TEST_OUTPUT = ROOT / "test" / "output"

app = modal.App("autopetv-baseline-smoketest")

image = modal.Image.from_dockerfile(BASELINE_DIR / "Dockerfile.modal", context_dir=BASELINE_DIR)


@app.function(image=image, gpu="T4", timeout=1200)
def run_inference(input_files: dict[str, bytes]) -> dict[str, bytes]:
    import subprocess

    input_root = Path("/input")
    for rel_path, data in input_files.items():
        dest = input_root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    print("Uploaded files landed at:", flush=True)
    for path in sorted(input_root.rglob("*")):
        print(" ", path, flush=True)

    subprocess.run(
        ["python", "-u", "-m", "process"], cwd="/opt/algorithm", check=True
    )

    output_root = Path("/output")
    outputs = {}
    for path in output_root.rglob("*"):
        if path.is_file():
            outputs[str(path.relative_to(output_root))] = path.read_bytes()
    return outputs


@app.local_entrypoint()
def main():
    input_files = {
        str(p.relative_to(TEST_INPUT)): p.read_bytes()
        for p in TEST_INPUT.rglob("*")
        if p.is_file()
    }
    print(f"Uploading {len(input_files)} input files, running remote inference...")

    outputs = run_inference.remote(input_files)

    TEST_OUTPUT.mkdir(parents=True, exist_ok=True)
    for rel_path, data in outputs.items():
        dest = TEST_OUTPUT / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        print(f"Wrote {dest}")

    print("Done.")
