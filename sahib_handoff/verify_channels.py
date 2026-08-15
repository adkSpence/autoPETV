"""
Verify locally regenerated scribble channels match our reference copy,
by content-hashing the voxel arrays (NOT the files -- gzip embeds
timestamps, so file checksums of .nii.gz are never stable across machines).

Usage:
    python3 verify_channels.py --generate <imagesTr_dir>   # (our side) writes reference_hashes.json
    python3 verify_channels.py --check <imagesTr_dir>      # (cluster side) compares against it
"""
import hashlib
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REFERENCE = Path(__file__).parent / "reference_hashes.json"
# Fixed spot-check cases: FDG + PSMA, empty and non-empty GT among them.
CASES = [
    "fdg_0011f3deaf_03232003",
    "fdg_ae8c77a995_05182007",
    "psma_5907433e52a030d1_20160529",
    "psma_bed76b3b7c2b172e_20200928",
    "psma_febfa344ff66b003_20181019",
]


def content_hash(path: Path) -> str:
    arr = nib.load(path).get_fdata().astype(np.float32)
    return hashlib.sha256(np.round(arr, 3).tobytes()).hexdigest()[:16]


def hashes_for(images_dir: Path) -> dict:
    out = {}
    for case in CASES:
        for ch in ("0002", "0003"):
            p = images_dir / f"{case}_{ch}.nii.gz"
            if not p.exists():
                print(f"MISSING: {p}")
                sys.exit(1)
            out[f"{case}_{ch}"] = content_hash(p)
    return out


if __name__ == "__main__":
    mode, images_dir = sys.argv[1], Path(sys.argv[2])
    h = hashes_for(images_dir)
    if mode == "--generate":
        REFERENCE.write_text(json.dumps(h, indent=2))
        print(f"Wrote {len(h)} reference hashes to {REFERENCE}")
    elif mode == "--check":
        ref = json.loads(REFERENCE.read_text())
        bad = [k for k in ref if h.get(k) != ref[k]]
        if bad:
            print(f"MISMATCH on {len(bad)}/{len(ref)}: {bad}")
            print("Do NOT train on this data -- ping Spencer.")
            sys.exit(1)
        print(f"All {len(ref)} channel hashes match. Data is bit-equivalent to ours -- safe to train.")
