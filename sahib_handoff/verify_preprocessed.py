#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--plans", default="nnUNetResEncUNetLPlans.json")
    args = parser.parse_args()

    dataset_json = json.loads((args.dataset / "dataset.json").read_text())
    channels = dataset_json.get("channel_names", dataset_json.get("modality", {}))
    if len(channels) != 4:
        raise SystemExit(f"FATAL: dataset.json declares {len(channels)} channels, expected 4")

    plans = json.loads((args.dataset / args.plans).read_text())
    configuration = plans["configurations"]["3d_fullres"]
    data_dir = args.dataset / configuration["data_identifier"]
    cases = sorted(data_dir.glob("*.npz"))
    if not cases:
        raise SystemExit(f"FATAL: no preprocessed cases in {data_dir}")

    with np.load(cases[0]) as case:
        shape = case["data"].shape
    if shape[0] != 4:
        raise SystemExit(f"FATAL: {cases[0].name} has {shape[0]} image channels, expected 4")
    print(f"4-channel preflight OK: {len(cases)} cases in {data_dir}")


if __name__ == "__main__":
    main()
