# Volyagers — autoPET V challenge submission

Interactive whole-body PET/CT lesion segmentation for the
[autoPET V challenge](https://autopet-v.grand-challenge.org).

Submitted algorithm: **autoPETV EDT+** — a 3D residual-encoder U-Net
trained with round supervision, taking CT, PET and two click-guidance
channels as input.

| Metric | Preliminary test set |
|---|---|
| Dice | **0.7702** |
| Lesion-level F1 | **0.7558** |

Single model, single fold, no ensembling, no test-time augmentation.

## The submitted model

| | |
|---|---|
| Dataset | `Dataset994_AutoPETInteractiveFull` (1611 studies) |
| Input | 4 channels: CT, PET, foreground guidance, background guidance |
| Architecture | `ResidualEncoderUNet`, 6 stages, features `[32,64,128,256,320,320]` |
| Plans | `nnUNetResEncUNetMPlans`, patch `112x160x128`, batch 2 |
| Trainer | `nnUNetTrainerResEnc4ChannelRoundSupervisionEDT` |
| Checkpoint | `checkpoint_deploy.pth` (epoch 950, SHA-256 pinned) |
| Training | 1x NVIDIA H200, ~97 h wall-clock |

### Click encoding

Clicks are rendered into two channels as a **spacing-aware local
Euclidean distance transform**: value 1 at the click, decaying linearly
to 0 at a 40 mm support radius, computed in millimetres so it is
invariant to voxel spacing. The channels are passed through the network
unnormalised.

This must match between training and inference. An earlier submission
used an inverted encoding (0 at the click, rising with distance, in
voxels) and scored **0.40** with identical weights; correcting the
encoding alone raised it to 0.7702. See `nnunet-baseline/utils.py`
(`generate_edt_heatmap`).

## Repository layout

```
nnunet-baseline/          submission container (build + package scripts)
  process.py              entrypoint: /input -> predict -> /output
  utils.py                click -> guidance channel encoding
  postprocess_filters.py  false-positive component filter
  check_weights.sh        verifies checkpoint presence + SHA-256
  package_model_weights.sh -> nnUNet_results.tar.gz  (Models page)
  export.sh                -> container tarball       (Container page)
interactive/              model weights, evaluation loop
  nnUNet_results/         Dataset994 checkpoints (git-lfs)
  interactive_loop.py     multi-round evaluation protocol
sahib_handoff/            scripts for training folds at scale
paper/                    LNCS manuscript
```

## Reproducing the submission

Requires `git-lfs` and Docker with GPU support.

```bash
git clone https://github.com/adkSpence/autoPETV.git
cd autoPETV
git lfs pull

cd nnunet-baseline
bash check_weights.sh            # verifies the deploy checkpoint's SHA-256
bash package_model_weights.sh    # -> nnUNet_results.tar.gz
bash export.sh                   # -> container tarball
bash test.sh                     # runs the container on the sample case
```

The weights are uploaded separately on Grand Challenge (Algorithm →
Models) rather than baked into the image; Grand Challenge extracts them
to `/opt/ml/model/`, matching `ENV nnUNet_results` in the Dockerfile.

## Post-processing

Predicted connected components (18-connectivity) are removed only if they
fail **both** a volume criterion (< 0.35 mL) and an uptake criterion
(SUV_max < 6.0). Requiring both avoids discarding small but metabolically
active lesions.

## Data

Challenge data only; no external data and no pre-trained weights. The raw
data is not redistributed here — obtain it from the challenge organisers.
If you use it, cite:

- Gatidis S., Hepp T., Früh M., et al. *A whole-body FDG-PET/CT dataset
  with manually annotated tumor lesions.* Sci Data 9, 601 (2022).
  [doi:10.1038/s41597-022-01718-3](https://doi.org/10.1038/s41597-022-01718-3)
- Gatidis S., Kuestner T. *FDG-PET-CT-Lesions* [Dataset]. TCIA (2022).
  [doi:10.7937/gkr0-xv29](https://doi.org/10.7937/gkr0-xv29)
- Jeblick K., et al. *PSMA-PET-CT-Lesions* (Version 1) [Dataset]. TCIA
  (2024). [doi:10.7937/r7ep-3x37](https://doi.org/10.7937/r7ep-3x37)

## Notes

`nnunet-baseline/` began as a fork of the organisers' baseline
([lab-midas/autoPETV](https://github.com/lab-midas/autoPETV)); the sample
test case under `test/` comes from that repository.

Some directories (`modal_deploy/`, `sandbox/`, `nnunet-baseline-2ch/`)
contain development and ablation code that is not part of the submission
and is kept for transparency rather than reuse.
