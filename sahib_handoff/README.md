# AutoPET V — ResEncL scale runs (handoff)

Goal: train nnU-Net **ResEncL** folds at **1000 epochs** on the full combined
FDG+PSMA dataset (Dataset990, 1614 cases, 4 channels: CT / PET / FG-scribble-EDT
/ BG-scribble-EDT). Priority order: **fold 2, fold 3, fold 4** (folds 0/1 exist
as plain-UNet 500ep models on our side; re-runs of 0/1 at ResEncL-1000 are
stretch goals if allocation allows).

Why ResEncL: measured on our fixed 250-case ablation (identical protocol,
full-volume validation): ResEncL Dice **0.7506** / lesion-F1 **0.8723** vs
plain UNet 0.7102 / 0.8214.

## Hardware requirements (per fold / per job)
- 1× GPU with **≥ 24 GB VRAM** (ResEncL plans target 24 GB; a 16 GB card OOMs — verified)
- ~64 GB system RAM recommended (data-aug workers), 8+ CPU cores
- **~200 GB free disk** (preprocessed dataset ~130 GB + checkpoints/logs)
- Rough per-fold wall clock: A100 ~1.5–2 days; L4/4090-class ~2.5–3.5 days.
  Folds are fully independent → run in parallel on separate GPUs if available.

## Steps
```bash
./01_setup_env.sh          # venv + nnunetv2==2.6.0 + env vars (source it thereafter)
./02_pull_data.sh          # pulls preprocessed ResEncL dataset from our GCS bucket
./03_train_fold.sh 2       # one fold per GPU/job; auto-resumes if interrupted
./04_push_results.sh 2     # sends back checkpoints + validation summary
```

Every script is idempotent: re-running is always safe. `03_train_fold.sh`
detects an existing `checkpoint_latest.pth` and resumes with `--c`
automatically — a killed/preempted job just needs the same command again.

## What we need back (per fold)
`04_push_results.sh N` uploads exactly these to the bucket:
- `checkpoint_final.pth` and `checkpoint_best.pth`
- `validation/summary.json` (the real Mean Validation Dice)
- training logs

## Invariants — do not change
- `-p nnUNetResEncUNetLPlans` and the stock trainer (its default IS 1000 epochs;
  no `-tr` flag anywhere)
- The shipped `splits_final.json` must be used untouched (fold definitions must
  match ours exactly or ensembling/comparability breaks)
- nnunetv2 pinned to 2.6.0 (matches the plans/preprocessing exactly)

## Before you start — three answers we need
1. **VRAM + concurrency**: how much VRAM per card, and how many cards can you
   hold for ~2 weeks? ResEncL **requires ≥ 24 GB** — a 16 GB card OOMs
   (verified). If you only have 16 GB cards, stop here and tell us; the
   architecture choice changes.
2. **Egress**: can your compute/login nodes reach the internet (pull ~130 GB
   from Modal's CLI or a GCS bucket)?
3. **Data path preference** (pick one):
   - **A (default)**: pull our verified, ready-made preprocessed data —
     `02_pull_data.sh` (GCS) or `modal volume get` (we invite you to the
     workspace; usually faster).
   - **B**: you already have the AutoPET V raw release (QIBA-aligned, Apr 26,
     FDG+PSMA) on-cluster → we send only our scribble-generation scripts +
     `splits_final.json` + plans (KBs, deterministic given seeds) and you
     preprocess locally; we cross-check a few case checksums before you train.

## Access
The GCS bucket is `gs://autopetv-data-transfer`. Send us the email of the
Google account you're authenticated as (`gcloud auth list`) and/or the email
to invite to the Modal workspace.

Questions: Spencer (spencerapeadjei@gmail.com).
