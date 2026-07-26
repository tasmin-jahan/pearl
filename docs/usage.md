# PEARL — Usage Guide

Canonical reference for running the PEARL pipeline. Every script's
flags are documented; every workflow has a copy-paste command block.

---

## Table of Contents

1. [Setup](#setup)
2. [End-to-end quick start (Phase 0 → 6)](#end-to-end-quick-start)
3. [CLI Reference](#cli-reference)
   - [scripts/dedup_data.py](#scriptsdedup_datapy)
   - [scripts/preprocess.py](#scriptspreprocesspy)
   - [scripts/train.py](#scriptstrainpy)
   - [scripts/sweep.py](#scriptssweeppy)
   - [scripts/tune.py](#scriptstunepy)
   - [scripts/sweep_hpo.py](#scriptssweep_hpopy)
   - [scripts/kfold_finalists.py](#scriptskfold_finalistspy)
   - [scripts/evaluate.py](#scriptsevaluatepy)
   - [scripts/run_calibration.py](#scriptsrun_calibrationpy)
   - [scripts/run_calibration_ensemble.py](#scriptsrun_calibration_ensemblepy)
   - [scripts/run_xai.py](#scriptsrun_xaipy)
   - [scripts/run_uncertainty.py](#scriptsrun_uncertaintypy)
   - [scripts/smoke_test.py](#scriptssmoke_testpy)
4. [Config schemas](#config-schemas)
5. [Dynamic CLI overrides (`--set`)](#dynamic-cli-overrides)
6. [Resume / checkpoints](#resume--checkpoints)
7. [Tests + smoke](#tests--smoke)
8. [Output directory layout](#output-directory-layout)

---

## Setup

```bash
cd /home/farhan/my-projects/pearl

# Create venv
python3.12 -m venv ~/pearl/.venv
source ~/pearl/.venv/bin/activate

# 1. PyTorch (CUDA 13 wheel — adjust cuXYZ for your GPU)
pip install torch==2.12.0 torchvision==0.27.0 torchaudio==2.11.0+cu130 \
    --index-url https://download.pytorch.org/whl/cu130

# 2. torchcam (--no-deps to bypass stale numpy<2.0 constraint)
pip install torchcam==0.4.1 --no-deps

# 3. Everything else
pip install -r requirements.txt
```

**Dataset layout** (Figshare PCOS v1):

```
data/
  infected/        # PCOS (label=1, 6784 images)
  noninfected/     # non-PCOS (label=0, 5000 images)
```

Set `DATA_DIR=/path/to/data` for the commands below.

---

## End-to-end quick start

This is the full v3 pipeline, in order. Each step requires the
previous one to have finished.

```bash
# Step 1 — Preprocess both ablation arms
python scripts/preprocess.py \
    --config configs/preprocessing/srad_clahe.yaml \
    --data_dir $DATA_DIR --split_seed 42

python scripts/preprocess.py \
    --config configs/preprocessing/gaussian_clahe.yaml \
    --data_dir $DATA_DIR --split_seed 42

# Step 2 — Run the 18-run ablation (9 archs × 2 denoising configs)
python scripts/sweep.py --experiment configs/experiment/ablation_18.yaml

# Step 3 — Pick the winner preprocessing config
python -c "
import pandas as pd
df = pd.read_csv('results/ablation/sweep_matrix.csv')
print(df.groupby('preprocessing')['val_auc'].agg(['mean', 'std', 'count']))
"
# Lock in the winner; assume srad_clahe from here on.

# Step 4 — Per-architecture Optuna HPO sweep (9 archs × N trials)
python scripts/sweep_hpo.py \
    --experiment configs/experiment/tune_per_arch.yaml \
    --out_dir results/sweep_hpo/

# Step 5 — k-fold CV on top-k finalists
python scripts/kfold_finalists.py \
    --experiment configs/experiment/kfold_finalists.yaml \
    --finalists results/sweep_hpo/finalists.csv \
    --params_dir results/sweep_hpo/ \
    --preprocessing srad_clahe \
    --data_dir $DATA_DIR

# Step 6 — Downstream analyses (per-model)
python scripts/run_calibration.py \
    --model configs/model/swin_tiny.yaml \
    --preprocessing configs/preprocessing/srad_clahe.yaml \
    --checkpoint results/checkpoints/swin_tiny__srad_clahe.pt

python scripts/run_xai.py \
    --model configs/model/swin_tiny.yaml \
    --preprocessing configs/preprocessing/srad_clahe.yaml \
    --checkpoint results/checkpoints/swin_tiny__srad_clahe.pt \
    --methods gradcam lrp shap --n_samples 20

python scripts/run_uncertainty.py \
    --model configs/model/swin_tiny.yaml \
    --preprocessing configs/preprocessing/srad_clahe.yaml \
    --checkpoint results/checkpoints/swin_tiny__srad_clahe.pt \
    --mc_passes 50

# Step 7 — Ensemble calibration (two passes)
python scripts/run_calibration_ensemble.py \
    --model_configs configs/model/swin_tiny.yaml configs/model/convnext_tiny.yaml \
    --checkpoints results/checkpoints/swin_tiny__srad_clahe.pt \
                  results/checkpoints/convnext_tiny__srad_clahe.pt \
    --preprocessing configs/preprocessing/srad_clahe.yaml
```

---

## CLI Reference

Every script is documented below with **Flag | Type | Default | Description**.

---

### `scripts/dedup_data.py`

Find and remove exact-byte duplicates from a class-folder dataset.
The Figshare PCOS dataset ships with substantial duplication
(only 16.2% of `noninfected/` and 46.9% of `infected/` files are
unique — see `notebooks/eda_figures/01_class_balance.png` and
`docs/methodology.tex § Data deduplication`). Run this **before**
`preprocess.py` to avoid wasting compute on repeated images.

```bash
python scripts/dedup_data.py --data_dir data --mode report      # audit only
python scripts/dedup_data.py --data_dir data --mode quarantine  # move dupes
python scripts/dedup_data.py --data_dir data --mode remove --yes  # destructive
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--data_dir` | str | `data` | Root directory containing the class folders. |
| `--mode` | str | `report` | One of `report`, `quarantine`, `remove`. |
| `--classes` | str list | `infected noninfected` | Class folder names to scan. |
| `--extensions` | str list | `.jpg .jpeg .png .bmp .tiff` | File extensions to consider. |
| `--report_dir` | str | `results/eda` | Where to write the CSV reports. |
| `--quarantine_dir` | str | `data/_duplicates` | Destination for quarantined files. |
| `--yes` | flag | off | Required for `mode=remove` (destructive). |

**Outputs**:
- `report` mode: writes `dedup_report.csv` (per-class counts) and `dedup_details.csv` (every duplicate path + MD5 + kept/removed flag). No files are touched.
- `quarantine` mode: also moves duplicates into `<quarantine_dir>/<class>/`. Reversible: `mv data/_duplicates/<class>/* data/<class>/`.
- `remove` mode: permanently deletes duplicates (requires `--yes`).

### `scripts/preprocess.py`

Reads a preprocessing config, runs the pipeline (letterbox resize →
CLAHE → denoise → Z-score), performs the 80/10/10 stratified split
with patient-level leakage check, and saves `.npy` files to disk.

**Required before**: nothing (this is the first step).

```bash
python scripts/preprocess.py \
    --config configs/preprocessing/<name>.yaml \
    --data_dir /path/to/data \
    [--split_seed 42] \
    [--input_size 224]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | str | **required** | Path to preprocessing config YAML. Use `configs/preprocessing/srad_clahe.yaml` or `gaussian_clahe.yaml` for the v3 ablation. |
| `--data_dir` | str | **required** | Path to raw dataset directory containing `infected/` and `noninfected/` subdirs. |
| `--split_seed` | int | `42` | Random seed for the stratified 80/10/10 split. Use the same seed across all preprocess runs to keep splits identical. |
| `--input_size` | int | `224` | Resize target. All v3 models use 224. The longer side is scaled to this size and the shorter side is letterbox-padded — see the `steps.padding` config key below. |

**Resize behaviour (v3.2)**: The preprocessor preserves aspect ratio
by default. The longer image side is scaled to `input_size` and the
shorter side is letterboxed. Three padding modes are supported via
the `steps.padding` key in the preprocessing YAML:

| `steps.padding` | Behaviour |
|---|---|
| `reflect` (default, recommended) | Pad with edge-replicated pixels. No black borders, no aspect distortion. Standard for medical imaging. |
| `constant` | Pad with `steps.pad_value` (default `0`). Equivalent to the dark-border variant. |
| `none` | Stretch to (input_size, input_size). Legacy v1/v2 behaviour. **Distorts follicle shape on non-square inputs** — only use for the v1/v2 ablation configs that explicitly require it. |

**Outputs** under `results/preprocessed/<config_name>/`:

```
train/infected/*.npy        test/infected/*.npy
train/noninfected/*.npy     test/noninfected/*.npy
val/infected/*.npy
val/noninfected/*.npy
```

**Augmentation**: the noise-robustness augmentations declared in the
YAML (`augmentation.jpeg_compression`, `augmentation.light_blur`) are
applied at *training time only*, inside the data loader, after the
preprocessed `.npy` array has been loaded. They never modify the
on-disk preprocessed artefacts and never run on validation or test
splits.

**Examples**

```bash
# v3 ablation: both arms
python scripts/preprocess.py \
    --config configs/preprocessing/srad_clahe.yaml \
    --data_dir $DATA_DIR --split_seed 42

python scripts/preprocess.py \
    --config configs/preprocessing/gaussian_clahe.yaml \
    --data_dir $DATA_DIR --split_seed 42
```

---

### `scripts/train.py`

Trains a single model on preprocessed data. Optionally resumes from
a checkpoint and supports inline `--set` overrides.

**Required before**: `scripts/preprocess.py` for the matching
`--preprocessing` config.

```bash
python scripts/train.py \
    --model configs/model/<arch>.yaml \
    --preprocessing configs/preprocessing/<name>.yaml \
    --experiment configs/experiment/<name>.yaml \
    [--run_dir results/runs/<name>] \
    [--resume results/runs/<run_id>/best.pt] \
    [--set <key.path>=<value> ...]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | str | **required** | Path to model config YAML. e.g. `configs/model/swin_tiny.yaml`. |
| `--preprocessing` | str | **required** | Path to preprocessing config YAML. Must already be on disk from `scripts/preprocess.py`. |
| `--experiment` | str | **required** | Path to experiment config YAML. e.g. `configs/experiment/best_model_xai.yaml`. Provides `training:` block (lr, epochs, etc.) and `seed:`. |
| `--run_dir` | str | auto-generated | Override the run output directory. Default: `results/runs/<arch>__<preproc>__<timestamp>/`. |
| `--resume` | str | `None` | Path to a checkpoint to resume from. Restores model + optimizer + scheduler + EMA + RNG. |
| `--set` | str (repeatable) | `[]` | Inline config override. See [Dynamic CLI overrides](#dynamic-cli-overrides). Repeatable. |

**Outputs** under `results/runs/<arch>__<preproc>__<timestamp>/`:

```
config.yaml              # snapshot of merged config
epoch_log.csv            # one row per epoch
final_metrics.json       # test-set metrics
training_curve.png       # loss + AUC plots
best.pt                  # best-by-val-AUC checkpoint
best__epoch<N>.pt        # rolling-window recent checkpoints
```

**Examples**

```bash
# Basic single run
python scripts/train.py \
    --model configs/model/swin_tiny.yaml \
    --preprocessing configs/preprocessing/srad_clahe.yaml \
    --experiment configs/experiment/best_model_xai.yaml

# Run with overridden hyperparameters
python scripts/train.py \
    --model configs/model/swin_tiny.yaml \
    --preprocessing configs/preprocessing/srad_clahe.yaml \
    --experiment configs/experiment/best_model_xai.yaml \
    --set training.lr=5e-4 training.batch_size=16

# Resume from a previous run
python scripts/train.py \
    --model configs/model/swin_tiny.yaml \
    --preprocessing configs/preprocessing/srad_clahe.yaml \
    --experiment configs/experiment/best_model_xai.yaml \
    --resume results/runs/swin_tiny__srad_clahe__20260518_143200/best.pt
```

---

### `scripts/sweep.py`

Iterates over `(model, preprocessing)` pairs from an experiment config
and runs each one through `scripts/train.py`-equivalent training. Used
for the Phase 0 ablation (18 runs).

**Required before**: `scripts/preprocess.py` for EVERY preprocessing
config named in the experiment config.

```bash
python scripts/sweep.py --experiment configs/experiment/<name>.yaml
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--experiment` | str | **required** | Experiment config YAML. Reads `models:`, `preprocessing:`, `training:`, `seed:`, `results_dir:`. |

**Examples**

```bash
# Phase 0 ablation: 9 archs × 2 denoising configs = 18 runs
python scripts/sweep.py --experiment configs/experiment/ablation_18.yaml

# Legacy 54-run sweep (kept for backward compatibility only)
python scripts/sweep.py --experiment configs/experiment/ablation_legacy_54.yaml
```

**Outputs**: `results/<results_dir>/sweep_matrix.csv` containing per-run metrics.

---

### `scripts/tune.py`

(Legacy single-model Optuna tuning. v3 prefers `sweep_hpo.py` for
per-architecture HPO across all 9 backbones.)

```bash
python scripts/tune.py \
    --experiment configs/experiment/tune_best.yaml \
    [--study_name <name>] \
    [--storage sqlite:///path/to/optuna.db]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--experiment` | str | **required** | Path to `tune_best.yaml`. Reads `model:`, `preprocessing:`, `n_trials:`, search space, etc. |
| `--study_name` | str | `pcos_tuning` | Optuna study name. |
| `--storage` | str | `None` | Optuna storage URL (e.g. SQLite). If set, the study is persisted and resumable. |

**Required before**: `scripts/preprocess.py` for the preprocessing
config named in the experiment config.

**Outputs** under `results/tuning/`:

```
optuna.db
best_params.yaml
all_trials.csv
```

---

### `scripts/sweep_hpo.py`

Per-architecture Optuna HPO sweep across all 9 architectures. Picks
the top-k finalists by val AUC.

**Required before**: `scripts/preprocess.py` for the winning
preprocessing config (the one whose name is in the experiment
config's `preprocessing:` field).

```bash
python scripts/sweep_hpo.py \
    --experiment configs/experiment/tune_per_arch.yaml \
    [--out_dir results/sweep_hpo/]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--experiment` | str | **required** | Path to `tune_per_arch.yaml`. Reads `models:`, `preprocessing:`, `n_trials_per_arch:`, `top_k_finalists:`, search space. |
| `--out_dir` | str | `results/sweep_hpo/` | Output directory. |

**Examples**

```bash
python scripts/sweep_hpo.py \
    --experiment configs/experiment/tune_per_arch.yaml \
    --out_dir results/sweep_hpo/
```

**Outputs** under `results/sweep_hpo/`:

```
summary.csv                 # per-arch best val AUC, ranked
finalists.csv               # top-k by val AUC
<arch>/
├── best_params.yaml        # winning hyperparameters
└── all_trials.csv          # full trial history
```

---

### `scripts/kfold_finalists.py`

K-fold cross-validation on top-k finalists only. Loads raw images
and applies the preprocessor in-memory per fold — **no prior
preprocess run is required**.

```bash
python scripts/kfold_finalists.py \
    --experiment configs/experiment/kfold_finalists.yaml \
    --finalists results/sweep_hpo/finalists.csv \
    --params_dir results/sweep_hpo/ \
    --preprocessing srad_clahe \
    --data_dir /path/to/data \
    [--out_dir results/kfold/]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--experiment` | str | **required** | Path to `kfold_finalists.yaml`. Reads `n_folds:`, `training:` block, `seed:`. |
| `--finalists` | str | **required** | Path to `finalists.csv` (output of `scripts/sweep_hpo.py`). Must have a column named `arch`. |
| `--params_dir` | str | **required** | Directory containing per-arch `best_params.yaml` files (the `results/sweep_hpo/<arch>/` outputs). |
| `--preprocessing` | str | `srad_clahe` | Name of the preprocessing config to use. (NOT a path — just the name, looked up under `configs/preprocessing/`. Use the same name as the ablation winner.) |
| `--data_dir` | str | **required** | Path to raw dataset directory (`infected/` + `notinfected/`). |
| `--out_dir` | str | `results/kfold/` | Output directory. |

**Examples**

```bash
python scripts/kfold_finalists.py \
    --experiment configs/experiment/kfold_finalists.yaml \
    --finalists results/sweep_hpo/finalists.csv \
    --params_dir results/sweep_hpo/ \
    --preprocessing srad_clahe \
    --data_dir $DATA_DIR
```

**Outputs** under `results/kfold/`:

```
kfold_summary.csv           # per-fold AUC per arch
<arch>/fold<N>/best.pt      # per-fold checkpoint
<arch>/fold<N>/epoch_log.csv
```

---

### `scripts/evaluate.py`

Standalone evaluation of a trained model on the test set. Useful for
recomputing metrics from a checkpoint outside the training flow.

**Required before**: `scripts/preprocess.py` for the matching
`--preprocessing` config.

```bash
python scripts/evaluate.py \
    --model configs/model/<arch>.yaml \
    --preprocessing configs/preprocessing/<name>.yaml \
    --checkpoint results/checkpoints/<arch>__<preproc>.pt \
    [--output results/eval/<arch>.json]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | str | **required** | Path to model config YAML. |
| `--preprocessing` | str | **required** | Path to preprocessing config YAML. |
| `--checkpoint` | str | **required** | Path to `.pt` checkpoint file. |
| `--output` | str | `None` | If set, write metrics JSON to this path. |

**Outputs**: prints all 7 metrics (accuracy, precision, recall,
specificity, F1, AUC-ROC, MCC). Optionally writes JSON if `--output` is set.

---

### `scripts/evaluate_external.py`

External-validation evaluator: runs a trained checkpoint against a
**different** dataset (no leakage from training/test splits). This is
the script that produces the "external validation" row in the paper's
comparison table.

The external dataset is loaded on-the-fly from raw images — no
preprocessing cache needed. The same preprocessing config used in
training is applied at inference time so the comparison is apples-to-
apples.

**Layouts supported**:
- `pcosgen` — the Sundari et al. 2025 PCOSGen Kaggle upload (train+test
  splits combined; uses `infected`/`healthy` folder names).
- `simple` — `<dataset>/<split>/<class>/*.jpg` (PCOSDataset convention).
- `flat` — `<dataset>/<class>/*.jpg` (no split subdir).

```bash
python scripts/evaluate_external.py \
    --model configs/model/efficientnet_b0.yaml \
    --preprocessing configs/preprocessing/srad_clahe.yaml \
    --checkpoint results/checkpoints/efficientnet_b0__srad_clahe.pt \
    --external_dir /home/farhan/my-projects/pearl/data_external/pcosgen \
    --external_layout pcosgen \
    --output results/external_validation/pcosgen_srad_clahe.json \
    --predictions_csv results/external_validation/pcosgen_srad_clahe.csv
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | str | **required** | Path to model config YAML. |
| `--preprocessing` | str | **required** | Path to preprocessing config YAML. Must match the preprocessing the model was trained with. |
| `--checkpoint` | str | **required** | Path to `.pt` checkpoint. |
| `--external_dir` | str | **required** | Root directory of the external dataset. |
| `--external_layout` | str | `pcosgen` | One of `pcosgen`, `simple`, `flat`. |
| `--split` | str | `test` | For `simple` layout: which subdir to use. |
| `--batch_size` | int | `32` | Inference batch size. |
| `--max_samples` | int | `None` | Optional cap on number of samples (debug only). |
| `--seed` | int | `42` | Seed for any RNG in the loader. |
| `--output` | str | `None` | If set, write metrics JSON to this path. |
| `--predictions_csv` | str | `None` | If set, write per-image `(path, label, pred, prob_infected)` rows. |

**Outputs**: prints all metrics including `n_samples`, `n_infected`,
`n_healthy`. Optionally writes JSON if `--output` is set, and per-image
predictions CSV if `--predictions_csv` is set.

**Note**: This script does not perform any model fine-tuning on the
external data. The reported numbers reflect pure domain-shift
performance of the model as trained on the Figshare PCOS dataset.

---

### `scripts/run_calibration.py`

Per-model temperature scaling (Pass 1 of two-pass calibration).

**Required before**: `scripts/preprocess.py` for the matching
`--preprocessing` config.

```bash
python scripts/run_calibration.py \
    --model configs/model/<arch>.yaml \
    --preprocessing configs/preprocessing/<name>.yaml \
    --checkpoint results/checkpoints/<arch>__<preproc>.pt \
    [--n_bins 15]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | str | **required** | Path to model config YAML. |
| `--preprocessing` | str | **required** | Path to preprocessing config YAML. |
| `--checkpoint` | str | **required** | Path to the trained `.pt` checkpoint. |
| `--n_bins` | int | `15` | Number of equal-width bins for ECE computation. |

**Outputs** under `results/calibration/<arch>__<preproc>/`:

```
calibration_results.json    # ECE before/after, optimal T, NLL
reliability_diagram_before.png
reliability_diagram_after.png
bin_data.csv
```

---

### `scripts/run_calibration_ensemble.py`

Two-pass ensemble calibration: per-model T (Pass 1) + ensemble T
(Pass 2).

**Required before**: `scripts/preprocess.py` for the matching
`--preprocessing` config.

```bash
python scripts/run_calibration_ensemble.py \
    --model_configs configs/model/a.yaml configs/model/b.yaml \
    --checkpoints results/checkpoints/a.pt results/checkpoints/b.pt \
    --preprocessing configs/preprocessing/<name>.yaml \
    [--n_bins 15] \
    [--out_dir results/calibration_ensemble]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model_configs` | str list | **required** | One model YAML per ensemble member. Pass multiple values. |
| `--checkpoints` | str list | **required** | One checkpoint path per ensemble member. Must match `--model_configs` in length and order. |
| `--preprocessing` | str | **required** | Path to preprocessing config YAML. |
| `--n_bins` | int | `15` | Number of ECE bins. |
| `--out_dir` | str | `results/calibration_ensemble` | Output directory. |

**Outputs** under `results/calibration_ensemble/`:

```
calibration_results.json    # per_model_temperatures, ensemble_temperature, ECE after each pass
reliability_p1.png          # after per-model T
reliability_p2.png          # after ensemble T
bin_data_pass1.csv
bin_data_pass2.csv
```

**Examples**

```bash
python scripts/run_calibration_ensemble.py \
    --model_configs configs/model/swin_tiny.yaml configs/model/convnext_tiny.yaml \
    --checkpoints results/checkpoints/swin_tiny__srad_clahe.pt \
                  results/checkpoints/convnext_tiny__srad_clahe.pt \
    --preprocessing configs/preprocessing/srad_clahe.yaml
```

---

### `scripts/run_xai.py`

Generates Grad-CAM, LRP, and SHAP explanations for 20 representative
test samples (5 per group: confident-correct, overconfident-error,
uncertain-correct, uncertain-wrong).

**Required before**: `scripts/preprocess.py` for the matching
`--preprocessing` config.

```bash
python scripts/run_xai.py \
    --model configs/model/<arch>.yaml \
    --preprocessing configs/preprocessing/<name>.yaml \
    --checkpoint results/checkpoints/<arch>__<preproc>.pt \
    [--methods gradcam lrp shap] \
    [--n_samples 20] \
    [--mc_passes 50]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | str | **required** | Path to model config YAML. |
| `--preprocessing` | str | **required** | Path to preprocessing config YAML. |
| `--checkpoint` | str | **required** | Path to the trained `.pt` checkpoint. |
| `--methods` | str list | `gradcam lrp shap` | Which XAI methods to run. Subset allowed: `--methods gradcam`. |
| `--n_samples` | int | `20` | Total samples to explain (split 4 ways, so choose a multiple of 4). |
| `--mc_passes` | int | `50` | Number of MC Dropout passes for entropy-based sample selection. |

**Outputs** under `results/xai/<arch>__<preproc>/`:

```
gradcam/sample_<id>_*.png
lrp/sample_<id>_relevance.png
shap/sample_<id>_shap.png
shap/mean_shap_summary.png
xai_metadata.csv
```

---

### `scripts/run_uncertainty.py`

MC Dropout uncertainty + referral system. 50 stochastic forward
passes (default) for predictive entropy, plus a coverage-accuracy
sweep.

**Required before**: `scripts/preprocess.py` for the matching
`--preprocessing` config.

```bash
python scripts/run_uncertainty.py \
    --model configs/model/<arch>.yaml \
    --preprocessing configs/preprocessing/<name>.yaml \
    --checkpoint results/checkpoints/<arch>__<preproc>.pt \
    [--mc_passes 50] \
    [--entropy_threshold 0.35] \
    [--coverage_thresholds 1.0 0.9 0.8 0.7 0.6]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | str | **required** | Path to model config YAML. |
| `--preprocessing` | str | **required** | Path to preprocessing config YAML. |
| `--checkpoint` | str | **required** | Path to the trained `.pt` checkpoint. |
| `--mc_passes` | int | `50` | Number of MC Dropout forward passes for averaging. |
| `--entropy_threshold` | float | `0.35` | Threshold (in nats) below which a sample is "auto-decided" (high confidence). |
| `--coverage_thresholds` | float list | `1.0 0.9 0.8 0.7 0.6` | Coverage levels to evaluate on the referral curve. |

**Outputs** under `results/uncertainty/<arch>__<preproc>/`:

```
uncertainty_results.json
entropy_per_sample.csv
referral_curve.csv
entropy_histogram.png
coverage_accuracy_curve.png
```

---

### `scripts/smoke_test.py`

Pre-flight check: 2-epoch synthetic training pass through the full
Trainer pipeline. Run before any long sweep.

```bash
python scripts/smoke_test.py [--device cpu]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--device` | str | `cpu` | Device to run on. Override to `cuda` if you want to test GPU paths. |

Target: < 15 seconds (PASSED if < 60s).

---

## Config schemas

### `configs/model/<arch>.yaml`

```yaml
name: swin_tiny
timm_name: swin_tiny_patch4_window7_224
input_size: 224                # all 9 v3 models = 224
pretrained: true
num_classes: 2
freeze_fraction: 0.60
head:
  hidden_dim: 256
  dropout: 0.5
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Short architecture name (used in output paths). |
| `timm_name` | str | timm model key. See `configs/model/*.yaml` for the v3 mappings. |
| `input_size` | int | Spatial input size. All v3 models standardized to 224. |
| `pretrained` | bool | Whether to load ImageNet-pretrained weights. |
| `num_classes` | int | Output classes. Wired to the head in `src/model/builder.py`. |
| `freeze_fraction` | float | Fraction of backbone params to freeze (0.0–1.0). Optuna sweeps over [0.4, 0.5, 0.6, 0.7]. |
| `head.hidden_dim` | int | Hidden dimension of the first FC layer in the head. |
| `head.dropout` | float | Dropout rate after the first head activation. Optuna sweeps over [0.3, 0.7]. |

### `configs/preprocessing/<name>.yaml`

```yaml
name: srad_clahe
steps:
  resize: true
  clahe: {enabled: true, clip_limit: 2.0, tile_size: [8, 8]}
  gaussian: {enabled: false}
  srad: {enabled: true, iterations: 20, kappa: 30, gamma: 0.1}
  anisotropic_diffusion: {enabled: false}
  zscore_normalize: {enabled: true}
  padding: reflect                   # letterbox mode: reflect | constant | none
augmentation:
  rotation: 15
  horizontal_flip: true
  scale: 0.10
  jpeg_compression:                  # noise-robustness augmentation
    enabled: true
    p: 0.3
    q_low: 50
    q_high: 95
  light_blur:                        # noise-robustness augmentation
    enabled: true
    p: 0.2
    sigma_low: 0.1
    sigma_high: 1.5
output_dir: results/preprocessed/srad_clahe
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Config name. Used as output dir suffix. |
| `steps.resize` | bool | Bicubic resize to `input_size` (default `224`). |
| `steps.padding` | str | Letterbox mode: `reflect` (default), `constant`, or `none` (legacy stretch). |
| `steps.clahe.{enabled, clip_limit, tile_size}` | bool/float/[int,int] | CLAHE. Fixed at `clip_limit=2.0, tile_size=[8,8]` in v3. |
| `steps.gaussian.{enabled, sigma}` | bool/float | Gaussian blur. `sigma=1.0` for the v3 ablation arm. |
| `steps.srad.{enabled, iterations, kappa, gamma}` | bool/int/float/float | SRAD denoising. Default `(20, 30, 0.1)`. |
| `steps.anisotropic_diffusion.{enabled, ...}` | bool | Generic Perona–Malik AD. **Not used in v3** (legacy kept for backward compatibility). |
| `steps.zscore_normalize.enabled` | bool | Per-channel z-score. |
| `augmentation.{rotation, horizontal_flip, scale}` | float/bool/float | Geometric augmentation, applied at training time only. |
| `augmentation.jpeg_compression.{enabled, p, q_low, q_high}` | bool/float/int/int | Simulates JPEG codec artefacts (random q in `[q_low, q_high]`) with probability `p`. Targets the ~q40 compression already present in the dataset. |
| `augmentation.light_blur.{enabled, p, sigma_low, sigma_high}` | bool/float/float/float | Light Gaussian blur (sigma in range) with probability `p`. Suppresses JPEG ringing without erasing anatomical edges. |
| `output_dir` | str | Where `.npy` files are saved. |

#### Noise-robustness augmentation rationale

The source dataset is heavily compressed (94.5% of images at estimated q ≤ 60, median ~q40). Preprocessing (SRAD + CLAHE) removes speckle and normalizes contrast but cannot remove codec artefacts because they have already destroyed the high-frequency DCT coefficients. Two augmentations address this:

- **JPEG re-compression** (`jpeg_compression`): randomly re-encode the training image at a fresh JPEG quality. Teaches the model to be invariant to the family of codec artefacts, not just one specific instance.
- **Light blur** (`light_blur`): sigma in [0.1, 1.5] suppresses JPEG ringing (1–2 px radius) without erasing anatomical edges (5–20 px radius).

Both are disabled by default. The v3 ablation YAMLs enable them at low probability (`p=0.3`, `p=0.2`) so the model is robust without losing signal.

### `configs/experiment/<name>.yaml`

The top-level keys. The `training:` block is consumed by `Trainer`.

```yaml
name: my_experiment
mode: ablation                    # ablation | sweep | single | sweep_tune | kfold
models:                            # for sweep.py / sweep_hpo.py
  - resnet50
  - swin_tiny
preprocessing: [srad_clahe, gaussian_clahe]   # for sweep.py
finalists: [swin_tiny, convnext_tiny]         # for kfold_finalists
n_trials_per_arch: 20             # HPO sweep
top_k_finalists: 3
n_folds: 5                        # k-fold
training:
  lr: 1.0e-4
  weight_decay: 1.0e-2
  batch_size: 32
  max_epochs: 100
  early_stopping_patience: 20
  warmup_epochs: 2                # Phase 2.1
  freeze_epochs: 1                # Phase 2.1 (set 0 to disable)
  bf16: true                      # Phase 2.5
  channels_last: true             # Phase 2.5
  grad_clip_norm: 1.0             # Phase 2.2
  ema: true                       # Phase 3.1
  ema_decay: 0.999
  keep_last_n: 3                  # Phase 3.4
seed: 42
results_dir: results/
```

---

## Dynamic CLI overrides

`scripts/train.py` and `scripts/sweep.py` accept a repeatable
`--set <key.path>=<value>` flag. The override is parsed with:

- `int` → `int`
- `float` (incl. `5e-4`) → `float`
- `true` / `false` → `bool`
- `null` / `none` → `None`
- everything else → `str`

Dotted keys navigate nested dicts.

**Examples**

```bash
# Single override
--set training.lr=5e-4

# Multiple overrides
--set training.lr=5e-4 training.batch_size=16

# Nested key
--set head.dropout=0.3

# Boolean toggle
--set bf16=false ema=true
```

Implementation: `src.utils.config.parse_overrides` +
`src.utils.config.apply_overrides`.

---

## Class imbalance handling

After deduplication the dataset imbalance is **3.92×** (3,184 infected vs
812 non-infected — see `docs/methodology.tex § Class imbalance`).
The pipeline supports two complementary mechanisms, both enabled by
default for the loss and configurable for the sampler:

| Mechanism | Where | Behaviour |
|---|---|---|
| Class-weighted CE loss | `src/training/losses.py` | `w_c = n_total / (2 * n_c)`; applied unconditionally. |
| WeightedRandomSampler | `src/data/dataloader.py:make_weighted_sampler` | Inverse-frequency sampling; **off by default**. |

**Enable weighted sampling per experiment** by adding to the training
config block:

```yaml
training:
  sampler: weighted      # or "shuffle" (default) | "none"
```

Or override on the CLI:

```bash
--set training.sampler=weighted
```

When `sampler=weighted`, leave loss weights at their default — the two
mechanisms compound rather than cancel. Set `sampler=none` for
deterministic iteration (debugging only).

---

## Resume / checkpoints

Checkpoints are **self-contained**: they store arch name, class names,
model + optimizer + scheduler state, EMA shadow, and RNG state (Python,
NumPy, PyTorch CPU + CUDA). A single `.pt` file is enough to resume a
run exactly.

```bash
python scripts/train.py \
    --model configs/model/swin_tiny.yaml \
    --preprocessing configs/preprocessing/srad_clahe.yaml \
    --experiment configs/experiment/best_model_xai.yaml \
    --resume results/runs/<run_id>/best.pt
```

The trainer picks up at `epoch + 1`, restores RNG so the augmentation
sequence is reproducible, and the logger **appends** to
`epoch_log.csv` rather than overwriting.

The checkpoint also stores rolling-window epochs (`__epochN.pt`) — the
last `keep_last_n` are kept alongside `best.pt`, so SWA can
average them later.

---

## Tests + smoke

### Pytest suite (fast, CPU-only)

```bash
python -m pytest tests/ -v
```

| Test file | What it checks |
|-----------|----------------|
| `tests/test_config_overrides.py` | `--set` parsing + deep-merge semantics |
| `tests/test_split_overlap.py` | 80/10/10 split ratios; group-level no-leakage across train/val/test |
| `tests/test_resume_correctness.py` | Checkpoint round-trip; resume state restoration; NaN guard raises `TrialDivergedError` |

### Smoke test (full integration, ~15s, CPU)

```bash
python scripts/smoke_test.py
```

Runs a 2-epoch synthetic training pass through the full Trainer +
Trainer-with-EMA + checkpoint pipeline. Prints a `PASSED` line if
total elapsed < 60s (target ~15s on CPU). Run before any long sweep to
catch integration breakage fast.

---

## Output directory layout

```
results/
├── ablation/                                  # Phase 0 (18 runs)
│   └── sweep_matrix.csv
├── runs/                                       # Phase 1 single runs
│   └── <arch>__<preproc>__<timestamp>/
│       ├── config.yaml
│       ├── epoch_log.csv
│       ├── final_metrics.json
│       ├── training_curve.png
│       ├── best.pt
│       └── best__epoch<N>.pt                  # rolling-window
├── sweep_hpo/                                  # Phase 2 (per-arch HPO)
│   ├── summary.csv
│   ├── finalists.csv
│   └── <arch>/{best_params.yaml, all_trials.csv}
├── kfold/                                      # Phase 3
│   ├── kfold_summary.csv
│   └── <arch>/fold<N>/{best.pt, epoch_log.csv, ...}
├── checkpoints/                                # shared checkpoint store
├── calibration/<arch>__<preproc>/             # Phase 5 pass 1
│   ├── calibration_results.json
│   ├── reliability_diagram_before.png
│   ├── reliability_diagram_after.png
│   └── bin_data.csv
├── calibration_ensemble/                      # Phase 5 pass 2
│   ├── calibration_results.json
│   ├── reliability_p1.png
│   ├── reliability_p2.png
│   ├── bin_data_pass1.csv
│   └── bin_data_pass2.csv
├── uncertainty/<arch>__<preproc>/             # Phase 6
├── xai/<arch>__<preproc>/{gradcam,lrp,shap}/
└── tuning/                                     # legacy single-arch tuning
```

All naming uses `<arch>__<preproc>` as the canonical key.
