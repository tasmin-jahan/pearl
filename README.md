# PEARL

**Probabilistic Explainability with Adaptive Reliability via transfer Learning**

A config-driven deep learning pipeline for PCOS (Polycystic Ovary
Syndrome) detection from ovarian ultrasound images, with model
comparison, explainability, calibration, and uncertainty quantification.

The v3 redesign (vs. v1/v2) added: 80/10/10 split with patient-level
leakage check, SRAD replacing generic AD, SiLU head with dynamic
`num_classes`, no-decay parameter groups, NaN divergence guard, LR
warmup + two-phase fine-tuning, EMA + SWA, two-pass calibration,
self-contained checkpoints with RNG state for resume. See
[`docs/methodology.tex`](docs/methodology.tex) for the full design rationale.

## Where to look

- **How to run the pipeline** → `## Usage` below (covers all CLI commands, configs, output layouts).
- **Pipeline redesign writeup** → [`docs/methodology.tex`](docs/methodology.tex).
- **Findings and analysis figures** → [`docs/analysis/`](docs/analysis/).

## What's in the codebase

```
pearl/
├── configs/
│   ├── preprocessing.yaml         # single unified preprocessing config
│   ├── model/                     # one YAML per architecture
│   └── search_space/              # figshare.yaml, pcosgen.yaml (Optuna)
├── src/
│   ├── data/dataloader.py         # PNG + CSV loader, weighted sampler
│   ├── preprocessing/             # preprocess.py (transform) + run_preprocessing/dedup/split (CLIs)
│   ├── model/builder.py           # timm-based model factory
│   ├── training/                  # Trainer, losses, checkpoint, swa
│   ├── evaluation/                # evaluator + metrics + run_evaluate CLI
│   ├── ensemble/                  # 00_load → 06_cli_uncertainty, by pipeline order
│   ├── calibration/               # ECE + temperature scaling + run_calibration CLI
│   ├── uncertainty/               # MC dropout + referral + run_uncertainty CLI
│   ├── xai/                       # gradcam + lrp + shap_explainer + run_xai CLI
│   ├── utils/                     # config, seed, logging
│   └── train.py                   # single training/HPO entrypoint
├── scripts/
│   ├── generate_figures.py        # dispatcher (--only {analysis,paper,thesis})
│   └── _figures_lib/              # private figure helpers
├── tests/                         # pytest suite
├── docs/
│   ├── methodology.tex
│   └── analysis/                  # findings, finetune_zenodo, figures
└── requirements.txt
```

## Quickstart: End-to-End Pipeline Runners

For immediate reproduction of the complete PEARL pipeline, 9 numbered runner scripts are provided in `scripts/`:

| Step | Script | Purpose |
|------|--------|---------|
| **01** | [`scripts/01_setup_env.py`](scripts/01_setup_env.py) | Verifies Python, PyTorch/CUDA, GPU VRAM, installs dependencies, tests forward pass. |
| **02** | [`scripts/02_prepare_data.py`](scripts/02_prepare_data.py) | Unpacks `data/*.zip`, runs deduplication, stratified splits, and materializes preprocessed PNGs. |
| **03** | [`scripts/03_train_models.py`](scripts/03_train_models.py) | Trains individual models (`--model <name>`) or all 5 architectures sequentially (`--model all`). |
| **04** | [`scripts/04_evaluate_models.py`](scripts/04_evaluate_models.py) | Evaluates trained models on in-distribution or external test cohorts (AUC, F1, MCC, Brier). |
| **05** | [`scripts/05_calibration_analysis.py`](scripts/05_calibration_analysis.py) | Fits temperature scaling calibration on validation set, evaluates ECE on test set. |
| **06** | [`scripts/06_create_ensemble.py`](scripts/06_create_ensemble.py) | Aggregates individual models into a calibrated, probability-averaged ensemble. |
| **07** | [`scripts/07_uncertainty_analysis.py`](scripts/07_uncertainty_analysis.py) | Runs MC-Dropout (50 passes), computes predictive entropy and risk-coverage trade-offs. |
| **08** | [`scripts/08_xai_analysis.py`](scripts/08_xai_analysis.py) | Generates Grad-CAM, LRP, and SHAP visual explanations across confidence quadrants. |
| **09** | [`scripts/09_generate_figures.py`](scripts/09_generate_figures.py) | Generates all paper, thesis, and analysis figures and LaTeX tables in `docs/`. |

### Cloud GPU Execution (Kaggle / Google Colab)

A self-contained, turnkey Jupyter notebook is located at:
[`notebooks/pearl_kaggle_colab.ipynb`](notebooks/pearl_kaggle_colab.ipynb)

- **Optimized for Nvidia T4 (16GB VRAM)**: Automatically configures FP16 Tensor Cores (`--fp16`) and batch sizes (32/64).
- **One-Click Execution**: Step-by-step cells run each stage cleanly with formatted progress outputs.
- **Automated Checkpoint Packaging**: Compresses `results/` into `pearl_results.zip` and `docs/` into `pearl_figures_and_tables.zip` with browser download prompts (`files.download` on Colab, working output on Kaggle).

---

## Detailed CLI Usage

All entrypoints are also available as standalone Python modules, run from the repo root. Every CLI
that touches preprocessed data expects a `--dataset-dir` of the form
`data/preprocessed/<dataset>/` (canonical layout produced by
`run_preprocessing`).

### 0. Setup

```bash
pip install -r requirements.txt
```

### 1. Smoke test (~5s, no GPU)

```bash
pytest tests/test_smoke.py -s
```

### 2. Prepare raw data (Figshare dedup → train/test stratified split)

```bash
# Three-stage pipeline: md5 + dHash dedup → canonical rename → stratified split.
python -m src.preprocessing.dedup --all
```

### 3. Carve validation split

```bash
python -m src.preprocessing.split --dataset figshare     # 10% val out of train
python -m src.preprocessing.split --dataset pcosgen      # same for PCOSGen
```

### 4. Materialize preprocessed PNGs

```bash
python -m src.preprocessing.run_preprocessing \
    --dataset-dir data/raw/figshare \
    --output-dir data/preprocessed/figshare
```

The output is a deterministic PNG tree mirroring the input split
boundaries. Augmentation happens only at training time. Re-running
wipes the output by default (`--no-clean` to append).

### 5. Train (single model, optional checkpoint resume)

```bash
# Foundation pass from scratch
python -m src.train \
    --dataset-dir data/preprocessed/figshare \
    --model swin_tiny \
    --output-dir results/figshare

# Optional: load a checkpoint to initialize weights (fine-tune mode)
python -m src.train \
    --dataset-dir data/preprocessed/pcosgen \
    --model swin_tiny \
    --checkpoint results/figshare/swin_tiny/best.pt \
    --output-dir results/pcosgen
```

`--checkpoint` initializes the model weights from `best.pt` but
**does not** carve a frozen fine-tune mode — the same trainer loop
runs in both cases. The full hyperparameter surface is exposed via
flags (`--lr`, `--weight-decay`, `--dropout`, `--freeze-fraction`,
`--epochs`, `--patience`, `--batch-size`, `--sampler`,
`--warmup-epochs`, `--freeze-epochs`). Throughput toggles are
`--ema/--no-ema`, `--bf16/--fp16`, `--channels-last/--no-channels-last`,
and `--compile`.

### 5a. Train (multiple models in one invocation)

`--model` accepts a space-separated list. Pass each architecture
once and they run sequentially, each in its own
`<output-dir>/<name>/` subdirectory (using the same name you passed
on the CLI). All other hyperparameters are shared.

```bash
python -m src.train \
    --dataset-dir data/preprocessed/figshare \
    --model swin_tiny vit_base convnext_tiny densenet169 efficientnet_b0 \
    --output-dir results/figshare
```

Saves to `results/figshare/swin_tiny/`, `results/figshare/vit_base/`,
etc. — one `best.pt` per model.

### 5b. Fine-tune multiple models at once

The same shape, with `--model-dir` pointing at the foundation-pass
output. For each `--model <name>`, `src.train` initializes from
`<model-dir>/<name>/best.pt` if it exists; otherwise it trains that
model from scratch (with a printed warning). This is the typical
cross-dataset fine-tune workflow: foundation pass already wrote one
`best.pt` per architecture under `results/<dataset>/<arch>/`.

```bash
python -m src.train \
    --dataset-dir data/preprocessed/pcosgen \
    --model swin_tiny vit_base convnext_tiny densenet169 efficientnet_b0 \
    --model-dir results/figshare \
    --output-dir results/pcosgen
```

Saves to `results/pcosgen/<name>/`. To use a single checkpoint for
all models instead, pass `--checkpoint <path>` (no `--model-dir`).

### 6. Train (Optuna HPO)

```bash
python -m src.train \
    --dataset-dir data/preprocessed/figshare \
    --model swin_tiny \
    --hpo \
    --search-space configs/search_space/figshare.yaml \
    --n-trials 30 \
    --study-name figshare_swin_tiny \
    --output-dir results/hpo/figshare/swin_tiny
```

Search spaces are dataset-specific (`figshare.yaml`, `pcosgen.yaml`).
Optuna uses TPE + MedianPruner, reports `val_auc` per trial, and
respects Trainer early stopping.

### 7. Evaluate a checkpoint (single or multi-model)

The same CLI covers in-distribution eval, fine-tune eval, and zero-shot
transfer eval. The CLI is dataset-agnostic — it never inspects which
dataset the checkpoint was trained on.

```bash
# Single model (writes <checkpoint-dir>/metrics.json)
python -m src.evaluation.run_evaluate \
    --checkpoint-dir results/figshare/swin_tiny \
    --test-dataset-dir data/preprocessed/figshare \
    --model-config configs/model/swin_tiny.yaml

# Multi-model zero-shot eval: every <arch>/best.pt under --model-dir
# is loaded and scored against --test-dataset-dir. Each architecture's
# YAML is resolved from configs/model/<arch>.yaml automatically.
python -m src.evaluation.run_evaluate \
    --model-dir results/stage1 \
    --test-dataset-dir data/preprocessed/pcosgen \
    --model-configs-dir configs/model \
    --output-dir results/zero_shot_pcosgen
```

`--test-dataset-dir` accepts either the dataset root
(`data/preprocessed/pcosgen`) or the test split directory directly
(`data/preprocessed/pcosgen/test`). Multi-model mode writes
`<output-dir>/<arch>/metrics.json` for each architecture and an
aggregated `<output-dir>/all_metrics.json`. Per-arch failures don't
abort the run — they're recorded as `{"error": "..."}` in the
aggregate so partial sweeps still produce usable output.

### 8. Calibration analysis (single + ensemble)

```bash
# Single model
python -m src.calibration.run_calibration \
    --dataset-dir data/preprocessed/figshare \
    --model swin_tiny \
    --checkpoint results/figshare/swin_tiny/best.pt \
    --out-dir results/calibration/figshare/swin_tiny

# Ensemble (two-pass: per-model T, then ensemble T)
python -m src.calibration.run_calibration \
    --dataset-dir data/preprocessed/figshare \
    --model swin_tiny     --checkpoint .../swin_tiny/best.pt \
    --model densenet169   --checkpoint .../densenet169/best.pt \
    --ensemble \
    --out-dir results/calibration/figshare/ensemble
```

Outputs `reliability_diagram_*.png`, `bin_data.csv` (or
`bin_data_pass1/2.csv` for ensemble), and `calibration_results.json`.
All ensemble logic lives under `src/ensemble/` (files `_00_` →
`_06_`, ordered by pipeline stage); `--ensemble` delegates there.

### 9. Uncertainty quantification (single + ensemble)

```bash
# Single model
python -m src.uncertainty.run_uncertainty \
    --dataset-dir data/preprocessed/figshare \
    --model swin_tiny \
    --checkpoint results/figshare/swin_tiny/best.pt \
    --out-dir results/uncertainty/figshare/swin_tiny \
    --mc-passes 50

# Ensemble (averaged MC-dropout entropy across members)
python -m src.uncertainty.run_uncertainty \
    --dataset-dir data/preprocessed/figshare \
    --model swin_tiny     --checkpoint .../swin_tiny/best.pt \
    --model densenet169   --checkpoint .../densenet169/best.pt \
    --ensemble \
    --out-dir results/uncertainty/figshare/ensemble
```

Outputs `entropy_per_sample.csv`, `entropy_histogram.png`,
`coverage_accuracy_curve.png`, `referral_curve.csv`,
`uncertainty_results.json`.

### 10. Explainability (Grad-CAM, LRP, SHAP)

```bash
python -m src.xai.run_xai \
    --dataset-dir data/preprocessed/figshare \
    --model swin_tiny \
    --checkpoint results/figshare/swin_tiny/best.pt \
    --out-dir results/xai/figshare/swin_tiny \
    --methods gradcam lrp shap --n-samples 20
```

Sample selection uses MC-Dropout entropy to surface four
`(confidence, correctness)` groups; `--n-samples` is split equally
across the four. SHAP is skipped automatically if `--shap-background`
training samples are not available.

### 11. Figures

```bash
# All figures for analysis, paper, and thesis
python scripts/generate_figures.py

# Restrict to one group
python scripts/generate_figures.py --only paper
python scripts/generate_figures.py --only analysis paper
```

The three figure-producing scripts live under
`scripts/_figures_lib/`; this is the only public entry point.

### Output directory conventions

```
results/
├── <dataset>/<arch>/          # training (one folder per model, default output root)
│   ├── best.pt                # the trained checkpoint (only one retained)
│   ├── config.yaml
│   ├── epoch_log.csv          # per-epoch metric log
│   ├── final_metrics.json
│   ├── training_curve.png
│   ├── roc_curve.png          # test-set ROC curve
│   ├── pr_curve.png           # test-set PR curve
│   └── confusion_matrix.png   # test-set confusion matrix
├── hpo/<dataset>/<arch>/      # Optuna (single-model) study.db + best_params.yaml
├── calibration/<dataset>/[<arch>|ensemble]/
├── uncertainty/<dataset>/[<arch>|ensemble]/
├── xai/<dataset>/<arch>/
└── eval/<dataset>_<arch>.json # written by src.evaluation.run_evaluate
```

Training only writes one checkpoint (`best.pt`). Per-epoch rolling
checkpoints and raw `.npz` curve arrays are no longer persisted on disk;
the downstream CLIs (calibration, uncertainty, XAI) re-run inference
from `best.pt` as needed.

`src.train` defaults `--output-dir` to `results/<dataset>/` and
writes one `<arch>/` subfolder per `--model` (matching the name you
passed). Downstream CLIs take their own `--out-dir`, so the
training root and the analysis root are independent.

Configs live in `configs/`:

- `configs/preprocessing.yaml` — single unified preprocessing config (steps + augmentation).
- `configs/model/<arch>.yaml` — one YAML per architecture (timm name, head, dropout, freeze).
- `configs/search_space/figshare.yaml`, `configs/search_space/pcosgen.yaml` — Optuna search spaces.

## Tests

```bash
pytest tests/                # full suite (~10s CPU-only, includes smoke)
pytest tests/test_smoke.py   # single Trainer end-to-end on synthetic data
```

The smoke test runs at every CI step. The rest cover the PNG dataloader,
preprocessing materializer, weighted samplers, augmentation, model
builder, calibration/ECE, split overlap, and resume correctness.

## License & dataset

PCOS dataset: Figshare PCOS Ultrasound Dataset (lives in `data/`; not
redistributed here).

This codebase is research software; not a medical device.
