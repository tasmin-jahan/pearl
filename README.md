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

## Usage

All entrypoints are Python modules, run from the repo root. Every CLI
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
    --output-dir results/training/figshare/swin_tiny

# Optional: load a checkpoint to initialize weights (fine-tune mode)
python -m src.train \
    --dataset-dir data/preprocessed/figshare \
    --model swin_tiny \
    --checkpoint results/training/figshare/swin_tiny/best.pt \
    --output-dir results/training/figshare/swin_tiny
```

`--checkpoint` initializes the model weights from `best.pt` but
**does not** carve a frozen fine-tune mode — the same trainer loop
runs in both cases. The full hyperparameter surface is exposed via
flags (`--lr`, `--weight-decay`, `--dropout`, `--freeze-fraction`,
`--epochs`, `--patience`, `--batch-size`, `--sampler`).

### 5a. Train (multiple models in one invocation)

`--model` is repeatable. Pass each architecture once and they run
sequentially, each in its own `<output-dir>/<dataset>/<arch>/`
subdirectory. All other hyperparameters are shared.

```bash
# Train all five supported architectures from scratch
python -m src.train \
    --dataset-dir data/preprocessed/figshare \
    --model swin_tiny --model vit_base --model convnext_tiny \
    --model densenet169 --model efficientnet_b0
```

### 5b. Fine-tune multiple models at once

Two ways:

```bash
# (a) Same checkpoint for every model (rare)
python -m src.train \
    --dataset-dir data/preprocessed/pcosgen \
    --model swin_tiny --model vit_base \
    --checkpoint results/training/figshare/swin_tiny/best.pt

# (b) Each model loads its own matching checkpoint — typical fine-tune
#     workflow: foundation pass already produced best.pt under
#     results/training/<dataset>/<arch>/ for every arch.
python -m src.train \
    --dataset-dir data/preprocessed/pcosgen \
    --model swin_tiny --model vit_base --model convnext_tiny \
    --model densenet169 --model efficientnet_b0 \
    --sweep-checkpoints results/training/figshare
```

With `--sweep-checkpoints <root>`, each model looks for
`<root>/<dataset>/<model>/best.pt`. If a model's checkpoint is
missing under the root, that model falls back to training from
scratch (with a printed warning) — so partial sweeps are safe.

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

### 7. Evaluate a single checkpoint

```bash
python -m src.evaluation.run_evaluate \
    --checkpoint-dir results/figshare/swin_tiny \
    --test-dataset-dir data/preprocessed/figshare \
    --model-config configs/model/swin_tiny.yaml
```

Writes one JSON of metrics to `<checkpoint-dir>/metrics.json` by
default. Call once per model — no cross-checkpoint aggregation.

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
├── training/<dataset>/<arch>/   # best.pt, metrics.json, training_curves.png, config.yaml
├── hpo/<dataset>/<arch>/        # best params, study.db, top-trial checkpoint
├── calibration/<dataset>/[<arch>|ensemble]/
├── uncertainty/<dataset>/[<arch>|ensemble]/
├── xai/<dataset>/<arch>/
└── eval/<dataset>_<arch>.json   # written by src.evaluation.run_evaluate
```

Downstream CLIs (`run_calibration`, `run_uncertainty`, `xai.run_xai`,
`evaluation.run_evaluate`) take a checkpoint path or `--checkpoint-dir`
and an `--out-dir` separately, so you can keep the training tree under
`results/training/<dataset>/<arch>/` and write the analysis outputs
anywhere (e.g. `results/calibration/<dataset>/<arch>/`). The two
roots are independent.

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
