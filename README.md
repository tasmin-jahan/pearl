# PEARL

Probabilistic Explainability with Adaptive Reliability via transfer Learning

A config-driven deep learning pipeline for PCOS (Polycystic Ovary Syndrome) detection from ovarian ultrasound images, with model comparison, explainability, calibration, and uncertainty quantification.

## Where to look

- **How to run the pipeline** → [`docs/usage.md`](docs/usage.md) — all CLI commands, config schemas, output layouts
- **Methodology** → [`docs/methodology.tex`](docs/methodology.tex) — full pipeline redesign writeup (LaTeX)

## What's in this codebase (v3)

A 7-phase pipeline, runnable end-to-end from YAML configs:

| Phase | Goal | Entry point |
|-------|------|-------------|
| 0 | Decide SRAD vs. Gaussian denoising (9 × 2 = 18 runs, no HPO) | `scripts/sweep.py --experiment ablation_18.yaml` |
| 1 | Single training run (debug, ablation inspection) | `scripts/train.py` |
| 2 | Per-architecture Optuna HPO + top-k finalists | `scripts/sweep_hpo.py` |
| 3 | k-fold CV on top-k finalists only | `scripts/kfold_finalists.py` |
| 4 | SWA + probability-averaging ensemble | `src/training/swa.py`, `src/evaluation/ensemble.py` |
| 5 | Two-pass calibration (per-model T + ensemble T) | `scripts/run_calibration*.py` |
| 6 | XAI (Grad-CAM, LRP, SHAP) and uncertainty (MC Dropout) | `scripts/run_xai.py`, `scripts/run_uncertainty.py` |

The v3 redesign (vs. v1/v2) added: 80/10/10 split with patient-level
leakage check, SRAD replacing generic AD, SiLU head with dynamic
`num_classes`, no-decay parameter groups, NaN divergence guard, LR
warmup + two-phase fine-tuning, EMA + SWA, two-pass calibration,
self-contained checkpoints with RNG state for resume. See
`docs/methodology.tex` for the full design rationale.

## Project layout (top level)

```
pearl/
├── configs/
│   ├── preprocessing/      # 8 configs (incl. v3 standard: srad, gauss)
│   ├── model/              # 9 v3 architectures (ResNet, DenseNet, EfficientNet-B0,
│   │                       #   ConvNeXt-T, MobileNetV3-L, ViT-B, Swin-T)
│   └── experiment/         # ablation_18, tune_per_arch, kfold_finalists, ...
├── src/                    # data / preprocessing / model / training / evaluation /
│                           # calibration / uncertainty / xai / utils
├── scripts/                # CLI entrypoints (preprocess, train, sweep, sweep_hpo,
│                           # kfold_finalists, smoke_test, run_xai, ...)
├── tests/                  # pytest suite (config overrides, split overlap, resume)
├── docs/
│   ├── usage.md            # full how-to-run
│   └── methodology.tex     # pipeline redesign paper (LaTeX)
└── requirements.txt
```

## Quick start (one-liner summary)

```bash
# 1. Setup (see docs/usage.md for full install)
pip install -r requirements.txt

# 2. Smoke test (~15s, no GPU needed)
python scripts/smoke_test.py

# 3. Deduplicate the raw dataset (the Figshare download ships with
#    ~83% byte-duplicate noninfected and ~53% infected files — see
#    notebooks/eda_figures and docs/methodology.tex). Reversible:
python scripts/dedup_data.py --data_dir /path/to/data --mode report     # audit only
python scripts/dedup_data.py --data_dir /path/to/data --mode quarantine # move to data/_duplicates/

# 4. Preprocess once
python scripts/preprocess.py \
    --config configs/preprocessing/srad.yaml \
    --data_dir /path/to/data

# 5. Run the full v3 pipeline (Phases 0 → 6)
python scripts/sweep.py --experiment configs/experiment/ablation_18.yaml
python scripts/sweep_hpo.py --experiment configs/experiment/tune_per_arch.yaml
python scripts/kfold_finalists.py \
    --experiment configs/experiment/kfold_finalists.yaml \
    --finalists results/sweep_hpo/finalists.csv \
    --params_dir results/sweep_hpo/ \
    --preprocessing srad \
    --data_dir /path/to/data
# (then run downstream XAI / uncertainty / calibration)
```

See [`docs/usage.md`](docs/usage.md) for the full pipeline reference.

## License & dataset

PCOS dataset: Figshare PCOS Ultrasound Dataset (in the `data/` directory
of the original project; not redistributed here).

This codebase is research software; not a medical device.
