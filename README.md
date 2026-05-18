# PEARL

Probabilistic Explainability with Adaptive Reliability via transfer Learning

# PCOS Detection from Ultrasound Images

A config-driven deep learning pipeline for PCOS (Polycystic Ovary Syndrome) detection from ovarian ultrasound images, with model comparison, explainability, calibration, and uncertainty quantification.

## Project Structure

```
pcos-detection/
├── configs/                     # All experiment configs (YAML)
│   ├── preprocessing/           # 6 preprocessing pipelines
│   ├── model/                   # 9 architecture configs
│   └── experiment/              # Sweep, tuning, and XAI configs
├── src/                         # Source modules
│   ├── data/                    # Dataset, splitter, dataloader
│   ├── preprocessing/           # Image preprocessing pipeline
│   ├── model/                   # Model builder + classification head
│   ├── training/                # Trainer, losses, checkpoints, tuner
│   ├── evaluation/              # Metrics + evaluator
│   ├── calibration/             # ECE, reliability diagrams, temperature scaling
│   ├── uncertainty/             # MC Dropout, referral system
│   ├── xai/                     # Grad-CAM, LRP, SHAP
│   └── utils/                   # Config, seed, logging
├── scripts/                     # CLI entrypoints
│   ├── preprocess.py            # Preprocess raw dataset
│   ├── train.py                 # Single training run
│   ├── sweep.py                 # Full 54-run sweep
│   ├── evaluate.py              # Standalone evaluation
│   ├── tune.py                  # Optuna hyperparameter tuning
│   ├── run_calibration.py       # Calibration analysis
│   ├── run_uncertainty.py       # MC Dropout + referral system
│   └── run_xai.py               # Grad-CAM + LRP + SHAP
└── results/                     # All outputs (auto-created)
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Download the [Figshare PCOS Ultrasound Dataset](https://figshare.com/) and organize as:

```
data/
  infected/        # PCOS images (6784)
  notinfected/     # Non-PCOS images (5000)
```

## Usage

### 1. Preprocess

```bash
python scripts/preprocess.py \
    --config configs/preprocessing/full_ad.yaml \
    --data_dir /path/to/data \
    --split_seed 42
```

### 2. Train a single model

```bash
python scripts/train.py \
    --model configs/model/efficientnet_b4.yaml \
    --preprocessing configs/preprocessing/full_ad.yaml \
    --experiment configs/experiment/best_model_xai.yaml
```

### 3. Run the full 54-run sweep

```bash
python scripts/sweep.py --experiment configs/experiment/sweep_all_54.yaml
```

### 4. Hyperparameter tuning (best model only)

```bash
python scripts/tune.py \
    --experiment configs/experiment/tune_best.yaml \
    --study_name efficientnet_b4_full_ad \
    --storage sqlite:///results/tuning/optuna.db
```

### 5. Calibration analysis

```bash
python scripts/run_calibration.py \
    --model configs/model/efficientnet_b4.yaml \
    --preprocessing configs/preprocessing/full_ad.yaml \
    --checkpoint results/checkpoints/efficientnet_b4__full_ad.pt
```

### 6. Uncertainty quantification

```bash
python scripts/run_uncertainty.py \
    --model configs/model/efficientnet_b4.yaml \
    --preprocessing configs/preprocessing/full_ad.yaml \
    --checkpoint results/checkpoints/efficientnet_b4__full_ad.pt \
    --mc_passes 50
```

### 7. Explainability (XAI)

```bash
python scripts/run_xai.py \
    --model configs/model/efficientnet_b4.yaml \
    --preprocessing configs/preprocessing/full_ad.yaml \
    --checkpoint results/checkpoints/efficientnet_b4__full_ad.pt \
    --methods gradcam lrp shap \
    --n_samples 20
```

## Key Design Decisions

- **Config-driven**: Swap YAML configs to run any experiment — no code changes needed
- **9 architectures**: VGG16/19, ResNet50/101, DenseNet121/169, EfficientNet-B0/B4, Inception V3
- **6 preprocessing pipelines**: Raw, CLAHE, CLAHE+Gaussian, CLAHE+AD, Full+Gaussian, Full+AD
- **Class-weighted loss**: Handles the 6784:5000 class imbalance
- **Stratified 70/15/15 split**: Preserves class distribution
- **Early stopping**: On validation AUC-ROC with patience=20
- **MC Dropout**: 50 stochastic forward passes for uncertainty
- **Temperature scaling**: Post-hoc calibration on validation set
- **XAI across 4 groups**: Confident-correct, overconfident-error, uncertain-correct, uncertain-wrong

## Naming Convention

All outputs use `{architecture}__{preprocessing}` format:
- `efficientnet_b4__full_ad.pt`
- `resnet50__clahe_only_metrics.csv`
