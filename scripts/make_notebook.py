#!/usr/bin/env python3
"""Generate notebooks/pearl_kaggle_colab.ipynb."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_FILE = NOTEBOOK_DIR / "pearl_kaggle_colab.ipynb"

cells = []

def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.splitlines()]
    }

def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in src.splitlines()]
    }

# Cell 1: Intro
cells.append(md("""# PEARL: Probabilistic Explainability with Adaptive Reliability via Transfer Learning
## End-to-End Execution Notebook (Optimized for Kaggle / Google Colab Nvidia T4 16GB GPU)

This notebook executes the complete PEARL research pipeline step-by-step:
1. **Environment Setup & GPU Verification** (Detects environment, T4 FP16 Tensor Cores)
2. **Data Preparation & Preprocessing** (De-duplication, Stratified Splitting, SRAD/CLAHE)
3. **Transfer Learning Training** (Individual & Sequential models with EMA and Mixed Precision)
4. **Model Evaluation** (In-distribution and zero-shot cross-dataset evaluation)
5. **Probability Calibration** (Two-pass Temperature Scaling & Reliability Diagrams)
6. **Ensemble Aggregation** (Calibrated probability-averaged ensemble)
7. **MC-Dropout Uncertainty Quantification** (Predictive entropy & Risk-Coverage curves)
8. **Explainable AI (XAI)** (Grad-CAM, LRP, SHAP attribution panels)
9. **Publication Figure Generation** (Paper figures, thesis tables, and analysis plots)
10. **Automated Artifact Packaging & Download** (Zip & direct browser download)"""))

# Cell 2: Environment Detection
cells.append(md("--- \n### 1. Environment Detection & Hardware Verification"))
cells.append(code("""import os
import sys
import platform
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
IN_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

print(f"Platform: {platform.system()} ({platform.release()})")
print(f"Colab Environment: {IN_COLAB}")
print(f"Kaggle Environment: {IN_KAGGLE}")

# Verify PyTorch and CUDA
import torch
print(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Active GPU: {device_name} ({vram_gb:.1f} GB VRAM)")
    if "T4" in device_name:
        print("[TIP] Detected Nvidia T4 GPU: using --fp16 for optimal Tensor Core throughput.")
else:
    print("[NOTE] CUDA not detected; running in CPU mode.")"""))

# Cell 3: Dependencies
cells.append(md("--- \n### 2. Dependency Installation & Verification"))
cells.append(code("""# Install dependencies
!pip install -q timm torchcam scikit-learn optuna zennit shap medpy opencv-python pyyaml pandas matplotlib pytest

# Ensure repository root is in sys.path
REPO_ROOT = Path(".").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Run Stage 01 Environment Verification
!python scripts/01_setup_env.py"""))

# Cell 4: Data Prep
cells.append(md("--- \n### 3. Data Unpacking, Deduplication & Preprocessing\n"
"Unpacks `data/PCOS.zip`, `data/updated train dataset.zip`, and `data/updated test dataset.zip`,\n"
"runs MD5 + perceptual hash deduplication, carves stratified splits, and materializes preprocessed PNGs."))
cells.append(code("""# Run Stage 02 Data Preparation
!python scripts/02_prepare_data.py"""))

# Cell 5: Training
cells.append(md("--- \n### 4. Transfer Learning Model Training (T4 16GB VRAM Optimized)\n"
"Train individual models or all five architectures (`convnext_tiny`, `densenet169`, `efficientnet_b0`, `swin_tiny`, `vit_base`)."))
cells.append(code('''\"\"\"
PEARL Architecture Selection & Model Training Guide
===================================================
You can train any of the 5 supported transfer-learning architectures individually,
or train all 5 models sequentially in an end-to-end sweep.

Supported Models (--model <name>):
----------------------------------
1. 'convnext_tiny'   : ConvNeXt-Tiny (~28.6M params)
                       - Modern Pure Convolutional Neural Network (ConvNet)
                       - 7x7 depthwise convolutions with inverted bottleneck design
                       - Recommended T4 batch size: 32 or 64

2. 'densenet169'     : DenseNet-169 (~14.1M params)
                       - Densely Connected Convolutional Network
                       - High feature reuse across dense blocks; gold-standard medical imaging baseline
                       - Recommended T4 batch size: 32 or 64

3. 'efficientnet_b0' : EfficientNet-B0 (~5.3M params)
                       - Neural Architecture Search (NAS) compound scaled CNN
                       - Lightweight inverted residuals with Squeeze-and-Excitation (SE) attention
                       - Recommended T4 batch size: 32 or 64

4. 'swin_tiny'       : Swin Transformer Tiny (~28.3M params)
                       - Hierarchical Vision Transformer with Shifted Windows
                       - Local window self-attention with cross-window shifted connections
                       - Recommended T4 batch size: 32

5. 'vit_base'        : Vision Transformer Base (~86.6M params)
                       - Non-hierarchical Pure Vision Transformer (16x16 patch projection)
                       - Global receptive field with 12 multi-head self-attention layers
                       - Recommended T4 batch size: 16 or 32 (use 16 or 32 with --fp16 to avoid OOM)

6. 'all'             : Trains all 5 architectures sequentially in a single pass.
                       - Saves checkpoints to: results/<dataset_name>/<model_name>/best.pt

Hardware & Throughput Flags (T4 GPU 16GB VRAM):
----------------------------------------------
- --fp16            : Enables FP16 mixed precision (native Tensor Cores on Turing T4).
- --batch-size 32   : Fits 16GB VRAM with headroom across all models.
- --epochs 50       : Default training epochs (early stopping patience=20).
- --dataset-dir     : Target preprocessed data (e.g. data/preprocessed/figshare or pcosgen).
\"\"\"

# Option A: Train a single model individually (e.g. ConvNeXt-Tiny)
!python scripts/03_train_models.py --model convnext_tiny --epochs 50 --batch-size 32 --fp16

# Option B: Train Vision Transformer Base (ViT-B)
# !python scripts/03_train_models.py --model vit_base --epochs 50 --batch-size 16 --fp16

# Option C: Train all 5 architectures sequentially
# !python scripts/03_train_models.py --model all --epochs 50 --batch-size 32 --fp16'''))

# Cell 6: Evaluation
cells.append(md("--- \n### 5. In-Distribution & Cross-Dataset Evaluation"))
cells.append(code("""# In-distribution evaluation on Figshare test set:
!python scripts/04_evaluate_models.py --model-dir results/figshare --test-dataset-dir data/preprocessed/figshare/test

# Zero-shot cross-dataset evaluation on PCOSgen test set:
if Path("data/preprocessed/pcosgen/test").exists():
    !python scripts/04_evaluate_models.py --model-dir results/figshare --test-dataset-dir data/preprocessed/pcosgen/test --output-dir results/evaluation/zero_shot_pcosgen"""))

# Cell 7: Calibration
cells.append(md("--- \n### 6. Probability Calibration (Temperature Scaling)"))
cells.append(code("""# Run temperature scaling calibration across all trained checkpoints:
!python scripts/05_calibration_analysis.py --model-dir results/figshare --dataset-dir data/preprocessed/figshare"""))

# Cell 8: Ensemble
cells.append(md("--- \n### 7. Ensemble Creation & Evaluation"))
cells.append(code("""# Aggregate trained members into calibrated probability-averaged ensemble:
!python scripts/06_create_ensemble.py --model-dir results/figshare --test-dataset-dir data/preprocessed/figshare/test --output results/ensemble/figshare_ensemble_metrics.json"""))

# Cell 9: Uncertainty
cells.append(md("--- \n### 8. MC-Dropout Uncertainty Quantification (50 Stochastic Passes)"))
cells.append(code("""# Run MC-Dropout inference and compute predictive entropy distributions:
!python scripts/07_uncertainty_analysis.py --model-dir results/figshare --dataset-dir data/preprocessed/figshare --mc-passes 50"""))

# Cell 10: XAI
cells.append(md("--- \n### 9. Explainable AI (XAI) Attribution Panels"))
cells.append(code("""# Generate Grad-CAM heatmaps across confident/uncertain error quadrants:
!python scripts/08_xai_analysis.py --model-dir results/figshare --dataset-dir data/preprocessed/figshare --methods gradcam --n-samples 20"""))

# Cell 11: Figures
cells.append(md("--- \n### 10. Generate Paper, Thesis & Analysis Figures"))
cells.append(code("""# Render all figures and LaTeX tables into docs/:
!python scripts/09_generate_figures.py --only all"""))

# Cell 12: Display figures
cells.append(md("--- \n### 11. Visual Inspection of Generated Figures"))
cells.append(code("""from IPython.display import Image, display

figures_to_show = [
    "docs/latex/figures/paper_fig_combined_roc.png",
    "docs/latex/figures/paper_fig_combined_pr.png",
    "docs/latex/figures/paper_fig_calibration_compare.png",
    "docs/latex/figures/paper_fig_uncertainty_compare.png",
    "docs/thesis/figures/thesis_per_class_pr.png",
    "docs/thesis/figures/thesis_threshold_sweep.png",
]

for p in figures_to_show:
    if Path(p).exists():
        print(f"Displaying: {p}")
        display(Image(filename=p, width=600))"""))

# Cell 13: Bundle and download
cells.append(md("--- \n### 12. Bundle Checkpoints & Results for Download\n"
"Zips model checkpoints (`results/`) and generated documentation/figures (`docs/`) for easy download."))
cells.append(code("""import shutil

print("Compressing experiment artifacts...")
if Path("results").exists():
    shutil.make_archive("pearl_results", "zip", "results")
    print("[OK] Created pearl_results.zip")

if Path("docs").exists():
    shutil.make_archive("pearl_figures_and_tables", "zip", "docs")
    print("[OK] Created pearl_figures_and_tables.zip")

# Direct browser download in Google Colab:
if IN_COLAB:
    from google.colab import files
    if Path("pearl_results.zip").exists():
        files.download("pearl_results.zip")
    if Path("pearl_figures_and_tables.zip").exists():
        files.download("pearl_figures_and_tables.zip")
else:
    print("Files ready for download in current working directory / Kaggle output pane:")
    for z in ("pearl_results.zip", "pearl_figures_and_tables.zip"):
        if Path(z).exists():
            size_mb = Path(z).stat().st_size / (1024 * 1024)
            print(f"  - {z} ({size_mb:.2f} MB)")"""))

nb_structure = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(NOTEBOOK_FILE, "w", encoding="utf-8") as f:
    json.dump(nb_structure, f, indent=2)

print(f"Successfully wrote {NOTEBOOK_FILE}")
