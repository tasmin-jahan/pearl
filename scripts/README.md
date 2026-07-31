# PEARL Scripts

All entrypoints for training, evaluation, and analysis. Folderised by task.

## Layout

```
scripts/
├── smoke/           Sanity checks before long sweeps
├── data/            Dataset preparation (dedup, splits, layout conversion)
├── preprocessing/   Apply preprocessing pipelines and cache outputs
├── train/           Single-model training + matrix/Optuna HPO sweeps
├── finetune/        Fine-tune foundation checkpoints on Zenodo PCOSgen
├── evaluation/      Internal/external validation + ablation comparisons
├── calibration/     Temperature scaling + reliability diagrams
├── uncertainty/     MC-Dropout inference + predictive entropy
├── xai/             Grad-CAM (+ LRP, SHAP) attributions
├── shortcut_audit/  Border-occlusion experiments that proved the
│                    letterbox-padding shortcut (Findings 8/9)
└── figures/         All analysis-section + paper figure generators
```

## Index

### `smoke/`
- `smoke_test.py` — 15 s, 2-epoch sanity check on 10 synthetic images. Run before every long sweep.

### `data/`
- `dedup_data.py` — md5 dedup on Figshare PCOS folder layout.
- `build_zenodo_splits.py` — reproducible 80/20 stratified Zenodo PCOSgen splits.
- `prepare_pcosgen_layout.py` — convert PCOSgen Kaggle folder into `discover_pcosgen` format.

### `preprocessing/`
- `preprocess.py` — CLI to apply a preprocessing pipeline and write train/val/test to disk.

### `train/`
- `train.py` — single-model trainer. Supports `--resume` for fine-tune.
- `sweep.py` — matrix sweep driver (e.g. 9 archs × 2 preps = 18 runs). Skip-if-best.pt.
- `sweep_hpo.py` — full Optuna sweep across all 9 architectures.
- `tune.py` — standalone Optuna HPO CLI for one experiment config.

### `finetune/`
- `sweep_finetune.py` — fine-tunes all 18 Figshare no-pad checkpoints on Zenodo PCOSgen.
- `sweep_hpo_finetune.py` — Optuna HPO for one fine-tune (prep, arch) pair.
- `run_hpo_finetune.py` — runs HPO + retrain + re-eval for top-3 finalists.
- `retrain_with_hpo.py` — retrain a checkpoint with its best Optuna params.
- `run_finetune_ensemble.py` — probability-averaged ensemble of top-K fine-tuned checkpoints.
- `run_top3_finetune_pipeline.py` — top-level orchestrator: HPO → retrain → eval → aggregate for top-3.
- `post_eval_finetune.py` — runs calibration + uncertainty + XAI on each fine-tuned checkpoint.
- `eval_finetune_zenodo.py` — aggregates Zenodo external-eval JSON/CSV across all fine-tuned runs.

### `evaluation/`
- `evaluate.py` — single-model evaluator (CLI for internal val/test).
- `evaluate_external.py` — full external-eval driver (both PCOSgen layouts).
- `eval_external_all.py` — driver: runs `evaluate_external.py` for every (prep, arch) pair.
- `eval_external_noproc.py` — external-eval variant with NO preprocessing (resize + ImageNet norm only).
- `eval_external_noproc_all.py` — driver for `eval_external_noproc.py` across all checkpoints.
- `kfold_finalists.py` — k-fold CV on top-k HPO finalists.

### `calibration/`
- `run_calibration.py` — ECE + temperature scaling + reliability diagrams. **Two modes** via `--test_dir`:
  - Figshare mode (default): uses train/val/test split.
  - Zenodo mode (`--test_dir data_external/test`): 50/50 split inside test set for honest post-T ECE.
- `run_calibration_ensemble.py` — two-pass ensemble calibration (per-model T, then ensemble T). Same two modes.

### `uncertainty/`
- `run_uncertainty.py` — MC-Dropout (50 passes) + predictive entropy + referral curve. **Two modes** via `--test_dir`.

### `xai/`
- `run_xai.py` — Grad-CAM (+ LRP, SHAP in Figshare mode) attributions on 20 stratified samples. **Two modes** via `--test_dir`.

### `shortcut_audit/`
- `eval_border_occluded.py` — border-occlusion inference ablation (median-replace + cross-label swap).
- `gradcam_shortcut_audit.py` — Grad-CAM border-attention fraction for one checkpoint.
- `gradcam_shortcut_audit_all.py` — driver for `gradcam_shortcut_audit.py` across all checkpoints.
- `generate_border_occlusion_figures.py` — figures for Finding 8 (border-occlusion causal ablation).
- `generate_shortcut_figures.py` — fig19/20/21 from per-run shortcut summary JSONs.
- `generate_shortcut_figures_nopad.py` — fig25: baseline vs no-padding external AUC.
- `generate_nopad_gradcam_figures.py` — fig27/28: border-attention before/after + side-by-side Grad-CAM.

### `figures/`
- `generate_analysis_figures.py` — master analysis-section figure generator (Findings 1–28).
- `thesis_figures.py` — thesis chapter figures + LaTeX table fragments regenerated from results.
- `make_paper_figures.py` — paper-specific combined ROC/PR/calibration/Grad-CAM panels.

## Common invocation patterns

```bash
# Single-model training (foundation pass)
python scripts/train/train.py \
    --experiment configs/experiment/ablation.yaml \
    --set preprocessing.name=srad_nopad model.name=densenet121

# Foundation sweep (9 archs × 2 preps)
python scripts/train/sweep.py \
    --experiment configs/experiment/ablation.yaml \
    --preprocessings srad_nopad gauss_nopad \
    --architectures resnet50 resnet101 densenet121 densenet169 \
                    efficientnet_b0 convnext_tiny mobilenetv3_large vit_base swin_tiny

# Fine-tune on Zenodo
python scripts/finetune/sweep_finetune.py \
    --experiment configs/experiment/finetune_zenodo.yaml

# External validation (Figshare)
python scripts/evaluation/evaluate_external.py \
    --run_dir results/ablation_nopad/checkpoints/srad_nopad/densenet121 \
    --external_dir data_external/test --layout zenodo_labeled

# External validation driver (all checkpoints)
python scripts/evaluation/eval_external_all.py \
    --checkpoints_root results/ablation_nopad/checkpoints

# Calibration (Figshare mode)
python scripts/calibration/run_calibration.py \
    --run_dir results/ablation_nopad/checkpoints/srad_nopad/densenet121

# Calibration (Zenodo mode: 50/50 split, honest ECE)
python scripts/calibration/run_calibration.py \
    --run_dir results/finetune_zenodo/checkpoints/srad_nopad/densenet121 \
    --test_dir data_external/test

# MC-Dropout uncertainty
python scripts/uncertainty/run_uncertainty.py \
    --run_dir results/finetune_zenodo/checkpoints/srad_nopad/densenet121 \
    --test_dir data_external/test --n_passes 50

# Grad-CAM XAI (20 samples across 4 entropy/correctness groups)
python scripts/xai/run_xai.py \
    --run_dir results/finetune_zenodo/checkpoints/srad_nopad/densenet121 \
    --test_dir data_external/test --n_samples 20

# Top-3 ensemble
python scripts/finetune/run_finetune_ensemble.py \
    --run_dirs results/finetune_zenodo/checkpoints/{srad_nopad/densenet121,srad_nopad/convnext_tiny,gauss_nopad/vit_base}

# Two-pass ensemble calibration
python scripts/calibration/run_calibration_ensemble.py \
    --run_dirs results/finetune_zenodo/checkpoints/{srad_nopad/densenet121,srad_nopad/convnext_tiny,gauss_nopad/vit_base} \
    --test_dir data_external/test

# Paper figures
python scripts/figures/make_paper_figures.py
```

## Path conventions

- All scripts use `sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))` to find the `src/` package.
- `src/` is already folderised: `src/{calibration,data,evaluation,model,preprocessing,training,uncertainty,xai,utils}/`.
- `configs/` is folderised: `configs/{experiment,model,preprocessing}/`.
