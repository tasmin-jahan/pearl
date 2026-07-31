#!/usr/bin/env python3
"""
CLI entrypoint for XAI analysis: Grad-CAM (+ LRP + SHAP in Figshare mode).

Generates explanations for samples across 4 entropy/correctness groups.

Two modes:

1. Figshare mode:
       python scripts/xai/run_xai.py \
           --model configs/model/swin_tiny.yaml \
           --preprocessing configs/preprocessing/srad.yaml \
           --checkpoint results/checkpoints/srad/swin_tiny.pt \
           --methods gradcam lrp shap --n_samples 20

2. Zenodo mode (--test_dir set, --run_dir set):
       python scripts/xai/run_xai.py \
           --run_dir results/finetune_zenodo/checkpoints/srad_nopad/densenet121 \
           --test_dir data_external/test \
           --method gradcam --n_samples 20

This file replaces both `run_xai.py` (original Figshare-only) and
`run_xai_zenodo.py` (Zenodo-only).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.dataloader import build_dataloaders
from src.data.zenodo_dataset import discover_zenodo_pairs, ZenodoDataset
from src.model.builder import build_model
from src.preprocessing.preprocess import Preprocessor
from src.training.checkpoint import load_checkpoint
from src.uncertainty.mc_dropout import mc_dropout_inference
from src.xai.gradcam import generate_gradcam, save_gradcam_visualization

# Optional dependencies (Figshare mode only).
try:
    from src.xai.lrp import generate_lrp, save_lrp_visualization
except ImportError:
    generate_lrp = save_lrp_visualization = None
try:
    from src.xai.shap_explainer import (
        generate_shap_values, get_background_samples,
        save_shap_visualization, save_mean_shap_summary,
    )
except ImportError:
    generate_shap_values = get_background_samples = save_shap_visualization = save_mean_shap_summary = None


def select_xai_samples(entropy, labels, predictions, n_per_group=5):
    """Select samples from 4 (entropy, correctness) groups."""
    correct = predictions == labels
    median_entropy = np.median(entropy)

    groups = {
        "confident_correct": (entropy <= median_entropy) & correct,
        "overconfident_error": (entropy <= median_entropy) & ~correct,
        "uncertain_correct": (entropy > median_entropy) & correct,
        "uncertain_wrong": (entropy > median_entropy) & ~correct,
    }

    selected = []
    for group_name, mask in groups.items():
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        if group_name.startswith("confident"):
            chosen = indices[np.argsort(entropy[indices])[:n_per_group]]
        else:
            chosen = indices[np.argsort(-entropy[indices])[:n_per_group]]

        for idx in chosen:
            entry = {
                "sample_id": int(idx),
                "true_label": int(labels[idx]),
                "predicted_label": int(predictions[idx]),
                "correct": bool(correct[idx]),
                "entropy_nats": round(float(entropy[idx]), 4),
                "split_group": group_name,
            }
            if generate_lrp is not None:  # Figshare-mode richer annotation
                entry["notes"] = (
                    f"{'low' if entropy[idx] <= median_entropy else 'high'} entropy "
                    f"{'TP' if correct[idx] and labels[idx]==1 else 'TN' if correct[idx] else 'FP' if predictions[idx]==1 else 'FN'}"
                )
            selected.append(entry)
    return selected


def _load_zenodo_model_and_data(args, device):
    """Build model + Zenodo DataLoader from run_dir."""
    cfg_path = os.path.join(args.run_dir, "config.yaml")
    cfg = load_config(cfg_path)
    model_config = cfg["model"]
    preproc_config = cfg["preprocessing"]

    model_config_freeze = dict(model_config)
    model_config_freeze["freeze_fraction"] = 0.0
    model = build_model(model_config_freeze)
    load_checkpoint(
        model, os.path.join(args.run_dir, "best.pt"),
        optimizer=None, scheduler=None, ema=None, device=device,
    )
    model.to(device)
    model.eval()

    pairs = discover_zenodo_pairs(args.test_dir)
    pre = Preprocessor(preproc_config, input_size=model_config.get("input_size", 224))
    ds = ZenodoDataset([p for p, _ in pairs], [l for _, l in pairs], pre, augment=False)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=False,
    )
    return model, loader, pairs, model_config


def _generate_gradcam_panels(model, all_images, selected, out_dir, methods):
    """Generate Grad-CAM panels (and optionally LRP/SHAP in Figshare mode)."""
    background = None
    if "shap" in methods and get_background_samples is not None:
        # SHAP background needs a train loader; Figshare mode supplies one.
        # In Zenodo mode this branch is skipped.
        pass  # caller-provided if Figshare mode

    all_shap_values = []
    for info in selected:
        sid = info["sample_id"]
        img_tensor = all_images[sid].unsqueeze(0)
        img_np = all_images[sid].permute(1, 2, 0).numpy()
        print(f"  Sample {sid} ({info['split_group']})")

        if "gradcam" in methods:
            heatmap = generate_gradcam(model, img_tensor, device=next(model.parameters()).device)
            save_gradcam_visualization(
                img_np, heatmap,
                os.path.join(out_dir, "gradcam"), sid,
            )

        if "lrp" in methods and generate_lrp is not None:
            relevance = generate_lrp(model, img_tensor, device=next(model.parameters()).device)
            save_lrp_visualization(relevance, os.path.join(out_dir, "lrp"), sid)

        if "shap" in methods and generate_shap_values is not None and background is not None:
            sv = generate_shap_values(
                model, img_tensor, background,
                device=next(model.parameters()).device,
            )
            save_shap_visualization(sv, img_np, os.path.join(out_dir, "shap"), sid)
            all_shap_values.append(sv)

    if all_shap_values and save_mean_shap_summary is not None:
        save_mean_shap_summary(
            all_shap_values,
            os.path.join(out_dir, "shap", "mean_shap_summary.png"),
        )


def main():
    parser = argparse.ArgumentParser(description="Run XAI analysis")
    # Figshare-mode args
    parser.add_argument("--model", type=str, default=None,
                        help="Figshare mode: model config YAML")
    parser.add_argument("--preprocessing", type=str, default=None,
                        help="Figshare mode: preprocessing config YAML")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Figshare mode: path to checkpoint .pt")
    # Common args
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Run dir containing config.yaml + best.pt "
                             "(required in Zenodo mode; output-only in Figshare mode)")
    parser.add_argument("--test_dir", type=str, default=None,
                        help="Zenodo mode: external test directory (e.g. data_external/test). "
                             "If set, switches to Zenodo mode.")
    parser.add_argument("--methods", "--method", dest="methods", nargs="+", default=["gradcam"])
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--mc_passes", "--n_passes", type=int, default=50, dest="mc_passes")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.test_dir is not None:
        # --- Zenodo mode ---
        if args.run_dir is None:
            parser.error("--run_dir is required when --test_dir is set")
        out_dir = os.path.join(args.run_dir, "xai")
        os.makedirs(out_dir, exist_ok=True)

        model, loader, pairs, _ = _load_zenodo_model_and_data(args, device)

        print(f"[XAI] Running {args.mc_passes} MC Dropout passes for sample selection...")
        mean_probs, entropy = mc_dropout_inference(
            model, loader, n_passes=args.mc_passes, device=device,
        )

        all_labels = np.array([l for _, l in pairs])
        all_images = []
        for x, _ in loader:
            all_images.append(x)
        all_images = torch.cat(all_images, dim=0)
        predictions = mean_probs.numpy().argmax(axis=1)
        entropy_np = entropy.numpy()

        n_per_group = max(1, args.n_samples // 4)
        selected = select_xai_samples(entropy_np, all_labels, predictions, n_per_group)
        pd.DataFrame(selected).to_csv(os.path.join(out_dir, "xai_metadata.csv"), index=False)
        print(f"[XAI] Selected {len(selected)} samples across 4 groups (target={args.n_samples})")

        _generate_gradcam_panels(model, all_images, selected, out_dir, args.methods)
        print(f"[XAI] Saved to {out_dir}")

    else:
        # --- Figshare mode ---
        if not (args.model and args.preprocessing and args.checkpoint):
            parser.error(
                "Figshare mode requires --model, --preprocessing, --checkpoint "
                "(or use --test_dir for Zenodo mode)"
            )
        model_config = load_config(args.model)
        preproc_config = load_config(args.preprocessing)
        if args.run_dir:
            out_dir = os.path.join(args.run_dir, "xai")
        else:
            out_dir = os.path.join(
                "results", "xai",
                f"{model_config['name']}__{preproc_config['name']}",
            )
        os.makedirs(out_dir, exist_ok=True)

        input_size = model_config.get("input_size", 224)
        train_loader, _, test_loader = build_dataloaders(
            preproc_config, batch_size=32, input_size=input_size,
        )

        model = build_model(model_config)
        load_checkpoint(model, args.checkpoint, device=device)

        print("[XAI] Running MC Dropout for sample selection...")
        mean_probs, entropy = mc_dropout_inference(
            model, test_loader, n_passes=args.mc_passes, device=device,
        )

        all_labels = []
        all_images = []
        for images, labels in test_loader:
            all_images.append(images)
            if isinstance(labels, torch.Tensor):
                all_labels.extend(labels.numpy())
            else:
                all_labels.extend(labels)

        all_labels = np.array(all_labels)
        all_images = torch.cat(all_images, dim=0)
        predictions = mean_probs.numpy().argmax(axis=1)
        entropy_np = entropy.numpy()

        n_per_group = args.n_samples // 4
        selected = select_xai_samples(entropy_np, all_labels, predictions, n_per_group)
        pd.DataFrame(selected).to_csv(os.path.join(out_dir, "xai_metadata.csv"), index=False)
        print(f"[XAI] Selected {len(selected)} samples across 4 groups")

        background = None
        if "shap" in args.methods and get_background_samples is not None:
            background = get_background_samples(train_loader, n_samples=100).to(device)

        _generate_gradcam_panels(model, all_images, selected, out_dir, args.methods)
        print(f"\n[XAI] Results saved to {out_dir}")


if __name__ == "__main__":
    main()
