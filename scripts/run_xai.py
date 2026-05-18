#!/usr/bin/env python3
"""
CLI entrypoint for XAI analysis: Grad-CAM + LRP + SHAP.

Generates explanations for 20 samples across 4 groups:
  1. Low entropy + correct    (5 samples)
  2. Low entropy + wrong      (5 samples)
  3. High entropy + correct   (5 samples)
  4. High entropy + wrong     (5 samples)

Usage:
    python scripts/run_xai.py \
        --model configs/model/efficientnet_b4.yaml \
        --preprocessing configs/preprocessing/full_ad.yaml \
        --checkpoint results/checkpoints/efficientnet_b4__full_ad.pt \
        --methods gradcam lrp shap \
        --n_samples 20
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.dataloader import build_dataloaders
from src.model.builder import build_model
from src.training.checkpoint import load_checkpoint
from src.uncertainty.mc_dropout import mc_dropout_inference
from src.xai.gradcam import generate_gradcam, save_gradcam_visualization
from src.xai.lrp import generate_lrp, save_lrp_visualization
from src.xai.shap_explainer import (
    generate_shap_values, get_background_samples,
    save_shap_visualization, save_mean_shap_summary,
)


def select_xai_samples(entropy, labels, predictions, n_per_group=5):
    """Select samples from 4 groups for XAI analysis."""
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
            selected.append({
                "sample_id": int(idx),
                "true_label": int(labels[idx]),
                "predicted_label": int(predictions[idx]),
                "correct": bool(correct[idx]),
                "entropy_nats": round(float(entropy[idx]), 4),
                "split_group": group_name,
                "notes": f"{'low' if entropy[idx] <= median_entropy else 'high'} entropy "
                         f"{'TP' if correct[idx] and labels[idx]==1 else 'TN' if correct[idx] else 'FP' if predictions[idx]==1 else 'FN'}",
            })

    return selected


def main():
    parser = argparse.ArgumentParser(description="Run XAI analysis")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--preprocessing", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--methods", type=str, nargs="+", default=["gradcam", "lrp", "shap"])
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--mc_passes", type=int, default=50)
    args = parser.parse_args()

    model_config = load_config(args.model)
    preproc_config = load_config(args.preprocessing)
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    arch = model_config["name"]
    preproc_name = preproc_config["name"]

    out_dir = os.path.join("results", "xai", f"{arch}__{preproc_name}")
    os.makedirs(out_dir, exist_ok=True)

    # Data
    input_size = model_config.get("input_size", 224)
    train_loader, _, test_loader = build_dataloaders(
        preproc_config, batch_size=32, input_size=input_size,
    )

    # Model
    model = build_model(model_config)
    load_checkpoint(model, args.checkpoint, device=device)

    # MC Dropout for entropy-based sample selection
    print("[XAI] Running MC Dropout for sample selection...")
    mean_probs, entropy = mc_dropout_inference(model, test_loader, n_passes=args.mc_passes, device=device)

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

    # Select samples
    n_per_group = args.n_samples // 4
    selected = select_xai_samples(entropy_np, all_labels, predictions, n_per_group)

    # Save metadata CSV
    pd.DataFrame(selected).to_csv(os.path.join(out_dir, "xai_metadata.csv"), index=False)
    print(f"[XAI] Selected {len(selected)} samples across 4 groups")

    # SHAP background
    background = None
    if "shap" in args.methods:
        background = get_background_samples(train_loader, n_samples=100).to(device)

    all_shap_values = []

    # Generate explanations
    for info in selected:
        sid = info["sample_id"]
        img_tensor = all_images[sid].unsqueeze(0)
        img_np = all_images[sid].permute(1, 2, 0).numpy()

        print(f"  Sample {sid} ({info['split_group']})")

        if "gradcam" in args.methods:
            heatmap = generate_gradcam(model, img_tensor, device=device)
            save_gradcam_visualization(
                img_np, heatmap,
                os.path.join(out_dir, "gradcam"), sid,
            )

        if "lrp" in args.methods:
            relevance = generate_lrp(model, img_tensor, device=device)
            save_lrp_visualization(relevance, os.path.join(out_dir, "lrp"), sid)

        if "shap" in args.methods and background is not None:
            sv = generate_shap_values(model, img_tensor, background, device=device)
            save_shap_visualization(sv, img_np, os.path.join(out_dir, "shap"), sid)
            all_shap_values.append(sv)

    # Mean SHAP summary
    if all_shap_values:
        save_mean_shap_summary(
            all_shap_values,
            os.path.join(out_dir, "shap", "mean_shap_summary.png"),
        )

    print(f"\n[XAI] Results saved to {out_dir}")


if __name__ == "__main__":
    main()
