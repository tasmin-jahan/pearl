#!/usr/bin/env python3
"""XAI (Grad-CAM) analysis on the Zenodo held-out test set.

Usage:
    python scripts/run_xai_zenodo.py \
        --run_dir results/finetune_zenodo/checkpoints/srad_nopad/densenet121 \
        --n_samples 20
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.zenodo_dataset import discover_zenodo_pairs, ZenodoDataset
from src.model.builder import build_model
from src.training.checkpoint import load_checkpoint
from src.preprocessing.preprocess import Preprocessor
from src.uncertainty.mc_dropout import mc_dropout_inference
from src.xai.gradcam import generate_gradcam, save_gradcam_visualization


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
            })
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--external_dir", default="data_external/test")
    parser.add_argument("--methods", nargs="+", default=["gradcam"])
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--n_passes", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    pairs = discover_zenodo_pairs(args.external_dir)
    pre = Preprocessor(preproc_config, input_size=model_config.get("input_size", 224))
    ds = ZenodoDataset([p for p, _ in pairs], [l for _, l in pairs], pre, augment=False)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=False,
    )

    out_dir = os.path.join(args.run_dir, "xai")
    os.makedirs(out_dir, exist_ok=True)

    # MC Dropout for entropy-based sample selection
    print(f"[XAI] Running {args.n_passes} MC Dropout passes for sample selection...")
    mean_probs, entropy = mc_dropout_inference(
        model, loader, n_passes=args.n_passes, device=device,
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

    # Generate Grad-CAM panels
    for info in selected:
        sid = info["sample_id"]
        img_tensor = all_images[sid].unsqueeze(0)
        img_np = all_images[sid].permute(1, 2, 0).numpy()
        if "gradcam" in args.methods:
            heatmap = generate_gradcam(model, img_tensor, device=device)
            save_gradcam_visualization(
                img_np, heatmap,
                os.path.join(out_dir, "gradcam"), sid,
            )

    print(f"[XAI] Saved to {out_dir}")


if __name__ == "__main__":
    main()