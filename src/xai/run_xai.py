"""
XAI analysis CLI: Grad-CAM (+ LRP, SHAP) on a preprocessed PNG dataset.

Sample selection uses MC Dropout entropy to surface four
``(confidence, correctness)`` groups; ``--n-samples`` is divided equally
across the four groups.

Usage::

    python -m src.xai.cli \
        --dataset-dir data/preprocessed/figshare \
        --model swin_tiny \
        --checkpoint results/figshare/swin_tiny/best.pt \
        --out-dir results/xai/figshare/swin_tiny \
        --methods gradcam lrp shap --n-samples 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from src.data.dataloader import build_dataloaders
from src.model.builder import build_model
from src.training.checkpoint import load_checkpoint
from src.uncertainty.mc_dropout import mc_dropout_inference
from src.xai.gradcam import generate_gradcam, save_gradcam_visualization
from src.utils.config import DEFAULT_PREPROCESSING_CONFIG, load_config
from src.utils.seed import set_seed


try:
    from src.xai.lrp import generate_lrp, save_lrp_visualization
except ImportError:
    generate_lrp = save_lrp_visualization = None

try:
    from src.xai.shap_explainer import (
        generate_shap_values,
        get_background_samples,
        save_mean_shap_summary,
        save_shap_visualization,
    )
except ImportError:
    generate_shap_values = get_background_samples = save_mean_shap_summary = save_shap_visualization = None


def _load_model(model_name: str, checkpoint: str, device: str) -> torch.nn.Module:
    cfg_path = (
        model_name
        if model_name.endswith((".yaml", ".yml"))
        else os.path.join("configs", "model", f"{model_name}.yaml")
    )
    model_config = load_config(cfg_path)
    model = build_model(model_config)
    load_checkpoint(model, checkpoint, device=device)
    model.to(device)
    model.eval()
    return model


def _select_samples(
    entropy: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    n_samples: int,
) -> List[dict]:
    """Pick samples from four ``(confidence, correctness)`` groups."""
    correct = predictions == labels
    median_entropy = float(np.median(entropy))
    groups = {
        "confident_correct": (entropy <= median_entropy) & correct,
        "overconfident_error": (entropy <= median_entropy) & ~correct,
        "uncertain_correct": (entropy > median_entropy) & correct,
        "uncertain_wrong": (entropy > median_entropy) & ~correct,
    }
    selected = []
    per_group = max(1, n_samples // 4)
    for name, mask in groups.items():
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        if name.startswith("confident"):
            chosen = indices[np.argsort(entropy[indices])[:per_group]]
        else:
            chosen = indices[np.argsort(-entropy[indices])[:per_group]]
        for idx in chosen:
            selected.append({
                "sample_id": int(idx),
                "true_label": int(labels[idx]),
                "predicted_label": int(predictions[idx]),
                "correct": bool(correct[idx]),
                "entropy_nats": round(float(entropy[idx]), 4),
                "split_group": name,
            })
    return selected


def _render_panels(
    model: torch.nn.Module,
    images: torch.Tensor,
    selected: Sequence[dict],
    methods: Sequence[str],
    out_dir: str,
    background: Optional[torch.Tensor] = None,
) -> None:
    device = next(model.parameters()).device
    all_shap = []
    for info in selected:
        sid = info["sample_id"]
        img_tensor = images[sid].unsqueeze(0).to(device)
        img_np = images[sid].permute(1, 2, 0).cpu().numpy()
        print(f"  Sample {sid} ({info['split_group']})")
        if "gradcam" in methods:
            heatmap = generate_gradcam(model, img_tensor, device=device)
            save_gradcam_visualization(
                img_np, heatmap, os.path.join(out_dir, "gradcam"), sid
            )
        if "lrp" in methods and generate_lrp is not None:
            relevance = generate_lrp(model, img_tensor, device=device)
            save_lrp_visualization(relevance, os.path.join(out_dir, "lrp"), sid)
        if (
            "shap" in methods
            and generate_shap_values is not None
            and background is not None
        ):
            sv = generate_shap_values(model, img_tensor, background, device=device)
            save_shap_visualization(sv, img_np, os.path.join(out_dir, "shap"), sid)
            all_shap.append(sv)
    if all_shap and save_mean_shap_summary is not None:
        save_mean_shap_summary(
            all_shap, os.path.join(out_dir, "shap", "mean_shap_summary.png")
        )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="XAI analysis on a preprocessed PNG dataset."
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--preprocessing", default=DEFAULT_PREPROCESSING_CONFIG)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--methods", nargs="+", default=["gradcam"],
        choices=("gradcam", "lrp", "shap"),
    )
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--mc-passes", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shap-background", type=int, default=100,
        help="Background samples for SHAP (skipped if --methods lacks shap).",
    )
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    preproc_config = load_config(args.preprocessing)

    train_loader, _, test_loader = build_dataloaders(
        args.dataset_dir,
        preproc_config,
        batch_size=args.batch_size,
        num_workers=2,
        sampler="none",
        drop_last=False,
    )

    model = _load_model(args.model, args.checkpoint, device)

    print(f"[XAI] Running {args.mc_passes} MC Dropout passes for sample selection...")
    mean_probs, entropy = mc_dropout_inference(
        model, test_loader, n_passes=args.mc_passes, device=device
    )

    labels = []
    images = []
    for x, y in test_loader:
        images.append(x)
        if isinstance(y, torch.Tensor):
            labels.extend(y.cpu().numpy())
        else:
            labels.extend(y)
    images = torch.cat(images, dim=0)
    labels = np.array(labels)
    predictions = mean_probs.numpy().argmax(axis=1)
    entropy_np = entropy.numpy()

    os.makedirs(args.out_dir, exist_ok=True)
    selected = _select_samples(entropy_np, labels, predictions, args.n_samples)
    pd.DataFrame(selected).to_csv(
        os.path.join(args.out_dir, "xai_metadata.csv"), index=False
    )
    print(f"[XAI] Selected {len(selected)} samples (target={args.n_samples})")

    background = None
    if "shap" in args.methods and get_background_samples is not None:
        background = get_background_samples(
            train_loader, n_samples=args.shap_background
        ).to(device)

    _render_panels(model, images, selected, args.methods, args.out_dir, background)
    with open(os.path.join(args.out_dir, "xai_summary.json"), "w") as f:
        json.dump(
            {
                "model": args.model,
                "dataset": args.dataset_dir,
                "checkpoint": args.checkpoint,
                "methods": list(args.methods),
                "n_samples": len(selected),
                "mc_passes": args.mc_passes,
            },
            f,
            indent=2,
        )
    print(f"[XAI] Saved to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())