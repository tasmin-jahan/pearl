"""
Uncertainty quantification CLI: MC Dropout predictive entropy + referral
curve.

Requires a preprocessed PNG dataset and a trained checkpoint::

    python -m src.uncertainty.run_uncertainty \
        --dataset-dir data/preprocessed/figshare \
        --model swin_tiny \
        --checkpoint results/figshare/swin_tiny/best.pt \
        --out-dir results/uncertainty/figshare/swin_tiny \
        --mc-passes 50

In ensemble mode, pass ``--model``/``--checkpoint`` repeatedly to compute
the average entropy across members::

    python -m src.uncertainty.run_uncertainty \
        --dataset-dir data/preprocessed/figshare \
        --model swin_tiny     --checkpoint .../swin_tiny/best.pt \
        --model densenet169   --checkpoint .../densenet169/best.pt \
        --ensemble \
        --out-dir results/uncertainty/figshare/ensemble

Single-model uncertainty lives here; ensemble uncertainty is delegated
to ``src.ensemble.run_uncertainty_ensemble`` so all ensemble code is in
one place.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.dataloader import build_dataloaders
from src.model.builder import build_model
from src.training.checkpoint import load_checkpoint
from src.uncertainty.mc_dropout import mc_dropout_inference
from src.uncertainty.referral import compute_referral_curve
from src.utils.config import DEFAULT_PREPROCESSING_CONFIG, load_config
from src.utils.seed import set_seed


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


def _predict_single(
    model: torch.nn.Module,
    loader,
    mc_passes: int,
    device: str,
):
    """Run MC dropout on a single model."""
    mean_probs, entropy = mc_dropout_inference(
        model, loader, n_passes=mc_passes, device=device
    )
    labels = []
    for _, lab in loader:
        if isinstance(lab, torch.Tensor):
            labels.extend(lab.cpu().numpy())
        else:
            labels.extend(lab)
    return mean_probs.numpy(), entropy.numpy(), np.array(labels)


def _summarize(
    mean_probs: np.ndarray,
    entropy: np.ndarray,
    labels: np.ndarray,
    arch_names: List[str],
    mc_passes: int,
    coverage_thresholds: List[float],
    out_dir: str,
) -> Dict[str, float]:
    os.makedirs(out_dir, exist_ok=True)
    predictions = mean_probs.argmax(axis=1)
    correct = predictions == labels

    per_sample = pd.DataFrame({
        "sample_id": range(len(labels)),
        "true_label": labels,
        "mean_prob_pcos": np.round(mean_probs[:, 1], 4),
        "mean_prob_non_pcos": np.round(mean_probs[:, 0], 4),
        "predicted_label": predictions,
        "correct": correct,
        "entropy_nats": np.round(entropy, 4),
    })
    per_sample.to_csv(
        os.path.join(out_dir, "entropy_per_sample.csv"), index=False
    )

    referral = compute_referral_curve(
        entropy, labels, predictions, coverage_thresholds
    )
    pd.DataFrame(referral).to_csv(
        os.path.join(out_dir, "referral_curve.csv"), index=False
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(entropy[correct], bins=50, alpha=0.7, label="Correct", color="tab:green")
    ax.hist(entropy[~correct], bins=50, alpha=0.7, label="Incorrect", color="tab:red")
    ax.set_xlabel("Predictive Entropy (nats)")
    ax.set_ylabel("Count")
    ax.set_title("Entropy Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(
        os.path.join(out_dir, "entropy_histogram.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)

    covs = [r["coverage_threshold"] for r in referral]
    accs = [r["accuracy_on_auto"] for r in referral]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(covs, accs, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Accuracy on Auto-decided")
    ax.set_title("Coverage vs Accuracy")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.55, 1.05)
    fig.savefig(
        os.path.join(out_dir, "coverage_accuracy_curve.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)

    summary = {
        "archs": arch_names,
        "ensemble": len(arch_names) > 1,
        "mc_passes": mc_passes,
        "n_test": int(len(labels)),
        "mean_entropy": round(float(entropy.mean()), 4),
        "median_entropy": round(float(np.median(entropy)), 4),
        "accuracy_full": round(float(correct.mean()), 4),
    }
    with open(os.path.join(out_dir, "uncertainty_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Uncertainty] Saved to {out_dir}")
    return summary


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MC Dropout uncertainty quantification on PNG dataset."
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--preprocessing", default=DEFAULT_PREPROCESSING_CONFIG)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mc-passes", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--coverage-thresholds", type=float, nargs="+",
        default=[1.0, 0.9, 0.8, 0.7, 0.6],
    )
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument(
        "--ensemble", action="store_true",
        help="Run MC-dropout over an ensemble (delegates to src.ensemble).",
    )
    args = parser.parse_args(argv)

    if len(args.model) != len(args.checkpoint) or not args.model:
        parser.error("Provide equal numbers of --model and --checkpoint flags")
    if not args.ensemble and len(args.model) != 1:
        args.ensemble = True

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    preproc_config = load_config(args.preprocessing)

    _, _, test_loader = build_dataloaders(
        args.dataset_dir,
        preproc_config,
        batch_size=args.batch_size,
        num_workers=2,
        sampler="none",
        drop_last=False,
    )

    if args.ensemble:
        from src.ensemble import run_uncertainty_ensemble
        run_uncertainty_ensemble(
            dataset_dir=args.dataset_dir,
            preproc_config=preproc_config,
            model_specs=list(zip(args.model, args.checkpoint)),
            test_loader=test_loader,
            mc_passes=args.mc_passes,
            coverage_thresholds=args.coverage_thresholds,
            out_dir=args.out_dir,
            device=device,
        )
    else:
        model = _load_model(args.model[0], args.checkpoint[0], device)
        mean_probs, entropy, labels = _predict_single(
            model, test_loader, args.mc_passes, device
        )
        _summarize(
            mean_probs=mean_probs,
            entropy=entropy,
            labels=labels,
            arch_names=[args.model[0]],
            mc_passes=args.mc_passes,
            coverage_thresholds=args.coverage_thresholds,
            out_dir=args.out_dir,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())