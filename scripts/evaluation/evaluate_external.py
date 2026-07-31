#!/usr/bin/env python3
"""
External-validation evaluator for a trained PCOS model.

Loads a trained checkpoint and runs inference on a separate dataset
(e.g. PCOSGen) using the SAME preprocessing pipeline as training. The
external dataset is loaded on-the-fly from raw images — no preprocessing
cache is required, and no retraining is performed.

This is the script that produces the "external validation" row in the
paper's comparison table (Figure 09 in the thesis narrative).

Supported external datasets:
  - PCOSGen (Sundari et al. 2025 / divyangambhir1 Kaggle upload):
        <dataset>/PCOSGen-train/test/infected/*.jpg
        <dataset>/PCOSGen-train/test/healthy/*.jpg
        <dataset>/PCOSGen-train/train/infected/*.jpg
        <dataset>/PCOSGen-train/train/healthy/*.jpg
  - Generic fold-out layout:
        <dataset>/infected/*.jpg
        <dataset>/noninfected/*.jpg (or healthy/, normal/)

Preprocessing is now the unified config (configs/preprocessing.yaml).

Usage — inference on an external dataset:

    python scripts/evaluate_external.py \
        --run_dir results/ablation/checkpoints/srad/efficientnet_b0/ \
        --model configs/model/efficientnet_b0.yaml \
        --preprocessing configs/preprocessing.yaml \
        --checkpoint results/ablation/checkpoints/srad/efficientnet_b0/best.pt \
        --external_dir data_external/pcosgen \
        --external_layout pcosgen

    (With --run_dir, the output and predictions_csv default to
     <run_dir>/external_validation/pcosgen.json and .csv respectively.)

Usage — aggregate all per-arch outputs into the paper-ready summary:

    python scripts/evaluate_external.py --aggregate_summary \
        --checkpoints_root results/ablation \
        --summary_csv results/external_validation/summary.csv
"""

import argparse
import csv
import glob
import json
import os
import sys
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.model.builder import build_model
from src.training.checkpoint import load_checkpoint
from src.preprocessing.preprocess import Preprocessor
from src.evaluation.metrics import compute_all_metrics


# =====================================================================
# External dataset drivers
# =====================================================================

# Canonical class-label mapping aligned with PCOSDataset.CLASS_MAP
INFECTED_NAMES = {"infected", "pcos", "pcosgen_infected", "1", "positive"}
HEALTHY_NAMES = {"noninfected", "healthy", "normal", "pcosgen_healthy", "0", "negative"}


def _is_infected(name: str) -> bool:
    return name.lower() in INFECTED_NAMES


def _is_healthy(name: str) -> bool:
    return name.lower() in HEALTHY_NAMES


def _label_from_dirname(dirname: str) -> int:
    """Map class folder name to integer label. PCOSDataset convention:
    noninfected=0, infected=1.
    """
    if _is_infected(dirname):
        return 1
    if _is_healthy(dirname):
        return 0
    raise ValueError(
        f"Cannot infer label for folder '{dirname}'. "
        f"Expected one of {sorted(INFECTED_NAMES | HEALTHY_NAMES)}."
    )


def _list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    out = []
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            out.append(os.path.join(folder, fname))
    return out


def discover_pcosgen(dataset_root: str) -> List[Tuple[str, int]]:
    """Discover (path, label) pairs from the PCOSGen Kaggle upload layout.

    Path conventions handled:
      <root>/PCOSGen-train/train/infected/*.jpg
      <root>/PCOSGen-train/train/healthy/*.jpg
      <root>/PCOSGen-train/test/infected/*.jpg
      <root>/PCOSGen-train/test/healthy/*.jpg
    """
    pairs = []
    for split in ("train", "test"):
        for cls in ("infected", "healthy"):
            folder = os.path.join(dataset_root, "PCOSGen-train", split, cls)
            for path in _list_images(folder):
                pairs.append((path, _label_from_dirname(cls)))
    return pairs


def discover_simple(dataset_root: str) -> List[Tuple[str, int]]:
    """Discover (path, label) pairs from a flat infected/healthy layout.

    Looks for any pair of folders under ``dataset_root`` whose names map
    to the infected / healthy name sets.
    """
    pairs = []
    if not os.path.isdir(dataset_root):
        return pairs
    for entry in sorted(os.listdir(dataset_root)):
        full = os.path.join(dataset_root, entry)
        if not os.path.isdir(full):
            continue
        try:
            label = _label_from_dirname(entry)
        except ValueError:
            continue
        for path in _list_images(full):
            pairs.append((path, label))
    return pairs


def discover_with_split(dataset_root: str, split: str) -> List[Tuple[str, int]]:
    """For simple layouts with a train/test/val subdirectory."""
    pairs = []
    split_dir = os.path.join(dataset_root, split)
    if not os.path.isdir(split_dir):
        return pairs
    for entry in sorted(os.listdir(split_dir)):
        full = os.path.join(split_dir, entry)
        if not os.path.isdir(full):
            continue
        try:
            label = _label_from_dirname(entry)
        except ValueError:
            continue
        for path in _list_images(full):
            pairs.append((path, label))
    return pairs


def discover_zenodo_labeled(dataset_root: str, split: str = "train") -> List[Tuple[str, int]]:
    """Discover (path, label) pairs from the Zenodo PCOSgen source layout.

    Layout:
        <dataset_root>/images/*.jpg
        <dataset_root>/class_label.xlsx  (train) or 'class label.csv' (test)

    The label is read from the LAST column (the "polycystic ovary visible"
    column); "Visible" -> 1, anything else -> 0. Pairs whose image filename
    is missing from the label table are skipped.

    The `split` argument is ignored — the layout is self-describing. It is
    accepted for parity with `discover_simple`.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from src.data.zenodo_dataset import discover_zenodo_pairs
    return discover_zenodo_pairs(dataset_root)


def discover(dataset_root: str, layout: str, split: str = "test") -> List[Tuple[str, int]]:
    """Top-level discovery dispatcher."""
    if layout == "pcosgen":
        # Use ONLY the test split for external validation.
        return discover_pcosgen(dataset_root)
    elif layout == "simple":
        return discover_with_split(dataset_root, split)
    elif layout == "flat":
        return discover_simple(dataset_root)
    elif layout == "zenodo_labeled":
        return discover_zenodo_labeled(dataset_root, split)
    else:
        raise ValueError(f"Unknown layout: {layout}")


# =====================================================================
# In-memory dataset
# =====================================================================

class _ExternalDataset(Dataset):
    """Loads raw images and runs the training preprocessing pipeline
    on-the-fly. Augmentation is FORCED OFF (this is evaluation)."""

    def __init__(self, pairs: List[Tuple[str, int]], preprocessor: Preprocessor):
        self.pairs = pairs
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path, label = self.pairs[idx]
        img = cv2.imread(path)
        if img is None:
            # Fail-safe: black image; should not happen on a healthy dataset.
            img = np.zeros(
                (self.preprocessor.input_size, self.preprocessor.input_size, 3),
                dtype=np.uint8,
            )
        x = self.preprocessor.apply(img, augment=False)
        if x.ndim == 2:
            x = x[None, :, :]
        else:
            x = np.transpose(x, (2, 0, 1))
        return torch.from_numpy(x.copy()).float(), label, path


# =====================================================================
# Main
# =====================================================================

def _aggregate_external_validation(checkpoints_root: str, output_path: str):
    """Walk ``results/<root>/checkpoints/<prep>/<arch>/external_validation/``,
    JSONs and write a single CSV summary at ``output_path``.

    Replaces the manual aggregation block in todo.md step 7.
    """
    pattern = os.path.join(
        checkpoints_root, "checkpoints", "*", "*", "external_validation", "*.json",
    )
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[Aggregate] No external_validation JSONs found at {pattern}")
        return

    rows = []
    for fp in files:
        # fp = .../checkpoints/<prep>/<arch>/external_validation/<dataset>.json
        rel = fp[len(checkpoints_root):].strip("/")
        parts = rel.split("/")
        prep = parts[1]
        arch = parts[2]
        dataset = os.path.splitext(os.path.basename(fp))[0]
        with open(fp) as f:
            d = json.load(f)
        rows.append({
            "prep": prep,
            "arch": arch,
            "dataset": dataset,
            "n_samples": d.get("n_samples"),
            # The per-arch JSONs use the ``test_*`` prefix (mirroring
            # ``compute_all_metrics`` output). Read both naming
            # conventions so older JSONs (and any external scripts)
            # still aggregate cleanly.
            "auc":        d.get("test_auc_roc", d.get("auc_roc", d.get("auc"))),
            "accuracy":   d.get("test_accuracy", d.get("accuracy")),
            "f1":         d.get("test_f1", d.get("f1")),
            "mcc":        d.get("test_mcc", d.get("mcc")),
            "precision":  d.get("test_precision", d.get("precision")),
            "recall":     d.get("test_recall", d.get("recall")),
            "specificity": d.get("test_specificity", d.get("specificity")),
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "prep", "arch", "dataset", "n_samples",
        "auc", "accuracy", "f1", "mcc",
        "precision", "recall", "specificity",
    ]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[Aggregate] Wrote {len(rows)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="External-validation evaluator",
        epilog=(
            "Two modes:\n"
            "  1. Standard: run inference on an external dataset and write\n"
            "     metrics + per-image preds. Use --checkpoint to load a model.\n"
            "     If --run_dir is set, --output / --predictions_csv default to\n"
            "     <run_dir>/external_validation/<dataset_slug>.{json,csv}.\n"
            "  2. --aggregate_summary: glob results/<root>/checkpoints/<prep>/"
            "<arch>/external_validation/*.json and write a single summary CSV."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model architecture YAML config.")
    parser.add_argument("--preprocessing", type=str, default=None,
                        help="Path to preprocessing YAML config.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained model checkpoint.")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Per-arch run directory (e.g. "
                             "results/ablation/checkpoints/srad/efficientnet_b0/). "
                             "If set, --output and --predictions_csv default to "
                             "<run_dir>/external_validation/<dataset_slug>.{json,csv}.")
    parser.add_argument("--dataset_slug", type=str, default="pcosgen",
                        help="Filename slug for the dataset in external_validation/ "
                             "(default 'pcosgen'). Used with --run_dir auto-defaulting.")
    parser.add_argument("--external_dir", type=str, default=None,
                        help="Root directory of the external dataset.")
    parser.add_argument("--layout", type=str, default="pcosgen",
                        choices=["pcosgen", "simple", "flat", "zenodo_labeled"],
                        help="External dataset layout.")
    parser.add_argument("--split", type=str, default="test",
                        help="For 'simple' layout: which split to use.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional cap on the number of samples (debug).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="Path to JSON output (metrics). Defaults to "
                             "<run_dir>/external_validation/<dataset_slug>.json "
                             "when --run_dir is set.")
    parser.add_argument("--predictions_csv", type=str, default=None,
                        help="Path to write per-image predictions CSV. Defaults "
                             "to <run_dir>/external_validation/<dataset_slug>.csv "
                             "when --run_dir is set.")
    parser.add_argument("--aggregate_summary", action="store_true",
                        help="Skip inference; rebuild the top-level aggregated "
                             "summary CSV from existing per-arch JSONs.")
    parser.add_argument("--checkpoints_root", type=str, default="results/",
                        help="Root results dir for --aggregate_summary "
                             "(default 'results/').")
    parser.add_argument("--summary_csv", type=str,
                        default="results/external_validation/summary.csv",
                        help="Output path for --aggregate_summary.")
    args = parser.parse_args()

    # ---- Aggregate mode (separate execution path) ----
    if args.aggregate_summary:
        _aggregate_external_validation(args.checkpoints_root, args.summary_csv)
        return

    # ---- Standard mode requires these ----
    if not (args.model and args.preprocessing and args.checkpoint
            and args.external_dir):
        parser.error(
            "Standard mode requires --model, --preprocessing, --checkpoint, "
            "--external_dir (or use --aggregate_summary)."
        )

    # ---- Apply --run_dir auto-defaults for output paths ----
    if args.run_dir:
        ext_dir = os.path.join(args.run_dir, "external_validation")
        os.makedirs(ext_dir, exist_ok=True)
        if args.output is None:
            args.output = os.path.join(ext_dir, f"{args.dataset_slug}.json")
        if args.predictions_csv is None:
            args.predictions_csv = os.path.join(ext_dir, f"{args.dataset_slug}.csv")

    set_seed(args.seed)

    model_config = load_config(args.model)
    preproc_config = load_config(args.preprocessing)
    input_size = int(model_config.get("input_size", 224))

    print(f"[ExternalEval] Model: {model_config.get('name', args.model)}")
    print(f"[ExternalEval] Preprocessing: {preproc_config.get('name', args.preprocessing)}")
    print(f"[ExternalEval] Input size: {input_size}")
    print(f"[ExternalEval] External dataset: {args.external_dir} (layout={args.layout})")

    # Discover samples
    pairs = discover(args.external_dir, args.layout, args.split)
    if args.max_samples is not None:
        pairs = pairs[: args.max_samples]
    if not pairs:
        raise RuntimeError(
            f"No images found under {args.external_dir} with layout={args.layout}. "
            f"Check the path and folder naming."
        )
    n_infected = sum(1 for _, l in pairs if l == 1)
    n_healthy = sum(1 for _, l in pairs if l == 0)
    print(f"[ExternalEval] Found {len(pairs)} images: {n_infected} infected, {n_healthy} healthy")

    # Build dataset + loader
    preprocessor = Preprocessor(preproc_config, input_size=input_size)
    dataset = _ExternalDataset(pairs, preprocessor)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=False,
    )

    # Build model + load checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(model_config)
    load_checkpoint(model, args.checkpoint, device=device)
    model.eval()
    model.to(device)

    # Inference
    all_paths, all_labels, all_probs, all_preds = [], [], [], []
    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_paths.extend(list(paths))
            all_labels.extend([int(l) for l in labels])
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    # Metrics
    metrics = compute_all_metrics(all_labels, all_preds, all_probs)
    metrics["n_samples"] = len(all_labels)
    metrics["n_infected"] = int(n_infected)
    metrics["n_healthy"] = int(n_healthy)
    metrics["model"] = model_config.get("name", args.model)
    metrics["preprocessing"] = preproc_config.get("name", args.preprocessing)
    metrics["checkpoint"] = args.checkpoint
    metrics["external_dir"] = args.external_dir
    metrics["external_layout"] = args.layout

    print("\n[ExternalEval] Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Save JSON
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[ExternalEval] Saved metrics to {args.output}")

    # Save per-image CSV
    if args.predictions_csv:
        os.makedirs(os.path.dirname(args.predictions_csv), exist_ok=True)
        with open(args.predictions_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "pred", "prob_infected"])
            for p, l, pr, pb in zip(all_paths, all_labels, all_preds, all_probs):
                w.writerow([p, int(l), int(pr), f"{pb:.6f}"])
        print(f"[ExternalEval] Saved per-image predictions to {args.predictions_csv}")


if __name__ == "__main__":
    main()
