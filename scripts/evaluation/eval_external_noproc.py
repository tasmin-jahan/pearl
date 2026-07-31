#!/usr/bin/env python3
"""External-eval variant: **skip** the srad/gauss denoising entirely.

Hypothesis: the figshare-trained models collapse on PCOSgen because the
ImageNet-style normalization + srad/gauss pipeline is tuned to figshare's
intensity distribution. If the model is fed PCOSgen with only resize +
standard ImageNet normalization (no denoising), the recovered metrics
would indicate whether the preprocessor is the bottleneck or the
features themselves are the problem.

Writes outputs to ``<run_dir>/external_validation_noproc/pcosgen.json``
and ``.csv`` so it never overwrites the existing
``external_validation/pcosgen.{json,csv}`` artifacts.
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
from src.evaluation.metrics import compute_all_metrics


# ImageNet mean/std (matches what timm's default pretrained models expect)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def noproc_apply(img: np.ndarray, input_size: int) -> np.ndarray:
    """Apply only resize (aspect-preserving letterbox) + ImageNet normalize.

    No CLAHE, no SRAD. This is the simplest
    preprocessing a timm pretrained model would expect.
    """
    h, w = img.shape[:2]
    scale = input_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Letterbox to input_size x input_size
    pad_h = input_size - new_h
    pad_w = input_size - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )
    # BGR -> RGB
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # ImageNet normalize
    normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return normalized.astype(np.float32)


class _NoProcDataset(Dataset):
    def __init__(self, pairs, input_size):
        self.pairs = pairs
        self.input_size = input_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path, label = self.pairs[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        x = noproc_apply(img, self.input_size)
        x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
        return torch.from_numpy(x.copy()).float(), label, path


# Reuse the discovery function from evaluate_external
from scripts.evaluation.evaluate_external import (  # noqa: E402
    discover, INFECTED_NAMES, HEALTHY_NAMES, _label_from_dirname,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--external_dir", required=True)
    ap.add_argument("--layout", default="pcosgen")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset_slug", default="pcosgen_noproc")
    ap.add_argument(
        "--out_subdir", default="external_validation_noproc",
        help="Subdirectory under run_dir for outputs (default: "
             "external_validation_noproc). Does NOT touch the existing "
             "external_validation/ folder.",
    )
    args = ap.parse_args()

    set_seed(args.seed)

    model_config = load_config(args.model)
    input_size = int(model_config.get("input_size", 224))
    pairs = discover(args.external_dir, args.layout, args.split)
    if args.max_samples is not None:
        pairs = pairs[: args.max_samples]
    if not pairs:
        raise RuntimeError(f"No images at {args.external_dir} (layout={args.layout})")
    n_inf = sum(1 for _, l in pairs if l == 1)
    n_h = sum(1 for _, l in pairs if l == 0)

    print(f"[NoProcEval] Model: {model_config.get('name', args.model)}")
    print(f"[NoProcEval] Input size: {input_size}  (resize+ImageNet normalize, NO denoising)")
    print(f"[NoProcEval] External dataset: {args.external_dir} (layout={args.layout})")
    print(f"[NoProcEval] Found {len(pairs)} images: {n_inf} infected, {n_h} healthy")

    dataset = _NoProcDataset(pairs, input_size=input_size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(model_config)
    load_checkpoint(model, args.checkpoint, device=device)
    model.eval()
    model.to(device)

    paths, labels, probs, preds = [], [], [], []
    with torch.no_grad():
        for images, lbl, p in loader:
            images = images.to(device)
            logits = model(images)
            p_inf = F.softmax(logits, dim=1)[:, 1]
            pr = logits.argmax(dim=1)
            paths.extend(list(p))
            labels.extend([int(x) for x in lbl])
            probs.extend(p_inf.cpu().numpy().tolist())
            preds.extend(pr.cpu().numpy().tolist())

    labels = np.array(labels)
    probs = np.array(probs)
    preds = np.array(preds)
    metrics = compute_all_metrics(labels, preds, probs)
    metrics.update({
        "n_samples": int(len(labels)),
        "n_infected": int(n_inf),
        "n_healthy": int(n_h),
        "model": model_config.get("name", args.model),
        "preprocessing": "none_resize_imagenet",
        "checkpoint": args.checkpoint,
        "external_dir": args.external_dir,
        "external_layout": args.layout,
    })

    print()
    print("[NoProcEval] Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Write outputs to a SEPARATE subdirectory under run_dir
    out_dir = os.path.join(args.run_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, f"{args.dataset_slug}.json")
    out_csv = os.path.join(out_dir, f"{args.dataset_slug}.csv")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[NoProcEval] Saved metrics to {out_json}")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "pred", "prob_infected"])
        for p, l, pr, pb in zip(paths, labels, preds, probs):
            w.writerow([p, int(l), int(pr), f"{pb:.6f}"])
    print(f"[NoProcEval] Saved per-image predictions to {out_csv}")


if __name__ == "__main__":
    main()
