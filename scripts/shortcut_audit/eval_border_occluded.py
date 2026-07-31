#!/usr/bin/env python3
"""Border-occlusion inference ablation.

Hypothesis: if the model relies on letterbox-padding artefacts to classify,
masking or replacing that border at *inference time* should degrade
performance.

Two occlusion modes are supported (selected with ``--mode``):

  median     — Replace every pixel in the outer ``--border_frac`` strip with
               the per-image median intensity of the inner (1 - 2*frac)
               region. Tests whether the *content* of the border matters.

  swap       — Swap the border strips between pairs of test images of
               opposite labels (PCOS ↔ non-PCOS). If the model's
               discrimination is driven by the actual border content
               rather than the global image, swapping the border should
               flip the model's decision more often than chance.

Outputs:
  <run_dir>/border_occlusion_<mode>/pcosgen_occluded.{json,csv}

This directory sits beside the existing ``external_validation/`` and
``shortcut_audit/`` directories so nothing is overwritten.

Usage:
    python scripts/eval_border_occluded.py \\
        --run_dir results/ablation/checkpoints/srad/densenet121 \\
        --model configs/model/densenet121.yaml \\
        --preprocessing configs/preprocessing/srad.yaml \\
        --checkpoint results/ablation/checkpoints/srad/densenet121/best.pt \\
        --mode median
"""

import argparse
import csv
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


# ImageNet mean/std (matches timm pretrained models)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ------------------------------------------------------------------
# Border mask + occlusion transforms
# ------------------------------------------------------------------

def make_border_strip(h: int, w: int, frac: float):
    """Return a (h, w) boolean mask that's True on the outer ``frac`` strip.

    The strip covers rows [:b] ∪ [-b:] and cols [:b] ∪ [-b:] (UNION, not
    symmetric-difference) — same convention used in the Grad-CAM audit.
    """
    b = max(1, int(round(min(h, w) * frac)))
    m = np.zeros((h, w), dtype=bool)
    m[:b, :] = True
    m[-b:, :] = True
    m[:, :b] = True
    m[:, -b:] = True
    return m


def median_border(x: np.ndarray, frac: float) -> np.ndarray:
    """Replace the outer ``frac`` strip with the median value of the inner
    region. Operates per-channel.

    Args:
        x: Image as CHW float32 array (post-zscore range, i.e. ~[-3, 3]).
        frac: Fraction of image to mask (e.g. 0.05 = outer 5%).

    Returns:
        New CHW float32 array with border replaced.
    """
    out = x.copy()
    h, w = x.shape[1], x.shape[2]
    mask = make_border_strip(h, w, frac)
    for c in range(x.shape[0]):
        ch = x[c]
        med = float(np.median(ch[~mask]))
        out[c] = np.where(mask, med, ch)
    return out


# ------------------------------------------------------------------
# Inference-only dataset: load preprocessed .npy, apply border transform
# ------------------------------------------------------------------

class _OcclusionDataset(Dataset):
    """Loads preprocessed .npy arrays and applies the per-image border transform.

    To support 'swap' mode we need cross-image knowledge at batch time, so
    'swap' is implemented separately in run_swap_evaluation() — this Dataset
    only handles 'median' mode.
    """

    def __init__(self, samples: List[Tuple[str, int]], mode: str = "median",
                 border_frac: float = 0.05):
        self.samples = samples
        self.mode = mode
        self.border_frac = border_frac

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = np.load(path)  # HWC float32
        if x.ndim == 3:
            chw = np.transpose(x, (2, 0, 1))
        else:
            chw = x[None, :, :]
        if self.mode == "median":
            chw = median_border(chw.astype(np.float32), self.border_frac)
        elif self.mode == "passthrough":
            pass
        else:
            raise ValueError(f"Unsupported mode in _OcclusionDataset: {self.mode}")
        return (
            torch.from_numpy(chw.copy()).float(),
            int(label),
            path,
        )


def _discover_pcos_test(preproc_config: dict) -> List[Tuple[str, int]]:
    """Discover Figshare test-set .npy paths from the preprocessor output dir."""
    out = preproc_config.get("output_dir", "results/preprocessed/default")
    test_root = os.path.join(out, "test")
    samples = []
    for cls_name, label in (("noninfected", 0), ("infected", 1)):
        cls_dir = os.path.join(test_root, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fn in sorted(os.listdir(cls_dir)):
            if fn.endswith(".npy"):
                samples.append((os.path.join(cls_dir, fn), label))
    return samples


# ------------------------------------------------------------------
# Main: median mode
# ------------------------------------------------------------------

def run_median(run_dir, model_cfg, preproc_cfg, ckpt, border_frac, args):
    samples = _discover_pcos_test(preproc_cfg)
    if not samples:
        raise RuntimeError(f"No test .npy files under {preproc_cfg.get('output_dir')}")
    n0 = sum(1 for _, l in samples if l == 0)
    n1 = sum(1 for _, l in samples if l == 1)
    print(f"[MedianOcclusion] {len(samples)} test images ({n1} infected, {n0} healthy), "
          f"border_frac={border_frac}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(model_cfg)
    load_checkpoint(model, ckpt, device=device)
    model.eval()
    model.to(device)

    ds = _OcclusionDataset(samples, mode="median", border_frac=border_frac)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=False)

    paths, labels, probs, preds = [], [], [], []
    with torch.no_grad():
        for x, y, p in loader:
            x = x.to(device)
            logits = model(x)
            pr_inf = F.softmax(logits, dim=1)[:, 1]
            pr = logits.argmax(dim=1)
            paths.extend(list(p))
            labels.extend([int(v) for v in y])
            probs.extend(pr_inf.cpu().numpy().tolist())
            preds.extend(pr.cpu().numpy().tolist())

    labels = np.array(labels); probs = np.array(probs); preds = np.array(preds)
    metrics = compute_all_metrics(labels, preds, probs)
    metrics.update({
        "mode": "median_border_replaced",
        "border_frac": border_frac,
        "n_samples": int(len(labels)),
        "n_infected": int(n1),
        "n_healthy": int(n0),
        "model": model_cfg.get("name", run_dir),
        "preprocessing": preproc_cfg.get("name", run_dir),
        "checkpoint": ckpt,
    })

    out_dir = os.path.join(run_dir, args.out_subdir or "border_occlusion_median")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "border_occluded.json")
    out_csv = os.path.join(out_dir, "border_occluded.csv")
    with open(out_json, "w") as f: json.dump(metrics, f, indent=2)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "pred", "prob_infected"])
        for pth, l, pr, pb in zip(paths, labels, preds, probs):
            w.writerow([pth, int(l), int(pr), f"{pb:.6f}"])
    print(f"[MedianOcclusion] wrote {out_json}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    return metrics


# ------------------------------------------------------------------
# Main: swap mode
# ------------------------------------------------------------------

def run_swap(run_dir, model_cfg, preproc_cfg, ckpt, border_frac, args):
    """Swap border strips between pairs of opposite-class test images.

    For every (PCOS, non-PCOS) pair we construct TWO ablated images:
        A_occluded = border-from-B-on-image-A
        B_occluded = border-from-A-on-image-B

    We then ask the model to classify both. If border content drives the
    decision, swapping should push the model toward the *donor's* class.
    """
    samples = _discover_pcos_test(preproc_cfg)
    inf = [s for s in samples if s[1] == 1]
    nor = [s for s in samples if s[1] == 0]
    n = min(len(inf), len(nor))
    print(f"[SwapOcclusion] {len(inf)} PCOS / {len(nor)} non-PCOS -> {n} pairs, "
          f"border_frac={border_frac}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(model_cfg)
    load_checkpoint(model, ckpt, device=device)
    model.eval()
    model.to(device)

    batch_size = args.batch_size
    rows = []

    def to_chw(arr):
        return np.transpose(arr, (2, 0, 1)) if arr.ndim == 3 else arr[None]

    def predict_batch(batch_chw):
        x = torch.from_numpy(np.stack(batch_chw).astype(np.float32)).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
        return probs, preds

    def strip_only(chw, mask):
        out = np.zeros_like(chw)
        for c in range(chw.shape[0]):
            out[c] = np.where(mask, chw[c], 0.0)
        return out

    def place_strip(target_chw, donor_strip, mask):
        out = target_chw.copy()
        for c in range(target_chw.shape[0]):
            out[c] = np.where(mask, donor_strip[c], target_chw[c])
        return out

    # Process in chunks of pairs
    inf_chw = [to_chw(np.load(s[0])) for s in inf[:n]]
    nor_chw = [to_chw(np.load(s[0])) for s in nor[:n]]
    h, w = inf_chw[0].shape[1], inf_chw[0].shape[2]
    mask = make_border_strip(h, w, border_frac)

    # For each pair, we evaluate 4 images:
    #   INF_clean   (control: original PCOS)
    #   NOR_clean   (control: original non-PCOS)
    #   INF_w_NOR_border  (PCOS image with non-PCOS border)
    #   NOR_w_INF_border  (non-PCOS image with PCOS border)
    # Predictions on the last two tell us if border content is informative.

    # Pre-build all donor strips
    inf_strips = [strip_only(x, mask) for x in inf_chw]
    nor_strips = [strip_only(x, mask) for x in nor_chw]

    print("[SwapOcclusion] Building pairs and evaluating in batches...")
    probs_inf_clean, preds_inf_clean = [], []
    probs_nor_clean, preds_nor_clean = [], []
    probs_inf_swap,  preds_inf_swap  = [], []
    probs_nor_swap,  preds_nor_swap  = [], []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b_inf = inf_chw[start:end]
        b_nor = nor_chw[start:end]
        b_inf_strip = inf_strips[start:end]
        b_nor_strip = nor_strips[start:end]
        # Build 4-image batches
        all_clean = b_inf + b_nor
        p, pr = predict_batch(all_clean)
        probs_inf_clean.extend(p[:len(b_inf)].tolist())
        preds_inf_clean.extend(pr[:len(b_inf)].tolist())
        probs_nor_clean.extend(p[len(b_inf):].tolist())
        preds_nor_clean.extend(pr[len(b_inf):].tolist())

        # Swapped versions
        inf_swapped = [place_strip(b_inf[i], b_nor_strip[i], mask) for i in range(len(b_inf))]
        nor_swapped = [place_strip(b_nor[i], b_inf_strip[i], mask) for i in range(len(b_nor))]
        all_swap = inf_swapped + nor_swapped
        p, pr = predict_batch(all_swap)
        probs_inf_swap.extend(p[:len(b_inf)].tolist())
        preds_inf_swap.extend(pr[:len(b_inf)].tolist())
        probs_nor_swap.extend(p[len(b_inf):].tolist())
        preds_nor_swap.extend(pr[len(b_inf):].tolist())

    arr = lambda x: np.array(x)
    summary = {
        "mode": "border_swap_opposite_class",
        "border_frac": border_frac,
        "n_pairs": n,
        # Clean predictions
        "clean_inf_p_infected_mean": float(arr(probs_inf_clean).mean()),
        "clean_nor_p_infected_mean": float(arr(probs_nor_clean).mean()),
        # Swapped
        "swapped_inf_p_infected_mean": float(arr(probs_inf_swap).mean()),
        "swapped_nor_p_infected_mean": float(arr(probs_nor_swap).mean()),
        # Δ (probability shift from clean → swapped)
        "delta_inf": float(arr(probs_inf_swap).mean() - arr(probs_inf_clean).mean()),
        "delta_nor": float(arr(probs_nor_swap).mean() - arr(probs_nor_clean).mean()),
        # If border drives decision:
        #   Δ_inf should be NEGATIVE  (PCOS-image-with-non-PCOS-border looks less PCOS)
        #   Δ_nor should be POSITIVE  (non-PCOS-with-PCOS-border looks more PCOS)
        "model": model_cfg.get("name", run_dir),
        "preprocessing": preproc_cfg.get("name", run_dir),
        "checkpoint": ckpt,
    }
    out_dir = os.path.join(run_dir, args.out_subdir or "border_occlusion_swap")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "border_swap.json")
    out_csv = os.path.join(out_dir, "border_swap.csv")
    with open(out_json, "w") as f: json.dump(summary, f, indent=2)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "inf_path", "nor_path",
            "inf_clean_p", "nor_clean_p",
            "inf_swap_p", "nor_swap_p",
            "inf_p_change", "nor_p_change",
        ])
        for i in range(n):
            w.writerow([
                inf[i][0], nor[i][0],
                f"{probs_inf_clean[i]:.6f}", f"{probs_nor_clean[i]:.6f}",
                f"{probs_inf_swap[i]:.6f}", f"{probs_nor_swap[i]:.6f}",
                f"{probs_inf_swap[i] - probs_inf_clean[i]:+.6f}",
                f"{probs_nor_swap[i] - probs_nor_clean[i]:+.6f}",
            ])
    print(f"[SwapOcclusion] wrote {out_json}")
    print(f"  clean INF mean p(PCOS)   = {summary['clean_inf_p_infected_mean']:.4f}")
    print(f"  clean NOR mean p(PCOS)   = {summary['clean_nor_p_infected_mean']:.4f}")
    print(f"  swapped INF mean p(PCOS) = {summary['swapped_inf_p_infected_mean']:.4f}  "
          f"(Δ={summary['delta_inf']:+.4f})")
    print(f"  swapped NOR mean p(PCOS) = {summary['swapped_nor_p_infected_mean']:.4f}  "
          f"(Δ={summary['delta_nor']:+.4f})")
    return summary


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--preprocessing", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mode", choices=["median", "swap"], default="median")
    ap.add_argument("--border_frac", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_subdir", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    model_cfg = load_config(args.model)
    preproc_cfg = load_config(args.preprocessing)

    if args.mode == "median":
        run_median(args.run_dir, model_cfg, preproc_cfg, args.checkpoint,
                   args.border_frac, args)
    elif args.mode == "swap":
        run_swap(args.run_dir, model_cfg, preproc_cfg, args.checkpoint,
                 args.border_frac, args)


if __name__ == "__main__":
    main()
