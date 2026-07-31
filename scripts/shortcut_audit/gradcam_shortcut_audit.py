#!/usr/bin/env python3
"""Grad-CAM-based audit of the "model looks at the border" shortcut hypothesis.

For each trained checkpoint, runs Grad-CAM on a sample of Figshare test
images, and records a *border-attention fraction* — the share of total
Grad-CAM activation mass that falls inside the outer 5% strip of the
preprocessed image. This single number tells us whether the model is
attending to anatomical content (interior) or to padding artefacts
(border).

Implementation:
  - Pure-PyTorch hooks (no torchcam dependency).
  - Auto-detects backbone output shape:
      * (B, C, H, W)   — ResNet, EfficientNet, ConvNeXt, MobileNet, DenseNet
      * (B, H, W, C)   — Swin
      * (B, N, C)      — ViT  (N = 1 + H_p * W_p tokens)
  - Re-uses the model's existing head (no surgical rewire). For ViT/Swin
    we synthesise a (B, C, H, W) feature map from the token grid before
    handing it to the head, which works because the head only does
    ``AdaptiveAvgPool2d`` and then MLP.

Outputs (under <run_dir>/shortcut_audit/):
  - per_image_gradcam.csv         — sample_id, true_label, pred_label,
                                    prob_infected, border_frac, interior_frac,
                                    peak_xy, peak_val, entropy
  - per_image_gradcam/<idx>.png   — input | gradcam | overlay panel
  - shortcut_summary.json         — per-class medians of border_frac +
                                    the two-tail p-value vs uniform (0.05)
  - shortcut_summary.txt          — human-readable summary
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.dataloader import build_dataloaders
from src.model.builder import build_model
from src.training.checkpoint import load_checkpoint


# =====================================================================
# Border / interior masks
# =====================================================================

def make_border_mask(h: int, w: int, frac: float = 0.05) -> np.ndarray:
    """Return a boolean (h, w) array that's True on the outer ``frac`` strip."""
    b = max(1, int(round(min(h, w) * frac)))
    mask = np.zeros((h, w), dtype=bool)
    mask[:b, :] = True
    mask[-b:, :] = True
    mask[:, :b] = True
    mask[:, -b:] = True
    return mask


# =====================================================================
# Grad-CAM via forward/backward hooks
# =====================================================================

class _HookedGradCAM:
    """Capture activations + gradients at ``backbone.forward_features`` output.

    ``model.backbone(x)`` runs ``forward_features`` followed by timm's global
    pool. Hooking the backbone module itself would therefore see only the
    already-pooled ``(B, C)`` tensor, which is useless for Grad-CAM. We instead
    temporarily wrap ``forward_features`` and retain its spatial output.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.activations = None
        self._orig_forward_features = None

    def _attach(self):
        backbone = self.model.backbone
        self._orig_forward_features = backbone.forward_features

        def wrapped_forward_features(*args, **kwargs):
            out = self._orig_forward_features(*args, **kwargs)
            out.retain_grad()
            self.activations = out
            return out

        backbone.forward_features = wrapped_forward_features

    def _detach(self):
        if self._orig_forward_features is not None:
            self.model.backbone.forward_features = self._orig_forward_features
            self._orig_forward_features = None

    @staticmethod
    def _normalise_spatial(act: torch.Tensor, grad: torch.Tensor
                           ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """Return a (B, C, H, W) feature map, matching gradient, and size.

        Handles all three timm output shapes.
        """
        # Standardise shape
        if act.dim() == 4:
            # (B, C, H, W) OR (B, H, W, C)  — distinguish by C in dim 1 vs 3
            if act.shape[1] <= 64 and act.shape[3] > 64:
                # (B, H, W, C) — Swin
                a = act.permute(0, 3, 1, 2).contiguous()
                g = grad.permute(0, 3, 1, 2).contiguous()
                return a, g, (a.shape[2], a.shape[3])
            elif act.shape[1] > 64:
                # (B, C, H, W) — ResNet/EffNet/ConvNeXt/DenseNet/MobileNet
                return act, grad, (act.shape[2], act.shape[3])
            else:
                # (B, H, W, C) — Swin fallback
                a = act.permute(0, 3, 1, 2).contiguous()
                g = grad.permute(0, 3, 1, 2).contiguous()
                return a, g, (a.shape[2], a.shape[3])
        elif act.dim() == 3:
            # (B, N, C) — ViT. Drop the CLS token (assumed at index 0).
            if act.shape[1] == grad.shape[1]:
                pass
            no_cls = act[:, 1:, :]
            no_cls_g = grad[:, 1:, :]
            n = no_cls.shape[1]
            side = int(round(n ** 0.5))
            assert side * side == n, f"ViT token count is not square: {n}"
            a = no_cls.transpose(1, 2).contiguous().view(
                no_cls.shape[0], no_cls.shape[2], side, side,
            )
            g = no_cls_g.transpose(1, 2).contiguous().view(
                no_cls_g.shape[0], no_cls_g.shape[2], side, side,
            )
            return a, g, (side, side)
        else:
            raise ValueError(f"Unsupported feature shape: {act.shape}")

    def __call__(self, x: torch.Tensor, target_class: int = None
                 ) -> Tuple[np.ndarray, int]:
        """Run a forward+backward pass and return a (H_in, W_in) heatmap."""
        self.model.zero_grad()
        self._attach()
        try:
            logits = self.model(x)
            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())
            score = logits[0, target_class]
            score.backward()
            act = self.activations
            grad = act.grad
        finally:
            self._detach()

        feat, g, (hp, wp) = self._normalise_spatial(act, grad)
        if g is None:
            g = torch.ones_like(feat)
        weights = g.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * feat).sum(dim=1, keepdim=False)  # (1, hp, wp)
        cam = F.relu(cam)[0].detach().cpu().numpy()

        # Resize to input spatial size
        h, w = x.shape[2], x.shape[3]
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        # Normalise to [0, 1]
        cam = cam - cam.min()
        mx = cam.max()
        if mx > 1e-8:
            cam = cam / mx
        return cam, target_class


# =====================================================================
# Image-level statistics
# =====================================================================

def border_attention_stats(cam: np.ndarray, border_frac: float = 0.05):
    """Compute border/interior attention fractions and peak location."""
    h, w = cam.shape
    border_mask = make_border_mask(h, w, border_frac)
    total = float(cam.sum() + 1e-8)
    border_mass = float(cam[border_mask].sum())
    interior_mass = float(cam[~border_mask].sum())
    border_frac_mass = border_mass / total
    interior_frac_mass = interior_mass / total
    # Border strip is ~ (1 - (1-2*frac)^2) of the area. For frac=0.05,
    # that's 1 - 0.9^2 = 0.19 of pixels. So the "uniform" expectation
    # for border_frac_mass is 0.19.
    expected_border = float(border_mask.mean())
    peak_idx = int(np.argmax(cam))
    py, px = divmod(peak_idx, w)
    peak_val = float(cam[py, px])
    return {
        "border_mass_frac": border_frac_mass,
        "interior_mass_frac": interior_frac_mass,
        "expected_border_uniform": expected_border,
        "peak_y": py,
        "peak_x": px,
        "peak_val": peak_val,
        "is_border_peak": bool(border_mask[py, px]),
    }


# =====================================================================
# Visualization (single panel per image)
# =====================================================================

def save_overlay_panel(
    out_path: str,
    original_uint8: np.ndarray,
    cam: np.ndarray,
    true_label: int,
    pred_label: int,
    prob: float,
    stats: dict,
):
    """Save a 1×3 panel: original | Grad-CAM jet | overlay with stats."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original_uint8)
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")
    overlay = (0.5 * original_uint8.astype(np.float32)
               + 0.5 * plt.cm.jet(cam)[..., :3] * 255)
    axes[2].imshow(overlay.astype(np.uint8))
    axes[2].set_title(
        f"Overlay — true={'P' if true_label else 'H'} "
        f"pred={'P' if pred_label else 'H'} "
        f"p(P)={prob:.2f}\n"
        f"border_attn={stats['border_mass_frac']:.2%}  "
        f"(uniform={stats['expected_border_uniform']:.0%})  "
        f"{'SHORTCUT' if stats['is_border_peak'] else ''}"
    )
    axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# Lazy import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================================
# Main audit driver
# =====================================================================

def _to_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert a (C, H, W) image-net-normalised tensor back to a displayable uint8."""
    a = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    a = a * std + mean
    a = np.clip(a, 0, 1)
    return (a * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--preprocessing", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--n_samples", type=int, default=80,
                    help="Total samples to draw from the test set (split "
                         "evenly across the 2 classes).")
    ap.add_argument("--border_frac", type=float, default=0.05)
    ap.add_argument("--out_subdir", default="shortcut_audit")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    model_config = load_config(args.model)
    preproc_config = load_config(args.preprocessing)
    input_size = int(model_config.get("input_size", 224))
    arch = model_config.get("name", args.model)
    prep_name = preproc_config.get("name", args.preprocessing)
    print(f"[ShortcutAudit] {arch} / {prep_name}  input={input_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_loader = build_dataloaders(
        preproc_config, batch_size=1, input_size=input_size,
    )
    model = build_model(model_config)
    load_checkpoint(model, args.checkpoint, device=device)
    model.to(device)
    model.eval()

    cam_engine = _HookedGradCAM(model)

    out_dir = os.path.join(args.run_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "panels"), exist_ok=True)

    # Collect the test set in memory (small enough; <5000 at 224×224)
    images, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            images.append(x)
            if torch.is_tensor(y):
                labels.append(int(y.item()))
            else:
                labels.append(int(y))
    images = torch.cat(images, dim=0)
    labels = np.array(labels)
    paths = [s[0] for s in test_loader.dataset.samples]
    print(f"[ShortcutAudit] test set: {len(labels)} samples "
          f"(infected={int((labels==1).sum())}, healthy={int((labels==0).sum())})")

    # Stratified sample: n/2 per class
    rng = np.random.default_rng(args.seed)
    n_per = args.n_samples // 2
    idx_inf = np.where(labels == 1)[0]
    idx_h = np.where(labels == 0)[0]
    sel_inf = rng.choice(idx_inf, size=min(n_per, len(idx_inf)), replace=False)
    sel_h = rng.choice(idx_h, size=min(n_per, len(idx_h)), replace=False)
    sel = np.concatenate([sel_inf, sel_h])
    rng.shuffle(sel)
    print(f"[ShortcutAudit] sampling {len(sel)} images "
          f"({len(sel_inf)} infected, {len(sel_h)} healthy)")

    rows = []
    class_border = defaultdict(list)
    class_peak_border = defaultdict(int)
    n_correct = 0

    for k, idx in enumerate(sel):
        x = images[idx:idx + 1].to(device)
        cam, pred = cam_engine(x, target_class=None)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        prob_inf = float(probs[1])
        true_lbl = int(labels[idx])
        pred_lbl = int(pred)
        if pred_lbl == true_lbl:
            n_correct += 1
        stats = border_attention_stats(cam, border_frac=args.border_frac)
        class_border[true_lbl].append(stats["border_mass_frac"])
        if stats["is_border_peak"]:
            class_peak_border[true_lbl] += 1
        entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        rows.append({
            "sample_id": int(idx),
            "path": paths[idx],
            "true_label": true_lbl,
            "pred_label": pred_lbl,
            "correct": int(pred_lbl == true_lbl),
            "prob_infected": prob_inf,
            "entropy": entropy,
            **stats,
        })
        if k < 24:  # save first 24 panel images for visual review
            img_uint8 = _to_uint8(images[idx])
            save_overlay_panel(
                os.path.join(out_dir, "panels", f"sample_{int(idx):04d}.png"),
                img_uint8, cam, true_lbl, pred_lbl, prob_inf, stats,
            )

    # Write per-image CSV
    csv_path = os.path.join(out_dir, "per_image_gradcam.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[ShortcutAudit] wrote {len(rows)} rows -> {csv_path}")

    # Per-class summary
    expected = float(make_border_mask(input_size, input_size, args.border_frac).mean())
    summary = {
        "arch": arch,
        "preprocessing": prep_name,
        "n_samples": len(rows),
        "n_correct": n_correct,
        "n_per_class": {int(k): len(v) for k, v in class_border.items()},
        "border_mass_frac_median": {
            int(k): float(np.median(v)) for k, v in class_border.items()
        },
        "border_mass_frac_mean": {
            int(k): float(np.mean(v)) for k, v in class_border.items()
        },
        "border_peak_count": {
            int(k): int(v) for k, v in class_peak_border.items()
        },
        "expected_border_uniform_frac": expected,
        "border_frac": args.border_frac,
    }
    json_path = os.path.join(out_dir, "shortcut_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[ShortcutAudit] wrote summary -> {json_path}")

    # Print one-line verdict
    for k in (0, 1):
        if k in class_border and class_border[k]:
            m = float(np.median(class_border[k]))
            print(f"  class={k}  median border_attn={m:.3f}  "
                  f"uniform_expected={expected:.3f}  "
                  f"border_peaks={class_peak_border[k]}/{len(class_border[k])}")


if __name__ == "__main__":
    main()
