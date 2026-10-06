"""
05_cli_calibration — Step 5 of the ensemble pipeline.

Orchestrates the two-pass ensemble calibration flow used by
``src.calibration.run_calibration`` when ``--ensemble`` is set.

Pipeline:
    00_load → 02_per_model_calibration (Pass 1)
            → 03_ensemble_calibration (Pass 2)
            → reliability diagrams + bin CSVs + results JSON.

Single-model calibration continues to live in
``src.calibration.run_calibration`` because it has no ensemble content.
"""
from __future__ import annotations

import json
import os
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch

from src.calibration.ece import compute_bin_data, compute_ece
from src.calibration.reliability import plot_reliability_diagram
from src.calibration.temperature_scaling import fit_temperature
from src.data.dataloader import build_dataloaders
from src.evaluation.evaluator import collect_logits_and_labels
from src.ensemble import (
    apply_ensemble_temperature,
    apply_per_model_temperature,
    fit_ensemble_temperature,
    fit_per_model_temperature,
    load_ensemble,
)
from src.utils.config import load_config


def _model_config_from_name(
    model_name: str, model_configs_dir: str = "configs/model",
) -> dict:
    """Resolve a model name or YAML path to a model-build config dict."""
    cfg_path = (
        model_name
        if model_name.endswith((".yaml", ".yml"))
        else os.path.join(model_configs_dir, f"{model_name}.yaml")
    )
    return load_config(cfg_path)


def _labels_from_loader(loader) -> np.ndarray:
    """Walk a DataLoader once and return its labels as a numpy array."""
    labels: List[np.ndarray] = []
    for _, lab in loader:
        if isinstance(lab, torch.Tensor):
            labels.append(lab.cpu().numpy())
        else:
            labels.append(np.asarray(lab))
    return np.concatenate(labels, axis=0)


def run_calibration_ensemble(
    dataset_dir: str,
    preproc_config: dict,
    model_specs: Sequence[tuple],
    n_bins: int,
    batch_size: int,
    out_dir: str,
    device: str,
    model_configs_dir: str = "configs/model",
) -> dict:
    """Two-pass ensemble calibration driver.

    Args:
        dataset_dir: Preprocessed dataset root with train/val/test.
        preproc_config: Preprocessing YAML loaded as dict.
        model_specs: Sequence of ``(model_name_or_yaml, checkpoint_path)``.
        n_bins: Number of reliability bins.
        batch_size: DataLoader batch size.
        out_dir: Where to write artifacts.
        device: Device string.
        model_configs_dir: Folder holding model-name YAMLs.

    Returns:
        Results dict (also written to ``out_dir/calibration_results.json``).
    """
    if len(model_specs) < 2:
        raise ValueError("Ensemble mode requires at least two --model/--checkpoint pairs")

    _, val_loader, test_loader = build_dataloaders(
        dataset_dir,
        preproc_config,
        batch_size=batch_size,
        num_workers=2,
        sampler="none",
        drop_last=False,
    )

    # ---- 00_load ----
    model_configs = [_model_config_from_name(name, model_configs_dir) for name, _ in model_specs]
    ckpt_paths = [path for _, path in model_specs]
    models = load_ensemble(model_configs, ckpt_paths, device=device)
    arch_names = [name for name, _ in model_specs]

    # ---- Pass 1: per-model T on val ----
    val_loaders = [val_loader] * len(models)
    temperatures: List[float] = fit_per_model_temperature(
        models, val_loaders, device=device,
    )

    # Average per-model calibrated probs on test (Pass 1 output).
    avg_probs, test_labels = apply_per_model_temperature(
        models, test_loader, temperatures, device=device,
    )

    pos_probs = avg_probs[:, 1]
    ece_after_pass1 = compute_ece(pos_probs, test_labels, n_bins=n_bins)
    print(f"[Pass 1] ECE after per-model calibration: {ece_after_pass1:.4f}")

    # ---- Pass 2: ensemble T on val average ----
    val_avg_probs, _ = apply_per_model_temperature(
        models, val_loader, temperatures, device=device,
    )
    val_labels_np = _labels_from_loader(val_loader)

    T_ens = fit_ensemble_temperature(val_avg_probs, val_labels_np)
    cal_probs = apply_ensemble_temperature(avg_probs, T_ens)
    ece_after_pass2 = compute_ece(cal_probs[:, 1], test_labels, n_bins=n_bins)
    print(f"[Pass 2] ECE after ensemble T={T_ens:.2f}: {ece_after_pass2:.4f}")

    # ---- Artifacts ----
    os.makedirs(out_dir, exist_ok=True)
    plot_reliability_diagram(
        compute_bin_data(pos_probs, test_labels, n_bins=n_bins),
        title="Ensemble — After per-model T",
        save_path=os.path.join(out_dir, "reliability_p1.png"),
    )
    plot_reliability_diagram(
        compute_bin_data(cal_probs[:, 1], test_labels, n_bins=n_bins),
        title=f"Ensemble — After ensemble T (T={T_ens:.2f})",
        save_path=os.path.join(out_dir, "reliability_p2.png"),
    )
    pd.DataFrame(
        compute_bin_data(pos_probs, test_labels, n_bins=n_bins)
    ).to_csv(os.path.join(out_dir, "bin_data_pass1.csv"), index=False)
    pd.DataFrame(
        compute_bin_data(cal_probs[:, 1], test_labels, n_bins=n_bins)
    ).to_csv(os.path.join(out_dir, "bin_data_pass2.csv"), index=False)

    results = {
        "archs": arch_names,
        "per_model_temperatures": [round(float(t), 4) for t in temperatures],
        "ensemble_temperature": round(float(T_ens), 4),
        "ece_after_pass1": round(float(ece_after_pass1), 4),
        "ece_after_pass2": round(float(ece_after_pass2), 4),
        "n_bins": n_bins,
        "n_test_samples": int(len(test_labels)),
    }
    with open(os.path.join(out_dir, "calibration_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Ensemble Calibration] Saved to {out_dir}")
    return results