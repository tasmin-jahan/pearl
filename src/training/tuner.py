"""
Optuna-based hyperparameter tuner for the best model configuration.

Tunes lr, weight_decay, dropout, freeze_fraction, and batch_size
using TPE sampler with median pruning.
"""

import copy
import time

import optuna
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score

from src.model.builder import build_model
from src.data.dataloader import build_dataloaders
from src.training.losses import build_weighted_loss


def make_objective(model_config, preproc_config, experiment_config, device):
    """Create an Optuna objective function.

    Args:
        model_config: Base model config dict (will be modified per trial).
        preproc_config: Preprocessing config dict.
        experiment_config: Tuning experiment config with search_space.
        device: Device string.

    Returns:
        Callable objective(trial) → float (val_auc).
    """
    search_space = experiment_config.get("search_space", {})
    max_epochs = experiment_config.get("max_epochs", 100)
    patience = experiment_config.get("early_stopping_patience", 20)

    def objective(trial):
        # Sample hyperparameters
        lr = trial.suggest_float("lr",
            search_space["lr"]["low"], search_space["lr"]["high"], log=True)
        weight_decay = trial.suggest_float("weight_decay",
            search_space["weight_decay"]["low"], search_space["weight_decay"]["high"], log=True)
        dropout = trial.suggest_float("dropout",
            search_space["dropout"]["low"], search_space["dropout"]["high"])
        freeze_fraction = trial.suggest_categorical("freeze_fraction",
            search_space["freeze_fraction"]["choices"])
        batch_size = trial.suggest_categorical("batch_size",
            search_space["batch_size"]["choices"])

        # Override config
        cfg = copy.deepcopy(model_config)
        cfg["freeze_fraction"] = freeze_fraction
        cfg["head"]["dropout"] = dropout

        # Build model
        model = build_model(cfg).to(device)

        # Build data
        input_size = cfg.get("input_size", 224)
        train_loader, val_loader, _ = build_dataloaders(
            preproc_config, batch_size=batch_size, input_size=input_size,
        )

        # Loss
        class_weights = train_loader.dataset.get_class_weights()
        criterion = build_weighted_loss(class_weights, device=device)

        # Optimizer + scheduler
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

        best_val_auc = 0.0
        epochs_no_improve = 0

        for epoch in range(1, max_epochs + 1):
            # Train
            model.train()
            for images, labels in train_loader:
                images = images.to(device)
                labels = torch.tensor(labels, dtype=torch.long).to(device) if not isinstance(labels, torch.Tensor) else labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()

            # Validate
            model.eval()
            all_labels, all_probs = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels_t = torch.tensor(labels, dtype=torch.long) if not isinstance(labels, torch.Tensor) else labels
                    probs = F.softmax(model(images), dim=1)
                    all_labels.extend(labels_t.cpu().numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy())

            val_auc = roc_auc_score(np.array(all_labels), np.array(all_probs))

            # Report to Optuna for pruning
            trial.report(val_auc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                break

            scheduler.step()

        return best_val_auc

    return objective
