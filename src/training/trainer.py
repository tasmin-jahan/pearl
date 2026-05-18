"""
Training loop with AdamW, cosine annealing, and early stopping on val AUC.
"""

import time
import datetime
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef

from src.training.checkpoint import save_checkpoint
from src.utils.logging import ExperimentLogger


class Trainer:
    """Full training loop manager.

    Args:
        model: The model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        test_loader: Test DataLoader.
        criterion: Loss function.
        config: Training config dict.
        logger: ExperimentLogger instance.
        device: Device string ('cuda' or 'cpu').
        checkpoint_path: Where to save the best checkpoint.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        criterion: nn.Module,
        config: dict,
        logger: ExperimentLogger,
        device: str = "cuda",
        checkpoint_path: str = "results/checkpoints/best.pt",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.config = config
        self.logger = logger
        self.device = device
        self.checkpoint_path = checkpoint_path

        # Optimizer
        lr = config.get("lr", 1e-4)
        weight_decay = config.get("weight_decay", 1e-2)
        beta1 = config.get("beta1", 0.9)
        beta2 = config.get("beta2", 0.999)
        eps = config.get("eps", 1e-8)

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )

        # Scheduler
        max_epochs = config.get("max_epochs", 100)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs
        )

        # Early stopping
        self.patience = config.get("early_stopping_patience", 20)
        self.max_epochs = max_epochs
        self.best_val_auc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0

    def train(self) -> dict:
        """Run the full training loop.

        Returns:
            Dict of final test-set metrics.
        """
        print(f"\n{'='*60}")
        print(f"Training for up to {self.max_epochs} epochs (patience={self.patience})")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(1, self.max_epochs + 1):
            epoch_start = time.time()

            # --- Train one epoch ---
            train_loss, train_acc = self._train_one_epoch()

            # --- Validate ---
            val_loss, val_acc, val_auc, val_f1, val_mcc = self._validate()

            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log
            self.logger.log_epoch(
                epoch, train_loss, val_loss,
                train_acc, val_acc, val_auc, val_f1, val_mcc,
                current_lr, epoch_time,
            )

            # Check for improvement
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_auc,
                    self.checkpoint_path,
                )
            else:
                self.epochs_without_improvement += 1

            # Step scheduler
            self.scheduler.step()

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\n[Trainer] Early stopping at epoch {epoch} "
                      f"(no improvement for {self.patience} epochs)")
                break

        total_time = time.time() - start_time
        stopped_early = self.epochs_without_improvement >= self.patience
        epochs_trained = epoch

        print(f"\n[Trainer] Training complete in {total_time:.0f}s")
        print(f"[Trainer] Best val AUC: {self.best_val_auc:.4f} at epoch {self.best_epoch}")

        # Load best checkpoint for test evaluation
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])

        # Evaluate on test set
        test_metrics = self._evaluate_test()

        # Build final metrics dict
        final_metrics = {
            "best_epoch": self.best_epoch,
            "epochs_trained": epochs_trained,
            "stopped_early": stopped_early,
            "val_auc_best": self.best_val_auc,
            "total_train_time_sec": round(total_time, 1),
            "params_total": sum(p.numel() for p in self.model.parameters()),
            "params_trainable": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "n_train": len(self.train_loader.dataset),
            "n_val": len(self.val_loader.dataset),
            "n_test": len(self.test_loader.dataset),
        }
        final_metrics.update(test_metrics)

        self.logger.log_final_metrics(final_metrics)
        self.logger.plot_training_curves()
        self.logger.close()

        return final_metrics

    def _train_one_epoch(self):
        """Train for one epoch. Returns (avg_loss, accuracy)."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = torch.tensor(labels, dtype=torch.long).to(self.device) if not isinstance(labels, torch.Tensor) else labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        avg_loss = running_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def _validate(self):
        """Validate and return (loss, acc, auc, f1, mcc)."""
        self.model.eval()
        running_loss = 0.0
        all_labels = []
        all_probs = []
        all_preds = []

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = torch.tensor(labels, dtype=torch.long).to(self.device) if not isinstance(labels, torch.Tensor) else labels.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        total = len(all_labels)
        avg_loss = running_loss / total
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)

        accuracy = (all_preds == all_labels).mean()
        auc = roc_auc_score(all_labels, all_probs)
        f1 = f1_score(all_labels, all_preds)
        mcc = matthews_corrcoef(all_labels, all_preds)

        return avg_loss, accuracy, auc, f1, mcc

    @torch.no_grad()
    def _evaluate_test(self):
        """Evaluate on test set and return metrics dict."""
        self.model.eval()
        all_labels = []
        all_probs = []
        all_preds = []

        for images, labels in self.test_loader:
            images = images.to(self.device)
            labels = torch.tensor(labels, dtype=torch.long).to(self.device) if not isinstance(labels, torch.Tensor) else labels.to(self.device)

            logits = self.model(images)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)

        # Compute all 7 metrics
        from src.evaluation.metrics import compute_all_metrics
        return compute_all_metrics(all_labels, all_preds, all_probs)
