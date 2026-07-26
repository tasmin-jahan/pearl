"""
Optuna-based hyperparameter tuner for the best model configuration.

Tunes lr, weight_decay, dropout, freeze_fraction, and batch_size
using TPE sampler with median pruning. Reuses :class:`Trainer` so that
training-pipeline improvements (no-decay param groups, NaN guard,
grad clipping, AMP, channels-last) only need to be applied once.
"""

import copy
from typing import Callable, Optional

import optuna

from src.model.builder import build_model
from src.training.trainer import Trainer


class OptunaCallback:
    """Adapter that lets Trainer drive Optuna pruning via a callback hook.

    Optuna's trial.report/should_prune live in the Optuna world, while
    Trainer's epoch loop only knows about logging/saving. The callback
    is invoked once per epoch with the current val_auc; it can prune
    by raising :class:`optuna.exceptions.TrialPruned`.
    """

    def __init__(self, trial: optuna.trial.Trial):
        self.trial = trial

    def __call__(self, epoch: int, val_auc: float) -> None:
        self.trial.report(val_auc, epoch)
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()


def make_objective(
    model_config: dict,
    preproc_config: dict,
    experiment_config: dict,
    device: str,
    build_dataloaders_fn=None,
    logger_factory: Optional[Callable] = None,
) -> Callable:
    """Create an Optuna objective function that delegates to Trainer.

    Args:
        model_config: Base model config dict (will be modified per trial).
        preproc_config: Preprocessing config dict.
        experiment_config: Tuning experiment config with search_space.
        device: Device string.
        build_dataloaders_fn: Optional override (default uses the project's
            build_dataloaders).
        logger_factory: Optional ExperimentLogger factory. Trials use a
            lightweight logger (or None) by default to avoid per-trial
            disk spam.

    Returns:
        Callable ``objective(trial) → float`` returning best_val_auc.
    """
    from src.data.dataloader import build_dataloaders as _default_loader_fn
    from src.training.losses import build_weighted_loss

    if build_dataloaders_fn is None:
        build_dataloaders_fn = _default_loader_fn

    search_space = experiment_config.get("search_space", {})

    def objective(trial: optuna.trial.Trial):
        # ---- 1. Sample hyperparameters ----
        lr = trial.suggest_float(
            "lr",
            search_space["lr"]["low"], search_space["lr"]["high"], log=True,
        )
        weight_decay = trial.suggest_float(
            "weight_decay",
            search_space["weight_decay"]["low"], search_space["weight_decay"]["high"],
            log=True,
        )
        dropout = trial.suggest_float(
            "dropout",
            search_space["dropout"]["low"], search_space["dropout"]["high"],
        )
        freeze_fraction = trial.suggest_categorical(
            "freeze_fraction", search_space["freeze_fraction"]["choices"],
        )
        batch_size = trial.suggest_categorical(
            "batch_size", search_space["batch_size"]["choices"],
        )

        # ---- 2. Build per-trial model config ----
        cfg = copy.deepcopy(model_config)
        cfg["freeze_fraction"] = freeze_fraction
        cfg["head"]["dropout"] = dropout
        cfg["lr"] = lr
        cfg["weight_decay"] = weight_decay

        # ---- 3. Build model, data, loss (mirrors trainer.py inputs) ----
        model = build_model(cfg)
        input_size = cfg.get("input_size", 224)
        sampler = cfg.get("sampler", "shuffle")
        train_loader, val_loader, test_loader = build_dataloaders_fn(
            preproc_config, batch_size=batch_size, input_size=input_size,
            sampler=sampler,
        )
        class_weights = train_loader.dataset.get_class_weights()
        criterion = build_weighted_loss(class_weights, device=device)

        # ---- 4. Trainer plumbing ----
        logger = (
            logger_factory(trial) if callable(logger_factory)
            else _NullLogger()
        )
        callback = OptunaCallback(trial)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            criterion=criterion,
            config=cfg,
            logger=logger,
            device=device,
            checkpoint_path=f"results/tuning/trial_{trial.number}.pt",
            on_epoch_end=callback,
            silent=True,
        )

        try:
            metrics = trainer.train()
            return metrics["val_auc_best"]
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            # TrialDivergedError caught here: prune the trial.
            from src.training.trainer import TrialDivergedError
            if isinstance(e, TrialDivergedError):
                print(f"[Tuner] Trial pruned: {e}")
                raise optuna.exceptions.TrialPruned()
            raise

    return objective


class _NullLogger:
    """No-op logger used by default during Optuna trials.

    Avoids per-trial CSV/JSON spam during sweeps. Trainer only calls
    ``log_epoch`` and a few other optional methods.
    """

    def log_epoch(self, *args, **kwargs):
        pass

    def log_final_metrics(self, *args, **kwargs):
        pass

    def plot_training_curves(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass
