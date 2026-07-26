"""
Experiment logger: CSV epoch logging, JSON final metrics, and training curve plots.

v3 features:
  - Dual-tqdm progress bars (validation + total epoch elapsed)
  - Safe logging: append-mode CSV on resume (not overwrite), so a
    resumed run continues the existing epoch_log.csv.
  - tqdm.write() everywhere to avoid breaking the progress bar.
"""

import csv
import datetime
import json
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import save_config


EPOCH_LOG_HEADER = [
    "epoch",
    "train_loss",
    "val_loss",
    "train_acc",
    "val_acc",
    "val_auc",
    "val_f1",
    "val_mcc",
    "lr",
    "epoch_time_sec",
]


class ExperimentLogger:
    """Logs training progress to CSV and produces summary outputs.

    Args:
        run_dir: Path to the run directory. Created if it doesn't exist.
        config: Full merged config dict; saved as config.yaml snapshot.
        resume: If True, append to existing epoch_log.csv instead of
            overwriting (used when resuming a checkpoint).
    """

    def __init__(self, run_dir: str, config: dict, resume: bool = False):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)

        # Save config snapshot for exact reproducibility
        if not resume:
            save_config(config, os.path.join(run_dir, "config.yaml"))

        self._csv_path = os.path.join(run_dir, "epoch_log.csv")
        if resume and os.path.isfile(self._csv_path):
            # Append mode — preserve past epochs
            self._csv_file = open(self._csv_path, "a", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._start_epoch = self._get_max_epoch() + 1
        else:
            self._csv_file = open(self._csv_path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(EPOCH_LOG_HEADER)
            self._csv_file.flush()
            self._start_epoch = 1

        print(f"[Logger] Run directory: {run_dir} (resume={resume})")

    def _get_max_epoch(self) -> int:
        if not os.path.isfile(self._csv_path):
            return 0
        try:
            df = pd.read_csv(self._csv_path)
            if "epoch" in df.columns and not df.empty:
                return int(df["epoch"].max())
        except Exception:
            return 0
        return 0

    # ------------------------------------------------------------------
    # Epoch-level logging
    # ------------------------------------------------------------------
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        val_auc: float,
        val_f1: float,
        val_mcc: float,
        lr: float,
        epoch_time: float,
    ) -> None:
        """Append one row to epoch_log.csv and print (using tqdm.write if active)."""
        row = [
            epoch,
            f"{train_loss:.6f}",
            f"{val_loss:.6f}",
            f"{train_acc:.4f}",
            f"{val_acc:.4f}",
            f"{val_auc:.4f}",
            f"{val_f1:.4f}",
            f"{val_mcc:.4f}",
            f"{lr:.2e}",
            f"{epoch_time:.1f}",
        ]
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        line = (
            f"  Epoch {epoch:>3d} | "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}  val_auc={val_auc:.4f}  val_f1={val_f1:.4f} | "
            f"lr={lr:.2e}  time={epoch_time:.1f}s"
        )
        # tqdm.write avoids corrupting progress bars; falls back to print if not active
        try:
            from tqdm import tqdm as _tqdm
            _tqdm.write(line)
        except ImportError:
            print(line)

    # ------------------------------------------------------------------
    # Final metrics
    # ------------------------------------------------------------------
    def log_final_metrics(self, metrics: dict) -> None:
        """Write final_metrics.json with test-set results."""
        path = os.path.join(self.run_dir, "final_metrics.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[Logger] Final metrics saved to {path}")

    # ------------------------------------------------------------------
    # Training curve plots
    # ------------------------------------------------------------------
    def plot_training_curves(self) -> None:
        """Read epoch_log.csv and produce training_curve.png."""
        if not os.path.isfile(self._csv_path):
            print("[Logger] No epoch_log.csv to plot.")
            return
        df = pd.read_csv(self._csv_path)
        if df.empty:
            print("[Logger] No data to plot.")
            return

        best_epoch = int(df.loc[df["val_auc"].idxmax(), "epoch"])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.plot(df["epoch"], df["train_loss"], label="Train Loss", linewidth=1.5)
        ax.plot(df["epoch"], df["val_loss"], label="Val Loss", linewidth=1.5)
        ax.axvline(best_epoch, color="grey", linestyle="--", alpha=0.7,
                   label=f"Best epoch ({best_epoch})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(df["epoch"], df["val_auc"], label="Val AUC",
                linewidth=1.5, color="tab:green")
        ax.axvline(best_epoch, color="grey", linestyle="--", alpha=0.7,
                   label=f"Best epoch ({best_epoch})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AUC-ROC")
        ax.set_title("Validation AUC-ROC")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.run_dir, "training_curve.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Logger] Training curves saved to {plot_path}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._csv_file and not self._csv_file.closed:
            self._csv_file.flush()
            self._csv_file.close()


def make_run_dir(results_dir: str, arch: str, preprocessing: str) -> str:
    """Generate a timestamped run directory path."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{arch}__{preprocessing}__{timestamp}"
    return os.path.join(results_dir, "runs", run_name)
