#!/usr/bin/env python3
"""Run gradcam_shortcut_audit.py across every checkpoint under
``--checkpoints_root`` and emit per-run summaries plus one combined
figure.

Defaults to the no-padding ablation root (``results/ablation_nopad/checkpoints``)
and the matching preprocessing configs, so re-running this gives the
Grad-CAM evidence for the no-pad run that closes Finding 9's causal
loop.

Usage:
  python scripts/gradcam_shortcut_audit_all.py
  python scripts/gradcam_shortcut_audit_all.py --checkpoints_root results/ablation/checkpoints --preprocessings srad gauss
"""
import argparse
import os
import subprocess
import sys
import time

from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
    TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)


ARCHES = [
    "swin_tiny", "vit_base", "convnext_tiny",
    "densenet169", "efficientnet_b0",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints_root", default="results/ablation_nopad/checkpoints")
    ap.add_argument("--configs_root", default="configs/preprocessing.yaml",
                    help="Path to the unified preprocessing config (kept as a "
                         "name for back-compat with the old per-preprocessing "
                         "YAML layout).")
    ap.add_argument("--preprocessings", nargs="+", default=["default"],
                    help="Logical preprocessing names to iterate over (for "
                         "labelling outputs only — all use the same config "
                         "now). E.g. ['srad', 'gauss'] to label output dirs.")
    ap.add_argument("--architectures", nargs="+", default=ARCHES)
    ap.add_argument("--n_samples", type=int, default=80)
    ap.add_argument("--border_frac", type=float, default=0.05)
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if per-run summary already exists.")
    args = ap.parse_args()

    console = Console()
    pending, skipped = [], []
    for prep in args.preprocessings:
        for arch in args.architectures:
            run_dir = os.path.join(args.checkpoints_root, prep, arch)
            ckpt = os.path.join(run_dir, "best.pt")
            out_json = os.path.join(run_dir, "shortcut_audit", "shortcut_summary.json")
            if not os.path.isfile(ckpt):
                continue
            if os.path.isfile(out_json) and not args.force:
                skipped.append((prep, arch))
                continue
            pending.append((prep, arch, run_dir, ckpt))

    total = len(pending) + len(skipped)
    if not pending:
        console.print(f"[green]✓[/] All {total} (prep, arch) pairs already audited. Nothing to do.")
        return

    console.print(
        f"[bold]Grad-CAM shortcut audit:[/] {len(pending)} to run, "
        f"{len(skipped)} already done."
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold magenta]GradCAM[/]"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("starting…", total=len(pending))
        for prep, arch, run_dir, ckpt in pending:
            cmd = [
                sys.executable, "scripts/gradcam_shortcut_audit.py",
                "--model", f"configs/model/{arch}.yaml",
                "--preprocessing", args.configs_root,
                "--checkpoint", ckpt,
                "--run_dir", run_dir,
                "--n_samples", str(args.n_samples),
                "--border_frac", str(args.border_frac),
            ]
            progress.update(task, description=f"[{prep}/{arch}]")
            t0 = time.time()
            res = subprocess.run(cmd, capture_output=True, text=True)
            dt = time.time() - t0
            if res.returncode != 0:
                tail = (res.stderr or "unknown").strip().splitlines()[-1]
                console.print(
                    f"[red]✗[/] [{prep}/{arch}] failed in {dt:.0f}s: {tail}"
                )
                progress.update(task, advance=1)
                continue
            console.print(f"[green]✓[/] [{prep}/{arch}] {dt:.0f}s")
            progress.update(task, advance=1)

    console.print(f"[bold green]✓ Grad-CAM audit complete[/] "
                  f"({len(pending)} runs, {len(skipped)} cached)")


if __name__ == "__main__":
    main()