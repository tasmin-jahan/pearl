#!/usr/bin/env python3
"""External-validation driver: NO preprocessing (resize + ImageNet normalize only).

Same semantics as ``scripts/eval_external_all.py`` but invokes
``scripts/eval_external_noproc.py`` and writes outputs to
``<run_dir>/external_validation_noproc/pcosgen_noproc.{json,csv}`` so
the existing ``external_validation/pcosgen.{json,csv}`` results are
never overwritten.
"""

import argparse
import os
import subprocess
import sys
import time

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints_root", default="results/ablation/checkpoints")
    ap.add_argument("--external_dir", default="data_external/pcosgen")
    ap.add_argument("--layout", default="pcosgen")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--preprocessings", nargs="+", default=["srad", "gauss"])
    ap.add_argument(
        "--architectures",
        nargs="+",
        default=[
            "swin_tiny", "vit_base", "convnext_tiny",
            "densenet169", "efficientnet_b0",
        ],
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    console = Console()
    pending, skipped = [], []
    for prep in args.preprocessings:
        for arch in args.architectures:
            run_dir = os.path.join(args.checkpoints_root, prep, arch)
            ckpt = os.path.join(run_dir, "best.pt")
            out_json = os.path.join(run_dir, "external_validation_noproc", "pcosgen_noproc.json")
            if not os.path.isfile(ckpt):
                continue
            if os.path.isfile(out_json) and not args.force:
                skipped.append((prep, arch))
                continue
            pending.append((prep, arch, run_dir, ckpt))

    if not pending:
        console.print(f"[green]✓[/] All {len(skipped)} (prep, arch) pairs already evaluated. Nothing to do.")
        return

    console.print(
        f"[bold]External noproc validation:[/] {len(pending)} to run, "
        f"{len(skipped)} already done, dataset={args.external_dir} (layout={args.layout})"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]ExtVal-NoProc[/]"),
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
                sys.executable, "scripts/evaluation/eval_external_noproc.py",
                "--run_dir", run_dir,
                "--model", f"configs/model/{arch}.yaml",
                "--checkpoint", ckpt,
                "--external_dir", args.external_dir,
                "--layout", args.layout,
                "--batch_size", str(args.batch_size),
            ]
            progress.update(task, description=f"[{prep}/{arch}]")
            t0 = time.time()
            res = subprocess.run(cmd, capture_output=True, text=True)
            dt = time.time() - t0
            if res.returncode != 0:
                console.print(f"[red]✗[/] [{prep}/{arch}] failed: {res.stderr.strip().splitlines()[-1] if res.stderr else 'unknown'}")
                if res.stderr:
                    console.print(f"[dim]{res.stderr[-2000:]}[/]")
                progress.update(task, advance=1)
                continue
            tag = "?"
            for line in res.stdout.splitlines():
                if "test_auc_roc:" in line:
                    tag = line.split("test_auc_roc:")[1].strip()
                    break
            console.print(f"[cyan]•[/] [{prep}/{arch}] {dt:.0f}s  noproc_auc={tag}")
            progress.update(task, advance=1)

    console.print(f"[bold green]✓ No-preproc validation complete[/] "
                  f"({len(pending)} runs, {len(skipped)} cached)")


if __name__ == "__main__":
    main()
