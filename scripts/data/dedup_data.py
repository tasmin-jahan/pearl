#!/usr/bin/env python3
"""
Find and remove exact-byte duplicates from a class-folder dataset.

Three modes of operation:

  report     — print duplicate counts per class, write a CSV listing
               every duplicate file. Nothing on disk is moved.
  quarantine — move duplicate files into data/_duplicates/<class>/ so
               the originals are preserved but excluded from training.
               Reversible: ``mv data/_duplicates/<class>/* data/<class>/``.
  remove     — permanently delete duplicates. DESTRUCTIVE; requires --yes.

The Figshare PCOS dataset ships with substantial exact-byte duplication:
as of the v3 EDA run, only 812 / 5,000 non-infected and 3,184 / 6,784
infected images are unique (16.2% and 46.9% respectively). Without
deduplication the model effectively trains on a much smaller unique
sample than reported.

Usage:
    python scripts/dedup_data.py --data_dir data --mode report
    python scripts/dedup_data.py --data_dir data --mode quarantine
    python scripts/dedup_data.py --data_dir data --mode remove --yes
"""

import argparse
import csv
import hashlib
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path


def md5_of(path: str, chunk: int = 1 << 16) -> str:
    """Stream-hash a file's bytes."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def collect_hashes(class_dir: Path, extensions: tuple) -> dict:
    """Return {md5: [path, ...]} for every image file under class_dir.

    Sorted paths so the *first* lexicographic occurrence is the canonical
    keeper — deterministic across runs.
    """
    hashes: dict = defaultdict(list)
    for fname in sorted(os.listdir(class_dir)):
        if not fname.lower().endswith(extensions):
            continue
        p = class_dir / fname
        if not p.is_file():
            continue
        hashes[md5_of(str(p))].append(p)
    return hashes


def build_report(per_class_hashes: dict, class_names: list) -> tuple:
    """Return (summary_rows, detail_rows) for CSV output.

    summary_rows: per-class totals + counts of duplicates that would be removed.
    detail_rows: one row per duplicate file (path, class, kept_or_removed, hash).
    """
    summary, detail = [], []
    for cls, hashes in zip(class_names, per_class_hashes):
        total = sum(len(v) for v in hashes.values())
        unique = len(hashes)
        n_dup = total - unique
        summary.append({
            "class": cls,
            "n_files": total,
            "n_unique": unique,
            "n_duplicates": n_dup,
            "duplicate_fraction": round(n_dup / total, 4) if total else 0.0,
        })
        for h, paths in sorted(hashes.items()):
            # First path (lexicographic) is the keeper; rest are dupes.
            kept = paths[0]
            for p in paths:
                detail.append({
                    "class": cls,
                    "path": str(p),
                    "md5": h,
                    "kept": int(p == kept),
                })
    return summary, detail


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Root dataset directory (with infected/ and noninfected/)")
    parser.add_argument("--mode", type=str, choices=["report", "quarantine", "remove"],
                        default="report",
                        help="Action: report (default), quarantine, or remove")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the destructive-mode confirmation prompt")
    parser.add_argument("--classes", type=str, nargs="+",
                        default=["infected", "noninfected"],
                        help="Class folder names to scan")
    parser.add_argument("--extensions", type=str, nargs="+",
                        default=[".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
                        help="File extensions to consider")
    parser.add_argument("--report_dir", type=str, default="results/eda",
                        help="Where to write dedup_report.csv and dedup_details.csv")
    parser.add_argument("--quarantine_dir", type=str, default="data/_duplicates",
                        help="Where to move quarantined duplicates")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit(f"ERROR: --data_dir {data_dir} does not exist")

    exts = tuple(e.lower() if e.startswith(".") else "." + e.lower()
                 for e in args.extensions)

    print(f"[Dedup] Scanning {data_dir} for {exts}")
    print(f"[Dedup] Mode: {args.mode}")
    print()

    per_class_hashes = []
    for cls in args.classes:
        cls_dir = data_dir / cls
        if not cls_dir.is_dir():
            sys.exit(f"ERROR: missing class folder {cls_dir}")
        per_class_hashes.append(collect_hashes(cls_dir, exts))

    summary, detail = build_report(per_class_hashes, args.classes)

    # ---- Print human-readable summary ----
    total_files = sum(s["n_files"] for s in summary)
    total_dup = sum(s["n_duplicates"] for s in summary)
    total_unique = sum(s["n_unique"] for s in summary)
    print(f"{'class':14s} {'files':>8s} {'unique':>8s} {'duplicates':>12s} {'%':>6s}")
    for s in summary:
        print(f"{s['class']:14s} {s['n_files']:>8,} {s['n_unique']:>8,} "
              f"{s['n_duplicates']:>12,} {s['duplicate_fraction']*100:>5.1f}%")
    print(f"{'TOTAL':14s} {total_files:>8,} {total_unique:>8,} "
          f"{total_dup:>12,} {total_dup/total_files*100:>5.1f}%")

    # ---- Write CSV report ----
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "dedup_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    with open(report_dir / "dedup_details.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0].keys()))
        w.writeheader()
        w.writerows(detail)
    print(f"\n[Dedup] Report  → {report_dir / 'dedup_report.csv'}")
    print(f"[Dedup] Details → {report_dir / 'dedup_details.csv'}")

    if args.mode == "report":
        return

    # ---- Quarantine / Remove ----
    n_acted = 0
    if args.mode == "remove":
        if not args.yes:
            print(f"\n[Dedup] About to PERMANENTLY DELETE {total_dup} files.")
            print(f"[Dedup] Pass --yes to confirm.")
            sys.exit(1)
        for row in detail:
            if row["kept"]:
                continue
            p = Path(row["path"])
            if p.exists():
                p.unlink()
                n_acted += 1
        print(f"\n[Dedup] Deleted {n_acted} duplicate files.")
    else:  # quarantine
        qdir = Path(args.quarantine_dir)
        qdir.mkdir(parents=True, exist_ok=True)
        for row in detail:
            if row["kept"]:
                continue
            src = Path(row["path"])
            if not src.exists():
                continue
            dest = qdir / row["class"] / src.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            # If a same-named file is already quarantined, suffix it
            if dest.exists():
                stem, suf = dest.stem, dest.suffix
                i = 1
                while True:
                    cand = qdir / row["class"] / f"{stem}__{i}{suf}"
                    if not cand.exists():
                        dest = cand
                        break
                    i += 1
            shutil.move(str(src), str(dest))
            n_acted += 1
        print(f"\n[Dedup] Moved {n_acted} duplicate files to {qdir}/")
        print(f"[Dedup] To restore: mv {qdir}/<class>/* {data_dir}/<class>/")


if __name__ == "__main__":
    main()