"""
Dataset-wide preprocessing materializer.

Reads the canonical raw layout produced by `src.preprocessing.dedup` and
`src.preprocessing.split`::

    data/raw/<dataset>/
        train/{images/*.jpg,label.csv}
        val/{images/*.jpg,label.csv}
        test/{images/*.jpg,label.csv}

Applies the configured ``Preprocessor`` to every image and writes
PNG files plus mirrored CSVs under::

    data/preprocessed/<dataset>/
        train/{images/*.png,label.csv}
        val/{images/*.png,label.csv}
        test/{images/*.png,label.csv}

The dataset directory name (e.g. ``figshare``, ``pcosgen``) is preserved,
and the train/val/test boundaries are never altered: every input split
becomes an output split. If a split directory is missing, preprocessing
fails with a clear ``FileNotFoundError`` rather than silently skipping it,
because the split must already exist on disk.

The disk pass uses ``augment=False``. Stochastic augmentation is applied
in memory at training time only.

Output PNGs are the deterministic 8-bit visualization of the
preprocessed image (no z-score is stored on disk). The dataloader applies
the configured normalization (z-score and/or ImageNet stats) at load
time, so the PNG is a faithful, visually inspectable record of the
processed image while model inputs retain the intended numerical
semantics.

Usage::

    python -m src.preprocessing.run_preprocessing \
        --dataset-dir data/raw/figshare \
        --output-dir data/preprocessed/figshare
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from src.preprocessing.preprocess import Preprocessor
from src.utils.config import load_config
from src.utils.seed import set_seed


SPLITS = ("train", "val", "test")
DEFAULT_CONFIG = "configs/preprocessing.yaml"
DEFAULT_INPUT_SIZE = 224
DEFAULT_SEED = 42

# Token-based label mapping for raw label cells.
POSITIVE_TOKENS = ("visible", "infected")
NEGATIVE_TOKENS = ("not-visible", "noninfected", "notinfected")


def _label_from_value(value: str) -> int:
    """Map a raw cell to 0/1. ``None``/blank/``nan``/``Not-visible`` → 0."""
    if value is None:
        return 0
    v = str(value).strip().lower()
    if not v or v == "nan":
        return 0
    # Check NEGATIVE tokens first so substrings like "visible" inside
    # "not-visible" don't trigger a false positive.
    if any(tok in v for tok in NEGATIVE_TOKENS):
        return 0
    if any(tok in v for tok in POSITIVE_TOKENS):
        return 1
    try:
        return 1 if int(float(v)) == 1 else 0
    except (ValueError, TypeError):
        return 0


def _detect_label_column(header: List[str]) -> str:
    """Pick the label column from a CSV header.

    Priority:
      1. Exact ``PCOS-visible`` (Figshare layout)
      2. Any header containing ``visible`` (PCOSGen's long header)
      3. Any header containing ``polycyst``
    """
    for col in header:
        if col.strip() == "PCOS-visible":
            return col
    for col in header:
        if "visible" in col.lower():
            return col
    for col in header:
        if "polycyst" in col.lower():
            return col
    raise ValueError(
        f"Could not detect label column in CSV header: {header}. "
        "Expected 'PCOS-visible' or any column containing 'visible'."
    )


def _read_split_items(dataset_dir: str, split: str) -> List[Tuple[str, int]]:
    """Read a split's ``images/*.jpg|png`` + ``label.csv``.

    Returns ``[(src_path, label_int), ...]``. Every CSV-referenced
    image must exist on disk; missing files raise ``FileNotFoundError``
    pointing at the upstream ``split.py``.
    """
    split_dir = Path(dataset_dir) / split
    img_dir = split_dir / "images"
    csv_path = split_dir / "label.csv"
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Missing {split_dir}/. Run `python -m src.preprocessing.dedup --all` "
            "and `python -m src.preprocessing.split --dataset <figshare|pcosgen>` "
            f"to create the {split}/ folder."
        )
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Missing {img_dir}/")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {csv_path}")

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"{csv_path} has no rows")

    header = list(rows[0].keys())
    label_col = _detect_label_column(header)
    img_col = header[0]
    items: List[Tuple[str, int]] = []
    for row in rows:
        img_name = (row.get(img_col) or "").strip()
        if not img_name or img_name.lower() == "nan":
            continue
        img_path = img_dir / img_name
        if not img_path.is_file():
            raise FileNotFoundError(
                f"{csv_path} references missing image {img_path}. "
                "Re-run `python -m src.preprocessing.split` to rebuild the "
                f"{split}/ directory."
            )
        items.append((str(img_path), _label_from_value(row[label_col])))
    return items


def _encode_png(image: np.ndarray) -> np.ndarray:
    """Convert a preprocessed image to a displayable uint8 for PNG storage.

    The ``Preprocessor.apply(..., augment=False)`` call returns a
    ``float32`` image in ``[0, 255]`` when ``zscore_normalize`` is disabled
    (the default for disk materialization — we deliberately do not store
    z-scored values to keep the PNG a faithful visual record). When
    ``zscore_normalize`` is enabled the float tensor is shifted into
    ``[0, 255]`` via min/max scaling so the encoded PNG remains visually
    faithful to the source after the SRAD/CLAHE pipeline.

    Args:
        image: HxWxC float32 image.

    Returns:
        HxWxC uint8 image, ready for ``cv2.imwrite``.
    """
    img = image.astype(np.float32, copy=False)
    if img.ndim == 2:
        img = img[..., None]
    # If the values are in z-scored range, rescale to [0, 255] so the PNG
    # is a faithful visualization of the *preprocessed* image. We use
    # min/max scaling rather than clipping so we don't lose contrast info.
    if img.max() > 1.5 and img.max() <= 255.5 and img.min() >= 0.0:
        # Already in [0, 255] float range.
        out = np.clip(img, 0, 255).astype(np.uint8)
    else:
        # Either z-scored or otherwise out of [0, 255]. Rescale.
        lo, hi = float(img.min()), float(img.max())
        if hi - lo < 1e-8:
            out = np.zeros_like(img, dtype=np.uint8)
        else:
            scaled = (img - lo) * (255.0 / (hi - lo))
            out = np.clip(scaled, 0, 255).astype(np.uint8)
    # cv2.imwrite expects 3-channel BGR uint8.
    if out.shape[2] == 3:
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    else:
        out_bgr = out
    return out_bgr


def _materialize_split(
    items: List[Tuple[str, int]],
    preprocessor: Preprocessor,
    out_split_dir: Path,
    split_name: str,
) -> Dict[str, int]:
    """Run ``Preprocessor.apply`` over a split and write PNGs.

    Returns a dict of class counts.
    """
    img_out = out_split_dir / "images"
    csv_out = out_split_dir / "label.csv"
    img_out.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    counts: Counter = Counter()
    for src_path, label in items:
        image = cv2.imread(src_path)
        if image is None:
            print(f"  WARNING: Could not read {src_path}, skipping.")
            continue
        processed = preprocessor.apply(image, augment=False)
        png_bytes = _encode_png(processed)
        stem = Path(src_path).stem
        out_path = img_out / f"{stem}.png"
        cv2.imwrite(str(out_path), png_bytes)
        rows.append({"imagePath": out_path.name, "label": int(label)})
        counts[int(label)] += 1

    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["imagePath", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"  [{split_name}] wrote {len(rows)} PNGs "
        f"(infected={counts.get(1, 0)}, noninfected={counts.get(0, 0)})"
    )
    return {"infected": counts.get(1, 0), "noninfected": counts.get(0, 0)}


def run_preprocessing(
    dataset_dir: str,
    output_dir: str,
    config_path: str = DEFAULT_CONFIG,
    input_size: int = DEFAULT_INPUT_SIZE,
    clean: bool = True,
) -> Dict[str, Dict[str, int]]:
    """Materialize a preprocessed PNG copy of ``dataset_dir``.

    Args:
        dataset_dir: Root containing pre-split ``train/``, ``val/``,
            ``test/`` with canonical raw layout.
        output_dir: Where to write the PNG copy. Preserves the dataset
            directory name (caller is expected to pass
            ``data/preprocessed/<dataset>``).
        config_path: Path to the unified preprocessing YAML.
        input_size: Spatial target size for the preprocessor.
        clean: Wipe ``output_dir`` before writing.

    Returns:
        Per-split counts ``{split: {"infected": int, "noninfected": int}}``.
    """
    set_seed(DEFAULT_SEED)
    config = load_config(config_path)
    print(f"[Preprocess] Config: {config.get('name', config_path)}")
    print(f"[Preprocess] Input : {dataset_dir}")
    print(f"[Preprocess] Output: {output_dir}")

    out_root = Path(output_dir)
    if out_root.exists() and clean:
        import shutil
        shutil.rmtree(out_root)
        print(f"[Preprocess] Cleared previous {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    preprocessor = Preprocessor(config, input_size=input_size)

    # Snapshot config so the loader can replicate any stored-side
    # normalization decisions (e.g. z-score enable flag).
    with open(out_root / "preprocessing_metadata.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    summary: Dict[str, Dict[str, int]] = {}
    for split in SPLITS:
        items = _read_split_items(dataset_dir, split)
        out_split_dir = out_root / split
        counts = _materialize_split(items, preprocessor, out_split_dir, split)
        summary[split] = counts
    print("\n[Preprocess] Summary:")
    print(f"  {'split':<8} {'infected':>10} {'noninfected':>14}")
    for split in SPLITS:
        if split in summary:
            print(
                f"  {split:<8} {summary[split]['infected']:>10} "
                f"{summary[split]['noninfected']:>14}"
            )
    print(f"\n[Preprocess] Done. Output -> {out_root}")
    return summary


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize preprocessed PNGs from a pre-split raw dataset."
    )
    parser.add_argument(
        "--dataset-dir", type=str, required=True,
        help="Pre-split raw dataset root (e.g. data/raw/figshare).",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Where to write PNGs (e.g. data/preprocessed/figshare).",
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help=f"Preprocessing YAML (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--input-size", type=int, default=DEFAULT_INPUT_SIZE,
        help=f"Spatial target size (default: {DEFAULT_INPUT_SIZE}).",
    )
    parser.add_argument(
        "--no-clean", action="store_true",
        help="Don't wipe output_dir before writing.",
    )
    args = parser.parse_args(argv)

    if not os.path.isdir(args.dataset_dir):
        print(f"ERROR: --dataset-dir {args.dataset_dir} is not a directory")
        return 1
    run_preprocessing(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        input_size=args.input_size,
        clean=not args.no_clean,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
