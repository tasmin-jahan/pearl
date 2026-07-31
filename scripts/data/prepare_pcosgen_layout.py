#!/usr/bin/env python3
"""Materialise the PCOSGen Kaggle layout expected by
``scripts/evaluate_external.py`` (layout="pcosgen") into
``data_external/pcosgen/PCOSGen-train/{train,test}/{infected,healthy}/``.

Source layout (current):
  data_external/train/images/<file>.jpg      + class_label.xlsx
  data_external/test/images/<file>.jpg       + "class label.csv"

Target layout (expected by discover_pcosgen):
  data_external/pcosgen/PCOSGen-train/train/infected/*.jpg
  data_external/pcosgen/PCOSGen-train/train/healthy/*.jpg
  data_external/pcosgen/PCOSGen-train/test/infected/*.jpg
  data_external/pcosgen/PCOSGen-train/test/healthy/*.jpg

Files are HARD-LINKED (no extra disk space) where possible, falling
back to copy if hardlink fails. Skips images whose target already
exists. Run once.

Usage:  python scripts/prepare_pcosgen_layout.py
"""
import csv
import os
import shutil
import sys
import zipfile
import xml.etree.ElementTree as ET


SRC_ROOT = "data_external"
DST_ROOT = "data_external/pcosgen/PCOSGen-train"

SPLITS = ("train", "test")


def read_xlsx(path):
    """Minimal xlsx reader — first sheet, first row = header."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as z:
        shared = [
            (t.text or "")
            for t in ET.fromstring(z.read("xl/sharedStrings.xml"))
            .iter(f"{ns}t")
        ]
        sh = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    out = []
    for row in sh.iter(f"{ns}row"):
        cells = []
        for c in row:
            t = c.attrib.get("t", "n")
            v_node = c.find(f"{ns}v")
            v = v_node.text if v_node is not None else ""
            if t == "s":
                v = shared[int(v)]
            elif t == "inlineStr":
                tnode = c.find(
                    f"{ns}is/{ns}t"
                )
                v = tnode.text if tnode is not None else ""
            cells.append(v)
        out.append(cells)
    return out


def read_csv_labels(path):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def labels_for_train():
    rows = read_xlsx(os.path.join(SRC_ROOT, "train", "class_label.xlsx"))
    header = rows[0]
    last = header[-1]
    out = {}
    for r in rows[1:]:
        img, lbl = r[0], r[-1]
        out[img] = 1 if lbl.strip().lower().startswith("visible") else 0
    return out


def labels_for_test():
    rows = read_csv_labels(os.path.join(SRC_ROOT, "test", "class label.csv"))
    header = list(rows[0].keys())
    last = header[-1]
    out = {}
    for r in rows:
        img = r[header[0]]
        lbl = r[last]
        out[img] = 1 if lbl.strip().lower().startswith("visible") else 0
    return out


def link_one(src, dst):
    if os.path.exists(dst):
        return False
    try:
        os.link(src, dst)  # hardlink, no extra disk
        return True
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)
        return True


def main():
    train_lbls = labels_for_train()
    test_lbls = labels_for_test()
    print(f"train labels: {len(train_lbls)}  test labels: {len(test_lbls)}")

    counts = {sp: {"infected": 0, "healthy": 0} for sp in SPLITS}
    for split, lbls in (("train", train_lbls), ("test", test_lbls)):
        img_dir = os.path.join(SRC_ROOT, split, "images")
        for img, lbl in lbls.items():
            if not img or str(img).lower() == "nan":
                continue
            cls = "infected" if lbl == 1 else "healthy"
            dst_dir = os.path.join(DST_ROOT, split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            src = os.path.join(img_dir, img)
            dst = os.path.join(dst_dir, img)
            if not os.path.isfile(src):
                print(f"  missing: {src}")
                continue
            if link_one(src, dst):
                counts[split][cls] += 1

    for sp in SPLITS:
        c = counts[sp]
        print(f"  {sp}: linked {c['infected']} infected, {c['healthy']} healthy")
    print(f"Done. Materialised layout under {DST_ROOT}/")


if __name__ == "__main__":
    main()