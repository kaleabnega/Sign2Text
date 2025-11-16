#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rename KArSL class folders from numeric IDs (e.g., 0001, 0502) to their label names
based on a labels.txt file (one label per line; line 1 -> id 0001, etc.).

Expected structure (before):
dataset_root/
  train/
    0001/
      sample_0001/
        frame_0001.jpg
        ...
      sample_0002/
        ...
    0002/
      ...
  val/
    0001/
    0002/
    ...

After renaming (default, no ID prefix):
dataset_root/
  train/
    <LABEL_OF_0001>/
      sample_0001/
      ...
    <LABEL_OF_0002>/
      ...
  val/
    <LABEL_OF_0001>/
    <LABEL_OF_0002>/
    ...

Features:
- Keeps Arabic/unicode labels.
- Prevents illegal path characters and trims whitespace.
- Optionally prefixes folders with the numeric ID (e.g., "0001_<LABEL>") to avoid ambiguity.
- If the destination folder already exists, contents are merged.
- Writes a CSV log of operations.

Usage (edit CONFIG or pass CLI args):
    python rename_karsl_folders.py \
        --dataset_root "/path/to/KArSL" \
        --labels_file "/path/to/labels.txt" \
        --splits train val \
        --id_width 4 \
        --include_id_prefix false \
        --dry_run

"""

from __future__ import annotations
import argparse
import csv
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


# =========================
# ========== CONFIG =======
# =========================
# Option A: Edit here and run the script with no CLI flags.
CONFIG_DATASET_ROOT = Path("/home/kaleab/Desktop/ArSL-dataset")        # contains 'train' and 'val'
CONFIG_LABELS_FILE  = Path("/home/kaleab/Desktop/ArSL-dataset/arsl-label.txt")   # one label per line; line 1 -> 0001
CONFIG_SPLITS       = ["train", "val"]              # which subfolders to process
CONFIG_ID_WIDTH     = 4                             # zero-padding width (e.g., 4 => 0001)
CONFIG_INCLUDE_ID_PREFIX = False                    # True => "0001_<LABEL>"; False => "<LABEL>"
CONFIG_DRY_RUN      = False                         # True => preview only

# =========================
# ===== Helper functions ==
# =========================

_ILLEGAL_CHARS = r'[\/:*?"<>|\x00-\x1F]'  # avoid slashes, control chars, Windows-illegal chars

def sanitize_folder_name(name: str) -> str:
    """
    Make a filesystem-safe folder name while preserving Unicode (e.g., Arabic).
    - Strip leading/trailing whitespace
    - Replace illegal characters with a single space
    - Collapse repeated spaces/underscores
    """
    s = name.strip()
    s = re.sub(_ILLEGAL_CHARS, " ", s)
    s = re.sub(r"\s+", " ", s)          # collapse whitespace runs
    s = s.strip(" ._")                   # trim trailing dots/underscores/spaces
    return s if s else "UNKNOWN"

def load_id_to_label(labels_file: Path, id_width: int) -> Dict[str, str]:
    """
    Build a mapping from zero-padded ID (e.g., '0001') to label string.
    """
    assert labels_file.exists(), f"labels file not found: {labels_file}"
    id2label: Dict[str, str] = {}
    with labels_file.open("r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    for idx, raw in enumerate(lines, start=1):
        label = sanitize_folder_name(raw)
        key = str(idx).zfill(id_width)
        id2label[key] = label
    return id2label

def is_numeric_folder(name: str) -> bool:
    return bool(re.fullmatch(r"\d+", name))

def merge_or_rename(src: Path, dst: Path, dry_run: bool) -> Tuple[int, int]:
    """
    If dst does not exist: rename src -> dst.
    If dst exists and is a directory: merge contents of src into dst, then remove src.
    Returns (moved_files, merged_dirs).
    """
    moved_files = 0
    merged_dirs = 0

    if dry_run:
        action = "RENAME" if not dst.exists() else "MERGE"
        print(f"[DRY] {action}: {src} -> {dst}")
        return (0, 0)

    if not dst.exists():
        src.rename(dst)
        return (0, 0)

    # Merge: move children of src into dst
    for item in src.iterdir():
        target = dst / item.name
        if target.exists():
            # Collision: add suffix to avoid overwrite
            stem, suffix = target.stem, target.suffix
            k = 1
            while True:
                candidate = dst / f"{stem}__dup{k}{suffix}"
                if not candidate.exists():
                    target = candidate
                    break
                k += 1
        if item.is_dir():
            shutil.move(str(item), str(target))
            merged_dirs += 1
        else:
            shutil.move(str(item), str(target))
            moved_files += 1

    # Remove now-empty source directory
    try:
        src.rmdir()
    except OSError:
        # If not empty (hidden files, etc.), remove tree
        shutil.rmtree(src, ignore_errors=True)

    return (moved_files, merged_dirs)

# =========================
# ===== Main procedure =====
# =========================

def rename_class_folders(
    dataset_root: Path,
    labels_file: Path,
    splits: List[str],
    id_width: int,
    include_id_prefix: bool,
    dry_run: bool,
) -> None:
    id2label = load_id_to_label(labels_file, id_width)

    log_rows = []
    summary = {
        "total_class_dirs": 0,
        "renamed_dirs": 0,
        "merged_dirs": 0,
        "skipped_dirs": 0,
        "missing_ids": 0,
        "moved_files_during_merge": 0,
    }

    for split in splits:
        split_dir = dataset_root / split
        if not split_dir.exists():
            print(f"[WARN] Split not found: {split_dir} (skipping)")
            continue

        for child in sorted(split_dir.iterdir()):
            if not child.is_dir():
                continue
            if not is_numeric_folder(child.name):
                # Already renamed or not a class folder
                summary["skipped_dirs"] += 1
                continue

            summary["total_class_dirs"] += 1
            class_id = child.name
            key = class_id.zfill(id_width)
            label = id2label.get(key)
            if label is None:
                print(f"[WARN] No label for ID={class_id} (labels.txt may be misaligned)")
                summary["missing_ids"] += 1
                continue

            new_name = f"{key}_{label}" if include_id_prefix else label
            new_name = sanitize_folder_name(new_name)
            dst = split_dir / new_name

            moved_files, merged_dirs = merge_or_rename(child, dst, dry_run=dry_run)

            summary["renamed_dirs"] += 1
            summary["merged_dirs"] += merged_dirs
            summary["moved_files_during_merge"] += moved_files

            log_rows.append({
                "split": split,
                "old_dir": str(child),
                "new_dir": str(dst),
                "class_id": key,
                "label": label,
                "merged": int(dst.exists()),
            })

    # Write CSV log next to the script run (inside dataset_root)
    log_path = dataset_root / "rename_log.csv"
    if dry_run:
        print(f"[DRY] Would write log to: {log_path}")
    else:
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()) if log_rows else
                                    ["split","old_dir","new_dir","class_id","label","merged"])
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"[INFO] Log written to: {log_path}")

    # Summary
    print("\n===== Summary =====")
    print(f"Dataset root:  {dataset_root}")
    print(f"Labels file:   {labels_file}")
    print(f"Splits:        {', '.join(splits)}")
    print(f"ID width:      {id_width}")
    print(f"ID prefix:     {include_id_prefix}")
    print(f"Dry run:       {dry_run}")
    print("---------------------------")
    for k, v in summary.items():
        print(f"{k.replace('_',' ').title():25s}: {v}")
    print("---------------------------")
    print("Done.")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rename KArSL class folders from numeric IDs to label names.")
    p.add_argument("--dataset_root", type=Path, help="Root directory containing 'train' and/or 'val'.")
    p.add_argument("--labels_file", type=Path, help="Path to labels.txt (one label per line).")
    p.add_argument("--splits", nargs="*", default=None, help="Which splits to process (default: train val).")
    p.add_argument("--id_width", type=int, default=None, help="Zero-padding width (e.g., 4 => 0001).")
    p.add_argument("--include_id_prefix", action="store_true", help="Prefix folder names with the numeric ID.")
    p.add_argument("--dry_run", action="store_true", help="Preview changes without modifying the filesystem.")
    return p.parse_args()

def main():
    args = parse_args()

    dataset_root = args.dataset_root or CONFIG_DATASET_ROOT
    labels_file  = args.labels_file  or CONFIG_LABELS_FILE
    splits       = args.splits       or CONFIG_SPLITS
    id_width     = args.id_width     or CONFIG_ID_WIDTH
    include_id_prefix = args.include_id_prefix or CONFIG_INCLUDE_ID_PREFIX
    dry_run      = args.dry_run or CONFIG_DRY_RUN

    rename_class_folders(
        dataset_root=dataset_root,
        labels_file=labels_file,
        splits=splits,
        id_width=id_width,
        include_id_prefix=include_id_prefix,
        dry_run=dry_run,
    )

if __name__ == "__main__":
    main()
