#!/usr/bin/env python3
"""
Organize WLASL videos into folders named by gloss (sign-word) using WLASL_v0.3.json.

Two ways to set paths:
  1) Edit the CONFIG section below (quickest).
  2) Use command-line flags to override the CONFIG values.

Examples:
  # Using CONFIG only (after editing paths)
  python organize_wlasl.py

  # Override via CLI and use train/val/test subfolders
  python organize_wlasl.py \
    --json "/data/WLASL/WLASL_v0.3.json" \
    --videos_dir "/data/WLASL/videos_raw" \
    --out_dir "/data/WLASL/processed" \
    --mode copy \
    --use_splits

  # Dry run to preview actions (no file changes)
  python organize_wlasl.py --dry_run
"""

from __future__ import annotations
import argparse
import json
import shutil
import sys
import re
from pathlib import Path

# =========================
# ========== CONFIG =======
# =========================
# Edit these three paths for the simplest usage, then run:  python organize_wlasl.py
CONFIG_JSON_PATH   = Path("/home/kaleab/Desktop/asl-words-dataset/wlasl-complete/WLASL_v0.3.json")
CONFIG_VIDEOS_DIR  = Path("/home/kaleab/Desktop/asl-words-dataset/wlasl-complete/videos")   # directory with files like 00001.mp4, 00295.mp4, ...
CONFIG_OUT_DIR     = Path("/home/kaleab/Desktop/WLASL_dataset_processed")    # where organized folders will be created

# Default behaviour (can be overridden by CLI flags)
CONFIG_MODE        = "move"       # one of: "copy", "move", "symlink"
CONFIG_USE_SPLITS  = False        # True => out_dir/train|val|test/<GLOSS>/
CONFIG_EXTS        = [".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"]
CONFIG_DRY_RUN     = False        # True => print actions, do not modify files


def sanitize_gloss(gloss: str) -> str:
    """
    Produce a filesystem-safe, consistent folder name for a gloss.
    - Uppercase
    - Spaces -> underscores
    - Remove characters unsafe for common filesystems
    """
    g = gloss.strip().upper().replace(" ", "_")
    g = re.sub(r"[^\w\-\.+]", "", g)     # keep letters/digits/_-.+
    g = re.sub(r"_+", "_", g).strip("_")
    return g or "UNKNOWN"


def resolve_video_path(videos_dir: Path, video_id: str, exts: list[str]) -> Path | None:
    """
    Find the actual file for a given numeric video_id by testing common extensions.
    Handles IDs with or without zero-padding.
    """
    # try zero-padded 5-digit id first (e.g., "00295"), then raw
    candidates = []
    vid5 = str(video_id).zfill(5)
    for base in (vid5, str(video_id)):
        for ext in exts:
            candidates.append(videos_dir / f"{base}{ext}")
    for p in candidates:
        if p.exists():
            return p
    return None


def load_mapping(json_path: Path) -> dict[str, dict]:
    """
    Load WLASL_v0.3.json and return a mapping:
        video_id (5-digit string) -> {"gloss": str, "split": str}
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    mapping = {}
    for entry in data:
        gloss = entry.get("gloss", "").strip()
        for inst in entry.get("instances", []):
            vid = str(inst.get("video_id", "")).strip()
            if not vid:
                continue
            split = str(inst.get("split", "unknown")).lower()
            mapping[vid.zfill(5)] = {"gloss": gloss, "split": split}
    return mapping


def place_file(src: Path, dst: Path, mode: str, dry_run: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"[DRY] {mode.upper():7}  {src}  ->  {dst}")
        return

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(src, dst)
    elif mode == "symlink":
        try:
            # Try relative symlink for portability
            rel_target = src.resolve()
            dst.symlink_to(rel_target)
        except OSError:
            # Fallback to copy if symlinks are not permitted
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def organize(
    json_path: Path,
    videos_dir: Path,
    out_dir: Path,
    mode: str = "copy",
    use_splits: bool = False,
    exts: list[str] = None,
    dry_run: bool = False,
) -> None:
    exts = exts or [".mp4"]
    assert json_path.exists(), f"JSON not found: {json_path}"
    assert videos_dir.exists(), f"Videos directory not found: {videos_dir}"
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = load_mapping(json_path)

    total, placed, missing, collisions = 0, 0, 0, 0

    for vid5, meta in mapping.items():
        total += 1
        gloss = meta["gloss"]
        split = meta["split"]  # 'train', 'val', 'test', or 'unknown'

        src = resolve_video_path(videos_dir, vid5, exts)
        if src is None:
            missing += 1
            print(f"[WARN] Missing file for video_id={vid5} (tried: {', '.join(exts)})")
            continue

        gloss_dir = sanitize_gloss(gloss)
        if use_splits:
            target_base = out_dir / split / gloss_dir
        else:
            target_base = out_dir / gloss_dir

        dst = target_base / src.name
        if dst.exists() and not dst.samefile(src):
            # Avoid overwriting if a file with the same name already exists
            collisions += 1
            stem, ext = dst.stem, dst.suffix
            k = 1
            while True:
                candidate = target_base / f"{stem}__dup{k}{ext}"
                if not candidate.exists():
                    dst = candidate
                    break
                k += 1

        place_file(src, dst, mode, dry_run)
        placed += 1

    print("\n==== Summary ====")
    print(f"JSON:        {json_path}")
    print(f"Videos dir:  {videos_dir}")
    print(f"Output dir:  {out_dir}")
    print(f"Mode:        {mode} | Use splits: {use_splits} | Dry run: {dry_run}")
    print(f"Extensions:  {exts}")
    print(f"Annotated:   {total}")
    print(f"Placed:      {placed}")
    print(f"Missing:     {missing}")
    print(f"Collisions:  {collisions}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Organize WLASL videos into gloss-named folders.")
    p.add_argument("--json", type=Path, help="Path to WLASL_v0.3.json")
    p.add_argument("--videos_dir", type=Path, help="Directory with raw videos named by numeric IDs.")
    p.add_argument("--out_dir", type=Path, help="Output directory for organized dataset.")
    p.add_argument("--mode", choices=["copy", "move", "symlink"], help="Placement mode (default: CONFIG_MODE).")
    p.add_argument("--use_splits", action="store_true", help="Create train/val/test subfolders.")
    p.add_argument("--exts", nargs="*", help="Extensions to try, e.g. .mp4 .mov (default: CONFIG_EXTS).")
    p.add_argument("--dry_run", action="store_true", help="Preview actions without changing files.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Use CLI args if provided, else fall back to CONFIG.
    json_path  = args.json       or CONFIG_JSON_PATH
    videos_dir = args.videos_dir or CONFIG_VIDEOS_DIR
    out_dir    = args.out_dir    or CONFIG_OUT_DIR
    mode       = args.mode       or CONFIG_MODE
    use_splits = args.use_splits or CONFIG_USE_SPLITS
    exts       = args.exts       or CONFIG_EXTS
    dry_run    = args.dry_run or CONFIG_DRY_RUN

    try:
        organize(
            json_path=json_path,
            videos_dir=videos_dir,
            out_dir=out_dir,
            mode=mode,
            use_splits=use_splits,
            exts=exts,
            dry_run=dry_run,
        )
    except AssertionError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
