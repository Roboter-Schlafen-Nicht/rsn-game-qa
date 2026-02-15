#!/usr/bin/env python
"""Prepare a captured dataset for YOLO training.

Restructures a flat dataset (frames at root, labels in ``labels/``) into
the standard YOLO training directory layout with train/val split and a
``data.yaml`` file.

Usage::

    python scripts/prepare_dataset.py --source output/dataset_20260215_175739
    python scripts/prepare_dataset.py --source output/dataset_20260215_175739 --val-ratio 0.15
    python scripts/prepare_dataset.py --source output/dataset_20260215_175739 --exclude-game-over -v
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from scripts._smoke_utils import Timer, setup_logging

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def prepare_dataset(
    source_dir: Path,
    output_dir: Path | None = None,
    val_ratio: float = 0.2,
    exclude_game_over: bool = False,
    seed: int = 42,
) -> Path:
    """Restructure a flat dataset into YOLO training format.

    Parameters
    ----------
    source_dir : Path
        Directory containing frame images at root level and YOLO label
        ``.txt`` files in a ``labels/`` subdirectory.
    output_dir : Path, optional
        Where to create the YOLO dataset.  Defaults to
        ``<source_dir>_yolo/``.
    val_ratio : float
        Fraction of images to use for validation (default: 0.2).
    exclude_game_over : bool
        If ``True`` and a ``manifest.json`` exists, skip frames that
        were captured during game-over state.
    seed : int
        Random seed for reproducible train/val split.

    Returns
    -------
    Path
        Path to the generated ``data.yaml`` file.

    Raises
    ------
    FileNotFoundError
        If the source directory or labels subdirectory does not exist.
    ValueError
        If no valid image/label pairs are found.
    """
    source_dir = Path(source_dir).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    labels_dir = source_dir / "labels"
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    # Read classes from labels/classes.txt
    classes_file = labels_dir / "classes.txt"
    if not classes_file.exists():
        raise FileNotFoundError(f"classes.txt not found: {classes_file}")

    class_names = [
        line.strip() for line in classes_file.read_text().splitlines() if line.strip()
    ]
    logger.info("Classes (%d): %s", len(class_names), class_names)

    # Determine which frames to exclude
    excluded_frames: set[str] = set()
    if exclude_game_over:
        manifest_path = source_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            for frame_info in manifest.get("frames", []):
                if frame_info.get("game_state") == "game_over":
                    excluded_frames.add(frame_info["filename"])
            logger.info(
                "Excluding %d game-over frames from dataset",
                len(excluded_frames),
            )
        else:
            logger.warning("No manifest.json found — cannot filter game-over frames")

    # Collect valid image/label pairs
    pairs: list[tuple[Path, Path]] = []
    for img_path in sorted(source_dir.glob("frame_*.png")):
        if img_path.name in excluded_frames:
            continue
        label_path = labels_dir / img_path.with_suffix(".txt").name
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            logger.warning("No label file for %s — skipping", img_path.name)

    if not pairs:
        raise ValueError(f"No valid image/label pairs found in {source_dir}")

    logger.info("Found %d valid image/label pairs", len(pairs))

    # Validate val_ratio
    if not (0 < val_ratio < 1):
        raise ValueError(
            f"val_ratio must be between 0 and 1 (exclusive), got {val_ratio}"
        )

    # Shuffle and split
    random.seed(seed)
    indices = list(range(len(pairs)))
    random.shuffle(indices)

    n_val = max(1, int(len(pairs) * val_ratio))
    # Ensure train set is non-empty
    if n_val >= len(pairs):
        n_val = len(pairs) - 1
    val_indices = set(indices[:n_val])
    train_indices = set(indices[n_val:])

    logger.info(
        "Split: %d train, %d val (ratio=%.2f)",
        len(train_indices),
        len(val_indices),
        val_ratio,
    )

    # Create output directory structure
    if output_dir is None:
        output_dir = source_dir.parent / f"{source_dir.name}_yolo"
    output_dir = Path(output_dir).resolve()

    train_images_dir = output_dir / "train" / "images"
    train_labels_dir = output_dir / "train" / "labels"
    val_images_dir = output_dir / "val" / "images"
    val_labels_dir = output_dir / "val" / "labels"

    for d in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Copy files
    for i, (img_path, label_path) in enumerate(pairs):
        if i in val_indices:
            shutil.copy2(img_path, val_images_dir / img_path.name)
            shutil.copy2(label_path, val_labels_dir / label_path.name)
        else:
            shutil.copy2(img_path, train_images_dir / img_path.name)
            shutil.copy2(label_path, train_labels_dir / label_path.name)

    # Write data.yaml
    data_yaml = {
        "path": str(output_dir),
        "train": "train/images",
        "val": "val/images",
        "nc": len(class_names),
        "names": class_names,
    }

    data_yaml_path = output_dir / "data.yaml"
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    logger.info("data.yaml written to: %s", data_yaml_path)
    logger.info("Train images: %d", len(list(train_images_dir.glob("*.png"))))
    logger.info("Val images:   %d", len(list(val_images_dir.glob("*.png"))))

    return data_yaml_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a captured dataset for YOLO training."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the source dataset directory",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: <source>_yolo/)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of images for validation (default: 0.2)",
    )
    parser.add_argument(
        "--exclude-game-over",
        action="store_true",
        help="Skip game-over frames (requires manifest.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    args = parser.parse_args()
    setup_logging(args.verbose)

    source = Path(args.source)
    if not source.is_absolute():
        source = _PROJECT_ROOT / source

    output = Path(args.output) if args.output else None
    if output and not output.is_absolute():
        output = _PROJECT_ROOT / output

    with Timer("dataset preparation") as t:
        data_yaml_path = prepare_dataset(
            source_dir=source,
            output_dir=output,
            val_ratio=args.val_ratio,
            exclude_game_over=args.exclude_game_over,
            seed=args.seed,
        )

    logger.info("Dataset prepared in %.1f seconds", t.elapsed)
    logger.info("data.yaml: %s", data_yaml_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as exc:
        logger.critical("FAILED: %s", exc, exc_info=True)
        sys.exit(1)
