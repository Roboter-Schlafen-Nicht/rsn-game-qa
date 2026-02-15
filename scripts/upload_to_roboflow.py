#!/usr/bin/env python
"""Upload captured frames to Roboflow for annotation.

Reads frames from a dataset directory (created by ``capture_dataset.py``)
and uploads them to a Roboflow project via the API.  Supports resuming
interrupted uploads by tracking which frames have already been uploaded.

Requirements::

    pip install roboflow

Usage::

    python scripts/upload_to_roboflow.py output/dataset_20260215_120000
    python scripts/upload_to_roboflow.py output/dataset_* --project breakout71
    python scripts/upload_to_roboflow.py output/dataset_20260215_120000 --batch 10 -v
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from scripts._smoke_utils import setup_logging

logger = logging.getLogger(__name__)

# Roboflow configuration (can be overridden via environment variables)
_DEFAULT_WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE", "")
_DEFAULT_PROJECT = os.environ.get("ROBOFLOW_PROJECT", "breakout71")
_DEFAULT_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")


def _load_upload_state(state_path: Path) -> set[str]:
    """Load the set of already-uploaded filenames from the state file.

    Parameters
    ----------
    state_path : Path
        Path to the JSON state file tracking uploads.

    Returns
    -------
    set[str]
        Filenames that have already been uploaded.
    """
    if state_path.exists():
        try:
            with open(state_path) as f:
                data = json.load(f)
            return set(data.get("uploaded", []))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Corrupted upload state file %s (%s) — starting fresh", state_path, exc
            )
            return set()
    return set()


def _save_upload_state(state_path: Path, uploaded: set[str]) -> None:
    """Persist the set of uploaded filenames.

    Parameters
    ----------
    state_path : Path
        Path to the JSON state file.
    uploaded : set[str]
        Filenames that have been uploaded.
    """
    with open(state_path, "w") as f:
        json.dump({"uploaded": sorted(uploaded)}, f, indent=2)


def upload_directory(
    dataset_dir: Path,
    api_key: str,
    workspace: str,
    project_name: str,
    batch_size: int = 50,
    split: str = "train",
) -> dict:
    """Upload all PNG frames from a dataset directory to Roboflow.

    Parameters
    ----------
    dataset_dir : Path
        Directory containing ``frame_*.png`` files and ``manifest.json``.
    api_key : str
        Roboflow API key.
    workspace : str
        Roboflow workspace name (slug).
    project_name : str
        Roboflow project name (slug).
    batch_size : int
        Log progress every N uploads.
    split : str
        Dataset split to assign (``"train"``, ``"valid"``, or ``"test"``).

    Returns
    -------
    dict
        Summary with keys ``uploaded``, ``skipped``, ``failed``, ``total``.
    """
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise RuntimeError(
            "roboflow package is required. Install with: pip install roboflow"
        ) from exc

    # Connect to Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    logger.info(
        "Connected to Roboflow: workspace=%s, project=%s",
        workspace,
        project_name,
    )

    # Find frames
    frames = sorted(dataset_dir.glob("frame_*.png"))
    if not frames:
        logger.warning("No frame_*.png files found in %s", dataset_dir)
        return {"uploaded": 0, "skipped": 0, "failed": 0, "total": 0}

    # Load upload state for resume support
    state_path = dataset_dir / ".upload_state.json"
    uploaded = _load_upload_state(state_path)
    logger.info("Found %d frames, %d already uploaded", len(frames), len(uploaded))

    stats = {"uploaded": 0, "skipped": 0, "failed": 0, "total": len(frames)}

    for i, frame_path in enumerate(frames):
        if frame_path.name in uploaded:
            stats["skipped"] += 1
            continue

        try:
            project.upload(
                image_path=str(frame_path),
                split=split,
            )
            uploaded.add(frame_path.name)
            stats["uploaded"] += 1
        except Exception as exc:
            logger.error("Failed to upload %s: %s", frame_path.name, exc)
            stats["failed"] += 1

        # Periodic progress + state save
        done = stats["uploaded"] + stats["skipped"]
        if done % batch_size == 0 or i == len(frames) - 1:
            _save_upload_state(state_path, uploaded)
            logger.info(
                "Progress: %d/%d (uploaded=%d, skipped=%d, failed=%d)",
                done,
                stats["total"],
                stats["uploaded"],
                stats["skipped"],
                stats["failed"],
            )

    # Final state save
    _save_upload_state(state_path, uploaded)
    return stats


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload captured frames to Roboflow for annotation."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Path to the dataset directory (from capture_dataset.py)",
    )
    parser.add_argument(
        "--api-key",
        default=_DEFAULT_API_KEY,
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)",
    )
    parser.add_argument(
        "--workspace",
        default=_DEFAULT_WORKSPACE,
        help="Roboflow workspace slug (or set ROBOFLOW_WORKSPACE env var)",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Roboflow project slug (or set ROBOFLOW_PROJECT env var, default: breakout71)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=50,
        help="Log progress every N uploads (default: %(default)s, min: 1)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "valid", "test"],
        default="train",
        help="Dataset split to assign (default: %(default)s)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    args = parser.parse_args()
    setup_logging(args.verbose)

    from dotenv import load_dotenv

    load_dotenv()

    # Re-read defaults after dotenv loads (CLI args override env vars)
    if not args.api_key:
        args.api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not args.workspace:
        args.workspace = os.environ.get("ROBOFLOW_WORKSPACE", "")
    if not args.project:
        args.project = os.environ.get("ROBOFLOW_PROJECT", "breakout71")

    if not args.api_key:
        logger.error(
            "No API key provided. Set ROBOFLOW_API_KEY env var or pass --api-key."
        )
        return 1

    if not args.workspace:
        logger.error(
            "No workspace provided. Set ROBOFLOW_WORKSPACE env var or pass --workspace."
        )
        return 1

    if args.batch < 1:
        logger.error("--batch must be >= 1")
        return 1

    if not args.dataset_dir.is_dir():
        logger.error("Dataset directory not found: %s", args.dataset_dir)
        return 1

    stats = upload_directory(
        dataset_dir=args.dataset_dir,
        api_key=args.api_key,
        workspace=args.workspace,
        project_name=args.project,
        batch_size=args.batch,
        split=args.split,
    )

    logger.info("--- Upload Summary ---")
    logger.info("Total frames  : %d", stats["total"])
    logger.info("Uploaded      : %d", stats["uploaded"])
    logger.info("Skipped       : %d", stats["skipped"])
    logger.info("Failed        : %d", stats["failed"])
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as exc:
        logger.critical("FAILED: %s", exc, exc_info=True)
        sys.exit(1)
