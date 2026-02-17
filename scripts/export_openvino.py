#!/usr/bin/env python
"""Export a trained YOLO model to OpenVINO IR format.

Converts a ``.pt`` PyTorch weights file into an OpenVINO Intermediate
Representation (IR) model directory that can be loaded directly by
``ultralytics.YOLO`` for accelerated inference on Intel hardware.

The exported model is saved alongside the source ``.pt`` file::

    weights/breakout71/best.pt
    weights/breakout71/best_openvino_model/   <-- created by this script

Usage::

    python scripts/export_openvino.py
    python scripts/export_openvino.py --game breakout71
    python scripts/export_openvino.py --weights weights/breakout71/best.pt --half
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._smoke_utils import Timer, setup_logging

logger = logging.getLogger(__name__)


def export_openvino(
    weights_path: Path,
    *,
    imgsz: int = 640,
    half: bool = False,
) -> Path:
    """Export a YOLO ``.pt`` model to OpenVINO format.

    Parameters
    ----------
    weights_path : Path
        Path to the trained ``.pt`` weights file.
    imgsz : int
        Inference image size (square).  Default 640.
    half : bool
        Export with FP16 precision.  Default False (FP32).

    Returns
    -------
    Path
        Path to the exported OpenVINO model directory.

    Raises
    ------
    FileNotFoundError
        If the weights file does not exist.
    RuntimeError
        If ultralytics is not installed or export fails.
    """
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for export. Install with: pip install ultralytics"
        ) from exc

    logger.info("Loading model from %s", weights_path)
    model = YOLO(str(weights_path))

    logger.info(
        "Exporting to OpenVINO (imgsz=%d, half=%s) ...",
        imgsz,
        half,
    )
    with Timer("openvino_export") as t:
        export_path = model.export(format="openvino", imgsz=imgsz, half=half)

    export_dir = Path(export_path)
    logger.info(
        "Export completed in %.1f seconds -> %s",
        t.elapsed,
        export_dir,
    )

    # Verify the exported model directory exists
    if not export_dir.exists():
        raise RuntimeError(
            f"Export reported success but output not found: {export_dir}"
        )

    return export_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a YOLO model to OpenVINO format for accelerated inference."
    )
    parser.add_argument(
        "--game",
        type=str,
        default="breakout71",
        help="Game plugin name (directory under games/). Default: breakout71",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to .pt weights file (default: from game plugin)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: %(default)s)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export with FP16 precision",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    args = parser.parse_args()
    setup_logging(args.verbose)

    from games import load_game_plugin

    plugin = load_game_plugin(args.game)
    weights = args.weights or Path(plugin.default_weights)

    export_dir = export_openvino(
        weights,
        imgsz=args.imgsz,
        half=args.half,
    )

    logger.info("OpenVINO model ready at: %s", export_dir)
    logger.info(
        "YoloDetector will auto-select this model when device is 'cpu' or Intel."
    )
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
