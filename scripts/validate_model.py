#!/usr/bin/env python
"""Validate a trained YOLO model against quality thresholds.

Loads trained weights, runs inference on a validation set, checks
mAP metrics against thresholds from the training config, and
optionally saves annotated sample images for visual inspection.

Usage::

    python scripts/validate_model.py --config breakout-71
    python scripts/validate_model.py --config breakout-71 --weights weights/breakout71/best.pt
    python scripts/validate_model.py --config breakout-71 --save-samples 10 -v
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from scripts._smoke_utils import Timer, setup_logging
from scripts.train_model import _PROJECT_ROOT, load_training_config, resolve_device

logger = logging.getLogger(__name__)


def validate_model(
    config: dict,
    weights_path: Path | None = None,
    save_samples: int = 0,
) -> dict:
    """Validate a trained YOLO model against config thresholds.

    Parameters
    ----------
    config : dict
        Training configuration (from :func:`load_training_config`).
    weights_path : Path, optional
        Path to the weights file. If ``None``, uses
        ``<output_dir>/best.pt`` from the config.
    save_samples : int
        Number of sample images to save with predictions drawn.
        Set to 0 to skip visual output.

    Returns
    -------
    dict
        Validation results with keys:

        - ``map50`` (float): mAP at IoU=0.5
        - ``map50_95`` (float): mAP at IoU=0.5:0.95
        - ``map50_pass`` (bool): whether mAP50 meets threshold
        - ``map50_95_pass`` (bool): whether mAP50-95 meets threshold
        - ``passed`` (bool): overall pass/fail
        - ``weights_path`` (str): path to validated weights
        - ``samples_dir`` (str or None): path to saved sample images

    Raises
    ------
    RuntimeError
        If ultralytics is not installed or weights not found.
    """
    # Resolve weights path (before heavy imports)
    if weights_path is None:
        output_dir = _PROJECT_ROOT / config.get("output_dir", "weights")
        weights_path = output_dir / "best.pt"

    if not weights_path.exists():
        raise RuntimeError(
            f"Weights file not found: {weights_path}\n"
            f"Train a model first with: python scripts/train_model.py --config <game>"
        )

    # Validate dataset_path (before importing ultralytics which pulls cv2)
    dataset_path = config.get("dataset_path")
    if not dataset_path:
        raise RuntimeError(
            "dataset_path is not set in the training config. "
            "Cannot run validation without a dataset."
        )
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise RuntimeError(f"Dataset path does not exist: {dataset_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for validation. "
            "Install with: pip install ultralytics"
        ) from exc

    # Load model
    logger.info("Loading model from %s", weights_path)
    model = YOLO(str(weights_path))

    # Resolve device
    device = resolve_device(config.get("device", "auto"))
    logger.info("Using device: %s", device)

    logger.info("Running validation on %s ...", dataset_path)
    with Timer("validation") as t:
        metrics = model.val(
            data=str(dataset_path),
            device=device,
            imgsz=config.get("imgsz", 640),
        )

    logger.info("Validation completed in %.1f seconds", t.elapsed)

    # Extract mAP metrics
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    # Check thresholds
    min_map50 = config.get("min_map50", 0.80)
    min_map50_95 = config.get("min_map50_95", 0.50)

    map50_pass = map50 >= min_map50
    map50_95_pass = map50_95 >= min_map50_95
    passed = map50_pass and map50_95_pass

    logger.info("--- Validation Metrics ---")
    logger.info(
        "mAP@0.5     : %.4f  (threshold: %.2f)  %s",
        map50,
        min_map50,
        "PASS" if map50_pass else "FAIL",
    )
    logger.info(
        "mAP@0.5:0.95: %.4f  (threshold: %.2f)  %s",
        map50_95,
        min_map50_95,
        "PASS" if map50_95_pass else "FAIL",
    )

    # Per-class metrics
    class_names = config.get("classes", [])
    if hasattr(metrics.box, "ap50") and len(class_names) > 0:
        logger.info("--- Per-Class AP@0.5 ---")
        ap50_per_class = metrics.box.ap50
        for i, cls_name in enumerate(class_names):
            if i < len(ap50_per_class):
                logger.info("  %-10s: %.4f", cls_name, ap50_per_class[i])

    # Save sample predictions
    samples_dir = None
    if save_samples > 0:
        samples_dir = _save_sample_predictions(model, config, device, save_samples)

    result = {
        "map50": map50,
        "map50_95": map50_95,
        "map50_pass": map50_pass,
        "map50_95_pass": map50_95_pass,
        "passed": passed,
        "weights_path": str(weights_path),
        "samples_dir": str(samples_dir) if samples_dir else None,
    }

    if passed:
        logger.info("VALIDATION PASSED — model meets quality thresholds")
    else:
        logger.warning("VALIDATION FAILED — model does not meet thresholds")

    return result


def _save_sample_predictions(
    model,
    config: dict,
    device: str,
    n_samples: int,
) -> Path | None:
    """Run prediction on sample images and save annotated results.

    Parameters
    ----------
    model : YOLO
        Loaded YOLO model.
    config : dict
        Training config.
    device : str
        Compute device.
    n_samples : int
        Number of images to process.

    Returns
    -------
    Path or None
        Directory where annotated images are saved, or None on failure.
    """
    import yaml

    dataset_path = Path(config["dataset_path"])

    # Parse data.yaml to find the val image directory
    with open(dataset_path) as f:
        data_cfg = yaml.safe_load(f)

    val_dir = data_cfg.get("val")
    if val_dir is None:
        logger.warning("No 'val' split in data.yaml — skipping samples")
        return None

    val_path = Path(val_dir)
    if not val_path.is_absolute():
        val_path = dataset_path.parent / val_path

    if not val_path.is_dir():
        logger.warning("Validation directory not found: %s", val_path)
        return None

    # Get sample images
    images = sorted(val_path.glob("*.png")) + sorted(val_path.glob("*.jpg"))
    images = images[:n_samples]

    if not images:
        logger.warning("No images found in %s", val_path)
        return None

    # Run prediction and save
    output_dir = _PROJECT_ROOT / "output" / "validation_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving %d sample predictions to %s", len(images), output_dir)

    for img_path in images:
        model.predict(
            source=str(img_path),
            device=device,
            imgsz=config.get("imgsz", 640),
            save=True,
            project=str(output_dir),
            name="samples",
            exist_ok=True,
        )

    return output_dir / "samples"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a trained YOLO model against quality thresholds."
    )
    parser.add_argument(
        "--config",
        default="breakout-71",
        help="Training config name (default: %(default)s)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to weights file (overrides config output_dir/best.pt)",
    )
    parser.add_argument(
        "--save-samples",
        type=int,
        default=0,
        help="Number of sample images to save with predictions (default: 0)",
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

    config = load_training_config(args.config)
    result = validate_model(
        config,
        weights_path=args.weights,
        save_samples=args.save_samples,
    )

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as exc:
        logger.critical("FAILED: %s", exc, exc_info=True)
        sys.exit(1)
