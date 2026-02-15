#!/usr/bin/env python
"""Train a YOLO model using a game-specific training configuration.

Reads a training config from ``configs/training/<game>.yaml``, resolves
the compute device (XPU > CUDA > CPU), and runs Ultralytics YOLOv8
fine-tuning on the specified dataset.

The trained weights are saved to the ``output_dir`` specified in the
config (default: ``weights/<game>/``).

Usage::

    python scripts/train_model.py --config breakout-71
    python scripts/train_model.py --config breakout-71 --epochs 50
    python scripts/train_model.py --config breakout-71 --device cpu -v
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from scripts._smoke_utils import Timer, setup_logging

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TRAINING_CONFIG_DIR = _PROJECT_ROOT / "configs" / "training"


def load_training_config(config_name: str) -> dict:
    """Load a training configuration YAML file.

    Parameters
    ----------
    config_name : str
        Name of the config (without ``.yaml`` extension).
        Looked up in ``configs/training/<config_name>.yaml``.

    Returns
    -------
    dict
        Parsed training configuration.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    import yaml

    config_path = _TRAINING_CONFIG_DIR / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Training config not found: {config_path}\n"
            f"Available configs: {[p.stem for p in _TRAINING_CONFIG_DIR.glob('*.yaml')]}"
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.debug("Loaded training config from %s", config_path)
    return config


def resolve_device(requested: str = "auto") -> str:
    """Resolve the best available compute device.

    Parameters
    ----------
    requested : str
        ``"auto"`` (try xpu > cuda > cpu), ``"xpu"``, ``"cuda"``,
        or ``"cpu"``.

    Returns
    -------
    str
        Device string suitable for Ultralytics ``model.train(device=...)``.
    """
    if requested != "auto":
        return requested

    import torch

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _patch_ultralytics_xpu() -> None:
    """Monkey-patch ultralytics to support Intel XPU.

    Ultralytics 8.4.x has partial XPU support in the trainer and
    validator but two things block XPU usage:

    1. ``select_device()`` rejects ``"xpu"`` as an invalid CUDA device.
    2. ``BaseTrainer`` hard-codes ``GradScaler("cuda", ...)`` which
       crashes on XPU-only torch builds.

    This function wraps the original functions so that XPU is handled
    correctly without modifying the ultralytics package on disk.
    """
    import torch
    from ultralytics.utils import torch_utils

    # --- Patch 1: select_device ---
    _original = torch_utils.select_device

    def _xpu_aware_select_device(device="", newline=False, verbose=True):
        dev = str(device).strip().lower()
        if dev == "xpu":
            if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
                logger.warning("XPU requested but not available, falling back to CPU")
                return _original("cpu", newline=newline, verbose=verbose)
            s = (
                f"Ultralytics {torch_utils.__version__} "
                f"Python-{torch_utils.PYTHON_VERSION} "
                f"torch-{torch_utils.TORCH_VERSION} "
                f"XPU ({torch.xpu.get_device_name(0)})\n"
            )
            if verbose:
                torch_utils.LOGGER.info(s if newline else s.rstrip())
            return torch.device("xpu")
        return _original(device, newline=newline, verbose=verbose)

    torch_utils.select_device = _xpu_aware_select_device

    # Also patch modules that import select_device directly into their
    # namespace (they hold a stale reference to the original function).
    for mod_name in [
        "ultralytics.engine.trainer",
        "ultralytics.engine.validator",
        "ultralytics.engine.predictor",
    ]:
        try:
            mod = __import__(mod_name, fromlist=["select_device"])
            if hasattr(mod, "select_device"):
                mod.select_device = _xpu_aware_select_device
        except ImportError:
            pass

    # --- Patch 2: GradScaler("cuda") → GradScaler("xpu") ---
    # BaseTrainer._setup_train hard-codes GradScaler("cuda", enabled=...)
    # which crashes on XPU-only torch even when amp=False.
    from ultralytics.engine.trainer import BaseTrainer

    _original_setup_train = BaseTrainer._setup_train

    def _patched_setup_train(self):
        _original_setup_train(self)
        # Replace the CUDA GradScaler with an XPU one after setup
        if hasattr(self, "device") and self.device.type == "xpu":
            self.scaler = torch.amp.GradScaler("xpu", enabled=self.amp)

    BaseTrainer._setup_train = _patched_setup_train

    # --- Patch 3: _get_memory() doesn't know about XPU ---
    # Falls through to torch.cuda.memory_reserved() which returns 0.
    _original_get_memory = BaseTrainer._get_memory

    def _patched_get_memory(self, fraction=False):
        if hasattr(self, "device") and self.device.type == "xpu":
            memory = torch.xpu.memory_reserved()
            if fraction:
                total = torch.xpu.get_device_properties(self.device).total_memory
                return (memory / total) if total > 0 else 0
            return memory / 2**30
        return _original_get_memory(self, fraction=fraction)

    BaseTrainer._get_memory = _patched_get_memory

    logger.debug("Patched ultralytics select_device + GradScaler + _get_memory for XPU")


def train(config: dict, overrides: dict | None = None) -> Path:
    """Run YOLO training with the given configuration.

    Parameters
    ----------
    config : dict
        Training configuration (from :func:`load_training_config`).
    overrides : dict, optional
        Key-value overrides for config values (e.g. from CLI args).

    Returns
    -------
    Path
        Path to the best weights file (``best.pt``), or the training
        output directory if ``best.pt`` was not found.

    Raises
    ------
    RuntimeError
        If ultralytics is not installed or dataset_path is not set.
    ValueError
        If the dataset path does not exist.
    """
    # Merge overrides (before heavy imports so validation is fast)
    cfg = {**config}
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})

    # Validate dataset path (before importing ultralytics which pulls cv2)
    dataset_path = cfg.get("dataset_path")
    if not dataset_path:
        raise RuntimeError(
            "dataset_path is not set in the training config. "
            "Export your annotated dataset from Roboflow in YOLOv8 format, "
            "then set dataset_path in configs/training/<game>.yaml to the "
            "path of the exported data.yaml file."
        )
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for training. "
            "Install with: pip install ultralytics"
        ) from exc

    # Resolve device
    device = resolve_device(cfg.get("device", "auto"))
    logger.info("Using device: %s", device)

    # Patch ultralytics select_device to support Intel XPU
    if device == "xpu":
        _patch_ultralytics_xpu()

    # Load base model
    base_model = cfg.get("base_model", "yolov8n.pt")
    logger.info("Loading base model: %s", base_model)
    model = YOLO(base_model)

    # Output directory
    output_dir = _PROJECT_ROOT / cfg.get("output_dir", "weights")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    epochs = cfg.get("epochs", 100)
    imgsz = cfg.get("imgsz", 640)
    batch = cfg.get("batch", 16)
    amp = cfg.get("amp", False)

    logger.info(
        "Starting training: epochs=%d, imgsz=%d, batch=%d, amp=%s",
        epochs,
        imgsz,
        batch,
        amp,
    )

    with Timer("training") as t:
        model.train(
            data=str(dataset_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            amp=amp,
            project=str(output_dir),
            name="train",
            exist_ok=True,
        )

    logger.info("Training completed in %.1f seconds", t.elapsed)

    # Copy best weights to a well-known location
    train_dir = output_dir / "train"
    best_weights = train_dir / "weights" / "best.pt"

    if best_weights.exists():
        final_weights = output_dir / "best.pt"
        shutil.copy2(best_weights, final_weights)
        logger.info("Best weights saved to: %s", final_weights)
        return final_weights

    logger.warning("best.pt not found at expected location: %s", best_weights)
    logger.warning("Returning train directory instead: %s", train_dir)
    return train_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a YOLO model using a game-specific config."
    )
    parser.add_argument(
        "--config",
        default="breakout-71",
        help="Training config name (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device (auto, xpu, cuda, cpu)",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Override dataset path (data.yaml from Roboflow export)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Override batch size",
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

    overrides = {
        "epochs": args.epochs,
        "device": args.device,
        "dataset_path": args.dataset_path,
        "batch": args.batch,
    }

    weights_path = train(config, overrides)
    logger.info("--- Training Complete ---")
    logger.info("Weights: %s", weights_path)
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
