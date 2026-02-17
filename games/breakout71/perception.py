"""Breakout 71 YOLO perception constants.

Defines the class names used by the YOLO object detection model
trained for Breakout 71.  This is the single source of truth for
class ordering — the training config, the detector, and the
environment all reference this list.
"""

BREAKOUT71_CLASSES: list[str] = ["paddle", "ball", "brick", "powerup", "wall"]
"""YOLO class names for the Breakout 71 object detection model.

Order must match the training config ``configs/training/breakout-71.yaml``
(now ``games/breakout71/training.yaml``) and the YOLO ``data.yaml``
produced by ``prepare_dataset.py``.
"""
