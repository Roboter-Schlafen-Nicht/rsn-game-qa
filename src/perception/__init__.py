"""Perception module — YOLO-based object detection and feature extraction."""

from .breakout_capture import detect_objects, grab_frame
from .yolo_detector import YoloDetector, resolve_device

__all__ = [
    "YoloDetector",
    "detect_objects",
    "grab_frame",
    "resolve_device",
]
