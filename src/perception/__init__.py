"""Perception module — YOLO-based object detection and feature extraction."""

from .breakout_capture import detect_objects, grab_frame
from .yolo_detector import YoloDetector, _find_openvino_model, resolve_device

__all__ = [
    "YoloDetector",
    "_find_openvino_model",
    "detect_objects",
    "grab_frame",
    "resolve_device",
]
