"""shapez.io perception constants.

shapez.io uses CNN-only observations (no YOLO model).  This module
provides an empty class list so the plugin conforms to the platform
convention.
"""

SHAPEZ_CLASSES: list[str] = []
"""YOLO class names for shapez.io.

Empty -- shapez.io uses pixel-based CNN observations, not YOLO
object detection.  The ``game_classes()`` method returns this list
to satisfy the platform's ``YoloDetector`` configuration, but YOLO
inference results will be empty dicts.
"""
