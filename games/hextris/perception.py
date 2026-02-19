"""Hextris perception constants.

Hextris uses CNN-only observations (no YOLO model).  This module
provides an empty class list so the plugin conforms to the platform
convention.
"""

HEXTRIS_CLASSES: list[str] = []
"""YOLO class names for Hextris.

Empty — Hextris uses pixel-based CNN observations, not YOLO
object detection.  The ``game_classes()`` method returns this list
to satisfy the platform's ``YoloDetector`` configuration, but YOLO
inference results will be empty dicts.
"""
