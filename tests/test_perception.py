"""Tests for the perception module (YoloDetector)."""

import pytest


class TestYoloDetector:
    """Placeholder tests for YoloDetector."""

    def test_detector_construction(self):
        """YoloDetector can be constructed with default args."""
        from src.perception.yolo_detector import YoloDetector

        detector = YoloDetector()
        assert detector.confidence_threshold == 0.5
        assert detector.device == "xpu"

    def test_default_classes(self):
        """Default class list should match Breakout 71 spec."""
        from src.perception.yolo_detector import YoloDetector

        detector = YoloDetector()
        assert "paddle" in detector.class_names
        assert "ball" in detector.class_names
        assert "brick" in detector.class_names

    def test_is_loaded_before_load(self):
        """is_loaded should return False before load() is called."""
        from src.perception.yolo_detector import YoloDetector

        detector = YoloDetector()
        assert not detector.is_loaded()

    def test_custom_classes(self):
        """Custom class names should override defaults."""
        from src.perception.yolo_detector import YoloDetector

        custom = ["player", "enemy", "coin"]
        detector = YoloDetector(classes=custom)
        assert detector.class_names == custom
