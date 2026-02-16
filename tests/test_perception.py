"""Tests for the perception module (YoloDetector + breakout_capture).

Tests cover:

- YoloDetector construction and ultralytics availability guard
- Custom / default class names
- is_loaded before and after load
- load() with valid weights, missing weights (FileNotFoundError)
- load() XPU fallback to CPU
- load() reading class names from model metadata
- detect() with mocked YOLO results — single and multiple detections
- detect() with empty results and no-boxes results
- detect() before load raises RuntimeError
- detect() with unknown class ID
- detect_to_game_state() — full state with paddle, ball, bricks, powerups
- detect_to_game_state() — missing paddle/ball returns None
- detect_to_game_state() — multiple paddle/ball picks highest confidence
- breakout_capture.grab_frame() — delegates to WindowCapture
- breakout_capture.grab_frame() — raises on invisible window
- breakout_capture.detect_objects() — delegates to YoloDetector
- breakout_capture.detect_objects() — auto-infers frame dimensions
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from src.perception.yolo_detector import (
    YoloDetector,
    _ULTRALYTICS_AVAILABLE,
    _find_openvino_model,
    _resolve_openvino_device,
)

pytestmark = pytest.mark.skipif(
    not _ULTRALYTICS_AVAILABLE,
    reason="ultralytics is not installed",
)


# ── Helpers ─────────────────────────────────────────────────────────


def _make_mock_box(cls_id: int, conf: float, xyxy: tuple[int, int, int, int]):
    """Create a single mock detection box mimicking ultralytics Results."""
    import torch

    box = mock.MagicMock()
    box.xyxy = torch.tensor([list(xyxy)], dtype=torch.float32)
    box.cls = torch.tensor([cls_id], dtype=torch.float32)
    box.conf = torch.tensor([conf], dtype=torch.float32)
    return box


def _make_mock_results(boxes_data: list[tuple[int, float, tuple[int, int, int, int]]]):
    """Create a mock ultralytics Results list.

    Parameters
    ----------
    boxes_data : list of (class_id, confidence, xyxy) tuples
    """
    result = mock.MagicMock()

    if not boxes_data:
        # Empty boxes — len 0, iterates to nothing
        mock_boxes = mock.MagicMock()
        mock_boxes.__len__ = mock.Mock(return_value=0)
        mock_boxes.__iter__ = mock.Mock(return_value=iter([]))
        result.boxes = mock_boxes
        return [result]

    mock_boxes_list = [
        _make_mock_box(cls_id, conf, xyxy) for cls_id, conf, xyxy in boxes_data
    ]
    result.boxes = mock_boxes_list
    return [result]


def _make_detector_with_mock_model(
    boxes_data: list[tuple[int, float, tuple[int, int, int, int]]] | None = None,
    class_names: list[str] | None = None,
) -> YoloDetector:
    """Create a YoloDetector with a mocked model already 'loaded'.

    Parameters
    ----------
    boxes_data : list or None
        Detection data to return from model calls.
        None means the model returns empty results.
    class_names : list[str] or None
        Class names to use. Defaults to BREAKOUT71_CLASSES.
    """
    detector = YoloDetector(classes=class_names)
    detector.model = mock.MagicMock(name="YOLO")

    data = boxes_data if boxes_data is not None else []
    detector.model.return_value = _make_mock_results(data)

    return detector


# ── YoloDetector Construction ───────────────────────────────────────


class TestYoloDetectorInit:
    """Construction and availability guard tests."""

    def test_detector_construction(self):
        """YoloDetector can be constructed with default args."""
        detector = YoloDetector()
        assert detector.confidence_threshold == 0.5
        assert detector.device in ("xpu", "cuda", "cpu")  # resolved from "auto"
        assert detector.iou_threshold == 0.45
        assert detector.img_size == 640

    def test_default_classes(self):
        """Default class list should match Breakout 71 spec."""
        detector = YoloDetector()
        assert "paddle" in detector.class_names
        assert "ball" in detector.class_names
        assert "brick" in detector.class_names
        assert "powerup" in detector.class_names
        assert "wall" in detector.class_names

    def test_is_loaded_before_load(self):
        """is_loaded should return False before load() is called."""
        detector = YoloDetector()
        assert not detector.is_loaded()

    def test_custom_classes(self):
        """Custom class names should override defaults."""
        custom = ["player", "enemy", "coin"]
        detector = YoloDetector(classes=custom)
        assert detector.class_names == custom

    def test_custom_weights_path(self):
        """Custom weights path is stored as a Path."""
        detector = YoloDetector(weights_path="/my/model.pt")
        assert detector.weights_path == Path("/my/model.pt")

    def test_custom_device(self):
        """Custom device string is stored."""
        detector = YoloDetector(device="cuda:0")
        assert detector.device == "cuda:0"

    def test_custom_thresholds(self):
        """Custom confidence and IoU thresholds are stored."""
        detector = YoloDetector(confidence_threshold=0.8, iou_threshold=0.3)
        assert detector.confidence_threshold == 0.8
        assert detector.iou_threshold == 0.3

    def test_raises_without_ultralytics(self):
        """YoloDetector raises RuntimeError when ultralytics is missing."""
        sys.modules.pop("src.perception.yolo_detector", None)
        with mock.patch.dict(sys.modules, {"ultralytics": None}):
            import src.perception.yolo_detector as yd_mod

            importlib.reload(yd_mod)
            assert yd_mod._ULTRALYTICS_AVAILABLE is False
            with pytest.raises(RuntimeError, match="ultralytics is required"):
                yd_mod.YoloDetector()

        # Restore module
        sys.modules.pop("src.perception.yolo_detector", None)
        importlib.import_module("src.perception.yolo_detector")


# ── load() ──────────────────────────────────────────────────────────


class TestYoloDetectorLoad:
    """Tests for model loading.

    These tests import ``YoloDetector`` freshly from the live module
    to ensure ``YOLO`` patching targets the same module object that
    the class's ``load()`` method resolves ``YOLO`` from.
    """

    @staticmethod
    def _fresh_import():
        """Return (module, YoloDetector) from the live sys.modules entry."""
        import src.perception.yolo_detector as yd

        return yd, yd.YoloDetector

    def test_load_missing_weights_raises(self, tmp_path):
        """load() raises FileNotFoundError for non-existent weights."""
        _, YD = self._fresh_import()
        detector = YD(weights_path=tmp_path / "nonexistent.pt")
        with pytest.raises(FileNotFoundError, match="not found"):
            detector.load()

    def test_load_success(self, tmp_path):
        """load() loads model and sets is_loaded to True."""
        yd, YD = self._fresh_import()
        weights = tmp_path / "best.pt"
        weights.touch()

        mock_model = mock.MagicMock()
        mock_model.names = {0: "paddle", 1: "ball", 2: "brick"}

        with mock.patch.object(yd, "YOLO", return_value=mock_model) as mock_cls:
            detector = YD(weights_path=weights, device="cpu")
            detector.load()

            assert detector.is_loaded()
            mock_cls.assert_called_once_with(str(weights))
            mock_model.to.assert_called_with("cpu")

    def test_load_xpu_fallback_to_cpu(self, tmp_path):
        """load() falls back to CPU if XPU device fails."""
        yd, YD = self._fresh_import()
        weights = tmp_path / "best.pt"
        weights.touch()

        mock_model = mock.MagicMock()
        mock_model.names = {}
        mock_model.to.side_effect = [RuntimeError("XPU not available"), None]

        with mock.patch.object(yd, "YOLO", return_value=mock_model):
            detector = YD(weights_path=weights, device="xpu")
            detector.load()

            assert detector.is_loaded()
            assert detector.device == "cpu"
            assert mock_model.to.call_count == 2
            mock_model.to.assert_any_call("xpu")
            mock_model.to.assert_any_call("cpu")

    def test_load_cpu_fallback_also_fails(self, tmp_path):
        """load() raises RuntimeError if both XPU and CPU fallback fail."""
        yd, YD = self._fresh_import()
        weights = tmp_path / "best.pt"
        weights.touch()

        mock_model = mock.MagicMock()
        mock_model.names = {}
        mock_model.to.side_effect = [
            RuntimeError("XPU not available"),
            RuntimeError("CPU also broken"),
        ]

        with mock.patch.object(yd, "YOLO", return_value=mock_model):
            detector = YD(weights_path=weights, device="xpu")
            with pytest.raises(RuntimeError, match="Failed to move YOLO model"):
                detector.load()
            assert not detector.is_loaded()

    def test_load_reads_model_class_names_dict(self, tmp_path):
        """load() reads class names from model.names dict when user didn't specify."""
        yd, YD = self._fresh_import()
        weights = tmp_path / "best.pt"
        weights.touch()

        mock_model = mock.MagicMock()
        mock_model.names = {0: "alpha", 1: "beta", 2: "gamma"}

        with mock.patch.object(yd, "YOLO", return_value=mock_model):
            detector = YD(weights_path=weights, device="cpu")
            detector.load()
            assert detector.class_names == ["alpha", "beta", "gamma"]

    def test_load_reads_model_class_names_list(self, tmp_path):
        """load() reads class names from model.names list."""
        yd, YD = self._fresh_import()
        weights = tmp_path / "best.pt"
        weights.touch()

        mock_model = mock.MagicMock()
        mock_model.names = ["x", "y", "z"]

        with mock.patch.object(yd, "YOLO", return_value=mock_model):
            detector = YD(weights_path=weights, device="cpu")
            detector.load()
            assert detector.class_names == ["x", "y", "z"]

    def test_load_user_classes_not_overwritten(self, tmp_path):
        """User-specified classes are preserved even when model has names."""
        yd, YD = self._fresh_import()
        weights = tmp_path / "best.pt"
        weights.touch()

        mock_model = mock.MagicMock()
        mock_model.names = {0: "alpha", 1: "beta"}

        with mock.patch.object(yd, "YOLO", return_value=mock_model):
            custom = ["one", "two", "three"]
            detector = YD(weights_path=weights, device="cpu", classes=custom)
            detector.load()
            assert detector.class_names == custom

    def test_is_loaded_after_load(self, tmp_path):
        """is_loaded returns True after successful load."""
        yd, YD = self._fresh_import()
        weights = tmp_path / "best.pt"
        weights.touch()

        mock_model = mock.MagicMock()
        mock_model.names = {}

        with mock.patch.object(yd, "YOLO", return_value=mock_model):
            detector = YD(weights_path=weights, device="cpu")
            assert not detector.is_loaded()
            detector.load()
            assert detector.is_loaded()


# ── detect() ────────────────────────────────────────────────────────


class TestYoloDetectorDetect:
    """Tests for inference and detection parsing."""

    def test_detect_not_loaded_raises(self):
        """detect() raises RuntimeError if model is not loaded."""
        detector = YoloDetector()
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="not loaded"):
            detector.detect(frame)

    def test_detect_single_detection(self):
        """detect() returns correct dict for a single detection."""
        detector = _make_detector_with_mock_model(
            boxes_data=[(0, 0.95, (100, 200, 300, 400))],
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert len(result) == 1
        det = result[0]
        assert det["class_name"] == "paddle"
        assert det["class_id"] == 0
        assert det["confidence"] == pytest.approx(0.95, abs=0.01)
        assert det["bbox_xyxy"] == (100, 200, 300, 400)

        # Normalised: cx=(100+300)/2/640=0.3125, cy=(200+400)/2/480=0.625
        cx, cy, w, h = det["bbox_xywh_norm"]
        assert cx == pytest.approx(0.3125, abs=0.001)
        assert cy == pytest.approx(0.625, abs=0.001)
        assert w == pytest.approx(200 / 640, abs=0.001)
        assert h == pytest.approx(200 / 480, abs=0.001)

    def test_detect_multiple_detections(self):
        """detect() returns all detections from the frame."""
        detector = _make_detector_with_mock_model(
            boxes_data=[
                (0, 0.9, (10, 10, 50, 30)),  # paddle
                (1, 0.85, (100, 100, 120, 120)),  # ball
                (2, 0.7, (200, 50, 250, 70)),  # brick
            ],
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert len(result) == 3
        assert result[0]["class_name"] == "paddle"
        assert result[1]["class_name"] == "ball"
        assert result[2]["class_name"] == "brick"

    def test_detect_empty_results(self):
        """detect() returns empty list when no detections found."""
        detector = _make_detector_with_mock_model(boxes_data=[])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert result == []

    def test_detect_none_results(self):
        """detect() returns empty list when model returns None boxes."""
        detector = _make_detector_with_mock_model()
        # Override model to return result with None boxes
        result_obj = mock.MagicMock()
        result_obj.boxes = None
        detector.model.return_value = [result_obj]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert result == []

    def test_detect_empty_result_list(self):
        """detect() returns empty list when model returns empty list."""
        detector = _make_detector_with_mock_model()
        detector.model.return_value = []

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert result == []

    def test_detect_unknown_class_id(self):
        """detect() assigns 'class_N' for class IDs beyond class_names."""
        detector = _make_detector_with_mock_model(
            boxes_data=[(99, 0.8, (10, 10, 50, 50))],
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert len(result) == 1
        assert result[0]["class_name"] == "class_99"
        assert result[0]["class_id"] == 99

    def test_detect_passes_parameters_to_model(self):
        """detect() passes img_size, conf, iou to model call."""
        detector = YoloDetector(
            confidence_threshold=0.7,
            iou_threshold=0.3,
            img_size=320,
        )
        detector.model = mock.MagicMock()
        detector.model.return_value = _make_mock_results([])

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.detect(frame)

        detector.model.assert_called_once()
        call_kwargs = detector.model.call_args
        assert call_kwargs.kwargs["imgsz"] == 320
        assert call_kwargs.kwargs["conf"] == 0.7
        assert call_kwargs.kwargs["iou"] == 0.3
        assert call_kwargs.kwargs["verbose"] is False


# ── detect_to_game_state() ──────────────────────────────────────────


class TestYoloDetectorGameState:
    """Tests for detect_to_game_state — structured game state extraction."""

    def test_full_game_state(self):
        """detect_to_game_state returns paddle, ball, bricks, powerups."""
        detector = _make_detector_with_mock_model(
            boxes_data=[
                (0, 0.95, (280, 440, 360, 460)),  # paddle
                (1, 0.90, (310, 200, 330, 220)),  # ball
                (2, 0.80, (100, 50, 150, 70)),  # brick 1
                (2, 0.75, (200, 50, 250, 70)),  # brick 2
                (3, 0.60, (300, 300, 320, 320)),  # powerup
            ],
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state = detector.detect_to_game_state(frame, 640, 480)

        assert state["paddle"] is not None
        assert state["ball"] is not None
        assert len(state["bricks"]) == 2
        assert len(state["powerups"]) == 1
        assert len(state["raw_detections"]) == 5

    def test_missing_paddle_returns_none(self):
        """detect_to_game_state returns None for paddle when not detected."""
        detector = _make_detector_with_mock_model(
            boxes_data=[
                (1, 0.90, (310, 200, 330, 220)),  # ball only
            ],
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state = detector.detect_to_game_state(frame, 640, 480)

        assert state["paddle"] is None
        assert state["ball"] is not None

    def test_missing_ball_returns_none(self):
        """detect_to_game_state returns None for ball when not detected."""
        detector = _make_detector_with_mock_model(
            boxes_data=[
                (0, 0.95, (280, 440, 360, 460)),  # paddle only
            ],
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state = detector.detect_to_game_state(frame, 640, 480)

        assert state["paddle"] is not None
        assert state["ball"] is None

    def test_no_detections_returns_empty_state(self):
        """detect_to_game_state returns all-empty when nothing detected."""
        detector = _make_detector_with_mock_model(boxes_data=[])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state = detector.detect_to_game_state(frame, 640, 480)

        assert state["paddle"] is None
        assert state["ball"] is None
        assert state["bricks"] == []
        assert state["powerups"] == []
        assert state["raw_detections"] == []

    def test_multiple_paddles_picks_highest_confidence(self):
        """When multiple paddles detected, pick the one with highest confidence."""
        detector = _make_detector_with_mock_model(
            boxes_data=[
                (0, 0.60, (100, 440, 180, 460)),  # paddle low conf
                (0, 0.95, (280, 440, 360, 460)),  # paddle high conf
            ],
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state = detector.detect_to_game_state(frame, 640, 480)

        # Should pick the second paddle (higher confidence)
        assert state["paddle"] is not None
        cx = state["paddle"][0]
        expected_cx = (280 + 360) / 2.0 / 640
        assert cx == pytest.approx(expected_cx, abs=0.001)

    def test_multiple_balls_picks_highest_confidence(self):
        """When multiple balls detected, pick the one with highest confidence."""
        detector = _make_detector_with_mock_model(
            boxes_data=[
                (1, 0.50, (50, 50, 70, 70)),  # ball low conf
                (1, 0.99, (310, 200, 330, 220)),  # ball high conf
            ],
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state = detector.detect_to_game_state(frame, 640, 480)

        assert state["ball"] is not None
        cx = state["ball"][0]
        expected_cx = (310 + 330) / 2.0 / 640
        assert cx == pytest.approx(expected_cx, abs=0.001)

    def test_normalisation_uses_provided_dimensions(self):
        """detect_to_game_state uses frame_width/frame_height, not frame.shape."""
        detector = _make_detector_with_mock_model(
            boxes_data=[
                (0, 0.95, (0, 0, 100, 50)),  # paddle at top-left
            ],
        )
        # Frame is 480x640 but we pass different dimensions
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state = detector.detect_to_game_state(frame, 1000, 500)

        assert state["paddle"] is not None
        cx, cy, w, h = state["paddle"]
        # cx = (0+100)/2/1000 = 0.05, cy = (0+50)/2/500 = 0.05
        assert cx == pytest.approx(0.05, abs=0.001)
        assert cy == pytest.approx(0.05, abs=0.001)
        assert w == pytest.approx(100 / 1000, abs=0.001)
        assert h == pytest.approx(50 / 500, abs=0.001)

    def test_wall_detections_not_in_game_state_keys(self):
        """Wall detections appear only in raw_detections, not in named keys."""
        detector = _make_detector_with_mock_model(
            boxes_data=[
                (4, 0.80, (0, 0, 640, 10)),  # wall
            ],
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state = detector.detect_to_game_state(frame, 640, 480)

        assert state["paddle"] is None
        assert state["ball"] is None
        assert state["bricks"] == []
        assert state["powerups"] == []
        assert len(state["raw_detections"]) == 1
        assert state["raw_detections"][0]["class_name"] == "wall"


# ── breakout_capture ────────────────────────────────────────────────


class TestBreakoutCapture:
    """Tests for the breakout_capture convenience functions."""

    def test_grab_frame_delegates_to_capture(self):
        """grab_frame calls capture.capture_frame()."""
        from src.perception.breakout_capture import grab_frame

        mock_capture = mock.MagicMock()
        mock_capture.is_window_visible.return_value = True
        expected_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_capture.capture_frame.return_value = expected_frame

        result = grab_frame(mock_capture)

        mock_capture.capture_frame.assert_called_once()
        np.testing.assert_array_equal(result, expected_frame)

    def test_grab_frame_raises_on_invisible_window(self):
        """grab_frame raises RuntimeError if window is not visible."""
        from src.perception.breakout_capture import grab_frame

        mock_capture = mock.MagicMock()
        mock_capture.is_window_visible.return_value = False

        with pytest.raises(RuntimeError, match="not visible"):
            grab_frame(mock_capture)

        mock_capture.capture_frame.assert_not_called()

    def test_detect_objects_delegates_to_detector(self):
        """detect_objects calls detector.detect_to_game_state()."""
        from src.perception.breakout_capture import detect_objects

        mock_detector = mock.MagicMock()
        expected_state = {
            "paddle": (0.5, 0.9, 0.1, 0.05),
            "ball": (0.5, 0.5, 0.02, 0.02),
            "bricks": [],
            "powerups": [],
            "raw_detections": [],
        }
        mock_detector.detect_to_game_state.return_value = expected_state

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_objects(mock_detector, frame, 640, 480)

        assert result == expected_state
        mock_detector.detect_to_game_state.assert_called_once_with(frame, 640, 480)

    def test_detect_objects_auto_infers_dimensions(self):
        """detect_objects uses frame.shape when dimensions not provided."""
        from src.perception.breakout_capture import detect_objects

        mock_detector = mock.MagicMock()
        mock_detector.detect_to_game_state.return_value = {}

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detect_objects(mock_detector, frame)

        mock_detector.detect_to_game_state.assert_called_once_with(frame, 640, 480)

    def test_detect_objects_explicit_dims_override_shape(self):
        """detect_objects uses explicit dimensions over frame.shape."""
        from src.perception.breakout_capture import detect_objects

        mock_detector = mock.MagicMock()
        mock_detector.detect_to_game_state.return_value = {}

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detect_objects(mock_detector, frame, frame_width=1000, frame_height=500)

        mock_detector.detect_to_game_state.assert_called_once_with(frame, 1000, 500)


# ── Module-level imports ────────────────────────────────────────────


class TestModuleImports:
    """Tests for module-level exports and imports."""

    def test_yolo_detector_importable_from_package(self):
        """YoloDetector is importable from src.perception."""
        from src.perception import YoloDetector as YD

        assert YD is YoloDetector

    def test_grab_frame_importable_from_package(self):
        """grab_frame is importable from src.perception."""
        from src.perception import grab_frame

        assert callable(grab_frame)

    def test_detect_objects_importable_from_package(self):
        """detect_objects is importable from src.perception."""
        from src.perception import detect_objects

        assert callable(detect_objects)

    def test_all_exports(self):
        """__all__ contains expected exports."""
        import src.perception as mod

        assert "YoloDetector" in mod.__all__
        assert "grab_frame" in mod.__all__
        assert "detect_objects" in mod.__all__


# ── _find_openvino_model ────────────────────────────────────────────


class TestFindOpenvinoModel:
    """Tests for _find_openvino_model helper."""

    def test_returns_none_when_dir_missing(self, tmp_path):
        """Returns None when no *_openvino_model/ directory exists."""
        weights = tmp_path / "best.pt"
        weights.touch()
        assert _find_openvino_model(weights) is None

    def test_returns_none_when_dir_empty(self, tmp_path):
        """Returns None when openvino dir exists but has no .xml files."""
        weights = tmp_path / "best.pt"
        weights.touch()
        ov_dir = tmp_path / "best_openvino_model"
        ov_dir.mkdir()
        assert _find_openvino_model(weights) is None

    def test_returns_dir_when_xml_exists(self, tmp_path):
        """Returns the openvino dir path when it contains .xml files."""
        weights = tmp_path / "best.pt"
        weights.touch()
        ov_dir = tmp_path / "best_openvino_model"
        ov_dir.mkdir()
        (ov_dir / "best.xml").touch()
        (ov_dir / "best.bin").touch()
        result = _find_openvino_model(weights)
        assert result == ov_dir

    def test_stem_based_naming(self, tmp_path):
        """Uses the weights file stem to find the openvino directory."""
        weights = tmp_path / "mymodel.pt"
        weights.touch()
        ov_dir = tmp_path / "mymodel_openvino_model"
        ov_dir.mkdir()
        (ov_dir / "mymodel.xml").touch()
        assert _find_openvino_model(weights) == ov_dir


# ── _resolve_openvino_device ────────────────────────────────────────


class TestResolveOpenvinoDevice:
    """Tests for _resolve_openvino_device helper."""

    def test_cpu_returns_intel_cpu(self):
        """CPU device maps to 'intel:CPU'."""
        assert _resolve_openvino_device("cpu") == "intel:CPU"

    def test_auto_returns_intel_auto(self):
        """Auto device maps to 'intel:AUTO'."""
        assert _resolve_openvino_device("auto") == "intel:AUTO"

    def test_unknown_device_returns_intel_auto(self):
        """Unknown device falls back to 'intel:AUTO'."""
        assert _resolve_openvino_device("cuda") == "intel:AUTO"

    def test_xpu_with_gpu0_available(self):
        """XPU maps to 'intel:GPU.0' when GPU.0 is in available_devices."""
        mock_core = mock.MagicMock()
        mock_core.available_devices = ["CPU", "GPU.0", "GPU.1"]

        mock_ov = mock.MagicMock()
        mock_ov.Core.return_value = mock_core

        with mock.patch.dict(sys.modules, {"openvino": mock_ov}):
            result = _resolve_openvino_device("xpu")

        assert result == "intel:GPU.0"

    def test_xpu_with_plain_gpu_available(self):
        """XPU maps to 'intel:GPU' when plain 'GPU' is in available_devices."""
        mock_core = mock.MagicMock()
        mock_core.available_devices = ["CPU", "GPU"]

        mock_ov = mock.MagicMock()
        mock_ov.Core.return_value = mock_core

        with mock.patch.dict(sys.modules, {"openvino": mock_ov}):
            result = _resolve_openvino_device("xpu")

        assert result == "intel:GPU"

    def test_xpu_without_openvino_installed(self):
        """XPU falls back to 'intel:GPU' when openvino is not installed."""
        with mock.patch.dict(sys.modules, {"openvino": None}):
            # Importing None from sys.modules raises ImportError
            result = _resolve_openvino_device("xpu")

        assert result == "intel:GPU"

    def test_xpu_openvino_query_fails(self):
        """XPU falls back to 'intel:GPU' when ov.Core() raises."""
        mock_ov = mock.MagicMock()
        mock_ov.Core.side_effect = RuntimeError("OpenVINO init failed")

        with mock.patch.dict(sys.modules, {"openvino": mock_ov}):
            result = _resolve_openvino_device("xpu")

        assert result == "intel:GPU"

    def test_xpu_no_gpu_in_available_devices(self):
        """XPU falls back to 'intel:GPU' when no GPU device is available."""
        mock_core = mock.MagicMock()
        mock_core.available_devices = ["CPU"]

        mock_ov = mock.MagicMock()
        mock_ov.Core.return_value = mock_core

        with mock.patch.dict(sys.modules, {"openvino": mock_ov}):
            result = _resolve_openvino_device("xpu")

        assert result == "intel:GPU"


# ── OpenVINO model selection in load() ──────────────────────────────


class TestOpenvinoModelSelection:
    """Tests for automatic OpenVINO model selection in load()."""

    @staticmethod
    def _fresh_import():
        """Return (module, YoloDetector) from the live sys.modules entry."""
        import src.perception.yolo_detector as yd

        return yd, yd.YoloDetector

    def _setup_openvino_dir(self, tmp_path):
        """Create a weights file and an openvino model dir with .xml."""
        weights = tmp_path / "best.pt"
        weights.touch()
        ov_dir = tmp_path / "best_openvino_model"
        ov_dir.mkdir()
        (ov_dir / "best.xml").touch()
        (ov_dir / "best.bin").touch()
        return weights, ov_dir

    def test_cpu_device_prefers_openvino(self, tmp_path):
        """On CPU, load() uses OpenVINO model when available."""
        yd, YD = self._fresh_import()
        weights, ov_dir = self._setup_openvino_dir(tmp_path)

        mock_model = mock.MagicMock()
        mock_model.names = {0: "paddle", 1: "ball"}

        with mock.patch.object(yd, "YOLO", return_value=mock_model) as mock_cls:
            detector = YD(weights_path=weights, device="cpu")
            detector.load()

            # Should load the OpenVINO directory, not the .pt file
            mock_cls.assert_called_once_with(str(ov_dir))
            assert detector.is_loaded()
            assert detector._using_openvino is True
            assert detector._ov_device == "intel:CPU"
            # Should NOT call model.to() for OpenVINO models
            mock_model.to.assert_not_called()

    def test_auto_device_prefers_openvino(self, tmp_path):
        """With auto device resolving to cpu, load() uses OpenVINO."""
        yd, YD = self._fresh_import()
        weights, ov_dir = self._setup_openvino_dir(tmp_path)

        mock_model = mock.MagicMock()
        mock_model.names = {}

        with (
            mock.patch.object(yd, "YOLO", return_value=mock_model) as mock_cls,
            mock.patch.object(yd, "resolve_device", return_value="cpu"),
        ):
            detector = YD(weights_path=weights, device="auto")
            detector.load()

            mock_cls.assert_called_once_with(str(ov_dir))
            assert detector._using_openvino is True
            assert detector._ov_device == "intel:CPU"

    def test_cuda_device_uses_pytorch(self, tmp_path):
        """On CUDA, load() uses PyTorch .pt even when OpenVINO exists."""
        yd, YD = self._fresh_import()
        weights, _ov_dir = self._setup_openvino_dir(tmp_path)

        mock_model = mock.MagicMock()
        mock_model.names = {}

        with mock.patch.object(yd, "YOLO", return_value=mock_model) as mock_cls:
            detector = YD(weights_path=weights, device="cuda")
            detector.load()

            # Should load the .pt file directly
            mock_cls.assert_called_once_with(str(weights))
            assert detector._using_openvino is False
            mock_model.to.assert_called_with("cuda")

    def test_cpu_no_openvino_dir_uses_pytorch(self, tmp_path):
        """On CPU without OpenVINO dir, load() falls back to PyTorch."""
        yd, YD = self._fresh_import()
        weights = tmp_path / "best.pt"
        weights.touch()
        # No openvino dir created

        mock_model = mock.MagicMock()
        mock_model.names = {}

        with mock.patch.object(yd, "YOLO", return_value=mock_model) as mock_cls:
            detector = YD(weights_path=weights, device="cpu")
            detector.load()

            mock_cls.assert_called_once_with(str(weights))
            assert detector._using_openvino is False
            mock_model.to.assert_called_with("cpu")

    def test_xpu_device_prefers_openvino(self, tmp_path):
        """On XPU, load() uses OpenVINO model when available."""
        yd, YD = self._fresh_import()
        weights, ov_dir = self._setup_openvino_dir(tmp_path)

        mock_model = mock.MagicMock()
        mock_model.names = {}

        # Mock openvino to report GPU.0 as available
        mock_core = mock.MagicMock()
        mock_core.available_devices = ["CPU", "GPU.0", "GPU.1"]
        mock_ov = mock.MagicMock()
        mock_ov.Core.return_value = mock_core

        with (
            mock.patch.object(yd, "YOLO", return_value=mock_model) as mock_cls,
            mock.patch.dict(sys.modules, {"openvino": mock_ov}),
        ):
            detector = YD(weights_path=weights, device="xpu")
            detector.load()

            mock_cls.assert_called_once_with(str(ov_dir))
            assert detector._using_openvino is True
            assert detector._ov_device == "intel:GPU.0"

    def test_openvino_model_reads_class_names(self, tmp_path):
        """OpenVINO model still reads class names from model metadata."""
        yd, YD = self._fresh_import()
        weights, _ov_dir = self._setup_openvino_dir(tmp_path)

        mock_model = mock.MagicMock()
        mock_model.names = {0: "paddle", 1: "ball", 2: "brick"}

        with mock.patch.object(yd, "YOLO", return_value=mock_model):
            detector = YD(weights_path=weights, device="cpu")
            detector.load()

            assert detector.class_names == ["paddle", "ball", "brick"]

    def test_detect_passes_ov_device_kwarg(self, tmp_path):
        """detect() passes device kwarg when using OpenVINO model."""
        yd, YD = self._fresh_import()
        weights, _ov_dir = self._setup_openvino_dir(tmp_path)

        mock_model = mock.MagicMock()
        mock_model.names = {}
        mock_model.return_value = _make_mock_results([])

        with mock.patch.object(yd, "YOLO", return_value=mock_model):
            detector = YD(weights_path=weights, device="cpu")
            detector.load()

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            detector.detect(frame)

            call_kwargs = mock_model.call_args.kwargs
            assert call_kwargs["device"] == "intel:CPU"

    def test_detect_no_device_kwarg_for_pytorch(self, tmp_path):
        """detect() does NOT pass device kwarg for PyTorch model."""
        yd, YD = self._fresh_import()
        weights = tmp_path / "best.pt"
        weights.touch()
        # No openvino dir

        mock_model = mock.MagicMock()
        mock_model.names = {}
        mock_model.return_value = _make_mock_results([])

        with mock.patch.object(yd, "YOLO", return_value=mock_model):
            detector = YD(weights_path=weights, device="cpu")
            detector.load()

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            detector.detect(frame)

            call_kwargs = mock_model.call_args.kwargs
            assert "device" not in call_kwargs

    def test_openvino_warmup_runs_on_target_device(self, tmp_path):
        """load() runs a warmup inference with device kwarg for OpenVINO."""
        yd, YD = self._fresh_import()
        weights, _ov_dir = self._setup_openvino_dir(tmp_path)

        mock_model = mock.MagicMock()
        mock_model.names = {}

        with mock.patch.object(yd, "YOLO", return_value=mock_model):
            detector = YD(weights_path=weights, device="cpu")
            detector.load()

            # model() should have been called once during load() for warmup
            assert mock_model.call_count == 1
            warmup_call = mock_model.call_args
            # Warmup frame is a zero array of img_size x img_size x 3
            warmup_frame = warmup_call.args[0]
            assert warmup_frame.shape == (640, 640, 3)
            assert warmup_frame.dtype == np.uint8
            # Device kwarg must match the resolved OpenVINO device
            assert warmup_call.kwargs["device"] == "intel:CPU"
            assert warmup_call.kwargs["verbose"] is False

    def test_openvino_warmup_uses_gpu_device(self, tmp_path):
        """load() warmup passes intel:GPU.0 when device is xpu."""
        yd, YD = self._fresh_import()
        weights, _ov_dir = self._setup_openvino_dir(tmp_path)

        mock_model = mock.MagicMock()
        mock_model.names = {}

        mock_core = mock.MagicMock()
        mock_core.available_devices = ["CPU", "GPU.0", "GPU.1"]
        mock_ov = mock.MagicMock()
        mock_ov.Core.return_value = mock_core

        with (
            mock.patch.object(yd, "YOLO", return_value=mock_model),
            mock.patch.dict(sys.modules, {"openvino": mock_ov}),
        ):
            detector = YD(weights_path=weights, device="xpu")
            detector.load()

            assert mock_model.call_count == 1
            warmup_kwargs = mock_model.call_args.kwargs
            assert warmup_kwargs["device"] == "intel:GPU.0"

    def test_pytorch_load_no_warmup_call(self, tmp_path):
        """load() does NOT call model() for warmup with PyTorch .pt."""
        yd, YD = self._fresh_import()
        weights = tmp_path / "best.pt"
        weights.touch()
        # No openvino dir

        mock_model = mock.MagicMock()
        mock_model.names = {}

        with mock.patch.object(yd, "YOLO", return_value=mock_model):
            detector = YD(weights_path=weights, device="cpu")
            detector.load()

            # model() should NOT be called during load for PyTorch
            assert mock_model.call_count == 0
            mock_model.assert_not_called()
