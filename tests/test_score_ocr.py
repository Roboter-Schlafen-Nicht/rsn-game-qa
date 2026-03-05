"""Tests for the ScoreOCR platform module.

Verifies score reading from game frames using OCR (pytesseract).
All tests mock pytesseract to avoid requiring the system binary
in CI.
"""

from __future__ import annotations

from unittest import mock

import numpy as np

from src.platform.score_ocr import ScoreOCR

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(h: int = 480, w: int = 640, channels: int = 3) -> np.ndarray:
    """Create a dummy BGR frame."""
    return np.zeros((h, w, channels), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestScoreOCRConstruction:
    """Test ScoreOCR initialisation and parameter validation."""

    def test_default_construction(self):
        """ScoreOCR can be created with no arguments."""
        ocr = ScoreOCR()
        assert ocr._region is None
        assert ocr._interval == 1
        assert ocr._last_score is None

    def test_custom_region(self):
        """ScoreOCR accepts a (x, y, w, h) region tuple."""
        ocr = ScoreOCR(region=(10, 20, 100, 30))
        assert ocr._region == (10, 20, 100, 30)

    def test_custom_interval(self):
        """ScoreOCR accepts an ocr_interval parameter."""
        ocr = ScoreOCR(ocr_interval=5)
        assert ocr._interval == 5

    def test_interval_minimum_clamped_to_1(self):
        """ocr_interval below 1 is clamped to 1."""
        ocr = ScoreOCR(ocr_interval=0)
        assert ocr._interval == 1
        ocr2 = ScoreOCR(ocr_interval=-3)
        assert ocr2._interval == 1


# ---------------------------------------------------------------------------
# Score reading
# ---------------------------------------------------------------------------


class TestReadScore:
    """Test read_score method."""

    def test_read_score_extracts_number_from_ocr_text(self):
        """read_score returns the detected number from OCR output."""
        ocr = ScoreOCR()
        frame = _make_frame()
        mock_pytesseract = mock.MagicMock()
        mock_pytesseract.image_to_string.return_value = "Score: 1234"
        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            score = ocr.read_score(frame)
        assert score == 1234

    def test_read_score_returns_largest_number(self):
        """When multiple numbers found, return the largest."""
        ocr = ScoreOCR()
        frame = _make_frame()
        mock_pytesseract = mock.MagicMock()
        mock_pytesseract.image_to_string.return_value = "Level 3  Score 5678"
        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            score = ocr.read_score(frame)
        assert score == 5678

    def test_read_score_returns_none_when_no_numbers(self):
        """read_score returns None when OCR finds no numbers."""
        ocr = ScoreOCR()
        frame = _make_frame()
        mock_pytesseract = mock.MagicMock()
        mock_pytesseract.image_to_string.return_value = "Game Over"
        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            score = ocr.read_score(frame)
        assert score is None

    def test_read_score_returns_none_when_pytesseract_missing(self):
        """read_score returns None gracefully when pytesseract is not installed."""
        ocr = ScoreOCR()
        frame = _make_frame()
        with mock.patch.dict("sys.modules", {"pytesseract": None}):
            score = ocr.read_score(frame)
        assert score is None

    def test_read_score_returns_none_on_ocr_exception(self):
        """read_score returns None when OCR raises an exception."""
        ocr = ScoreOCR()
        frame = _make_frame()
        mock_pytesseract = mock.MagicMock()
        mock_pytesseract.image_to_string.side_effect = RuntimeError("OCR failed")
        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            score = ocr.read_score(frame)
        assert score is None

    def test_read_score_crops_to_region(self):
        """read_score crops the frame to the specified region before OCR."""
        ocr = ScoreOCR(region=(10, 20, 100, 30))
        frame = _make_frame(480, 640)
        mock_pytesseract = mock.MagicMock()
        mock_pytesseract.image_to_string.return_value = "42"
        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            score = ocr.read_score(frame)
        # Verify the cropped region was passed to OCR
        call_args = mock_pytesseract.image_to_string.call_args
        cropped = call_args[0][0]
        assert cropped.shape[0] == 30  # height
        assert cropped.shape[1] == 100  # width
        assert score == 42


# ---------------------------------------------------------------------------
# Throttling
# ---------------------------------------------------------------------------


class TestThrottling:
    """Test OCR interval throttling."""

    def test_throttle_skips_intermediate_frames(self):
        """With interval=3, OCR runs on frame 1, 4, 7, etc."""
        ocr = ScoreOCR(ocr_interval=3)
        mock_pytesseract = mock.MagicMock()
        mock_pytesseract.image_to_string.return_value = "100"
        frame = _make_frame()

        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            # Frame 1: OCR runs
            s1 = ocr.read_score(frame)
            assert s1 == 100
            assert mock_pytesseract.image_to_string.call_count == 1

            # Frame 2: throttled, returns cached
            s2 = ocr.read_score(frame)
            assert s2 == 100
            assert mock_pytesseract.image_to_string.call_count == 1

            # Frame 3: throttled, returns cached
            s3 = ocr.read_score(frame)
            assert s3 == 100
            assert mock_pytesseract.image_to_string.call_count == 1

            # Frame 4: OCR runs again
            mock_pytesseract.image_to_string.return_value = "200"
            s4 = ocr.read_score(frame)
            assert s4 == 200
            assert mock_pytesseract.image_to_string.call_count == 2

    def test_interval_1_runs_every_frame(self):
        """With interval=1, OCR runs every frame."""
        ocr = ScoreOCR(ocr_interval=1)
        mock_pytesseract = mock.MagicMock()
        mock_pytesseract.image_to_string.return_value = "50"
        frame = _make_frame()

        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            ocr.read_score(frame)
            ocr.read_score(frame)
            ocr.read_score(frame)
            assert mock_pytesseract.image_to_string.call_count == 3


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    """Test reset method."""

    def test_reset_clears_state(self):
        """reset() clears cached score and frame counter."""
        ocr = ScoreOCR(ocr_interval=3)
        mock_pytesseract = mock.MagicMock()
        mock_pytesseract.image_to_string.return_value = "500"
        frame = _make_frame()

        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            # Read a score to populate cache
            ocr.read_score(frame)
            assert ocr._last_score == 500

            # Reset
            ocr.reset()
            assert ocr._last_score is None
            assert ocr._frame_count == 0

    def test_reset_allows_immediate_ocr_on_next_read(self):
        """After reset(), the next read_score runs OCR immediately."""
        ocr = ScoreOCR(ocr_interval=5)
        mock_pytesseract = mock.MagicMock()
        mock_pytesseract.image_to_string.return_value = "100"
        frame = _make_frame()

        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            ocr.read_score(frame)  # frame 1: runs OCR
            ocr.read_score(frame)  # frame 2: throttled
            assert mock_pytesseract.image_to_string.call_count == 1

            ocr.reset()
            mock_pytesseract.image_to_string.return_value = "999"
            s = ocr.read_score(frame)  # should run OCR immediately
            assert s == 999
            assert mock_pytesseract.image_to_string.call_count == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases in score parsing."""

    def test_handles_zero_score(self):
        """Score of 0 is returned correctly (not None)."""
        ocr = ScoreOCR()
        mock_pytesseract = mock.MagicMock()
        mock_pytesseract.image_to_string.return_value = "Score: 0"
        frame = _make_frame()
        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            score = ocr.read_score(frame)
        assert score == 0

    def test_handles_large_numbers(self):
        """Large numbers are handled correctly."""
        ocr = ScoreOCR()
        mock_pytesseract = mock.MagicMock()
        mock_pytesseract.image_to_string.return_value = "1234567890"
        frame = _make_frame()
        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            score = ocr.read_score(frame)
        assert score == 1234567890

    def test_handles_comma_separated_numbers(self):
        """Numbers with commas (e.g. '1,234') are parsed correctly."""
        ocr = ScoreOCR()
        mock_pytesseract = mock.MagicMock()
        mock_pytesseract.image_to_string.return_value = "Score: 1,234,567"
        frame = _make_frame()
        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            score = ocr.read_score(frame)
        assert score == 1234567

    def test_handles_period_separated_numbers(self):
        """Numbers with periods as thousand separators (e.g. '1.234') are parsed."""
        ocr = ScoreOCR()
        mock_pytesseract = mock.MagicMock()
        # European-style thousand separator: "1.234.567"
        mock_pytesseract.image_to_string.return_value = "Score: 1.234.567"
        frame = _make_frame()
        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            score = ocr.read_score(frame)
        assert score == 1234567

    def test_grayscale_frame_handled(self):
        """Grayscale (2D) frames are handled without error."""
        ocr = ScoreOCR()
        frame = np.zeros((480, 640), dtype=np.uint8)
        mock_pytesseract = mock.MagicMock()
        mock_pytesseract.image_to_string.return_value = "99"
        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            score = ocr.read_score(frame)
        assert score == 99

    def test_region_out_of_bounds_returns_none(self):
        """If the region extends beyond frame bounds, returns None."""
        ocr = ScoreOCR(region=(600, 400, 200, 100))
        frame = _make_frame(480, 640)  # region extends past right edge
        mock_pytesseract = mock.MagicMock()
        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            score = ocr.read_score(frame)
        assert score is None
        # OCR should not have been called
        mock_pytesseract.image_to_string.assert_not_called()

    def test_negative_numbers_ignored(self):
        """Negative numbers in OCR text are treated as positive (score is unsigned)."""
        ocr = ScoreOCR()
        mock_pytesseract = mock.MagicMock()
        mock_pytesseract.image_to_string.return_value = "Score: -500"
        frame = _make_frame()
        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            score = ocr.read_score(frame)
        assert score == 500


# ---------------------------------------------------------------------------
# Threshold preprocessing
# ---------------------------------------------------------------------------


class TestThresholdPreprocessing:
    """Test binary threshold preprocessing before OCR.

    cv2 is lazy-imported inside ``_run_ocr``.  In CI Docker the real
    cv2 may not be importable (missing libGL.so.1).  All tests that
    verify thresholding behaviour therefore inject a mock cv2 module
    that delegates to numpy for the actual binarisation.
    """

    @staticmethod
    def _make_mock_cv2():
        """Build a mock cv2 module with a working ``threshold``."""
        mock_cv2 = mock.MagicMock()
        mock_cv2.THRESH_BINARY = 0  # value doesn't matter for mock

        def _threshold(src, thresh, maxval, typ):
            binary = np.where(src > thresh, maxval, 0).astype(np.uint8)
            return thresh, binary

        mock_cv2.threshold = _threshold
        return mock_cv2

    def test_threshold_applied_before_ocr(self):
        """_run_ocr applies binary threshold to improve OCR accuracy."""
        ocr = ScoreOCR()
        gray = np.full((30, 100), 180, dtype=np.uint8)
        mock_pytesseract = mock.MagicMock()
        mock_pytesseract.image_to_string.return_value = "1234"
        mock_cv2 = self._make_mock_cv2()

        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract, "cv2": mock_cv2}):
            result = ocr._run_ocr(gray)

        assert result == "1234"
        call_args = mock_pytesseract.image_to_string.call_args
        processed = call_args[0][0]
        unique_vals = set(np.unique(processed))
        assert unique_vals <= {0, 255}, f"Expected binary image, got values: {unique_vals}"

    def test_threshold_converts_above_128_to_white(self):
        """Pixels above 128 become 255 (white) after thresholding."""
        ocr = ScoreOCR()
        gray = np.full((10, 10), 200, dtype=np.uint8)
        mock_pytesseract = mock.MagicMock()
        mock_pytesseract.image_to_string.return_value = ""
        mock_cv2 = self._make_mock_cv2()

        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract, "cv2": mock_cv2}):
            ocr._run_ocr(gray)

        call_args = mock_pytesseract.image_to_string.call_args
        processed = call_args[0][0]
        assert np.all(processed == 255)

    def test_threshold_converts_below_128_to_black(self):
        """Pixels at or below 128 become 0 (black) after thresholding."""
        ocr = ScoreOCR()
        gray = np.full((10, 10), 50, dtype=np.uint8)
        mock_pytesseract = mock.MagicMock()
        mock_pytesseract.image_to_string.return_value = ""
        mock_cv2 = self._make_mock_cv2()

        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract, "cv2": mock_cv2}):
            ocr._run_ocr(gray)

        call_args = mock_pytesseract.image_to_string.call_args
        processed = call_args[0][0]
        assert np.all(processed == 0)

    def test_fallback_when_cv2_unavailable(self):
        """When cv2 is not installed, OCR runs on raw grayscale (no crash)."""
        ocr = ScoreOCR()
        gray = np.full((10, 10), 180, dtype=np.uint8)
        mock_pytesseract = mock.MagicMock()
        mock_pytesseract.image_to_string.return_value = "42"

        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract, "cv2": None}):
            result = ocr._run_ocr(gray)

        assert result == "42"
        call_args = mock_pytesseract.image_to_string.call_args
        processed = call_args[0][0]
        assert np.all(processed == 180)

    def test_threshold_preserves_image_shape(self):
        """Thresholding does not change the image dimensions."""
        ocr = ScoreOCR(region=(10, 20, 100, 30))
        frame = _make_frame(480, 640)
        mock_pytesseract = mock.MagicMock()
        mock_pytesseract.image_to_string.return_value = "99"
        mock_cv2 = self._make_mock_cv2()

        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract, "cv2": mock_cv2}):
            score = ocr.read_score(frame)

        call_args = mock_pytesseract.image_to_string.call_args
        processed = call_args[0][0]
        assert processed.shape == (30, 100)
        assert score == 99

    def test_cv2_threshold_error_falls_back_to_raw(self):
        """If cv2.threshold raises, fall back to raw grayscale."""
        ocr = ScoreOCR()
        gray = np.full((10, 10), 180, dtype=np.uint8)
        mock_pytesseract = mock.MagicMock()
        mock_pytesseract.image_to_string.return_value = "42"
        mock_cv2 = mock.MagicMock()
        mock_cv2.threshold.side_effect = RuntimeError("unexpected cv2 error")

        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract, "cv2": mock_cv2}):
            result = ocr._run_ocr(gray)

        assert result == "42"
        call_args = mock_pytesseract.image_to_string.call_args
        processed = call_args[0][0]
        assert np.all(processed == 180)

    def test_non_uint8_input_normalized_before_threshold(self):
        """Non-uint8 grayscale input is normalized to uint8 before threshold."""
        ocr = ScoreOCR()
        gray = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
        mock_pytesseract = mock.MagicMock()
        mock_pytesseract.image_to_string.return_value = "7"
        mock_cv2 = self._make_mock_cv2()

        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract, "cv2": mock_cv2}):
            result = ocr._run_ocr(gray)

        assert result == "7"
        call_args = mock_pytesseract.image_to_string.call_args
        processed = call_args[0][0]
        assert processed.dtype == np.uint8
        unique_vals = set(np.unique(processed))
        assert unique_vals <= {0, 255}


# ---------------------------------------------------------------------------
# detect_score_region (standalone function)
# ---------------------------------------------------------------------------


class TestDetectScoreRegion:
    """Test heuristic score region auto-detection."""

    def test_detect_score_region_exists_as_function(self):
        """detect_score_region is importable from score_ocr module."""
        from src.platform.score_ocr import detect_score_region

        assert callable(detect_score_region)

    def test_returns_tuple_of_four_ints_when_score_found(self):
        """Returns (x, y, w, h) tuple when a numeric region is found."""
        from src.platform.score_ocr import detect_score_region

        frame = _make_frame(480, 640)
        mock_pytesseract = mock.MagicMock()
        # Simulate OCR finding a score in a candidate region
        mock_pytesseract.image_to_string.return_value = "Score: 1234"

        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            result = detect_score_region(frame)

        assert result is not None
        assert len(result) == 4
        x, y, w, h = result
        assert isinstance(x, int)
        assert isinstance(y, int)
        assert isinstance(w, int)
        assert isinstance(h, int)

    def test_returns_none_when_no_score_found(self):
        """Returns None when no candidate region contains numbers."""
        from src.platform.score_ocr import detect_score_region

        frame = _make_frame(480, 640)
        mock_pytesseract = mock.MagicMock()
        mock_pytesseract.image_to_string.return_value = ""

        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            result = detect_score_region(frame)

        assert result is None

    def test_returns_none_when_pytesseract_missing(self):
        """Returns None gracefully when pytesseract is not installed."""
        from src.platform.score_ocr import detect_score_region

        frame = _make_frame(480, 640)

        with mock.patch.dict("sys.modules", {"pytesseract": None}):
            result = detect_score_region(frame)

        assert result is None

    def test_region_within_frame_bounds(self):
        """Returned region is within the frame boundaries."""
        from src.platform.score_ocr import detect_score_region

        frame = _make_frame(480, 640)
        mock_pytesseract = mock.MagicMock()
        mock_pytesseract.image_to_string.return_value = "42"

        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            result = detect_score_region(frame)

        if result is not None:
            x, y, w, h = result
            assert x >= 0
            assert y >= 0
            assert x + w <= 640
            assert y + h <= 480

    def test_scans_top_strip_of_frame(self):
        """Auto-detection checks the top portion of the frame (scores are typically at top or bottom)."""
        from src.platform.score_ocr import detect_score_region

        frame = _make_frame(480, 640)
        mock_pytesseract = mock.MagicMock()
        # Only return a score for the first call (top strip)
        call_count = 0

        def _side_effect(img):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Score: 999"
            return ""

        mock_pytesseract.image_to_string.side_effect = _side_effect

        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            result = detect_score_region(frame)

        # Should find a region since the first candidate returned digits
        assert result is not None


# ---------------------------------------------------------------------------
# ScoreOCR auto_detect parameter
# ---------------------------------------------------------------------------


class TestScoreOCRAutoDetect:
    """Test ScoreOCR auto_detect parameter for automatic region detection."""

    def test_auto_detect_parameter_accepted(self):
        """ScoreOCR accepts auto_detect parameter."""
        ocr = ScoreOCR(auto_detect=True)
        assert ocr._auto_detect is True

    def test_auto_detect_defaults_to_false(self):
        """auto_detect defaults to False for backward compatibility."""
        ocr = ScoreOCR()
        assert ocr._auto_detect is False

    def test_auto_detect_sets_region_on_first_read(self):
        """When auto_detect=True and region=None, region is set on first read_score()."""
        ocr = ScoreOCR(auto_detect=True)
        frame = _make_frame(480, 640)
        mock_pytesseract = mock.MagicMock()
        mock_pytesseract.image_to_string.return_value = "Score: 500"

        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            with mock.patch(
                "src.platform.score_ocr.detect_score_region",
                return_value=(10, 5, 200, 40),
            ):
                ocr.read_score(frame)

        # Region should now be set
        assert ocr._region == (10, 5, 200, 40)

    def test_auto_detect_skipped_when_region_already_set(self):
        """When region is already provided, auto_detect does not override it."""
        ocr = ScoreOCR(region=(100, 200, 150, 50), auto_detect=True)
        frame = _make_frame(480, 640)
        mock_pytesseract = mock.MagicMock()
        mock_pytesseract.image_to_string.return_value = "42"

        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            with mock.patch("src.platform.score_ocr.detect_score_region") as mock_detect:
                ocr.read_score(frame)

        # detect_score_region should NOT have been called
        mock_detect.assert_not_called()
        assert ocr._region == (100, 200, 150, 50)

    def test_auto_detect_only_runs_once(self):
        """Auto-detection only runs on the first read_score() call, not subsequent ones."""
        ocr = ScoreOCR(auto_detect=True)
        frame = _make_frame(480, 640)
        mock_pytesseract = mock.MagicMock()
        mock_pytesseract.image_to_string.return_value = "100"

        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            with mock.patch(
                "src.platform.score_ocr.detect_score_region",
                return_value=(10, 5, 200, 40),
            ) as mock_detect:
                ocr.read_score(frame)
                ocr.read_score(frame)
                ocr.read_score(frame)

        # Only called once (on first read)
        assert mock_detect.call_count == 1

    def test_auto_detect_falls_back_gracefully_when_detection_fails(self):
        """When auto-detect returns None, OCR uses full frame (no crash)."""
        ocr = ScoreOCR(auto_detect=True)
        frame = _make_frame(480, 640)
        mock_pytesseract = mock.MagicMock()
        mock_pytesseract.image_to_string.return_value = "77"

        with mock.patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            with mock.patch(
                "src.platform.score_ocr.detect_score_region",
                return_value=None,
            ):
                score = ocr.read_score(frame)

        # Should still read the score from full frame
        assert score == 77
        # Region stays None
        assert ocr._region is None


# ---------------------------------------------------------------------------
# Plugin default_score_region metadata
# ---------------------------------------------------------------------------


class TestPluginDefaultScoreRegion:
    """Test that game plugins can export default_score_region metadata."""

    def test_discover_plugins_collects_default_score_region(self):
        """discover_plugins collects default_score_region into PluginEntry.extra."""
        from games import GameRegistry, discover_plugins

        mock_module = mock.MagicMock()
        mock_module.env_class = type("MockEnv", (), {})
        mock_module.loader_class = None
        mock_module.game_name = "test-game"
        mock_module.default_config = "configs/games/test.yaml"
        mock_module.default_weights = ""
        mock_module.default_score_region = (100, 10, 200, 50)
        # Make dir() return the expected attributes
        mock_module.__dir__ = lambda self: [
            "env_class",
            "loader_class",
            "game_name",
            "default_config",
            "default_weights",
            "default_score_region",
        ]

        registry = GameRegistry()
        with mock.patch("games.pkgutil.iter_modules", return_value=[("", "mockgame", True)]):
            with mock.patch("games.importlib.import_module", return_value=mock_module):
                discover_plugins(registry)

        entry = registry.get("mockgame")
        assert entry is not None
        assert entry.extra.get("default_score_region") == (100, 10, 200, 50)

    def test_plugin_entry_extra_contains_score_region_when_present(self):
        """PluginEntry.extra stores default_score_region from plugin module."""
        from games import GameRegistry

        registry = GameRegistry()
        registry.register(
            "test",
            env_class=type("E", (), {}),
            loader_class=None,
            game_name="test",
            default_config="c.yaml",
            default_weights="",
            default_score_region=(540, 402, 200, 80),
        )
        entry = registry.get("test")
        assert entry is not None
        assert entry.extra["default_score_region"] == (540, 402, 200, 80)
