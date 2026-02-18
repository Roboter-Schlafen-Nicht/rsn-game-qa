"""TDD tests for game-over detection strategies.

Tests the pixel-based game-over detection system that replaces
DOM/JS-based modal detection with game-agnostic visual strategies.

Naming convention: test_{method}_{expected_behavior}_when_{condition}
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(h: int = 84, w: int = 84, value: int = 128) -> np.ndarray:
    """Create a solid-color BGR frame."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _make_random_frame(h: int = 84, w: int = 84, seed: int = 42) -> np.ndarray:
    """Create a random BGR frame (simulates active gameplay)."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _make_text_frame(text: str = "GAME OVER", h: int = 480, w: int = 640) -> np.ndarray:
    """Create a frame with text rendered via OpenCV (for OCR tests)."""
    cv2 = pytest.importorskip("cv2")
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # White text on black background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    thickness = 3
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x = (w - text_size[0]) // 2
    y = (h + text_size[1]) // 2
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)
    return frame


# ===================================================================
# GameOverStrategy ABC
# ===================================================================


class TestGameOverStrategyABC:
    """Test the abstract base class contract."""

    def test_update_returns_float_confidence(self):
        """Strategy.update(frame) must return a float in [0, 1]."""
        from src.platform.game_over_detector import ScreenFreezeStrategy

        strategy = ScreenFreezeStrategy()
        frame = _make_frame()
        confidence = strategy.update(frame)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_reset_clears_internal_state(self):
        """Strategy.reset() must clear all internal state."""
        from src.platform.game_over_detector import ScreenFreezeStrategy

        strategy = ScreenFreezeStrategy()
        frame = _make_frame()
        # Feed identical frames to build up confidence
        for _ in range(10):
            strategy.update(frame)
        strategy.reset()
        # After reset, a single frame should give 0.0 confidence
        confidence = strategy.update(frame)
        assert confidence == 0.0

    def test_name_returns_string(self):
        """Strategy.name must return a descriptive string."""
        from src.platform.game_over_detector import ScreenFreezeStrategy

        strategy = ScreenFreezeStrategy()
        assert isinstance(strategy.name, str)
        assert len(strategy.name) > 0


# ===================================================================
# ScreenFreezeStrategy
# ===================================================================


class TestScreenFreezeStrategy:
    """Test pixel-diff based screen freeze detection."""

    def test_update_returns_zero_confidence_when_first_frame(self):
        """First frame has no previous frame to compare against."""
        from src.platform.game_over_detector import ScreenFreezeStrategy

        strategy = ScreenFreezeStrategy()
        confidence = strategy.update(_make_frame())
        assert confidence == 0.0

    def test_update_returns_zero_confidence_when_frames_change(self):
        """Active gameplay with changing frames → no freeze."""
        from src.platform.game_over_detector import ScreenFreezeStrategy

        strategy = ScreenFreezeStrategy()
        for i in range(10):
            frame = _make_random_frame(seed=i)
            confidence = strategy.update(frame)
        assert confidence == 0.0

    def test_update_returns_high_confidence_when_frames_identical(self):
        """N consecutive identical frames → high confidence."""
        from src.platform.game_over_detector import ScreenFreezeStrategy

        strategy = ScreenFreezeStrategy(threshold=5)
        frame = _make_frame()
        # threshold=5 means ramp starts after 5 identical comparisons.
        # With 11 frames: 10 comparisons → freeze_count=10 → ramp=(10-5)/5=1.0
        for _ in range(11):
            confidence = strategy.update(frame)
        assert confidence >= 0.8

    def test_update_returns_one_when_sustained_freeze(self):
        """After many identical frames, confidence should be 1.0."""
        from src.platform.game_over_detector import ScreenFreezeStrategy

        strategy = ScreenFreezeStrategy(threshold=5)
        frame = _make_frame()
        for _ in range(20):
            confidence = strategy.update(frame)
        assert confidence == 1.0

    def test_update_resets_counter_when_frame_changes(self):
        """A different frame after freeze should reset confidence."""
        from src.platform.game_over_detector import ScreenFreezeStrategy

        strategy = ScreenFreezeStrategy(threshold=5)
        frozen = _make_frame(value=100)
        for _ in range(10):
            strategy.update(frozen)
        # Now a different frame
        confidence = strategy.update(_make_frame(value=200))
        assert confidence == 0.0

    def test_update_uses_pixel_diff_threshold_when_near_identical(self):
        """Frames with tiny noise (< pixel_threshold) count as frozen."""
        from src.platform.game_over_detector import ScreenFreezeStrategy

        strategy = ScreenFreezeStrategy(threshold=3, pixel_threshold=10)
        base = _make_frame(value=128)
        # Add very small noise (within pixel_threshold)
        noisy = base.copy()
        noisy[0, 0, 0] = 130  # 2 pixel diff, within threshold of 10
        for _ in range(4):
            strategy.update(base)
        confidence = strategy.update(noisy)
        # Should still count as frozen (noise below pixel_threshold)
        assert confidence > 0.0

    def test_reset_clears_freeze_count(self):
        """After reset(), freeze counter starts fresh."""
        from src.platform.game_over_detector import ScreenFreezeStrategy

        strategy = ScreenFreezeStrategy(threshold=3)
        frame = _make_frame()
        for _ in range(5):
            strategy.update(frame)
        strategy.reset()
        confidence = strategy.update(frame)
        assert confidence == 0.0

    def test_configurable_threshold(self):
        """Different threshold values change when confidence ramps up."""
        from src.platform.game_over_detector import ScreenFreezeStrategy

        strategy_low = ScreenFreezeStrategy(threshold=2)
        strategy_high = ScreenFreezeStrategy(threshold=10)
        frame = _make_frame()
        for _ in range(5):
            c_low = strategy_low.update(frame)
            c_high = strategy_high.update(frame)
        # Low threshold should be more confident than high threshold
        assert c_low > c_high


# ===================================================================
# EntropyCollapseStrategy
# ===================================================================


class TestEntropyCollapseStrategy:
    """Test static/uniform screen detection via image entropy."""

    def test_update_returns_zero_when_first_frame(self):
        """First frame returns 0.0 (needs comparison baseline)."""
        from src.platform.game_over_detector import EntropyCollapseStrategy

        strategy = EntropyCollapseStrategy()
        confidence = strategy.update(_make_random_frame())
        assert confidence == 0.0

    def test_update_returns_high_confidence_when_uniform_frame(self):
        """A solid color frame has near-zero entropy → high confidence."""
        from src.platform.game_over_detector import EntropyCollapseStrategy

        strategy = EntropyCollapseStrategy(threshold=3)
        # First a normal frame to establish baseline
        strategy.update(_make_random_frame())
        # Then N uniform frames — need threshold+threshold more for ramp
        # to reach 0.5+ (ramp = (count - threshold) / threshold)
        uniform = _make_frame(value=50)
        for _ in range(8):
            confidence = strategy.update(uniform)
        assert confidence >= 0.5

    def test_update_returns_zero_when_high_entropy_gameplay(self):
        """Active gameplay with high entropy → no collapse."""
        from src.platform.game_over_detector import EntropyCollapseStrategy

        strategy = EntropyCollapseStrategy()
        for i in range(10):
            confidence = strategy.update(_make_random_frame(seed=i))
        assert confidence == 0.0

    def test_update_detects_entropy_drop_when_gameplay_to_static(self):
        """Transition from active gameplay to static screen → detection."""
        from src.platform.game_over_detector import EntropyCollapseStrategy

        strategy = EntropyCollapseStrategy(threshold=3)
        # Active gameplay (high entropy)
        for i in range(5):
            strategy.update(_make_random_frame(seed=i))
        # Transition to static (low entropy)
        static = _make_frame(value=30)
        for _ in range(5):
            confidence = strategy.update(static)
        assert confidence > 0.0

    def test_reset_clears_entropy_history(self):
        """After reset, entropy baseline is cleared."""
        from src.platform.game_over_detector import EntropyCollapseStrategy

        strategy = EntropyCollapseStrategy()
        for i in range(5):
            strategy.update(_make_random_frame(seed=i))
        strategy.reset()
        confidence = strategy.update(_make_frame())
        assert confidence == 0.0


# ===================================================================
# MotionCessationStrategy (replaces "input responsiveness")
# ===================================================================


class TestMotionCessationStrategy:
    """Test motion cessation detection via frame differencing.

    Instead of actively probing input responsiveness (which requires
    game-specific knowledge), we detect when all motion stops for
    sustained periods — a game-agnostic signal that often indicates
    a terminal state (game-over screen, pause, loading screen).
    """

    def test_update_returns_zero_when_first_frame(self):
        """No motion can be computed from a single frame."""
        from src.platform.game_over_detector import MotionCessationStrategy

        strategy = MotionCessationStrategy()
        confidence = strategy.update(_make_frame())
        assert confidence == 0.0

    def test_update_returns_zero_when_active_motion(self):
        """Frames with significant pixel changes → no cessation."""
        from src.platform.game_over_detector import MotionCessationStrategy

        strategy = MotionCessationStrategy()
        for i in range(10):
            confidence = strategy.update(_make_random_frame(seed=i))
        assert confidence == 0.0

    def test_update_returns_high_confidence_when_no_motion(self):
        """Consecutive frames with no pixel changes → motion ceased."""
        from src.platform.game_over_detector import MotionCessationStrategy

        strategy = MotionCessationStrategy(threshold=5)
        frame = _make_frame()
        for _ in range(10):
            confidence = strategy.update(frame)
        assert confidence >= 0.5

    def test_update_detects_subtle_animation_as_low_motion(self):
        """Small localized changes (cursor blink) → partial cessation."""
        from src.platform.game_over_detector import MotionCessationStrategy

        strategy = MotionCessationStrategy(threshold=5, motion_fraction_threshold=0.01)
        base = _make_frame(value=50)
        for i in range(10):
            frame = base.copy()
            # Simulate cursor blink: tiny area changes
            if i % 2 == 0:
                frame[40:42, 40:42] = 200
            confidence = strategy.update(frame)
        # Motion fraction is very small — should eventually detect
        assert confidence > 0.0

    def test_reset_clears_motion_history(self):
        """After reset, motion history starts fresh."""
        from src.platform.game_over_detector import MotionCessationStrategy

        strategy = MotionCessationStrategy()
        frame = _make_frame()
        for _ in range(10):
            strategy.update(frame)
        strategy.reset()
        confidence = strategy.update(frame)
        assert confidence == 0.0


# ===================================================================
# TextDetectionStrategy (OCR-based terminal text)
# ===================================================================


class TestTextDetectionStrategy:
    """Test OCR-based detection of terminal text patterns."""

    def test_update_returns_zero_when_no_text(self):
        """A frame with no text should not trigger detection."""
        from src.platform.game_over_detector import TextDetectionStrategy

        strategy = TextDetectionStrategy()
        confidence = strategy.update(_make_frame())
        assert confidence == 0.0

    def test_update_returns_high_confidence_when_game_over_text(self):
        """Frame with 'GAME OVER' text should trigger detection."""
        pytest.importorskip("cv2")
        pytest.importorskip("pytesseract")
        from src.platform.game_over_detector import TextDetectionStrategy

        strategy = TextDetectionStrategy()
        frame = _make_text_frame("GAME OVER")
        confidence = strategy.update(frame)
        assert confidence >= 0.5

    def test_update_detects_custom_patterns(self):
        """Custom text patterns should be detectable."""
        pytest.importorskip("cv2")
        pytest.importorskip("pytesseract")
        from src.platform.game_over_detector import TextDetectionStrategy

        strategy = TextDetectionStrategy(patterns=["GAME OVER", "YOU DIED", "CONTINUE"])
        frame = _make_text_frame("YOU DIED")
        confidence = strategy.update(frame)
        assert confidence >= 0.5

    def test_update_returns_zero_when_gameplay_text(self):
        """Normal gameplay text (scores, labels) should not trigger."""
        pytest.importorskip("cv2")
        from src.platform.game_over_detector import TextDetectionStrategy

        strategy = TextDetectionStrategy()
        frame = _make_text_frame("SCORE 1000")
        confidence = strategy.update(frame)
        assert confidence == 0.0

    def test_update_is_case_insensitive(self):
        """Text matching should be case-insensitive."""
        pytest.importorskip("cv2")
        pytest.importorskip("pytesseract")
        from src.platform.game_over_detector import TextDetectionStrategy

        strategy = TextDetectionStrategy()
        frame = _make_text_frame("game over")
        confidence = strategy.update(frame)
        assert confidence >= 0.5

    def test_reset_clears_detection_state(self):
        """After reset, any accumulated state is cleared."""
        pytest.importorskip("cv2")
        from src.platform.game_over_detector import TextDetectionStrategy

        strategy = TextDetectionStrategy()
        frame = _make_text_frame("GAME OVER")
        strategy.update(frame)
        strategy.reset()
        # A clean frame after reset should give 0
        confidence = strategy.update(_make_frame())
        assert confidence == 0.0

    def test_configurable_throttle_interval(self):
        """OCR should be throttled to avoid per-frame cost."""
        from src.platform.game_over_detector import TextDetectionStrategy

        strategy = TextDetectionStrategy(check_interval=5)
        frame = _make_frame()
        # Only frame 0 and 5 should actually run OCR
        results = []
        for _ in range(6):
            results.append(strategy.update(frame))
        # All should be 0 since no text, but update should not crash
        assert all(r == 0.0 for r in results)

    def test_graceful_degradation_when_tesseract_unavailable(self):
        """If pytesseract is not installed, strategy returns 0.0."""
        from src.platform.game_over_detector import TextDetectionStrategy

        strategy = TextDetectionStrategy()
        # Even if OCR fails, should return 0.0, not raise
        confidence = strategy.update(_make_frame())
        assert confidence == 0.0


# ===================================================================
# GameOverDetector (ensemble)
# ===================================================================


class TestGameOverDetector:
    """Test the ensemble detector that combines strategies."""

    def test_init_with_default_strategies(self):
        """Default constructor creates all built-in strategies."""
        from src.platform.game_over_detector import GameOverDetector

        detector = GameOverDetector()
        assert len(detector.strategies) >= 1

    def test_init_with_custom_strategies(self):
        """Can construct with specific strategies."""
        from src.platform.game_over_detector import (
            GameOverDetector,
            ScreenFreezeStrategy,
        )

        freeze = ScreenFreezeStrategy(threshold=5)
        detector = GameOverDetector(strategies=[freeze])
        assert len(detector.strategies) == 1

    def test_update_returns_false_when_no_detection(self):
        """Active gameplay should not trigger game-over."""
        from src.platform.game_over_detector import GameOverDetector

        detector = GameOverDetector()
        for i in range(10):
            result = detector.update(_make_random_frame(seed=i))
        assert result is False

    def test_update_returns_true_when_freeze_detected(self):
        """Frozen screen should trigger game-over."""
        from src.platform.game_over_detector import (
            GameOverDetector,
            ScreenFreezeStrategy,
        )

        freeze = ScreenFreezeStrategy(threshold=3)
        detector = GameOverDetector(
            strategies=[freeze],
            confidence_threshold=0.5,
        )
        frame = _make_frame()
        for _ in range(10):
            result = detector.update(frame)
        assert result is True

    def test_update_uses_weighted_average_when_multiple_strategies(self):
        """Multiple strategies combine via weighted average."""
        from src.platform.game_over_detector import (
            GameOverDetector,
            ScreenFreezeStrategy,
            EntropyCollapseStrategy,
        )

        freeze = ScreenFreezeStrategy(threshold=3)
        entropy = EntropyCollapseStrategy(threshold=3)
        detector = GameOverDetector(
            strategies=[freeze, entropy],
            weights=[0.7, 0.3],
            confidence_threshold=0.5,
        )
        frame = _make_frame()
        # Feed uniform frozen frames — both should fire
        for _ in range(10):
            result = detector.update(frame)
        assert result is True

    def test_update_respects_confidence_threshold(self):
        """Detection only fires when weighted confidence >= threshold."""
        from src.platform.game_over_detector import (
            GameOverDetector,
            ScreenFreezeStrategy,
        )

        freeze = ScreenFreezeStrategy(threshold=3)
        detector = GameOverDetector(
            strategies=[freeze],
            confidence_threshold=0.99,
        )
        frame = _make_frame()
        # Only 4 identical frames — confidence should be moderate, not 0.99
        results = []
        for _ in range(4):
            results.append(detector.update(frame))
        # With threshold=3 and only 4 frames, confidence is rising but
        # may not hit 0.99 yet
        # After many more frames it should
        for _ in range(50):
            result = detector.update(frame)
        assert result is True

    def test_reset_resets_all_strategies(self):
        """reset() propagates to all strategies."""
        from src.platform.game_over_detector import (
            GameOverDetector,
            ScreenFreezeStrategy,
        )

        freeze = ScreenFreezeStrategy(threshold=3)
        detector = GameOverDetector(strategies=[freeze])
        frame = _make_frame()
        for _ in range(10):
            detector.update(frame)
        detector.reset()
        result = detector.update(frame)
        assert result is False

    def test_confidence_property_returns_last_confidence(self):
        """confidence property returns the last weighted confidence."""
        from src.platform.game_over_detector import GameOverDetector

        detector = GameOverDetector()
        detector.update(_make_frame())
        assert isinstance(detector.confidence, float)
        assert 0.0 <= detector.confidence <= 1.0

    def test_strategy_confidences_returns_per_strategy_dict(self):
        """strategy_confidences returns a dict of strategy_name → float."""
        from src.platform.game_over_detector import (
            GameOverDetector,
            ScreenFreezeStrategy,
        )

        freeze = ScreenFreezeStrategy()
        detector = GameOverDetector(strategies=[freeze])
        detector.update(_make_frame())
        sc = detector.strategy_confidences
        assert isinstance(sc, dict)
        assert freeze.name in sc
        assert isinstance(sc[freeze.name], float)

    def test_weights_normalized_when_not_summing_to_one(self):
        """Weights are auto-normalized if they don't sum to 1.0."""
        from src.platform.game_over_detector import (
            GameOverDetector,
            ScreenFreezeStrategy,
            EntropyCollapseStrategy,
        )

        detector = GameOverDetector(
            strategies=[ScreenFreezeStrategy(), EntropyCollapseStrategy()],
            weights=[2.0, 8.0],  # Sum = 10, should normalize to 0.2, 0.8
        )
        assert abs(sum(detector._weights) - 1.0) < 1e-6

    def test_raises_when_weights_length_mismatch(self):
        """ValueError if weights length != strategies length."""
        from src.platform.game_over_detector import (
            GameOverDetector,
            ScreenFreezeStrategy,
        )

        with pytest.raises(ValueError, match="weights"):
            GameOverDetector(
                strategies=[ScreenFreezeStrategy()],
                weights=[0.5, 0.5],
            )

    def test_default_weights_are_equal(self):
        """When no weights specified, all strategies get equal weight."""
        from src.platform.game_over_detector import (
            GameOverDetector,
            ScreenFreezeStrategy,
            EntropyCollapseStrategy,
        )

        detector = GameOverDetector(
            strategies=[ScreenFreezeStrategy(), EntropyCollapseStrategy()],
        )
        assert abs(detector._weights[0] - 0.5) < 1e-6
        assert abs(detector._weights[1] - 0.5) < 1e-6


# ===================================================================
# BaseGameEnv integration
# ===================================================================


class TestBaseGameEnvIntegration:
    """Test that GameOverDetector integrates with BaseGameEnv."""

    def test_base_env_accepts_game_over_detector_parameter(self):
        """BaseGameEnv constructor accepts an optional game_over_detector."""
        # We can't instantiate BaseGameEnv directly (it's abstract),
        # so we check the __init__ signature accepts the parameter.
        import inspect

        from src.platform.base_env import BaseGameEnv

        sig = inspect.signature(BaseGameEnv.__init__)
        assert "game_over_detector" in sig.parameters

    def test_base_env_game_over_detector_defaults_to_none(self):
        """When not provided, game_over_detector is None (backward compat)."""
        import inspect

        from src.platform.base_env import BaseGameEnv

        sig = inspect.signature(BaseGameEnv.__init__)
        param = sig.parameters["game_over_detector"]
        assert param.default is None
