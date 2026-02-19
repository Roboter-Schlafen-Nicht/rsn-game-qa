"""Pixel-based game-over detection strategies.

Provides game-agnostic strategies for detecting terminal game states
from raw frames, without requiring DOM access or JS injection.  This
enables the platform to work with **any** game — not just web games
with inspectable DOM.

Architecture
------------
- **GameOverStrategy** (ABC): Each strategy analyses a sequence of
  frames and returns a confidence score in [0, 1].
- **GameOverDetector** (ensemble): Combines multiple strategies via
  weighted average.  When the combined confidence exceeds a configurable
  threshold, the detector signals game-over.

Built-in strategies
-------------------
- ``ScreenFreezeStrategy``: Detects when frames are identical (or
  near-identical) for N consecutive steps.
- ``EntropyCollapseStrategy``: Detects when image entropy drops below
  a threshold for N consecutive frames (solid color / loading screen).
- ``MotionCessationStrategy``: Detects when the fraction of changed
  pixels drops to near-zero for N consecutive frames.
- ``TextDetectionStrategy``: Uses OCR (pytesseract) to detect terminal
  text patterns like "GAME OVER", "YOU DIED", "CONTINUE?".

Integration
-----------
Injected into ``BaseGameEnv`` via the ``game_over_detector`` constructor
parameter.  When configured, the detector's ``update(frame)`` is called
every step.  If it returns True, the episode terminates.
"""

from __future__ import annotations

import abc
import logging

import numpy as np

logger = logging.getLogger(__name__)


# ===================================================================
# Abstract base
# ===================================================================


class GameOverStrategy(abc.ABC):
    """Abstract base for a single game-over detection strategy.

    Each strategy maintains internal state across frames within an
    episode.  Call ``reset()`` at the start of each episode.

    Subclasses must implement:

    - ``update(frame)`` — process one frame, return confidence [0, 1].
    - ``reset()`` — clear per-episode state.
    - ``name`` (property) — human-readable strategy name.
    """

    @abc.abstractmethod
    def update(self, frame: np.ndarray) -> float:
        """Process a new frame and return detection confidence.

        Parameters
        ----------
        frame : np.ndarray
            BGR uint8 image, shape ``(H, W, 3)``.

        Returns
        -------
        float
            Confidence that the game is over, in [0.0, 1.0].
        """

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear all per-episode state."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""


# ===================================================================
# ScreenFreezeStrategy
# ===================================================================


class ScreenFreezeStrategy(GameOverStrategy):
    """Detect game-over by consecutive identical (or near-identical) frames.

    Parameters
    ----------
    threshold : int
        Number of consecutive frozen frames before confidence starts
        rising.  After ``threshold`` frames, confidence ramps linearly
        from 0 to 1 over the next ``threshold`` frames, reaching 1.0
        at ``2 * threshold`` consecutive frozen frames.
    pixel_threshold : int
        Maximum per-pixel absolute difference to consider frames as
        "identical".  This handles minor compression artefacts or
        cursor blinks.  When 0, frames must be exactly equal.
    """

    def __init__(
        self,
        threshold: int = 10,
        pixel_threshold: int = 0,
    ) -> None:
        self._threshold = threshold
        self._pixel_threshold = pixel_threshold
        self._prev_frame: np.ndarray | None = None
        self._freeze_count: int = 0

    def update(self, frame: np.ndarray) -> float:
        """Return freeze confidence based on consecutive identical frames."""
        if self._prev_frame is None:
            self._prev_frame = frame.copy()
            return 0.0

        if self._frames_match(frame, self._prev_frame):
            self._freeze_count += 1
        else:
            self._freeze_count = 0
            self._prev_frame = frame.copy()
            return 0.0

        self._prev_frame = frame.copy()

        if self._freeze_count < self._threshold:
            return 0.0

        # Ramp from 0 to 1 over [threshold, 2*threshold]
        ramp = (self._freeze_count - self._threshold) / max(self._threshold, 1)
        return min(ramp, 1.0)

    def _frames_match(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Return True if frames are identical within pixel_threshold."""
        if self._pixel_threshold == 0:
            return np.array_equal(a, b)
        diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
        return bool(np.max(diff) <= self._pixel_threshold)

    def reset(self) -> None:
        """Clear freeze state."""
        self._prev_frame = None
        self._freeze_count = 0

    @property
    def name(self) -> str:
        return "screen_freeze"


# ===================================================================
# EntropyCollapseStrategy
# ===================================================================


class EntropyCollapseStrategy(GameOverStrategy):
    """Detect game-over by sustained low image entropy.

    Image entropy measures the information content of a frame.  Active
    gameplay produces high entropy (many distinct pixel values); a
    solid-colour or loading screen has near-zero entropy.

    The strategy fires when entropy stays below ``entropy_threshold``
    for ``threshold`` consecutive frames, after an initial baseline
    period.

    Parameters
    ----------
    threshold : int
        Consecutive low-entropy frames before confidence rises.
    entropy_threshold : float
        Frames with entropy below this are "collapsed".  Shannon
        entropy of a uniform 8-bit image is ~8.0; a solid colour
        is 0.0.  Default 2.0 catches solid and near-solid screens.
    baseline_frames : int
        Number of initial frames to observe before activating
        detection (avoids false positives during startup).
    """

    def __init__(
        self,
        threshold: int = 5,
        entropy_threshold: float = 2.0,
        baseline_frames: int = 1,
    ) -> None:
        self._threshold = threshold
        self._entropy_threshold = entropy_threshold
        self._baseline_frames = baseline_frames
        self._frame_count: int = 0
        self._low_entropy_count: int = 0

    def update(self, frame: np.ndarray) -> float:
        """Return collapse confidence based on sustained low entropy."""
        self._frame_count += 1

        if self._frame_count <= self._baseline_frames:
            return 0.0

        entropy = self._compute_entropy(frame)

        if entropy < self._entropy_threshold:
            self._low_entropy_count += 1
        else:
            self._low_entropy_count = 0
            return 0.0

        if self._low_entropy_count < self._threshold:
            return 0.0

        # Ramp from 0 to 1 over [threshold, 2*threshold]
        ramp = (self._low_entropy_count - self._threshold) / max(self._threshold, 1)
        return min(ramp, 1.0)

    @staticmethod
    def _compute_entropy(frame: np.ndarray) -> float:
        """Compute Shannon entropy of a grayscale version of the frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR uint8 image, shape ``(H, W, 3)``.

        Returns
        -------
        float
            Shannon entropy in bits (0.0 for solid colour, ~8.0 max).
        """
        # Convert to grayscale via simple average (avoids cv2 dependency)
        gray = frame.mean(axis=2).astype(np.uint8)
        # Histogram of pixel values
        hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
        hist = hist[hist > 0]  # Remove zero bins
        probs = hist / hist.sum()
        return float(-np.sum(probs * np.log2(probs)))

    def reset(self) -> None:
        """Clear entropy history."""
        self._frame_count = 0
        self._low_entropy_count = 0

    @property
    def name(self) -> str:
        return "entropy_collapse"


# ===================================================================
# MotionCessationStrategy
# ===================================================================


class MotionCessationStrategy(GameOverStrategy):
    """Detect game-over by sustained absence of inter-frame motion.

    Uses ``cv2.absdiff`` (or numpy fallback) to compute the fraction
    of pixels that changed between consecutive frames.  When the
    motion fraction stays below ``motion_fraction_threshold`` for
    ``threshold`` consecutive frames, confidence rises.

    Parameters
    ----------
    threshold : int
        Consecutive low-motion frames before confidence rises.
    pixel_change_threshold : int
        Minimum per-pixel absolute difference to count as "changed".
    motion_fraction_threshold : float
        Maximum fraction of changed pixels to count as "no motion".
        Default 0.005 (0.5% of pixels).
    """

    def __init__(
        self,
        threshold: int = 10,
        pixel_change_threshold: int = 15,
        motion_fraction_threshold: float = 0.005,
    ) -> None:
        self._threshold = threshold
        self._pixel_change_threshold = pixel_change_threshold
        self._motion_fraction_threshold = motion_fraction_threshold
        self._prev_frame: np.ndarray | None = None
        self._low_motion_count: int = 0

    def update(self, frame: np.ndarray) -> float:
        """Return motion cessation confidence."""
        if self._prev_frame is None:
            self._prev_frame = frame.copy()
            return 0.0

        motion_fraction = self._compute_motion_fraction(frame, self._prev_frame)
        self._prev_frame = frame.copy()

        if motion_fraction <= self._motion_fraction_threshold:
            self._low_motion_count += 1
        else:
            self._low_motion_count = 0
            return 0.0

        if self._low_motion_count < self._threshold:
            return 0.0

        # Ramp from 0 to 1
        ramp = (self._low_motion_count - self._threshold) / max(self._threshold, 1)
        return min(ramp, 1.0)

    def _compute_motion_fraction(self, a: np.ndarray, b: np.ndarray) -> float:
        """Fraction of pixels that changed above the threshold."""
        # Convert to grayscale for efficiency
        gray_a = a.mean(axis=2).astype(np.uint8)
        gray_b = b.mean(axis=2).astype(np.uint8)
        diff = np.abs(gray_a.astype(np.int16) - gray_b.astype(np.int16))
        changed = np.count_nonzero(diff > self._pixel_change_threshold)
        total = gray_a.size
        return changed / total if total > 0 else 0.0

    def reset(self) -> None:
        """Clear motion history."""
        self._prev_frame = None
        self._low_motion_count = 0

    @property
    def name(self) -> str:
        return "motion_cessation"


# ===================================================================
# TextDetectionStrategy
# ===================================================================


class TextDetectionStrategy(GameOverStrategy):
    """Detect game-over by OCR recognition of terminal text patterns.

    Uses pytesseract (Tesseract OCR) to read text from frames and
    match against configurable patterns like "GAME OVER", "YOU DIED",
    "CONTINUE?".

    OCR is expensive, so it's throttled to run every ``check_interval``
    frames.  Between checks, the last result is cached.

    Parameters
    ----------
    patterns : list[str] or None
        Text patterns to search for (case-insensitive substring match).
        Defaults to common terminal patterns.
    check_interval : int
        Run OCR every N frames (default 5).
    """

    _DEFAULT_PATTERNS = [
        "game over",
        "you died",
        "you lose",
        "continue",
        "try again",
        "restart",
        "play again",
    ]

    def __init__(
        self,
        patterns: list[str] | None = None,
        check_interval: int = 5,
    ) -> None:
        self._patterns = [p.lower() for p in (patterns or self._DEFAULT_PATTERNS)]
        self._check_interval = max(check_interval, 1)
        self._frame_count: int = 0
        self._last_confidence: float = 0.0
        self._consecutive_detections: int = 0

    def update(self, frame: np.ndarray) -> float:
        """Return text detection confidence."""
        self._frame_count += 1

        # Throttle OCR — only run every check_interval frames
        if (self._frame_count - 1) % self._check_interval != 0:
            return self._last_confidence

        detected = self._run_ocr(frame)
        if detected:
            self._consecutive_detections += 1
            # Confidence ramps with consecutive detections
            self._last_confidence = min(self._consecutive_detections * 0.5, 1.0)
        else:
            self._consecutive_detections = 0
            self._last_confidence = 0.0

        return self._last_confidence

    def _run_ocr(self, frame: np.ndarray) -> bool:
        """Run OCR and check for pattern matches.

        Returns True if any pattern is found in the OCR text.
        Gracefully degrades to False if pytesseract is not installed.
        """
        try:
            import pytesseract
        except ImportError:
            logger.debug("pytesseract not installed; TextDetectionStrategy disabled")
            return False

        try:
            # Convert BGR to grayscale for better OCR
            gray = frame.mean(axis=2).astype(np.uint8)
            text = pytesseract.image_to_string(gray).lower()
            return any(pattern in text for pattern in self._patterns)
        except Exception:
            logger.debug("OCR failed", exc_info=True)
            return False

    def reset(self) -> None:
        """Clear detection state."""
        self._frame_count = 0
        self._last_confidence = 0.0
        self._consecutive_detections = 0

    @property
    def name(self) -> str:
        return "text_detection"


# ===================================================================
# GameOverDetector (ensemble)
# ===================================================================


class GameOverDetector:
    """Ensemble game-over detector combining multiple strategies.

    Parameters
    ----------
    strategies : list[GameOverStrategy] or None
        Strategies to use.  Defaults to ``[ScreenFreezeStrategy(),
        MotionCessationStrategy()]`` (lightweight, no external deps).
    weights : list[float] or None
        Per-strategy weights for the weighted average.  Auto-normalised
        to sum to 1.0.  Defaults to equal weights.
    confidence_threshold : float
        Weighted confidence must exceed this to signal game-over.
    """

    def __init__(
        self,
        strategies: list[GameOverStrategy] | None = None,
        weights: list[float] | None = None,
        confidence_threshold: float = 0.6,
    ) -> None:
        if strategies is None:
            strategies = [
                ScreenFreezeStrategy(),
                MotionCessationStrategy(),
            ]
        if len(strategies) == 0:
            raise ValueError("strategies list must contain at least one strategy")
        self._strategies = strategies

        if weights is not None:
            if len(weights) != len(strategies):
                raise ValueError(
                    f"len(weights)={len(weights)} != len(strategies)={len(strategies)}"
                )
            total = sum(weights)
            if total == 0.0:
                raise ValueError("weights must contain at least one non-zero value")
            self._weights = [w / total for w in weights]
        else:
            n = len(strategies)
            self._weights = [1.0 / n] * n

        self._confidence_threshold = confidence_threshold
        self._last_confidence: float = 0.0
        self._last_per_strategy: dict[str, float] = {}

    @property
    def strategies(self) -> list[GameOverStrategy]:
        """Return the list of active strategies."""
        return self._strategies

    @property
    def confidence(self) -> float:
        """Return the last computed weighted confidence."""
        return self._last_confidence

    @property
    def strategy_confidences(self) -> dict[str, float]:
        """Return per-strategy confidence from the last update."""
        return dict(self._last_per_strategy)

    def get_confidence(self) -> dict[str, float]:
        """Return per-strategy confidence dict for info reporting.

        Convenience method used by ``BaseGameEnv.step()`` to populate
        the ``game_over_detector`` key in the info dict.

        Returns
        -------
        dict[str, float]
            Per-strategy confidence values from the last ``update()``.
        """
        return self.strategy_confidences

    def update(self, frame: np.ndarray) -> bool:
        """Process a frame and return whether game-over is detected.

        Parameters
        ----------
        frame : np.ndarray
            BGR uint8 image, shape ``(H, W, 3)``.

        Returns
        -------
        bool
            True if the weighted confidence exceeds the threshold.
        """
        per_strategy: dict[str, float] = {}
        weighted_sum = 0.0

        for strategy, weight in zip(self._strategies, self._weights, strict=False):
            conf = strategy.update(frame)
            per_strategy[strategy.name] = conf
            weighted_sum += conf * weight

        self._last_confidence = weighted_sum
        self._last_per_strategy = per_strategy

        return weighted_sum >= self._confidence_threshold

    def reset(self) -> None:
        """Reset all strategies (call at episode start)."""
        for strategy in self._strategies:
            strategy.reset()
        self._last_confidence = 0.0
        self._last_per_strategy = {}
