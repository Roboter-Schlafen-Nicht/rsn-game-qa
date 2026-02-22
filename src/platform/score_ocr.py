"""OCR-based score reader for game-agnostic reward signals.

Uses pytesseract (Tesseract OCR) to read numeric score values from
game frames.  Provides throttling to avoid running OCR every frame,
and graceful degradation when pytesseract is not installed.

Parameters
----------
region : tuple[int, int, int, int] or None
    ``(x, y, width, height)`` bounding box to crop before OCR.
    If None, the entire frame is used.
ocr_interval : int
    Run OCR every N calls to :meth:`read_score`.  Between calls,
    the last cached score is returned.  Default ``1`` (every frame).
"""

from __future__ import annotations

import logging
import re

import numpy as np

logger = logging.getLogger(__name__)


class ScoreOCR:
    """Read numeric scores from game frames via OCR.

    Designed as a platform-level utility: no game-specific knowledge.
    The caller (e.g. ``BaseGameEnv._compute_score_reward``) uses
    score deltas as a reward signal.

    Parameters
    ----------
    region : tuple[int, int, int, int] or None
        ``(x, y, width, height)`` crop region.  ``None`` means full frame.
    ocr_interval : int
        Run OCR every *N* calls.  Cached value returned in between.
    """

    # Regex: sequences of digits, optionally separated by commas/periods
    # (thousand separators).  Matches "1,234,567" or "1.234.567" or "12345".
    _NUMBER_PATTERN = re.compile(r"\d[\d,.]*\d|\d")

    def __init__(
        self,
        region: tuple[int, int, int, int] | None = None,
        ocr_interval: int = 1,
    ) -> None:
        self._region = region
        self._interval = max(ocr_interval, 1)
        self._frame_count: int = 0
        self._last_score: int | None = None

    def read_score(self, frame: np.ndarray) -> int | None:
        """Read the numeric score from *frame*.

        Parameters
        ----------
        frame : np.ndarray
            BGR or grayscale game frame.

        Returns
        -------
        int or None
            Detected score, or ``None`` if no number was found
            or OCR is unavailable.
        """
        self._frame_count += 1

        # Throttle: only run OCR on designated frames
        if (self._frame_count - 1) % self._interval != 0:
            return self._last_score

        # Crop to region if specified
        cropped = self._crop(frame)
        if cropped is None:
            return None

        # Convert to grayscale if needed
        gray = cropped.mean(axis=2).astype(np.uint8) if cropped.ndim == 3 else cropped

        # Run OCR
        text = self._run_ocr(gray)
        if text is None:
            return self._last_score

        # Extract largest number from OCR text
        score = self._extract_score(text)
        self._last_score = score
        return score

    def reset(self) -> None:
        """Clear cached state for a new episode."""
        self._frame_count = 0
        self._last_score = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _crop(self, frame: np.ndarray) -> np.ndarray | None:
        """Crop frame to region, or return full frame if no region set.

        Returns None if the region is out of bounds.
        """
        if self._region is None:
            return frame

        x, y, w, h = self._region
        fh = frame.shape[0]
        fw = frame.shape[1]

        # Validate bounds
        if x + w > fw or y + h > fh or x < 0 or y < 0:
            logger.debug(
                "Score region (%d,%d,%d,%d) out of bounds for frame %dx%d",
                x,
                y,
                w,
                h,
                fw,
                fh,
            )
            return None

        return frame[y : y + h, x : x + w]

    def _run_ocr(self, gray: np.ndarray) -> str | None:
        """Run pytesseract OCR on a grayscale image.

        Applies binary thresholding to improve OCR accuracy on game
        frames where score text may have low contrast or anti-aliasing.

        Returns the raw OCR text, or None if pytesseract is not
        available or OCR fails.
        """
        try:
            import pytesseract
        except ImportError:
            logger.debug("pytesseract not installed; ScoreOCR disabled")
            return None

        try:
            import cv2

            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        except ImportError:
            logger.debug("cv2 not available; skipping threshold preprocessing")
            binary = gray

        try:
            return pytesseract.image_to_string(binary)
        except Exception:
            logger.debug("OCR failed", exc_info=True)
            return None

    def _extract_score(self, text: str) -> int | None:
        """Extract the largest integer from OCR text.

        Handles thousand separators (commas and periods).
        Returns None if no numbers are found.
        """
        matches = self._NUMBER_PATTERN.findall(text)
        if not matches:
            return None

        scores: list[int] = []
        for m in matches:
            # Remove thousand separators (commas and periods)
            cleaned = m.replace(",", "").replace(".", "")
            try:
                scores.append(int(cleaned))
            except ValueError:
                continue

        return max(scores) if scores else None
