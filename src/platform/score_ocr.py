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
auto_detect : bool
    When ``True`` and *region* is ``None``, auto-detect the score
    region on the first :meth:`read_score` call by scanning candidate
    strips of the frame for numeric text.  Default ``False``.
"""

from __future__ import annotations

import logging
import re

import numpy as np

logger = logging.getLogger(__name__)

# Number of horizontal strips to scan during auto-detection.
# Each strip spans the full frame width.
_AUTO_DETECT_STRIPS = 4

# Height fraction of the frame for each candidate strip.
_STRIP_HEIGHT_FRAC = 0.10


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
        auto_detect: bool = False,
    ) -> None:
        self._region = region
        self._interval = max(ocr_interval, 1)
        self._auto_detect = auto_detect
        self._auto_detect_done: bool = region is not None
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
        # Auto-detect region on the very first call if requested
        if self._auto_detect and not self._auto_detect_done:
            self._auto_detect_done = True
            detected = detect_score_region(frame)
            if detected is not None:
                self._region = detected
                logger.info("Auto-detected score region: %s", detected)

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

            # Normalize to uint8 for 2D grayscale inputs to avoid cv2 dtype issues.
            gray_for_cv = gray
            if gray_for_cv.ndim == 2 and gray_for_cv.dtype != np.uint8:
                g_min = float(np.min(gray_for_cv))
                g_max = float(np.max(gray_for_cv))
                if g_max > g_min:
                    scale = 255.0 / (g_max - g_min)
                    gray_for_cv = ((gray_for_cv - g_min) * scale).astype(np.uint8)
                else:
                    gray_for_cv = np.zeros_like(gray_for_cv, dtype=np.uint8)

            _, binary = cv2.threshold(gray_for_cv, 128, 255, cv2.THRESH_BINARY)
        except ImportError:
            logger.debug("cv2 not available; skipping threshold preprocessing")
            binary = gray
        except Exception:
            logger.debug(
                "cv2 threshold preprocessing failed; falling back to raw grayscale",
                exc_info=True,
            )
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


# -----------------------------------------------------------------------
# Standalone auto-detection function
# -----------------------------------------------------------------------

# Regex: sequences of digits, optionally separated by commas/periods.
_NUMBER_RE = re.compile(r"\d[\d,.]*\d|\d")


def detect_score_region(
    frame: np.ndarray,
) -> tuple[int, int, int, int] | None:
    """Heuristically detect a score region by scanning candidate strips.

    Scans horizontal strips at the top and bottom of the frame (where
    scores typically appear) and runs OCR on each strip.  Returns the
    first strip where OCR detects at least one number, as an
    ``(x, y, width, height)`` tuple.

    Parameters
    ----------
    frame : np.ndarray
        BGR or grayscale game frame.

    Returns
    -------
    tuple[int, int, int, int] or None
        ``(x, y, w, h)`` of the detected region, or ``None`` if no
        numeric text was found in any candidate strip.
    """
    try:
        import pytesseract
    except ImportError:
        logger.debug("pytesseract not installed; score region detection disabled")
        return None

    fh, fw = frame.shape[:2]
    strip_h = max(int(fh * _STRIP_HEIGHT_FRAC), 20)

    # Candidate strips: top-left, top-right, bottom-left, bottom-right
    candidates = [
        (0, 0, fw, strip_h),  # full top strip
        (0, fh - strip_h, fw, strip_h),  # full bottom strip
        (0, 0, fw // 2, strip_h),  # top-left half
        (fw // 2, 0, fw - fw // 2, strip_h),  # top-right half
    ]

    for x, y, w, h in candidates:
        crop = frame[y : y + h, x : x + w]
        # Convert to grayscale if needed
        gray = crop.mean(axis=2).astype(np.uint8) if crop.ndim == 3 else crop

        try:
            text = pytesseract.image_to_string(gray)
        except Exception:
            continue

        if _NUMBER_RE.search(text or ""):
            return (x, y, w, h)

    return None
