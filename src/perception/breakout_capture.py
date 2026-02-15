"""Breakout 71 perception helpers — frame capture and object detection.

Convenience functions that bridge the :mod:`src.capture` and
:mod:`src.perception` subsystems, providing a simple two-step pipeline:
grab a frame from the game window, then run YOLO detection on it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.capture import WindowCapture

from .yolo_detector import YoloDetector


def grab_frame(capture: WindowCapture) -> np.ndarray:
    """Capture a single frame from the game window.

    Parameters
    ----------
    capture : WindowCapture
        An initialised :class:`~src.capture.WindowCapture` instance
        bound to the game window.

    Returns
    -------
    np.ndarray
        BGR image as ``(H, W, 3)`` uint8 array.

    Raises
    ------
    RuntimeError
        If the window is not visible or capture fails.
    """
    if not capture.is_window_visible():
        raise RuntimeError("Game window is not visible — cannot capture frame.")
    return capture.capture_frame()


def detect_objects(
    detector: YoloDetector,
    frame: np.ndarray,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> dict[str, Any]:
    """Run YOLO detection on a game frame and return the game state.

    Parameters
    ----------
    detector : YoloDetector
        A loaded :class:`YoloDetector` instance.
    frame : np.ndarray
        BGR image as ``(H, W, 3)`` uint8 array (from :func:`grab_frame`).
    frame_width : int, optional
        Frame width for normalisation.  Defaults to ``frame.shape[1]``.
    frame_height : int, optional
        Frame height for normalisation.  Defaults to ``frame.shape[0]``.

    Returns
    -------
    dict[str, Any]
        Game-state dict as returned by
        :meth:`YoloDetector.detect_to_game_state`:

        - ``"paddle"`` : normalised bbox tuple or None
        - ``"ball"``   : normalised bbox tuple or None
        - ``"bricks"`` : list of normalised bbox tuples
        - ``"powerups"``: list of normalised bbox tuples
        - ``"raw_detections"``: full detection list

    Raises
    ------
    RuntimeError
        If the detector model has not been loaded.
    """
    h, w = frame.shape[:2]
    fw = frame_width if frame_width is not None else w
    fh = frame_height if frame_height is not None else h
    return detector.detect_to_game_state(frame, fw, fh)
