"""YOLO-based object detector for game frame analysis.

Loads a trained YOLOv8 model and extracts structured detections
(bounding boxes, classes, confidences) from game frames.  Designed
to run on Intel Arc A770 GPUs via the XPU backend, with fallback
to CPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from ultralytics import YOLO  # noqa: F401

    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False


class YoloDetector:
    """Runs YOLOv8 inference on game frames and returns structured detections.

    Parameters
    ----------
    weights_path : str or Path
        Path to the trained ``.pt`` weights file.
    device : str
        PyTorch device string.  Default is ``"xpu"`` (Intel Arc).
        Falls back to ``"cpu"`` if XPU is unavailable.
    confidence_threshold : float
        Minimum detection confidence.  Default is 0.5.
    iou_threshold : float
        IoU threshold for NMS.  Default is 0.45.
    img_size : int
        Inference image size (square).  Default is 640.
    classes : list[str], optional
        Expected class names in order.  If None, read from the model.

    Attributes
    ----------
    model : YOLO
        The loaded Ultralytics YOLO model.
    class_names : list[str]
        Class names from the model.

    Raises
    ------
    RuntimeError
        If ultralytics is not installed.
    FileNotFoundError
        If the weights file does not exist.
    """

    # Default class names for Breakout 71 (from session1.md spec)
    BREAKOUT71_CLASSES = ["paddle", "ball", "brick", "powerup", "wall"]

    def __init__(
        self,
        weights_path: str | Path = "weights/best.pt",
        device: str = "xpu",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        classes: Optional[list[str]] = None,
    ) -> None:
        if not _ULTRALYTICS_AVAILABLE:
            raise RuntimeError(
                "ultralytics is required for YoloDetector. "
                "Install it with: pip install ultralytics"
            )

        self.weights_path = Path(weights_path)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.class_names = classes or self.BREAKOUT71_CLASSES

        self.model: Any = None  # Loaded lazily

    def load(self) -> None:
        """Load the YOLO model and move it to the target device.

        Falls back from XPU to CPU if the XPU backend is not available.

        Raises
        ------
        FileNotFoundError
            If ``weights_path`` does not exist.
        """
        raise NotImplementedError("Model loading not yet implemented")

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Run inference on a single frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image as ``(H, W, 3)`` uint8 array.

        Returns
        -------
        list[dict[str, Any]]
            List of detections, each a dict with keys:

            - ``"class_name"`` (str): detected class
            - ``"class_id"`` (int): class index
            - ``"confidence"`` (float): detection confidence
            - ``"bbox_xyxy"`` (tuple[int,int,int,int]): pixel coords
            - ``"bbox_xywh_norm"`` (tuple[float,float,float,float]):
              normalised center-x, center-y, width, height
        """
        raise NotImplementedError("Detection inference not yet implemented")

    def detect_to_game_state(
        self,
        frame: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> dict[str, Any]:
        """Run detection and convert results to game-state dict.

        This is the primary interface used by ``Breakout71Env``.  It
        runs ``detect()`` and groups results by class, extracting the
        paddle position, ball position, and brick grid.

        Parameters
        ----------
        frame : np.ndarray
            BGR game frame.
        frame_width : int
            Frame width in pixels (for normalisation).
        frame_height : int
            Frame height in pixels (for normalisation).

        Returns
        -------
        dict[str, Any]
            Game state with keys:

            - ``"paddle"`` : ``(cx_norm, cy_norm, w_norm, h_norm)`` or None
            - ``"ball"``   : ``(cx_norm, cy_norm, w_norm, h_norm)`` or None
            - ``"bricks"`` : list of ``(cx_norm, cy_norm, w_norm, h_norm)``
            - ``"powerups"``: list of ``(cx_norm, cy_norm, w_norm, h_norm)``
            - ``"raw_detections"``: full detection list
        """
        raise NotImplementedError("Game state extraction not yet implemented")

    def is_loaded(self) -> bool:
        """Check if the model has been loaded.

        Returns
        -------
        bool
            True if the model is loaded and ready for inference.
        """
        return self.model is not None
