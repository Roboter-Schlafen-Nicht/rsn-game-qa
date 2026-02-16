"""YOLO-based object detector for game frame analysis.

Loads a trained YOLOv8 model and extracts structured detections
(bounding boxes, classes, confidences) from game frames.  Uses
``resolve_device()`` to auto-detect the best available device
(XPU > CUDA > CPU) by default.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from ultralytics import YOLO

    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    YOLO = None  # type: ignore[assignment,misc]
    _ULTRALYTICS_AVAILABLE = False

logger = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> str:
    """Resolve the best available compute device for inference.

    Parameters
    ----------
    requested : str
        ``"auto"`` (try xpu > cuda > cpu), ``"xpu"``, ``"cuda"``,
        or ``"cpu"``.

    Returns
    -------
    str
        Device string suitable for ``YoloDetector(device=...)``.
    """
    if requested != "auto":
        return requested

    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


class YoloDetector:
    """Runs YOLOv8 inference on game frames and returns structured detections.

    Parameters
    ----------
    weights_path : str or Path
        Path to the trained ``.pt`` weights file.
    device : str
        ``"auto"`` to auto-detect (xpu > cuda > cpu), or an explicit
        device string (``"xpu"``, ``"cuda"``, ``"cpu"``).  Default
        ``"auto"``.
    confidence_threshold : float
        Minimum detection confidence.  Default is 0.5.
    iou_threshold : float
        IoU threshold for NMS.  Default is 0.45.
    img_size : int
        Inference image size (square).  Default is 640.
    classes : list[str], optional
        Expected class names in order.  If None, read from the model
        after loading, or fall back to ``BREAKOUT71_CLASSES``.

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
        device: str = "auto",
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
        self.device = resolve_device(device)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self._user_classes = classes  # Store user override separately
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
        if not self.weights_path.exists():
            raise FileNotFoundError(f"YOLO weights file not found: {self.weights_path}")

        self.model = YOLO(str(self.weights_path))

        # Attempt to move model to the requested device, fall back to CPU
        try:
            self.model.to(self.device)
            logger.info("YOLO model loaded on device: %s", self.device)
        except Exception:
            logger.warning("Device '%s' unavailable, falling back to CPU", self.device)
            self.device = "cpu"
            try:
                self.model.to("cpu")
                logger.info("YOLO model loaded on device: cpu (fallback)")
            except Exception as cpu_exc:
                self.model = None
                raise RuntimeError(
                    "Failed to move YOLO model to any device (requested and CPU)."
                ) from cpu_exc

        # Read class names from the model if the user didn't override
        if self._user_classes is None and hasattr(self.model, "names"):
            model_names = self.model.names
            if isinstance(model_names, dict):
                self.class_names = list(model_names.values())
            elif isinstance(model_names, (list, tuple)):
                self.class_names = list(model_names)

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

        Raises
        ------
        RuntimeError
            If the model has not been loaded yet.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() before detect().")

        results = self.model(
            frame,
            imgsz=self.img_size,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections: list[dict[str, Any]] = []
        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        h, w = frame.shape[:2]

        for box in boxes:
            # Extract coordinates — boxes are tensors, move to CPU
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

            cls_id = int(box.cls[0].cpu().item())
            conf = float(box.conf[0].cpu().item())

            # Resolve class name
            if cls_id < len(self.class_names):
                cls_name = self.class_names[cls_id]
            else:
                cls_name = f"class_{cls_id}"

            # Normalised center-x, center-y, width, height
            cx_norm = ((x1 + x2) / 2.0) / w if w > 0 else 0.0
            cy_norm = ((y1 + y2) / 2.0) / h if h > 0 else 0.0
            w_norm = (x2 - x1) / w if w > 0 else 0.0
            h_norm = (y2 - y1) / h if h > 0 else 0.0

            detections.append(
                {
                    "class_name": cls_name,
                    "class_id": cls_id,
                    "confidence": conf,
                    "bbox_xyxy": (x1, y1, x2, y2),
                    "bbox_xywh_norm": (cx_norm, cy_norm, w_norm, h_norm),
                }
            )

        return detections

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
        raw = self.detect(frame)

        paddle: tuple[float, float, float, float] | None = None
        ball: tuple[float, float, float, float] | None = None
        bricks: list[tuple[float, float, float, float]] = []
        powerups: list[tuple[float, float, float, float]] = []

        # Track best confidence for paddle/ball (pick highest if multiple)
        best_paddle_conf = -1.0
        best_ball_conf = -1.0

        for det in raw:
            name = det["class_name"]
            conf = det["confidence"]
            xyxy = det["bbox_xyxy"]
            x1, y1, x2, y2 = xyxy

            # Normalise using provided frame dimensions
            fw = frame_width if frame_width > 0 else 1
            fh = frame_height if frame_height > 0 else 1
            cx_norm = ((x1 + x2) / 2.0) / fw
            cy_norm = ((y1 + y2) / 2.0) / fh
            w_norm = (x2 - x1) / fw
            h_norm = (y2 - y1) / fh
            bbox_norm = (cx_norm, cy_norm, w_norm, h_norm)

            if name == "paddle" and conf > best_paddle_conf:
                paddle = bbox_norm
                best_paddle_conf = conf
            elif name == "ball" and conf > best_ball_conf:
                ball = bbox_norm
                best_ball_conf = conf
            elif name == "brick":
                bricks.append(bbox_norm)
            elif name == "powerup":
                powerups.append(bbox_norm)

        return {
            "paddle": paddle,
            "ball": ball,
            "bricks": bricks,
            "powerups": powerups,
            "raw_detections": raw,
        }

    def is_loaded(self) -> bool:
        """Check if the model has been loaded.

        Returns
        -------
        bool
            True if the model is loaded and ready for inference.
        """
        return self.model is not None
