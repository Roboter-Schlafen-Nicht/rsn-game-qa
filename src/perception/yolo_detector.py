"""YOLO-based object detector for game frame analysis.

Loads a trained YOLOv8 model and extracts structured detections
(bounding boxes, classes, confidences) from game frames.  Uses
``resolve_device()`` to auto-detect the best available device
(XPU > CUDA > CPU) by default.

When an OpenVINO-exported model directory exists alongside the
``.pt`` weights (e.g. ``best_openvino_model/``), the detector
automatically uses it for inference on CPU or Intel devices,
providing a significant speed-up without requiring code changes.

Ultralytics requires ``device="intel:<OV_DEVICE>"`` for OpenVINO
device routing (e.g. ``"intel:GPU"`` for Intel Arc GPUs).  The
detector translates ``resolve_device()`` outputs (``"xpu"`` →
``"intel:GPU"``, ``"cpu"`` → ``"intel:CPU"``) automatically.
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

# Devices where OpenVINO IR models are preferred over PyTorch .pt
_OPENVINO_PREFERRED_DEVICES = frozenset({"cpu", "auto"})

# Mapping from resolve_device() output to OpenVINO device names.
# ultralytics requires "intel:<OV_DEVICE>" for OpenVINO device routing
# (see ultralytics/nn/autobackend.py lines 315-351).
# NOTE: The "intel:" prefix is an ultralytics routing convention, NOT a
# hardware vendor constraint.  OpenVINO's "CPU" plugin is vendor-agnostic
# and works on AMD, Intel, and ARM CPUs alike.  "intel:" simply tells
# ultralytics to use the OpenVINO backend instead of PyTorch.
# NOTE: ultralytics validates device_name against core.available_devices
# using exact string match.  OpenVINO lists GPUs as "GPU.0", "GPU.1",
# not "GPU", so we must query for the actual device name at runtime.
_OPENVINO_DEVICE_MAP: dict[str, str] = {
    "xpu": "GPU",  # mapped to "intel:GPU.N" at runtime
    "cpu": "CPU",  # works on any CPU vendor (AMD, Intel, ARM)
    "auto": "AUTO",
}


def _resolve_openvino_device(requested_device: str) -> str:
    """Map a ``resolve_device()`` output to an ``"intel:<OV_DEVICE>"`` string.

    ultralytics validates the OpenVINO device name against
    ``ov.Core().available_devices`` using an exact string match.
    Since OpenVINO lists discrete GPUs as ``"GPU.0"``, ``"GPU.1"``
    (not plain ``"GPU"``), we query the runtime for the first
    matching device.

    Parameters
    ----------
    requested_device : str
        Device string from ``resolve_device()`` (e.g. ``"xpu"``,
        ``"cpu"``, ``"auto"``).

    Returns
    -------
    str
        ``"intel:<OV_DEVICE>"`` string for ultralytics.
    """
    ov_hint = _OPENVINO_DEVICE_MAP.get(requested_device, "AUTO")

    if ov_hint in ("CPU", "AUTO"):
        return f"intel:{ov_hint}"

    # For GPU, find the exact device name from OpenVINO runtime
    try:
        import openvino as ov

        core = ov.Core()
        available = core.available_devices
        # Check exact match first (e.g. "GPU")
        if ov_hint in available:
            return f"intel:{ov_hint}"
        # Find first device starting with the hint (e.g. "GPU.0")
        for dev in available:
            if dev.startswith(ov_hint):
                logger.info(
                    "OpenVINO device '%s' not in available_devices, using '%s' instead",
                    ov_hint,
                    dev,
                )
                return f"intel:{dev}"
    except ImportError:
        logger.debug("openvino not installed, falling back to intel:AUTO")
    except Exception as exc:
        logger.debug("OpenVINO device query failed: %s", exc)

    return f"intel:{ov_hint}"


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


def _find_openvino_model(weights_path: Path) -> Path | None:
    """Locate an OpenVINO model directory next to the ``.pt`` weights.

    Ultralytics exports produce a directory named
    ``<stem>_openvino_model/`` containing the IR files.

    Parameters
    ----------
    weights_path : Path
        Path to the ``.pt`` weights file.

    Returns
    -------
    Path or None
        Path to the OpenVINO model directory if it exists and contains
        the expected ``.xml`` model file, otherwise ``None``.
    """
    ov_dir = weights_path.parent / f"{weights_path.stem}_openvino_model"
    if not ov_dir.is_dir():
        return None
    # Verify the directory actually contains an OpenVINO model
    xml_files = list(ov_dir.glob("*.xml"))
    if not xml_files:
        logger.debug("OpenVINO dir %s exists but contains no .xml files", ov_dir)
        return None
    return ov_dir


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
        self._using_openvino: bool = False
        self._ov_device: str | None = None

    def load(self) -> None:
        """Load the YOLO model and move it to the target device.

        When the resolved device is CPU or Intel (XPU), this method
        checks for an OpenVINO-exported model directory alongside the
        ``.pt`` weights (e.g. ``best_openvino_model/``).  If found,
        it loads the OpenVINO model and runs a warmup inference on the
        target device so that the OpenVINO graph is compiled once
        upfront — avoiding a latency spike on the first real frame.
        Otherwise falls back to the PyTorch ``.pt`` model.

        Falls back from XPU to CPU if the XPU backend is not available.

        Raises
        ------
        FileNotFoundError
            If ``weights_path`` does not exist.
        """
        if not self.weights_path.exists():
            raise FileNotFoundError(f"YOLO weights file not found: {self.weights_path}")

        # Check if an OpenVINO model should be used
        effective_path = self.weights_path
        self._using_openvino = False
        self._ov_device: str | None = None  # ultralytics device for OpenVINO

        if self.device in _OPENVINO_PREFERRED_DEVICES or self.device == "xpu":
            ov_dir = _find_openvino_model(self.weights_path)
            if ov_dir is not None:
                effective_path = ov_dir
                self._using_openvino = True
                # Map resolved device to ultralytics OpenVINO device string
                self._ov_device = _resolve_openvino_device(self.device)
                logger.info(
                    "OpenVINO model found — using %s (device: %s → %s)",
                    ov_dir,
                    self.device,
                    self._ov_device,
                )
            else:
                logger.debug(
                    "No OpenVINO model found alongside %s — using PyTorch",
                    self.weights_path,
                )

        self.model = YOLO(str(effective_path))

        # OpenVINO models handle their own device routing — skip .to().
        # Run a warmup inference on the target device so that the
        # OpenVINO graph is compiled once (for the correct device)
        # rather than compiled for CPU by ultralytics' default warmup
        # and then recompiled for GPU.0 on the first real detect().
        if self._using_openvino:
            logger.info(
                "YOLO model loaded via OpenVINO: %s (target: %s)",
                effective_path,
                self._ov_device,
            )
            # Warmup: compile the OpenVINO graph on the target device
            _warmup_frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            self.model(
                _warmup_frame,
                imgsz=self.img_size,
                verbose=False,
                device=self._ov_device,
            )
            logger.info("OpenVINO warmup complete on %s", self._ov_device)
        else:
            # Attempt to move model to the requested device, fall back to CPU
            try:
                self.model.to(self.device)
                logger.info("YOLO model loaded on device: %s", self.device)
            except Exception:
                logger.warning(
                    "Device '%s' unavailable, falling back to CPU", self.device
                )
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
            **({"device": self._ov_device} if self._using_openvino else {}),
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
