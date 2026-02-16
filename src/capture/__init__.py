"""Capture module — window frame capture and input injection for Windows games."""

from .window_capture import WindowCapture
from .input_controller import InputController

# WinCamCapture requires wincam (DirectX 11 GPU, Windows 10+ only).
# Not imported eagerly to avoid breaking CI/headless environments.
# Use: from src.capture.wincam_capture import WinCamCapture
__all__ = ["WindowCapture", "InputController"]
