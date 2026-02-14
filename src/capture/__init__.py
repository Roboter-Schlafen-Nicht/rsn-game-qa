"""Capture module — window frame capture and input injection for Windows games."""

from .window_capture import WindowCapture
from .input_controller import InputController

__all__ = ["WindowCapture", "InputController"]
