"""Crash Oracle — detects game crashes, freezes, and unresponsive windows.

Monitors the target window's responsiveness and process state.  If the
window stops responding or the process exits unexpectedly, a critical
finding is raised.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Oracle


class CrashOracle(Oracle):
    """Detects game crashes by monitoring window/process liveness.

    Detection strategy
    ------------------
    1. Check if the game window handle is still valid.
    2. Check if the game process is still running (via ``psutil``).
    3. Optionally compare consecutive frames — if the frame is completely
       black or unchanged for ``freeze_threshold`` steps, flag a freeze.

    Parameters
    ----------
    freeze_threshold : int
        Number of consecutive identical frames before declaring a freeze.
        Default is 30 (roughly 1 second at 30 FPS).
    process_name : str, optional
        Name of the game process to monitor (e.g. ``"chrome.exe"``).
        If not set, only window-level checks are performed.
    """

    def __init__(
        self,
        freeze_threshold: int = 30,
        process_name: str | None = None,
    ) -> None:
        super().__init__(name="crash")
        self.freeze_threshold = freeze_threshold
        self.process_name = process_name
        self._prev_frame: np.ndarray | None = None
        self._identical_count: int = 0
        self._step_count: int = 0

    def on_reset(self, obs: np.ndarray, info: dict[str, Any]) -> None:
        """Reset internal state at episode start.

        Parameters
        ----------
        obs : np.ndarray
            Initial observation.
        info : dict[str, Any]
            Reset info dict.
        """
        self._prev_frame = None
        self._identical_count = 0
        self._step_count = 0

    def on_step(
        self,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        """Check for crashes/freezes after each step.

        Parameters
        ----------
        obs : np.ndarray
            Current observation.
        reward : float
            Step reward.
        terminated : bool
            Episode terminated flag.
        truncated : bool
            Episode truncated flag.
        info : dict[str, Any]
            Step info dict.  Expected to contain ``"frame"`` (raw BGR
            image) if frame-level freeze detection is desired.
        """
        raise NotImplementedError(
            "CrashOracle.on_step: crash/freeze detection not yet implemented"
        )

    def _check_process_alive(self) -> bool:
        """Check whether the game process is still running.

        Returns
        -------
        bool
            True if the process is alive or ``process_name`` is not set.
        """
        raise NotImplementedError("Process liveness check not yet implemented")

    def _check_frame_frozen(self, frame: np.ndarray) -> bool:
        """Compare current frame to previous; update freeze counter.

        Parameters
        ----------
        frame : np.ndarray
            Current BGR frame.

        Returns
        -------
        bool
            True if the frame has been identical for ``freeze_threshold``
            consecutive steps.
        """
        raise NotImplementedError("Frame freeze detection not yet implemented")
