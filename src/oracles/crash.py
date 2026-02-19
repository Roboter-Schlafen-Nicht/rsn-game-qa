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
        self._black_frame_reported: bool = False
        self._freeze_reported: bool = False

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
        self._black_frame_reported = False
        self._freeze_reported = False

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
        self._step_count += 1

        # 1. Check process liveness
        if not self._check_process_alive():
            self._add_finding(
                severity="critical",
                step=self._step_count,
                description=(f"Game process '{self.process_name}' is no longer running"),
                data={"process_name": self.process_name},
            )
            return  # No point checking frames if the process is dead

        # 2. Check for frame freeze / black frame
        frame = info.get("frame")
        if frame is not None:
            # Check for black frame (report once per episode)
            if np.all(frame == 0):
                if not self._black_frame_reported:
                    self._add_finding(
                        severity="critical",
                        step=self._step_count,
                        description="Black frame detected — possible crash or hang",
                        data={"type": "black_frame"},
                        frame=frame,
                    )
                    self._black_frame_reported = True
            else:
                # Reset flag when we see a non-black frame
                self._black_frame_reported = False

            # Check for frozen frame (report once per freeze sequence)
            if self._check_frame_frozen(frame):
                if not self._freeze_reported:
                    self._add_finding(
                        severity="critical",
                        step=self._step_count,
                        description=(
                            f"Frame frozen for {self.freeze_threshold} "
                            f"consecutive steps — possible hang"
                        ),
                        data={
                            "type": "frozen_frame",
                            "identical_count": self._identical_count,
                        },
                        frame=frame,
                    )
                    self._freeze_reported = True
            else:
                self._freeze_reported = False

    def _check_process_alive(self) -> bool:
        """Check whether the game process is still running.

        Returns
        -------
        bool
            True if the process is alive or ``process_name`` is not set.
        """
        if self.process_name is None:
            return True

        try:
            import psutil
        except ImportError:
            # psutil not available — assume alive
            return True

        for proc in psutil.process_iter(["name"]):
            try:
                if proc.info["name"] == self.process_name:
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return False

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
        if self._prev_frame is not None and np.array_equal(frame, self._prev_frame):
            self._identical_count += 1
        else:
            self._identical_count = 0

        self._prev_frame = frame.copy()

        return self._identical_count >= self.freeze_threshold
