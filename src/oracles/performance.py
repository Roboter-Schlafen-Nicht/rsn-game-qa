"""Performance Oracle — monitors FPS, frame times, and resource usage.

Uses ``psutil`` and ``time.perf_counter`` to track system-level
performance metrics during gameplay and flag degradations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Oracle


class PerformanceOracle(Oracle):
    """Detects performance degradation during gameplay.

    Detection strategy
    ------------------
    1. Measure wall-clock time between ``on_step`` calls to estimate
       effective FPS.
    2. If FPS drops below ``min_fps`` for ``sustained_frames``
       consecutive frames, raise a warning.
    3. Monitor CPU and RAM usage via ``psutil`` — flag if they exceed
       ``cpu_threshold`` or ``ram_threshold_mb``.

    Parameters
    ----------
    min_fps : float
        Minimum acceptable FPS.  Default is 20.0.
    sustained_frames : int
        Number of consecutive low-FPS frames before raising a finding.
        Default is 30.
    cpu_threshold : float
        Maximum acceptable CPU usage as a percentage (0-100).
        Default is 90.0.
    ram_threshold_mb : float
        Maximum acceptable RAM usage in megabytes.
        Default is 4096.0 (4 GB).
    process_name : str, optional
        Name of the game process for per-process monitoring.
    """

    def __init__(
        self,
        min_fps: float = 20.0,
        sustained_frames: int = 30,
        cpu_threshold: float = 90.0,
        ram_threshold_mb: float = 4096.0,
        process_name: str | None = None,
    ) -> None:
        super().__init__(name="performance")
        self.min_fps = min_fps
        self.sustained_frames = sustained_frames
        self.cpu_threshold = cpu_threshold
        self.ram_threshold_mb = ram_threshold_mb
        self.process_name = process_name

        self._last_step_time: float | None = None
        self._low_fps_count: int = 0
        self._fps_history: list[float] = []
        self._step_count: int = 0

    def on_reset(self, obs: np.ndarray, info: dict[str, Any]) -> None:
        """Reset performance tracking at episode start.

        Parameters
        ----------
        obs : np.ndarray
            Initial observation.
        info : dict[str, Any]
            Reset info dict.
        """
        self._last_step_time = None
        self._low_fps_count = 0
        self._fps_history.clear()
        self._step_count = 0

    def on_step(
        self,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        """Measure frame timing and resource usage after each step.

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
            Step info dict.
        """
        raise NotImplementedError(
            "PerformanceOracle.on_step: performance monitoring not yet implemented"
        )

    def _measure_fps(self) -> float | None:
        """Calculate instantaneous FPS from wall-clock step timing.

        Returns
        -------
        float or None
            FPS estimate, or None if this is the first step.
        """
        raise NotImplementedError("FPS measurement not yet implemented")

    def _check_resource_usage(self) -> dict[str, float]:
        """Query CPU and RAM usage via psutil.

        Returns
        -------
        dict[str, float]
            Keys: ``"cpu_percent"``, ``"ram_mb"``.
        """
        raise NotImplementedError("Resource usage check not yet implemented")

    def get_fps_summary(self) -> dict[str, float]:
        """Return summary statistics for FPS over the episode.

        Returns
        -------
        dict[str, float]
            Keys: ``"mean_fps"``, ``"min_fps"``, ``"max_fps"``,
            ``"std_fps"``.
        """
        raise NotImplementedError("FPS summary not yet implemented")
