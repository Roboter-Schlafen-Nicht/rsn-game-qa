"""Performance Oracle — monitors FPS, frame times, and resource usage.

Uses ``psutil`` and ``time.perf_counter`` to track system-level
performance metrics during gameplay and flag degradations.
"""

from __future__ import annotations

import time
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
        self._step_count += 1

        # 1. Measure FPS
        fps = self._measure_fps()
        if fps is not None:
            self._fps_history.append(fps)

            if fps < self.min_fps:
                self._low_fps_count += 1
            else:
                self._low_fps_count = 0

            # Fire finding when sustained low FPS threshold hit (exactly once)
            if self._low_fps_count == self.sustained_frames:
                self._add_finding(
                    severity="warning",
                    step=self._step_count,
                    description=(
                        f"FPS dropped below {self.min_fps} for "
                        f"{self.sustained_frames} consecutive frames "
                        f"(current: {fps:.1f})"
                    ),
                    data={
                        "type": "low_fps",
                        "current_fps": fps,
                        "min_fps": self.min_fps,
                        "sustained_frames": self._low_fps_count,
                    },
                )

        # 2. Check resource usage (every 60 steps to avoid overhead)
        if self._step_count % 60 == 0:
            resources = self._check_resource_usage()
            if resources:
                cpu = resources.get("cpu_percent", 0.0)
                ram = resources.get("ram_mb", 0.0)

                if cpu > self.cpu_threshold:
                    self._add_finding(
                        severity="warning",
                        step=self._step_count,
                        description=(
                            f"CPU usage {cpu:.1f}% exceeds threshold "
                            f"{self.cpu_threshold}%"
                        ),
                        data={
                            "type": "high_cpu",
                            "cpu_percent": cpu,
                            "threshold": self.cpu_threshold,
                        },
                    )

                if ram > self.ram_threshold_mb:
                    self._add_finding(
                        severity="warning",
                        step=self._step_count,
                        description=(
                            f"RAM usage {ram:.1f} MB exceeds threshold "
                            f"{self.ram_threshold_mb} MB"
                        ),
                        data={
                            "type": "high_ram",
                            "ram_mb": ram,
                            "threshold": self.ram_threshold_mb,
                        },
                    )

    def _measure_fps(self) -> float | None:
        """Calculate instantaneous FPS from wall-clock step timing.

        Returns
        -------
        float or None
            FPS estimate, or None if this is the first step.
        """
        now = time.perf_counter()

        if self._last_step_time is None:
            self._last_step_time = now
            return None

        elapsed = now - self._last_step_time
        self._last_step_time = now

        if elapsed <= 0:
            return None

        return 1.0 / elapsed

    def _check_resource_usage(self) -> dict[str, float]:
        """Query CPU and RAM usage via psutil.

        Returns
        -------
        dict[str, float]
            Keys: ``"cpu_percent"``, ``"ram_mb"``.
            Returns empty dict if psutil is not available.
        """
        try:
            import psutil
        except ImportError:
            return {}

        cpu_percent = psutil.cpu_percent(interval=None)
        ram_info = psutil.virtual_memory()
        ram_mb = ram_info.used / (1024 * 1024)

        return {
            "cpu_percent": float(cpu_percent),
            "ram_mb": float(ram_mb),
        }

    def get_fps_summary(self) -> dict[str, float]:
        """Return summary statistics for FPS over the episode.

        Returns
        -------
        dict[str, float]
            Keys: ``"mean_fps"``, ``"min_fps"``, ``"max_fps"``,
            ``"std_fps"``.  Returns zeros if no FPS data available.
        """
        if not self._fps_history:
            return {
                "mean_fps": 0.0,
                "min_fps": 0.0,
                "max_fps": 0.0,
                "std_fps": 0.0,
            }

        arr = np.array(self._fps_history)
        return {
            "mean_fps": float(np.mean(arr)),
            "min_fps": float(np.min(arr)),
            "max_fps": float(np.max(arr)),
            "std_fps": float(np.std(arr)),
        }
