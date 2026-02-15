"""Soak Oracle — detects gradual degradation over long play sessions.

Monitors resource usage trends across multiple episodes to detect
memory leaks, performance degradation, and other long-running
stability issues that only manifest after extended play.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base import Oracle


class SoakOracle(Oracle):
    """Detects gradual performance degradation over multiple episodes.

    Detection strategy
    ------------------
    1. Sample RAM usage at regular intervals and track the trend.
    2. If RAM usage shows a statistically significant upward trend
       (linear regression slope > ``leak_threshold_mb_per_min``),
       flag a potential memory leak.
    3. Track per-episode mean FPS and flag if it trends downward
       over successive episodes.

    Parameters
    ----------
    sample_interval_steps : int
        Number of steps between resource usage samples.
        Default is 300 (~10 seconds at 30 FPS).
    leak_threshold_mb_per_min : float
        Minimum RAM growth rate (MB/minute) to flag as a leak.
        Default is 10.0.
    min_samples_for_trend : int
        Minimum number of samples before trend analysis is applied.
        Default is 10.
    fps_degradation_threshold : float
        Minimum FPS drop (as fraction of initial) across episodes
        to flag performance degradation.  Default is 0.2 (20% drop).
    """

    def __init__(
        self,
        sample_interval_steps: int = 300,
        leak_threshold_mb_per_min: float = 10.0,
        min_samples_for_trend: int = 10,
        fps_degradation_threshold: float = 0.2,
    ) -> None:
        super().__init__(name="soak")
        self.sample_interval_steps = sample_interval_steps
        self.leak_threshold_mb_per_min = leak_threshold_mb_per_min
        self.min_samples_for_trend = min_samples_for_trend
        self.fps_degradation_threshold = fps_degradation_threshold

        # RAM tracking: (wall_time, ram_mb)
        self._ram_samples: list[tuple[float, float]] = []
        self._start_time: float = time.perf_counter()
        self._step_count: int = 0
        self._total_steps: int = 0  # Across all episodes

        # FPS tracking per episode
        self._episode_fps_means: list[float] = []
        self._episode_step_times: list[float] = []
        self._last_step_time: float | None = None

        self._leak_warned: bool = False
        self._fps_degradation_warned: bool = False

    def on_reset(self, obs: np.ndarray, info: dict[str, Any]) -> None:
        """Record per-episode stats and reset step counter.

        Parameters
        ----------
        obs : np.ndarray
            Initial observation.
        info : dict[str, Any]
            Reset info dict.
        """
        # Record FPS for the completed episode
        if self._episode_step_times:
            valid_times = [t for t in self._episode_step_times if t > 0]
            if valid_times:
                fps_values = [1.0 / t for t in valid_times]
                self._episode_fps_means.append(float(np.mean(fps_values)))

        self._step_count = 0
        self._episode_step_times = []
        self._last_step_time = None

    def on_step(
        self,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        """Sample resources and track timing.

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
        self._total_steps += 1
        now = time.perf_counter()

        # Track step timing for FPS calculation
        if self._last_step_time is not None:
            elapsed = now - self._last_step_time
            self._episode_step_times.append(elapsed)
        self._last_step_time = now

        # Sample RAM at regular intervals
        if self._total_steps % self.sample_interval_steps == 0:
            ram_mb = self._get_ram_usage_mb()
            if ram_mb is not None:
                wall_time = now - self._start_time
                self._ram_samples.append((wall_time, ram_mb))
                self._check_memory_leak()

        # Check FPS degradation across episodes
        if terminated or truncated:
            self._check_fps_degradation()

    def _get_ram_usage_mb(self) -> float | None:
        """Get current process RAM usage in megabytes.

        Returns
        -------
        float or None
            RAM usage in MB, or None if psutil is not available.
        """
        try:
            import psutil

            process = psutil.Process()
            mem_info = process.memory_info()
            return mem_info.rss / (1024 * 1024)
        except (ImportError, Exception):
            return None

    def _check_memory_leak(self) -> None:
        """Check RAM usage trend for potential memory leak."""
        if len(self._ram_samples) < self.min_samples_for_trend or self._leak_warned:
            return

        times = np.array([s[0] for s in self._ram_samples])
        rams = np.array([s[1] for s in self._ram_samples])

        # Convert time to minutes for rate calculation
        times_min = times / 60.0

        # Simple linear regression: slope = rate of RAM growth
        if len(times_min) < 2 or np.std(times_min) == 0:
            return

        slope = float(np.polyfit(times_min, rams, 1)[0])  # MB per minute

        if slope > self.leak_threshold_mb_per_min:
            self._leak_warned = True
            duration_min = float(times_min[-1] - times_min[0])
            total_growth = float(rams[-1] - rams[0])
            self._add_finding(
                severity="warning",
                step=self._total_steps,
                description=(
                    f"Potential memory leak: RAM growing at "
                    f"{slope:.1f} MB/min over {duration_min:.1f} minutes "
                    f"(total growth: {total_growth:.1f} MB)"
                ),
                data={
                    "type": "memory_leak",
                    "slope_mb_per_min": slope,
                    "duration_min": duration_min,
                    "total_growth_mb": total_growth,
                    "start_ram_mb": float(rams[0]),
                    "end_ram_mb": float(rams[-1]),
                    "num_samples": len(self._ram_samples),
                },
            )

    def _check_fps_degradation(self) -> None:
        """Check if FPS is degrading across episodes."""
        if len(self._episode_fps_means) < 3 or self._fps_degradation_warned:
            return

        first_fps = self._episode_fps_means[0]
        if first_fps <= 0:
            return

        latest_fps = self._episode_fps_means[-1]
        drop_fraction = (first_fps - latest_fps) / first_fps

        if drop_fraction > self.fps_degradation_threshold:
            self._fps_degradation_warned = True
            self._add_finding(
                severity="warning",
                step=self._total_steps,
                description=(
                    f"FPS degradation: {first_fps:.1f} -> {latest_fps:.1f} "
                    f"({drop_fraction * 100:.0f}% drop over "
                    f"{len(self._episode_fps_means)} episodes)"
                ),
                data={
                    "type": "fps_degradation",
                    "first_episode_fps": first_fps,
                    "latest_episode_fps": latest_fps,
                    "drop_fraction": drop_fraction,
                    "num_episodes": len(self._episode_fps_means),
                    "all_fps_means": self._episode_fps_means.copy(),
                },
            )

    def get_soak_summary(self) -> dict[str, Any]:
        """Return summary of soak test metrics.

        Returns
        -------
        dict[str, Any]
            Soak test statistics including RAM trend and FPS history.
        """
        result: dict[str, Any] = {
            "total_steps": self._total_steps,
            "num_episodes": len(self._episode_fps_means),
            "num_ram_samples": len(self._ram_samples),
        }

        if self._ram_samples:
            rams = [s[1] for s in self._ram_samples]
            result["ram_start_mb"] = rams[0]
            result["ram_end_mb"] = rams[-1]
            result["ram_growth_mb"] = rams[-1] - rams[0]

        if self._episode_fps_means:
            result["fps_first_episode"] = self._episode_fps_means[0]
            result["fps_latest_episode"] = self._episode_fps_means[-1]
            result["fps_all_episodes"] = self._episode_fps_means.copy()

        return result
