"""Episode Length Oracle — detects abnormal episode durations.

Tracks episode lengths across multiple episodes and flags statistical
outliers, which can indicate instant-death bugs, infinite loops,
unkillable states, or other progression issues.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Oracle


class EpisodeLengthOracle(Oracle):
    """Detects abnormally short or long episodes.

    Detection strategy
    ------------------
    1. Count steps within each episode.
    2. Flag episodes shorter than ``min_steps`` (instant death / crash).
    3. Flag episodes longer than ``max_steps`` (infinite loop / stuck).
    4. After accumulating enough episodes, flag statistical outliers
       using z-score analysis on episode lengths.

    Parameters
    ----------
    min_steps : int
        Minimum expected episode length.  Episodes shorter than this
        trigger a warning.  Default is 10.
    max_steps : int
        Maximum expected episode length.  Episodes longer than this
        trigger a warning.  Default is 10000.
    z_threshold : float
        Number of standard deviations for statistical outlier detection.
        Default is 3.0.
    min_episodes_for_stats : int
        Minimum number of completed episodes before statistical
        analysis is applied.  Default is 5.
    """

    def __init__(
        self,
        min_steps: int = 10,
        max_steps: int = 10000,
        z_threshold: float = 3.0,
        min_episodes_for_stats: int = 5,
    ) -> None:
        super().__init__(name="episode_length")
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.z_threshold = z_threshold
        self.min_episodes_for_stats = min_episodes_for_stats

        self._step_count: int = 0
        self._episode_lengths: list[int] = []
        self._max_steps_warned: bool = False

    def on_reset(self, obs: np.ndarray, info: dict[str, Any]) -> None:
        """Record previous episode length and reset step counter.

        Parameters
        ----------
        obs : np.ndarray
            Initial observation.
        info : dict[str, Any]
            Reset info dict.
        """
        # Record the previous episode length (if any)
        if self._step_count > 0:
            self._episode_lengths.append(self._step_count)

        self._step_count = 0
        self._max_steps_warned = False

    def on_step(
        self,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        """Track episode length and check for anomalies.

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

        # Check for excessively long episode (once per episode)
        if self._step_count >= self.max_steps and not self._max_steps_warned:
            self._max_steps_warned = True
            self._add_finding(
                severity="warning",
                step=self._step_count,
                description=(
                    f"Episode exceeds {self.max_steps} steps — "
                    f"possible infinite loop or stuck state"
                ),
                data={
                    "type": "long_episode",
                    "step_count": self._step_count,
                    "max_steps": self.max_steps,
                },
            )

        # On episode end, check for abnormally short episode
        if terminated or truncated:
            if self._step_count < self.min_steps:
                self._add_finding(
                    severity="warning",
                    step=self._step_count,
                    description=(
                        f"Episode ended after only {self._step_count} steps "
                        f"(min expected: {self.min_steps}) — "
                        f"possible instant-death bug"
                    ),
                    data={
                        "type": "short_episode",
                        "step_count": self._step_count,
                        "min_steps": self.min_steps,
                    },
                )

            # Statistical outlier detection across episodes
            self._check_statistical_outlier()

            # Record this episode
            self._episode_lengths.append(self._step_count)

    def _check_statistical_outlier(self) -> None:
        """Check if current episode length is a statistical outlier.

        Only runs if enough previous episodes have been recorded.
        """
        if len(self._episode_lengths) < self.min_episodes_for_stats:
            return

        arr = np.array(self._episode_lengths)
        mean = float(np.mean(arr))
        std = float(np.std(arr))

        if std == 0:
            return

        z_score = abs(self._step_count - mean) / std
        if z_score > self.z_threshold:
            self._add_finding(
                severity="info",
                step=self._step_count,
                description=(
                    f"Episode length {self._step_count} is a statistical "
                    f"outlier (z={z_score:.2f}, mean={mean:.0f}, "
                    f"std={std:.0f})"
                ),
                data={
                    "type": "statistical_outlier",
                    "step_count": self._step_count,
                    "z_score": z_score,
                    "mean": mean,
                    "std": std,
                    "num_episodes": len(self._episode_lengths),
                },
            )

    def get_episode_stats(self) -> dict[str, float]:
        """Return summary statistics for episode lengths.

        Returns
        -------
        dict[str, float]
            Keys: ``"mean_length"``, ``"min_length"``, ``"max_length"``,
            ``"std_length"``, ``"num_episodes"``.
        """
        if not self._episode_lengths:
            return {
                "mean_length": 0.0,
                "min_length": 0.0,
                "max_length": 0.0,
                "std_length": 0.0,
                "num_episodes": 0,
            }

        arr = np.array(self._episode_lengths)
        return {
            "mean_length": float(np.mean(arr)),
            "min_length": float(np.min(arr)),
            "max_length": float(np.max(arr)),
            "std_length": float(np.std(arr)),
            "num_episodes": len(self._episode_lengths),
        }
