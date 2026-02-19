"""Score Anomaly Oracle — detects abnormal score jumps or impossible values.

Monitors the game score (extracted from observations or info dict) and
flags sudden jumps, negative values, or statistically unlikely
trajectories.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Oracle


class ScoreAnomalyOracle(Oracle):
    """Detects anomalous score behaviour during gameplay.

    Detection strategy
    ------------------
    1. Track score deltas between consecutive steps.
    2. Flag if a single-step delta exceeds ``max_delta`` (impossible jump).
    3. Flag if the score goes negative (for games where it shouldn't).
    4. Flag if the cumulative score deviates more than ``z_threshold``
       standard deviations from the running mean (statistical anomaly).

    Parameters
    ----------
    max_delta : float
        Maximum allowed score change in a single step.  Default is 100.
    allow_negative : bool
        Whether negative scores are valid.  Default is False.
    z_threshold : float
        Number of standard deviations for statistical anomaly detection.
        Default is 3.0.
    score_key : str
        Key in the ``info`` dict that holds the current score.
        Default is ``"score"``.
    """

    def __init__(
        self,
        max_delta: float = 100.0,
        allow_negative: bool = False,
        z_threshold: float = 3.0,
        score_key: str = "score",
    ) -> None:
        super().__init__(name="score_anomaly")
        self.max_delta = max_delta
        self.allow_negative = allow_negative
        self.z_threshold = z_threshold
        self.score_key = score_key

        self._prev_score: float | None = None
        self._score_history: list[float] = []
        self._step_count: int = 0
        self._negative_score_reported: bool = False

    def on_reset(self, obs: np.ndarray, info: dict[str, Any]) -> None:
        """Reset score tracking at episode start.

        Parameters
        ----------
        obs : np.ndarray
            Initial observation.
        info : dict[str, Any]
            Reset info dict.  May contain initial score.
        """
        self._prev_score = info.get(self.score_key)
        self._score_history.clear()
        self._step_count = 0
        self._negative_score_reported = False

    def on_step(
        self,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        """Check for score anomalies after each step.

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
            Step info dict.  Expected to contain ``self.score_key``.
        """
        self._step_count += 1

        current_score = info.get(self.score_key)
        if current_score is None:
            return  # No score data available this step

        current_score = float(current_score)

        # 1. Check for negative score (report once per negative streak)
        if not self.allow_negative and current_score < 0:
            if not self._negative_score_reported:
                self._add_finding(
                    severity="warning",
                    step=self._step_count,
                    description=f"Negative score detected: {current_score}",
                    data={"score": current_score},
                )
                self._negative_score_reported = True
        else:
            self._negative_score_reported = False

        # 2. Check for impossible jump
        if self._prev_score is not None:
            if self._check_impossible_jump(self._prev_score, current_score):
                delta = current_score - self._prev_score
                self._add_finding(
                    severity="critical",
                    step=self._step_count,
                    description=(
                        f"Impossible score jump: {self._prev_score} -> "
                        f"{current_score} (delta={delta:.1f}, "
                        f"max_delta={self.max_delta})"
                    ),
                    data={
                        "prev_score": self._prev_score,
                        "current_score": current_score,
                        "delta": delta,
                    },
                )

        # 3. Record score and check for statistical anomaly
        self._score_history.append(current_score)
        if self._check_statistical_anomaly(current_score):
            mean = float(np.mean(self._score_history[:-1]))
            std = float(np.std(self._score_history[:-1]))
            z_score = (current_score - mean) / std if std > 0 else 0.0
            self._add_finding(
                severity="warning",
                step=self._step_count,
                description=(
                    f"Statistical score anomaly: score={current_score:.1f}, z-score={z_score:.2f}"
                ),
                data={
                    "score": current_score,
                    "mean": mean,
                    "std": std,
                    "z_score": z_score,
                },
            )

        self._prev_score = current_score

    def _check_impossible_jump(self, prev: float, current: float) -> bool:
        """Check if the score delta exceeds the allowed maximum.

        Parameters
        ----------
        prev : float
            Previous score.
        current : float
            Current score.

        Returns
        -------
        bool
            True if |current - prev| > ``max_delta``.
        """
        return abs(current - prev) > self.max_delta

    def _check_statistical_anomaly(self, score: float) -> bool:
        """Check if the score is a statistical outlier.

        Parameters
        ----------
        score : float
            Current score value.

        Returns
        -------
        bool
            True if score is > ``z_threshold`` std devs from the mean.
        """
        # Need at least a few data points for meaningful statistics
        # Use history *excluding* the current score (already appended)
        if len(self._score_history) < 3:
            return False

        history = self._score_history[:-1]
        mean = float(np.mean(history))
        std = float(np.std(history))

        if std == 0:
            return False

        z_score = abs(score - mean) / std
        return z_score > self.z_threshold
