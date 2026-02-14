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
        raise NotImplementedError(
            "ScoreAnomalyOracle.on_step: score anomaly detection not yet implemented"
        )

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
        raise NotImplementedError("Impossible jump check not yet implemented")

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
        raise NotImplementedError("Statistical anomaly check not yet implemented")
