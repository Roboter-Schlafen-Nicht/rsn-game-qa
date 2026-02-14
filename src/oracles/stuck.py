"""Stuck Oracle — detects when the RL agent is stuck or making no progress.

Monitors the agent's observation and reward trajectory.  If the agent
stays in substantially the same state for too many steps, a warning
finding is raised.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Oracle


class StuckOracle(Oracle):
    """Detects the agent being stuck in a repetitive or non-progressing state.

    Detection strategy
    ------------------
    1. Track the rolling mean reward over a sliding window.
    2. If the rolling mean stays near zero (below ``reward_epsilon``) for
       ``patience`` steps, flag a "stuck" warning.
    3. Optionally track observation variance — if observations barely
       change, the agent may be stuck in a corner.

    Parameters
    ----------
    patience : int
        Number of steps of near-zero reward before raising a finding.
        Default is 300 (~10 seconds at 30 FPS).
    reward_epsilon : float
        Absolute reward threshold below which a step is considered
        "no progress".  Default is 1e-4.
    obs_var_threshold : float
        If the variance of recent observations falls below this,
        consider the agent stuck.  Default is 1e-6.
    window_size : int
        Size of the sliding window for reward/observation tracking.
        Default is 60.
    """

    def __init__(
        self,
        patience: int = 300,
        reward_epsilon: float = 1e-4,
        obs_var_threshold: float = 1e-6,
        window_size: int = 60,
    ) -> None:
        super().__init__(name="stuck")
        self.patience = patience
        self.reward_epsilon = reward_epsilon
        self.obs_var_threshold = obs_var_threshold
        self.window_size = window_size

        self._reward_buffer: list[float] = []
        self._obs_buffer: list[np.ndarray] = []
        self._no_progress_steps: int = 0
        self._step_count: int = 0

    def on_reset(self, obs: np.ndarray, info: dict[str, Any]) -> None:
        """Reset tracking state at episode start.

        Parameters
        ----------
        obs : np.ndarray
            Initial observation.
        info : dict[str, Any]
            Reset info dict.
        """
        self._reward_buffer.clear()
        self._obs_buffer.clear()
        self._no_progress_steps = 0
        self._step_count = 0

    def on_step(
        self,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        """Update progress tracking and check for stuck conditions.

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
            "StuckOracle.on_step: stuck detection not yet implemented"
        )

    def _check_reward_stagnation(self) -> bool:
        """Check if recent rewards indicate no progress.

        Returns
        -------
        bool
            True if rolling mean reward is below ``reward_epsilon``.
        """
        raise NotImplementedError("Reward stagnation check not yet implemented")

    def _check_observation_variance(self) -> bool:
        """Check if recent observations show minimal change.

        Returns
        -------
        bool
            True if observation variance is below ``obs_var_threshold``.
        """
        raise NotImplementedError("Observation variance check not yet implemented")
