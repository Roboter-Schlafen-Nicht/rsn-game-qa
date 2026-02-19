"""Epsilon-greedy action noise wrapper for VecEnv.

Replaces agent actions with uniform random samples with probability
``epsilon``.  Forces visual diversity in observations, preventing the
RND predictor from learning a single static frame embedding and
collapsing the intrinsic reward signal.

The wrapper is **stateless** per environment — each ``step()`` call
independently decides whether to replace the action.  There is no
correlation between steps or environments.

Usage::

    from src.platform.epsilon_greedy_wrapper import EpsilonGreedyWrapper

    vec_env = DummyVecEnv([lambda: env])
    vec_env = EpsilonGreedyWrapper(vec_env, epsilon=0.1)
"""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper


class EpsilonGreedyWrapper(VecEnvWrapper):
    """Replace agent actions with random actions with probability epsilon.

    Parameters
    ----------
    venv : VecEnv
        Vectorised environment to wrap.
    epsilon : float
        Probability of replacing each environment's action with a
        uniform random sample from the action space.  Must be in
        ``[0.0, 1.0]``.  Default ``0.1`` (10% random actions).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        venv: VecEnv,
        epsilon: float = 0.1,
        seed: int | None = None,
    ) -> None:
        super().__init__(venv)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0.0, 1.0], got {epsilon}")
        self.epsilon = epsilon
        self._rng = np.random.default_rng(seed)

    def step_async(self, actions: np.ndarray) -> None:
        """Replace actions with random ones with probability epsilon."""
        if self.epsilon > 0.0:
            n_envs = actions.shape[0]
            # Per-environment coin flip
            mask = self._rng.random(n_envs) < self.epsilon
            if mask.any():
                for i in range(n_envs):
                    if mask[i]:
                        actions[i] = self.action_space.sample()
        self.venv.step_async(actions)

    def step_wait(self) -> tuple:
        """Pass through to the wrapped environment."""
        return self.venv.step_wait()

    def reset(self) -> np.ndarray:
        """Pass through to the wrapped environment."""
        return self.venv.reset()
