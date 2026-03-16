"""LTC (Liquid Time-Constant) evaluation wrapper for game environments.

Provides ``LtcEvalWrapper``, a Gymnasium wrapper that converts a
:class:`BaseGameEnv`'s MLP observation into a single-frame CHW
grayscale image suitable for recurrent CNN policies (CfC/LTC).

Unlike :class:`CnnEvalWrapper`, this wrapper does **not** perform
frame stacking.  Temporal context is handled by the recurrent hidden
state (CfC cell), so only the current frame is needed.

Usage::

    from src.platform.ltc_wrapper import LtcEvalWrapper

    base_env = Breakout71Env(...)
    eval_env = LtcEvalWrapper(base_env)

    model = RecurrentPPO.load("ltc_breakout71.zip")
    obs, info = eval_env.reset()
    lstm_states = None
    episode_starts = np.array([True])

    for _ in range(1000):
        action, lstm_states = model.predict(
            obs[np.newaxis],  # add batch dim
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True,
        )
        obs, reward, terminated, truncated, info = eval_env.step(action)
        episode_starts = np.array([terminated or truncated])

Notes
-----
- Observation shape: ``(1, obs_size, obs_size)`` — CHW, single channel.
- The original MLP observation is preserved in ``info["mlp_obs"]``.
- When no ``"frame"`` key is present in ``info``, a black frame is
  returned (all zeros).
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)

# Default target resolution — matches CNN pipeline (84x84).
_LTC_OBS_SIZE: int = 84


class LtcEvalWrapper(gym.Wrapper):
    """Gymnasium wrapper for evaluating LTC/CfC-trained recurrent models.

    Converts the base environment's MLP observation into a single-frame
    CHW grayscale image.  No frame stacking is performed — the recurrent
    hidden state carries temporal context.

    Parameters
    ----------
    env : gym.Env
        The base environment (a :class:`BaseGameEnv` subclass).  Must
        produce an ``info`` dict containing a ``"frame"`` key with a
        BGR ``np.ndarray``.
    obs_size : int
        Target height and width for the square observation image.
        Default is 84.

    Attributes
    ----------
    observation_space : gym.spaces.Box
        ``Box(0, 255, (1, obs_size, obs_size), uint8)`` — single-channel
        CHW grayscale image.
    """

    def __init__(self, env: gym.Env, obs_size: int = _LTC_OBS_SIZE) -> None:
        if obs_size < 1:
            raise ValueError(f"obs_size must be >= 1, got {obs_size}")
        super().__init__(env)
        self._obs_size = obs_size

        # Override observation space: CHW uint8 single-frame image
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1, obs_size, obs_size),
            dtype=np.uint8,
        )

        logger.info(
            "LtcEvalWrapper: obs_size=%d, space=%s",
            obs_size,
            self.observation_space.shape,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _frame_to_chw(self, frame: np.ndarray | None) -> np.ndarray:
        """Convert a raw BGR frame to a CHW grayscale ``(1, H, W)`` array.

        Parameters
        ----------
        frame : np.ndarray or None
            BGR image from the game window.  If ``None``, returns a
            black frame of shape ``(1, obs_size, obs_size)``.

        Returns
        -------
        np.ndarray
            Grayscale image of shape ``(1, obs_size, obs_size)``,
            dtype ``uint8``.
        """
        if frame is None:
            return np.zeros((1, self._obs_size, self._obs_size), dtype=np.uint8)

        # Reuse the shared _frame_to_obs helper from cnn_wrapper
        from src.platform.cnn_wrapper import _frame_to_obs

        obs_hwc = _frame_to_obs(frame, self._obs_size)  # (H, W, 1)
        # HWC → CHW: (H, W, 1) → (1, H, W)
        return obs_hwc.transpose(2, 0, 1)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment and return a single-frame CHW observation.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        options : dict, optional
            Additional reset options.

        Returns
        -------
        obs : np.ndarray
            CHW observation of shape ``(1, obs_size, obs_size)``.
        info : dict
            Info dict from the base environment, with added
            ``"mlp_obs"`` key containing the original observation.
        """
        mlp_obs, info = self.env.reset(seed=seed, options=options)

        frame = info.get("frame")
        obs = self._frame_to_chw(frame)

        info["mlp_obs"] = mlp_obs
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Take a step and return a single-frame CHW observation.

        Parameters
        ----------
        action : np.ndarray
            Action to pass to the base environment.

        Returns
        -------
        obs : np.ndarray
            CHW observation of shape ``(1, obs_size, obs_size)``.
        reward : float
            Reward from the base environment (unchanged).
        terminated : bool
            Termination flag (unchanged).
        truncated : bool
            Truncation flag (unchanged).
        info : dict
            Info dict with added ``"mlp_obs"`` key.
        """
        mlp_obs, reward, terminated, truncated, info = self.env.step(action)

        frame = info.get("frame")
        obs = self._frame_to_chw(frame)

        info["mlp_obs"] = mlp_obs
        return obs, reward, terminated, truncated, info
