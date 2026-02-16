"""CNN observation wrapper for Breakout71Env.

Converts the 8-element feature vector observation from
:class:`~src.env.breakout71_env.Breakout71Env` into an 84x84
single-channel (grayscale) image observation suitable for SB3's
``CnnPolicy`` (NatureCNN architecture).

The raw game frame captured each step is available in
``info["frame"]`` from the base environment.  This wrapper intercepts
it, resizes to 84x84, converts to grayscale, and returns it as the
observation.

**YOLO inference is unchanged** in the base environment — it still
runs every frame to compute reward, termination, and oracle checks.
Only the observation *returned to the policy* changes from a feature
vector to a pixel image.  This ensures a fair A/B comparison with
``MlpPolicy``: same reward signal, same episode boundaries, only
the policy input differs.

Usage with SB3::

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

    base_env = Breakout71Env(...)
    cnn_env = CnnObservationWrapper(base_env)

    vec_env = DummyVecEnv([lambda: cnn_env])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    vec_env = VecTransposeImage(vec_env)

    model = PPO("CnnPolicy", vec_env, device="xpu:1")

Notes
-----
- Frame stacking is handled externally by ``VecFrameStack``, not here.
- ``VecTransposeImage`` converts HWC → CHW for PyTorch's NatureCNN.
- The wrapper passes through reward, terminated, truncated, and info
  unchanged.  The original 8-element obs is available in
  ``info["mlp_obs"]`` for logging or debugging.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)

# Default target resolution for the CNN observation.
# 84x84 is the standard used by DQN (Mnih et al., 2013) and most
# Atari RL research.  NatureCNN is designed for this input size.
CNN_OBS_SIZE: int = 84


class CnnObservationWrapper(gym.ObservationWrapper):
    """Wrap a Breakout71Env to produce 84x84 grayscale image observations.

    This is a thin :class:`gymnasium.ObservationWrapper` that replaces the
    8-element feature vector with a grayscale image derived from the raw
    game frame already captured by the base environment.

    Parameters
    ----------
    env : gym.Env
        The base environment (typically ``Breakout71Env``).  Must produce
        an ``info`` dict containing a ``"frame"`` key with a BGR
        ``np.ndarray``.
    obs_size : int
        Target height and width for the square observation image.
        Default is 84.

    Attributes
    ----------
    observation_space : gym.spaces.Box
        ``Box(0, 255, (obs_size, obs_size, 1), uint8)`` — single-channel
        grayscale image.
    """

    def __init__(self, env: gym.Env, obs_size: int = CNN_OBS_SIZE) -> None:
        super().__init__(env)
        self._obs_size = obs_size

        # Override observation space: HWC uint8 image
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_size, obs_size, 1),
            dtype=np.uint8,
        )

        # Cache for the last raw frame (used by observation())
        self._last_raw_frame: np.ndarray | None = None

        logger.info(
            "CnnObservationWrapper: obs_size=%d, space=%s",
            obs_size,
            self.observation_space.shape,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Convert the base env's observation to an 84x84 grayscale image.

        The base env's 8-element vector is ignored.  Instead, the raw
        game frame from ``self._last_raw_frame`` (set during step/reset)
        is resized and converted to grayscale.

        Parameters
        ----------
        observation : np.ndarray
            The original observation from the base env (ignored).

        Returns
        -------
        np.ndarray
            Grayscale image of shape ``(obs_size, obs_size, 1)``,
            dtype ``uint8``.
        """
        frame = self._last_raw_frame
        if frame is None:
            # Fallback: return a black frame
            return np.zeros((self._obs_size, self._obs_size, 1), dtype=np.uint8)

        return _frame_to_obs(frame, self._obs_size)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Take a step and convert the observation to a grayscale image.

        Parameters
        ----------
        action : np.ndarray
            Action to pass to the base environment.

        Returns
        -------
        obs : np.ndarray
            Grayscale image observation.
        reward : float
            Reward from the base environment (unchanged).
        terminated : bool
            Termination flag (unchanged).
        truncated : bool
            Truncation flag (unchanged).
        info : dict
            Info dict with added ``"mlp_obs"`` key containing the
            original 8-element observation vector.
        """
        mlp_obs, reward, terminated, truncated, info = self.env.step(action)

        # Stash the raw frame for observation()
        self._last_raw_frame = info.get("frame")

        # Store original MLP observation for debugging/logging
        info["mlp_obs"] = mlp_obs

        # Convert via observation()
        obs = self.observation(mlp_obs)
        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment and return a grayscale image observation.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        options : dict, optional
            Additional reset options.

        Returns
        -------
        obs : np.ndarray
            Initial grayscale image observation.
        info : dict
            Info dict with added ``"mlp_obs"`` key.
        """
        mlp_obs, info = self.env.reset(seed=seed, options=options)

        # Stash the raw frame for observation()
        self._last_raw_frame = info.get("frame")

        # Store original MLP observation for debugging/logging
        info["mlp_obs"] = mlp_obs

        obs = self.observation(mlp_obs)
        return obs, info


def _frame_to_obs(frame: np.ndarray, obs_size: int) -> np.ndarray:
    """Convert a BGR game frame to a grayscale observation image.

    Parameters
    ----------
    frame : np.ndarray
        BGR image from the game window (any resolution).
    obs_size : int
        Target height and width (square).

    Returns
    -------
    np.ndarray
        Grayscale image of shape ``(obs_size, obs_size, 1)``,
        dtype ``uint8``.
    """
    # Lazy import to avoid breaking CI (Docker lacks libGL)
    import cv2

    # Convert BGR to grayscale
    if frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 2:
        gray = frame
    else:
        # Unexpected shape — take first channel
        gray = frame[:, :, 0] if frame.ndim == 3 else frame

    # Resize to target resolution
    resized = cv2.resize(gray, (obs_size, obs_size), interpolation=cv2.INTER_AREA)

    # Add channel dimension: (H, W) -> (H, W, 1)
    return resized[:, :, np.newaxis]
