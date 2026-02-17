"""CNN observation wrappers for game environments.

Provides two wrappers for converting MLP observations to CNN-compatible
image observations:

``CnnObservationWrapper``
    Training wrapper — converts to 84x84 grayscale single-channel image.
    Used with SB3's ``DummyVecEnv → VecFrameStack → VecTransposeImage``
    pipeline during training.

``CnnEvalWrapper``
    Evaluation wrapper — combines grayscale conversion, frame stacking,
    and CHW transpose in a single Gymnasium wrapper.  Produces the same
    observation shape ``(N, 84, 84)`` as the training pipeline without
    requiring VecEnv wrapping.  Preserves access to ``env.unwrapped``
    for oracle findings and step count.

Training usage::

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

    base_env = Breakout71Env(...)
    cnn_env = CnnObservationWrapper(base_env)

    vec_env = DummyVecEnv([lambda: cnn_env])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    vec_env = VecTransposeImage(vec_env)

    model = PPO("CnnPolicy", vec_env, device="xpu:1")

Evaluation usage::

    from src.platform.cnn_wrapper import CnnEvalWrapper

    base_env = Breakout71Env(...)
    eval_env = CnnEvalWrapper(base_env, frame_stack=4)

    model = PPO.load("ppo_breakout71.zip")
    obs, info = eval_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)

Notes
-----
- Frame stacking is handled externally by ``VecFrameStack`` (training)
  or internally by ``CnnEvalWrapper`` (evaluation).
- ``VecTransposeImage`` converts HWC → CHW for PyTorch's NatureCNN.
- Both wrappers pass through reward, terminated, truncated, and info
  unchanged.  The original MLP obs is available in ``info["mlp_obs"]``.
"""

from __future__ import annotations

import logging
from collections import deque
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
    """Wrap a BaseGameEnv to produce 84x84 grayscale image observations.

    This is a thin :class:`gymnasium.ObservationWrapper` that replaces the
    feature vector with a grayscale image derived from the raw game frame
    already captured by the base environment.

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


class CnnEvalWrapper(gym.Wrapper):
    """Gymnasium wrapper for evaluating CNN-trained models.

    Combines grayscale conversion, frame stacking, and HWC→CHW
    transpose into a single wrapper that produces observations
    matching the training pipeline's output:
    ``CnnObservationWrapper → DummyVecEnv → VecFrameStack → VecTransposeImage``.

    Unlike the training pipeline, this wrapper does NOT require
    VecEnv wrapping.  It operates at the Gymnasium level, preserving
    access to ``env.unwrapped`` for oracle findings and step counts.

    Parameters
    ----------
    env : gym.Env
        The base environment (a :class:`BaseGameEnv` subclass).  Must
        produce an ``info`` dict containing a ``"frame"`` key with a
        BGR ``np.ndarray``.
    frame_stack : int
        Number of frames to stack.  Default is 4.
    obs_size : int
        Target height and width for the square observation image.
        Default is 84.

    Attributes
    ----------
    observation_space : gym.spaces.Box
        ``Box(0, 255, (frame_stack, obs_size, obs_size), uint8)`` —
        CHW stacked grayscale frames.
    """

    def __init__(
        self,
        env: gym.Env,
        frame_stack: int = 4,
        obs_size: int = CNN_OBS_SIZE,
    ) -> None:
        super().__init__(env)
        self._frame_stack = frame_stack
        self._obs_size = obs_size

        # Frame buffer: deque of grayscale (H, W) arrays
        self._frames: deque[np.ndarray] = deque(maxlen=frame_stack)

        # Override observation space: CHW uint8 image stack
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(frame_stack, obs_size, obs_size),
            dtype=np.uint8,
        )

        logger.info(
            "CnnEvalWrapper: obs_size=%d, frame_stack=%d, space=%s",
            obs_size,
            frame_stack,
            self.observation_space.shape,
        )

    def _frame_to_grayscale(self, frame: np.ndarray | None) -> np.ndarray:
        """Convert a raw BGR frame to a grayscale (H, W) array.

        Parameters
        ----------
        frame : np.ndarray or None
            BGR image from the game window.  If None, returns a black
            frame.

        Returns
        -------
        np.ndarray
            Grayscale image of shape ``(obs_size, obs_size)``,
            dtype ``uint8``.
        """
        if frame is None:
            return np.zeros((self._obs_size, self._obs_size), dtype=np.uint8)

        # Reuse _frame_to_obs and squeeze out the channel dim
        obs_hwc = _frame_to_obs(frame, self._obs_size)  # (H, W, 1)
        return obs_hwc[:, :, 0]  # (H, W)

    def _get_stacked_obs(self) -> np.ndarray:
        """Stack buffered frames into a CHW array.

        Returns
        -------
        np.ndarray
            Stacked observation of shape ``(frame_stack, obs_size, obs_size)``,
            dtype ``uint8``.
        """
        return np.stack(list(self._frames), axis=0)  # (N, H, W) = CHW

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment and return a stacked CHW observation.

        The frame stack is initialised by repeating the first frame
        ``frame_stack`` times, matching ``VecFrameStack`` reset behaviour.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        options : dict, optional
            Additional reset options.

        Returns
        -------
        obs : np.ndarray
            Stacked CHW observation of shape
            ``(frame_stack, obs_size, obs_size)``.
        info : dict
            Info dict from the base environment, with added
            ``"mlp_obs"`` key containing the original observation.
        """
        mlp_obs, info = self.env.reset(seed=seed, options=options)

        # Extract frame and convert to grayscale
        frame = info.get("frame")
        gray = self._frame_to_grayscale(frame)

        # Fill the frame stack with the initial frame
        self._frames.clear()
        for _ in range(self._frame_stack):
            self._frames.append(gray)

        info["mlp_obs"] = mlp_obs
        return self._get_stacked_obs(), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Take a step and return a stacked CHW observation.

        Parameters
        ----------
        action : np.ndarray
            Action to pass to the base environment.

        Returns
        -------
        obs : np.ndarray
            Stacked CHW observation.
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

        # Extract frame and push onto stack
        frame = info.get("frame")
        gray = self._frame_to_grayscale(frame)
        self._frames.append(gray)

        info["mlp_obs"] = mlp_obs
        return self._get_stacked_obs(), reward, terminated, truncated, info
