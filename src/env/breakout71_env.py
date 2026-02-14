"""Breakout 71 Gymnasium environment for RL-driven game QA.

Wraps the Breakout 71 browser game running in a native Windows window.
Uses ``WindowCapture`` for frame acquisition, ``YoloDetector`` for
object detection, ``InputController`` for action injection, and a
battery of ``Oracle`` instances for bug detection.

Observation vector layout (from session1.md spec)::

    [paddle_x, ball_x, ball_y, ball_vx, ball_vy, bricks_norm]

6-element vector where:

- paddle_x    : normalised x-position of the paddle centre  [0.0, 1.0]
- ball_x/y    : normalised position of the ball centre       [0.0, 1.0]
- ball_vx/vy  : estimated velocity (frame delta), clipped    [-1.0, 1.0]
- bricks_norm : fraction of bricks remaining (count/initial) [0.0, 1.0]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Breakout71Env(gym.Env):
    """Gymnasium environment wrapping the Breakout 71 browser game.

    This environment captures frames from the game window, runs YOLO
    inference to extract game-object positions, converts them to a
    structured observation vector, and injects actions via
    ``InputController``.

    Parameters
    ----------
    window_title : str
        Title of the browser window running Breakout 71.
        Default is ``"Breakout - 71"``.
    yolo_weights : str or Path
        Path to the trained YOLOv8 weights file.
    max_steps : int
        Maximum steps per episode before truncation.  Default is 10000.
    render_mode : str, optional
        Gymnasium render mode (``"human"`` or ``"rgb_array"``).
    oracles : list, optional
        List of ``Oracle`` instances to attach.  If None, all default
        oracles are used.

    Attributes
    ----------
    observation_space : gym.spaces.Box
        6-element continuous observation vector:
        ``[paddle_x, ball_x, ball_y, ball_vx, ball_vy, bricks_norm]``.
        Positions in [0, 1], velocities in [-1, 1].
    action_space : gym.spaces.Discrete
        Discrete(3): 0=NOOP, 1=LEFT, 2=RIGHT.
        FIRE/Space is only used in ``reset()`` to start a new game.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        window_title: str = "Breakout - 71",
        yolo_weights: str | Path = "weights/best.pt",
        max_steps: int = 10_000,
        render_mode: Optional[str] = None,
        oracles: Optional[list[Any]] = None,
    ) -> None:
        super().__init__()

        self.window_title = window_title
        self.yolo_weights = Path(yolo_weights)
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Observation: 6-element vector
        # [paddle_x, ball_x, ball_y, ball_vx, ball_vy, bricks_norm]
        # Positions in [0, 1], velocities in [-1, 1], bricks_norm in [0, 1]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Actions: 0=NOOP, 1=LEFT, 2=RIGHT
        # FIRE/Space is only used in reset() to start a new game
        self.action_space = spaces.Discrete(3)

        # Internal state
        self._step_count: int = 0
        self._prev_ball_pos: tuple[float, float] | None = None
        self._bricks_total: int | None = None  # set on first reset
        self._oracles: list[Any] = oracles or []
        self._last_frame: np.ndarray | None = None

        # Sub-components (initialised lazily)
        self._capture = None  # WindowCapture instance
        self._detector = None  # YoloDetector instance
        self._input = None  # InputController instance

    def _lazy_init(self) -> None:
        """Lazily initialise capture, detector, and input sub-components.

        Called on the first ``reset()`` so that the env can be
        constructed without requiring a live game window (e.g. for
        testing or config validation).
        """
        raise NotImplementedError(
            "Lazy initialisation of capture/detector/input not yet implemented"
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment for a new episode.

        Parameters
        ----------
        seed : int, optional
            Random seed (for reproducibility of any stochastic elements).
        options : dict, optional
            Additional reset options.

        Returns
        -------
        obs : np.ndarray
            Initial observation vector.
        info : dict[str, Any]
            Auxiliary information (``"frame"``, ``"detections"``, etc.).
        """
        raise NotImplementedError("Breakout71Env.reset not yet implemented")

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one action and return the resulting transition.

        Parameters
        ----------
        action : int
            Discrete action (0=NOOP, 1=LEFT, 2=RIGHT).

        Returns
        -------
        obs : np.ndarray
            Observation vector after the action.
        reward : float
            Reward signal (e.g. +1 per brick destroyed, -1 on life lost).
        terminated : bool
            True if the game is over (all lives lost or all bricks cleared).
        truncated : bool
            True if ``max_steps`` has been reached.
        info : dict[str, Any]
            Auxiliary information including ``"frame"``, ``"score"``,
            ``"oracle_findings"``.
        """
        raise NotImplementedError("Breakout71Env.step not yet implemented")

    def render(self) -> np.ndarray | None:
        """Render the current frame.

        Returns
        -------
        np.ndarray or None
            BGR frame if ``render_mode="rgb_array"``, else None.
        """
        if self.render_mode == "rgb_array":
            return self._last_frame
        return None

    def close(self) -> None:
        """Release capture and input resources."""
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._detector = None
        self._input = None

    def _capture_frame(self) -> np.ndarray:
        """Capture a frame from the game window.

        Returns
        -------
        np.ndarray
            RGB image of the game window's client area.
        """
        raise NotImplementedError("Frame capture not yet implemented")

    def _detect_objects(self, frame: np.ndarray) -> dict[str, Any]:
        """Run YOLO inference on a frame and extract detections.

        Parameters
        ----------
        frame : np.ndarray
            BGR game frame.

        Returns
        -------
        dict[str, Any]
            Detection results keyed by class name:
            ``{"paddle": (cx, cy, w, h), "ball": (cx, cy, w, h),
              "bricks": [(cx, cy, w, h), ...], ...}``.
        """
        raise NotImplementedError("Object detection not yet implemented")

    def _build_observation(self, detections: dict[str, Any]) -> np.ndarray:
        """Convert YOLO detections into the flat observation vector.

        Parameters
        ----------
        detections : dict[str, Any]
            Detection results from ``_detect_objects``.

        Returns
        -------
        np.ndarray
            Float32 observation vector (6 elements):
            ``[paddle_x, ball_x, ball_y, ball_vx, ball_vy, bricks_norm]``.
        """
        raise NotImplementedError("Observation building not yet implemented")

    def _compute_reward(
        self, detections: dict[str, Any], info: dict[str, Any]
    ) -> float:
        """Compute the reward for the current step.

        Parameters
        ----------
        detections : dict[str, Any]
            Current detections.
        info : dict[str, Any]
            Current step info.

        Returns
        -------
        float
            Reward signal.
        """
        raise NotImplementedError("Reward computation not yet implemented")

    def _apply_action(self, action: int) -> None:
        """Send the chosen action to the game via InputController.

        Parameters
        ----------
        action : int
            Discrete action to apply.
        """
        raise NotImplementedError("Action application not yet implemented")

    def _run_oracles(
        self,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> list[Any]:
        """Run all attached oracles and collect findings.

        Parameters
        ----------
        obs : np.ndarray
            Current observation.
        reward : float
            Current reward.
        terminated : bool
            Terminated flag.
        truncated : bool
            Truncated flag.
        info : dict[str, Any]
            Step info dict.

        Returns
        -------
        list[Finding]
            Aggregated findings from all oracles.
        """
        raise NotImplementedError("Oracle execution not yet implemented")
