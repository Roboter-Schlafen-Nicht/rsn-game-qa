"""Hextris Gymnasium environment for RL-driven game QA.

Game-specific subclass of :class:`BaseGameEnv` implementing the Hextris
browser game.  Hextris is a hexagonal puzzle game where colored blocks
fall toward a central hexagon and the player rotates the hexagon to
match block colors on each side.

Key differences from Breakout 71:

- **Discrete action space**: ``Discrete(3)`` — 0=no-op, 1=rotate left,
  2=rotate right.  (Breakout uses continuous ``Box(-1, 1)``.)
- **No YOLO**: CNN-only pixel observations.  The game's visual state is
  the entire observation.
- **No levels/transitions**: Single continuous game until death.  No perk
  picker, no level clear.
- **Keyboard/JS rotation input**: ``MainHex.rotate(-1)`` and
  ``MainHex.rotate(1)`` via JS injection.
- **Game state via JS globals**: ``window.gameState`` (int),
  ``window.score`` (int), ``window.blocks`` (array).

The action space is ``Discrete(3)``:

- ``0`` — no action (do nothing)
- ``1`` — rotate hexagon counter-clockwise (left)
- ``2`` — rotate hexagon clockwise (right)

Observation is an 8-element MLP vector (matching platform convention)::

    [score_norm, block_count_norm, game_active, 0, 0, 0, 0, 0]

In practice, CNN mode (``--policy cnn``) is the primary observation
mode.  The MLP vector is a minimal placeholder.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces

from games.hextris.modal_handler import (
    DETECT_STATE_JS,
    DISMISS_GAME_OVER_JS,
    READ_GAME_STATE_JS,
    ROTATE_LEFT_JS,
    ROTATE_RIGHT_JS,
    START_GAME_JS,
)
from src.platform.base_env import BaseGameEnv

logger = logging.getLogger(__name__)


class HextrisEnv(BaseGameEnv):
    """Gymnasium environment wrapping the Hextris browser game.

    This environment captures frames from the game window, uses CNN
    pixel observations (no YOLO), and injects actions via JavaScript
    calls to ``MainHex.rotate()``.

    Parameters
    ----------
    window_title : str
        Title of the browser window running Hextris.
        Default is ``"HEXTRIS"``.
    yolo_weights : str or Path
        Path to YOLO weights file.  Not used by Hextris (CNN-only)
        but required by the platform interface.
    max_steps : int
        Maximum steps per episode before truncation.  Default is 5000.
    render_mode : str, optional
        Gymnasium render mode (``"human"`` or ``"rgb_array"``).
    oracles : list, optional
        List of ``Oracle`` instances to attach.
    driver : object, optional
        Selenium WebDriver instance for DOM interaction and input.
    device : str
        Device for YOLO inference.  Largely irrelevant for Hextris
        (CNN-only).  Default ``"auto"``.
    headless : bool
        If ``True``, use Selenium-based capture.  Default ``False``.
    reward_mode : str
        Reward strategy.  ``"survival"`` recommended for Hextris.
    game_over_detector : object, optional
        Pixel-based game-over detector.
    survival_bonus : float
        Per-step survival reward.  Default ``0.01``.
    browser_instance : object, optional
        Reference to ``BrowserInstance`` for crash recovery.
    """

    # No ball-lost or level-clear — game over is detected via JS gameState
    _GAME_OVER_CONFIRM_FRAMES: int = 3
    """Consecutive frames with gameState==2 before confirming game over."""

    def __init__(
        self,
        window_title: str = "HEXTRIS",
        yolo_weights: str | Path = "weights/best.pt",
        max_steps: int = 5_000,
        render_mode: str | None = None,
        oracles: list[Any] | None = None,
        driver: Any | None = None,
        device: str = "auto",
        headless: bool = False,
        reward_mode: str = "survival",
        game_over_detector: Any | None = None,
        survival_bonus: float = 0.01,
        browser_instance: Any | None = None,
    ) -> None:
        super().__init__(
            window_title=window_title,
            yolo_weights=yolo_weights,
            max_steps=max_steps,
            render_mode=render_mode,
            oracles=oracles,
            driver=driver,
            device=device,
            headless=headless,
            reward_mode=reward_mode,
            game_over_detector=game_over_detector,
            survival_bonus=survival_bonus,
            browser_instance=browser_instance,
        )

        # Observation: 8-element vector (minimal MLP placeholder)
        # [score_norm, block_count_norm, game_active, 0, 0, 0, 0, 0]
        self.observation_space = spaces.Box(
            low=np.zeros(8, dtype=np.float32),
            high=np.ones(8, dtype=np.float32),
            dtype=np.float32,
        )

        # Actions: Discrete(3) — 0=noop, 1=rotate_left, 2=rotate_right
        self.action_space = spaces.Discrete(3)

        # Game-specific state
        self._prev_score: int = 0
        self._game_over_count: int = 0

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def game_classes(self) -> list[str]:
        """Return empty class list (Hextris uses CNN-only observations).

        Returns
        -------
        list[str]
            Empty list.
        """
        return []

    def canvas_selector(self) -> str:
        """Return the CSS ID of the Hextris game canvas.

        Returns
        -------
        str
            ``"canvas"``
        """
        return "canvas"

    def build_observation(self, detections: dict[str, Any], *, reset: bool = False) -> np.ndarray:
        """Convert game state into the flat observation vector.

        For Hextris, YOLO detections are empty (no model).  The MLP
        observation is a minimal placeholder reading JS state.  The
        primary observation mode is CNN (84x84 grayscale pixels via
        ``CnnObservationWrapper``).

        Parameters
        ----------
        detections : dict[str, Any]
            Detection results (empty for Hextris).
        reset : bool
            If True, this is the first observation of a new episode.

        Returns
        -------
        np.ndarray
            Float32 observation vector (8 elements).
        """
        game_state = self._read_game_state()
        score = game_state.get("score", 0)
        block_count = game_state.get("blockCount", 0)
        running = game_state.get("running", False)

        # Normalise score to [0, 1] with soft cap at 10000
        score_norm = min(score / 10000.0, 1.0) if score > 0 else 0.0
        # Normalise block count (typical max ~30-50 on screen)
        block_norm = min(block_count / 50.0, 1.0)
        game_active = 1.0 if running else 0.0

        obs = np.array(
            [score_norm, block_norm, game_active, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )

        if reset:
            self._prev_score = score

        return obs

    def compute_reward(
        self,
        detections: dict[str, Any],
        terminated: bool,
        level_cleared: bool,
    ) -> float:
        """Compute the reward for the current step.

        In ``yolo`` mode (the game-specific reward), uses score delta
        from the JS bridge.  In ``survival`` mode, the base class
        handles reward computation.

        Parameters
        ----------
        detections : dict[str, Any]
            Current detections (empty for Hextris).
        terminated : bool
            Whether the episode ended this step.
        level_cleared : bool
            Always False for Hextris (no levels).

        Returns
        -------
        float
            Reward signal.
        """
        game_state = self._read_game_state()
        score = game_state.get("score", 0)

        # Score delta reward
        score_delta = score - self._prev_score
        reward = score_delta * 0.01

        # Small time penalty to encourage action
        reward -= 0.001

        # Terminal penalty
        if terminated:
            reward -= 5.0

        self._prev_score = score
        return reward

    def check_termination(self, detections: dict[str, Any]) -> tuple[bool, bool]:
        """Check whether the episode should terminate.

        Reads ``window.gameState`` via JS.  Game over when
        ``gameState == 2`` for ``_GAME_OVER_CONFIRM_FRAMES`` consecutive
        frames.

        Parameters
        ----------
        detections : dict[str, Any]
            Current detections (empty for Hextris).

        Returns
        -------
        terminated : bool
            True if the game is over.
        level_cleared : bool
            Always False (Hextris has no levels).
        """
        game_state = self._read_game_state()
        gs = game_state.get("gameState", 1)

        if gs == 2:  # Game over state
            self._game_over_count += 1
        else:
            self._game_over_count = 0

        terminated = self._game_over_count >= self._GAME_OVER_CONFIRM_FRAMES
        return terminated, False

    def apply_action(self, action: np.ndarray | int) -> None:
        """Send the chosen action to the game via JavaScript.

        Parameters
        ----------
        action : int or np.ndarray
            Discrete action: 0=noop, 1=rotate_left, 2=rotate_right.
        """
        if self._driver is None:
            return

        act = int(action)

        if act == 0:
            return  # No-op

        try:
            if act == 1:
                self._driver.execute_script(ROTATE_LEFT_JS)
            elif act == 2:
                self._driver.execute_script(ROTATE_RIGHT_JS)
        except Exception as exc:
            logger.debug("Action failed: %s", exc)

    def handle_modals(self, *, dismiss_game_over: bool = True) -> str:
        """Detect and handle game UI state.

        Hextris has simpler state management than Breakout — no perk
        picker, just start screen, gameplay, and game over.

        Parameters
        ----------
        dismiss_game_over : bool
            If True, restart the game on game over.

        Returns
        -------
        str
            Detected state: ``"gameplay"``, ``"game_over"``,
            ``"menu"``, or ``"unknown"``.
        """
        if self._driver is None:
            return "gameplay"

        try:
            state_info = self._driver.execute_script(DETECT_STATE_JS)
        except Exception as exc:
            logger.debug("State detection failed: %s", exc)
            return "unknown"

        if state_info is None:
            return "unknown"

        state = state_info.get("state", "unknown")

        if state == "game_over" and dismiss_game_over:
            try:
                self._driver.execute_script(DISMISS_GAME_OVER_JS)
                logger.info("Game over dismissed — game restarted")
            except Exception as exc:
                logger.warning("Game over dismiss failed: %s", exc)
            time.sleep(0.5)

        elif state == "menu":
            try:
                self._driver.execute_script(START_GAME_JS)
                logger.info("Game started from menu/start screen")
            except Exception as exc:
                logger.warning("Start game failed: %s", exc)
            time.sleep(0.5)

        return state

    def start_game(self) -> None:
        """Start or unpause the game.

        Calls ``resumeGame()`` or clicks the start button via JS.
        """
        if self._driver is None:
            return

        try:
            self._driver.execute_script(START_GAME_JS)
        except Exception as exc:
            logger.debug("Start game failed: %s", exc)

    def build_info(self, detections: dict[str, Any]) -> dict[str, Any]:
        """Build the game-specific portion of the info dict.

        Parameters
        ----------
        detections : dict[str, Any]
            Current detections (empty for Hextris).

        Returns
        -------
        dict[str, Any]
            Game-specific info entries.
        """
        game_state = self._read_game_state()
        return {
            "detections": detections,
            "score": game_state.get("score", 0),
            "block_count": game_state.get("blockCount", 0),
            "game_state_raw": game_state.get("gameState", -99),
            "game_over_count": self._game_over_count,
        }

    def terminal_reward(self) -> float:
        """Return the fixed terminal penalty for game-over via modal.

        Returns
        -------
        float
            ``-5.01`` (terminal penalty + time penalty).
        """
        return -5.0 - 0.001

    def on_reset_detections(self, detections: dict[str, Any]) -> bool:
        """Check whether reset detections are valid to start an episode.

        For Hextris (CNN-only, no YOLO), always accept the frame as
        long as the game is in a playable state.

        Parameters
        ----------
        detections : dict[str, Any]
            Detections from a reset attempt frame.

        Returns
        -------
        bool
            True if the game is in a playable state.
        """
        if self._driver is None:
            return True

        game_state = self._read_game_state()
        gs = game_state.get("gameState", -99)
        # Accept if playing (1) or about to play (0, -1, 4)
        return gs in (0, 1, -1, 4)

    def reset_termination_state(self) -> None:
        """Reset game-specific termination counters for a new episode."""
        self._prev_score = 0
        self._game_over_count = 0

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def _should_check_modals(self) -> bool:
        """Always check modals for Hextris.

        Hextris has no ball-loss heuristic like Breakout.  Always
        check for game-over state.

        Returns
        -------
        bool
            Always True.
        """
        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_game_state(self) -> dict[str, Any]:
        """Read game state (score, gameState, blockCount) via JS bridge.

        Returns
        -------
        dict[str, Any]
            ``{"score": int, "gameState": int, "blockCount": int,
            "running": bool}``
        """
        defaults: dict[str, Any] = {
            "score": 0,
            "gameState": -99,
            "blockCount": 0,
            "running": False,
        }
        if self._driver is None:
            return defaults
        try:
            result = self._driver.execute_script(READ_GAME_STATE_JS)
            if result is not None:
                return result
        except Exception as exc:
            logger.debug("Failed to read game state: %s", exc)
        return defaults
