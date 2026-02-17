"""Breakout 71 Gymnasium environment for RL-driven game QA.

Game-specific subclass of :class:`BaseGameEnv` implementing the
Breakout 71 browser game.  Uses ``WinCamCapture`` (Direct3D11, <1 ms)
or ``WindowCapture`` (PrintWindow fallback) for frame acquisition,
``YoloDetector`` for object detection, Selenium ``ActionChains`` for
paddle control, and Selenium WebDriver for modal handling (game-over,
perk picker, menu overlays).

In **headless** mode, capture uses Selenium screenshots
(``get_screenshot_as_png``) instead of Win32-based screen capture.
This is much slower (~2-3 FPS) but avoids capturing the host mouse
and does not require a physical display.

All input (paddle movement, canvas clicks) uses Selenium
``ActionChains`` in both native and headless modes.  This avoids
OS-level mouse events (e.g. ``pydirectinput``) which can trigger
unintended game behaviour (click-to-pause, focus loss, etc.) and
are inherently game-specific.

The action space is a continuous ``Box(-1, 1, shape=(1,))`` value that
maps directly to the absolute paddle position via
``ActionChains.move_to_element_with_offset()``.
A value of ``-1`` moves the paddle to the left edge of the game canvas,
``+1`` to the right edge, and ``0`` to the centre.

Observation vector layout (8 elements)::

    [paddle_x, ball_x, ball_y, ball_vx, ball_vy, bricks_norm,
     coins_norm, score_norm]

Where:

- paddle_x    : normalised x-position of the paddle centre  [0.0, 1.0]
- ball_x/y    : normalised position of the ball centre       [0.0, 1.0]
- ball_vx/vy  : estimated velocity (frame delta), clipped    [-1.0, 1.0]
- bricks_norm : fraction of bricks remaining (count/initial) [0.0, 1.0]
- coins_norm  : normalised coin count on screen (v1: 0.0)    [0.0, 1.0]
- score_norm  : normalised score delta since last step (v1: 0.0) [0.0, 1.0]

Note: The game uses coin-based scoring (bricks spawn coins caught by
paddle), not direct brick-based scoring.  The ``coins_norm`` and
``score_norm`` slots are placeholders for v1 and will be populated once
YOLO coin tracking or OCR/JS bridge is available.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from gymnasium import spaces

from games.breakout71.modal_handler import (
    CLICK_PERK_JS,
    DETECT_STATE_JS,
    DISMISS_GAME_OVER_JS,
    DISMISS_MENU_JS,
    MOVE_MOUSE_JS,
)
from src.platform.base_env import BaseGameEnv

logger = logging.getLogger(__name__)


class Breakout71Env(BaseGameEnv):
    """Gymnasium environment wrapping the Breakout 71 browser game.

    This environment captures frames from the game window, runs YOLO
    inference to extract game-object positions, converts them to a
    structured observation vector, and injects actions via Selenium
    ``ActionChains.move_to_element_with_offset()`` (both native and
    headless modes).

    Frame capture uses ``WinCamCapture`` (Direct3D11, <1 ms) in native
    mode or Selenium ``get_screenshot_as_png`` in headless mode.

    Selenium WebDriver is used for modal handling (game-over, perk
    picker, menu overlays) and all input (paddle movement, canvas
    clicks) in both modes.

    Parameters
    ----------
    window_title : str
        Title of the browser window running Breakout 71.
        Default is ``"Breakout"``.
    yolo_weights : str or Path
        Path to the trained YOLOv8 weights file.
    max_steps : int
        Maximum steps per episode before truncation.  Default is 10000.
    render_mode : str, optional
        Gymnasium render mode (``"human"`` or ``"rgb_array"``).
    oracles : list, optional
        List of ``Oracle`` instances to attach.  If None, no oracles
        are used.
    driver : object, optional
        Selenium WebDriver instance for modal handling and input.
        Required for both native and headless modes.  In native mode,
        capture still uses Win32 screen capture (wincam/PrintWindow)
        for speed.  In headless mode, capture also uses Selenium.
    device : str
        Device for YOLO inference: ``"auto"`` (default), ``"xpu"``,
        ``"cuda"``, ``"cpu"``.  Passed to ``YoloDetector``.
    headless : bool
        If ``True``, use Selenium-based capture instead of Win32 APIs.
        Input always uses Selenium in both modes.
        Requires ``driver`` to be set.  Much slower (~2-3 FPS) but does
        not capture the host mouse.  Default is ``False``.

    Attributes
    ----------
    observation_space : gym.spaces.Box
        8-element continuous observation vector.
    action_space : gym.spaces.Box
        Continuous Box(-1, 1, shape=(1,)).  The value maps to the
        absolute paddle position: -1 = left edge, 0 = centre,
        +1 = right edge of the game canvas.
    """

    # Termination thresholds
    _BALL_LOST_THRESHOLD: int = 5
    """Consecutive frames without ball detection before game-over."""

    _LEVEL_CLEAR_THRESHOLD: int = 3
    """Consecutive frames with zero bricks before level-cleared."""

    def __init__(
        self,
        window_title: str = "Breakout",
        yolo_weights: str | Path = "weights/best.pt",
        max_steps: int = 10_000,
        render_mode: Optional[str] = None,
        oracles: Optional[list[Any]] = None,
        driver: Optional[Any] = None,
        device: str = "auto",
        headless: bool = False,
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
        )

        # Observation: 8-element vector
        # [paddle_x, ball_x, ball_y, ball_vx, ball_vy,
        #  bricks_norm, coins_norm, score_norm]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Actions: continuous paddle position [-1, 1]
        # -1 = left edge, 0 = centre, +1 = right edge
        self.action_space = spaces.Box(
            low=np.float32(-1.0),
            high=np.float32(1.0),
            shape=(1,),
            dtype=np.float32,
        )

        # Game-specific state
        self._prev_ball_pos: tuple[float, float] | None = None
        self._bricks_total: int | None = None  # set on first reset
        self._prev_bricks_norm: float = 1.0

        # Canvas bounding rect (left, top) for JS mousemove dispatch
        self._canvas_rect: tuple[float, float] | None = None

        # Termination counters
        self._no_ball_count: int = 0
        self._no_bricks_count: int = 0

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def on_lazy_init(self) -> None:
        """Cache the canvas bounding rect for JS mousemove dispatch."""
        self._update_canvas_rect()

    def _update_canvas_rect(self) -> None:
        """Query the canvas bounding rect from the DOM.

        Caches ``(left, top)`` in viewport coordinates for use by
        ``apply_action()``.  Falls back to ``(0, 0)`` on error.
        """
        if self._driver is None:
            return
        try:
            rect = self._driver.execute_script(
                "var c = document.getElementById(arguments[0]);"
                "if (!c) return null;"
                "var r = c.getBoundingClientRect();"
                "return {left: r.left, top: r.top};",
                self.canvas_selector(),
            )
            if rect is not None:
                self._canvas_rect = (rect["left"], rect["top"])
                logger.info(
                    "Canvas rect: left=%.1f, top=%.1f",
                    self._canvas_rect[0],
                    self._canvas_rect[1],
                )
        except Exception as exc:
            logger.debug("Failed to get canvas rect: %s", exc)

    def game_classes(self) -> list[str]:
        """Return Breakout 71 YOLO class names.

        Returns
        -------
        list[str]
            ``["ball", "brick", "paddle", "powerup", "wall"]``
        """
        return ["ball", "brick", "paddle", "powerup", "wall"]

    def canvas_selector(self) -> str:
        """Return the CSS ID of the Breakout 71 game canvas.

        Returns
        -------
        str
            ``"game"``
        """
        return "game"

    def build_observation(
        self, detections: dict[str, Any], *, reset: bool = False
    ) -> np.ndarray:
        """Convert YOLO detections into the flat observation vector.

        Parameters
        ----------
        detections : dict[str, Any]
            Detection results from ``_detect_objects``.
        reset : bool
            If True, zero velocities and set ``_bricks_total`` from
            the current brick count (first frame of episode).

        Returns
        -------
        np.ndarray
            Float32 observation vector (8 elements).
        """
        # Extract paddle position (normalised cx)
        paddle = detections.get("paddle")
        paddle_x = paddle[0] if paddle is not None else 0.5

        # Extract ball position (normalised cx, cy)
        ball = detections.get("ball")
        ball_x = ball[0] if ball is not None else 0.5
        ball_y = ball[1] if ball is not None else 0.5

        # Compute velocity from frame delta
        if reset or self._prev_ball_pos is None or ball is None:
            ball_vx, ball_vy = 0.0, 0.0
        else:
            ball_vx = ball_x - self._prev_ball_pos[0]
            ball_vy = ball_y - self._prev_ball_pos[1]

        # Update previous ball position
        if ball is not None:
            self._prev_ball_pos = (ball_x, ball_y)
        elif reset:
            self._prev_ball_pos = None

        # Brick normalisation
        bricks = detections.get("bricks", [])
        bricks_left = len(bricks)

        if reset or self._bricks_total is None:
            # Retry detection if 0 bricks on first frame (transition
            # screen or slow render can cause false zero).
            # Guard: only retry when subsystems are initialized.
            if reset and bricks_left == 0 and self._initialized:
                for _retry in range(3):
                    time.sleep(0.2)
                    retry_frame = self._capture_frame()
                    retry_dets = self._detect_objects(retry_frame)
                    retry_bricks = len(retry_dets.get("bricks", []))
                    if retry_bricks > 0:
                        bricks_left = retry_bricks
                        bricks = retry_dets.get("bricks", [])
                        break
            self._bricks_total = max(bricks_left, 1)

        bricks_norm = bricks_left / self._bricks_total

        # Placeholder slots for v1
        coins_norm = 0.0
        score_norm = 0.0

        obs = np.array(
            [
                paddle_x,
                ball_x,
                ball_y,
                np.clip(ball_vx, -1.0, 1.0),
                np.clip(ball_vy, -1.0, 1.0),
                np.clip(bricks_norm, 0.0, 1.0),
                coins_norm,
                score_norm,
            ],
            dtype=np.float32,
        )

        return obs

    def compute_reward(
        self,
        detections: dict[str, Any],
        terminated: bool,
        level_cleared: bool,
    ) -> float:
        """Compute the reward for the current step.

        Parameters
        ----------
        detections : dict[str, Any]
            Current detections.
        terminated : bool
            Whether the episode ended this step.
        level_cleared : bool
            Whether the level was cleared (all bricks destroyed).

        Returns
        -------
        float
            Reward signal.
        """
        # Brick destruction reward
        bricks_left = len(detections.get("bricks", []))
        bricks_total = self._bricks_total if self._bricks_total else 1
        bricks_norm = bricks_left / bricks_total
        brick_delta = self._prev_bricks_norm - bricks_norm
        reward = brick_delta * 10.0

        # Score delta reward (placeholder -- will activate with OCR/JS bridge)
        score_delta = 0.0
        reward += score_delta * 0.01

        # Time penalty
        reward -= 0.01

        # Terminal rewards
        if terminated and level_cleared:
            reward += 5.0
        elif terminated:
            reward -= 5.0

        # Update state for next step
        self._prev_bricks_norm = bricks_norm

        return reward

    def check_termination(self, detections: dict[str, Any]) -> tuple[bool, bool]:
        """Check whether the episode should terminate.

        Reads ``_no_ball_count`` (already updated by
        ``_check_late_game_over``) and updates ``_no_bricks_count``.

        Parameters
        ----------
        detections : dict[str, Any]
            Current YOLO detections.

        Returns
        -------
        terminated : bool
            True if the episode should end.
        level_cleared : bool
            True if the level was cleared.
        """
        brick_count = len(detections.get("bricks", []))

        if brick_count == 0 and self._bricks_total is not None:
            self._no_bricks_count += 1
        else:
            self._no_bricks_count = 0

        level_cleared = self._no_bricks_count >= self._LEVEL_CLEAR_THRESHOLD
        game_over = (
            self._no_ball_count >= self._BALL_LOST_THRESHOLD
            and self._bricks_total is not None
            and brick_count == self._prev_brick_count()
        )
        terminated = level_cleared or game_over

        return terminated, level_cleared

    def apply_action(self, action: "np.ndarray") -> None:
        """Send the chosen action to the game via JavaScript mousemove.

        Dispatches a synthetic ``mousemove`` event on the game canvas
        with the correct ``clientX`` value.  The game reads
        ``e.clientX`` directly from mousemove events to set the paddle
        position.

        This uses ``driver.execute_script()`` (~5 ms) instead of
        Selenium ``ActionChains`` (~270 ms) for a ~50× speedup on the
        action dispatch path.

        Parameters
        ----------
        action : np.ndarray
            Continuous action value in ``[-1, 1]``.  Accepts a scalar,
            0-d array, or shape ``(1,)`` array.  Maps to absolute
            paddle position: -1 = left edge, 0 = centre, +1 = right
            edge of the game canvas.
        """
        if self._driver is None or self._game_canvas is None:
            return
        if self._canvas_size is None:
            return

        # Normalize: accept scalar, 0-d, or (1,) array
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 1:
            raise ValueError(f"Expected action of size 1, got size {action.size}")

        value = float(np.clip(action[0], -1.0, 1.0))
        canvas_w, canvas_h = self._canvas_size

        # Convert [-1, 1] to clientX in viewport coordinates.
        # The canvas left edge is at _canvas_rect.left; we need the
        # absolute viewport pixel.  Use the cached rect or fall back
        # to element-centre-based offset.
        if self._canvas_rect is not None:
            # clientX = rect.left + (value + 1) / 2 * rect.width
            client_x = self._canvas_rect[0] + (value + 1.0) / 2.0 * canvas_w
            # clientY at 90% of canvas height
            client_y = self._canvas_rect[1] + 0.9 * canvas_h
        else:
            # Fallback: assume canvas starts at (0, 0)
            client_x = (value + 1.0) / 2.0 * canvas_w
            client_y = 0.9 * canvas_h

        try:
            self._driver.execute_script(
                MOVE_MOUSE_JS,
                self.canvas_selector(),
                client_x,
                client_y,
            )
        except Exception as exc:
            logger.debug("Action failed: %s", exc)

    def handle_modals(self, *, dismiss_game_over: bool = True) -> str:
        """Detect and handle game UI state (modals, game over, perks).

        Uses JavaScript execution via Selenium to query the game DOM
        and dismiss modals as needed.  Headless mode also relies on
        Selenium for screenshots and ``ActionChains`` input, but in
        native mode this DOM-level interaction is the only Selenium
        use — all observation and control is otherwise pixel-based.

        Parameters
        ----------
        dismiss_game_over : bool
            If True (default), dismiss game-over modals by clicking
            "Restart game".  If False, only detect the state but leave
            the modal in place.  ``step()`` passes False so that
            game-over terminates the episode; ``reset()`` passes True
            (the default) to clear the modal before starting a new
            episode.

        Returns
        -------
        str
            The detected game state (``"gameplay"``, ``"game_over"``,
            ``"perk_picker"``, ``"menu"``, or ``"unknown"``).
        """
        if self._driver is None:
            return "gameplay"

        try:
            state_info = self._driver.execute_script(DETECT_STATE_JS)
        except Exception as exc:
            logger.debug("State detection failed: %s", exc)
            return "unknown"

        if state_info is None:
            logger.warning("State detection returned None")
            return "unknown"

        state = state_info.get("state", "unknown")
        details = state_info.get("details", {})
        logger.info(
            "Game state: %s (buttons=%s)",
            state,
            details.get("buttonTexts", []),
        )

        if state == "perk_picker":
            try:
                result = self._driver.execute_script(CLICK_PERK_JS)
                logger.info(
                    "Perk picker: clicked button %d (%s)",
                    result.get("clicked", -1),
                    result.get("text", "?"),
                )
            except Exception:
                pass
            time.sleep(0.5)

        elif state == "game_over":
            if dismiss_game_over:
                try:
                    result = self._driver.execute_script(DISMISS_GAME_OVER_JS)
                    logger.info(
                        "Game over dismissed: %s (%s)",
                        result.get("action", "?"),
                        result.get("text", ""),
                    )
                except Exception as exc:
                    logger.warning("Game over dismiss failed: %s", exc)
                time.sleep(1.0)
            else:
                logger.info("Game over detected — episode will terminate")

        elif state == "menu":
            try:
                self._driver.execute_script(DISMISS_MENU_JS)
                logger.info("Menu dismissed")
            except Exception:
                pass
            time.sleep(0.5)

        elif state == "unknown":
            logger.warning("Unknown game state: details=%s", details)

        return state

    def start_game(self) -> None:
        """Start or unpause the game.

        Sets ``gameState.running = true`` and
        ``gameState.ballStickToPuck = false`` via JavaScript.  This
        is more reliable than dispatching click/mouseup events because
        the game's ``play()`` function is module-scoped and its
        ``async applyFullScreenChoice()`` guard can silently block
        synthetic events in headless Chrome.

        Falls back to ``ActionChains.click()`` if JS execution fails.
        """
        if self._driver is None:
            return

        try:
            self._driver.execute_script(
                "gameState.running = true;gameState.ballStickToPuck = false;"
            )
        except Exception:
            # Fallback: try ActionChains click
            if self._game_canvas is not None:
                try:
                    from selenium.webdriver.common.action_chains import ActionChains

                    ActionChains(self._driver).move_to_element(
                        self._game_canvas
                    ).click().perform()
                except Exception as exc:
                    logger.debug("Canvas click failed: %s", exc)

    def build_info(self, detections: dict[str, Any]) -> dict[str, Any]:
        """Build the game-specific portion of the info dict.

        The base class adds ``"frame"`` and ``"step"`` automatically.

        Parameters
        ----------
        detections : dict[str, Any]
            Current YOLO detections.

        Returns
        -------
        dict[str, Any]
            Game-specific info entries.
        """
        paddle = detections.get("paddle")
        ball = detections.get("ball")

        return {
            "detections": detections,
            "ball_pos": [ball[0], ball[1]] if ball is not None else None,
            "paddle_pos": [paddle[0], paddle[1]] if paddle is not None else None,
            "brick_count": len(detections.get("bricks", [])),
            "score": 0.0,  # placeholder for v1
            "no_ball_count": self._no_ball_count,
        }

    def terminal_reward(self) -> float:
        """Return the fixed terminal penalty for game-over via modal.

        Returns
        -------
        float
            ``-5.01`` (terminal penalty + time penalty).
        """
        return -5.0 - 0.01

    def on_reset_detections(self, detections: dict[str, Any]) -> bool:
        """Check whether the ball is detected (required to start playing).

        Parameters
        ----------
        detections : dict[str, Any]
            Detections from a reset attempt frame.

        Returns
        -------
        bool
            True if a ball is detected.
        """
        return detections.get("ball") is not None

    def reset_termination_state(self) -> None:
        """Reset game-specific termination counters for a new episode."""
        self._prev_bricks_norm = 1.0
        self._no_ball_count = 0
        self._no_bricks_count = 0

    # ------------------------------------------------------------------
    # Modal throttling hooks
    # ------------------------------------------------------------------

    def _should_check_modals(self) -> bool:
        """Check modals only when ball has been missing.

        Returns
        -------
        bool
            True when ``_no_ball_count > 0``.
        """
        return self._no_ball_count > 0

    def _check_late_game_over(self, detections: dict[str, Any]) -> bool:
        """Check for game-over on the 0→1 ball-miss transition.

        When the ball was visible last step and disappears this step,
        check for a game-over modal immediately to prevent spurious
        positive rewards from modal-occluded brick detections.

        Parameters
        ----------
        detections : dict[str, Any]
            Current YOLO detections.

        Returns
        -------
        bool
            True if late game-over modal detected.
        """
        ball_detected = detections.get("ball") is not None

        if not ball_detected:
            self._no_ball_count += 1
            if self._no_ball_count == 1:
                late_state = self.handle_modals(dismiss_game_over=False)
                if late_state == "game_over":
                    return True
        else:
            self._no_ball_count = 0

        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prev_brick_count(self) -> int:
        """Return the previous brick count based on ``_prev_bricks_norm``.

        Returns
        -------
        int
            Estimated previous brick count.
        """
        if self._bricks_total is None:
            return 0
        return round(self._prev_bricks_norm * self._bricks_total)
