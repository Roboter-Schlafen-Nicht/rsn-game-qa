"""Breakout 71 Gymnasium environment for RL-driven game QA.

Wraps the Breakout 71 browser game running in a native Windows window.
Uses ``WinCamCapture`` (Direct3D11, <1 ms) or ``WindowCapture`` (PrintWindow
fallback) for frame acquisition, ``YoloDetector`` for object detection,
``pydirectinput`` for pixel-based paddle control, and Selenium WebDriver
for **modal handling only** (game-over, perk picker, menu overlays).

In **headless** mode, capture uses Selenium screenshots (``get_screenshot_as_png``)
and paddle control uses ``ActionChains.move_to_element_with_offset()``.  This is
much slower (~2-3 FPS) but avoids capturing the host mouse and does not
require a physical display.

The action space is a continuous ``Box(-1, 1, shape=(1,))`` value that
maps directly to the absolute paddle screen position via
``pydirectinput.moveTo()`` (native) or ``ActionChains`` (headless).
A value of ``-1`` moves the paddle to the left edge of the client area,
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

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JavaScript snippets for modal handling ONLY (DOM overlays)
# ---------------------------------------------------------------------------

DETECT_STATE_JS = """
return (function() {
    var result = {state: "gameplay", details: {}};

    var hasAlert = document.body.classList.contains('has-alert-open');
    var popup = document.getElementById('popup');
    var closeBtn = document.getElementById('close-modale');

    if (!hasAlert) {
        result.state = "gameplay";
        return result;
    }

    if (!popup) {
        result.state = "unknown";
        return result;
    }

    var popupText = popup.innerText || "";
    var buttons = popup.querySelectorAll('button');
    var buttonTexts = [];
    for (var i = 0; i < buttons.length; i++) {
        buttonTexts.push(buttons[i].innerText.trim());
    }
    result.details.buttonTexts = buttonTexts;

    var closeBtnVisible = false;
    if (closeBtn) {
        var style = window.getComputedStyle(closeBtn);
        closeBtnVisible = (style.display !== 'none'
                           && style.visibility !== 'hidden');
    }

    if (!closeBtnVisible && buttons.length >= 2) {
        result.state = "perk_picker";
        result.details.numPerks = buttons.length;
        return result;
    }

    if (closeBtnVisible) {
        var hasRestart = false;
        for (var j = 0; j < buttonTexts.length; j++) {
            var t = buttonTexts[j].toLowerCase();
            if (t.indexOf("new") >= 0 || t.indexOf("restart") >= 0 ||
                t.indexOf("run") >= 0 || t.indexOf("yes") >= 0 ||
                t.indexOf("again") >= 0) {
                hasRestart = true;
                break;
            }
        }
        if (hasRestart || popupText.toLowerCase().indexOf("game over") >= 0 ||
            popupText.toLowerCase().indexOf("score") >= 0) {
            result.state = "game_over";
        } else {
            result.state = "menu";
        }
        return result;
    }

    result.state = "unknown";
    return result;
})();
"""

CLICK_PERK_JS = """
return (function() {
    var popup = document.getElementById('popup');
    if (!popup) return {clicked: -1, text: ""};
    var buttons = popup.querySelectorAll('button');
    if (buttons.length === 0) return {clicked: -1, text: ""};
    var idx = Math.floor(Math.random() * buttons.length);
    var text = buttons[idx].innerText.trim();
    buttons[idx].click();
    return {clicked: idx, text: text};
})();
"""

DISMISS_GAME_OVER_JS = """
return (function() {
    var popup = document.getElementById('popup');
    var closeBtn = document.getElementById('close-modale');
    if (popup) {
        var buttons = popup.querySelectorAll('button');
        for (var i = 0; i < buttons.length; i++) {
            var t = buttons[i].innerText.trim().toLowerCase();
            if (t.indexOf("new") >= 0 || t.indexOf("restart") >= 0 ||
                t.indexOf("run") >= 0 || t.indexOf("yes") >= 0 ||
                t.indexOf("again") >= 0) {
                buttons[i].click();
                return {action: "restart_button",
                        text: buttons[i].innerText.trim()};
            }
        }
    }
    if (closeBtn) {
        closeBtn.click();
        return {action: "close_button", text: ""};
    }
    return {action: "none", text: ""};
})();
"""

DISMISS_MENU_JS = """
return (function() {
    var closeBtn = document.getElementById('close-modale');
    if (closeBtn) {
        closeBtn.click();
        return {action: "close_button"};
    }
    document.dispatchEvent(
        new KeyboardEvent('keydown', {key: 'Escape', code: 'Escape'}));
    return {action: "escape_key"};
})();
"""


def _get_client_origin(hwnd: int) -> tuple[int, int]:
    """Return screen-absolute (x, y) of the client area's top-left corner.

    Parameters
    ----------
    hwnd : int
        Window handle.

    Returns
    -------
    tuple[int, int]
        ``(screen_x, screen_y)`` of client area origin.
    """
    import win32gui

    return win32gui.ClientToScreen(hwnd, (0, 0))


def _norm_to_screen(
    x_norm: float,
    y_norm: float,
    client_w: int,
    client_h: int,
    client_origin: tuple[int, int],
) -> tuple[int, int]:
    """Convert normalised [0,1] coords to absolute screen pixels.

    Parameters
    ----------
    x_norm, y_norm : float
        Normalised position within the captured frame.
    client_w, client_h : int
        Client area dimensions (from capture).
    client_origin : tuple[int, int]
        Screen-absolute (x, y) of client area top-left.

    Returns
    -------
    tuple[int, int]
        ``(screen_x, screen_y)`` absolute pixel coordinates.
    """
    x_norm_clamped = max(0.0, min(1.0, x_norm))
    y_norm_clamped = max(0.0, min(1.0, y_norm))
    x_client = min(int(x_norm_clamped * client_w), max(client_w - 1, 0))
    y_client = min(int(y_norm_clamped * client_h), max(client_h - 1, 0))
    return client_origin[0] + x_client, client_origin[1] + y_client


class Breakout71Env(gym.Env):
    """Gymnasium environment wrapping the Breakout 71 browser game.

    This environment captures frames from the game window, runs YOLO
    inference to extract game-object positions, converts them to a
    structured observation vector, and injects actions via
    ``pydirectinput.moveTo()`` (native mode) or Selenium
    ``ActionChains`` (headless mode).

    Selenium WebDriver is used for modal handling (game-over, perk
    picker, menu overlays).  In headless mode it is also used for
    frame capture (``get_screenshot_as_png``) and paddle input
    (``move_to_element_with_offset``).

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
        Selenium WebDriver instance for modal handling (and for
        capture/input in headless mode).  If provided in native mode,
        the env uses JavaScript execution to detect and dismiss
        game-over, perk-picker, and menu modals.  In headless mode,
        this parameter is **required**.
    device : str
        Device for YOLO inference: ``"auto"`` (default), ``"xpu"``,
        ``"cuda"``, ``"cpu"``.  Passed to ``YoloDetector``.
    headless : bool
        If ``True``, use Selenium-based capture and input instead of
        Win32 APIs (pydirectinput/WinCamCapture/WindowCapture).
        Requires ``driver`` to be set.  Much slower (~2-3 FPS) but does
        not capture the host mouse.  Default is ``False``.

    Attributes
    ----------
    observation_space : gym.spaces.Box
        8-element continuous observation vector.
    action_space : gym.spaces.Box
        Continuous Box(-1, 1, shape=(1,)).  The value maps to the
        absolute paddle position: -1 = left edge, 0 = centre,
        +1 = right edge of the client area.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

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
        super().__init__()

        self.window_title = window_title
        self.yolo_weights = Path(yolo_weights)
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.device = device
        self.headless = headless

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

        # Internal state
        self._step_count: int = 0
        self._prev_ball_pos: tuple[float, float] | None = None
        self._bricks_total: int | None = None  # set on first reset
        self._prev_bricks_norm: float = 1.0
        self._oracles: list[Any] = oracles or []
        self._last_frame: np.ndarray | None = None

        # Termination counters
        self._no_ball_count: int = 0
        self._no_bricks_count: int = 0

        # Sub-components (initialised lazily)
        self._capture = None  # WinCamCapture or WindowCapture instance
        self._detector = None  # YoloDetector instance
        self._driver = driver  # Selenium WebDriver (modals only)
        self._initialized: bool = False

        # Pixel-based control state
        self._client_origin: tuple[int, int] | None = None  # screen coords

        # Headless mode state (Selenium-based capture/input)
        self._game_canvas: Any | None = None  # Selenium WebElement
        self._canvas_size: tuple[int, int] | None = None  # (width, height)

    def _lazy_init(self) -> None:
        """Lazily initialise capture, detector, and input sub-components.

        Called on the first ``reset()`` so that the env can be
        constructed without requiring a live game window (e.g. for
        testing or config validation).

        All imports are performed inside this method to avoid breaking
        CI in Docker where pywin32/wincam/pydirectinput are unavailable.

        **Native mode** (default): Prefers ``WinCamCapture`` (Direct3D11,
        <1 ms per frame) and falls back to ``WindowCapture`` (PrintWindow,
        ~25 ms) if wincam is unavailable.

        **Headless mode**: Skips Win32 capture/input entirely.  Uses
        Selenium ``get_screenshot_as_png`` for frames and
        ``ActionChains`` for paddle control.  Finds the ``#game`` canvas
        element for coordinate mapping.
        """
        if self._initialized:
            return

        from src.perception.yolo_detector import YoloDetector

        if self.headless:
            # -- Headless: Selenium-based capture/input --------------------
            if self._driver is None:
                raise RuntimeError(
                    "Headless mode requires a Selenium WebDriver "
                    "(pass driver= to Breakout71Env)"
                )

            # Find the game canvas element for coordinate mapping
            try:
                from selenium.webdriver.common.by import By

                self._game_canvas = self._driver.find_element(By.ID, "game")
            except Exception:
                logger.warning("Canvas #game not found, falling back to <body>")
                from selenium.webdriver.common.by import By

                self._game_canvas = self._driver.find_element(By.TAG_NAME, "body")

            # Read canvas size for action coordinate mapping
            size = self._game_canvas.size
            self._canvas_size = (size["width"], size["height"])
            logger.info(
                "Headless mode: canvas %dx%d",
                self._canvas_size[0],
                self._canvas_size[1],
            )
        else:
            # -- Native: Win32-based capture/input -------------------------
            try:
                from src.capture.wincam_capture import WinCamCapture

                self._capture = WinCamCapture(window_title=self.window_title, fps=60)
                logger.info(
                    "Capture: WinCamCapture (Direct3D11) HWND=%s, %dx%d",
                    self._capture.hwnd,
                    self._capture.width,
                    self._capture.height,
                )
            except (ImportError, RuntimeError, OSError) as exc:
                logger.warning(
                    "WinCamCapture unavailable (%s), falling back to PrintWindow",
                    exc,
                )
                from src.capture.window_capture import WindowCapture

                self._capture = WindowCapture(window_title=self.window_title)
                logger.info(
                    "Capture: WindowCapture (PrintWindow) HWND=%s, %dx%d",
                    self._capture.hwnd,
                    self._capture.width,
                    self._capture.height,
                )

            # Client origin for pydirectinput coordinate mapping
            self._update_client_origin()

            # Disable pydirectinput's built-in 100ms sleep
            try:
                import pydirectinput

                pydirectinput.PAUSE = 0
            except ImportError:
                logger.debug("pydirectinput not available (CI environment)")

        # -- YOLO detector (shared by both modes) --------------------------
        self._detector = YoloDetector(
            weights_path=self.yolo_weights,
            device=self.device,
        )
        self._detector.load()

        self._initialized = True

    def _update_client_origin(self) -> None:
        """Update the screen-absolute origin of the client area.

        Reads the window handle from the capture subsystem and converts
        client (0,0) to screen coordinates.  Called during init and on
        each reset to track window movement.
        """
        if self._capture is None or self._capture.hwnd == 0:
            self._client_origin = None
            return

        try:
            self._client_origin = _get_client_origin(self._capture.hwnd)
            logger.debug("Client origin: (%d, %d)", *self._client_origin)
        except Exception as exc:
            logger.warning("Failed to get client origin: %s", exc)
            self._client_origin = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment for a new episode.

        Handles game-over modals, perk picker screens, and initial
        game start by using Selenium JavaScript execution for modal
        dismissal and pydirectinput for canvas clicks.

        Parameters
        ----------
        seed : int, optional
            Random seed (for reproducibility of any stochastic elements).
        options : dict, optional
            Additional reset options.

        Returns
        -------
        obs : np.ndarray
            Initial observation vector (8 elements).
        info : dict[str, Any]
            Auxiliary information (``"frame"``, ``"detections"``, etc.).
        """
        super().reset(seed=seed)

        if not self._initialized:
            self._lazy_init()

        # Refresh client origin (window may have moved) — native only
        if not self.headless:
            self._update_client_origin()

        # Dismiss any modals and start the game
        detections: dict[str, Any] = {}
        for attempt in range(5):
            logger.info("reset() attempt %d/5", attempt + 1)
            self._handle_game_state()
            self._click_canvas()
            time.sleep(0.5)

            frame = self._capture_frame()
            detections = self._detect_objects(frame)

            ball = detections.get("ball")
            brick_count = len(detections.get("bricks", []))
            logger.info(
                "reset() attempt %d: ball=%s, bricks=%d",
                attempt + 1,
                ball is not None,
                brick_count,
            )

            if ball is not None:
                break

        if detections.get("ball") is None:
            raise RuntimeError(
                "Breakout71Env.reset() failed to detect a ball after "
                "5 attempts; the game may not have initialized correctly."
            )

        # Build observation with reset semantics
        obs = self._build_observation(detections, reset=True)

        # Reset episode counters
        self._step_count = 0
        self._prev_bricks_norm = 1.0
        self._no_ball_count = 0
        self._no_bricks_count = 0

        # Build info dict
        info = self._build_info(detections)

        # Clear and notify oracles
        for oracle in self._oracles:
            oracle.clear()
            oracle.on_reset(obs, info)

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one action and return the resulting transition.

        Parameters
        ----------
        action : np.ndarray
            Continuous action array of shape ``(1,)`` with value in
            ``[-1, 1]``.  Maps to absolute paddle position: -1 = left
            edge, 0 = centre, +1 = right edge of the client area.

        Returns
        -------
        obs : np.ndarray
            Observation vector after the action (8 elements).
        reward : float
            Reward signal.
        terminated : bool
            True if the game is over (ball lost or level cleared).
        truncated : bool
            True if ``max_steps`` has been reached.
        info : dict[str, Any]
            Auxiliary information including ``"frame"``, ``"score"``,
            ``"oracle_findings"``.
        """
        # Apply action — no artificial throttle; the pipeline runs as
        # fast as capture + inference allow.
        self._apply_action(action)

        # Handle any modals that appeared mid-episode (perk picker,
        # game over, menu).  Without this the modal overlay blocks the
        # game canvas causing YOLO to miss the ball and the episode to
        # terminate prematurely.
        mid_state = self._handle_game_state()
        if mid_state in ("game_over", "perk_picker", "menu"):
            self._click_canvas()
            time.sleep(0.3)

        # Capture and detect
        frame = self._capture_frame()
        detections = self._detect_objects(frame)

        # Build observation
        obs = self._build_observation(detections)

        # Update termination counters
        ball_detected = detections.get("ball") is not None
        brick_count = len(detections.get("bricks", []))

        if not ball_detected:
            self._no_ball_count += 1
        else:
            self._no_ball_count = 0

        if brick_count == 0 and self._bricks_total is not None:
            self._no_bricks_count += 1
        else:
            self._no_bricks_count = 0

        # Increment step counter (before truncation check so max_steps is
        # the exact number of transitions allowed per episode)
        self._step_count += 1

        # Determine termination
        level_cleared = self._no_bricks_count >= self._LEVEL_CLEAR_THRESHOLD
        game_over = (
            self._no_ball_count >= self._BALL_LOST_THRESHOLD
            and self._bricks_total is not None
            and brick_count == self._prev_brick_count()
        )
        terminated = level_cleared or game_over
        truncated = self._step_count >= self.max_steps

        # Compute reward
        reward = self._compute_reward(detections, terminated, level_cleared)

        # Build info dict
        info = self._build_info(detections)

        # Run oracles
        findings = self._run_oracles(obs, reward, terminated, truncated, info)
        info["oracle_findings"] = findings

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render the current frame.

        Returns
        -------
        np.ndarray or None
            RGB frame if ``render_mode="rgb_array"``, else None.
        """
        if self.render_mode == "rgb_array":
            return self._last_frame
        return None

    @property
    def step_count(self) -> int:
        """Return the current step count (read-only).

        Returns
        -------
        int
            Number of steps taken in the current episode.
        """
        return self._step_count

    def close(self) -> None:
        """Release capture resources.

        Does **not** close the Selenium driver — that is owned by the
        caller (e.g. ``train_rl.py``).
        """
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._detector = None
        self._client_origin = None
        self._game_canvas = None
        self._canvas_size = None
        self._initialized = False

    # -- Private helpers -------------------------------------------------------

    def _capture_frame(self) -> np.ndarray:
        """Capture a frame from the game window.

        In native mode, uses ``WinCamCapture`` or ``WindowCapture``.
        In headless mode, uses Selenium ``get_screenshot_as_png``.

        Returns
        -------
        np.ndarray
            BGR image of the game window's client area.
        """
        if self.headless:
            return self._capture_frame_headless()
        frame = self._capture.capture_frame()
        self._last_frame = frame
        return frame

    def _capture_frame_headless(self) -> np.ndarray:
        """Capture a frame via Selenium screenshot.

        Takes a full-page PNG screenshot, decodes it to a BGR numpy
        array.  Slower than native capture (~50-100 ms) but works
        without a physical display.

        Returns
        -------
        np.ndarray
            BGR image decoded from the screenshot.
        """
        import cv2

        png_bytes = self._driver.get_screenshot_as_png()
        nparr = np.frombuffer(png_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        self._last_frame = frame
        return frame

    def _detect_objects(self, frame: np.ndarray) -> dict[str, Any]:
        """Run YOLO inference on a frame and extract detections.

        Parameters
        ----------
        frame : np.ndarray
            BGR game frame.

        Returns
        -------
        dict[str, Any]
            Detection results from ``YoloDetector.detect_to_game_state``.
        """
        h, w = frame.shape[:2]
        return self._detector.detect_to_game_state(frame, w, h)

    def _build_observation(
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

    def _compute_reward(
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

    def _apply_action(self, action: "np.ndarray") -> None:
        """Send the chosen action to the game.

        In native mode, uses ``pydirectinput.moveTo()`` to set the
        mouse position on screen.  In headless mode, uses Selenium
        ``ActionChains.move_to_element_with_offset()`` on the game
        canvas.

        Parameters
        ----------
        action : np.ndarray
            Continuous action value in ``[-1, 1]``.  Accepts a scalar,
            0-d array, or shape ``(1,)`` array.  Maps to absolute
            paddle position: -1 = left edge, 0 = centre, +1 = right
            edge of the client area.
        """
        if self.headless:
            self._apply_action_headless(action)
            return

        if self._capture is None or self._client_origin is None:
            return

        # Normalize: accept scalar, 0-d, or (1,) array
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 1:
            raise ValueError(f"Expected action of size 1, got size {action.size}")

        # Extract scalar from the action array
        value = float(np.clip(action[0], -1.0, 1.0))

        # Map [-1, 1] to normalised [0, 1] for screen mapping
        x_norm = (value + 1.0) / 2.0

        # Fixed paddle Y position (~90% of client height)
        y_norm = 0.90

        screen_x, screen_y = _norm_to_screen(
            x_norm,
            y_norm,
            self._capture.width,
            self._capture.height,
            self._client_origin,
        )

        try:
            import pydirectinput

            pydirectinput.moveTo(screen_x, screen_y)
        except ImportError:
            logger.debug("pydirectinput not available — action skipped")
        except Exception as exc:
            logger.debug("Action failed: %s", exc)

    def _apply_action_headless(self, action: "np.ndarray") -> None:
        """Send action via Selenium ActionChains (headless mode).

        Maps the continuous ``[-1, 1]`` action to a pixel offset from
        the canvas centre and moves the mouse there.  The Y position
        is fixed at 90% of canvas height (relative to centre).

        Parameters
        ----------
        action : np.ndarray
            Continuous action value in ``[-1, 1]``.
        """
        if self._driver is None or self._game_canvas is None:
            return
        if self._canvas_size is None:
            return

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 1:
            raise ValueError(f"Expected action of size 1, got size {action.size}")

        value = float(np.clip(action[0], -1.0, 1.0))
        canvas_w, canvas_h = self._canvas_size

        # x_offset from centre: value * half-width
        x_offset = int(value * (canvas_w / 2))

        # y_offset from centre: 90% of height → 0.4 * height from centre
        y_offset = int(canvas_h * 0.4)

        try:
            from selenium.webdriver.common.action_chains import ActionChains

            ActionChains(self._driver).move_to_element_with_offset(
                self._game_canvas, x_offset, y_offset
            ).perform()
        except Exception as exc:
            logger.debug("Headless action failed: %s", exc)

    def _handle_game_state(self) -> str:
        """Detect and handle game UI state (modals, game over, perks).

        Uses JavaScript execution via Selenium to query the game DOM
        and dismiss modals as needed.  This is the **only** use of
        Selenium at runtime — all observation and control is pixel-based.

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

    def _click_canvas(self) -> None:
        """Click the game canvas centre to start/unpause the game.

        In native mode, uses ``pydirectinput.click()`` at the centre
        of the client area.  In headless mode, uses Selenium
        ``ActionChains.click()``.
        """
        if self.headless:
            self._click_canvas_headless()
            return

        if self._capture is None or self._client_origin is None:
            return

        try:
            import pydirectinput

            screen_x, screen_y = _norm_to_screen(
                0.5,
                0.5,
                self._capture.width,
                self._capture.height,
                self._client_origin,
            )
            pydirectinput.click(screen_x, screen_y)
        except ImportError:
            logger.debug("pydirectinput not available — click skipped")
        except Exception as exc:
            logger.debug("Canvas click failed: %s", exc)

    def _click_canvas_headless(self) -> None:
        """Click the game canvas via Selenium ActionChains (headless mode).

        Moves to the canvas element centre and clicks.
        """
        if self._driver is None or self._game_canvas is None:
            return

        try:
            from selenium.webdriver.common.action_chains import ActionChains

            ActionChains(self._driver).move_to_element(
                self._game_canvas
            ).click().perform()
        except Exception as exc:
            logger.debug("Headless canvas click failed: %s", exc)

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
        list
            Aggregated findings from all oracles.
        """
        findings: list[Any] = []
        for oracle in self._oracles:
            oracle.on_step(obs, reward, terminated, truncated, info)
            findings.extend(oracle.get_findings())
        return findings

    def _build_info(self, detections: dict[str, Any]) -> dict[str, Any]:
        """Build the info dict returned by reset/step.

        Parameters
        ----------
        detections : dict[str, Any]
            Current YOLO detections.

        Returns
        -------
        dict[str, Any]
            Info dict with frame, detections, positions, counts.
        """
        paddle = detections.get("paddle")
        ball = detections.get("ball")

        return {
            "frame": self._last_frame,
            "detections": detections,
            "ball_pos": [ball[0], ball[1]] if ball is not None else None,
            "paddle_pos": [paddle[0], paddle[1]] if paddle is not None else None,
            "brick_count": len(detections.get("bricks", [])),
            "step": self._step_count,
            "score": 0.0,  # placeholder for v1
            "no_ball_count": self._no_ball_count,
        }

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
