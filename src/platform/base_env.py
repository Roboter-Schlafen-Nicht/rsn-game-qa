"""Base game environment ABC for RL-driven game QA.

Provides the generic lifecycle (capture → detect → observe → reward →
terminate → oracle) that all game environments share.  Game-specific
behaviour is implemented by subclasses via abstract methods.

The base class handles:

- Lazy initialisation of capture (``WinCamCapture`` / ``WindowCapture``
  / Selenium headless) and YOLO detector.
- Frame capture in native and headless modes.
- YOLO object detection.
- Oracle wiring (``on_step``, ``on_reset``, ``clear``).
- The ``reset()`` / ``step()`` Gymnasium lifecycle, including modal
  check throttling for mid-step game-over detection.
- Rendering and resource cleanup.

Subclasses must implement:

- ``game_classes()`` — YOLO class names for this game.
- ``observation_space`` / ``action_space`` — Gymnasium spaces.
- ``build_observation()`` — convert YOLO detections to obs vector.
- ``compute_reward()`` — reward signal from detections.
- ``check_termination()`` — terminated / level-cleared flags.
- ``apply_action()`` — translate action to game input.
- ``handle_modals()`` — detect and handle game UI state.
- ``start_game()`` — initial click / key to begin gameplay.
- ``canvas_selector()`` — CSS selector for the game canvas.
- ``build_info()`` — game-specific info dict entries.

Optional hooks:

- ``on_lazy_init()`` — called after base lazy init completes.
- ``on_reset_complete()`` — called after a successful reset.
"""

from __future__ import annotations

import abc
import logging
import time
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


class BaseGameEnv(gym.Env, abc.ABC):
    """Abstract base class for game QA environments.

    Subclasses must define ``observation_space`` and ``action_space``
    in their ``__init__`` (after calling ``super().__init__()``).

    Parameters
    ----------
    window_title : str
        Title of the browser window running the game.
    yolo_weights : str or Path
        Path to trained YOLO weights file.
    max_steps : int
        Maximum steps per episode before truncation.
    render_mode : str, optional
        Gymnasium render mode (``"human"`` or ``"rgb_array"``).
    oracles : list, optional
        List of Oracle instances to attach.
    driver : object, optional
        Selenium WebDriver for modal handling and input.
    device : str
        Device for YOLO inference (``"auto"``, ``"xpu"``, ``"cuda"``,
        ``"cpu"``).
    headless : bool
        If True, use Selenium screenshots instead of Win32 capture.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        window_title: str = "",
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

        # Internal state
        self._step_count: int = 0
        self._oracles: list[Any] = oracles or []
        self._last_frame: np.ndarray | None = None

        # Sub-components (initialised lazily)
        self._capture = None  # WinCamCapture or WindowCapture
        self._detector = None  # YoloDetector
        self._driver = driver  # Selenium WebDriver
        self._initialized: bool = False

        # Selenium canvas element for ActionChains input
        self._game_canvas: Any | None = None
        self._canvas_size: tuple[int, int] | None = None

    # ------------------------------------------------------------------
    # Abstract methods — subclasses MUST implement
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def game_classes(self) -> list[str]:
        """Return the YOLO class names for this game.

        Used by the platform to configure the YOLO detector when
        dynamically loading game plugins (``--game`` flag).

        Returns
        -------
        list[str]
            Ordered list of class names matching the YOLO model.
        """

    @abc.abstractmethod
    def build_observation(
        self, detections: dict[str, Any], *, reset: bool = False
    ) -> np.ndarray:
        """Convert YOLO detections into the observation vector.

        Parameters
        ----------
        detections : dict[str, Any]
            Detection results from ``_detect_objects``.
        reset : bool
            If True, this is the first observation of a new episode.

        Returns
        -------
        np.ndarray
            Observation matching ``self.observation_space``.
        """

    @abc.abstractmethod
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
            Current YOLO detections.
        terminated : bool
            Whether the episode ended this step.
        level_cleared : bool
            Whether a level/stage was cleared.

        Returns
        -------
        float
            Reward signal.
        """

    @abc.abstractmethod
    def check_termination(self, detections: dict[str, Any]) -> tuple[bool, bool]:
        """Check whether the episode should terminate.

        Called every step after detection.  The subclass is responsible
        for maintaining its own termination counters (e.g. consecutive
        frames without a ball).

        Parameters
        ----------
        detections : dict[str, Any]
            Current YOLO detections.

        Returns
        -------
        terminated : bool
            True if the episode should end (game over or level cleared).
        level_cleared : bool
            True if the level/stage was cleared (subset of terminated).
        """

    @abc.abstractmethod
    def apply_action(self, action: np.ndarray) -> None:
        """Send the chosen action to the game.

        Parameters
        ----------
        action : np.ndarray
            Action from the policy, matching ``self.action_space``.
        """

    @abc.abstractmethod
    def handle_modals(self, *, dismiss_game_over: bool = True) -> str:
        """Detect and handle game UI modals (game over, menus, etc.).

        Parameters
        ----------
        dismiss_game_over : bool
            If True, dismiss game-over modals.  If False, only detect.

        Returns
        -------
        str
            Detected state: ``"gameplay"``, ``"game_over"``,
            ``"perk_picker"``, ``"menu"``, or ``"unknown"``.
        """

    @abc.abstractmethod
    def start_game(self) -> None:
        """Perform the initial action to start/unpause the game.

        Called during ``reset()`` after modal dismissal.
        """

    @abc.abstractmethod
    def canvas_selector(self) -> str:
        """Return the CSS ID of the game canvas element.

        Used by ``_init_canvas()`` to locate the element for
        Selenium ``ActionChains`` coordinate mapping.

        Returns
        -------
        str
            CSS element ID (without ``#`` prefix), e.g. ``"game"``.
        """

    @abc.abstractmethod
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

    @abc.abstractmethod
    def terminal_reward(self) -> float:
        """Return the fixed reward for game-over detected via modal.

        This is used when game-over is detected mid-step via modal
        check, avoiding computing reward from modal-occluded detections.

        Returns
        -------
        float
            Terminal penalty (typically negative).
        """

    @abc.abstractmethod
    def on_reset_detections(self, detections: dict[str, Any]) -> bool:
        """Check whether reset detections are valid to start an episode.

        Parameters
        ----------
        detections : dict[str, Any]
            Detections from a reset attempt frame.

        Returns
        -------
        bool
            True if the detections are sufficient to start playing
            (e.g. ball detected for Breakout).
        """

    @abc.abstractmethod
    def reset_termination_state(self) -> None:
        """Reset game-specific termination counters for a new episode.

        Called at the start of ``reset()`` after a valid detection frame.
        """

    # ------------------------------------------------------------------
    # Optional hooks — subclasses MAY override
    # ------------------------------------------------------------------

    def on_lazy_init(self) -> None:
        """Hook called after base lazy initialisation completes.

        Override to perform game-specific setup (e.g. query game zone
        dimensions from the DOM).
        """

    def on_reset_complete(self, obs: np.ndarray, info: dict[str, Any]) -> None:
        """Hook called after a successful reset.

        Parameters
        ----------
        obs : np.ndarray
            Initial observation.
        info : dict[str, Any]
            Info dict from reset.
        """

    # ------------------------------------------------------------------
    # Gymnasium lifecycle — generic implementation
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment for a new episode.

        Handles modals, starts the game, and waits for valid detections.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict, optional
            Additional reset options.

        Returns
        -------
        obs : np.ndarray
            Initial observation.
        info : dict[str, Any]
            Auxiliary information.
        """
        super().reset(seed=seed)

        if not self._initialized:
            self._lazy_init()

        # Dismiss any modals and start the game
        detections: dict[str, Any] = {}
        for attempt in range(5):
            logger.info("reset() attempt %d/5", attempt + 1)
            self.handle_modals()
            self.start_game()
            time.sleep(0.5)

            frame = self._capture_frame()
            detections = self._detect_objects(frame)

            if self.on_reset_detections(detections):
                logger.info("reset() attempt %d: valid detections", attempt + 1)
                break

            logger.info("reset() attempt %d: invalid detections", attempt + 1)

        if not self.on_reset_detections(detections):
            raise RuntimeError(
                f"{type(self).__name__}.reset() failed to get valid "
                f"detections after 5 attempts; the game may not have "
                f"initialized correctly."
            )

        # Build observation with reset semantics
        obs = self.build_observation(detections, reset=True)

        # Reset counters
        self._step_count = 0
        self.reset_termination_state()

        # Build info dict
        info = self._make_info(detections)

        # Clear and notify oracles
        for oracle in self._oracles:
            oracle.clear()
            oracle.on_reset(obs, info)

        self.on_reset_complete(obs, info)

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one action and return the resulting transition.

        Parameters
        ----------
        action : np.ndarray
            Action matching ``self.action_space``.

        Returns
        -------
        obs : np.ndarray
            Observation after the action.
        reward : float
            Reward signal.
        terminated : bool
            True if the game is over.
        truncated : bool
            True if max_steps reached.
        info : dict[str, Any]
            Auxiliary information.
        """
        self.apply_action(action)

        # Modal check throttling: only check when subclass signals
        # a missing key object (e.g. ball not detected for N frames).
        mid_state = "gameplay"
        if self._should_check_modals():
            mid_state = self.handle_modals(dismiss_game_over=False)

        if mid_state == "game_over":
            frame = self._capture_frame()
            detections = self._detect_objects(frame)
            obs = self.build_observation(detections)
            self._step_count += 1
            truncated = self._step_count >= self.max_steps
            reward = self.terminal_reward()
            info = self._make_info(detections)
            findings = self._run_oracles(obs, reward, True, truncated, info)
            info["oracle_findings"] = findings
            return obs, reward, True, truncated, info

        # Handle non-terminal modals (perk picker, menu)
        if mid_state in ("perk_picker", "menu"):
            self.start_game()
            time.sleep(0.3)

        # Capture and detect
        frame = self._capture_frame()
        detections = self._detect_objects(frame)

        # Build observation
        obs = self.build_observation(detections)

        # Let subclass check for late game-over (e.g. ball just
        # disappeared → immediate modal check)
        late_game_over = self._check_late_game_over(detections)
        if late_game_over:
            self._step_count += 1
            truncated = self._step_count >= self.max_steps
            reward = self.terminal_reward()
            info = self._make_info(detections)
            findings = self._run_oracles(obs, reward, True, truncated, info)
            info["oracle_findings"] = findings
            return obs, reward, True, truncated, info

        # Increment step counter
        self._step_count += 1

        # Determine termination (game-specific)
        terminated, level_cleared = self.check_termination(detections)
        truncated = self._step_count >= self.max_steps

        # Compute reward (game-specific)
        reward = self.compute_reward(detections, terminated, level_cleared)

        # Build info dict
        info = self._make_info(detections)

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
        """Return the current step count (read-only)."""
        return self._step_count

    def close(self) -> None:
        """Release capture resources.

        Does **not** close the Selenium driver — that is owned by the
        caller.
        """
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._detector = None
        self._game_canvas = None
        self._canvas_size = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Modal throttling hooks — subclasses override for game-specific
    # ------------------------------------------------------------------

    def _should_check_modals(self) -> bool:
        """Return True if a modal check is warranted this step.

        The default returns True (always check — safe for new games).
        Subclasses should override to skip expensive Selenium HTTP
        round-trips when game state indicates normal gameplay
        (e.g., only check when ball is missing).

        Returns
        -------
        bool
            True to check for modals this step, False to skip.
        """
        return True

    def _check_late_game_over(self, detections: dict[str, Any]) -> bool:
        """Check for game-over on the 0→1 transition of a missing object.

        Called after ``build_observation`` but before ``check_termination``.
        Return True to terminate the episode immediately with
        ``terminal_reward()``.

        Parameters
        ----------
        detections : dict[str, Any]
            Current YOLO detections.

        Returns
        -------
        bool
            True if a late game-over modal was detected.
        """
        return False

    # ------------------------------------------------------------------
    # Private helpers — shared infrastructure
    # ------------------------------------------------------------------

    def _lazy_init(self) -> None:
        """Lazily initialise capture, detector, and canvas.

        Called on the first ``reset()``.  All imports are deferred to
        avoid breaking CI in Docker where pywin32/wincam are unavailable.
        """
        if self._initialized:
            return

        from src.perception.yolo_detector import YoloDetector

        if self.headless:
            if self._driver is None:
                raise RuntimeError(
                    "Headless mode requires a Selenium WebDriver "
                    f"(pass driver= to {type(self).__name__})"
                )
        else:
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

        self._init_canvas()

        self._detector = YoloDetector(
            weights_path=self.yolo_weights,
            device=self.device,
            classes=self.game_classes(),
        )
        self._detector.load()

        self._initialized = True
        self.on_lazy_init()

    def _init_canvas(self) -> None:
        """Find the game canvas element for ActionChains input."""
        if self._driver is None:
            return

        selector = self.canvas_selector()
        try:
            from selenium.webdriver.common.by import By

            self._game_canvas = self._driver.find_element(By.ID, selector)
        except Exception:
            logger.warning("Canvas #%s not found, falling back to <body>", selector)
            try:
                from selenium.webdriver.common.by import By

                self._game_canvas = self._driver.find_element(By.TAG_NAME, "body")
            except Exception:
                logger.warning("Could not find <body> either")
                return

        size = self._game_canvas.size
        self._canvas_size = (size["width"], size["height"])
        logger.info(
            "Canvas: %dx%d",
            self._canvas_size[0],
            self._canvas_size[1],
        )

    def _capture_frame(self) -> np.ndarray:
        """Capture a frame from the game window.

        Returns
        -------
        np.ndarray
            BGR image of the game window.
        """
        if self.headless:
            return self._capture_frame_headless()
        frame = self._capture.capture_frame()
        self._last_frame = frame
        return frame

    def _capture_frame_headless(self) -> np.ndarray:
        """Capture a frame via canvas ``toDataURL`` or Selenium screenshot.

        Prefers ``canvas.toDataURL('image/jpeg')`` (~11 ms) over
        ``get_screenshot_as_png()`` (~85 ms) when a canvas selector is
        available.  Falls back to full-page screenshot if the canvas
        capture fails.

        Returns
        -------
        np.ndarray
            BGR image decoded from the screenshot.

        Raises
        ------
        RuntimeError
            If the driver is unavailable or decoding fails.
        """
        if self._driver is None:
            raise RuntimeError("Cannot capture headless frame: driver is None")

        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "cv2 (opencv-python) is required for headless capture"
            ) from exc

        frame = None

        # -- Dismiss any unexpected browser alerts -------------------------
        # The game can occasionally spawn JS alert() dialogs (e.g. "Two
        # alerts where opened at once") that block all Selenium commands
        # including screenshots.  Dismiss them preemptively.
        try:
            alert = self._driver.switch_to.alert
            logger.warning("Dismissing unexpected browser alert: %s", alert.text)
            alert.dismiss()
        except Exception:  # noqa: BLE001
            pass  # No alert present — normal case

        # Fast path: canvas toDataURL (JPEG, ~11ms vs ~85ms for PNG)
        try:
            selector = self.canvas_selector()
        except NotImplementedError:
            selector = None

        if selector is not None:
            try:
                import base64

                data_url = self._driver.execute_script(
                    "var c = document.getElementById(arguments[0]);"
                    "if (!c) return null;"
                    "return c.toDataURL('image/jpeg', 0.8);",
                    selector,
                )
                if data_url is not None:
                    b64_data = data_url.split(",", 1)[1]
                    img_bytes = base64.b64decode(b64_data)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception:
                logger.debug("Canvas toDataURL failed, falling back to screenshot")

        # Fallback: full-page screenshot
        if frame is None:
            try:
                png_bytes = self._driver.get_screenshot_as_png()
                nparr = np.frombuffer(png_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception:  # noqa: BLE001
                # Alert may have appeared between toDataURL and screenshot;
                # dismiss and retry once.
                try:
                    alert = self._driver.switch_to.alert
                    logger.warning(
                        "Dismissing alert before screenshot retry: %s",
                        alert.text,
                    )
                    alert.dismiss()
                    png_bytes = self._driver.get_screenshot_as_png()
                    nparr = np.frombuffer(png_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                except Exception as inner_exc:  # noqa: BLE001
                    logger.error(
                        "Screenshot failed even after alert dismissal: %s",
                        inner_exc,
                    )
        if frame is None:
            raise RuntimeError("Failed to decode screenshot PNG to BGR frame")
        self._last_frame = frame
        return frame

    def _detect_objects(self, frame: np.ndarray) -> dict[str, Any]:
        """Run YOLO inference on a frame.

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

    def _make_info(self, detections: dict[str, Any]) -> dict[str, Any]:
        """Build the complete info dict (base + game-specific).

        Parameters
        ----------
        detections : dict[str, Any]
            Current YOLO detections.

        Returns
        -------
        dict[str, Any]
            Combined info dict.
        """
        info = self.build_info(detections)
        info["frame"] = self._last_frame
        info["step"] = self._step_count
        return info

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
