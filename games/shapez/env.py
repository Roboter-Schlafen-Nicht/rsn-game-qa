"""shapez.io Gymnasium environment for RL-driven game QA.

Game-specific subclass of :class:`BaseGameEnv` implementing the shapez.io
factory-builder browser game.  shapez.io is an open-source factory game
where the player builds conveyor networks to produce and deliver shapes
to a central hub.

Key differences from Breakout 71 and Hextris:

- **MultiDiscrete action space**: ``MultiDiscrete([7, 10, 16, 16, 4])``
  — action_type (7), building_id (10), grid_x (16), grid_y (16),
  rotation (4).  Far more complex than Breakout (continuous Box) or
  Hextris (Discrete(3)).
- **No YOLO**: CNN-only pixel observations.
- **No natural game-over**: Factory builders have no death state.
  Session ends via ``max_steps`` truncation or idle detection.
- **Level transitions**: 26 levels + freeplay.  Unlock notifications
  must be dismissed to continue.
- **Dev mode access**: ``window.shapez.GLOBAL_APP`` provides app-level
  access (settings, state manager, savegame manager) from boot time.
  ``window.globalRoot`` provides in-game access (level, entity count,
  shape delivery progress, upgrade levels) but is only available during
  active gameplay (``InGameState``).
- **Mouse + keyboard input**: Building placement via click, selection
  via digit keys, rotation via R, camera pan via WASD.
- **Tutorial**: Disabled via ``offerHints = false`` in
  ``SETUP_TRAINING_JS``.

The action space is ``MultiDiscrete([7, 10, 16, 16, 4])``:

- Dim 0 — action type:
    0 = noop
    1 = select building (use dim 1 for building ID)
    2 = place / click at grid position (use dims 2, 3)
    3 = delete at grid position (right-click, use dims 2, 3)
    4 = rotate selected building
    5 = pan camera (use dim 4 for direction)
    6 = center camera on hub
- Dim 1 — building ID (0-9, mapped to digit keys 1-9, 0)
- Dim 2 — grid X position (0-15, mapped to canvas pixel coords)
- Dim 3 — grid Y position (0-15, mapped to canvas pixel coords)
- Dim 4 — pan direction (0=up, 1=down, 2=left, 3=right)

Observation is an 8-element MLP vector (matching platform convention)::

    [level_norm, goal_progress, entity_norm, running, 0, 0, 0, 0]

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

from games.shapez.modal_handler import (
    CENTER_HUB_JS,
    CLICK_AT_JS,
    CLOSE_SETTINGS_JS,
    DELETE_AT_JS,
    DETECT_STATE_JS,
    DISMISS_MODAL_JS,
    DISMISS_UNLOCK_JS,
    NOOP_JS,
    PAN_CAMERA_JS,
    READ_GAME_STATE_JS,
    ROTATE_BUILDING_JS,
    SELECT_BUILDING_JS,
    SETUP_TRAINING_JS,
    START_NEW_GAME_JS,
)
from src.platform.base_env import BaseGameEnv

logger = logging.getLogger(__name__)

# Number of grid cells along each axis for the discretised placement grid.
_GRID_SIZE = 16

# Pan direction mapping for action dim 4.
_PAN_DIRECTIONS = ("up", "down", "left", "right")

# Steps with no progress (shapes delivered or entities placed) before
# the idle detector fires.  With ~15 FPS headless this is ~200 seconds.
_IDLE_THRESHOLD = 3000


class ShapezEnv(BaseGameEnv):
    """Gymnasium environment wrapping the shapez.io browser game.

    This environment captures frames from the game window, uses CNN
    pixel observations (no YOLO), and injects actions via JavaScript
    calls to the shapez.io dev mode API.

    Parameters
    ----------
    window_title : str
        Title of the browser window running shapez.io.
        Default is ``"shapez.io"``.
    yolo_weights : str or Path
        Path to YOLO weights file.  Not used by shapez.io (CNN-only)
        but required by the platform interface.
    max_steps : int
        Maximum steps per episode before truncation.  Default is 10000.
    render_mode : str, optional
        Gymnasium render mode (``"human"`` or ``"rgb_array"``).
    oracles : list, optional
        List of ``Oracle`` instances to attach.
    driver : object, optional
        Selenium WebDriver instance for DOM interaction and input.
    device : str
        Device for YOLO inference.  Irrelevant for shapez.io (CNN-only).
    headless : bool
        If ``True``, use Selenium-based capture.
    reward_mode : str
        Reward strategy.  ``"survival"`` recommended for shapez.io.
    game_over_detector : object, optional
        Pixel-based game-over detector.
    survival_bonus : float
        Per-step survival reward.  Default ``0.01``.
    browser_instance : object, optional
        Reference to ``BrowserInstance`` for crash recovery.
    idle_threshold : int
        Steps without progress before idle termination.
        Default ``3000``.
    """

    def __init__(
        self,
        window_title: str = "shapez.io",
        yolo_weights: str | Path = "weights/best.pt",
        max_steps: int = 10_000,
        render_mode: str | None = None,
        oracles: list[Any] | None = None,
        driver: Any | None = None,
        device: str = "auto",
        headless: bool = False,
        reward_mode: str = "survival",
        game_over_detector: Any | None = None,
        survival_bonus: float = 0.01,
        browser_instance: Any | None = None,
        idle_threshold: int = _IDLE_THRESHOLD,
        score_region: tuple[int, int, int, int] | None = None,
        score_ocr_interval: int = 1,
        score_reward_coeff: float = 0.01,
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
            score_region=score_region,
            score_ocr_interval=score_ocr_interval,
            score_reward_coeff=score_reward_coeff,
        )

        # Observation: 8-element vector (minimal MLP placeholder)
        self.observation_space = spaces.Box(
            low=np.zeros(8, dtype=np.float32),
            high=np.ones(8, dtype=np.float32),
            dtype=np.float32,
        )

        # Actions: MultiDiscrete([action_type, building_id, grid_x, grid_y, pan_dir])
        self.action_space = spaces.MultiDiscrete([7, 10, _GRID_SIZE, _GRID_SIZE, 4])

        # Game-specific state
        self._prev_level: int = 0
        self._prev_entity_count: int = 0
        self._prev_shapes_delivered: int = 0
        self._idle_count: int = 0
        self._idle_threshold: int = idle_threshold
        self._training_configured: bool = False
        self._menu_start_requested: bool = False
        self._canvas_ready: bool = False

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def game_classes(self) -> list[str]:
        """Return empty class list (shapez.io uses CNN-only observations).

        Returns
        -------
        list[str]
            Empty list.
        """
        return []

    def canvas_selector(self) -> str:
        """Return the CSS ID of the shapez.io game canvas.

        Returns
        -------
        str
            ``"ingame_Canvas"``
        """
        return "ingame_Canvas"

    def build_observation(self, detections: dict[str, Any], *, reset: bool = False) -> np.ndarray:
        """Convert game state into the flat observation vector.

        For shapez.io, YOLO detections are empty (no model).  The MLP
        observation is a minimal placeholder reading JS state.  The
        primary observation mode is CNN (84x84 grayscale pixels via
        ``CnnObservationWrapper``).

        Parameters
        ----------
        detections : dict[str, Any]
            Detection results (empty for shapez.io).
        reset : bool
            If True, this is the first observation of a new episode.

        Returns
        -------
        np.ndarray
            Float32 observation vector (8 elements).
        """
        game_state = self._read_game_state()
        level = game_state.get("level", 0)
        goal_progress = game_state.get("goalProgress", 0.0)
        entity_count = game_state.get("entityCount", 0)
        running = game_state.get("running", False)

        # Normalise level to [0, 1] with soft cap at 26 (max story levels)
        level_norm = min(level / 26.0, 1.0) if level > 0 else 0.0
        # Goal progress is already [0, 1]
        goal_norm = min(max(goal_progress, 0.0), 1.0)
        # Normalise entity count (typical large factory ~500-1000)
        entity_norm = min(entity_count / 1000.0, 1.0)
        running_flag = 1.0 if running else 0.0

        obs = np.array(
            [level_norm, goal_norm, entity_norm, running_flag, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )

        if reset:
            self._prev_level = level
            self._prev_entity_count = entity_count
            self._prev_shapes_delivered = game_state.get("shapesDelivered", 0)

        return obs

    def compute_reward(
        self,
        detections: dict[str, Any],
        terminated: bool,
        level_cleared: bool,
    ) -> float:
        """Compute the reward for the current step.

        In ``yolo`` mode (the game-specific reward), uses shape delivery
        progress and entity placement as reward signals.

        Parameters
        ----------
        detections : dict[str, Any]
            Current detections (empty for shapez.io).
        terminated : bool
            Whether the episode ended this step.
        level_cleared : bool
            Whether a level was completed this step.

        Returns
        -------
        float
            Reward signal.
        """
        game_state = self._read_game_state()
        shapes_delivered = game_state.get("shapesDelivered", 0)
        entity_count = game_state.get("entityCount", 0)
        level = game_state.get("level", 0)

        reward = 0.0

        # Shape delivery reward
        shapes_delta = shapes_delivered - self._prev_shapes_delivered
        if shapes_delta > 0:
            reward += shapes_delta * 0.01

        # Entity placement reward (small bonus for building)
        entity_delta = entity_count - self._prev_entity_count
        if entity_delta > 0:
            reward += entity_delta * 0.005

        # Level completion bonus
        level_delta = level - self._prev_level
        if level_delta > 0:
            reward += level_delta * 1.0

        # Small time penalty to encourage action
        reward -= 0.001

        # Update tracking state
        self._prev_shapes_delivered = shapes_delivered
        self._prev_entity_count = entity_count
        self._prev_level = level

        return reward

    def check_termination(self, detections: dict[str, Any]) -> tuple[bool, bool]:
        """Check whether the episode should terminate.

        shapez.io has no natural game-over.  Termination occurs via:

        - ``max_steps`` truncation (handled by base class).
        - Idle detection: no shapes delivered and no entities placed
          for ``idle_threshold`` consecutive steps.

        Level completion is detected via the JS bridge (level change).

        Parameters
        ----------
        detections : dict[str, Any]
            Current detections (empty for shapez.io).

        Returns
        -------
        terminated : bool
            True if idle threshold exceeded.
        level_cleared : bool
            True if the level was completed since last check.
        """
        game_state = self._read_game_state()
        shapes_delivered = game_state.get("shapesDelivered", 0)
        entity_count = game_state.get("entityCount", 0)
        level = game_state.get("level", 0)

        # Level completion detection
        level_cleared = level > self._prev_level

        # Idle detection: no progress for too many steps
        if (
            shapes_delivered == self._prev_shapes_delivered
            and entity_count == self._prev_entity_count
        ):
            self._idle_count += 1
        else:
            self._idle_count = 0

        terminated = self._idle_count >= self._idle_threshold

        return terminated, level_cleared

    def _ensure_canvas_ready(self) -> None:
        """Re-initialise the canvas reference if not yet valid.

        Called before coordinate-dependent actions (place/delete) so that
        the very first action of an episode uses correct canvas
        dimensions, even if ``handle_modals()`` has not run yet.
        """
        if self._canvas_ready:
            return
        self._init_canvas()
        if self._canvas_size and self._canvas_size[1] > 100:
            self._canvas_ready = True
            logger.info(
                "Canvas re-initialised (from apply_action): %dx%d",
                self._canvas_size[0],
                self._canvas_size[1],
            )

    def apply_action(self, action: np.ndarray | int) -> None:
        """Send the chosen action to the game via JavaScript.

        Parameters
        ----------
        action : np.ndarray
            MultiDiscrete action: [action_type, building_id, grid_x,
            grid_y, pan_direction].
        """
        if self._driver is None:
            return

        act = np.asarray(action).flatten()
        if len(act) < 5:
            return  # Invalid action shape

        action_type = int(act[0])
        building_id = int(act[1])
        grid_x = int(act[2])
        grid_y = int(act[3])
        pan_dir = int(act[4])

        try:
            if action_type == 0:
                # Noop
                self._driver.execute_script(NOOP_JS)
            elif action_type == 1:
                # Select building by digit key
                key = (building_id + 1) % 10  # 0->1, 1->2, ..., 9->0
                self._driver.execute_script(SELECT_BUILDING_JS, key)
            elif action_type == 2:
                # Place at grid position
                self._ensure_canvas_ready()
                px, py = self._grid_to_canvas(grid_x, grid_y)
                self._driver.execute_script(CLICK_AT_JS, px, py)
            elif action_type == 3:
                # Delete at grid position
                self._ensure_canvas_ready()
                px, py = self._grid_to_canvas(grid_x, grid_y)
                self._driver.execute_script(DELETE_AT_JS, px, py)
            elif action_type == 4:
                # Rotate building
                self._driver.execute_script(ROTATE_BUILDING_JS)
            elif action_type == 5:
                # Pan camera
                direction = _PAN_DIRECTIONS[min(pan_dir, 3)]
                self._driver.execute_script(PAN_CAMERA_JS, direction)
            elif action_type == 6:
                # Center hub
                self._driver.execute_script(CENTER_HUB_JS)
        except Exception as exc:
            logger.debug("Action failed (type=%d): %s", action_type, exc)

    def handle_modals(self, *, dismiss_game_over: bool = True) -> str:
        """Detect and handle game UI state.

        shapez.io modals include level-complete unlock notifications,
        settings menus, and generic modal dialogs.  There is no
        traditional game-over state.

        Parameters
        ----------
        dismiss_game_over : bool
            If True, dismiss modals.  For shapez.io this controls
            whether unlock notifications are auto-dismissed.

        Returns
        -------
        str
            Detected state: ``"gameplay"``, ``"level_complete"``,
            ``"settings"``, ``"modal"``, ``"main_menu"``,
            ``"loading"``, ``"menu"``, or ``"unknown"``.
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

        # Clear menu-start guard once we've transitioned to gameplay
        if state not in ("main_menu", "loading") and self._menu_start_requested:
            self._menu_start_requested = False

        # Re-initialise the canvas reference once the in-game canvas
        # exists.  During ``_lazy_init()`` the game is still on the main
        # menu so ``#ingame_Canvas`` is absent and the env falls back to
        # ``<body>`` with incorrect dimensions.
        if state == "gameplay" and not self._canvas_ready:
            self._ensure_canvas_ready()

        if state == "level_complete" and dismiss_game_over:
            try:
                self._driver.execute_script(DISMISS_UNLOCK_JS)
                logger.info("Level-complete notification dismissed")
            except Exception as exc:
                logger.warning("Unlock dismiss failed: %s", exc)
            time.sleep(0.5)

        elif state == "settings":
            try:
                self._driver.execute_script(CLOSE_SETTINGS_JS)
                logger.info("Settings menu closed")
            except Exception as exc:
                logger.warning("Settings close failed: %s", exc)
            time.sleep(0.3)

        elif state == "modal":
            try:
                self._driver.execute_script(DISMISS_MODAL_JS)
                logger.info("Modal dialog dismissed")
            except Exception as exc:
                logger.warning("Modal dismiss failed: %s", exc)
            time.sleep(0.3)

        elif state == "main_menu":
            if not self._menu_start_requested:
                try:
                    result = self._driver.execute_script(START_NEW_GAME_JS)
                    action = result.get("action", "none") if result else "none"
                    if action == "play_invoked":
                        self._menu_start_requested = True
                        logger.info("New game requested from main menu")
                    else:
                        error = result.get("error", "") if result else ""
                        logger.warning("Start new game returned: %s (%s)", action, error)
                except Exception as exc:
                    logger.warning("Start new game failed: %s", exc)
                time.sleep(2.0)
            else:
                # Already requested, wait for state transition
                logger.debug("Waiting for main menu -> InGameState transition")
                time.sleep(1.0)

        return state

    def start_game(self) -> None:
        """Start a new game and wait for the InGameState transition.

        Navigates from main menu to in-game state by calling
        ``onPlayButtonClicked()`` on the main menu state via JS, then
        polls until the game enters ``InGameState`` (up to 15 s).
        Once gameplay is active, re-initialises the canvas reference
        so that ``#ingame_Canvas`` is used instead of the ``<body>``
        fallback from ``_lazy_init()``.

        The polling wait is essential because shapez.io takes 3-5 s to
        transition from ``MainMenuState`` through the save picker dialog
        to ``InGameState``.  Without it, the first several ``step()``
        calls run against an uninitialised canvas with wrong dimensions.

        If the initial ``START_NEW_GAME_JS`` call fails (e.g. because
        ``GLOBAL_APP`` is not yet available after a page refresh), the
        polling loop retries the JS call once the detected state
        transitions to ``main_menu``.
        """
        if self._driver is None:
            return

        # Check if already in gameplay before issuing a new game request.
        try:
            pre_state = self._detect_ui_state().get("state", "unknown")
        except Exception:
            pre_state = "unknown"

        if pre_state == "gameplay":
            logger.info("start_game: already in gameplay")
            self._menu_start_requested = False
            self._canvas_ready = False
            self._ensure_canvas_ready()
            return

        # Only request a new game if we haven't already (e.g. from
        # ``handle_modals()`` seeing the main menu during a prior step).
        if not self._menu_start_requested:
            try:
                result = self._driver.execute_script(START_NEW_GAME_JS)
                action = result.get("action", "none") if result else "none"
                if action == "play_invoked":
                    self._menu_start_requested = True
                    logger.info("start_game: new game requested")
                else:
                    logger.debug("start_game: %s", result)
            except Exception as exc:
                logger.debug("Start game failed: %s", exc)
                return
        else:
            logger.debug("start_game: transition already requested, polling")

        # Poll until the game enters InGameState (gameplay).  shapez.io
        # needs 3-5 s for the full MainMenu → SavePicker → InGame flow.
        #
        # After a ``driver.refresh()`` the page reloads through
        # ``PreloadState`` → ``MainMenuState``.  The initial
        # ``START_NEW_GAME_JS`` call above may fail because
        # ``GLOBAL_APP`` is not yet available during ``PreloadState``.
        # We must retry the JS call once the state transitions to
        # ``main_menu`` (meaning ``GLOBAL_APP`` is now available).
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            state_info = self._detect_ui_state()
            state = state_info.get("state", "unknown")
            if state == "gameplay":
                logger.info("start_game: InGameState reached")
                self._menu_start_requested = False
                # Re-init canvas now that #ingame_Canvas exists
                self._canvas_ready = False
                self._ensure_canvas_ready()
                return
            # Retry START_NEW_GAME_JS if we're on the main menu but
            # haven't successfully requested a game start yet.
            if state == "main_menu" and not self._menu_start_requested:
                try:
                    result = self._driver.execute_script(START_NEW_GAME_JS)
                    action = result.get("action", "none") if result else "none"
                    if action == "play_invoked":
                        self._menu_start_requested = True
                        logger.info("start_game: retried START_NEW_GAME_JS — success")
                    else:
                        logger.debug("start_game: retry returned: %s", result)
                except Exception as exc:
                    logger.debug("start_game: retry failed: %s", exc)
            logger.debug("start_game: waiting for InGameState (current: %s)", state)
            time.sleep(0.5)

        logger.warning("start_game: timed out waiting for InGameState")

    def build_info(self, detections: dict[str, Any]) -> dict[str, Any]:
        """Build the game-specific portion of the info dict.

        Parameters
        ----------
        detections : dict[str, Any]
            Current detections (empty for shapez.io).

        Returns
        -------
        dict[str, Any]
            Game-specific info entries.
        """
        game_state = self._read_game_state()
        return {
            "detections": detections,
            "level": game_state.get("level", 0),
            "shapes_delivered": game_state.get("shapesDelivered", 0),
            "goal_progress": game_state.get("goalProgress", 0.0),
            "entity_count": game_state.get("entityCount", 0),
            "idle_count": self._idle_count,
            "running": game_state.get("running", False),
        }

    def terminal_reward(self) -> float:
        """Return the fixed terminal penalty for idle termination.

        Returns
        -------
        float
            ``-5.01`` (terminal penalty + time penalty).
        """
        return -5.0 - 0.001

    def on_reset_detections(self, detections: dict[str, Any]) -> bool:
        """Check whether reset detections are valid to start an episode.

        For shapez.io (CNN-only, no YOLO), accept any frame where the
        game is in a playable or menu state.

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

        state_info = self._detect_ui_state()
        state = state_info.get("state", "unknown")
        # Accept gameplay, main_menu (will auto-start), or loading states
        return state in ("gameplay", "main_menu", "loading")

    def reset_termination_state(self) -> None:
        """Reset game-specific termination counters for a new episode."""
        game_state = self._read_game_state()
        self._prev_level = game_state.get("level", 0)
        self._prev_entity_count = game_state.get("entityCount", 0)
        self._prev_shapes_delivered = game_state.get("shapesDelivered", 0)
        self._idle_count = 0
        self._menu_start_requested = False
        self._canvas_ready = False

    # ------------------------------------------------------------------
    # Platform overrides (skip YOLO for CNN-only game)
    # ------------------------------------------------------------------

    def _lazy_init(self) -> None:
        """Initialise capture and canvas, skipping YOLO detector.

        shapez.io is a CNN-only game with no YOLO model.  This override
        performs the same setup as ``BaseGameEnv._lazy_init()`` but
        skips ``YoloDetector`` construction and loading, which would
        fail because there are no weights to load.
        """
        if self._initialized:
            return

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
        # Skip YoloDetector — shapez.io is CNN-only (no weights)
        self._initialized = True
        self.on_lazy_init()

    def _detect_objects(self, frame: np.ndarray) -> dict[str, Any]:
        """Return empty detections (shapez.io has no YOLO model).

        Parameters
        ----------
        frame : np.ndarray
            Captured game frame (unused).

        Returns
        -------
        dict[str, Any]
            Empty detection results.
        """
        return {}

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def on_lazy_init(self) -> None:
        """Configure training settings after lazy initialisation.

        Disables tutorials and hints via the ``SETUP_TRAINING_JS``
        snippet.
        """
        if self._driver is not None and not self._training_configured:
            try:
                self._driver.execute_script(SETUP_TRAINING_JS)
                self._training_configured = True
                logger.info("Training settings configured (tutorials disabled)")
            except Exception as exc:
                logger.debug("Failed to configure training settings: %s", exc)

    def _should_check_modals(self) -> bool:
        """Always check modals for shapez.io.

        shapez.io can show unlock notifications, settings, and modal
        dialogs at any time during gameplay.

        Returns
        -------
        bool
            Always True.
        """
        return True

    def _handle_level_transition(self) -> bool:
        """Handle a level-complete transition.

        Dismisses the unlock notification and waits for the next
        level to load.

        Returns
        -------
        bool
            True if the transition was handled successfully.
        """
        if self._driver is None:
            return False

        try:
            self._driver.execute_script(DISMISS_UNLOCK_JS)
            logger.info("Level transition: unlock notification dismissed")
            time.sleep(0.5)
            return True
        except Exception as exc:
            logger.warning("Level transition failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_game_state(self) -> dict[str, Any]:
        """Read game state via the JS bridge.

        Returns
        -------
        dict[str, Any]
            Game state dictionary with keys: ``level``,
            ``shapesDelivered``, ``goalRequired``, ``goalProgress``,
            ``entityCount``, ``upgradeLevels``, ``running``,
            ``inGame``.
        """
        defaults: dict[str, Any] = {
            "level": 0,
            "shapesDelivered": 0,
            "goalRequired": 0,
            "goalProgress": 0.0,
            "entityCount": 0,
            "upgradeLevels": {},
            "running": False,
            "inGame": False,
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

    def _detect_ui_state(self) -> dict[str, Any]:
        """Detect the current UI state via the JS bridge.

        Returns
        -------
        dict[str, Any]
            State info dict with ``state`` key.
        """
        defaults: dict[str, Any] = {"state": "unknown", "details": {}}
        if self._driver is None:
            return defaults
        try:
            result = self._driver.execute_script(DETECT_STATE_JS)
            if result is not None:
                return result
        except Exception as exc:
            logger.debug("Failed to detect UI state: %s", exc)
        return defaults

    def _grid_to_canvas(self, grid_x: int, grid_y: int) -> tuple[int, int]:
        """Convert a discrete grid position to canvas pixel coordinates.

        Maps a ``_GRID_SIZE x _GRID_SIZE`` grid onto the canvas
        dimensions.  If canvas size is unknown, falls back to the
        configured window dimensions.

        Parameters
        ----------
        grid_x : int
            Grid X position (0 to ``_GRID_SIZE - 1``).
        grid_y : int
            Grid Y position (0 to ``_GRID_SIZE - 1``).

        Returns
        -------
        tuple[int, int]
            (pixel_x, pixel_y) canvas coordinates.
        """
        if self._canvas_size is not None:
            w, h = self._canvas_size
        else:
            # Fallback to default window size
            w, h = 1280, 1024

        # Map grid to canvas, centered in each grid cell
        cell_w = w / _GRID_SIZE
        cell_h = h / _GRID_SIZE
        px = int((grid_x + 0.5) * cell_w)
        py = int((grid_y + 0.5) * cell_h)

        return px, py
