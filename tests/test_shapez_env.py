"""Tests for the ShapezEnv Gymnasium environment.

Covers:
- Construction and space definitions (observation, action)
- Game class list (CNN-only, no YOLO)
- Canvas selector
- Observation building (build_observation)
- Reward computation (compute_reward)
- Termination detection (check_termination) with idle detection
- Action application (apply_action — MultiDiscrete to JS calls)
- Modal handling (handle_modals — level_complete, settings, modal, main_menu)
- Game start (start_game)
- Info dict contents (build_info)
- Terminal reward
- Reset detection validation (on_reset_detections)
- Reset termination state
- JS game state reading (_read_game_state, _detect_ui_state)
- _should_check_modals always returns True
- _lazy_init override (skip YOLO for CNN-only game)
- _detect_objects override (empty detections)
- _grid_to_canvas coordinate mapping
- _handle_level_transition
- on_lazy_init (training configuration)
- Idle detection with idle_threshold
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import numpy as np
import pytest

from games.shapez.env import _GRID_SIZE, _IDLE_THRESHOLD, ShapezEnv
from games.shapez.modal_handler import (
    CENTER_HUB_JS,
    CLICK_AT_JS,
    DELETE_AT_JS,
    DISMISS_UNLOCK_JS,
    NOOP_JS,
    PAN_CAMERA_JS,
    ROTATE_BUILDING_JS,
    SELECT_BUILDING_JS,
)

# -- Helpers -------------------------------------------------------------------


def _mock_driver(
    *,
    level: int = 1,
    shapes_delivered: int = 0,
    goal_required: int = 30,
    goal_progress: float = 0.0,
    entity_count: int = 0,
    running: bool = True,
    in_game: bool = True,
    ui_state: str = "gameplay",
) -> mock.MagicMock:
    """Create a mock Selenium WebDriver for ShapezEnv testing.

    Parameters
    ----------
    level : int
        Current game level.
    shapes_delivered : int
        Number of shapes delivered to hub.
    goal_required : int
        Shapes required for current level.
    goal_progress : float
        Goal progress [0, 1].
    entity_count : int
        Number of buildings placed.
    running : bool
        Whether the game is in s10_gameRunning stage.
    in_game : bool
        Whether we are in InGameState.
    ui_state : str
        UI state returned by DETECT_STATE_JS.
    """
    driver = mock.MagicMock()

    def _execute_script(js, *args):
        # READ_GAME_STATE_JS
        if "hubGoals" in js and "entityCount" in js and "goalProgress" in js:
            return {
                "level": level,
                "shapesDelivered": shapes_delivered,
                "goalRequired": goal_required,
                "goalProgress": goal_progress,
                "entityCount": entity_count,
                "upgradeLevels": {},
                "running": running,
                "inGame": in_game,
            }
        # DETECT_STATE_JS
        if "bodyId" in js and "unlockNotification" in js:
            return {"state": ui_state, "details": {"bodyId": "state_InGameState"}}
        # SETUP_TRAINING_JS
        if "offerHints" in js:
            return {"actions": ["tutorials_disabled"]}
        # START_NEW_GAME_JS
        if "createNewSavegame" in js:
            return {"action": "play_invoked"}
        # DISMISS_UNLOCK_JS
        if "unlockNotification" in js and "mainButton" in js:
            return {"action": "unlock_dismissed"}
        # CLOSE_SETTINGS_JS
        if "settingsMenu" in js and "close" in js:
            return {"action": "settings_closed"}
        # DISMISS_MODAL_JS
        if "modalDialogs" in js and "dialogStack" in js and "ok" in js:
            return {"action": "modal_dismissed"}
        # SELECT_BUILDING_JS
        if "Digit" in js:
            return {"ok": True, "key": str(args[0]) if args else "1"}
        # ROTATE_BUILDING_JS
        if "KeyR" in js:
            return {"ok": True, "action": "rotate"}
        # CLICK_AT_JS
        if "mousedown" in js and "button: 0" in js:
            return {"ok": True, "x": args[0] if args else 0, "y": args[1] if len(args) > 1 else 0}
        # DELETE_AT_JS
        if "mousedown" in js and "button: 2" in js:
            return {"ok": True, "action": "delete"}
        # PAN_CAMERA_JS
        if "KeyW" in js and "KeyS" in js:
            return {"ok": True, "direction": args[0] if args else "up"}
        # CENTER_HUB_JS
        if "Space" in js and "center_hub" in js:
            return {"ok": True, "action": "center_hub"}
        # NOOP_JS
        if "noop" in js:
            return {"ok": True, "action": "noop"}
        return None

    driver.execute_script.side_effect = _execute_script
    return driver


def _make_env(**kwargs) -> ShapezEnv:
    """Create a ShapezEnv with sensible test defaults."""
    defaults: dict[str, Any] = {
        "window_title": "shapez.io",
        "max_steps": 5000,
    }
    defaults.update(kwargs)
    return ShapezEnv(**defaults)


# -- Construction & Spaces ----------------------------------------------------


class TestConstruction:
    """Tests for ShapezEnv constructor and space definitions."""

    def test_env_construction_default(self):
        """ShapezEnv can be constructed with default args."""
        env = _make_env()
        assert env is not None

    def test_observation_space_shape(self):
        """Observation space should be an 8-element Box."""
        env = _make_env()
        assert env.observation_space.shape == (8,)

    def test_observation_space_bounds(self):
        """All observation dimensions in [0, 1]."""
        env = _make_env()
        assert float(env.observation_space.low[0]) == 0.0
        assert float(env.observation_space.high[0]) == 1.0
        assert float(env.observation_space.low[7]) == 0.0
        assert float(env.observation_space.high[7]) == 1.0

    def test_action_space_multidiscrete(self):
        """Action space should be MultiDiscrete([7, 10, 16, 16, 4])."""
        env = _make_env()
        expected = [7, 10, _GRID_SIZE, _GRID_SIZE, 4]
        np.testing.assert_array_equal(env.action_space.nvec, expected)

    def test_custom_max_steps(self):
        """Constructor accepts custom max_steps."""
        env = _make_env(max_steps=2000)
        assert env.max_steps == 2000

    def test_driver_parameter_stored(self):
        """Constructor stores the Selenium driver."""
        driver = mock.MagicMock()
        env = _make_env(driver=driver)
        assert env._driver is driver

    def test_default_window_title(self):
        """Default window title is 'shapez.io'."""
        env = ShapezEnv()
        assert env.window_title == "shapez.io"

    def test_default_reward_mode(self):
        """Default reward_mode is survival."""
        env = ShapezEnv()
        assert env.reward_mode == "survival"

    def test_headless_parameter(self):
        """Constructor accepts headless parameter."""
        env = _make_env(headless=True)
        assert env.headless is True

    def test_initial_game_state(self):
        """Initial game-specific state is zeroed."""
        env = _make_env()
        assert env._prev_level == 0
        assert env._prev_entity_count == 0
        assert env._prev_shapes_delivered == 0
        assert env._idle_count == 0
        assert env._canvas_ready is False

    def test_custom_idle_threshold(self):
        """Constructor accepts custom idle_threshold."""
        env = _make_env(idle_threshold=500)
        assert env._idle_threshold == 500

    def test_default_idle_threshold(self):
        """Default idle threshold matches module constant."""
        env = _make_env()
        assert env._idle_threshold == _IDLE_THRESHOLD

    def test_training_configured_initially_false(self):
        """_training_configured is False initially."""
        env = _make_env()
        assert env._training_configured is False


# -- Game classes & canvas selector -------------------------------------------


class TestGameClasses:
    """Tests for game_classes() — should return empty list (CNN-only)."""

    def test_game_classes_empty(self):
        """shapez.io uses CNN-only: game_classes returns empty list."""
        env = _make_env()
        assert env.game_classes() == []

    def test_game_classes_returns_list(self):
        """Return type is list."""
        env = _make_env()
        result = env.game_classes()
        assert isinstance(result, list)


class TestCanvasSelector:
    """Tests for canvas_selector()."""

    def test_canvas_selector_returns_id(self):
        """Canvas CSS selector is 'ingame_Canvas'."""
        env = _make_env()
        assert env.canvas_selector() == "ingame_Canvas"


# -- Observation building -----------------------------------------------------


class TestBuildObservation:
    """Tests for build_observation()."""

    def test_observation_shape(self):
        """Observation has 8 elements."""
        driver = _mock_driver(level=5, entity_count=100)
        env = _make_env(driver=driver)
        obs = env.build_observation({})
        assert obs.shape == (8,)

    def test_observation_dtype(self):
        """Observation dtype is float32."""
        driver = _mock_driver()
        env = _make_env(driver=driver)
        obs = env.build_observation({})
        assert obs.dtype == np.float32

    def test_level_normalisation(self):
        """Level 13 normalises to 0.5 (soft cap at 26)."""
        driver = _mock_driver(level=13)
        env = _make_env(driver=driver)
        obs = env.build_observation({})
        assert abs(obs[0] - 0.5) < 1e-6

    def test_level_normalisation_caps_at_one(self):
        """Level above 26 caps at 1.0."""
        driver = _mock_driver(level=30)
        env = _make_env(driver=driver)
        obs = env.build_observation({})
        assert abs(obs[0] - 1.0) < 1e-6

    def test_level_zero(self):
        """Level 0 gives 0.0."""
        driver = _mock_driver(level=0)
        env = _make_env(driver=driver)
        obs = env.build_observation({})
        assert abs(obs[0] - 0.0) < 1e-6

    def test_goal_progress_normalisation(self):
        """Goal progress 0.5 maps directly."""
        driver = _mock_driver(goal_progress=0.5)
        env = _make_env(driver=driver)
        obs = env.build_observation({})
        assert abs(obs[1] - 0.5) < 1e-6

    def test_goal_progress_clamps(self):
        """Goal progress > 1 clamps to 1.0."""
        driver = _mock_driver(goal_progress=1.5)
        env = _make_env(driver=driver)
        obs = env.build_observation({})
        assert abs(obs[1] - 1.0) < 1e-6

    def test_entity_count_normalisation(self):
        """500 entities normalises to 0.5 (soft cap at 1000)."""
        driver = _mock_driver(entity_count=500)
        env = _make_env(driver=driver)
        obs = env.build_observation({})
        assert abs(obs[2] - 0.5) < 1e-6

    def test_running_flag_true(self):
        """When game is running, running=1.0."""
        driver = _mock_driver(running=True)
        env = _make_env(driver=driver)
        obs = env.build_observation({})
        assert abs(obs[3] - 1.0) < 1e-6

    def test_running_flag_false(self):
        """When game is not running, running=0.0."""
        driver = _mock_driver(running=False)
        env = _make_env(driver=driver)
        obs = env.build_observation({})
        assert abs(obs[3] - 0.0) < 1e-6

    def test_unused_dimensions_zero(self):
        """Dimensions 4-7 are unused placeholders (0.0)."""
        driver = _mock_driver()
        env = _make_env(driver=driver)
        obs = env.build_observation({})
        for i in range(4, 8):
            assert abs(obs[i] - 0.0) < 1e-6

    def test_reset_flag_stores_prev_state(self):
        """When reset=True, tracking state is updated."""
        driver = _mock_driver(level=3, entity_count=42, shapes_delivered=10)
        env = _make_env(driver=driver)
        env.build_observation({}, reset=True)
        assert env._prev_level == 3
        assert env._prev_entity_count == 42
        assert env._prev_shapes_delivered == 10

    def test_no_driver_returns_zeros(self):
        """Without driver, observation is all zeros."""
        env = _make_env()
        obs = env.build_observation({})
        assert np.allclose(obs, 0.0)

    def test_observation_within_space_bounds(self):
        """Observation should be within the observation space."""
        driver = _mock_driver(level=10, goal_progress=0.5, entity_count=200)
        env = _make_env(driver=driver)
        obs = env.build_observation({})
        assert env.observation_space.contains(obs)


# -- Reward computation -------------------------------------------------------


class TestComputeReward:
    """Tests for compute_reward()."""

    def test_shape_delivery_reward(self):
        """Positive shape delivery delta gives reward."""
        driver = _mock_driver(shapes_delivered=10, entity_count=0, level=1)
        env = _make_env(driver=driver)
        env._prev_shapes_delivered = 5
        env._prev_entity_count = 0
        env._prev_level = 1
        reward = env.compute_reward({}, terminated=False, level_cleared=False)
        # shapes_delta=5, reward = 5*0.01 + 0 + 0 - 0.001 = 0.049
        assert abs(reward - 0.049) < 1e-6

    def test_entity_placement_reward(self):
        """Positive entity delta gives reward."""
        driver = _mock_driver(shapes_delivered=0, entity_count=10, level=1)
        env = _make_env(driver=driver)
        env._prev_shapes_delivered = 0
        env._prev_entity_count = 5
        env._prev_level = 1
        reward = env.compute_reward({}, terminated=False, level_cleared=False)
        # entity_delta=5, reward = 0 + 5*0.005 + 0 - 0.001 = 0.024
        assert abs(reward - 0.024) < 1e-6

    def test_level_completion_bonus(self):
        """Level completion gives +1.0 bonus."""
        driver = _mock_driver(shapes_delivered=0, entity_count=0, level=2)
        env = _make_env(driver=driver)
        env._prev_shapes_delivered = 0
        env._prev_entity_count = 0
        env._prev_level = 1
        reward = env.compute_reward({}, terminated=False, level_cleared=True)
        # level_delta=1, reward = 0 + 0 + 1.0 - 0.001 = 0.999
        assert abs(reward - 0.999) < 1e-6

    def test_time_penalty_only(self):
        """No progress gives only time penalty."""
        driver = _mock_driver(shapes_delivered=0, entity_count=0, level=1)
        env = _make_env(driver=driver)
        env._prev_shapes_delivered = 0
        env._prev_entity_count = 0
        env._prev_level = 1
        reward = env.compute_reward({}, terminated=False, level_cleared=False)
        assert abs(reward - (-0.001)) < 1e-6

    def test_combined_rewards(self):
        """Multiple reward components combine correctly."""
        driver = _mock_driver(shapes_delivered=20, entity_count=15, level=3)
        env = _make_env(driver=driver)
        env._prev_shapes_delivered = 10
        env._prev_entity_count = 10
        env._prev_level = 2
        reward = env.compute_reward({}, terminated=False, level_cleared=True)
        # shapes=10*0.01 + entity=5*0.005 + level=1*1.0 - 0.001
        expected = 0.1 + 0.025 + 1.0 - 0.001
        assert abs(reward - expected) < 1e-6

    def test_prev_state_updated_after_reward(self):
        """Tracking state is updated after reward computation."""
        driver = _mock_driver(shapes_delivered=50, entity_count=30, level=5)
        env = _make_env(driver=driver)
        env._prev_shapes_delivered = 0
        env._prev_entity_count = 0
        env._prev_level = 1
        env.compute_reward({}, terminated=False, level_cleared=False)
        assert env._prev_shapes_delivered == 50
        assert env._prev_entity_count == 30
        assert env._prev_level == 5

    def test_negative_delta_no_reward(self):
        """Negative deltas don't give reward (guard)."""
        driver = _mock_driver(shapes_delivered=0, entity_count=0, level=1)
        env = _make_env(driver=driver)
        env._prev_shapes_delivered = 10  # Decreased
        env._prev_entity_count = 5
        env._prev_level = 1
        reward = env.compute_reward({}, terminated=False, level_cleared=False)
        # Only time penalty (deltas are negative, not added)
        assert abs(reward - (-0.001)) < 1e-6


# -- Termination detection ----------------------------------------------------


class TestCheckTermination:
    """Tests for check_termination()."""

    def test_not_terminated_with_progress(self):
        """Active game with progress: not terminated."""
        driver = _mock_driver(shapes_delivered=10, entity_count=5, level=1)
        env = _make_env(driver=driver)
        env._prev_shapes_delivered = 5
        env._prev_entity_count = 3
        terminated, level_cleared = env.check_termination({})
        assert terminated is False

    def test_idle_count_increments(self):
        """Idle count increments when no progress."""
        driver = _mock_driver(shapes_delivered=0, entity_count=0, level=1)
        env = _make_env(driver=driver)
        env._prev_shapes_delivered = 0
        env._prev_entity_count = 0
        env.check_termination({})
        assert env._idle_count == 1

    def test_idle_count_resets_on_shape_delivery(self):
        """Idle count resets when shapes are delivered."""
        driver = _mock_driver(shapes_delivered=5, entity_count=0, level=1)
        env = _make_env(driver=driver)
        env._prev_shapes_delivered = 0
        env._prev_entity_count = 0
        env._idle_count = 100
        env.check_termination({})
        assert env._idle_count == 0

    def test_idle_count_resets_on_entity_placement(self):
        """Idle count resets when entities are placed."""
        driver = _mock_driver(shapes_delivered=0, entity_count=5, level=1)
        env = _make_env(driver=driver)
        env._prev_shapes_delivered = 0
        env._prev_entity_count = 0
        env._idle_count = 100
        env.check_termination({})
        assert env._idle_count == 0

    def test_idle_termination_at_threshold(self):
        """Terminates when idle count reaches threshold."""
        driver = _mock_driver(shapes_delivered=0, entity_count=0, level=1)
        env = _make_env(driver=driver, idle_threshold=3)
        env._prev_shapes_delivered = 0
        env._prev_entity_count = 0
        env._idle_count = 2  # Will increment to 3
        terminated, _ = env.check_termination({})
        assert terminated is True

    def test_not_terminated_below_threshold(self):
        """Not terminated just below idle threshold."""
        driver = _mock_driver(shapes_delivered=0, entity_count=0, level=1)
        env = _make_env(driver=driver, idle_threshold=5)
        env._prev_shapes_delivered = 0
        env._prev_entity_count = 0
        env._idle_count = 3  # Will increment to 4
        terminated, _ = env.check_termination({})
        assert terminated is False

    def test_level_cleared_detection(self):
        """Level cleared when level increases."""
        driver = _mock_driver(shapes_delivered=0, entity_count=0, level=2)
        env = _make_env(driver=driver)
        env._prev_level = 1
        env._prev_shapes_delivered = 0
        env._prev_entity_count = 0
        _, level_cleared = env.check_termination({})
        assert level_cleared is True

    def test_no_level_cleared_when_same_level(self):
        """No level cleared when level is same."""
        driver = _mock_driver(shapes_delivered=0, entity_count=0, level=1)
        env = _make_env(driver=driver)
        env._prev_level = 1
        env._prev_shapes_delivered = 0
        env._prev_entity_count = 0
        _, level_cleared = env.check_termination({})
        assert level_cleared is False

    def test_no_driver_defaults(self):
        """Without driver, no termination."""
        env = _make_env()
        env._prev_shapes_delivered = 0
        env._prev_entity_count = 0
        terminated, _ = env.check_termination({})
        # idle_count will increment (both are 0 == 0)
        assert env._idle_count == 1


# -- Action application -------------------------------------------------------


class TestApplyAction:
    """Tests for apply_action()."""

    def test_noop_action(self):
        """Action type 0 executes NOOP_JS."""
        driver = _mock_driver()
        env = _make_env(driver=driver)
        env.apply_action(np.array([0, 0, 0, 0, 0]))
        driver.execute_script.assert_called_once_with(NOOP_JS)

    def test_select_building(self):
        """Action type 1 executes SELECT_BUILDING_JS with key."""
        driver = _mock_driver()
        env = _make_env(driver=driver)
        env.apply_action(np.array([1, 3, 0, 0, 0]))
        # building_id=3 -> key=(3+1)%10=4
        driver.execute_script.assert_called_once_with(SELECT_BUILDING_JS, 4)

    def test_select_building_id_wraps(self):
        """Building ID 9 wraps to key 0."""
        driver = _mock_driver()
        env = _make_env(driver=driver)
        env.apply_action(np.array([1, 9, 0, 0, 0]))
        # building_id=9 -> key=(9+1)%10=0
        driver.execute_script.assert_called_once_with(SELECT_BUILDING_JS, 0)

    def test_place_at_grid(self):
        """Action type 2 places at grid position via CLICK_AT_JS."""
        driver = _mock_driver()
        env = _make_env(driver=driver)
        env._canvas_size = (1280, 1024)
        env._canvas_ready = True
        env.apply_action(np.array([2, 0, 8, 8, 0]))
        # grid (8,8) -> center of canvas
        assert driver.execute_script.called
        call_args = driver.execute_script.call_args
        assert call_args[0][0] == CLICK_AT_JS

    def test_delete_at_grid(self):
        """Action type 3 deletes at grid position via DELETE_AT_JS."""
        driver = _mock_driver()
        env = _make_env(driver=driver)
        env._canvas_size = (1280, 1024)
        env._canvas_ready = True
        env.apply_action(np.array([3, 0, 5, 5, 0]))
        call_args = driver.execute_script.call_args
        assert call_args[0][0] == DELETE_AT_JS

    def test_rotate_building(self):
        """Action type 4 executes ROTATE_BUILDING_JS."""
        driver = _mock_driver()
        env = _make_env(driver=driver)
        env.apply_action(np.array([4, 0, 0, 0, 0]))
        driver.execute_script.assert_called_once_with(ROTATE_BUILDING_JS)

    def test_pan_camera(self):
        """Action type 5 executes PAN_CAMERA_JS with direction."""
        driver = _mock_driver()
        env = _make_env(driver=driver)
        env.apply_action(np.array([5, 0, 0, 0, 2]))
        # pan_dir=2 -> "left"
        driver.execute_script.assert_called_once_with(PAN_CAMERA_JS, "left")

    def test_pan_camera_directions(self):
        """All 4 pan directions map correctly."""
        expected = ["up", "down", "left", "right"]
        for i, direction in enumerate(expected):
            driver = _mock_driver()
            env = _make_env(driver=driver)
            env.apply_action(np.array([5, 0, 0, 0, i]))
            driver.execute_script.assert_called_once_with(PAN_CAMERA_JS, direction)

    def test_center_hub(self):
        """Action type 6 executes CENTER_HUB_JS."""
        driver = _mock_driver()
        env = _make_env(driver=driver)
        env.apply_action(np.array([6, 0, 0, 0, 0]))
        driver.execute_script.assert_called_once_with(CENTER_HUB_JS)

    def test_no_driver_noop(self):
        """Without driver, apply_action is a no-op."""
        env = _make_env()
        env.apply_action(np.array([1, 0, 0, 0, 0]))  # Should not raise

    def test_js_exception_caught(self):
        """JS execution errors are caught and logged, not raised."""
        driver = mock.MagicMock()
        driver.execute_script.side_effect = Exception("JS error")
        env = _make_env(driver=driver)
        env.apply_action(np.array([0, 0, 0, 0, 0]))  # Should not raise

    def test_short_action_ignored(self):
        """Action with fewer than 5 elements is ignored."""
        driver = _mock_driver()
        env = _make_env(driver=driver)
        env.apply_action(np.array([0, 1]))  # Too short
        driver.execute_script.assert_not_called()

    def test_ndarray_action(self):
        """Works with numpy array actions."""
        driver = _mock_driver()
        env = _make_env(driver=driver)
        env.apply_action(np.array([4, 0, 0, 0, 0], dtype=np.int64))
        driver.execute_script.assert_called_once_with(ROTATE_BUILDING_JS)

    def test_place_calls_ensure_canvas_ready(self):
        """Place action calls _ensure_canvas_ready before grid mapping."""
        driver = _mock_driver()
        env = _make_env(driver=driver)
        env._canvas_size = (1280, 1024)
        with mock.patch.object(env, "_ensure_canvas_ready") as mock_ensure:
            env.apply_action(np.array([2, 0, 8, 8, 0]))
            mock_ensure.assert_called_once()

    def test_delete_calls_ensure_canvas_ready(self):
        """Delete action calls _ensure_canvas_ready before grid mapping."""
        driver = _mock_driver()
        env = _make_env(driver=driver)
        env._canvas_size = (1280, 1024)
        with mock.patch.object(env, "_ensure_canvas_ready") as mock_ensure:
            env.apply_action(np.array([3, 0, 5, 5, 0]))
            mock_ensure.assert_called_once()

    def test_noop_does_not_call_ensure_canvas_ready(self):
        """Non-coordinate actions skip _ensure_canvas_ready."""
        driver = _mock_driver()
        env = _make_env(driver=driver)
        with mock.patch.object(env, "_ensure_canvas_ready") as mock_ensure:
            env.apply_action(np.array([0, 0, 0, 0, 0]))
            mock_ensure.assert_not_called()


# -- Canvas readiness --------------------------------------------------------


class TestEnsureCanvasReady:
    """Tests for _ensure_canvas_ready()."""

    def test_calls_init_canvas_when_not_ready(self):
        """_init_canvas is called when _canvas_ready is False."""
        driver = _mock_driver()
        env = _make_env(driver=driver)
        env._canvas_ready = False
        with mock.patch.object(env, "_init_canvas") as mock_init:

            def set_canvas():
                env._canvas_size = (1280, 1024)

            mock_init.side_effect = set_canvas
            env._ensure_canvas_ready()
            mock_init.assert_called_once()
        assert env._canvas_ready is True

    def test_skips_when_already_ready(self):
        """_init_canvas is not called when _canvas_ready is True."""
        driver = _mock_driver()
        env = _make_env(driver=driver)
        env._canvas_ready = True
        with mock.patch.object(env, "_init_canvas") as mock_init:
            env._ensure_canvas_ready()
            mock_init.assert_not_called()

    def test_not_marked_ready_for_small_canvas(self):
        """Canvas not marked ready if height is too small (body fallback)."""
        driver = _mock_driver()
        env = _make_env(driver=driver)
        env._canvas_ready = False
        with mock.patch.object(env, "_init_canvas") as mock_init:

            def set_small_canvas():
                env._canvas_size = (1264, 15)

            mock_init.side_effect = set_small_canvas
            env._ensure_canvas_ready()
            mock_init.assert_called_once()
        assert env._canvas_ready is False

    def test_not_marked_ready_when_canvas_size_is_none(self):
        """Canvas not marked ready if _canvas_size is None."""
        driver = _mock_driver()
        env = _make_env(driver=driver)
        env._canvas_ready = False
        with mock.patch.object(env, "_init_canvas") as mock_init:
            env._ensure_canvas_ready()
            mock_init.assert_called_once()
        assert env._canvas_ready is False


# -- Modal handling -----------------------------------------------------------


class TestHandleModals:
    """Tests for handle_modals()."""

    def test_gameplay_state_returns_gameplay(self):
        """When game is playing, returns 'gameplay'."""
        driver = _mock_driver(ui_state="gameplay")
        env = _make_env(driver=driver)
        env._canvas_ready = True  # Avoid canvas re-init in this test
        state = env.handle_modals()
        assert state == "gameplay"

    def test_level_complete_dismissed(self):
        """Level complete notification is dismissed when dismiss_game_over=True."""
        driver = _mock_driver(ui_state="level_complete")
        env = _make_env(driver=driver)
        state = env.handle_modals(dismiss_game_over=True)
        assert state == "level_complete"
        # Should have called DISMISS_UNLOCK_JS
        calls = [str(c) for c in driver.execute_script.call_args_list]
        dismiss_calls = [c for c in calls if "unlock" in c.lower() or "mainButton" in c]
        assert len(dismiss_calls) > 0

    def test_level_complete_not_dismissed(self):
        """Level complete not dismissed when dismiss_game_over=False."""
        driver = _mock_driver(ui_state="level_complete")
        env = _make_env(driver=driver)
        state = env.handle_modals(dismiss_game_over=False)
        assert state == "level_complete"
        # Only DETECT_STATE_JS called, not DISMISS_UNLOCK_JS
        assert driver.execute_script.call_count == 1

    def test_settings_state_closes(self):
        """Settings menu is closed."""
        driver = _mock_driver(ui_state="settings")
        env = _make_env(driver=driver)
        state = env.handle_modals()
        assert state == "settings"

    def test_modal_state_dismissed(self):
        """Generic modal is dismissed."""
        driver = _mock_driver(ui_state="modal")
        env = _make_env(driver=driver)
        state = env.handle_modals()
        assert state == "modal"

    def test_main_menu_starts_game(self):
        """Main menu triggers START_NEW_GAME_JS and sets guard flag."""
        driver = _mock_driver(ui_state="main_menu")
        env = _make_env(driver=driver)
        state = env.handle_modals()
        assert state == "main_menu"
        # Should have called DETECT_STATE_JS + START_NEW_GAME_JS
        assert driver.execute_script.call_count >= 2
        assert env._menu_start_requested is True

    def test_main_menu_no_repeat_after_requested(self):
        """Second handle_modals with main_menu skips START_NEW_GAME_JS."""
        driver = _mock_driver(ui_state="main_menu")
        env = _make_env(driver=driver)
        env._menu_start_requested = True
        state = env.handle_modals()
        assert state == "main_menu"
        # Only DETECT_STATE_JS should be called — no START_NEW_GAME_JS
        assert driver.execute_script.call_count == 1

    def test_menu_start_guard_cleared_on_gameplay(self):
        """_menu_start_requested is cleared once state leaves main_menu."""
        driver = _mock_driver(ui_state="gameplay")
        env = _make_env(driver=driver)
        env._menu_start_requested = True
        env._canvas_ready = True  # Avoid canvas re-init in this test
        env.handle_modals()
        assert env._menu_start_requested is False

    def test_canvas_reinit_on_gameplay_transition(self):
        """Canvas re-init is triggered when gameplay state is first detected."""
        driver = _mock_driver(ui_state="gameplay")
        env = _make_env(driver=driver)
        env._canvas_ready = False
        with mock.patch.object(env, "_ensure_canvas_ready") as mock_ensure:
            env.handle_modals()
            mock_ensure.assert_called_once()

    def test_canvas_reinit_skipped_when_already_ready(self):
        """Canvas re-init is skipped if _canvas_ready is True."""
        driver = _mock_driver(ui_state="gameplay")
        env = _make_env(driver=driver)
        env._canvas_ready = True
        with mock.patch.object(env, "_ensure_canvas_ready") as mock_ensure:
            env.handle_modals()
            mock_ensure.assert_not_called()

    def test_canvas_reinit_not_marked_ready_for_small_canvas(self):
        """Canvas not marked ready if height is too small (body fallback)."""
        driver = _mock_driver(ui_state="gameplay")
        env = _make_env(driver=driver)
        env._canvas_ready = False
        with mock.patch.object(env, "_init_canvas") as mock_init:

            def set_small_canvas():
                env._canvas_size = (1264, 15)

            mock_init.side_effect = set_small_canvas
            env.handle_modals()
            mock_init.assert_called_once()
        assert env._canvas_ready is False

    def test_no_driver_returns_gameplay(self):
        """Without driver, returns 'gameplay'."""
        env = _make_env()
        assert env.handle_modals() == "gameplay"

    def test_js_exception_returns_unknown(self):
        """JS execution error returns 'unknown'."""
        driver = mock.MagicMock()
        driver.execute_script.side_effect = Exception("JS error")
        env = _make_env(driver=driver)
        assert env.handle_modals() == "unknown"

    def test_null_state_returns_unknown(self):
        """None result from JS returns 'unknown'."""
        driver = mock.MagicMock()
        driver.execute_script.return_value = None
        env = _make_env(driver=driver)
        assert env.handle_modals() == "unknown"

    def test_loading_state_returned(self):
        """Loading state is returned as-is (no action taken)."""
        driver = _mock_driver(ui_state="loading")
        env = _make_env(driver=driver)
        state = env.handle_modals()
        assert state == "loading"

    def test_menu_state_returned(self):
        """Generic menu state is returned as-is."""
        driver = _mock_driver(ui_state="menu")
        env = _make_env(driver=driver)
        state = env.handle_modals()
        assert state == "menu"


# -- start_game ---------------------------------------------------------------


class TestStartGame:
    """Tests for start_game()."""

    @mock.patch("games.shapez.env.time")
    def test_start_game_calls_js_and_waits_for_ingame(self, mock_time):
        """start_game executes START_NEW_GAME_JS and polls until gameplay."""
        # Simulate monotonic clock advancing past each sleep
        mock_time.monotonic.side_effect = [0.0, 0.5, 1.0]
        mock_time.sleep = mock.MagicMock()

        # detect call 1 = pre-check (main_menu), 2 = poll (loading),
        # 3 = poll (gameplay)
        call_count = {"detect": 0}

        driver = _mock_driver(ui_state="main_menu")
        original_side_effect = driver.execute_script.side_effect

        def _script(js, *args):
            # DETECT_STATE_JS calls during pre-check and polling
            if "bodyId" in js and "unlockNotification" in js:
                call_count["detect"] += 1
                if call_count["detect"] >= 3:
                    return {"state": "gameplay", "details": {}}
                if call_count["detect"] == 1:
                    return {"state": "main_menu", "details": {}}
                return {"state": "loading", "details": {}}
            return original_side_effect(js, *args)

        driver.execute_script.side_effect = _script
        env = _make_env(driver=driver)
        with mock.patch.object(env, "_ensure_canvas_ready"):
            env.start_game()

        # Pre-check + at least 2 polls
        assert call_count["detect"] >= 3
        # Menu start guard should be cleared after reaching gameplay
        assert env._menu_start_requested is False
        # Verify START_NEW_GAME_JS was called (contains createNewSavegame)
        js_calls = [str(c) for c in driver.execute_script.call_args_list]
        assert any("createNewSavegame" in c for c in js_calls)

    def test_start_game_no_driver(self):
        """start_game without driver is a no-op."""
        env = _make_env()
        env.start_game()  # Should not raise

    @mock.patch("games.shapez.env.time")
    def test_start_game_js_exception_caught(self, mock_time):
        """JS execution error in START_NEW_GAME_JS is caught."""
        mock_time.monotonic.side_effect = [0.0]
        mock_time.sleep = mock.MagicMock()
        driver = mock.MagicMock()
        driver.execute_script.side_effect = Exception("JS error")
        env = _make_env(driver=driver)
        env.start_game()  # Should not raise

    @mock.patch("games.shapez.env.time")
    def test_start_game_timeout_logs_warning(self, mock_time):
        """start_game logs warning when InGameState is not reached."""
        # Clock jumps past deadline immediately
        mock_time.monotonic.side_effect = [0.0, 100.0]
        mock_time.sleep = mock.MagicMock()

        driver = _mock_driver(ui_state="main_menu")
        original_side_effect = driver.execute_script.side_effect

        def _script(js, *args):
            if "bodyId" in js and "unlockNotification" in js:
                return {"state": "loading", "details": {}}
            return original_side_effect(js, *args)

        driver.execute_script.side_effect = _script
        env = _make_env(driver=driver)
        env.start_game()
        # Should not raise — just logs a warning
        assert env._menu_start_requested is True

    @mock.patch("games.shapez.env.time")
    def test_start_game_immediate_gameplay(self, mock_time):
        """start_game returns immediately if already in gameplay (pre-check)."""
        mock_time.monotonic.return_value = 0.0
        mock_time.sleep = mock.MagicMock()

        driver = _mock_driver(ui_state="gameplay")
        env = _make_env(driver=driver)
        with mock.patch.object(env, "_ensure_canvas_ready"):
            env.start_game()
        # Gameplay detected on pre-check — guard cleared, no polling
        assert env._menu_start_requested is False
        # START_NEW_GAME_JS should NOT have been called
        js_calls = [str(c) for c in driver.execute_script.call_args_list]
        assert not any("createNewSavegame" in c for c in js_calls)

    @mock.patch("games.shapez.env.time")
    def test_start_game_reinits_canvas_on_gameplay(self, mock_time):
        """start_game resets _canvas_ready and calls _ensure_canvas_ready."""
        mock_time.monotonic.return_value = 0.0
        mock_time.sleep = mock.MagicMock()

        driver = _mock_driver(ui_state="gameplay")
        env = _make_env(driver=driver)
        env._canvas_ready = True  # Simulate stale canvas
        with mock.patch.object(env, "_ensure_canvas_ready") as mock_ecr:
            env.start_game()
        # _canvas_ready reset to False, then _ensure_canvas_ready called
        mock_ecr.assert_called_once()

    @mock.patch("games.shapez.env.time")
    def test_start_game_skips_js_when_already_requested(self, mock_time):
        """start_game skips START_NEW_GAME_JS if _menu_start_requested."""
        mock_time.monotonic.side_effect = [0.0, 0.5]
        mock_time.sleep = mock.MagicMock()

        call_count = {"detect": 0}
        driver = _mock_driver(ui_state="main_menu")
        original_side_effect = driver.execute_script.side_effect

        def _script(js, *args):
            if "bodyId" in js and "unlockNotification" in js:
                call_count["detect"] += 1
                # Pre-check returns main_menu, first poll returns gameplay
                if call_count["detect"] >= 2:
                    return {"state": "gameplay", "details": {}}
                return {"state": "main_menu", "details": {}}
            return original_side_effect(js, *args)

        driver.execute_script.side_effect = _script
        env = _make_env(driver=driver)
        env._menu_start_requested = True  # Already requested
        with mock.patch.object(env, "_ensure_canvas_ready"):
            env.start_game()

        # START_NEW_GAME_JS should NOT have been called
        js_calls = [str(c) for c in driver.execute_script.call_args_list]
        assert not any("createNewSavegame" in c for c in js_calls)
        # But guard should be cleared after reaching gameplay
        assert env._menu_start_requested is False

    @mock.patch("games.shapez.env.time")
    def test_start_game_retries_js_when_loading_then_main_menu(self, mock_time):
        """start_game retries START_NEW_GAME_JS after loading -> main_menu.

        Simulates the post-refresh scenario: the initial JS call fails
        because GLOBAL_APP is not available during PreloadState, then the
        page transitions to MainMenuState where the retry succeeds,
        followed by InGameState.
        """
        # 0.0 = deadline start, 0.5..1.5 = 3 poll iterations (2.0 spare)
        mock_time.monotonic.side_effect = [0.0, 0.5, 1.0, 1.5, 2.0]
        mock_time.sleep = mock.MagicMock()

        call_count = {"detect": 0, "start_js": 0}
        driver = _mock_driver(ui_state="loading")

        def _script(js, *args):
            # DETECT_STATE_JS
            if "bodyId" in js and "unlockNotification" in js:
                call_count["detect"] += 1
                # Pre-check (1) = loading, poll 1 (2) = loading,
                # poll 2 (3) = main_menu (retry happens here),
                # poll 3 (4) = gameplay
                if call_count["detect"] <= 2:
                    return {"state": "loading", "details": {}}
                if call_count["detect"] == 3:
                    return {"state": "main_menu", "details": {}}
                return {"state": "gameplay", "details": {}}
            # START_NEW_GAME_JS
            if "createNewSavegame" in js:
                call_count["start_js"] += 1
                if call_count["start_js"] == 1:
                    # First call fails — GLOBAL_APP not available
                    return {"action": "none", "error": "GLOBAL_APP not available"}
                # Retry succeeds
                return {"action": "play_invoked"}
            return None

        driver.execute_script.side_effect = _script
        env = _make_env(driver=driver)
        with mock.patch.object(env, "_ensure_canvas_ready"):
            env.start_game()

        # START_NEW_GAME_JS should have been called twice (initial + retry)
        assert call_count["start_js"] == 2
        # Guard cleared after reaching gameplay
        assert env._menu_start_requested is False
        # At least 4 detect calls (pre-check + 3 polls)
        assert call_count["detect"] >= 4


class TestBuildInfo:
    """Tests for build_info()."""

    def test_info_contains_level(self):
        """Info dict includes level."""
        driver = _mock_driver(level=5)
        env = _make_env(driver=driver)
        info = env.build_info({})
        assert info["level"] == 5

    def test_info_contains_shapes_delivered(self):
        """Info dict includes shapes_delivered."""
        driver = _mock_driver(shapes_delivered=42)
        env = _make_env(driver=driver)
        info = env.build_info({})
        assert info["shapes_delivered"] == 42

    def test_info_contains_goal_progress(self):
        """Info dict includes goal_progress."""
        driver = _mock_driver(goal_progress=0.75)
        env = _make_env(driver=driver)
        info = env.build_info({})
        assert abs(info["goal_progress"] - 0.75) < 1e-6

    def test_info_contains_entity_count(self):
        """Info dict includes entity_count."""
        driver = _mock_driver(entity_count=100)
        env = _make_env(driver=driver)
        info = env.build_info({})
        assert info["entity_count"] == 100

    def test_info_contains_idle_count(self):
        """Info dict includes idle_count."""
        env = _make_env()
        env._idle_count = 42
        info = env.build_info({})
        assert info["idle_count"] == 42

    def test_info_contains_running(self):
        """Info dict includes running flag."""
        driver = _mock_driver(running=True)
        env = _make_env(driver=driver)
        info = env.build_info({})
        assert info["running"] is True

    def test_info_contains_detections(self):
        """Info dict passes through detections."""
        env = _make_env()
        info = env.build_info({"foo": "bar"})
        assert info["detections"] == {"foo": "bar"}


# -- terminal_reward ----------------------------------------------------------


class TestTerminalReward:
    """Tests for terminal_reward()."""

    def test_terminal_reward_value(self):
        """Terminal reward is -5.001 (penalty + time penalty)."""
        env = _make_env()
        assert abs(env.terminal_reward() - (-5.001)) < 1e-6


# -- on_reset_detections ------------------------------------------------------


class TestOnResetDetections:
    """Tests for on_reset_detections()."""

    def test_accepts_gameplay_state(self):
        """Gameplay state is accepted."""
        driver = _mock_driver(ui_state="gameplay")
        env = _make_env(driver=driver)
        assert env.on_reset_detections({}) is True

    def test_accepts_main_menu(self):
        """Main menu state is accepted."""
        driver = _mock_driver(ui_state="main_menu")
        env = _make_env(driver=driver)
        assert env.on_reset_detections({}) is True

    def test_accepts_loading(self):
        """Loading state is accepted."""
        driver = _mock_driver(ui_state="loading")
        env = _make_env(driver=driver)
        assert env.on_reset_detections({}) is True

    def test_rejects_settings(self):
        """Settings state is rejected."""
        driver = _mock_driver(ui_state="settings")
        env = _make_env(driver=driver)
        assert env.on_reset_detections({}) is False

    def test_rejects_modal(self):
        """Modal state is rejected."""
        driver = _mock_driver(ui_state="modal")
        env = _make_env(driver=driver)
        assert env.on_reset_detections({}) is False

    def test_rejects_unknown(self):
        """Unknown state is rejected."""
        driver = _mock_driver(ui_state="unknown")
        env = _make_env(driver=driver)
        assert env.on_reset_detections({}) is False

    def test_no_driver_always_true(self):
        """Without driver, always accept."""
        env = _make_env()
        assert env.on_reset_detections({}) is True


# -- reset_termination_state --------------------------------------------------


class TestResetTerminationState:
    """Tests for reset_termination_state()."""

    def test_resets_idle_count(self):
        """Idle count is reset to 0."""
        env = _make_env()
        env._idle_count = 500
        env.reset_termination_state()
        assert env._idle_count == 0

    def test_resets_menu_start_flag(self):
        """_menu_start_requested is reset to False."""
        env = _make_env()
        env._menu_start_requested = True
        driver = _mock_driver()
        env._driver = driver
        env.reset_termination_state()
        assert env._menu_start_requested is False

    def test_resets_canvas_ready_flag(self):
        """_canvas_ready is reset to False for canvas re-init on next episode."""
        env = _make_env()
        env._canvas_ready = True
        env.reset_termination_state()
        assert env._canvas_ready is False

    def test_resets_prev_tracking_state(self):
        """Previous tracking state is updated from JS bridge."""
        driver = _mock_driver(level=5, entity_count=42, shapes_delivered=100)
        env = _make_env(driver=driver)
        env._prev_level = 0
        env._prev_entity_count = 0
        env._prev_shapes_delivered = 0
        env.reset_termination_state()
        assert env._prev_level == 5
        assert env._prev_entity_count == 42
        assert env._prev_shapes_delivered == 100


# -- _read_game_state ---------------------------------------------------------


class TestReadGameState:
    """Tests for _read_game_state() private helper."""

    def test_returns_js_values(self):
        """Returns values from JS bridge."""
        driver = _mock_driver(level=3, shapes_delivered=15, entity_count=8)
        env = _make_env(driver=driver)
        state = env._read_game_state()
        assert state["level"] == 3
        assert state["shapesDelivered"] == 15
        assert state["entityCount"] == 8
        assert state["running"] is True

    def test_no_driver_returns_defaults(self):
        """Without driver, returns default dict."""
        env = _make_env()
        state = env._read_game_state()
        assert state["level"] == 0
        assert state["shapesDelivered"] == 0
        assert state["entityCount"] == 0
        assert state["running"] is False

    def test_js_exception_returns_defaults(self):
        """JS error returns default dict."""
        driver = mock.MagicMock()
        driver.execute_script.side_effect = Exception("JS error")
        env = _make_env(driver=driver)
        state = env._read_game_state()
        assert state["level"] == 0

    def test_null_result_returns_defaults(self):
        """None result from JS returns defaults."""
        driver = mock.MagicMock()
        driver.execute_script.return_value = None
        env = _make_env(driver=driver)
        state = env._read_game_state()
        assert state["level"] == 0


# -- _detect_ui_state ---------------------------------------------------------


class TestDetectUiState:
    """Tests for _detect_ui_state() private helper."""

    def test_returns_ui_state(self):
        """Returns UI state from JS bridge."""
        driver = _mock_driver(ui_state="gameplay")
        env = _make_env(driver=driver)
        state = env._detect_ui_state()
        assert state["state"] == "gameplay"

    def test_no_driver_returns_defaults(self):
        """Without driver, returns default dict."""
        env = _make_env()
        state = env._detect_ui_state()
        assert state["state"] == "unknown"

    def test_js_exception_returns_defaults(self):
        """JS error returns default dict."""
        driver = mock.MagicMock()
        driver.execute_script.side_effect = Exception("JS error")
        env = _make_env(driver=driver)
        state = env._detect_ui_state()
        assert state["state"] == "unknown"


# -- _should_check_modals -----------------------------------------------------


class TestShouldCheckModals:
    """Tests for _should_check_modals() override."""

    def test_always_returns_true(self):
        """shapez.io always checks modals."""
        env = _make_env()
        assert env._should_check_modals() is True


# -- _grid_to_canvas ----------------------------------------------------------


class TestGridToCanvas:
    """Tests for _grid_to_canvas() coordinate mapping."""

    def test_center_of_grid(self):
        """Grid center maps to canvas center."""
        env = _make_env()
        env._canvas_size = (1600, 1200)
        # grid (8, 6) should map to roughly the center region
        px, py = env._grid_to_canvas(8, 6)
        cell_w = 1600 / _GRID_SIZE
        cell_h = 1200 / _GRID_SIZE
        expected_x = int((8 + 0.5) * cell_w)
        expected_y = int((6 + 0.5) * cell_h)
        assert px == expected_x
        assert py == expected_y

    def test_origin_grid(self):
        """Grid (0, 0) maps to top-left cell center."""
        env = _make_env()
        env._canvas_size = (1280, 1024)
        px, py = env._grid_to_canvas(0, 0)
        cell_w = 1280 / _GRID_SIZE
        cell_h = 1024 / _GRID_SIZE
        assert px == int(0.5 * cell_w)
        assert py == int(0.5 * cell_h)

    def test_max_grid(self):
        """Grid (15, 15) maps to bottom-right cell center."""
        env = _make_env()
        env._canvas_size = (1280, 1024)
        px, py = env._grid_to_canvas(15, 15)
        cell_w = 1280 / _GRID_SIZE
        cell_h = 1024 / _GRID_SIZE
        assert px == int(15.5 * cell_w)
        assert py == int(15.5 * cell_h)

    def test_fallback_dimensions(self):
        """Without canvas size, falls back to 1280x1024."""
        env = _make_env()
        env._canvas_size = None
        px, py = env._grid_to_canvas(0, 0)
        cell_w = 1280 / _GRID_SIZE
        cell_h = 1024 / _GRID_SIZE
        assert px == int(0.5 * cell_w)
        assert py == int(0.5 * cell_h)


# -- _handle_level_transition -------------------------------------------------


class TestHandleLevelTransition:
    """Tests for _handle_level_transition() override."""

    def test_dismisses_unlock_notification(self):
        """Level transition dismisses unlock notification."""
        driver = _mock_driver()
        env = _make_env(driver=driver)
        result = env._handle_level_transition()
        assert result is True
        # Should have called DISMISS_UNLOCK_JS
        driver.execute_script.assert_called_once_with(DISMISS_UNLOCK_JS)

    def test_no_driver_returns_false(self):
        """Without driver, returns False."""
        env = _make_env()
        assert env._handle_level_transition() is False

    def test_js_exception_returns_false(self):
        """JS error returns False."""
        driver = mock.MagicMock()
        driver.execute_script.side_effect = Exception("JS error")
        env = _make_env(driver=driver)
        assert env._handle_level_transition() is False


# -- on_lazy_init (training configuration) ------------------------------------


class TestOnLazyInit:
    """Tests for on_lazy_init() hook."""

    def test_configures_training_settings(self):
        """on_lazy_init executes SETUP_TRAINING_JS."""
        driver = _mock_driver()
        env = _make_env(driver=driver)
        env.on_lazy_init()
        # Should have called SETUP_TRAINING_JS
        calls = [str(c) for c in driver.execute_script.call_args_list]
        setup_calls = [c for c in calls if "offerHints" in c]
        assert len(setup_calls) > 0
        assert env._training_configured is True

    def test_idempotent(self):
        """Second call is a no-op."""
        driver = _mock_driver()
        env = _make_env(driver=driver)
        env.on_lazy_init()
        call_count = driver.execute_script.call_count
        env.on_lazy_init()
        assert driver.execute_script.call_count == call_count

    def test_no_driver_skips(self):
        """Without driver, does not crash."""
        env = _make_env()
        env.on_lazy_init()  # Should not raise
        assert env._training_configured is False

    def test_js_exception_sets_not_configured(self):
        """JS error leaves _training_configured as False."""
        driver = mock.MagicMock()
        driver.execute_script.side_effect = Exception("JS error")
        env = _make_env(driver=driver)
        env.on_lazy_init()
        assert env._training_configured is False


# -- _lazy_init (skip YOLO) ---------------------------------------------------


class TestLazyInit:
    """Tests for _lazy_init() override that skips YOLO initialization."""

    def test_lazy_init_skips_yolo_headless(self):
        """In headless mode, _lazy_init completes without YOLO detector."""
        driver = _mock_driver()
        env = _make_env(driver=driver, headless=True)
        env._lazy_init()
        assert env._initialized is True
        assert env._detector is None

    def test_lazy_init_idempotent(self):
        """Second call to _lazy_init is a no-op."""
        driver = _mock_driver()
        env = _make_env(driver=driver, headless=True)
        env._lazy_init()
        env._lazy_init()  # Should not raise
        assert env._initialized is True

    def test_lazy_init_requires_driver_in_headless(self):
        """Headless mode without driver raises RuntimeError."""
        env = _make_env(headless=True)
        with pytest.raises(RuntimeError, match="Headless mode requires"):
            env._lazy_init()

    def test_lazy_init_calls_on_lazy_init_hook(self):
        """on_lazy_init() hook is called after initialization."""
        driver = _mock_driver()
        env = _make_env(driver=driver, headless=True)
        with mock.patch.object(env, "on_lazy_init") as mock_hook:
            env._lazy_init()
            mock_hook.assert_called_once()

    def test_lazy_init_sets_initialized_flag(self):
        """_initialized flag is set to True after init."""
        driver = _mock_driver()
        env = _make_env(driver=driver, headless=True)
        assert env._initialized is False
        env._lazy_init()
        assert env._initialized is True


# -- _detect_objects (empty detections) ----------------------------------------


class TestDetectObjects:
    """Tests for _detect_objects() override returning empty detections."""

    def test_returns_empty_dict(self):
        """_detect_objects always returns empty dict (no YOLO)."""
        env = _make_env()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = env._detect_objects(frame)
        assert result == {}

    def test_ignores_frame_content(self):
        """Frame content is irrelevant (no detection model)."""
        env = _make_env()
        frame = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        result = env._detect_objects(frame)
        assert result == {}
        assert isinstance(result, dict)


# -- Score OCR parameter forwarding -------------------------------------------


class TestShapezScoreParamForwarding:
    """Tests for score OCR parameter forwarding to BaseGameEnv."""

    def test_accepts_score_region(self):
        """ShapezEnv accepts score_region and forwards to BaseGameEnv."""
        env = _make_env(
            reward_mode="score",
            score_region=(10, 20, 100, 30),
        )
        assert env._score_region == (10, 20, 100, 30)
        assert env._score_ocr is not None

    def test_accepts_score_ocr_interval(self):
        """ShapezEnv accepts score_ocr_interval."""
        env = _make_env(
            reward_mode="score",
            score_ocr_interval=10,
        )
        assert env._score_ocr_interval == 10

    def test_accepts_score_reward_coeff(self):
        """ShapezEnv accepts score_reward_coeff."""
        env = _make_env(
            reward_mode="score",
            score_reward_coeff=0.5,
        )
        assert env._score_reward_coeff == 0.5
