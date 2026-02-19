"""Tests for the HextrisEnv Gymnasium environment.

Covers:
- Construction and space definitions (observation, action)
- Game class list (CNN-only, no YOLO)
- Canvas selector
- Observation building (build_observation)
- Reward computation (compute_reward)
- Termination detection (check_termination)
- Action application (apply_action via JS rotation)
- Modal handling (handle_modals — start screen, gameplay, game over)
- Game start (start_game)
- Info dict contents (build_info)
- Terminal reward
- Reset detection validation (on_reset_detections)
- Reset termination state
- JS game state reading (_read_game_state)
- _should_check_modals always returns True
- _lazy_init override (skip YOLO for CNN-only game)
- _detect_objects override (empty detections)
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import numpy as np
import pytest

from games.hextris.env import HextrisEnv
from games.hextris.modal_handler import (
    ROTATE_LEFT_JS,
    ROTATE_RIGHT_JS,
    START_GAME_JS,
)

# -- Helpers -------------------------------------------------------------------


def _mock_driver(
    *,
    game_state: int = 1,
    score: int = 0,
    block_count: int = 5,
    state_str: str = "gameplay",
) -> mock.MagicMock:
    """Create a mock Selenium WebDriver for HextrisEnv testing.

    Parameters
    ----------
    game_state : int
        JS ``gameState`` value (0=start, 1=playing, 2=game_over).
    score : int
        Current JS ``score``.
    block_count : int
        Number of blocks (``blocks.length``).
    state_str : str
        State string returned by DETECT_STATE_JS.
    """
    driver = mock.MagicMock()

    def _execute_script(js, *args):
        if "gameState" in js and "blockCount" in js:
            # READ_GAME_STATE_JS
            return {
                "score": score,
                "gameState": game_state,
                "blockCount": block_count,
                "running": game_state == 1,
            }
        if "result" in js and "state" in js and "details" in js:
            # DETECT_STATE_JS
            return {"state": state_str, "details": {"gameState": game_state}}
        if "rotate" in js:
            return {"ok": True, "direction": "left" if "-1" in js else "right"}
        if "init" in js:
            return {"action": "init_restart", "text": ""}
        if "resumeGame" in js:
            return {"action": "resume_game"}
        return None

    driver.execute_script.side_effect = _execute_script
    return driver


def _make_env(**kwargs) -> HextrisEnv:
    """Create a HextrisEnv with sensible test defaults."""
    defaults: dict[str, Any] = {
        "window_title": "HEXTRIS",
        "max_steps": 5000,
    }
    defaults.update(kwargs)
    return HextrisEnv(**defaults)


# -- Construction & Spaces ----------------------------------------------------


class TestConstruction:
    """Tests for HextrisEnv constructor and space definitions."""

    def test_env_construction_default(self):
        """HextrisEnv can be constructed with default args."""
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

    def test_action_space_discrete_3(self):
        """Action space should be Discrete(3)."""
        env = _make_env()
        assert env.action_space.n == 3

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
        """Default window title is HEXTRIS."""
        env = HextrisEnv()
        assert env.window_title == "HEXTRIS"

    def test_default_reward_mode(self):
        """Default reward_mode is survival."""
        env = HextrisEnv()
        assert env.reward_mode == "survival"

    def test_headless_parameter(self):
        """Constructor accepts headless parameter."""
        env = _make_env(headless=True)
        assert env.headless is True

    def test_initial_game_state(self):
        """Initial game-specific state is zeroed."""
        env = _make_env()
        assert env._prev_score == 0
        assert env._game_over_count == 0


# -- Game classes & canvas selector -------------------------------------------


class TestGameClasses:
    """Tests for game_classes() — should return empty list (CNN-only)."""

    def test_game_classes_empty(self):
        """Hextris uses CNN-only: game_classes returns empty list."""
        env = _make_env()
        assert env.game_classes() == []

    def test_game_classes_returns_list(self):
        """Return type is list."""
        env = _make_env()
        result = env.game_classes()
        assert isinstance(result, list)


class TestCanvasSelector:
    """Tests for canvas_selector()."""

    def test_canvas_selector_returns_canvas(self):
        """Canvas CSS selector is 'canvas' (Hextris canvas ID)."""
        env = _make_env()
        assert env.canvas_selector() == "canvas"


# -- Observation building -----------------------------------------------------


class TestBuildObservation:
    """Tests for build_observation()."""

    def test_observation_shape(self):
        """Observation has 8 elements."""
        driver = _mock_driver(score=100, block_count=10)
        env = _make_env(driver=driver)
        obs = env.build_observation({})
        assert obs.shape == (8,)

    def test_observation_dtype(self):
        """Observation dtype is float32."""
        driver = _mock_driver()
        env = _make_env(driver=driver)
        obs = env.build_observation({})
        assert obs.dtype == np.float32

    def test_score_normalisation(self):
        """Score 5000 normalises to 0.5 (soft cap at 10000)."""
        driver = _mock_driver(score=5000)
        env = _make_env(driver=driver)
        obs = env.build_observation({})
        assert abs(obs[0] - 0.5) < 1e-6

    def test_score_normalisation_caps_at_one(self):
        """Score above 10000 caps at 1.0."""
        driver = _mock_driver(score=20000)
        env = _make_env(driver=driver)
        obs = env.build_observation({})
        assert abs(obs[0] - 1.0) < 1e-6

    def test_block_count_normalisation(self):
        """25 blocks normalises to 0.5 (soft cap at 50)."""
        driver = _mock_driver(block_count=25)
        env = _make_env(driver=driver)
        obs = env.build_observation({})
        assert abs(obs[1] - 0.5) < 1e-6

    def test_game_active_flag(self):
        """When game is running, game_active=1.0."""
        driver = _mock_driver(game_state=1)
        env = _make_env(driver=driver)
        obs = env.build_observation({})
        assert abs(obs[2] - 1.0) < 1e-6

    def test_game_inactive_flag(self):
        """When game is not running, game_active=0.0."""
        driver = _mock_driver(game_state=2)
        env = _make_env(driver=driver)
        obs = env.build_observation({})
        assert abs(obs[2] - 0.0) < 1e-6

    def test_unused_dimensions_zero(self):
        """Dimensions 3-7 are unused placeholders (0.0)."""
        driver = _mock_driver()
        env = _make_env(driver=driver)
        obs = env.build_observation({})
        for i in range(3, 8):
            assert abs(obs[i] - 0.0) < 1e-6

    def test_reset_flag_stores_prev_score(self):
        """When reset=True, _prev_score is updated."""
        driver = _mock_driver(score=42)
        env = _make_env(driver=driver)
        env.build_observation({}, reset=True)
        assert env._prev_score == 42

    def test_no_driver_returns_zeros(self):
        """Without driver, observation is all zeros."""
        env = _make_env()  # No driver
        obs = env.build_observation({})
        assert np.allclose(obs, 0.0)

    def test_observation_within_space_bounds(self):
        """Observation should be within the observation space."""
        driver = _mock_driver(score=3000, block_count=15)
        env = _make_env(driver=driver)
        obs = env.build_observation({})
        assert env.observation_space.contains(obs)


# -- Reward computation -------------------------------------------------------


class TestComputeReward:
    """Tests for compute_reward()."""

    def test_score_delta_reward(self):
        """Positive score delta gives positive reward component."""
        driver = _mock_driver(score=100)
        env = _make_env(driver=driver)
        env._prev_score = 50
        reward = env.compute_reward({}, terminated=False, level_cleared=False)
        # score_delta=50, reward=50*0.01 - 0.001 = 0.499
        assert abs(reward - 0.499) < 1e-6

    def test_no_score_delta_gives_time_penalty(self):
        """Zero score delta gives only the time penalty."""
        driver = _mock_driver(score=0)
        env = _make_env(driver=driver)
        env._prev_score = 0
        reward = env.compute_reward({}, terminated=False, level_cleared=False)
        assert abs(reward - (-0.001)) < 1e-6

    def test_terminal_penalty(self):
        """Termination applies a -5.0 penalty."""
        driver = _mock_driver(score=0)
        env = _make_env(driver=driver)
        env._prev_score = 0
        reward = env.compute_reward({}, terminated=True, level_cleared=False)
        # 0 * 0.01 - 0.001 - 5.0 = -5.001
        assert abs(reward - (-5.001)) < 1e-6

    def test_prev_score_updated_after_reward(self):
        """_prev_score is updated to current score after reward computation."""
        driver = _mock_driver(score=200)
        env = _make_env(driver=driver)
        env._prev_score = 100
        env.compute_reward({}, terminated=False, level_cleared=False)
        assert env._prev_score == 200

    def test_level_cleared_ignored(self):
        """Hextris has no levels — level_cleared=True has no effect."""
        driver = _mock_driver(score=0)
        env = _make_env(driver=driver)
        env._prev_score = 0
        r1 = env.compute_reward({}, terminated=False, level_cleared=False)
        env._prev_score = 0
        r2 = env.compute_reward({}, terminated=False, level_cleared=True)
        assert abs(r1 - r2) < 1e-6


# -- Termination detection ----------------------------------------------------


class TestCheckTermination:
    """Tests for check_termination()."""

    def test_not_game_over_when_playing(self):
        """gameState=1 (playing): not terminated."""
        driver = _mock_driver(game_state=1)
        env = _make_env(driver=driver)
        terminated, level_cleared = env.check_termination({})
        assert terminated is False
        assert level_cleared is False

    def test_game_over_needs_confirmation_frames(self):
        """gameState=2 on first frame: not yet terminated (needs 3 frames)."""
        driver = _mock_driver(game_state=2)
        env = _make_env(driver=driver)
        terminated, _ = env.check_termination({})
        assert terminated is False
        assert env._game_over_count == 1

    def test_game_over_confirmed_after_3_frames(self):
        """gameState=2 for 3 consecutive frames: terminated."""
        driver = _mock_driver(game_state=2)
        env = _make_env(driver=driver)
        for _ in range(2):
            terminated, _ = env.check_termination({})
            assert terminated is False
        terminated, level_cleared = env.check_termination({})
        assert terminated is True
        assert level_cleared is False

    def test_game_over_counter_resets_on_non_game_over(self):
        """Counter resets if gameState switches away from 2."""
        driver = _mock_driver(game_state=2)
        env = _make_env(driver=driver)
        env.check_termination({})
        assert env._game_over_count == 1

        # Switch to playing
        driver.execute_script.side_effect = lambda js, *a: {
            "score": 0,
            "gameState": 1,
            "blockCount": 0,
            "running": True,
        }
        env.check_termination({})
        assert env._game_over_count == 0

    def test_level_cleared_always_false(self):
        """Hextris never signals level_cleared."""
        driver = _mock_driver(game_state=1)
        env = _make_env(driver=driver)
        _, level_cleared = env.check_termination({})
        assert level_cleared is False

    def test_no_driver_defaults(self):
        """Without driver, gameState defaults to -99, no termination."""
        env = _make_env()
        terminated, _ = env.check_termination({})
        assert terminated is False


# -- Action application -------------------------------------------------------


class TestApplyAction:
    """Tests for apply_action()."""

    def test_noop_action(self):
        """Action 0 does nothing (no JS executed)."""
        driver = _mock_driver()
        env = _make_env(driver=driver)
        env.apply_action(0)
        driver.execute_script.assert_not_called()

    def test_rotate_left(self):
        """Action 1 executes ROTATE_LEFT_JS."""
        driver = _mock_driver()
        env = _make_env(driver=driver)
        env.apply_action(1)
        driver.execute_script.assert_called_once_with(ROTATE_LEFT_JS)

    def test_rotate_right(self):
        """Action 2 executes ROTATE_RIGHT_JS."""
        driver = _mock_driver()
        env = _make_env(driver=driver)
        env.apply_action(2)
        driver.execute_script.assert_called_once_with(ROTATE_RIGHT_JS)

    def test_ndarray_action_converted(self):
        """np.ndarray action is converted to int."""
        driver = _mock_driver()
        env = _make_env(driver=driver)
        env.apply_action(np.int64(2))
        driver.execute_script.assert_called_once_with(ROTATE_RIGHT_JS)

    def test_no_driver_noop(self):
        """Without driver, apply_action is a no-op."""
        env = _make_env()
        env.apply_action(1)  # Should not raise

    def test_js_exception_caught(self):
        """JS execution errors are caught and logged, not raised."""
        driver = mock.MagicMock()
        driver.execute_script.side_effect = Exception("JS error")
        env = _make_env(driver=driver)
        env.apply_action(1)  # Should not raise


# -- Modal handling -----------------------------------------------------------


class TestHandleModals:
    """Tests for handle_modals()."""

    def test_gameplay_state_returns_gameplay(self):
        """When game is playing, returns 'gameplay'."""
        driver = _mock_driver(state_str="gameplay")
        env = _make_env(driver=driver)
        state = env.handle_modals()
        assert state == "gameplay"

    def test_game_over_dismissed(self):
        """When game_over detected with dismiss_game_over=True, restart is called."""
        driver = _mock_driver(state_str="game_over", game_state=2)
        env = _make_env(driver=driver)
        state = env.handle_modals(dismiss_game_over=True)
        assert state == "game_over"
        # Should have called DISMISS_GAME_OVER_JS
        calls = [str(c) for c in driver.execute_script.call_args_list]
        dismiss_calls = [c for c in calls if "init" in c]
        assert len(dismiss_calls) > 0

    def test_game_over_not_dismissed(self):
        """When dismiss_game_over=False, game_over is detected but not dismissed."""
        driver = _mock_driver(state_str="game_over", game_state=2)
        env = _make_env(driver=driver)
        state = env.handle_modals(dismiss_game_over=False)
        assert state == "game_over"
        # Only DETECT_STATE_JS should be called, not DISMISS_GAME_OVER_JS
        assert driver.execute_script.call_count == 1

    def test_menu_state_starts_game(self):
        """When on menu, start_game is called."""
        driver = _mock_driver(state_str="menu", game_state=0)
        env = _make_env(driver=driver)
        state = env.handle_modals()
        assert state == "menu"
        # Should have called START_GAME_JS
        assert driver.execute_script.call_count >= 2

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


# -- start_game ---------------------------------------------------------------


class TestStartGame:
    """Tests for start_game()."""

    def test_start_game_calls_js(self):
        """start_game executes START_GAME_JS."""
        driver = mock.MagicMock()
        env = _make_env(driver=driver)
        env.start_game()
        driver.execute_script.assert_called_once_with(START_GAME_JS)

    def test_start_game_no_driver(self):
        """start_game without driver is a no-op."""
        env = _make_env()
        env.start_game()  # Should not raise

    def test_start_game_js_exception_caught(self):
        """JS execution error is caught."""
        driver = mock.MagicMock()
        driver.execute_script.side_effect = Exception("JS error")
        env = _make_env(driver=driver)
        env.start_game()  # Should not raise


# -- build_info ----------------------------------------------------------------


class TestBuildInfo:
    """Tests for build_info()."""

    def test_info_contains_score(self):
        """Info dict includes score from JS bridge."""
        driver = _mock_driver(score=42)
        env = _make_env(driver=driver)
        info = env.build_info({})
        assert info["score"] == 42

    def test_info_contains_block_count(self):
        """Info dict includes block count."""
        driver = _mock_driver(block_count=15)
        env = _make_env(driver=driver)
        info = env.build_info({})
        assert info["block_count"] == 15

    def test_info_contains_game_state_raw(self):
        """Info dict includes raw gameState value."""
        driver = _mock_driver(game_state=1)
        env = _make_env(driver=driver)
        info = env.build_info({})
        assert info["game_state_raw"] == 1

    def test_info_contains_game_over_count(self):
        """Info dict includes game_over_count."""
        env = _make_env()
        env._game_over_count = 2
        info = env.build_info({})
        assert info["game_over_count"] == 2

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

    def test_accepts_playing_state(self):
        """gameState=1 (playing) is accepted."""
        driver = _mock_driver(game_state=1)
        env = _make_env(driver=driver)
        assert env.on_reset_detections({}) is True

    def test_accepts_start_screen(self):
        """gameState=0 (start screen) is accepted."""
        driver = _mock_driver(game_state=0)
        env = _make_env(driver=driver)
        assert env.on_reset_detections({}) is True

    def test_accepts_paused(self):
        """gameState=-1 (paused) is accepted."""
        driver = _mock_driver(game_state=-1)
        env = _make_env(driver=driver)
        assert env.on_reset_detections({}) is True

    def test_accepts_main_menu(self):
        """gameState=4 (main menu) is accepted."""
        driver = _mock_driver(game_state=4)
        env = _make_env(driver=driver)
        assert env.on_reset_detections({}) is True

    def test_rejects_game_over(self):
        """gameState=2 (game over) is rejected."""
        driver = _mock_driver(game_state=2)
        env = _make_env(driver=driver)
        assert env.on_reset_detections({}) is False

    def test_rejects_fade_out(self):
        """gameState=3 (fade out) is rejected."""
        driver = _mock_driver(game_state=3)
        env = _make_env(driver=driver)
        assert env.on_reset_detections({}) is False

    def test_no_driver_always_true(self):
        """Without driver, always accept."""
        env = _make_env()
        assert env.on_reset_detections({}) is True


# -- reset_termination_state --------------------------------------------------


class TestResetTerminationState:
    """Tests for reset_termination_state()."""

    def test_resets_prev_score(self):
        """_prev_score is reset to 0."""
        env = _make_env()
        env._prev_score = 500
        env.reset_termination_state()
        assert env._prev_score == 0

    def test_resets_game_over_count(self):
        """_game_over_count is reset to 0."""
        env = _make_env()
        env._game_over_count = 3
        env.reset_termination_state()
        assert env._game_over_count == 0


# -- _read_game_state ---------------------------------------------------------


class TestReadGameState:
    """Tests for _read_game_state() private helper."""

    def test_returns_js_values(self):
        """Returns values from JS bridge."""
        driver = _mock_driver(score=100, game_state=1, block_count=8)
        env = _make_env(driver=driver)
        state = env._read_game_state()
        assert state["score"] == 100
        assert state["gameState"] == 1
        assert state["blockCount"] == 8
        assert state["running"] is True

    def test_no_driver_returns_defaults(self):
        """Without driver, returns default dict."""
        env = _make_env()
        state = env._read_game_state()
        assert state["score"] == 0
        assert state["gameState"] == -99
        assert state["blockCount"] == 0
        assert state["running"] is False

    def test_js_exception_returns_defaults(self):
        """JS error returns default dict."""
        driver = mock.MagicMock()
        driver.execute_script.side_effect = Exception("JS error")
        env = _make_env(driver=driver)
        state = env._read_game_state()
        assert state["score"] == 0
        assert state["gameState"] == -99

    def test_null_result_returns_defaults(self):
        """None result from JS returns defaults."""
        driver = mock.MagicMock()
        driver.execute_script.return_value = None
        env = _make_env(driver=driver)
        state = env._read_game_state()
        assert state["score"] == 0


# -- _should_check_modals -----------------------------------------------------


class TestShouldCheckModals:
    """Tests for _should_check_modals() override."""

    def test_always_returns_true(self):
        """Hextris always checks modals (no ball-loss heuristic)."""
        env = _make_env()
        assert env._should_check_modals() is True


# -- Confirm frames constant ---------------------------------------------------


class TestConfirmFrames:
    """Tests for the _GAME_OVER_CONFIRM_FRAMES class constant."""

    def test_confirm_frames_is_3(self):
        """Game over requires 3 consecutive confirmed frames."""
        assert HextrisEnv._GAME_OVER_CONFIRM_FRAMES == 3


# -- _lazy_init (skip YOLO) ---------------------------------------------------


class TestLazyInit:
    """Tests for _lazy_init() override that skips YOLO initialization."""

    def test_lazy_init_skips_yolo_headless(self):
        """In headless mode, _lazy_init completes without YOLO detector."""
        driver = _mock_driver()
        env = _make_env(driver=driver, headless=True)

        # _lazy_init should succeed without needing YOLO weights
        env._lazy_init()

        assert env._initialized is True
        # _detector should remain None (no YOLO loaded)
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
