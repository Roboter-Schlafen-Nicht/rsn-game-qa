"""Tests for the Breakout71Env Gymnasium environment.

Covers:
- Construction and space definitions (observation, action)
- Lazy initialisation (_lazy_init)
- Frame capture delegation (_capture_frame)
- Object detection delegation (_detect_objects)
- Observation building (_build_observation) with reset, missing detections,
  velocity computation, clipping
- Reward computation (_compute_reward) with brick delta, time penalty,
  terminal rewards, score_delta placeholder
- Action application (_apply_action) via JS puckPosition injection
- Game state handling (_handle_game_state, _click_canvas)
- Oracle execution (_run_oracles)
- Full reset() lifecycle
- Full step() lifecycle with termination logic (ball lost, level cleared,
  max steps)
- Info dict contents (_build_info)
- Render and close
"""

import sys
from unittest import mock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure the ``selenium`` package is importable even in CI where it is
# not installed.  The lazy imports inside the env module use
# ``from selenium.webdriver.common.action_chains import ActionChains``
# and ``mock.patch("selenium.webdriver.common.action_chains.ActionChains")``
# requires the module path to be resolvable.
# ---------------------------------------------------------------------------
_SELENIUM_MODULES = [
    "selenium",
    "selenium.webdriver",
    "selenium.webdriver.common",
    "selenium.webdriver.common.action_chains",
]
_injected: list[str] = []
for _mod in _SELENIUM_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = mock.MagicMock()
        _injected.append(_mod)

from src.env.breakout71_env import Breakout71Env  # noqa: E402


# -- Helpers -------------------------------------------------------------------


def _detections(
    paddle=(0.5, 0.9, 0.1, 0.02),
    ball=(0.5, 0.5, 0.02, 0.02),
    bricks=None,
    powerups=None,
):
    """Build a fake detections dict matching YoloDetector.detect_to_game_state."""
    if bricks is None:
        bricks = [(0.1 * i, 0.1, 0.05, 0.03) for i in range(10)]
    return {
        "paddle": paddle,
        "ball": ball,
        "bricks": bricks,
        "powerups": powerups or [],
        "raw_detections": [],
    }


def _frame(h=480, w=640, color=(128, 128, 128)):
    """Create a synthetic BGR frame."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = color
    return frame


def _action(value=0.0):
    """Create a continuous action array for the env."""
    return np.array([value], dtype=np.float32)


def _mock_driver():
    """Create a mock Selenium WebDriver with canvas element."""
    driver = mock.MagicMock()
    canvas = mock.MagicMock()
    canvas.rect = {"x": 0, "y": 0, "width": 1280, "height": 1024}
    driver.find_element.return_value = canvas
    # Default: gameplay state (no modals)
    driver.execute_script.return_value = {"state": "gameplay", "details": {}}
    return driver, canvas


# -- Construction & Spaces -----------------------------------------------------


class TestConstruction:
    """Tests for Breakout71Env constructor and space definitions."""

    def test_env_construction(self):
        """Breakout71Env can be constructed with default args."""
        env = Breakout71Env()
        assert env.action_space.shape == (1,)
        assert env.observation_space.shape == (8,)

    def test_observation_space_shape(self):
        """Observation space should be 8-element Box."""
        env = Breakout71Env()
        assert env.observation_space.shape == (8,)

    def test_observation_space_bounds(self):
        """Observation space should have correct low/high bounds."""
        env = Breakout71Env()
        # Positions [0,1], velocities [-1,1], bricks/coins/score [0,1]
        assert env.observation_space.low[0] == 0.0  # paddle_x low
        assert env.observation_space.low[3] == -1.0  # ball_vx low
        assert env.observation_space.high[5] == 1.0  # bricks_norm high
        assert env.observation_space.low[6] == 0.0  # coins_norm low
        assert env.observation_space.high[7] == 1.0  # score_norm high

    def test_action_space_continuous_box(self):
        """Action space should be continuous Box(-1, 1, shape=(1,))."""
        env = Breakout71Env()
        assert env.action_space.shape == (1,)
        assert float(env.action_space.low[0]) == -1.0
        assert float(env.action_space.high[0]) == 1.0

    def test_custom_parameters(self):
        """Constructor should accept and store custom parameters."""
        env = Breakout71Env(
            window_title="Test",
            yolo_weights="test.pt",
            max_steps=500,
            render_mode="rgb_array",
        )
        assert env.window_title == "Test"
        assert env.max_steps == 500
        assert env.render_mode == "rgb_array"

    def test_driver_parameter(self):
        """Constructor should accept and store a Selenium driver."""
        driver = mock.MagicMock()
        env = Breakout71Env(driver=driver)
        assert env._driver is driver

    def test_initial_state(self):
        """Internal state should be properly initialised."""
        env = Breakout71Env()
        assert env._step_count == 0
        assert env._prev_ball_pos is None
        assert env._bricks_total is None
        assert env._initialized is False
        assert env._capture is None
        assert env._detector is None
        assert env._driver is None
        assert env._game_canvas is None

    def test_render_mode_rgb_array_no_frame(self):
        """render() should return None when no frame captured yet."""
        env = Breakout71Env(render_mode="rgb_array")
        assert env.render() is None

    def test_close_without_init(self):
        """close() should not raise if sub-components were never initialized."""
        env = Breakout71Env()
        env.close()  # should not raise


# -- Lazy Init -----------------------------------------------------------------


class TestLazyInit:
    """Tests for _lazy_init sub-component wiring."""

    @mock.patch("src.env.breakout71_env.Breakout71Env._lazy_init")
    def test_lazy_init_not_called_on_construction(self, mock_init):
        """_lazy_init should NOT be called during construction."""
        Breakout71Env()
        mock_init.assert_not_called()

    def test_lazy_init_sets_initialized_flag(self):
        """_lazy_init should set _initialized to True."""
        env = Breakout71Env()
        with (
            mock.patch("src.capture.window_capture.WindowCapture") as mock_cap,
            mock.patch("src.perception.yolo_detector.YoloDetector") as mock_det,
        ):
            mock_det_instance = mock.MagicMock()
            mock_det.return_value = mock_det_instance
            mock_cap_instance = mock.MagicMock()
            mock_cap.return_value = mock_cap_instance

            env._lazy_init()

            assert env._initialized is True
            assert env._capture is not None
            assert env._detector is not None

    def test_lazy_init_idempotent(self):
        """_lazy_init should only execute once (idempotent)."""
        env = Breakout71Env()
        with (
            mock.patch("src.capture.window_capture.WindowCapture") as mock_cap,
            mock.patch("src.perception.yolo_detector.YoloDetector") as mock_det,
        ):
            mock_det.return_value = mock.MagicMock()
            mock_cap.return_value = mock.MagicMock()

            env._lazy_init()
            env._lazy_init()  # second call should be a no-op

            mock_cap.assert_called_once()
            mock_det.assert_called_once()

    def test_lazy_init_loads_detector(self):
        """_lazy_init should call detector.load()."""
        env = Breakout71Env()
        with (
            mock.patch("src.capture.window_capture.WindowCapture"),
            mock.patch("src.perception.yolo_detector.YoloDetector") as mock_det,
        ):
            mock_det_instance = mock.MagicMock()
            mock_det.return_value = mock_det_instance

            env._lazy_init()

            mock_det_instance.load.assert_called_once()

    def test_lazy_init_finds_canvas_with_driver(self):
        """_lazy_init should find the game canvas when driver is provided."""
        driver, canvas = _mock_driver()
        env = Breakout71Env(driver=driver)
        with (
            mock.patch("src.capture.window_capture.WindowCapture"),
            mock.patch("src.perception.yolo_detector.YoloDetector") as mock_det,
        ):
            mock_det.return_value = mock.MagicMock()

            env._lazy_init()

            assert env._game_canvas is canvas
            assert env._canvas_dims == (0, 0, 1280, 1024)
            driver.find_element.assert_called_with("css selector", "#game")

    def test_lazy_init_falls_back_to_body(self):
        """_lazy_init should fall back to <body> if #game not found."""
        driver = mock.MagicMock()
        body_element = mock.MagicMock()
        body_element.rect = {"x": 0, "y": 0, "width": 1280, "height": 1024}
        # First call (#game) raises, second call (body) returns body_element
        driver.find_element.side_effect = [Exception("not found"), body_element]
        env = Breakout71Env(driver=driver)
        with (
            mock.patch("src.capture.window_capture.WindowCapture"),
            mock.patch("src.perception.yolo_detector.YoloDetector") as mock_det,
        ):
            mock_det.return_value = mock.MagicMock()

            env._lazy_init()

            assert env._game_canvas is body_element


# -- Capture Frame -------------------------------------------------------------


class TestCaptureFrame:
    """Tests for _capture_frame delegation."""

    def test_capture_frame_delegates(self):
        """_capture_frame should delegate to WindowCapture.capture_frame."""
        env = Breakout71Env()
        expected = _frame()
        env._capture = mock.MagicMock()
        env._capture.capture_frame.return_value = expected

        result = env._capture_frame()

        env._capture.capture_frame.assert_called_once()
        np.testing.assert_array_equal(result, expected)

    def test_capture_frame_stores_last_frame(self):
        """_capture_frame should store the frame as _last_frame."""
        env = Breakout71Env()
        expected = _frame()
        env._capture = mock.MagicMock()
        env._capture.capture_frame.return_value = expected

        env._capture_frame()

        np.testing.assert_array_equal(env._last_frame, expected)


# -- Detect Objects ------------------------------------------------------------


class TestDetectObjects:
    """Tests for _detect_objects delegation."""

    def test_detect_objects_delegates(self):
        """_detect_objects should delegate to YoloDetector.detect_to_game_state."""
        env = Breakout71Env()
        frame = _frame(480, 640)
        expected_detections = _detections()
        env._detector = mock.MagicMock()
        env._detector.detect_to_game_state.return_value = expected_detections

        result = env._detect_objects(frame)

        env._detector.detect_to_game_state.assert_called_once_with(frame, 640, 480)
        assert result == expected_detections

    def test_detect_objects_passes_correct_dimensions(self):
        """_detect_objects should pass frame width and height correctly."""
        env = Breakout71Env()
        frame = _frame(100, 200)
        env._detector = mock.MagicMock()
        env._detector.detect_to_game_state.return_value = _detections()

        env._detect_objects(frame)

        env._detector.detect_to_game_state.assert_called_once_with(frame, 200, 100)


# -- Build Observation ---------------------------------------------------------


class TestBuildObservation:
    """Tests for _build_observation with various detection scenarios."""

    def test_normal_detection(self):
        """Normal detections should produce correct observation values."""
        env = Breakout71Env()
        det = _detections(
            paddle=(0.3, 0.9, 0.1, 0.02),
            ball=(0.6, 0.4, 0.02, 0.02),
            bricks=[(0.1, 0.1, 0.05, 0.03)] * 8,
        )
        obs = env._build_observation(det, reset=True)

        assert obs[0] == 0.3  # paddle_x
        assert obs[1] == 0.6  # ball_x
        assert obs[2] == 0.4  # ball_y
        assert obs.shape == (8,)
        assert obs.dtype == np.float32

    def test_missing_paddle(self):
        """Missing paddle detection should default to 0.5."""
        env = Breakout71Env()
        det = _detections(paddle=None)
        obs = env._build_observation(det, reset=True)

        assert obs[0] == 0.5  # default paddle_x

    def test_missing_ball(self):
        """Missing ball detection should default to 0.5 for position."""
        env = Breakout71Env()
        det = _detections(ball=None)
        obs = env._build_observation(det, reset=True)

        assert obs[1] == 0.5  # default ball_x
        assert obs[2] == 0.5  # default ball_y

    def test_missing_both(self):
        """Missing paddle and ball should both default to 0.5."""
        env = Breakout71Env()
        det = _detections(paddle=None, ball=None)
        obs = env._build_observation(det, reset=True)

        assert obs[0] == 0.5  # paddle_x
        assert obs[1] == 0.5  # ball_x
        assert obs[2] == 0.5  # ball_y

    def test_reset_zeroes_velocity(self):
        """On reset, velocities should be zero regardless of prev position."""
        env = Breakout71Env()
        env._prev_ball_pos = (0.3, 0.3)  # simulate prior state

        det = _detections(ball=(0.6, 0.4, 0.02, 0.02))
        obs = env._build_observation(det, reset=True)

        assert obs[3] == 0.0  # ball_vx
        assert obs[4] == 0.0  # ball_vy

    def test_reset_sets_bricks_total(self):
        """On reset, _bricks_total should be set from current brick count."""
        env = Breakout71Env()
        bricks = [(0.1 * i, 0.1, 0.05, 0.03) for i in range(15)]
        det = _detections(bricks=bricks)
        env._build_observation(det, reset=True)

        assert env._bricks_total == 15

    def test_reset_bricks_total_minimum_1(self):
        """On reset with no bricks, _bricks_total should be at least 1."""
        env = Breakout71Env()
        det = _detections(bricks=[])
        env._build_observation(det, reset=True)

        assert env._bricks_total == 1

    def test_velocity_computation(self):
        """Velocity should be computed from consecutive ball positions."""
        env = Breakout71Env()
        # First frame (reset)
        det1 = _detections(ball=(0.5, 0.5, 0.02, 0.02))
        env._build_observation(det1, reset=True)

        # Second frame
        det2 = _detections(ball=(0.6, 0.45, 0.02, 0.02))
        obs = env._build_observation(det2)

        assert abs(obs[3] - 0.1) < 1e-6  # ball_vx = 0.6 - 0.5
        assert abs(obs[4] - (-0.05)) < 1e-6  # ball_vy = 0.45 - 0.5

    def test_velocity_clipping(self):
        """Velocities exceeding [-1, 1] should be clipped."""
        env = Breakout71Env()
        det1 = _detections(ball=(0.0, 0.0, 0.02, 0.02))
        env._build_observation(det1, reset=True)

        # Large jump -- would give velocity > 1.0
        det2 = _detections(ball=(1.0, 1.0, 0.02, 0.02))
        # Override prev_ball_pos to create extreme delta
        env._prev_ball_pos = (-1.0, -1.0)
        obs = env._build_observation(det2)

        assert obs[3] == 1.0  # clipped to 1.0
        assert obs[4] == 1.0  # clipped to 1.0

    def test_bricks_norm_fraction(self):
        """bricks_norm should be fraction of remaining bricks."""
        env = Breakout71Env()
        bricks_initial = [(0.1 * i, 0.1, 0.05, 0.03) for i in range(10)]
        det1 = _detections(bricks=bricks_initial)
        env._build_observation(det1, reset=True)

        # Simulate 3 bricks destroyed
        bricks_remaining = [(0.1 * i, 0.1, 0.05, 0.03) for i in range(7)]
        det2 = _detections(bricks=bricks_remaining)
        obs = env._build_observation(det2)

        assert abs(obs[5] - 0.7) < 1e-6  # 7/10

    def test_placeholder_slots_are_zero(self):
        """coins_norm and score_norm should be 0.0 in v1."""
        env = Breakout71Env()
        det = _detections()
        obs = env._build_observation(det, reset=True)

        assert obs[6] == 0.0  # coins_norm
        assert obs[7] == 0.0  # score_norm

    def test_observation_in_bounds(self):
        """Observation should always be within the observation space bounds."""
        env = Breakout71Env()
        det = _detections(
            paddle=(0.8, 0.9, 0.1, 0.02),
            ball=(0.2, 0.3, 0.02, 0.02),
        )
        obs = env._build_observation(det, reset=True)

        assert env.observation_space.contains(obs)


# -- Compute Reward ------------------------------------------------------------


class TestComputeReward:
    """Tests for _compute_reward with various scenarios."""

    def test_brick_destruction_reward(self):
        """Destroying bricks should give positive reward proportional to delta."""
        env = Breakout71Env()
        env._bricks_total = 10
        env._prev_bricks_norm = 1.0

        # 8 bricks remaining (2 destroyed)
        det = _detections(bricks=[(0.1 * i, 0.1, 0.05, 0.03) for i in range(8)])
        reward = env._compute_reward(det, terminated=False, level_cleared=False)

        # brick_delta = 1.0 - 0.8 = 0.2; reward = 0.2 * 10 - 0.01 = 1.99
        assert abs(reward - 1.99) < 1e-6

    def test_time_penalty_only(self):
        """No brick change should give only time penalty."""
        env = Breakout71Env()
        env._bricks_total = 10
        env._prev_bricks_norm = 0.5

        det = _detections(bricks=[(0.1 * i, 0.1, 0.05, 0.03) for i in range(5)])
        reward = env._compute_reward(det, terminated=False, level_cleared=False)

        assert abs(reward - (-0.01)) < 1e-6

    def test_game_over_penalty(self):
        """Game over should give -5.0 penalty on top of time penalty."""
        env = Breakout71Env()
        env._bricks_total = 10
        env._prev_bricks_norm = 0.5

        det = _detections(bricks=[(0.1 * i, 0.1, 0.05, 0.03) for i in range(5)])
        reward = env._compute_reward(det, terminated=True, level_cleared=False)

        # time penalty (-0.01) + game_over (-5.0) = -5.01
        assert abs(reward - (-5.01)) < 1e-6

    def test_level_cleared_bonus(self):
        """Level cleared should give +5.0 bonus."""
        env = Breakout71Env()
        env._bricks_total = 10
        env._prev_bricks_norm = 0.0  # already at 0

        det = _detections(bricks=[])
        reward = env._compute_reward(det, terminated=True, level_cleared=True)

        # brick_delta = 0.0; time penalty (-0.01) + level_clear (+5.0) = 4.99
        assert abs(reward - 4.99) < 1e-6

    def test_combined_brick_and_level_clear(self):
        """Last brick destroyed + level cleared should give both rewards."""
        env = Breakout71Env()
        env._bricks_total = 10
        env._prev_bricks_norm = 0.1  # 1 brick left

        det = _detections(bricks=[])
        reward = env._compute_reward(det, terminated=True, level_cleared=True)

        # brick_delta = 0.1 * 10 = 1.0; + time (-0.01) + level (+5.0) = 5.99
        assert abs(reward - 5.99) < 1e-6

    def test_score_delta_placeholder(self):
        """Score delta should be 0.0 in v1 (no effect on reward)."""
        env = Breakout71Env()
        env._bricks_total = 10
        env._prev_bricks_norm = 1.0

        det = _detections(bricks=[(0.1 * i, 0.1, 0.05, 0.03) for i in range(10)])
        reward = env._compute_reward(det, terminated=False, level_cleared=False)

        # Only time penalty since no bricks changed
        assert abs(reward - (-0.01)) < 1e-6

    def test_prev_bricks_norm_updated(self):
        """_prev_bricks_norm should be updated after reward computation."""
        env = Breakout71Env()
        env._bricks_total = 10
        env._prev_bricks_norm = 1.0

        det = _detections(bricks=[(0.1 * i, 0.1, 0.05, 0.03) for i in range(7)])
        env._compute_reward(det, terminated=False, level_cleared=False)

        assert abs(env._prev_bricks_norm - 0.7) < 1e-6


# -- Apply Action --------------------------------------------------------------


class TestApplyAction:
    """Tests for _apply_action via JS puckPosition injection."""

    def test_centre_action_sets_puck_position(self):
        """Action 0.0 (centre) should set puckPosition to zone midpoint."""
        driver, canvas = _mock_driver()
        env = Breakout71Env(driver=driver)
        env._game_canvas = canvas
        env._canvas_dims = (0, 0, 1280, 1024)
        env._game_zone_left = 320.0
        env._game_zone_right = 960.0

        env._apply_action(np.array([0.0], dtype=np.float32))

        driver.execute_script.assert_called_once()
        call_args = driver.execute_script.call_args
        pixel_x = call_args[0][1]
        assert abs(pixel_x - 640.0) < 1e-6  # midpoint

    def test_left_edge_action(self):
        """Action -1.0 should set puckPosition to game zone left edge."""
        driver, canvas = _mock_driver()
        env = Breakout71Env(driver=driver)
        env._game_canvas = canvas
        env._canvas_dims = (0, 0, 1280, 1024)
        env._game_zone_left = 320.0
        env._game_zone_right = 960.0

        env._apply_action(np.array([-1.0], dtype=np.float32))

        call_args = driver.execute_script.call_args
        pixel_x = call_args[0][1]
        assert abs(pixel_x - 320.0) < 1e-6

    def test_right_edge_action(self):
        """Action +1.0 should set puckPosition to game zone right edge."""
        driver, canvas = _mock_driver()
        env = Breakout71Env(driver=driver)
        env._game_canvas = canvas
        env._canvas_dims = (0, 0, 1280, 1024)
        env._game_zone_left = 320.0
        env._game_zone_right = 960.0

        env._apply_action(np.array([1.0], dtype=np.float32))

        call_args = driver.execute_script.call_args
        pixel_x = call_args[0][1]
        assert abs(pixel_x - 960.0) < 1e-6

    def test_action_clamped_to_bounds(self):
        """Action values outside [-1, 1] should be clipped."""
        driver, canvas = _mock_driver()
        env = Breakout71Env(driver=driver)
        env._game_canvas = canvas
        env._canvas_dims = (0, 0, 1280, 1024)
        env._game_zone_left = 320.0
        env._game_zone_right = 960.0

        env._apply_action(np.array([2.0], dtype=np.float32))

        call_args = driver.execute_script.call_args
        pixel_x = call_args[0][1]
        # Clipped to +1.0 -> right edge
        assert abs(pixel_x - 960.0) < 1e-6

    def test_fallback_without_game_zone(self):
        """Without game zone boundaries, should use canvas-based fallback."""
        driver, canvas = _mock_driver()
        env = Breakout71Env(driver=driver)
        env._game_canvas = canvas
        env._canvas_dims = (0, 0, 1280, 1024)
        # _game_zone_left/right are None (not queried)

        env._apply_action(np.array([0.0], dtype=np.float32))

        # Fallback: 25%-75% of canvas = 320-960, centre = 640
        call_args = driver.execute_script.call_args
        pixel_x = call_args[0][1]
        assert abs(pixel_x - 640.0) < 1e-6

    def test_no_driver_is_noop(self):
        """Without a driver, _apply_action should be a no-op."""
        env = Breakout71Env()  # no driver

        # Should not raise
        env._apply_action(np.array([0.5], dtype=np.float32))
        env._apply_action(np.array([-0.5], dtype=np.float32))

    def test_quarter_position(self):
        """Action -0.5 should map to 25% between left and right edges."""
        driver, canvas = _mock_driver()
        env = Breakout71Env(driver=driver)
        env._game_canvas = canvas
        env._canvas_dims = (0, 0, 1280, 1024)
        env._game_zone_left = 320.0
        env._game_zone_right = 960.0

        env._apply_action(np.array([-0.5], dtype=np.float32))

        call_args = driver.execute_script.call_args
        pixel_x = call_args[0][1]
        # -0.5 -> (−0.5 + 1) / 2 = 0.25 -> 320 + 0.25 * 640 = 480
        assert abs(pixel_x - 480.0) < 1e-6

    def test_action_scalar_input(self):
        """Scalar action (0-d array or float) is normalised to (1,)."""
        driver, canvas = _mock_driver()
        env = Breakout71Env(driver=driver)
        env._game_zone_left = 320.0
        env._game_zone_right = 960.0

        # 0-d numpy array
        env._apply_action(np.float32(0.0))
        pixel_x = driver.execute_script.call_args[0][1]
        assert abs(pixel_x - 640.0) < 1e-6

        # Plain Python float
        driver.execute_script.reset_mock()
        env._apply_action(0.0)
        pixel_x = driver.execute_script.call_args[0][1]
        assert abs(pixel_x - 640.0) < 1e-6

    def test_action_wrong_size_raises(self):
        """Action with size != 1 should raise ValueError."""
        driver, canvas = _mock_driver()
        env = Breakout71Env(driver=driver)
        env._game_zone_left = 320.0
        env._game_zone_right = 960.0

        with pytest.raises(ValueError, match="size 1"):
            env._apply_action(np.array([0.1, 0.2], dtype=np.float32))


# -- Game State Handling -------------------------------------------------------


class TestHandleGameState:
    """Tests for _handle_game_state and _click_canvas."""

    def test_gameplay_state_no_action(self):
        """Gameplay state should return 'gameplay' without JS calls."""
        driver, canvas = _mock_driver()
        driver.execute_script.return_value = {
            "state": "gameplay",
            "details": {},
        }
        env = Breakout71Env(driver=driver)

        state = env._handle_game_state()

        assert state == "gameplay"
        # Only the detect call, no dismiss calls
        driver.execute_script.assert_called_once()

    @mock.patch("src.env.breakout71_env.time")
    def test_game_over_dismissed(self, mock_time):
        """Game over state should trigger dismiss JS execution."""
        driver, canvas = _mock_driver()
        driver.execute_script.side_effect = [
            {"state": "game_over", "details": {}},
            {"action": "restart_button", "text": "New Run"},
        ]
        env = Breakout71Env(driver=driver)

        state = env._handle_game_state()

        assert state == "game_over"
        assert driver.execute_script.call_count == 2

    @mock.patch("src.env.breakout71_env.time")
    def test_perk_picker_clicks_perk(self, mock_time):
        """Perk picker state should click a random perk button."""
        driver, canvas = _mock_driver()
        driver.execute_script.side_effect = [
            {"state": "perk_picker", "details": {"numPerks": 3}},
            {"clicked": 1, "text": "Extra Life"},
        ]
        env = Breakout71Env(driver=driver)

        state = env._handle_game_state()

        assert state == "perk_picker"
        assert driver.execute_script.call_count == 2

    @mock.patch("src.env.breakout71_env.time")
    def test_menu_dismissed(self, mock_time):
        """Menu state should trigger dismiss."""
        driver, canvas = _mock_driver()
        driver.execute_script.side_effect = [
            {"state": "menu", "details": {}},
            {"action": "close_button"},
        ]
        env = Breakout71Env(driver=driver)

        state = env._handle_game_state()

        assert state == "menu"

    def test_no_driver_returns_gameplay(self):
        """Without a driver, _handle_game_state returns 'gameplay'."""
        env = Breakout71Env()

        assert env._handle_game_state() == "gameplay"

    def test_click_canvas(self):
        """_click_canvas should click the game canvas element."""
        driver, canvas = _mock_driver()
        env = Breakout71Env(driver=driver)
        env._game_canvas = canvas

        with mock.patch(
            "selenium.webdriver.common.action_chains.ActionChains"
        ) as mock_ac_cls:
            mock_chain = mock.MagicMock()
            mock_ac_cls.return_value = mock_chain
            mock_chain.move_to_element.return_value = mock_chain
            mock_chain.click.return_value = mock_chain

            env._click_canvas()

            mock_chain.move_to_element.assert_called_once_with(canvas)
            mock_chain.click.assert_called_once()
            mock_chain.perform.assert_called_once()

    def test_click_canvas_no_driver(self):
        """_click_canvas without driver should be a no-op."""
        env = Breakout71Env()
        env._click_canvas()  # should not raise


# -- Run Oracles ---------------------------------------------------------------


class TestRunOracles:
    """Tests for _run_oracles oracle execution."""

    def test_calls_on_step_on_all_oracles(self):
        """_run_oracles should call on_step on every attached oracle."""
        env = Breakout71Env()
        oracle1 = mock.MagicMock()
        oracle1.get_findings.return_value = []
        oracle2 = mock.MagicMock()
        oracle2.get_findings.return_value = []
        env._oracles = [oracle1, oracle2]

        obs = np.zeros(8, dtype=np.float32)
        info = {"frame": None}
        env._run_oracles(obs, 0.0, False, False, info)

        oracle1.on_step.assert_called_once_with(obs, 0.0, False, False, info)
        oracle2.on_step.assert_called_once_with(obs, 0.0, False, False, info)

    def test_collects_findings(self):
        """_run_oracles should aggregate findings from all oracles."""
        env = Breakout71Env()
        finding1 = mock.MagicMock()
        finding2 = mock.MagicMock()
        oracle1 = mock.MagicMock()
        oracle1.get_findings.return_value = [finding1]
        oracle2 = mock.MagicMock()
        oracle2.get_findings.return_value = [finding2]
        env._oracles = [oracle1, oracle2]

        obs = np.zeros(8, dtype=np.float32)
        findings = env._run_oracles(obs, 0.0, False, False, {})

        assert len(findings) == 2
        assert finding1 in findings
        assert finding2 in findings

    def test_empty_oracle_list(self):
        """_run_oracles with no oracles should return empty list."""
        env = Breakout71Env()
        env._oracles = []

        obs = np.zeros(8, dtype=np.float32)
        findings = env._run_oracles(obs, 0.0, False, False, {})

        assert findings == []

    def test_oracle_with_no_findings(self):
        """Oracle that produces no findings should contribute empty list."""
        env = Breakout71Env()
        oracle = mock.MagicMock()
        oracle.get_findings.return_value = []
        env._oracles = [oracle]

        obs = np.zeros(8, dtype=np.float32)
        findings = env._run_oracles(obs, 0.0, False, False, {})

        assert findings == []


# -- Build Info ----------------------------------------------------------------


class TestBuildInfo:
    """Tests for _build_info helper."""

    def test_info_contains_expected_keys(self):
        """Info dict should contain all expected keys."""
        env = Breakout71Env()
        env._last_frame = _frame()
        env._step_count = 5
        det = _detections()

        info = env._build_info(det)

        assert "frame" in info
        assert "detections" in info
        assert "ball_pos" in info
        assert "paddle_pos" in info
        assert "brick_count" in info
        assert "step" in info
        assert "score" in info

    def test_info_ball_pos_from_detections(self):
        """Info ball_pos should come from detections."""
        env = Breakout71Env()
        env._last_frame = _frame()
        det = _detections(ball=(0.3, 0.7, 0.02, 0.02))

        info = env._build_info(det)

        assert info["ball_pos"] == [0.3, 0.7]

    def test_info_missing_ball(self):
        """Info ball_pos should be None when ball not detected."""
        env = Breakout71Env()
        env._last_frame = _frame()
        det = _detections(ball=None)

        info = env._build_info(det)

        assert info["ball_pos"] is None

    def test_info_brick_count(self):
        """Info brick_count should match detection count."""
        env = Breakout71Env()
        env._last_frame = _frame()
        bricks = [(0.1 * i, 0.1, 0.05, 0.03) for i in range(7)]
        det = _detections(bricks=bricks)

        info = env._build_info(det)

        assert info["brick_count"] == 7


# -- Reset ---------------------------------------------------------------------


class TestReset:
    """Tests for the full reset() lifecycle."""

    def _make_env_with_mocks(self):
        """Create env with mocked sub-components."""
        driver, canvas = _mock_driver()
        env = Breakout71Env(driver=driver)
        env._initialized = True
        env._capture = mock.MagicMock()
        env._capture.capture_frame.return_value = _frame()
        env._detector = mock.MagicMock()
        env._detector.detect_to_game_state.return_value = _detections()
        env._game_canvas = canvas
        env._canvas_dims = (0, 0, 1280, 1024)
        return env

    @mock.patch("src.env.breakout71_env.time")
    def test_reset_returns_obs_and_info(self, mock_time):
        """reset() should return (obs, info) tuple."""
        env = self._make_env_with_mocks()

        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (8,)
        assert isinstance(info, dict)

    @mock.patch("src.env.breakout71_env.time")
    def test_reset_handles_game_state(self, mock_time):
        """reset() should call _handle_game_state to dismiss modals."""
        env = self._make_env_with_mocks()

        env.reset()

        # Driver.execute_script is called at least once for state detection
        env._driver.execute_script.assert_called()

    @mock.patch("src.env.breakout71_env.time")
    def test_reset_resets_counters(self, mock_time):
        """reset() should reset all episode counters."""
        env = self._make_env_with_mocks()
        env._step_count = 100
        env._no_ball_count = 3
        env._no_bricks_count = 2
        env._prev_bricks_norm = 0.3

        env.reset()

        assert env._step_count == 0
        assert env._no_ball_count == 0
        assert env._no_bricks_count == 0
        assert env._prev_bricks_norm == 1.0

    @mock.patch("src.env.breakout71_env.time")
    def test_reset_clears_oracles(self, mock_time):
        """reset() should clear and call on_reset on all oracles."""
        env = self._make_env_with_mocks()
        oracle = mock.MagicMock()
        env._oracles = [oracle]

        env.reset()

        oracle.clear.assert_called_once()
        oracle.on_reset.assert_called_once()

    @mock.patch("src.env.breakout71_env.time")
    def test_reset_calls_lazy_init_when_not_initialized(self, mock_time):
        """reset() should call _lazy_init on first call."""
        driver, canvas = _mock_driver()
        env = Breakout71Env(driver=driver)
        env._lazy_init = mock.MagicMock()
        # After lazy_init, we need the sub-components set up
        env._capture = mock.MagicMock()
        env._capture.capture_frame.return_value = _frame()
        env._detector = mock.MagicMock()
        env._detector.detect_to_game_state.return_value = _detections()
        env._game_canvas = canvas
        env._canvas_dims = (0, 0, 1280, 1024)

        env.reset()

        env._lazy_init.assert_called_once()

    @mock.patch("src.env.breakout71_env.time")
    def test_reset_raises_after_ball_retry_exhausted(self, mock_time):
        """reset() raises RuntimeError if ball never detected after 5 retries."""
        driver, canvas = _mock_driver()
        env = Breakout71Env(driver=driver)
        env._initialized = True
        env._capture = mock.MagicMock()
        env._capture.capture_frame.return_value = _frame()
        env._detector = mock.MagicMock()
        env._detector.detect_to_game_state.return_value = _detections(ball=None)
        env._game_canvas = canvas
        env._canvas_dims = (0, 0, 1280, 1024)

        with pytest.raises(RuntimeError, match="failed to detect a ball"):
            env.reset()


# -- Step ----------------------------------------------------------------------


class TestStep:
    """Tests for the full step() lifecycle."""

    def _make_env_ready(self, bricks_count=10):
        """Create env in a post-reset state with mocked sub-components."""
        driver, canvas = _mock_driver()
        env = Breakout71Env(driver=driver)
        env._initialized = True
        env._bricks_total = bricks_count
        env._prev_bricks_norm = 1.0
        env._prev_ball_pos = (0.5, 0.5)
        env._step_count = 0
        env._no_ball_count = 0
        env._no_bricks_count = 0
        env._last_frame = _frame()
        env._game_canvas = canvas
        env._canvas_dims = (0, 0, 1280, 1024)

        env._capture = mock.MagicMock()
        env._capture.capture_frame.return_value = _frame()
        env._detector = mock.MagicMock()
        env._detector.detect_to_game_state.return_value = _detections(
            bricks=[(0.1 * i, 0.1, 0.05, 0.03) for i in range(bricks_count)]
        )
        return env

    @mock.patch("src.env.breakout71_env.time")
    def test_step_returns_5_tuple(self, mock_time):
        """step() should return (obs, reward, terminated, truncated, info)."""
        env = self._make_env_ready()

        result = env.step(_action())

        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    @mock.patch("src.env.breakout71_env.time")
    def test_step_increments_counter(self, mock_time):
        """step() should increment _step_count."""
        env = self._make_env_ready()
        assert env._step_count == 0

        env.step(_action())

        assert env._step_count == 1

    @mock.patch("src.env.breakout71_env.time")
    def test_step_applies_action(self, mock_time):
        """step() should set puckPosition via JS execution."""
        env = self._make_env_ready()
        env._game_zone_left = 320.0
        env._game_zone_right = 960.0

        env.step(_action(0.5))  # Move right of centre

        # execute_script called for puckPosition (and game state)
        assert env._driver.execute_script.call_count >= 1

    @mock.patch("src.env.breakout71_env.time")
    def test_step_normal_not_terminated(self, mock_time):
        """Normal step with ball and bricks should not terminate."""
        env = self._make_env_ready()

        _, _, terminated, truncated, _ = env.step(_action())

        assert terminated is False
        assert truncated is False

    @mock.patch("src.env.breakout71_env.time")
    def test_step_truncation_at_max_steps(self, mock_time):
        """step() should set truncated=True when max_steps is reached."""
        env = self._make_env_ready()
        env.max_steps = 5
        env._step_count = 4  # will become 5 after increment -> equals max_steps

        _, _, terminated, truncated, _ = env.step(_action())

        assert truncated is True

    @mock.patch("src.env.breakout71_env.time")
    def test_step_ball_lost_game_over(self, mock_time):
        """Game over when ball not detected for BALL_LOST_THRESHOLD frames."""
        env = self._make_env_ready()
        # Simulate ball already missing for threshold-1 frames
        env._no_ball_count = Breakout71Env._BALL_LOST_THRESHOLD - 1

        # Return detections with no ball
        env._detector.detect_to_game_state.return_value = _detections(
            ball=None,
            bricks=[(0.1 * i, 0.1, 0.05, 0.03) for i in range(10)],
        )

        _, _, terminated, _, _ = env.step(_action())

        assert terminated is True

    @mock.patch("src.env.breakout71_env.time")
    def test_step_ball_found_resets_counter(self, mock_time):
        """Finding ball should reset _no_ball_count to 0."""
        env = self._make_env_ready()
        env._no_ball_count = 3

        env.step(_action())  # default detections have ball

        assert env._no_ball_count == 0

    @mock.patch("src.env.breakout71_env.time")
    def test_step_level_cleared(self, mock_time):
        """Level cleared when no bricks for LEVEL_CLEAR_THRESHOLD frames."""
        env = self._make_env_ready()
        env._no_bricks_count = Breakout71Env._LEVEL_CLEAR_THRESHOLD - 1

        # Return detections with no bricks
        env._detector.detect_to_game_state.return_value = _detections(bricks=[])

        _, _, terminated, _, _ = env.step(_action())

        assert terminated is True

    @mock.patch("src.env.breakout71_env.time")
    def test_step_bricks_present_resets_counter(self, mock_time):
        """Having bricks should reset _no_bricks_count to 0."""
        env = self._make_env_ready()
        env._no_bricks_count = 2

        env.step(_action())  # default detections have bricks

        assert env._no_bricks_count == 0

    @mock.patch("src.env.breakout71_env.time")
    def test_step_info_has_oracle_findings(self, mock_time):
        """step() info dict should include oracle_findings."""
        env = self._make_env_ready()

        _, _, _, _, info = env.step(_action())

        assert "oracle_findings" in info

    @mock.patch("src.env.breakout71_env.time")
    def test_step_runs_oracles(self, mock_time):
        """step() should call on_step on all attached oracles."""
        env = self._make_env_ready()
        oracle = mock.MagicMock()
        oracle.get_findings.return_value = []
        env._oracles = [oracle]

        env.step(_action())

        oracle.on_step.assert_called_once()

    @mock.patch("src.env.breakout71_env.time")
    def test_step_observation_in_bounds(self, mock_time):
        """step() observation should be within observation_space."""
        env = self._make_env_ready()

        obs, _, _, _, _ = env.step(_action())

        assert env.observation_space.contains(obs)


# -- Render & Close ------------------------------------------------------------


class TestRenderClose:
    """Tests for render() and close()."""

    def test_render_rgb_array(self):
        """render() should return last_frame when render_mode is rgb_array."""
        env = Breakout71Env(render_mode="rgb_array")
        frame = _frame()
        env._last_frame = frame

        result = env.render()

        np.testing.assert_array_equal(result, frame)

    def test_render_no_mode(self):
        """render() should return None when render_mode is not set."""
        env = Breakout71Env()
        env._last_frame = _frame()

        assert env.render() is None

    def test_close_releases_capture(self):
        """close() should call release on capture if initialised."""
        env = Breakout71Env()
        mock_capture = mock.MagicMock()
        env._capture = mock_capture
        env._detector = mock.MagicMock()

        env.close()

        mock_capture.release.assert_called_once()
        assert env._capture is None
        assert env._detector is None

    def test_close_safe_without_init(self):
        """close() should not raise if nothing was initialised."""
        env = Breakout71Env()
        env.close()  # should not raise

    def test_close_does_not_close_driver(self):
        """close() should NOT close the Selenium driver (caller owns it)."""
        driver, canvas = _mock_driver()
        env = Breakout71Env(driver=driver)
        env._game_canvas = canvas

        env.close()

        driver.quit.assert_not_called()
        # Driver reference is not cleared -- caller still owns it
        assert env._driver is driver

    def test_close_resets_initialized(self):
        """close() should reset _initialized flag."""
        env = Breakout71Env()
        env._initialized = True
        env._capture = mock.MagicMock()

        env.close()

        assert env._initialized is False


# -- Prev Brick Count ---------------------------------------------------------


class TestPrevBrickCount:
    """Tests for _prev_brick_count helper."""

    def test_with_total(self):
        """Should compute previous brick count from norm and total."""
        env = Breakout71Env()
        env._bricks_total = 10
        env._prev_bricks_norm = 0.7

        assert env._prev_brick_count() == 7

    def test_without_total(self):
        """Should return 0 when _bricks_total is None."""
        env = Breakout71Env()
        env._bricks_total = None

        assert env._prev_brick_count() == 0
