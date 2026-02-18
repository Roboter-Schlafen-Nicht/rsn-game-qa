"""Tests for the Breakout71Env Gymnasium environment.

Covers:
- Construction and space definitions (observation, action)
- Lazy initialisation (_lazy_init) with WinCamCapture/WindowCapture fallback
- Canvas element lookup (_init_canvas) for ActionChains input
- Frame capture delegation (_capture_frame)
- Object detection delegation (_detect_objects)
- Observation building (build_observation) with reset, missing detections,
  velocity computation, clipping
- Reward computation (compute_reward) with brick delta, time penalty,
  terminal rewards, score_delta placeholder
- Action application (apply_action) via Selenium ActionChains
- Game state handling (handle_modals, start_game)
- Oracle execution (_run_oracles)
- Full reset() lifecycle
- Full step() lifecycle with termination logic (ball lost, level cleared,
  max steps)
- Info dict contents (build_info)
- Render and close
"""

import sys
from unittest import mock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure the ``selenium`` package is importable even in CI where it is
# not installed.  The env module uses Selenium for modal handling only,
# via ``driver.execute_script()``.
# ---------------------------------------------------------------------------
_SELENIUM_MODULES = [
    "selenium",
    "selenium.webdriver",
    "selenium.webdriver.common",
    "selenium.webdriver.common.action_chains",
    "selenium.webdriver.common.by",
]
_injected: list[str] = []
for _mod in _SELENIUM_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = mock.MagicMock()
        _injected.append(_mod)

from games.breakout71.env import (  # noqa: E402
    Breakout71Env,
)
from games.breakout71.modal_handler import (  # noqa: E402
    DISMISS_GAME_OVER_JS,
    READ_GAME_STATE_JS,
)


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
    """Create a mock Selenium WebDriver for modal handling.

    Returns a mock driver with ``execute_script`` set to return
    a gameplay state by default.
    """
    driver = mock.MagicMock()
    # Default: gameplay state (no modals)
    driver.execute_script.return_value = {"state": "gameplay", "details": {}}
    return driver


def _setup_capture_mock(env, *, width=640, height=480, hwnd=12345):
    """Set up a mock capture subsystem on the env."""
    env._capture = mock.MagicMock()
    env._capture.width = width
    env._capture.height = height
    env._capture.hwnd = hwnd
    env._capture.capture_frame.return_value = _frame(height, width)
    return env._capture


def _make_env_ready(bricks_count=10):
    """Create a Breakout71Env in a post-reset state with mocked sub-components.

    Shared helper used by TestStep, TestModalCheckThrottling, and any
    other test class that needs a fully-wired env for step() testing.
    """
    driver = _mock_driver()
    env = Breakout71Env(driver=driver)
    env._initialized = True
    env._bricks_total = bricks_count
    env._prev_bricks_norm = 1.0
    env._prev_ball_pos = (0.5, 0.5)
    env._step_count = 0
    env._no_ball_count = 0
    env._no_bricks_count = 0
    env._last_frame = _frame()

    _setup_capture_mock(env)
    env._game_canvas = mock.MagicMock()
    env._canvas_size = (640, 480)
    env._detector = mock.MagicMock()
    env._detector.detect_to_game_state.return_value = _detections(
        bricks=[(0.1 * i, 0.1, 0.05, 0.03) for i in range(bricks_count)]
    )
    return env


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

    def test_device_parameter(self):
        """Constructor should accept and store device parameter."""
        env = Breakout71Env(device="cpu")
        assert env.device == "cpu"

    def test_device_default_auto(self):
        """Device should default to 'auto'."""
        env = Breakout71Env()
        assert env.device == "auto"

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

    def test_render_mode_rgb_array_no_frame(self):
        """render() should return None when no frame captured yet."""
        env = Breakout71Env(render_mode="rgb_array")
        assert env.render() is None

    def test_close_without_init(self):
        """close() should not raise if sub-components were never initialized."""
        env = Breakout71Env()
        env.close()  # should not raise


# -- Init Canvas ---------------------------------------------------------------


class TestInitCanvas:
    """Tests for _init_canvas helper method."""

    def test_finds_game_canvas(self):
        """_init_canvas should find the #game canvas element."""
        driver = _mock_driver()
        canvas_el = mock.MagicMock()
        canvas_el.size = {"width": 1280, "height": 1024}
        driver.find_element.return_value = canvas_el

        env = Breakout71Env(driver=driver)
        env._init_canvas()

        assert env._game_canvas is canvas_el
        assert env._canvas_size == (1280, 1024)

    def test_falls_back_to_body(self):
        """_init_canvas should fall back to <body> if #game not found."""
        driver = _mock_driver()
        body_el = mock.MagicMock()
        body_el.size = {"width": 800, "height": 600}
        driver.find_element.side_effect = [Exception("not found"), body_el]

        env = Breakout71Env(driver=driver)
        env._init_canvas()

        assert env._game_canvas is body_el
        assert env._canvas_size == (800, 600)

    def test_no_driver_skips(self):
        """_init_canvas without driver should do nothing."""
        env = Breakout71Env()
        env._init_canvas()

        assert env._game_canvas is None
        assert env._canvas_size is None

    def test_both_elements_missing(self):
        """_init_canvas should handle both #game and <body> missing."""
        driver = _mock_driver()
        driver.find_element.side_effect = Exception("nothing found")

        env = Breakout71Env(driver=driver)
        env._init_canvas()

        assert env._game_canvas is None
        assert env._canvas_size is None


# -- Lazy Init -----------------------------------------------------------------


class TestLazyInit:
    """Tests for _lazy_init sub-component wiring."""

    @mock.patch("games.breakout71.env.Breakout71Env._lazy_init")
    def test_lazy_init_not_called_on_construction(self, mock_init):
        """_lazy_init should NOT be called during construction."""
        Breakout71Env()
        mock_init.assert_not_called()

    @mock.patch("src.perception.yolo_detector.YoloDetector")
    @mock.patch("src.capture.wincam_capture.WinCamCapture")
    def test_lazy_init_prefers_wincam(self, mock_wincam, mock_det):
        """_lazy_init should prefer WinCamCapture when available."""
        mock_det_instance = mock.MagicMock()
        mock_det.return_value = mock_det_instance
        mock_wincam_instance = mock.MagicMock()
        mock_wincam.return_value = mock_wincam_instance
        mock_wincam_instance.hwnd = 12345
        mock_wincam_instance.width = 1280
        mock_wincam_instance.height = 1024

        env = Breakout71Env()
        env._lazy_init()

        assert env._initialized is True
        assert env._capture is mock_wincam_instance
        assert env._detector is not None
        mock_wincam.assert_called_once()

    @mock.patch("src.perception.yolo_detector.YoloDetector")
    @mock.patch(
        "src.capture.wincam_capture.WinCamCapture",
        side_effect=ImportError("no wincam"),
    )
    @mock.patch("src.capture.window_capture.WindowCapture")
    def test_lazy_init_falls_back_to_window_capture(
        self, mock_wc, mock_wincam, mock_det
    ):
        """_lazy_init should fall back to WindowCapture when WinCamCapture fails."""
        mock_det.return_value = mock.MagicMock()
        mock_wc_instance = mock.MagicMock()
        mock_wc.return_value = mock_wc_instance

        env = Breakout71Env()
        env._lazy_init()

        assert env._initialized is True
        assert env._capture is mock_wc_instance
        mock_wc.assert_called_once()

    @mock.patch("src.perception.yolo_detector.YoloDetector")
    @mock.patch("src.capture.wincam_capture.WinCamCapture")
    def test_lazy_init_idempotent(self, mock_wincam, mock_det):
        """_lazy_init should only execute once (idempotent)."""
        mock_det.return_value = mock.MagicMock()
        mock_wincam.return_value = mock.MagicMock()

        env = Breakout71Env()
        env._lazy_init()
        env._lazy_init()  # second call should be a no-op

        mock_wincam.assert_called_once()
        mock_det.assert_called_once()

    @mock.patch("src.perception.yolo_detector.YoloDetector")
    @mock.patch("src.capture.wincam_capture.WinCamCapture")
    def test_lazy_init_loads_detector(self, mock_wincam, mock_det):
        """_lazy_init should call detector.load()."""
        mock_det_instance = mock.MagicMock()
        mock_det.return_value = mock_det_instance
        mock_wincam.return_value = mock.MagicMock()

        env = Breakout71Env()
        env._lazy_init()

        mock_det_instance.load.assert_called_once()

    @mock.patch("src.perception.yolo_detector.YoloDetector")
    @mock.patch("src.capture.wincam_capture.WinCamCapture")
    def test_lazy_init_passes_device_to_detector(self, mock_wincam, mock_det):
        """_lazy_init should pass device param to YoloDetector."""
        mock_det.return_value = mock.MagicMock()
        mock_wincam.return_value = mock.MagicMock()

        env = Breakout71Env(device="cpu")
        env._lazy_init()

        mock_det.assert_called_once_with(
            weights_path=env.yolo_weights,
            device="cpu",
            classes=["ball", "brick", "paddle", "powerup", "wall"],
        )

    @mock.patch("src.perception.yolo_detector.YoloDetector")
    @mock.patch("src.capture.wincam_capture.WinCamCapture")
    def test_lazy_init_calls_init_canvas(self, mock_wincam, mock_det):
        """_lazy_init should call _init_canvas."""
        mock_det.return_value = mock.MagicMock()
        mock_wincam.return_value = mock.MagicMock()

        env = Breakout71Env()
        with mock.patch.object(env, "_init_canvas") as mock_canvas:
            env._lazy_init()
            mock_canvas.assert_called_once()


# -- Capture Frame -------------------------------------------------------------


class TestCaptureFrame:
    """Tests for _capture_frame delegation."""

    def test_capture_frame_delegates(self):
        """_capture_frame should delegate to capture.capture_frame."""
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
    """Tests for build_observation with various detection scenarios."""

    def test_normal_detection(self):
        """Normal detections should produce correct observation values."""
        env = Breakout71Env()
        det = _detections(
            paddle=(0.3, 0.9, 0.1, 0.02),
            ball=(0.6, 0.4, 0.02, 0.02),
            bricks=[(0.1, 0.1, 0.05, 0.03)] * 8,
        )
        obs = env.build_observation(det, reset=True)

        assert obs[0] == 0.3  # paddle_x
        assert obs[1] == 0.6  # ball_x
        assert obs[2] == 0.4  # ball_y
        assert obs.shape == (8,)
        assert obs.dtype == np.float32

    def test_missing_paddle(self):
        """Missing paddle detection should default to 0.5."""
        env = Breakout71Env()
        det = _detections(paddle=None)
        obs = env.build_observation(det, reset=True)

        assert obs[0] == 0.5  # default paddle_x

    def test_missing_ball(self):
        """Missing ball detection should default to 0.5 for position."""
        env = Breakout71Env()
        det = _detections(ball=None)
        obs = env.build_observation(det, reset=True)

        assert obs[1] == 0.5  # default ball_x
        assert obs[2] == 0.5  # default ball_y

    def test_missing_both(self):
        """Missing paddle and ball should both default to 0.5."""
        env = Breakout71Env()
        det = _detections(paddle=None, ball=None)
        obs = env.build_observation(det, reset=True)

        assert obs[0] == 0.5  # paddle_x
        assert obs[1] == 0.5  # ball_x
        assert obs[2] == 0.5  # ball_y

    def test_reset_zeroes_velocity(self):
        """On reset, velocities should be zero regardless of prev position."""
        env = Breakout71Env()
        env._prev_ball_pos = (0.3, 0.3)  # simulate prior state

        det = _detections(ball=(0.6, 0.4, 0.02, 0.02))
        obs = env.build_observation(det, reset=True)

        assert obs[3] == 0.0  # ball_vx
        assert obs[4] == 0.0  # ball_vy

    def test_reset_sets_bricks_total(self):
        """On reset, _bricks_total should be set from current brick count."""
        env = Breakout71Env()
        bricks = [(0.1 * i, 0.1, 0.05, 0.03) for i in range(15)]
        det = _detections(bricks=bricks)
        env.build_observation(det, reset=True)

        assert env._bricks_total == 15

    def test_reset_bricks_total_minimum_1(self):
        """On reset with no bricks, _bricks_total should be at least 1."""
        env = Breakout71Env()
        det = _detections(bricks=[])
        env.build_observation(det, reset=True)

        assert env._bricks_total == 1

    def test_velocity_computation(self):
        """Velocity should be computed from consecutive ball positions."""
        env = Breakout71Env()
        # First frame (reset)
        det1 = _detections(ball=(0.5, 0.5, 0.02, 0.02))
        env.build_observation(det1, reset=True)

        # Second frame
        det2 = _detections(ball=(0.6, 0.45, 0.02, 0.02))
        obs = env.build_observation(det2)

        assert abs(obs[3] - 0.1) < 1e-6  # ball_vx = 0.6 - 0.5
        assert abs(obs[4] - (-0.05)) < 1e-6  # ball_vy = 0.45 - 0.5

    def test_velocity_clipping(self):
        """Velocities exceeding [-1, 1] should be clipped."""
        env = Breakout71Env()
        det1 = _detections(ball=(0.0, 0.0, 0.02, 0.02))
        env.build_observation(det1, reset=True)

        # Large jump -- would give velocity > 1.0
        det2 = _detections(ball=(1.0, 1.0, 0.02, 0.02))
        # Override prev_ball_pos to create extreme delta
        env._prev_ball_pos = (-1.0, -1.0)
        obs = env.build_observation(det2)

        assert obs[3] == 1.0  # clipped to 1.0
        assert obs[4] == 1.0  # clipped to 1.0

    def test_bricks_norm_fraction(self):
        """bricks_norm should be fraction of remaining bricks."""
        env = Breakout71Env()
        bricks_initial = [(0.1 * i, 0.1, 0.05, 0.03) for i in range(10)]
        det1 = _detections(bricks=bricks_initial)
        env.build_observation(det1, reset=True)

        # Simulate 3 bricks destroyed
        bricks_remaining = [(0.1 * i, 0.1, 0.05, 0.03) for i in range(7)]
        det2 = _detections(bricks=bricks_remaining)
        obs = env.build_observation(det2)

        assert abs(obs[5] - 0.7) < 1e-6  # 7/10

    def test_placeholder_slots_are_zero(self):
        """coins_norm and score_norm should be 0.0 in v1."""
        env = Breakout71Env()
        det = _detections()
        obs = env.build_observation(det, reset=True)

        assert obs[6] == 0.0  # coins_norm
        assert obs[7] == 0.0  # score_norm

    def test_observation_in_bounds(self):
        """Observation should always be within the observation space bounds."""
        env = Breakout71Env()
        det = _detections(
            paddle=(0.8, 0.9, 0.1, 0.02),
            ball=(0.2, 0.3, 0.02, 0.02),
        )
        obs = env.build_observation(det, reset=True)

        assert env.observation_space.contains(obs)


# -- Compute Reward ------------------------------------------------------------


class TestComputeReward:
    """Tests for compute_reward with various scenarios."""

    def test_brick_destruction_reward(self):
        """Destroying bricks should give positive reward proportional to delta."""
        env = Breakout71Env()
        env._bricks_total = 10
        env._prev_bricks_norm = 1.0

        # 8 bricks remaining (2 destroyed)
        det = _detections(bricks=[(0.1 * i, 0.1, 0.05, 0.03) for i in range(8)])
        reward = env.compute_reward(det, terminated=False, level_cleared=False)

        # brick_delta = 1.0 - 0.8 = 0.2; reward = 0.2 * 10 - 0.01 = 1.99
        assert abs(reward - 1.99) < 1e-6

    def test_time_penalty_only(self):
        """No brick change should give only time penalty."""
        env = Breakout71Env()
        env._bricks_total = 10
        env._prev_bricks_norm = 0.5

        det = _detections(bricks=[(0.1 * i, 0.1, 0.05, 0.03) for i in range(5)])
        reward = env.compute_reward(det, terminated=False, level_cleared=False)

        assert abs(reward - (-0.01)) < 1e-6

    def test_game_over_penalty(self):
        """Game over should give -5.0 penalty on top of time penalty."""
        env = Breakout71Env()
        env._bricks_total = 10
        env._prev_bricks_norm = 0.5

        det = _detections(bricks=[(0.1 * i, 0.1, 0.05, 0.03) for i in range(5)])
        reward = env.compute_reward(det, terminated=True, level_cleared=False)

        # time penalty (-0.01) + game_over (-5.0) = -5.01
        assert abs(reward - (-5.01)) < 1e-6

    def test_level_cleared_bonus(self):
        """Level cleared should give +1.0 bonus (multi-level play)."""
        env = Breakout71Env()
        env._bricks_total = 10
        env._prev_bricks_norm = 0.0  # already at 0

        det = _detections(bricks=[])
        # In multi-level play, level_cleared=True with terminated=False
        reward = env.compute_reward(det, terminated=False, level_cleared=True)

        # brick_delta = 0.0; time penalty (-0.01) + level_clear (+1.0) = 0.99
        assert abs(reward - 0.99) < 1e-6

    def test_combined_brick_and_level_clear(self):
        """Last brick destroyed + level cleared should give both rewards."""
        env = Breakout71Env()
        env._bricks_total = 10
        env._prev_bricks_norm = 0.1  # 1 brick left

        det = _detections(bricks=[])
        # In multi-level play, level_cleared=True with terminated=False
        reward = env.compute_reward(det, terminated=False, level_cleared=True)

        # brick_delta = 0.1 * 10 = 1.0; + time (-0.01) + level (+1.0) = 1.99
        assert abs(reward - 1.99) < 1e-6

    def test_no_brick_or_score_change_time_penalty_only(self):
        """When bricks and score are unchanged, reward is just the time penalty."""
        env = Breakout71Env()
        env._bricks_total = 10
        env._prev_bricks_norm = 1.0
        env._prev_score = 0

        det = _detections(bricks=[(0.1 * i, 0.1, 0.05, 0.03) for i in range(10)])
        reward = env.compute_reward(det, terminated=False, level_cleared=False, score=0)

        # Only time penalty since no bricks or score changed
        assert abs(reward - (-0.01)) < 1e-6

    def test_prev_bricks_norm_updated(self):
        """_prev_bricks_norm should be updated after reward computation."""
        env = Breakout71Env()
        env._bricks_total = 10
        env._prev_bricks_norm = 1.0

        det = _detections(bricks=[(0.1 * i, 0.1, 0.05, 0.03) for i in range(7)])
        env.compute_reward(det, terminated=False, level_cleared=False)

        assert abs(env._prev_bricks_norm - 0.7) < 1e-6


class TestRewardModePassthrough:
    """Verify Breakout71Env passes reward_mode to BaseGameEnv."""

    def test_default_is_yolo(self):
        """Default reward_mode is 'yolo'."""
        env = Breakout71Env()
        assert env.reward_mode == "yolo"

    def test_survival_mode_passthrough(self):
        """reward_mode='survival' is forwarded to BaseGameEnv."""
        env = Breakout71Env(reward_mode="survival")
        assert env.reward_mode == "survival"

    def test_invalid_mode_raises(self):
        """Invalid reward_mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid reward_mode"):
            Breakout71Env(reward_mode="bogus")


# -- Apply Action --------------------------------------------------------------


class TestApplyAction:
    """Tests for apply_action via JS mousemove dispatch."""

    def _make_env_for_action(self, *, canvas_w=1280, canvas_h=1024, rect=(0, 0)):
        """Create an env with canvas set up for action tests."""
        driver = _mock_driver()
        env = Breakout71Env(driver=driver)
        env._game_canvas = mock.MagicMock()
        env._canvas_size = (canvas_w, canvas_h)
        env._canvas_rect = rect
        return env

    def test_centre_action_maps_to_centre_clientx(self):
        """Action 0.0 (centre) should produce clientX at canvas centre."""
        env = self._make_env_for_action(canvas_w=1000, rect=(0, 0))

        env.apply_action(_action(0.0))

        env._driver.execute_script.assert_called_once()
        args = env._driver.execute_script.call_args[0]
        # clientX = rect_left + (0 + 1) / 2 * 1000 = 500
        assert args[2] == 500.0  # clientX

    def test_left_edge_action(self):
        """Action -1.0 should produce clientX at canvas left edge."""
        env = self._make_env_for_action(canvas_w=1000, rect=(0, 0))

        env.apply_action(_action(-1.0))

        args = env._driver.execute_script.call_args[0]
        # clientX = 0 + (-1 + 1) / 2 * 1000 = 0
        assert args[2] == 0.0  # clientX

    def test_right_edge_action(self):
        """Action +1.0 should produce clientX at canvas right edge."""
        env = self._make_env_for_action(canvas_w=1000, rect=(0, 0))

        env.apply_action(_action(1.0))

        args = env._driver.execute_script.call_args[0]
        # clientX = 0 + (1 + 1) / 2 * 1000 = 1000
        assert args[2] == 1000.0  # clientX

    def test_action_clamped_to_bounds(self):
        """Action values outside [-1, 1] should be clipped."""
        env = self._make_env_for_action(canvas_w=1000, rect=(0, 0))

        env.apply_action(_action(2.0))

        args = env._driver.execute_script.call_args[0]
        # clipped to +1.0 -> clientX = 1000
        assert args[2] == 1000.0

    def test_quarter_position(self):
        """Action -0.5 should map to 25% from left on 1000px canvas."""
        env = self._make_env_for_action(canvas_w=1000, rect=(0, 0))

        env.apply_action(_action(-0.5))

        args = env._driver.execute_script.call_args[0]
        # clientX = 0 + (-0.5 + 1) / 2 * 1000 = 250
        assert args[2] == 250.0

    def test_canvas_rect_offset_applied(self):
        """Canvas rect left/top should offset the clientX/clientY."""
        env = self._make_env_for_action(canvas_w=1000, canvas_h=800, rect=(100, 50))

        env.apply_action(_action(0.0))

        args = env._driver.execute_script.call_args[0]
        # clientX = 100 + (0 + 1) / 2 * 1000 = 600
        assert args[2] == 600.0
        # clientY = 50 + 0.9 * 800 = 770
        assert args[3] == 770.0

    def test_no_driver_is_noop(self):
        """Without driver, apply_action should be a no-op."""
        env = Breakout71Env()
        env.apply_action(_action(0.5))  # should not raise

    def test_no_canvas_is_noop(self):
        """Without game_canvas, apply_action should be a no-op."""
        env = Breakout71Env(driver=_mock_driver())
        env._game_canvas = None
        env.apply_action(_action(0.5))  # should not raise

    def test_no_canvas_size_is_noop(self):
        """Without canvas_size, apply_action should be a no-op."""
        env = Breakout71Env(driver=_mock_driver())
        env._game_canvas = mock.MagicMock()
        env._canvas_size = None
        env.apply_action(_action(0.5))  # should not raise

    def test_action_scalar_input(self):
        """Scalar action (0-d array or float) is normalised to (1,)."""
        env = self._make_env_for_action(canvas_w=1000, rect=(0, 0))

        # 0-d numpy array
        env.apply_action(np.float32(0.0))
        args = env._driver.execute_script.call_args[0]
        assert args[2] == 500.0  # centre

        # Plain Python float
        env._driver.reset_mock()
        env.apply_action(0.0)
        args = env._driver.execute_script.call_args[0]
        assert args[2] == 500.0  # centre

    def test_action_wrong_size_raises(self):
        """Action with size != 1 should raise ValueError."""
        env = self._make_env_for_action()

        with pytest.raises(ValueError, match="size 1"):
            env.apply_action(np.array([0.1, 0.2], dtype=np.float32))

    def test_y_position_is_paddle_row(self):
        """clientY should be at 90% of canvas height."""
        env = self._make_env_for_action(canvas_h=1000, rect=(0, 0))

        env.apply_action(_action(0.0))

        args = env._driver.execute_script.call_args[0]
        # clientY = 0 + 0.9 * 1000 = 900
        assert args[3] == 900.0

    def test_execute_script_called(self):
        """driver.execute_script should be called with MOVE_MOUSE_JS."""
        env = self._make_env_for_action()

        env.apply_action(_action(0.0))

        env._driver.execute_script.assert_called_once()
        args = env._driver.execute_script.call_args[0]
        # First arg is the JS snippet
        assert "mousemove" in args[0]
        # Second arg is the canvas selector
        assert args[1] == "game"

    def test_fallback_without_canvas_rect(self):
        """Without _canvas_rect, should use (0, 0) as origin."""
        env = self._make_env_for_action(canvas_w=1000, canvas_h=800)
        env._canvas_rect = None

        env.apply_action(_action(0.0))

        args = env._driver.execute_script.call_args[0]
        # clientX = (0 + 1) / 2 * 1000 = 500 (no rect offset)
        assert args[2] == 500.0
        # clientY = 0.9 * 800 = 720
        assert args[3] == 720.0


# -- Game State Handling -------------------------------------------------------


class TestHandleGameState:
    """Tests for handle_modals and start_game."""

    def test_gameplay_state_no_action(self):
        """Gameplay state should return 'gameplay' without extra JS calls."""
        driver = _mock_driver()
        driver.execute_script.return_value = {
            "state": "gameplay",
            "details": {},
        }
        env = Breakout71Env(driver=driver)

        state = env.handle_modals()

        assert state == "gameplay"
        # Only the detect call, no dismiss calls
        driver.execute_script.assert_called_once()

    @mock.patch("games.breakout71.env.time")
    def test_game_over_dismissed(self, mock_time):
        """Game over state should trigger dismiss JS execution by default."""
        driver = _mock_driver()
        driver.execute_script.side_effect = [
            {"state": "game_over", "details": {}},
            {"action": "restart_button", "text": "New Run"},
        ]
        env = Breakout71Env(driver=driver)

        state = env.handle_modals()

        assert state == "game_over"
        # Verify the dismiss JS was actually called (not just call count)
        js_calls = [args[0] for args, _ in driver.execute_script.call_args_list]
        assert DISMISS_GAME_OVER_JS in js_calls

    @mock.patch("games.breakout71.env.time")
    def test_game_over_not_dismissed_when_flag_false(self, mock_time):
        """Game over should be detected but NOT dismissed when dismiss_game_over=False."""
        driver = _mock_driver()
        driver.execute_script.return_value = {
            "state": "game_over",
            "details": {},
        }
        env = Breakout71Env(driver=driver)

        state = env.handle_modals(dismiss_game_over=False)

        assert state == "game_over"
        # Verify dismiss JS was NOT called
        js_calls = [args[0] for args, _ in driver.execute_script.call_args_list]
        assert DISMISS_GAME_OVER_JS not in js_calls

    @mock.patch("games.breakout71.env.time")
    def test_perk_picker_clicks_perk(self, mock_time):
        """Perk picker state should click a random perk button."""
        driver = _mock_driver()
        driver.execute_script.side_effect = [
            {"state": "perk_picker", "details": {"numPerks": 3}},
            {"clicked": 1, "text": "Extra Life"},
        ]
        env = Breakout71Env(driver=driver)

        state = env.handle_modals()

        assert state == "perk_picker"
        assert driver.execute_script.call_count == 2

    @mock.patch("games.breakout71.env.time")
    def test_menu_dismissed(self, mock_time):
        """Menu state should trigger dismiss."""
        driver = _mock_driver()
        driver.execute_script.side_effect = [
            {"state": "menu", "details": {}},
            {"action": "close_button"},
        ]
        env = Breakout71Env(driver=driver)

        state = env.handle_modals()

        assert state == "menu"

    def test_no_driver_returns_gameplay(self):
        """Without a driver, handle_modals returns 'gameplay'."""
        env = Breakout71Env()

        assert env.handle_modals() == "gameplay"

    def test_state_detection_exception_returns_unknown(self):
        """Exception during state detection should return 'unknown'."""
        driver = _mock_driver()
        driver.execute_script.side_effect = RuntimeError("WebDriver error")
        env = Breakout71Env(driver=driver)

        state = env.handle_modals()

        assert state == "unknown"

    def test_state_detection_returns_none(self):
        """Null state detection result should return 'unknown'."""
        driver = _mock_driver()
        driver.execute_script.return_value = None
        env = Breakout71Env(driver=driver)

        state = env.handle_modals()

        assert state == "unknown"

    def test_start_game_sets_gamestate_via_js(self):
        """start_game should set gameState.running and ballStickToPuck via JS."""
        driver = _mock_driver()
        env = Breakout71Env(driver=driver)

        env.start_game()

        driver.execute_script.assert_called_with(
            "gameState.running = true;gameState.ballStickToPuck = false;"
        )

    def test_start_game_falls_back_to_action_chains_on_js_error(self):
        """start_game should try ActionChains click if JS mutation fails."""
        driver = _mock_driver()
        driver.execute_script.side_effect = Exception("JS failed")
        env = Breakout71Env(driver=driver)
        env._game_canvas = mock.MagicMock()

        mock_ac_module = sys.modules["selenium.webdriver.common.action_chains"]
        mock_chains = mock.MagicMock()
        mock_chains.move_to_element.return_value = mock_chains
        mock_chains.click.return_value = mock_chains
        mock_ac_module.ActionChains.return_value = mock_chains

        env.start_game()

        mock_chains.move_to_element.assert_called_once_with(env._game_canvas)
        mock_chains.click.assert_called_once()
        mock_chains.perform.assert_called_once()

    def test_start_game_no_driver(self):
        """start_game without driver should be a no-op."""
        env = Breakout71Env()
        env.start_game()  # should not raise

    def test_start_game_no_canvas_js_fails_gracefully(self):
        """start_game with JS failure and no canvas should not raise."""
        driver = _mock_driver()
        driver.execute_script.side_effect = Exception("JS failed")
        env = Breakout71Env(driver=driver)
        env._game_canvas = None
        env.start_game()  # should not raise


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
    """Tests for build_info helper and _make_info wrapper."""

    def test_info_contains_expected_keys(self):
        """Combined info dict (_make_info) should contain all expected keys."""
        env = Breakout71Env()
        env._last_frame = _frame()
        env._step_count = 5
        det = _detections()

        info = env._make_info(det)

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

        info = env.build_info(det)

        assert info["ball_pos"] == [0.3, 0.7]

    def test_info_missing_ball(self):
        """Info ball_pos should be None when ball not detected."""
        env = Breakout71Env()
        env._last_frame = _frame()
        det = _detections(ball=None)

        info = env.build_info(det)

        assert info["ball_pos"] is None

    def test_info_brick_count(self):
        """Info brick_count should match detection count."""
        env = Breakout71Env()
        env._last_frame = _frame()
        bricks = [(0.1 * i, 0.1, 0.05, 0.03) for i in range(7)]
        det = _detections(bricks=bricks)

        info = env.build_info(det)

        assert info["brick_count"] == 7


# -- Reset ---------------------------------------------------------------------


class TestReset:
    """Tests for the full reset() lifecycle."""

    def _make_env_with_mocks(self):
        """Create env with mocked sub-components for reset/step tests."""
        driver = _mock_driver()
        env = Breakout71Env(driver=driver)
        env._initialized = True
        _setup_capture_mock(env)
        env._game_canvas = mock.MagicMock()
        env._canvas_size = (640, 480)
        env._detector = mock.MagicMock()
        env._detector.detect_to_game_state.return_value = _detections()
        return env

    @mock.patch("games.breakout71.env.time")
    def test_reset_returns_obs_and_info(self, mock_time):
        """reset() should return (obs, info) tuple."""
        env = self._make_env_with_mocks()

        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (8,)
        assert isinstance(info, dict)

    @mock.patch("games.breakout71.env.time")
    def test_reset_handles_game_state(self, mock_time):
        """reset() should call handle_modals to dismiss modals."""
        env = self._make_env_with_mocks()

        env.reset()

        # Driver.execute_script is called at least once for state detection
        env._driver.execute_script.assert_called()

    @mock.patch("games.breakout71.env.time")
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

    @mock.patch("games.breakout71.env.time")
    def test_reset_clears_oracles(self, mock_time):
        """reset() should clear and call on_reset on all oracles."""
        env = self._make_env_with_mocks()
        oracle = mock.MagicMock()
        env._oracles = [oracle]

        env.reset()

        oracle.clear.assert_called_once()
        oracle.on_reset.assert_called_once()

    @mock.patch("games.breakout71.env.time")
    def test_reset_calls_lazy_init_when_not_initialized(self, mock_time):
        """reset() should call _lazy_init on first call."""
        driver = _mock_driver()
        env = Breakout71Env(driver=driver)
        env._lazy_init = mock.MagicMock()
        # After lazy_init, we need the sub-components set up
        _setup_capture_mock(env)
        env._detector = mock.MagicMock()
        env._detector.detect_to_game_state.return_value = _detections()
        env._game_canvas = mock.MagicMock()
        env._canvas_size = (640, 480)

        env.reset()

        env._lazy_init.assert_called_once()

    @mock.patch("games.breakout71.env.time")
    def test_reset_raises_after_ball_retry_exhausted(self, mock_time):
        """reset() raises RuntimeError if ball never detected after 5 retries."""
        driver = _mock_driver()
        env = Breakout71Env(driver=driver)
        env._initialized = True
        _setup_capture_mock(env)
        env._game_canvas = mock.MagicMock()
        env._canvas_size = (640, 480)
        env._detector = mock.MagicMock()
        env._detector.detect_to_game_state.return_value = _detections(ball=None)

        with pytest.raises(RuntimeError, match="failed to get valid detections"):
            env.reset()


# -- Step ----------------------------------------------------------------------


class TestStep:
    """Tests for the full step() lifecycle."""

    def _make_env_ready(self, bricks_count=10):
        """Create env in a post-reset state with mocked sub-components."""
        return _make_env_ready(bricks_count=bricks_count)

    @mock.patch("games.breakout71.env.time")
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

    @mock.patch("games.breakout71.env.time")
    def test_step_increments_counter(self, mock_time):
        """step() should increment _step_count."""
        env = self._make_env_ready()
        assert env._step_count == 0

        env.step(_action())

        assert env._step_count == 1

    @mock.patch("games.breakout71.env.time")
    def test_step_applies_action(self, mock_time):
        """step() should call apply_action."""
        env = self._make_env_ready()

        with mock.patch.object(env, "apply_action") as mock_apply:
            env.step(_action(0.5))
            mock_apply.assert_called_once()

    @mock.patch("games.breakout71.env.time")
    def test_step_normal_not_terminated(self, mock_time):
        """Normal step with ball and bricks should not terminate."""
        env = self._make_env_ready()

        _, _, terminated, truncated, _ = env.step(_action())

        assert terminated is False
        assert truncated is False

    @mock.patch("games.breakout71.env.time")
    def test_step_truncation_at_max_steps(self, mock_time):
        """step() should set truncated=True when max_steps is reached."""
        env = self._make_env_ready()
        env.max_steps = 5
        env._step_count = 4  # will become 5 after increment -> equals max_steps

        _, _, terminated, truncated, _ = env.step(_action())

        assert truncated is True

    @mock.patch("games.breakout71.env.time")
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

    @mock.patch("games.breakout71.env.time")
    def test_step_ball_found_resets_counter(self, mock_time):
        """Finding ball should reset _no_ball_count to 0."""
        env = self._make_env_ready()
        env._no_ball_count = 3

        env.step(_action())  # default detections have ball

        assert env._no_ball_count == 0

    @mock.patch("games.breakout71.env.time")
    def test_step_level_cleared(self, mock_time):
        """Level cleared should trigger level transition, not termination.

        In multi-level play, level clear calls _handle_level_transition()
        which (with a mock driver) succeeds and continues the episode.
        """
        env = self._make_env_ready()
        env._no_bricks_count = Breakout71Env._LEVEL_CLEAR_THRESHOLD - 1

        # Return detections with no bricks
        env._detector.detect_to_game_state.return_value = _detections(bricks=[])

        _, _, terminated, _, _ = env.step(_action())

        # Transition succeeds (mock driver present) → episode continues
        assert terminated is False
        assert env._levels_cleared == 1

    @mock.patch("games.breakout71.env.time")
    def test_step_bricks_present_resets_counter(self, mock_time):
        """Having bricks should reset _no_bricks_count to 0."""
        env = self._make_env_ready()
        env._no_bricks_count = 2

        env.step(_action())  # default detections have bricks

        assert env._no_bricks_count == 0

    @mock.patch("games.breakout71.env.time")
    def test_step_info_has_oracle_findings(self, mock_time):
        """step() info dict should include oracle_findings."""
        env = self._make_env_ready()

        _, _, _, _, info = env.step(_action())

        assert "oracle_findings" in info

    @mock.patch("games.breakout71.env.time")
    def test_step_runs_oracles(self, mock_time):
        """step() should call on_step on all attached oracles."""
        env = self._make_env_ready()
        oracle = mock.MagicMock()
        oracle.get_findings.return_value = []
        env._oracles = [oracle]

        env.step(_action())

        oracle.on_step.assert_called_once()

    @mock.patch("games.breakout71.env.time")
    def test_step_observation_in_bounds(self, mock_time):
        """step() observation should be within observation_space."""
        env = self._make_env_ready()

        obs, _, _, _, _ = env.step(_action())

        assert env.observation_space.contains(obs)

    @mock.patch("games.breakout71.env.time")
    def test_step_handles_mid_episode_modal(self, mock_time):
        """step() should handle perk_picker modals that appear mid-episode."""
        env = self._make_env_ready()
        env._no_ball_count = 1  # ball missing → triggers modal check
        # Make game state return perk_picker on first call, gameplay after
        env._driver.execute_script.side_effect = [
            {"state": "perk_picker", "details": {"numPerks": 3}},
            {"clicked": 0, "text": "Speed"},
        ]

        # Should not crash; modal is handled
        env.step(_action())

    @mock.patch("games.breakout71.env.time")
    def test_step_terminates_on_game_over_modal(self, mock_time):
        """step() should return terminated=True when game_over modal detected."""
        env = self._make_env_ready()
        env._no_ball_count = 1  # ball missing → triggers modal check
        env._driver.execute_script.return_value = {
            "state": "game_over",
            "details": {},
        }

        _, _, terminated, _, _ = env.step(_action())

        assert terminated is True

    @mock.patch("games.breakout71.env.time")
    def test_step_game_over_does_not_dismiss_modal(self, mock_time):
        """step() should NOT dismiss game_over modal (only reset() does)."""
        env = self._make_env_ready()
        env._no_ball_count = 1  # ball missing → triggers modal check
        env._driver.execute_script.return_value = {
            "state": "game_over",
            "details": {},
        }

        env.step(_action())

        # Verify dismiss JS was NOT called (check call_args, not count)
        js_calls = [args[0] for args, _ in env._driver.execute_script.call_args_list]
        assert DISMISS_GAME_OVER_JS not in js_calls

    @mock.patch("games.breakout71.env.time")
    def test_step_game_over_still_increments_step_count(self, mock_time):
        """step() should still increment _step_count when game_over detected."""
        env = self._make_env_ready()
        env._no_ball_count = 1  # ball missing → triggers modal check
        env._driver.execute_script.return_value = {
            "state": "game_over",
            "details": {},
        }
        assert env._step_count == 0

        env.step(_action())

        assert env._step_count == 1

    @mock.patch("games.breakout71.env.time")
    def test_step_game_over_returns_fixed_terminal_penalty(self, mock_time):
        """step() should use fixed penalty on game_over, not detection-based reward.

        The game-over modal occludes bricks, so detection-based reward
        would produce incorrect brick-count deltas.  The reward must
        be the fixed terminal penalty (-5.01) regardless of detections.
        """
        env = self._make_env_ready()
        env._no_ball_count = 1  # ball missing → triggers modal check
        env._driver.execute_script.return_value = {
            "state": "game_over",
            "details": {},
        }

        _, reward, terminated, _, _ = env.step(_action())

        assert terminated is True
        assert reward == pytest.approx(-5.01)

    @mock.patch("games.breakout71.env.time")
    def test_step_perk_picker_does_not_terminate(self, mock_time):
        """step() should NOT terminate on perk_picker — it's part of gameplay."""
        env = self._make_env_ready()
        env._no_ball_count = 1  # ball missing → triggers modal check
        env._driver.execute_script.side_effect = [
            {"state": "perk_picker", "details": {"numPerks": 3}},
            {"clicked": 0, "text": "Speed"},
        ]

        _, _, terminated, _, _ = env.step(_action())

        assert terminated is False


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
        driver = _mock_driver()
        env = Breakout71Env(driver=driver)

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

    def test_close_clears_canvas(self):
        """close() should clear _game_canvas and _canvas_size."""
        env = Breakout71Env()
        env._initialized = True
        env._capture = mock.MagicMock()
        env._game_canvas = mock.MagicMock()
        env._canvas_size = (640, 480)

        env.close()

        assert env._game_canvas is None
        assert env._canvas_size is None


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


# -- Headless Mode -------------------------------------------------------------


class TestHeadless:
    """Tests for headless mode (Selenium-based capture/input)."""

    def test_headless_param_default_false(self):
        """Headless mode should default to False."""
        env = Breakout71Env()
        assert env.headless is False

    def test_headless_param_set(self):
        """Constructor should accept headless parameter."""
        env = Breakout71Env(headless=True)
        assert env.headless is True

    def test_headless_init_state(self):
        """Headless mode should initialise canvas state to None."""
        env = Breakout71Env(headless=True)
        assert env._game_canvas is None
        assert env._canvas_size is None

    def test_headless_lazy_init_requires_driver(self):
        """_lazy_init in headless mode should raise if no driver."""
        env = Breakout71Env(headless=True, driver=None)
        with pytest.raises(RuntimeError, match="Headless mode requires"):
            env._lazy_init()

    @mock.patch("src.perception.yolo_detector.YoloDetector")
    def test_headless_lazy_init_finds_canvas(self, mock_det):
        """_lazy_init in headless mode should find the game canvas."""
        mock_det.return_value = mock.MagicMock()
        driver = _mock_driver()
        canvas_el = mock.MagicMock()
        canvas_el.size = {"width": 1280, "height": 1024}
        driver.find_element.return_value = canvas_el

        env = Breakout71Env(headless=True, driver=driver)
        env._lazy_init()

        assert env._initialized is True
        assert env._game_canvas is canvas_el
        assert env._canvas_size == (1280, 1024)
        assert env._capture is None  # No Win32 capture in headless

    @mock.patch("src.perception.yolo_detector.YoloDetector")
    def test_headless_lazy_init_body_fallback(self, mock_det):
        """_lazy_init should fall back to <body> if #game not found."""
        mock_det.return_value = mock.MagicMock()
        driver = _mock_driver()
        body_el = mock.MagicMock()
        body_el.size = {"width": 800, "height": 600}
        # First call raises (for By.ID "game"), second returns body
        driver.find_element.side_effect = [Exception("not found"), body_el]

        env = Breakout71Env(headless=True, driver=driver)
        env._lazy_init()

        assert env._initialized is True
        assert env._game_canvas is body_el
        assert env._canvas_size == (800, 600)

    @mock.patch("src.perception.yolo_detector.YoloDetector")
    def test_headless_lazy_init_sets_canvas_via_driver(self, mock_det):
        """Headless init should set up _game_canvas and _canvas_size via driver."""
        mock_det.return_value = mock.MagicMock()
        driver = _mock_driver()
        canvas_el = mock.MagicMock()
        canvas_el.size = {"width": 1280, "height": 1024}
        driver.find_element.return_value = canvas_el

        env = Breakout71Env(headless=True, driver=driver)
        env._lazy_init()

        assert env._game_canvas is canvas_el
        assert env._canvas_size == (1280, 1024)

    def test_headless_capture_frame_via_screenshot(self):
        """_capture_frame in headless mode should use Selenium screenshot."""
        env = Breakout71Env(headless=True)
        env._driver = mock.MagicMock()

        # Provide dummy PNG bytes (content doesn't matter — cv2 is mocked)
        env._driver.get_screenshot_as_png.return_value = b"fake-png-bytes"

        # Mock cv2 to avoid libGL.so.1 dependency in CI Docker
        dummy_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        dummy_frame[:] = (255, 0, 0)  # blue BGR
        mock_cv2 = mock.MagicMock()
        mock_cv2.IMREAD_COLOR = 1
        mock_cv2.imdecode.return_value = dummy_frame

        with mock.patch.dict(sys.modules, {"cv2": mock_cv2}):
            frame = env._capture_frame()

        env._driver.get_screenshot_as_png.assert_called_once()
        mock_cv2.imdecode.assert_called_once()
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (10, 10, 3)
        assert env._last_frame is frame

    def test_headless_capture_frame_no_driver_raises(self):
        """_capture_frame_headless should raise RuntimeError if driver is None."""
        env = Breakout71Env(headless=True)
        env._driver = None

        with pytest.raises(RuntimeError, match="driver is None"):
            env._capture_frame_headless()

    def test_headless_capture_frame_decode_fails_raises(self):
        """_capture_frame_headless should raise if imdecode returns None."""
        env = Breakout71Env(headless=True)
        env._driver = mock.MagicMock()
        env._driver.get_screenshot_as_png.return_value = b"corrupt-png"

        mock_cv2 = mock.MagicMock()
        mock_cv2.IMREAD_COLOR = 1
        mock_cv2.imdecode.return_value = None

        with mock.patch.dict(sys.modules, {"cv2": mock_cv2}):
            with pytest.raises(RuntimeError, match="Failed to decode"):
                env._capture_frame_headless()

    def test_headless_apply_action_uses_js_mousemove(self):
        """apply_action in headless mode should use execute_script for mousemove."""
        env = Breakout71Env(headless=True)
        env._driver = mock.MagicMock()
        env._game_canvas = mock.MagicMock()
        env._canvas_size = (1280, 1024)
        env._canvas_rect = (0, 0)

        env.apply_action(_action(0.5))

        env._driver.execute_script.assert_called_once()
        args = env._driver.execute_script.call_args[0]
        assert "mousemove" in args[0]
        assert args[1] == "game"  # canvas selector

    def test_headless_apply_action_maps_position(self):
        """apply_action should map [-1,1] to clientX in viewport coords."""
        env = Breakout71Env(headless=True)
        env._driver = mock.MagicMock()
        env._game_canvas = mock.MagicMock()
        env._canvas_size = (1000, 800)
        env._canvas_rect = (0, 0)

        env.apply_action(_action(-1.0))

        args = env._driver.execute_script.call_args[0]
        # clientX = 0 + (-1 + 1) / 2 * 1000 = 0
        assert args[2] == 0.0  # clientX (left edge)
        # clientY = 0 + 0.9 * 800 = 720
        assert args[3] == 720.0  # clientY

    def test_headless_apply_action_no_driver(self):
        """apply_action in headless mode should do nothing without driver."""
        env = Breakout71Env(headless=True)
        env._driver = None
        env.apply_action(_action(0.0))  # should not raise

    def test_headless_apply_action_no_canvas(self):
        """apply_action in headless mode should do nothing without canvas."""
        env = Breakout71Env(headless=True)
        env._driver = mock.MagicMock()
        env._game_canvas = None
        env.apply_action(_action(0.0))  # should not raise

    def test_headless_start_game_uses_js_mutation(self):
        """start_game in headless mode should set gameState via JS."""
        env = Breakout71Env(headless=True)
        env._driver = mock.MagicMock()
        env._game_canvas = mock.MagicMock()

        env.start_game()

        env._driver.execute_script.assert_called_with(
            "gameState.running = true;gameState.ballStickToPuck = false;"
        )

    def test_headless_start_game_no_driver(self):
        """start_game in headless mode should do nothing without driver."""
        env = Breakout71Env(headless=True)
        env._driver = None
        env.start_game()  # should not raise

    def test_close_clears_headless_state(self):
        """close() should clear headless-specific state."""
        env = Breakout71Env(headless=True)
        env._game_canvas = mock.MagicMock()
        env._canvas_size = (1280, 1024)
        env._initialized = True

        env.close()

        assert env._game_canvas is None
        assert env._canvas_size is None
        assert env._initialized is False


# -- Build Info (no_ball_count) ------------------------------------------------


class TestBuildInfoNoBallCount:
    """Tests for no_ball_count in build_info."""

    def test_info_includes_no_ball_count(self):
        """build_info should include no_ball_count field."""
        env = Breakout71Env()
        env._no_ball_count = 3
        det = _detections()

        info = env.build_info(det)

        assert "no_ball_count" in info
        assert info["no_ball_count"] == 3

    def test_info_no_ball_count_default_zero(self):
        """no_ball_count should be 0 initially."""
        env = Breakout71Env()
        det = _detections()

        info = env.build_info(det)

        assert info["no_ball_count"] == 0


class TestModalCheckThrottling:
    """TDD specs for modal check throttling (option 3).

    The behavioral contract: ``handle_modals()`` should only be
    called when ``_no_ball_count > 0`` (ball is missing).  During normal
    gameplay (ball visible), we skip the Selenium round-trip entirely,
    removing the ~100-150ms overhead per step.
    """

    def _make_env_ready(self, bricks_count=10):
        """Create env in a post-reset state with mocked sub-components."""
        return _make_env_ready(bricks_count=bricks_count)

    @mock.patch("games.breakout71.env.time")
    def test_step_skips_handle_modals_when_ball_detected(self, mock_time):
        """step() should NOT call handle_modals when _no_ball_count == 0.

        This is the key optimization: during normal gameplay with the ball
        visible, we skip the Selenium round-trip entirely.
        """
        env = self._make_env_ready()
        assert env._no_ball_count == 0  # ball was detected last step

        with mock.patch.object(env, "handle_modals") as mock_hgs:
            env.step(_action())
            mock_hgs.assert_not_called()

    @mock.patch("games.breakout71.env.time")
    def test_step_calls_handle_modals_when_ball_missing(self, mock_time):
        """step() should call handle_modals when _no_ball_count > 0.

        When the ball has been missing for one or more frames, we check
        for modals since the missing ball might be caused by a modal overlay.
        """
        env = self._make_env_ready()
        env._no_ball_count = 3  # ball has been missing for 3 frames

        with mock.patch.object(
            env, "handle_modals", return_value="gameplay"
        ) as mock_hgs:
            env.step(_action())
            mock_hgs.assert_called_once()

    @mock.patch("games.breakout71.env.time")
    def test_step_detects_game_over_modal_when_ball_missing(self, mock_time):
        """step() should still detect game_over modals and terminate
        the episode when _no_ball_count > 0 triggers the check.
        """
        env = self._make_env_ready()
        env._no_ball_count = 2  # ball missing → triggers modal check

        with mock.patch.object(env, "handle_modals", return_value="game_over"):
            _, _, terminated, _, _ = env.step(_action())
            assert terminated is True

    @mock.patch("games.breakout71.env.time")
    def test_step_handles_perk_picker_when_ball_missing(self, mock_time):
        """step() should still handle perk_picker modals when triggered
        by _no_ball_count > 0.
        """
        env = self._make_env_ready()
        env._no_ball_count = 1  # ball missing → triggers modal check

        with mock.patch.object(
            env, "handle_modals", return_value="perk_picker"
        ) as mock_hgs:
            _, _, terminated, _, _ = env.step(_action())
            # Perk picker is part of normal gameplay — not terminated
            assert terminated is False
            mock_hgs.assert_called_once()

    @mock.patch("games.breakout71.env.time")
    def test_step_skips_handle_modals_on_first_step_after_reset(self, mock_time):
        """The first step after reset (ball visible) should NOT call
        handle_modals — no unnecessary Selenium overhead.
        """
        env = self._make_env_ready()
        # Fresh after reset: _no_ball_count == 0, _step_count == 0
        assert env._no_ball_count == 0
        assert env._step_count == 0

        with mock.patch.object(env, "handle_modals") as mock_hgs:
            env.step(_action())
            mock_hgs.assert_not_called()

    @mock.patch("games.breakout71.env.time")
    def test_step_calls_handle_modals_at_threshold_boundary(self, mock_time):
        """step() should call handle_modals when _no_ball_count
        is exactly 1 (just crossed from 0).
        """
        env = self._make_env_ready()
        env._no_ball_count = 1  # exactly at boundary

        with mock.patch.object(
            env, "handle_modals", return_value="gameplay"
        ) as mock_hgs:
            env.step(_action())
            mock_hgs.assert_called_once()

    @mock.patch("games.breakout71.env.time")
    def test_step_consecutive_ball_visible_frames_never_check_modals(self, mock_time):
        """Multiple consecutive steps with ball visible should never
        call handle_modals.
        """
        env = self._make_env_ready()
        # Detector always returns ball detected
        assert env._no_ball_count == 0

        with mock.patch.object(env, "handle_modals") as mock_hgs:
            for _ in range(5):
                env.step(_action())
            mock_hgs.assert_not_called()

    @mock.patch("games.breakout71.env.time")
    def test_step_detects_game_over_on_zero_to_one_ball_miss(self, mock_time):
        """When ball was visible last step (_no_ball_count == 0) and
        disappears this step (game-over modal appeared), the late
        modal check on the 0->1 transition should detect game_over
        and return terminated=True with fixed penalty reward.

        This prevents spurious positive rewards from modal-occluded
        brick detections — the exact scenario the fixed terminal
        penalty path is designed to handle.
        """
        env = self._make_env_ready()
        assert env._no_ball_count == 0  # ball was visible last step

        # YOLO detects no ball (modal occludes it) but sees bricks
        env._detector.detect_to_game_state.return_value = _detections(
            ball=None, bricks=[(0.1 * i, 0.1, 0.05, 0.03) for i in range(5)]
        )

        # Late modal check (after detection, on 0->1 transition)
        # returns game_over
        with mock.patch.object(env, "handle_modals", return_value="game_over"):
            _, reward, terminated, _, _ = env.step(_action())
            assert terminated is True
            assert reward == pytest.approx(-5.01)

    @mock.patch("games.breakout71.env.time")
    def test_step_no_spurious_reward_on_zero_to_one_game_over(self, mock_time):
        """When game-over is detected on the 0->1 ball-miss transition,
        compute_reward should NOT be called — the fixed terminal
        penalty is used directly to avoid modal-occluded brick-delta
        producing a spurious positive reward.
        """
        env = self._make_env_ready()
        assert env._no_ball_count == 0

        # Set up YOLO to return fewer bricks (simulating modal occlusion
        # that would normally produce a large positive brick-delta reward)
        env._detector.detect_to_game_state.return_value = _detections(
            ball=None, bricks=[(0.5, 0.1, 0.05, 0.03)]
        )

        with (
            mock.patch.object(env, "handle_modals", return_value="game_over"),
            mock.patch.object(env, "compute_reward") as mock_cr,
        ):
            _, reward, terminated, _, _ = env.step(_action())
            assert terminated is True
            # compute_reward should NOT have been called
            mock_cr.assert_not_called()
            # Fixed penalty used instead
            assert reward == pytest.approx(-5.01)


# -- Multi-Level Play ---------------------------------------------------------


class TestReadGameStateJS:
    """Tests for the READ_GAME_STATE_JS snippet."""

    def test_snippet_exists_and_is_string(self):
        """READ_GAME_STATE_JS should be a non-empty string."""
        assert isinstance(READ_GAME_STATE_JS, str)
        assert len(READ_GAME_STATE_JS) > 0

    def test_snippet_reads_score_level_lives(self):
        """READ_GAME_STATE_JS should reference score, level, lives keys."""
        assert "score" in READ_GAME_STATE_JS
        assert "level" in READ_GAME_STATE_JS
        assert "lives" in READ_GAME_STATE_JS


class TestMultiLevelState:
    """Tests for multi-level state tracking in Breakout71Env."""

    def test_initial_multi_level_state(self):
        """New env should have multi-level state initialized to defaults."""
        env = Breakout71Env()
        assert env._prev_score == 0
        assert env._current_level == 0
        assert env._levels_cleared == 0

    def test_reset_clears_multi_level_state(self):
        """reset_termination_state should reset multi-level counters."""
        env = Breakout71Env()
        env._prev_score = 100
        env._current_level = 3
        env._levels_cleared = 2
        env.reset_termination_state()
        assert env._prev_score == 0
        assert env._current_level == 0
        assert env._levels_cleared == 0


class TestReadGameScore:
    """Tests for _read_game_state() JS bridge."""

    def test_read_game_state_returns_score_and_level(self):
        """_read_game_state should return score and level from JS."""
        driver = _mock_driver()
        env = Breakout71Env(driver=driver)

        driver.execute_script.return_value = {
            "score": 42,
            "level": 2,
            "lives": 3,
            "running": True,
        }
        state = env._read_game_state()
        assert state["score"] == 42
        assert state["level"] == 2
        assert state["lives"] == 3

    def test_read_game_state_no_driver(self):
        """_read_game_state without driver should return defaults."""
        env = Breakout71Env()
        state = env._read_game_state()
        assert state["score"] == 0
        assert state["level"] == 0

    def test_read_game_state_handles_exception(self):
        """_read_game_state should return defaults on JS error."""
        driver = _mock_driver()
        env = Breakout71Env(driver=driver)
        driver.execute_script.side_effect = Exception("JS error")
        state = env._read_game_state()
        assert state["score"] == 0
        assert state["level"] == 0


class TestHandleLevelTransition:
    """Tests for _handle_level_transition() perk picker loop."""

    def test_handle_level_transition_clicks_perks(self):
        """_handle_level_transition should click perks until modal closes."""
        driver = _mock_driver()
        env = Breakout71Env(driver=driver)
        env._initialized = True
        env._bricks_total = 10
        env._no_bricks_count = 3

        # Simulate: first call = perk_picker, second = perk_picker,
        # third = gameplay (modal closed)
        call_count = [0]

        def mock_detect_state(js):
            from games.breakout71.modal_handler import (
                CLICK_PERK_JS,
                DETECT_STATE_JS,
            )

            if js == DETECT_STATE_JS:
                call_count[0] += 1
                if call_count[0] <= 2:
                    return {"state": "perk_picker", "details": {"numPerks": 3}}
                return {"state": "gameplay", "details": {}}
            if js == CLICK_PERK_JS:
                return {"clicked": 0, "text": "Multiball"}
            if js == READ_GAME_STATE_JS:
                return {"score": 0, "level": 1, "lives": 3, "running": True}
            return None

        driver.execute_script.side_effect = mock_detect_state

        result = env._handle_level_transition()
        assert result is True
        assert env._levels_cleared == 1
        assert env._no_bricks_count == 0

    def test_handle_level_transition_no_driver_returns_false(self):
        """_handle_level_transition without driver should return False."""
        env = Breakout71Env()
        result = env._handle_level_transition()
        assert result is False

    def test_handle_level_transition_resets_brick_state(self):
        """After transition, brick tracking state should be reset."""
        driver = _mock_driver()
        env = Breakout71Env(driver=driver)
        env._initialized = True
        env._bricks_total = 10
        env._prev_bricks_norm = 0.0
        env._no_bricks_count = 3

        # Modal immediately closes (gameplay state)
        driver.execute_script.return_value = {
            "state": "gameplay",
            "details": {},
        }

        env._handle_level_transition()
        assert env._bricks_total is None  # reset for re-calibration
        assert env._prev_bricks_norm == 1.0
        assert env._no_bricks_count == 0

    def test_handle_level_transition_max_retries(self):
        """_handle_level_transition should give up after max retries."""
        driver = _mock_driver()
        env = Breakout71Env(driver=driver)
        env._initialized = True
        env._bricks_total = 10

        # Always returns perk_picker (stuck modal)
        driver.execute_script.return_value = {
            "state": "perk_picker",
            "details": {"numPerks": 3},
        }

        result = env._handle_level_transition()
        # Should still return True (best effort) but not loop forever
        assert result is True


class TestStepMultiLevel:
    """Tests for step() behavior with multi-level play."""

    def _make_env_ready(self, bricks_count=10):
        """Create a ready env (reuse existing helper pattern)."""
        return _make_env_ready(bricks_count)

    @mock.patch("games.breakout71.env.time")
    def test_step_level_cleared_continues_when_transition_succeeds(self, mock_time):
        """When level is cleared and transition succeeds, episode continues."""
        env = self._make_env_ready(bricks_count=10)

        # Simulate zero bricks for 3+ frames → level_cleared
        env._no_bricks_count = 2  # one more zero will trigger
        env._detector.detect_to_game_state.return_value = _detections(bricks=[])

        with mock.patch.object(env, "_handle_level_transition", return_value=True):
            _, reward, terminated, truncated, info = env.step(_action())
            assert terminated is False
            assert truncated is False

    @mock.patch("games.breakout71.env.time")
    def test_step_level_cleared_terminates_when_transition_fails(self, mock_time):
        """When level is cleared but transition fails, episode terminates."""
        env = self._make_env_ready(bricks_count=10)

        # Simulate level_cleared
        env._no_bricks_count = 2
        env._detector.detect_to_game_state.return_value = _detections(bricks=[])

        with mock.patch.object(env, "_handle_level_transition", return_value=False):
            _, reward, terminated, truncated, info = env.step(_action())
            assert terminated is True

    @mock.patch("games.breakout71.env.time")
    def test_step_level_cleared_gives_bonus_reward(self, mock_time):
        """Level clear should give a positive bonus reward."""
        env = self._make_env_ready(bricks_count=10)

        env._no_bricks_count = 2
        env._detector.detect_to_game_state.return_value = _detections(bricks=[])

        with mock.patch.object(env, "_handle_level_transition", return_value=True):
            _, reward, terminated, _, _ = env.step(_action())
            # Should have level clear bonus (positive)
            assert reward > 0

    @mock.patch("games.breakout71.env.time")
    def test_step_game_over_still_terminates(self, mock_time):
        """Game-over (ball lost) should still terminate the episode."""
        env = self._make_env_ready(bricks_count=10)

        # Ball lost for enough frames
        env._no_ball_count = env._BALL_LOST_THRESHOLD
        env._detector.detect_to_game_state.return_value = _detections(
            ball=None,
            bricks=[(0.1 * i, 0.1, 0.05, 0.03) for i in range(10)],
        )

        _, _, terminated, _, _ = env.step(_action())
        assert terminated is True


class TestScoreReward:
    """Tests for JS score-based reward computation."""

    def test_compute_reward_includes_score_delta(self):
        """compute_reward should include score delta component."""
        env = Breakout71Env()
        env._bricks_total = 10
        env._prev_bricks_norm = 1.0
        env._prev_score = 0

        dets = _detections(bricks=[(0.1 * i, 0.1, 0.05, 0.03) for i in range(10)])
        reward = env.compute_reward(dets, False, False, score=100)
        # Score delta should contribute positively
        # Without score: reward = 0 * 10 - 0.01 = -0.01
        # With score 100: reward = -0.01 + 100 * scale
        assert reward > -0.01

    def test_compute_reward_updates_prev_score(self):
        """compute_reward should update _prev_score for next step."""
        env = Breakout71Env()
        env._bricks_total = 10
        env._prev_bricks_norm = 1.0
        env._prev_score = 0

        dets = _detections(bricks=[(0.1 * i, 0.1, 0.05, 0.03) for i in range(10)])
        env.compute_reward(dets, False, False, score=100)
        assert env._prev_score == 100

    def test_compute_reward_level_cleared_bonus(self):
        """Level cleared should add bonus when not terminated."""
        env = Breakout71Env()
        env._bricks_total = 10
        env._prev_bricks_norm = 0.0
        env._prev_score = 0

        dets = _detections(bricks=[])
        # level_cleared=True, terminated=False (multi-level continues)
        reward = env.compute_reward(dets, False, True, score=0)
        # Should include level_cleared bonus
        assert reward > 0

    def test_compute_reward_game_over_penalty(self):
        """Game over should give negative terminal reward."""
        env = Breakout71Env()
        env._bricks_total = 10
        env._prev_bricks_norm = 1.0
        env._prev_score = 0

        dets = _detections(bricks=[(0.1 * i, 0.1, 0.05, 0.03) for i in range(10)])
        reward = env.compute_reward(dets, True, False, score=0)
        assert reward < 0


class TestBuildInfoMultiLevel:
    """Tests for build_info with multi-level state."""

    def test_build_info_includes_level_and_score(self):
        """build_info should include current_level and levels_cleared."""
        env = Breakout71Env()
        env._current_level = 2
        env._levels_cleared = 1
        env._prev_score = 50

        dets = _detections()
        info = env.build_info(dets)
        assert "current_level" in info
        assert info["current_level"] == 2
        assert "levels_cleared" in info
        assert info["levels_cleared"] == 1
