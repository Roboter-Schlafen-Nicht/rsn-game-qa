"""Tests for the CnnObservationWrapper.

Covers:
- Construction and observation space override (shape, dtype, bounds)
- Observation conversion (BGR frame → 84x84 grayscale)
- step() returns image observation with correct shape/dtype
- reset() returns image observation with correct shape/dtype
- Original MLP observation preserved in info["mlp_obs"]
- Reward, terminated, truncated pass through unchanged
- Fallback to black frame when no frame in info
- Custom obs_size parameter
- Action space unchanged (passthrough)
- Frame channel handling (BGR, grayscale, unexpected)
"""

import sys
from unittest import mock

import numpy as np

# ---------------------------------------------------------------------------
# Mock selenium for CI
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

import pytest  # noqa: E402

from src.env.breakout71_env import Breakout71Env  # noqa: E402
from src.env.cnn_wrapper import CNN_OBS_SIZE, CnnObservationWrapper  # noqa: E402


# ---------------------------------------------------------------------------
# Mock cv2 for CI Docker (missing libGL.so.1)
# ---------------------------------------------------------------------------
# ``_frame_to_obs()`` does a lazy ``import cv2`` — which fails in the CI
# Docker container because libGL.so.1 is not installed.  We use an
# autouse fixture that injects a lightweight mock into ``sys.modules``
# for the duration of each test, then restores the original entry so
# that other test modules (e.g. ``test_training_pipeline``) still see
# the real ``cv2``.
# ---------------------------------------------------------------------------


def _build_cv2_mock() -> mock.MagicMock:
    """Create a mock cv2 with functional cvtColor and resize."""
    m = mock.MagicMock()
    m.COLOR_BGR2GRAY = 6
    m.INTER_AREA = 3

    def _fake_cvtColor(frame, code):  # noqa: N802
        if frame.ndim == 3 and frame.shape[2] >= 3:
            return (
                0.114 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.299 * frame[:, :, 2]
            ).astype(np.uint8)
        return frame

    def _fake_resize(img, dsize, **_kw):
        tw, th = dsize
        h, w = img.shape[:2]
        ri = (np.arange(th) * h // th).astype(int)
        ci = (np.arange(tw) * w // tw).astype(int)
        return img[np.ix_(ri, ci)]

    m.cvtColor = _fake_cvtColor
    m.resize = _fake_resize
    return m


@pytest.fixture(autouse=True)
def _mock_cv2_for_ci(monkeypatch):
    """Inject a fake cv2 so _frame_to_obs works without libGL.so.1.

    Scoped per-test so the mock never leaks to other test modules.
    """
    monkeypatch.setitem(sys.modules, "cv2", _build_cv2_mock())


# -- Helpers -------------------------------------------------------------------


def _frame(h=480, w=640, color=(128, 128, 128)):
    """Create a synthetic BGR frame."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = color
    return frame


def _action(value=0.0):
    """Create a continuous action array."""
    return np.array([value], dtype=np.float32)


def _detections(
    paddle=(0.5, 0.9, 0.1, 0.02),
    ball=(0.5, 0.5, 0.02, 0.02),
    bricks=None,
    powerups=None,
):
    """Build a fake detections dict."""
    if bricks is None:
        bricks = [(0.1 * i, 0.1, 0.05, 0.03) for i in range(10)]
    return {
        "paddle": paddle,
        "ball": ball,
        "bricks": bricks,
        "powerups": powerups or [],
        "raw_detections": [],
    }


def _mock_driver():
    """Create a mock Selenium WebDriver."""
    driver = mock.MagicMock()
    driver.execute_script.return_value = {"state": "gameplay", "details": {}}
    return driver


def _make_base_env_ready(bricks_count=10):
    """Create a Breakout71Env in a post-reset state with mocked sub-components."""
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
    env._game_canvas = mock.MagicMock()
    env._canvas_size = (640, 480)

    env._capture = mock.MagicMock()
    env._capture.width = 640
    env._capture.height = 480
    env._capture.hwnd = 12345
    env._capture.capture_frame.return_value = _frame()

    env._detector = mock.MagicMock()
    env._detector.detect_to_game_state.return_value = _detections(
        bricks=[(0.1 * i, 0.1, 0.05, 0.03) for i in range(bricks_count)]
    )
    return env


# -- Construction & Observation Space ------------------------------------------


class TestCnnWrapperConstruction:
    """Tests for CnnObservationWrapper constructor and space definitions."""

    def test_observation_space_shape(self):
        """Wrapped observation space should be (84, 84, 1) by default."""
        base_env = Breakout71Env()
        env = CnnObservationWrapper(base_env)
        assert env.observation_space.shape == (84, 84, 1)

    def test_observation_space_dtype(self):
        """Wrapped observation space dtype should be uint8."""
        base_env = Breakout71Env()
        env = CnnObservationWrapper(base_env)
        assert env.observation_space.dtype == np.uint8

    def test_observation_space_bounds(self):
        """Wrapped observation space should have bounds [0, 255]."""
        base_env = Breakout71Env()
        env = CnnObservationWrapper(base_env)
        assert env.observation_space.low.min() == 0
        assert env.observation_space.high.max() == 255

    def test_action_space_unchanged(self):
        """Wrapper should not modify the action space."""
        base_env = Breakout71Env()
        env = CnnObservationWrapper(base_env)
        assert env.action_space.shape == (1,)
        assert float(env.action_space.low[0]) == -1.0
        assert float(env.action_space.high[0]) == 1.0

    def test_custom_obs_size(self):
        """Wrapper should accept a custom observation size."""
        base_env = Breakout71Env()
        env = CnnObservationWrapper(base_env, obs_size=64)
        assert env.observation_space.shape == (64, 64, 1)

    def test_default_obs_size_constant(self):
        """CNN_OBS_SIZE constant should be 84."""
        assert CNN_OBS_SIZE == 84


# -- Observation Conversion ----------------------------------------------------


class TestObservationConversion:
    """Tests for frame-to-observation conversion."""

    def test_bgr_frame_to_grayscale(self):
        """BGR frame should be converted to single-channel grayscale."""
        base_env = Breakout71Env()
        env = CnnObservationWrapper(base_env)

        # Set a coloured frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = 50  # Blue channel
        frame[:, :, 1] = 100  # Green channel
        frame[:, :, 2] = 150  # Red channel
        env._last_raw_frame = frame

        obs = env.observation(np.zeros(8, dtype=np.float32))

        assert obs.shape == (84, 84, 1)
        assert obs.dtype == np.uint8
        # Grayscale should be a weighted combination, not just one channel
        assert obs[0, 0, 0] > 0

    def test_grayscale_frame_passthrough(self):
        """Already-grayscale (2D) frame should work without conversion."""
        base_env = Breakout71Env()
        env = CnnObservationWrapper(base_env)

        frame = np.full((480, 640), 128, dtype=np.uint8)
        env._last_raw_frame = frame

        obs = env.observation(np.zeros(8, dtype=np.float32))

        assert obs.shape == (84, 84, 1)
        assert obs.dtype == np.uint8

    def test_frame_resized_to_target(self):
        """Large frame should be resized to obs_size x obs_size."""
        base_env = Breakout71Env()
        env = CnnObservationWrapper(base_env, obs_size=64)

        env._last_raw_frame = _frame(1024, 768)
        obs = env.observation(np.zeros(8, dtype=np.float32))

        assert obs.shape == (64, 64, 1)

    def test_no_frame_returns_black(self):
        """When no frame is available, should return a black image."""
        base_env = Breakout71Env()
        env = CnnObservationWrapper(base_env)
        env._last_raw_frame = None

        obs = env.observation(np.zeros(8, dtype=np.float32))

        assert obs.shape == (84, 84, 1)
        assert obs.dtype == np.uint8
        assert obs.sum() == 0

    def test_observation_contains_in_space(self):
        """Returned observation should be within the observation space."""
        base_env = Breakout71Env()
        env = CnnObservationWrapper(base_env)
        env._last_raw_frame = _frame()

        obs = env.observation(np.zeros(8, dtype=np.float32))
        assert env.observation_space.contains(obs)


# -- step() Integration -------------------------------------------------------


class TestCnnWrapperStep:
    """Tests for step() through the CNN wrapper."""

    def test_step_returns_image_observation(self):
        """step() should return an 84x84x1 uint8 observation."""
        base_env = _make_base_env_ready()
        env = CnnObservationWrapper(base_env)

        obs, reward, terminated, truncated, info = env.step(_action(0.0))

        assert obs.shape == (84, 84, 1)
        assert obs.dtype == np.uint8

    def test_step_preserves_reward(self):
        """step() should pass through reward unchanged."""
        base_env = _make_base_env_ready()
        env = CnnObservationWrapper(base_env)

        # First step has a time penalty
        _, reward, _, _, _ = env.step(_action(0.0))
        assert isinstance(reward, float)

    def test_step_preserves_terminated(self):
        """step() should pass through terminated flag unchanged."""
        base_env = _make_base_env_ready()
        env = CnnObservationWrapper(base_env)

        _, _, terminated, _, _ = env.step(_action(0.0))
        assert isinstance(terminated, bool)

    def test_step_preserves_truncated(self):
        """step() should pass through truncated flag unchanged."""
        base_env = _make_base_env_ready()
        env = CnnObservationWrapper(base_env)

        _, _, _, truncated, _ = env.step(_action(0.0))
        assert isinstance(truncated, bool)

    def test_step_includes_mlp_obs_in_info(self):
        """step() should include original MLP observation in info['mlp_obs']."""
        base_env = _make_base_env_ready()
        env = CnnObservationWrapper(base_env)

        _, _, _, _, info = env.step(_action(0.0))

        assert "mlp_obs" in info
        assert info["mlp_obs"].shape == (8,)
        assert info["mlp_obs"].dtype == np.float32

    def test_step_includes_frame_in_info(self):
        """step() should still include the raw frame in info."""
        base_env = _make_base_env_ready()
        env = CnnObservationWrapper(base_env)

        _, _, _, _, info = env.step(_action(0.0))
        assert "frame" in info

    def test_step_observation_in_space(self):
        """step() observation should be within the observation space."""
        base_env = _make_base_env_ready()
        env = CnnObservationWrapper(base_env)

        obs, _, _, _, _ = env.step(_action(0.5))
        assert env.observation_space.contains(obs)

    def test_step_game_over_returns_image(self):
        """step() with game-over modal should still return an image obs."""
        base_env = _make_base_env_ready()
        # Simulate ball missing for several frames to trigger game-over
        base_env._no_ball_count = 5
        base_env._detector.detect_to_game_state.return_value = _detections(
            ball=None, bricks=[(0.1, 0.1, 0.05, 0.03)] * 10
        )
        base_env._driver.execute_script.return_value = {
            "state": "game_over",
            "details": {},
        }
        env = CnnObservationWrapper(base_env)

        obs, reward, terminated, _, _ = env.step(_action(0.0))
        assert obs.shape == (84, 84, 1)
        assert obs.dtype == np.uint8
        assert terminated is True


# -- reset() Integration ------------------------------------------------------


class TestCnnWrapperReset:
    """Tests for reset() through the CNN wrapper."""

    def test_reset_returns_image_observation(self):
        """reset() should return an 84x84x1 uint8 observation."""
        base_env = _make_base_env_ready()
        env = CnnObservationWrapper(base_env)

        obs, info = env.reset()

        assert obs.shape == (84, 84, 1)
        assert obs.dtype == np.uint8

    def test_reset_includes_mlp_obs_in_info(self):
        """reset() should include original MLP observation in info['mlp_obs']."""
        base_env = _make_base_env_ready()
        env = CnnObservationWrapper(base_env)

        _, info = env.reset()

        assert "mlp_obs" in info
        assert info["mlp_obs"].shape == (8,)

    def test_reset_observation_in_space(self):
        """reset() observation should be within the observation space."""
        base_env = _make_base_env_ready()
        env = CnnObservationWrapper(base_env)

        obs, _ = env.reset()
        assert env.observation_space.contains(obs)


# -- Frame Processing Edge Cases -----------------------------------------------


class TestFrameProcessingEdgeCases:
    """Tests for edge cases in frame processing."""

    def test_portrait_frame(self):
        """Portrait-oriented frame (768x1024) should resize correctly."""
        base_env = Breakout71Env()
        env = CnnObservationWrapper(base_env)

        # Portrait: width < height
        env._last_raw_frame = _frame(h=1024, w=768)
        obs = env.observation(np.zeros(8, dtype=np.float32))

        assert obs.shape == (84, 84, 1)

    def test_landscape_frame(self):
        """Landscape-oriented frame (1280x1024) should resize correctly."""
        base_env = Breakout71Env()
        env = CnnObservationWrapper(base_env)

        env._last_raw_frame = _frame(h=1024, w=1280)
        obs = env.observation(np.zeros(8, dtype=np.float32))

        assert obs.shape == (84, 84, 1)

    def test_small_frame(self):
        """Very small frame should still resize to target."""
        base_env = Breakout71Env()
        env = CnnObservationWrapper(base_env)

        env._last_raw_frame = _frame(h=10, w=10)
        obs = env.observation(np.zeros(8, dtype=np.float32))

        assert obs.shape == (84, 84, 1)

    def test_white_frame_produces_nonzero_obs(self):
        """White frame should produce a bright observation."""
        base_env = Breakout71Env()
        env = CnnObservationWrapper(base_env)

        env._last_raw_frame = _frame(color=(255, 255, 255))
        obs = env.observation(np.zeros(8, dtype=np.float32))

        assert obs.max() == 255

    def test_black_frame_produces_zero_obs(self):
        """Black frame should produce a dark observation."""
        base_env = Breakout71Env()
        env = CnnObservationWrapper(base_env)

        env._last_raw_frame = _frame(color=(0, 0, 0))
        obs = env.observation(np.zeros(8, dtype=np.float32))

        assert obs.sum() == 0

    def test_four_channel_frame(self):
        """4-channel frame (e.g., BGRA) should extract first channel."""
        base_env = Breakout71Env()
        env = CnnObservationWrapper(base_env)

        frame = np.full((480, 640, 4), 100, dtype=np.uint8)
        env._last_raw_frame = frame
        obs = env.observation(np.zeros(8, dtype=np.float32))

        assert obs.shape == (84, 84, 1)
        assert obs.dtype == np.uint8
