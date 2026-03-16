"""Tests for the LtcEvalWrapper.

Covers:
- Observation space is Box(0, 255, (1, 84, 84), uint8)
- Action space unchanged from wrapped env
- reset() returns (1, 84, 84) CHW observation
- step() returns (1, 84, 84) CHW observation
- Original MLP observation preserved in info["mlp_obs"]
- Reward, terminated, truncated passthrough
- Fallback to black frame when no frame in info
- Custom obs_size parameter
"""

import sys
from unittest import mock

import gymnasium as gym
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

# ---------------------------------------------------------------------------
# Mock cv2 for CI Docker (missing libGL.so.1)
# ---------------------------------------------------------------------------


def _build_cv2_mock() -> mock.MagicMock:
    """Create a mock cv2 with functional cvtColor and resize."""
    m = mock.MagicMock()
    m.COLOR_BGR2GRAY = 6
    m.INTER_AREA = 3

    def _fake_cvtColor(frame, code):
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
    """Inject mock cv2 so that _frame_to_obs works without libGL."""
    cv2_mock = _build_cv2_mock()
    monkeypatch.setitem(sys.modules, "cv2", cv2_mock)
    yield


# ---------------------------------------------------------------------------
# Helper: create a mock Breakout71Env with controlled frame output
# ---------------------------------------------------------------------------


class _FakeBreakout71Env(gym.Env):
    """Minimal mock of Breakout71Env for testing wrappers.

    Produces BGR frames of a configurable size with a known pattern.
    Extends ``gym.Env`` so that ``gym.Wrapper`` accepts it.
    """

    metadata: dict = {"render_modes": []}

    def __init__(self, frame_h: int = 600, frame_w: int = 400):
        super().__init__()
        self._frame_h = frame_h
        self._frame_w = frame_w
        from gymnasium import spaces

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.step_count = 0
        self._terminated = False

    def _make_frame(self) -> np.ndarray:
        """Create a BGR frame with a known pattern."""
        frame = np.zeros((self._frame_h, self._frame_w, 3), dtype=np.uint8)
        frame[:, :, 0] = 50  # B
        frame[:, :, 1] = 100  # G
        frame[:, :, 2] = 150  # R
        return frame

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.step_count = 0
        self._terminated = False
        mlp_obs = np.zeros(8, dtype=np.float32)
        info = {"frame": self._make_frame()}
        return mlp_obs, info

    def step(self, action):
        self.step_count += 1
        mlp_obs = np.ones(8, dtype=np.float32) * 0.5
        reward = 1.0
        terminated = self.step_count >= 5
        truncated = False
        info = {"frame": self._make_frame()}
        return mlp_obs, reward, terminated, truncated, info

    def render(self):
        pass


# ---------------------------------------------------------------------------
# Tests for LtcEvalWrapper
# ---------------------------------------------------------------------------


class TestLtcEvalWrapperConstruction:
    """Tests for LtcEvalWrapper construction and observation space."""

    def test_observation_space_shape(self):
        """Observation space is (1, 84, 84) CHW for single frame."""
        from src.platform.ltc_wrapper import LtcEvalWrapper

        env = _FakeBreakout71Env()
        wrapper = LtcEvalWrapper(env)
        assert wrapper.observation_space.shape == (1, 84, 84)

    def test_observation_space_dtype(self):
        """Observation space dtype is uint8."""
        from src.platform.ltc_wrapper import LtcEvalWrapper

        env = _FakeBreakout71Env()
        wrapper = LtcEvalWrapper(env)
        assert wrapper.observation_space.dtype == np.uint8

    def test_observation_space_bounds(self):
        """Observation space bounds are [0, 255]."""
        from src.platform.ltc_wrapper import LtcEvalWrapper

        env = _FakeBreakout71Env()
        wrapper = LtcEvalWrapper(env)
        assert wrapper.observation_space.low.min() == 0
        assert wrapper.observation_space.high.max() == 255

    def test_action_space_unchanged(self):
        """Action space is unchanged from the wrapped env."""
        from src.platform.ltc_wrapper import LtcEvalWrapper

        env = _FakeBreakout71Env()
        wrapper = LtcEvalWrapper(env)
        assert wrapper.action_space == env.action_space

    def test_custom_obs_size(self):
        """Custom obs_size changes observation shape."""
        from src.platform.ltc_wrapper import LtcEvalWrapper

        env = _FakeBreakout71Env()
        wrapper = LtcEvalWrapper(env, obs_size=64)
        assert wrapper.observation_space.shape == (1, 64, 64)


class TestLtcEvalWrapperReset:
    """Tests for LtcEvalWrapper.reset()."""

    def test_reset_returns_correct_shape(self):
        """reset() returns (1, 84, 84) CHW observation."""
        from src.platform.ltc_wrapper import LtcEvalWrapper

        env = _FakeBreakout71Env()
        wrapper = LtcEvalWrapper(env)
        obs, info = wrapper.reset()
        assert obs.shape == (1, 84, 84)

    def test_reset_returns_uint8(self):
        """reset() returns uint8 observation."""
        from src.platform.ltc_wrapper import LtcEvalWrapper

        env = _FakeBreakout71Env()
        wrapper = LtcEvalWrapper(env)
        obs, info = wrapper.reset()
        assert obs.dtype == np.uint8

    def test_reset_preserves_mlp_obs(self):
        """reset() preserves original MLP observation in info['mlp_obs']."""
        from src.platform.ltc_wrapper import LtcEvalWrapper

        env = _FakeBreakout71Env()
        wrapper = LtcEvalWrapper(env)
        obs, info = wrapper.reset()
        assert "mlp_obs" in info
        assert info["mlp_obs"].shape == (8,)

    def test_reset_non_zero_observation(self):
        """reset() returns non-zero observation from colored frame."""
        from src.platform.ltc_wrapper import LtcEvalWrapper

        env = _FakeBreakout71Env()
        wrapper = LtcEvalWrapper(env)
        obs, info = wrapper.reset()
        # The fake frame has non-zero BGR values, so grayscale should be non-zero
        assert obs.sum() > 0


class TestLtcEvalWrapperStep:
    """Tests for LtcEvalWrapper.step()."""

    def test_step_returns_correct_shape(self):
        """step() returns (1, 84, 84) CHW observation."""
        from src.platform.ltc_wrapper import LtcEvalWrapper

        env = _FakeBreakout71Env()
        wrapper = LtcEvalWrapper(env)
        wrapper.reset()
        obs, reward, terminated, truncated, info = wrapper.step(np.array([0.0]))
        assert obs.shape == (1, 84, 84)

    def test_step_returns_uint8(self):
        """step() returns uint8 observation."""
        from src.platform.ltc_wrapper import LtcEvalWrapper

        env = _FakeBreakout71Env()
        wrapper = LtcEvalWrapper(env)
        wrapper.reset()
        obs, reward, terminated, truncated, info = wrapper.step(np.array([0.0]))
        assert obs.dtype == np.uint8

    def test_step_reward_passthrough(self):
        """step() passes reward through unchanged."""
        from src.platform.ltc_wrapper import LtcEvalWrapper

        env = _FakeBreakout71Env()
        wrapper = LtcEvalWrapper(env)
        wrapper.reset()
        _, reward, _, _, _ = wrapper.step(np.array([0.0]))
        assert reward == 1.0

    def test_step_terminated_passthrough(self):
        """step() passes terminated through unchanged."""
        from src.platform.ltc_wrapper import LtcEvalWrapper

        env = _FakeBreakout71Env()
        wrapper = LtcEvalWrapper(env)
        wrapper.reset()
        # Step 5 times to trigger termination
        for _ in range(4):
            _, _, terminated, _, _ = wrapper.step(np.array([0.0]))
            assert not terminated
        _, _, terminated, _, _ = wrapper.step(np.array([0.0]))
        assert terminated

    def test_step_truncated_passthrough(self):
        """step() passes truncated through unchanged."""
        from src.platform.ltc_wrapper import LtcEvalWrapper

        env = _FakeBreakout71Env()
        wrapper = LtcEvalWrapper(env)
        wrapper.reset()
        _, _, _, truncated, _ = wrapper.step(np.array([0.0]))
        assert not truncated

    def test_step_preserves_mlp_obs(self):
        """step() preserves original MLP observation in info['mlp_obs']."""
        from src.platform.ltc_wrapper import LtcEvalWrapper

        env = _FakeBreakout71Env()
        wrapper = LtcEvalWrapper(env)
        wrapper.reset()
        _, _, _, _, info = wrapper.step(np.array([0.0]))
        assert "mlp_obs" in info
        assert info["mlp_obs"].shape == (8,)

    def test_step_no_frame_fallback(self):
        """step() produces black frame when no frame in info."""
        from src.platform.ltc_wrapper import LtcEvalWrapper

        env = _FakeBreakout71Env()
        wrapper = LtcEvalWrapper(env)
        wrapper.reset()

        # Patch step to return no frame
        original_step = env.step

        def _step_no_frame(action):
            obs, r, t, tr, info = original_step(action)
            del info["frame"]
            return obs, r, t, tr, info

        env.step = _step_no_frame

        obs, _, _, _, _ = wrapper.step(np.array([0.0]))
        assert obs.shape == (1, 84, 84)
        assert obs.sum() == 0  # All black

    def test_reset_no_frame_fallback(self):
        """reset() produces black frame when no frame in info."""
        from src.platform.ltc_wrapper import LtcEvalWrapper

        env = _FakeBreakout71Env()

        # Patch reset to return no frame
        original_reset = env.reset

        def _reset_no_frame(**kwargs):
            obs, info = original_reset(**kwargs)
            del info["frame"]
            return obs, info

        env.reset = _reset_no_frame

        wrapper = LtcEvalWrapper(env)
        obs, _ = wrapper.reset()
        assert obs.shape == (1, 84, 84)
        assert obs.sum() == 0  # All black
