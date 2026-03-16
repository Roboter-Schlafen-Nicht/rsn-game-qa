"""Tests for LTC (CfC) integration into train_rl.py and session_runner.py.

Covers:
- parse_args accepts --policy ltc
- parse_args rejects invalid --policy values
- SessionRunner accepts policy="ltc"
- SessionRunner rejects invalid policy values
- SessionRunner _wrap_env_for_cnn uses LtcEvalWrapper for policy="ltc"
- LTC training pipeline builds correct env stack (no VecFrameStack)
- RecurrentPPO is used instead of PPO for LTC policy
"""

import sys
from unittest import mock

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Mock heavy dependencies that are unavailable in CI
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
# Helper: minimal game-like environment
# ---------------------------------------------------------------------------


class _FakeGameEnv(gym.Env):
    """Minimal gym env with MLP-like obs and a 'frame' in info.

    Extends ``gym.Env`` for DummyVecEnv compatibility.
    """

    metadata: dict = {"render_modes": []}

    def __init__(self, **kwargs):
        super().__init__()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._step = 0
        self.step_count = 0
        self._oracles = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._step = 0
        self.step_count = 0
        obs = np.zeros(8, dtype=np.float32)
        frame = np.zeros((600, 400, 3), dtype=np.uint8)
        return obs, {"frame": frame}

    def step(self, action):
        self._step += 1
        self.step_count = self._step
        obs = np.ones(8, dtype=np.float32) * 0.5
        reward = 0.0
        terminated = self._step >= 100
        truncated = False
        frame = np.zeros((600, 400, 3), dtype=np.uint8)
        return obs, reward, terminated, truncated, {"frame": frame}

    def render(self):
        pass


# ---------------------------------------------------------------------------
# Tests for parse_args (train_rl.py)
# ---------------------------------------------------------------------------


class TestParseArgsLtcPolicy:
    """Tests for LTC policy argument parsing in train_rl.py."""

    def test_ltc_policy_accepted(self):
        """parse_args accepts --policy ltc."""
        from scripts.train_rl import parse_args

        args = parse_args(["--policy", "ltc"])
        assert args.policy == "ltc"

    def test_mlp_policy_still_works(self):
        """parse_args still accepts --policy mlp (regression)."""
        from scripts.train_rl import parse_args

        args = parse_args(["--policy", "mlp"])
        assert args.policy == "mlp"

    def test_cnn_policy_still_works(self):
        """parse_args still accepts --policy cnn (regression)."""
        from scripts.train_rl import parse_args

        args = parse_args(["--policy", "cnn"])
        assert args.policy == "cnn"

    def test_invalid_policy_rejected(self):
        """parse_args rejects unknown --policy values."""
        from scripts.train_rl import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--policy", "invalid"])

    def test_ltc_default_frame_stack_ignored(self):
        """--frame-stack is irrelevant for LTC (no VecFrameStack)."""
        from scripts.train_rl import parse_args

        args = parse_args(["--policy", "ltc"])
        # frame_stack default exists but won't be used for LTC pipeline
        assert hasattr(args, "frame_stack")


# ---------------------------------------------------------------------------
# Tests for SessionRunner with policy="ltc"
# ---------------------------------------------------------------------------


class TestSessionRunnerLtcInit:
    """Tests for SessionRunner.__init__ with policy='ltc'."""

    def test_ltc_policy_accepted(self):
        """SessionRunner accepts policy='ltc'."""
        from src.orchestrator.session_runner import SessionRunner

        runner = SessionRunner(policy="ltc")
        assert runner.policy == "ltc"

    def test_invalid_policy_rejected(self):
        """SessionRunner rejects unknown policy values."""
        from src.orchestrator.session_runner import SessionRunner

        with pytest.raises(ValueError, match="policy must be one of"):
            SessionRunner(policy="invalid")


# ---------------------------------------------------------------------------
# Tests for _wrap_env_for_cnn with LTC
# ---------------------------------------------------------------------------


class TestSessionRunnerLtcWrap:
    """Tests for SessionRunner._wrap_env_for_cnn with policy='ltc'."""

    def _build_runner_with_env(self, policy: str):  # -> SessionRunner
        """Build a SessionRunner with a fake env for wrapping tests."""
        from src.orchestrator.session_runner import SessionRunner

        runner = SessionRunner(policy=policy)
        runner._env = _FakeGameEnv()
        return runner

    def test_ltc_wraps_with_ltc_eval_wrapper(self):
        """_wrap_env_for_cnn uses LtcEvalWrapper when policy='ltc'."""
        from src.platform.ltc_wrapper import LtcEvalWrapper

        runner = self._build_runner_with_env("ltc")
        runner._wrap_env_for_cnn()

        assert isinstance(runner._env, LtcEvalWrapper)
        assert runner._raw_env is not None

    def test_ltc_observation_space_shape(self):
        """LTC-wrapped env has (1, 84, 84) observation space."""
        runner = self._build_runner_with_env("ltc")
        runner._wrap_env_for_cnn()

        assert runner._env.observation_space.shape == (1, 84, 84)

    def test_cnn_still_wraps_with_cnn_eval_wrapper(self):
        """_wrap_env_for_cnn still uses CnnEvalWrapper when policy='cnn' (regression)."""
        # Mock cv2 for CI
        cv2_mock = mock.MagicMock()
        cv2_mock.COLOR_BGR2GRAY = 6
        cv2_mock.INTER_AREA = 3
        cv2_mock.cvtColor = lambda f, c: np.mean(f, axis=2).astype(np.uint8)
        cv2_mock.resize = lambda img, dsize, **kw: np.zeros((dsize[1], dsize[0]), dtype=np.uint8)
        with mock.patch.dict(sys.modules, {"cv2": cv2_mock}):
            from src.platform.cnn_wrapper import CnnEvalWrapper

            runner = self._build_runner_with_env("cnn")
            runner._wrap_env_for_cnn()

            assert isinstance(runner._env, CnnEvalWrapper)

    def test_mlp_no_wrapping(self):
        """_wrap_env_for_cnn is a no-op when policy='mlp'."""
        runner = self._build_runner_with_env("mlp")
        original_env = runner._env
        runner._wrap_env_for_cnn()

        assert runner._env is original_env
        assert runner._raw_env is None


# ---------------------------------------------------------------------------
# Tests for LTC training pipeline construction
# ---------------------------------------------------------------------------


class TestLtcTrainingPipeline:
    """Tests for the LTC training pipeline (env wrapping for train_rl.py).

    These verify that the correct env stack is built for LTC:
    - No VecFrameStack (temporal context in CfC hidden state)
    - Single-frame CHW observation
    - RecurrentPPO instead of PPO
    """

    def test_ltc_vec_env_no_framestack(self):
        """LTC pipeline does not use VecFrameStack."""
        from stable_baselines3.common.vec_env import DummyVecEnv

        from src.platform.ltc_wrapper import LtcEvalWrapper

        # Build pipeline: LtcEvalWrapper → DummyVecEnv (no VecTransposeImage
        # because LtcEvalWrapper already outputs CHW)
        fake_env = _FakeGameEnv()
        ltc_env = LtcEvalWrapper(fake_env)
        vec_env = DummyVecEnv([lambda: ltc_env])

        # Obs shape should be (1, 84, 84) — NOT (4, 84, 84) from frame stacking
        assert vec_env.observation_space.shape == (1, 84, 84)
        vec_env.close()

    def test_ltc_recurrent_ppo_construction(self):
        """RecurrentPPO constructs correctly with CnnCfCPolicy and LTC pipeline."""
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        from src.platform.ltc_policy import CnnCfCPolicy
        from src.platform.ltc_wrapper import LtcEvalWrapper

        fake_env = _FakeGameEnv()
        ltc_env = LtcEvalWrapper(fake_env)
        vec_env = DummyVecEnv([lambda: ltc_env])

        model = RecurrentPPO(
            CnnCfCPolicy,
            vec_env,
            n_steps=16,
            batch_size=16,
            policy_kwargs={"lstm_hidden_size": 32},
            device="cpu",
            verbose=0,
        )
        assert model is not None
        vec_env.close()

    def test_ltc_learn_with_game_env(self):
        """RecurrentPPO.learn() runs on LTC pipeline with game-like env."""
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        from src.platform.ltc_policy import CnnCfCPolicy
        from src.platform.ltc_wrapper import LtcEvalWrapper

        fake_env = _FakeGameEnv()
        ltc_env = LtcEvalWrapper(fake_env)
        vec_env = DummyVecEnv([lambda: ltc_env])

        model = RecurrentPPO(
            CnnCfCPolicy,
            vec_env,
            n_steps=32,
            batch_size=32,
            n_epochs=1,
            policy_kwargs={"lstm_hidden_size": 32},
            device="cpu",
            verbose=0,
        )
        model.learn(total_timesteps=64)
        vec_env.close()
