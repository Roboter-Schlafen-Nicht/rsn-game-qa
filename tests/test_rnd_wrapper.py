"""Tests for the RND (Random Network Distillation) reward wrapper.

TDD tests for ``src/platform/rnd_wrapper.py``.

Covers:
- RND target/predictor network architecture (output dims, fixed vs trainable)
- Observation normalisation (running mean/std, clipping)
- Intrinsic reward computation (MSE between target and predictor embeddings)
- Reward normalisation (running variance, no mean subtraction)
- VecEnvWrapper integration (step_wait modifies reward, obs/done/info pass through)
- Predictor network training (loss decreases on repeated observations)
- Non-episodic intrinsic reward (not reset on done=True)
- Hyperparameter configuration (int_coeff, ext_coeff, embedding_dim, etc.)
- Device handling (CPU fallback)
- State dict save/load for checkpointing
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

# Lazy-import torch to match CI pattern (torch may not be installed in all envs)
torch = pytest.importorskip("torch")

from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: simple image env for testing
# ---------------------------------------------------------------------------
class _DummyImageEnv(gym.Env):
    """Minimal gym env that returns CHW float32 observations (default 1x84x84)."""

    def __init__(self, obs_shape: tuple[int, ...] = (1, 84, 84)):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        self._step_count = 0
        obs = self.observation_space.sample()
        return obs, {}

    def step(self, action):
        self._step_count += 1
        obs = self.observation_space.sample()
        reward = 0.01  # survival reward
        terminated = self._step_count >= 100
        truncated = False
        return obs, reward, terminated, truncated, {}


def _make_vec_env(
    n_envs: int = 1,
    obs_shape: tuple[int, ...] = (1, 84, 84),
) -> DummyVecEnv:
    """Create a DummyVecEnv with image observations (CHW format, post-transpose)."""
    return DummyVecEnv([lambda: _DummyImageEnv(obs_shape) for _ in range(n_envs)])


# ---------------------------------------------------------------------------
# Fixture: import RND wrapper (deferred so tests fail clearly if missing)
# ---------------------------------------------------------------------------
@pytest.fixture
def rnd_module():
    """Import and return the rnd_wrapper module."""
    from src.platform import rnd_wrapper

    return rnd_wrapper


@pytest.fixture
def RNDRewardWrapper(rnd_module):
    """Return the RNDRewardWrapper class."""
    return rnd_module.RNDRewardWrapper


@pytest.fixture
def RNDTargetNetwork(rnd_module):
    """Return the RNDTargetNetwork class."""
    return rnd_module.RNDTargetNetwork


@pytest.fixture
def RNDPredictorNetwork(rnd_module):
    """Return the RNDPredictorNetwork class."""
    return rnd_module.RNDPredictorNetwork


@pytest.fixture
def RunningMeanStd(rnd_module):
    """Return the RunningMeanStd class."""
    return rnd_module.RunningMeanStd


# ===========================================================================
# Target Network Tests
# ===========================================================================
class TestRNDTargetNetwork:
    """Target network: fixed random CNN -> 512-dim embedding."""

    def test_output_shape_is_embedding_dim(self, RNDTargetNetwork):
        """Target network output has shape (batch, embedding_dim)."""
        net = RNDTargetNetwork(obs_shape=(1, 84, 84), embedding_dim=512)
        x = torch.randn(4, 1, 84, 84)
        out = net(x)
        assert out.shape == (4, 512)

    def test_custom_embedding_dim(self, RNDTargetNetwork):
        """Embedding dim is configurable."""
        net = RNDTargetNetwork(obs_shape=(1, 84, 84), embedding_dim=256)
        x = torch.randn(2, 1, 84, 84)
        out = net(x)
        assert out.shape == (2, 256)

    def test_parameters_are_frozen(self, RNDTargetNetwork):
        """Target network parameters should not require gradients."""
        net = RNDTargetNetwork(obs_shape=(1, 84, 84), embedding_dim=512)
        for param in net.parameters():
            assert not param.requires_grad

    def test_output_is_deterministic(self, RNDTargetNetwork):
        """Same input produces same output (no dropout, no stochasticity)."""
        net = RNDTargetNetwork(obs_shape=(1, 84, 84), embedding_dim=512)
        net.eval()
        x = torch.randn(2, 1, 84, 84)
        out1 = net(x)
        out2 = net(x)
        torch.testing.assert_close(out1, out2)

    def test_architecture_has_three_conv_layers(self, RNDTargetNetwork):
        """Target has Conv2d -> Conv2d -> Conv2d -> Flatten -> Linear."""
        net = RNDTargetNetwork(obs_shape=(1, 84, 84), embedding_dim=512)
        conv_count = sum(1 for m in net.modules() if isinstance(m, torch.nn.Conv2d))
        assert conv_count == 3

    def test_accepts_multi_channel_input(self, RNDTargetNetwork):
        """Target works with multi-channel observations (e.g. 4-frame stack)."""
        net = RNDTargetNetwork(obs_shape=(4, 84, 84), embedding_dim=512)
        x = torch.randn(2, 4, 84, 84)
        out = net(x)
        assert out.shape == (2, 512)


# ===========================================================================
# Predictor Network Tests
# ===========================================================================
class TestRNDPredictorNetwork:
    """Predictor network: same conv backbone + deeper FC head."""

    def test_output_shape_matches_target(self, RNDPredictorNetwork):
        """Predictor output has same shape as target (batch, embedding_dim)."""
        net = RNDPredictorNetwork(obs_shape=(1, 84, 84), embedding_dim=512)
        x = torch.randn(4, 1, 84, 84)
        out = net(x)
        assert out.shape == (4, 512)

    def test_parameters_are_trainable(self, RNDPredictorNetwork):
        """Predictor network parameters require gradients."""
        net = RNDPredictorNetwork(obs_shape=(1, 84, 84), embedding_dim=512)
        trainable = [p for p in net.parameters() if p.requires_grad]
        assert len(trainable) > 0

    def test_has_deeper_head_than_target(self, RNDPredictorNetwork, RNDTargetNetwork):
        """Predictor has more parameters than target (deeper FC head)."""
        target = RNDTargetNetwork(obs_shape=(1, 84, 84), embedding_dim=512)
        predictor = RNDPredictorNetwork(obs_shape=(1, 84, 84), embedding_dim=512)
        target_params = sum(p.numel() for p in target.parameters())
        predictor_params = sum(p.numel() for p in predictor.parameters())
        assert predictor_params > target_params

    def test_gradient_flows(self, RNDPredictorNetwork):
        """Gradients flow through the predictor network."""
        net = RNDPredictorNetwork(obs_shape=(1, 84, 84), embedding_dim=512)
        x = torch.randn(2, 1, 84, 84)
        out = net(x)
        loss = out.sum()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())
        assert has_grad


# ===========================================================================
# Running Mean/Std Tests
# ===========================================================================
class TestRunningMeanStd:
    """Running mean/std tracker for observation and reward normalisation."""

    def test_initial_mean_is_zero(self, RunningMeanStd):
        """Initial mean should be zero."""
        rms = RunningMeanStd(shape=(84, 84))
        np.testing.assert_array_equal(rms.mean, np.zeros((84, 84)))

    def test_initial_var_is_one(self, RunningMeanStd):
        """Initial variance should be one."""
        rms = RunningMeanStd(shape=(84, 84))
        np.testing.assert_array_equal(rms.var, np.ones((84, 84)))

    def test_update_changes_mean(self, RunningMeanStd):
        """After update, mean should reflect the data."""
        rms = RunningMeanStd(shape=(3,))
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        rms.update(data)
        assert rms.count == 2
        np.testing.assert_allclose(rms.mean, [2.5, 3.5, 4.5])

    def test_update_changes_var(self, RunningMeanStd):
        """After update, variance should reflect the data."""
        rms = RunningMeanStd(shape=(1,))
        data = np.array([[10.0], [20.0], [30.0], [40.0]])
        rms.update(data)
        # Var should be population variance
        expected_var = np.var(data, axis=0)
        np.testing.assert_allclose(rms.var, expected_var, rtol=1e-5)

    def test_scalar_shape(self, RunningMeanStd):
        """Works with scalar (empty tuple) shape for reward normalisation."""
        rms = RunningMeanStd(shape=())
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rms.update(data)
        assert rms.count == 5
        np.testing.assert_allclose(rms.mean, 3.0)


# ===========================================================================
# RND Reward Wrapper Tests
# ===========================================================================
class TestRNDRewardWrapper:
    """VecEnvWrapper that adds RND intrinsic reward to environment rewards."""

    def test_construction_preserves_spaces(self, RNDRewardWrapper):
        """Wrapper preserves observation and action spaces from wrapped env."""
        venv = _make_vec_env()
        wrapper = RNDRewardWrapper(venv)
        assert wrapper.observation_space == venv.observation_space
        assert wrapper.action_space == venv.action_space

    def test_step_wait_returns_modified_reward(self, RNDRewardWrapper):
        """step_wait adds intrinsic reward to extrinsic reward."""
        venv = _make_vec_env()
        wrapper = RNDRewardWrapper(venv)
        obs = wrapper.reset()
        wrapper.step_async(np.array([[0.0]]))
        obs, reward, done, info = wrapper.step_wait()
        # Reward should differ from pure extrinsic (0.01) because
        # intrinsic bonus is added
        # With ext_coeff=2.0, extrinsic part = 0.02
        # Total = 0.02 + int_coeff * normalized_intrinsic
        assert reward.shape == (1,)
        # Just check it's a valid float, not NaN
        assert np.isfinite(reward[0])

    def test_obs_pass_through_unchanged(self, RNDRewardWrapper):
        """Observations are not modified by the RND wrapper."""
        venv = _make_vec_env()
        wrapper = RNDRewardWrapper(venv)
        obs = wrapper.reset()
        assert obs.shape == (1, 1, 84, 84)

    def test_done_pass_through_unchanged(self, RNDRewardWrapper):
        """Done flags are not modified by the RND wrapper."""
        venv = _make_vec_env()
        wrapper = RNDRewardWrapper(venv)
        wrapper.reset()
        wrapper.step_async(np.array([[0.0]]))
        _, _, done, _ = wrapper.step_wait()
        assert done.dtype == bool

    def test_info_contains_intrinsic_reward(self, RNDRewardWrapper):
        """Info dict should contain the raw intrinsic reward for logging."""
        venv = _make_vec_env()
        wrapper = RNDRewardWrapper(venv)
        wrapper.reset()
        wrapper.step_async(np.array([[0.0]]))
        _, _, _, infos = wrapper.step_wait()
        assert "rnd_intrinsic_reward" in infos[0]
        assert np.isfinite(infos[0]["rnd_intrinsic_reward"])

    def test_custom_coefficients(self, RNDRewardWrapper):
        """int_coeff and ext_coeff are configurable."""
        venv = _make_vec_env()
        wrapper = RNDRewardWrapper(venv, int_coeff=0.5, ext_coeff=1.0)
        assert wrapper.int_coeff == 0.5
        assert wrapper.ext_coeff == 1.0

    def test_default_coefficients(self, RNDRewardWrapper):
        """Default coefficients match spec: int_coeff=1.0, ext_coeff=2.0."""
        venv = _make_vec_env()
        wrapper = RNDRewardWrapper(venv)
        assert wrapper.int_coeff == 1.0
        assert wrapper.ext_coeff == 2.0

    def test_multi_env(self, RNDRewardWrapper):
        """Works with multiple vectorised environments."""
        venv = _make_vec_env(n_envs=3)
        wrapper = RNDRewardWrapper(venv)
        wrapper.reset()
        wrapper.step_async(np.array([[0.0], [0.5], [-0.5]]))
        obs, reward, done, info = wrapper.step_wait()
        assert obs.shape == (3, 1, 84, 84)
        assert reward.shape == (3,)
        assert done.shape == (3,)

    def test_intrinsic_reward_is_positive(self, RNDRewardWrapper):
        """Intrinsic reward (MSE) is non-negative."""
        venv = _make_vec_env()
        wrapper = RNDRewardWrapper(venv)
        wrapper.reset()
        for _ in range(5):
            wrapper.step_async(np.array([[0.0]]))
            _, _, _, infos = wrapper.step_wait()
            assert infos[0]["rnd_intrinsic_reward"] >= 0.0

    def test_predictor_loss_decreases_on_repeated_obs(self, RNDRewardWrapper):
        """Predictor should learn to match target on repeated observations.

        When the same observation is seen many times, the prediction error
        (intrinsic reward) should decrease on average, demonstrating that
        the predictor is learning.
        """
        venv = _make_vec_env()
        wrapper = RNDRewardWrapper(venv, predictor_lr=1e-2)
        wrapper.reset()

        # Use a fixed observation
        fixed_obs = np.random.RandomState(42).randn(1, 1, 84, 84).astype(np.float32)

        intrinsic_rewards = []
        for _ in range(100):
            # Compute intrinsic reward for the fixed observation
            reward = wrapper._compute_intrinsic_reward(fixed_obs)
            intrinsic_rewards.append(reward[0])
            # Update predictor on this observation
            wrapper._update_predictor(fixed_obs)

        # Average of first 10 should be higher than average of last 10
        first_avg = np.mean(intrinsic_rewards[:10])
        last_avg = np.mean(intrinsic_rewards[-10:])
        assert first_avg > last_avg, (
            f"Predictor did not learn: first_avg={first_avg:.6f}, last_avg={last_avg:.6f}"
        )

    def test_novel_obs_has_higher_intrinsic_reward(self, RNDRewardWrapper):
        """Novel observations should have higher intrinsic reward than familiar ones.

        After training the predictor on a set of observations, a new
        observation should produce higher prediction error.
        """
        venv = _make_vec_env()
        wrapper = RNDRewardWrapper(venv, predictor_lr=1e-2)
        wrapper.reset()

        # Train on a fixed observation
        familiar_obs = np.random.RandomState(42).randn(1, 1, 84, 84).astype(np.float32)
        for _ in range(200):
            wrapper._update_predictor(familiar_obs)

        familiar_reward = wrapper._compute_intrinsic_reward(familiar_obs)[0]

        # Novel observations should have higher intrinsic reward on average
        novel_rewards = []
        for seed in range(99, 109):
            novel_obs = np.random.RandomState(seed).randn(1, 1, 84, 84).astype(np.float32)
            novel_rewards.append(wrapper._compute_intrinsic_reward(novel_obs)[0])
        mean_novel_reward = np.mean(novel_rewards)

        assert mean_novel_reward > familiar_reward, (
            f"Novel obs not more surprising: novel_avg={mean_novel_reward:.6f}, "
            f"familiar={familiar_reward:.6f}"
        )

    def test_reward_normalisation_stabilises(self, RNDRewardWrapper):
        """Reward normalisation should prevent extreme reward magnitudes.

        After sufficient steps, normalised intrinsic rewards should be
        roughly unit-scale (not orders of magnitude larger than extrinsic).
        """
        venv = _make_vec_env()
        wrapper = RNDRewardWrapper(venv)
        wrapper.reset()

        rewards = []
        for _ in range(100):
            wrapper.step_async(np.array([[0.0]]))
            _, reward, _, _ = wrapper.step_wait()
            rewards.append(reward[0])

        # After 100 steps, rewards should be finite and not extreme
        rewards = np.array(rewards)
        assert np.all(np.isfinite(rewards))
        # Should be roughly bounded (not 1e6 or similar)
        assert np.abs(rewards).max() < 100.0

    def test_obs_normalisation_applied(self, RNDRewardWrapper):
        """Observations fed to RND networks should be normalised."""
        venv = _make_vec_env()
        wrapper = RNDRewardWrapper(venv)
        wrapper.reset()

        # After some steps, obs normaliser should have non-trivial statistics
        for _ in range(10):
            wrapper.step_async(np.array([[0.0]]))
            wrapper.step_wait()

        # Obs normaliser should have been updated
        assert wrapper.obs_rms.count > 0

    def test_non_episodic_intrinsic_reward(self, RNDRewardWrapper):
        """Intrinsic reward statistics should NOT reset on episode boundaries.

        The reward normaliser's running statistics should persist across
        episodes (non-episodic), as specified in Burda et al. (2018).
        """
        venv = _make_vec_env()
        wrapper = RNDRewardWrapper(venv)
        wrapper.reset()

        # Run some steps
        for _ in range(10):
            wrapper.step_async(np.array([[0.0]]))
            wrapper.step_wait()

        count_before = wrapper.reward_rms.count

        # Simulate episode boundary (reset)
        wrapper.reset()

        # Run more steps
        for _ in range(10):
            wrapper.step_async(np.array([[0.0]]))
            wrapper.step_wait()

        # Count should have increased, not reset
        assert wrapper.reward_rms.count > count_before

    def test_update_proportion(self, RNDRewardWrapper):
        """update_proportion controls fraction of batch used for training."""
        venv = _make_vec_env(n_envs=4)
        wrapper = RNDRewardWrapper(venv, update_proportion=0.5)
        assert wrapper.update_proportion == 0.5

    def test_update_proportion_rejects_zero(self, RNDRewardWrapper):
        """update_proportion <= 0 raises ValueError."""
        venv = _make_vec_env()
        with pytest.raises(ValueError, match="update_proportion"):
            RNDRewardWrapper(venv, update_proportion=0.0)

    def test_update_proportion_rejects_negative(self, RNDRewardWrapper):
        """update_proportion < 0 raises ValueError."""
        venv = _make_vec_env()
        with pytest.raises(ValueError, match="update_proportion"):
            RNDRewardWrapper(venv, update_proportion=-0.5)

    def test_update_proportion_rejects_above_one(self, RNDRewardWrapper):
        """update_proportion > 1.0 raises ValueError."""
        venv = _make_vec_env()
        with pytest.raises(ValueError, match="update_proportion"):
            RNDRewardWrapper(venv, update_proportion=1.5)

    def test_rejects_non_3d_observations(self, RNDRewardWrapper):
        """Non-3D observation space raises ValueError (not assert)."""
        # Create a 1D observation env
        env = gym.make("CartPole-v1")
        venv = DummyVecEnv([lambda: env])
        with pytest.raises(ValueError, match="3D.*CHW"):
            RNDRewardWrapper(venv)

    def test_close_delegates(self, RNDRewardWrapper):
        """close() delegates to wrapped environment."""
        venv = _make_vec_env()
        wrapper = RNDRewardWrapper(venv)
        wrapper.close()  # Should not raise

    def test_state_dict_save_load(self, RNDRewardWrapper):
        """RND state can be saved and loaded for checkpointing."""
        venv = _make_vec_env()
        wrapper = RNDRewardWrapper(venv)
        wrapper.reset()

        # Run some steps to build up statistics
        for _ in range(10):
            wrapper.step_async(np.array([[0.0]]))
            wrapper.step_wait()

        state = wrapper.get_rnd_state()
        assert "predictor_state_dict" in state
        assert "obs_rms" in state
        assert "reward_rms" in state

        # Create new wrapper and load state
        venv2 = _make_vec_env()
        wrapper2 = RNDRewardWrapper(venv2)
        wrapper2.load_rnd_state(state)

        # Statistics should match
        np.testing.assert_allclose(wrapper2.obs_rms.mean, wrapper.obs_rms.mean)
        np.testing.assert_allclose(wrapper2.reward_rms.var, wrapper.reward_rms.var)

    def test_cpu_device_fallback(self, RNDRewardWrapper):
        """RND networks should work on CPU."""
        venv = _make_vec_env()
        wrapper = RNDRewardWrapper(venv, device="cpu")
        wrapper.reset()
        wrapper.step_async(np.array([[0.0]]))
        obs, reward, done, info = wrapper.step_wait()
        assert np.isfinite(reward[0])

    def test_device_auto_resolves_to_concrete_device(self, RNDRewardWrapper):
        """device='auto' resolves to a concrete torch device (not 'auto')."""
        venv = _make_vec_env()
        wrapper = RNDRewardWrapper(venv, device="auto")
        # Should not raise RuntimeError from torch.device("auto")
        assert wrapper.device.type in ("cpu", "cuda", "xpu")
        # Verify the wrapper is functional
        wrapper.reset()
        wrapper.step_async(np.array([[0.0]]))
        obs, reward, done, info = wrapper.step_wait()
        assert np.isfinite(reward[0])
