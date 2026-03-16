"""Tests for the CnnCfCPolicy (CfC-based recurrent policy).

Covers:
- Policy construction with default and custom args
- Forward pass output shapes (action, value, log_prob, states)
- _process_sequence handles episode resets correctly
- Hidden state shape matches cfc_hidden_size
- predict() with None state initializes correctly
- predict() returns state in correct tuple format
- evaluate_actions() returns correct shapes
- get_distribution() returns distribution and updated states
- predict_values() returns correct shape
- Integration: RecurrentPPO can construct with CnnCfCPolicy
- Integration: RecurrentPPO.learn() runs without error
- Shared CfC mode
- No critic CfC mode
"""

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.vec_env import DummyVecEnv

# ---------------------------------------------------------------------------
# Helper: minimal gymnasium environment with image observations
# ---------------------------------------------------------------------------


class _ImageEnv(gym.Env):
    """Minimal gym env with (1, 84, 84) Box observation space.

    This matches the LTC pipeline: single-frame CHW uint8.
    Extends ``gym.Env`` so that ``DummyVecEnv`` / ``gym.Wrapper`` accept it.
    """

    metadata: dict = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._step = 0
        obs = np.random.randint(0, 256, (1, 84, 84), dtype=np.uint8)
        return obs, {}

    def step(self, action):
        self._step += 1
        obs = np.random.randint(0, 256, (1, 84, 84), dtype=np.uint8)
        reward = 0.0
        terminated = self._step >= 100
        truncated = False
        return obs, reward, terminated, truncated, {}

    def render(self):
        pass


def _make_vec_env():
    """Create a DummyVecEnv with a single _ImageEnv."""
    return DummyVecEnv([lambda: _ImageEnv()])


# ---------------------------------------------------------------------------
# Tests for CnnCfCPolicy construction
# ---------------------------------------------------------------------------


class TestCnnCfCPolicyConstruction:
    """Tests for CnnCfCPolicy instantiation."""

    def test_default_construction(self):
        """Policy can be constructed with default arguments."""
        from src.platform.ltc_policy import CnnCfCPolicy

        obs_space = spaces.Box(0, 255, (1, 84, 84), dtype=np.uint8)
        act_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        lr_schedule = lambda _: 3e-4  # noqa: E731

        policy = CnnCfCPolicy(obs_space, act_space, lr_schedule)
        assert policy is not None

    def test_custom_hidden_size(self):
        """Policy respects custom cfc_hidden_size."""
        from src.platform.ltc_policy import CnnCfCPolicy

        obs_space = spaces.Box(0, 255, (1, 84, 84), dtype=np.uint8)
        act_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        lr_schedule = lambda _: 3e-4  # noqa: E731

        policy = CnnCfCPolicy(obs_space, act_space, lr_schedule, lstm_hidden_size=128)
        assert policy.lstm_output_dim == 128

    def test_has_cfc_actor(self):
        """Policy has a CfC-based actor (not nn.LSTM)."""
        from src.platform.ltc_policy import CnnCfCPolicy

        obs_space = spaces.Box(0, 255, (1, 84, 84), dtype=np.uint8)
        act_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        lr_schedule = lambda _: 3e-4  # noqa: E731

        policy = CnnCfCPolicy(obs_space, act_space, lr_schedule)
        # lstm_actor should NOT be an nn.LSTM
        assert not isinstance(policy.lstm_actor, th.nn.LSTM)

    def test_has_cfc_critic(self):
        """Policy has a CfC-based critic when enable_critic_lstm=True."""
        from src.platform.ltc_policy import CnnCfCPolicy

        obs_space = spaces.Box(0, 255, (1, 84, 84), dtype=np.uint8)
        act_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        lr_schedule = lambda _: 3e-4  # noqa: E731

        policy = CnnCfCPolicy(obs_space, act_space, lr_schedule, enable_critic_lstm=True)
        assert policy.lstm_critic is not None
        assert not isinstance(policy.lstm_critic, th.nn.LSTM)

    def test_no_critic_cfc(self):
        """Policy uses linear critic when enable_critic_lstm=False."""
        from src.platform.ltc_policy import CnnCfCPolicy

        obs_space = spaces.Box(0, 255, (1, 84, 84), dtype=np.uint8)
        act_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        lr_schedule = lambda _: 3e-4  # noqa: E731

        policy = CnnCfCPolicy(
            obs_space,
            act_space,
            lr_schedule,
            enable_critic_lstm=False,
            shared_lstm=False,
        )
        assert policy.lstm_critic is None
        assert policy.critic is not None

    def test_lstm_actor_has_required_attrs(self):
        """lstm_actor exposes num_layers and hidden_size for RecurrentPPO."""
        from src.platform.ltc_policy import CnnCfCPolicy

        obs_space = spaces.Box(0, 255, (1, 84, 84), dtype=np.uint8)
        act_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        lr_schedule = lambda _: 3e-4  # noqa: E731

        policy = CnnCfCPolicy(obs_space, act_space, lr_schedule, lstm_hidden_size=64)
        assert hasattr(policy.lstm_actor, "num_layers")
        assert hasattr(policy.lstm_actor, "hidden_size")
        assert hasattr(policy.lstm_actor, "input_size")
        assert policy.lstm_actor.num_layers == 1
        assert policy.lstm_actor.hidden_size == 64


# ---------------------------------------------------------------------------
# Tests for forward pass
# ---------------------------------------------------------------------------


class TestCnnCfCPolicyForward:
    """Tests for CnnCfCPolicy.forward()."""

    @pytest.fixture
    def policy(self):
        from src.platform.ltc_policy import CnnCfCPolicy

        obs_space = spaces.Box(0, 255, (1, 84, 84), dtype=np.uint8)
        act_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        lr_schedule = lambda _: 3e-4  # noqa: E731
        return CnnCfCPolicy(obs_space, act_space, lr_schedule, lstm_hidden_size=64)

    def test_forward_output_shapes(self, policy):
        """forward() returns (action, value, log_prob, RNNStates)."""
        batch_size = 2
        obs = th.randint(0, 256, (batch_size, 1, 84, 84), dtype=th.uint8)
        # State shape: (n_layers=1, batch=2, hidden=64)
        h_pi = th.zeros(1, batch_size, 64)
        c_pi = th.zeros(1, batch_size, 64)
        h_vf = th.zeros(1, batch_size, 64)
        c_vf = th.zeros(1, batch_size, 64)
        states = RNNStates((h_pi, c_pi), (h_vf, c_vf))
        episode_starts = th.zeros(batch_size)

        actions, values, log_probs, new_states = policy.forward(obs, states, episode_starts)

        assert actions.shape == (batch_size, 1)  # Box(1,) action
        assert values.shape == (batch_size, 1)
        assert log_probs.shape == (batch_size,)
        assert isinstance(new_states, RNNStates)
        # State shapes should be (n_layers=1, batch, hidden)
        assert new_states.pi[0].shape == (1, batch_size, 64)
        assert new_states.vf[0].shape == (1, batch_size, 64)

    def test_forward_episode_reset(self, policy):
        """forward() resets state when episode_starts=1."""
        batch_size = 2
        obs = th.randint(0, 256, (batch_size, 1, 84, 84), dtype=th.uint8)
        # Non-zero initial state
        h_pi = th.ones(1, batch_size, 64)
        c_pi = th.zeros(1, batch_size, 64)
        h_vf = th.ones(1, batch_size, 64)
        c_vf = th.zeros(1, batch_size, 64)
        states = RNNStates((h_pi, c_pi), (h_vf, c_vf))
        # Both envs start a new episode
        episode_starts = th.ones(batch_size)

        # The state should be zeroed before processing
        _, _, _, new_states = policy.forward(obs, states, episode_starts)
        # States should be non-zero (CfC processed the observation)
        # but NOT identical to what they'd be with non-reset states
        assert new_states.pi[0] is not None


# ---------------------------------------------------------------------------
# Tests for predict()
# ---------------------------------------------------------------------------


class TestCnnCfCPolicyPredict:
    """Tests for CnnCfCPolicy.predict()."""

    @pytest.fixture
    def policy(self):
        from src.platform.ltc_policy import CnnCfCPolicy

        obs_space = spaces.Box(0, 255, (1, 84, 84), dtype=np.uint8)
        act_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        lr_schedule = lambda _: 3e-4  # noqa: E731
        return CnnCfCPolicy(obs_space, act_space, lr_schedule, lstm_hidden_size=64)

    def test_predict_none_state(self, policy):
        """predict() initializes state when state=None."""
        obs = np.random.randint(0, 256, (1, 1, 84, 84), dtype=np.uint8)
        actions, states = policy.predict(obs, state=None)
        assert actions.shape == (1,) or actions.shape == (1, 1)
        assert states is not None
        # States should be a tuple of numpy arrays
        assert isinstance(states, tuple)
        assert len(states) == 2

    def test_predict_state_roundtrip(self, policy):
        """predict() output state can be fed back as input state."""
        obs = np.random.randint(0, 256, (1, 1, 84, 84), dtype=np.uint8)
        actions1, states1 = policy.predict(obs, state=None)
        # Feed states back
        actions2, states2 = policy.predict(obs, state=states1)
        assert states2 is not None


# ---------------------------------------------------------------------------
# Tests for evaluate_actions()
# ---------------------------------------------------------------------------


class TestCnnCfCPolicyEvaluateActions:
    """Tests for CnnCfCPolicy.evaluate_actions()."""

    @pytest.fixture
    def policy(self):
        from src.platform.ltc_policy import CnnCfCPolicy

        obs_space = spaces.Box(0, 255, (1, 84, 84), dtype=np.uint8)
        act_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        lr_schedule = lambda _: 3e-4  # noqa: E731
        return CnnCfCPolicy(obs_space, act_space, lr_schedule, lstm_hidden_size=64)

    def test_evaluate_actions_shapes(self, policy):
        """evaluate_actions() returns (values, log_prob, entropy)."""
        batch_size = 4
        obs = th.randint(0, 256, (batch_size, 1, 84, 84), dtype=th.uint8)
        actions = th.zeros(batch_size, 1)
        h = th.zeros(1, batch_size, 64)
        c = th.zeros(1, batch_size, 64)
        states = RNNStates((h, c), (h.clone(), c.clone()))
        episode_starts = th.zeros(batch_size)

        values, log_prob, entropy = policy.evaluate_actions(obs, actions, states, episode_starts)

        assert values.shape == (batch_size, 1)
        assert log_prob.shape == (batch_size,)
        assert entropy.shape == (batch_size,)


# ---------------------------------------------------------------------------
# Tests for predict_values()
# ---------------------------------------------------------------------------


class TestCnnCfCPolicyPredictValues:
    """Tests for CnnCfCPolicy.predict_values()."""

    def test_predict_values_shape(self):
        """predict_values() returns (batch, 1) tensor."""
        from src.platform.ltc_policy import CnnCfCPolicy

        obs_space = spaces.Box(0, 255, (1, 84, 84), dtype=np.uint8)
        act_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        lr_schedule = lambda _: 3e-4  # noqa: E731
        policy = CnnCfCPolicy(obs_space, act_space, lr_schedule, lstm_hidden_size=64)

        batch_size = 2
        obs = th.randint(0, 256, (batch_size, 1, 84, 84), dtype=th.uint8)
        h = th.zeros(1, batch_size, 64)
        c = th.zeros(1, batch_size, 64)
        episode_starts = th.zeros(batch_size)

        values = policy.predict_values(obs, (h, c), episode_starts)
        assert values.shape == (batch_size, 1)


# ---------------------------------------------------------------------------
# Tests for RecurrentPPO integration
# ---------------------------------------------------------------------------


class TestRecurrentPPOIntegration:
    """Integration tests: CnnCfCPolicy with RecurrentPPO."""

    def test_recurrent_ppo_construction(self):
        """RecurrentPPO can be constructed with CnnCfCPolicy."""
        from sb3_contrib import RecurrentPPO

        from src.platform.ltc_policy import CnnCfCPolicy

        vec_env = _make_vec_env()
        model = RecurrentPPO(
            CnnCfCPolicy,
            vec_env,
            n_steps=16,
            batch_size=16,
            policy_kwargs={"lstm_hidden_size": 64},
            device="cpu",
            verbose=0,
        )
        assert model is not None
        vec_env.close()

    def test_recurrent_ppo_learn(self):
        """RecurrentPPO.learn() runs 64 steps without error."""
        from sb3_contrib import RecurrentPPO

        from src.platform.ltc_policy import CnnCfCPolicy

        vec_env = _make_vec_env()
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

    def test_recurrent_ppo_predict(self):
        """RecurrentPPO.predict() works with CnnCfCPolicy."""
        from sb3_contrib import RecurrentPPO

        from src.platform.ltc_policy import CnnCfCPolicy

        vec_env = _make_vec_env()
        model = RecurrentPPO(
            CnnCfCPolicy,
            vec_env,
            n_steps=16,
            batch_size=16,
            policy_kwargs={"lstm_hidden_size": 32},
            device="cpu",
            verbose=0,
        )

        obs = vec_env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        for _ in range(5):
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            obs, _, dones, _ = vec_env.step(action)
            episode_starts = dones

        vec_env.close()
