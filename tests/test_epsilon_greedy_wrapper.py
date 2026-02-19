"""Tests for the EpsilonGreedyWrapper VecEnv wrapper.

Covers:
- Action replacement probability at epsilon=0, 0.5, 1.0
- Passthrough of step_wait, reset, observation/reward/done/info
- Epsilon validation (rejects out-of-range values)
- Seed reproducibility
- Action space sampling for replaced actions
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from src.platform.epsilon_greedy_wrapper import EpsilonGreedyWrapper


# ---------------------------------------------------------------------------
# Helper: simple env for testing
# ---------------------------------------------------------------------------
class _DummyEnv(gym.Env):
    """Minimal gym env with continuous action space."""

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(4,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self._last_action = None

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        self._last_action = action
        obs = self.observation_space.sample()
        return obs, 1.0, False, False, {}


def _make_vec_env(n_envs: int = 1) -> DummyVecEnv:
    return DummyVecEnv([lambda: _DummyEnv() for _ in range(n_envs)])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEpsilonValidation:
    """Epsilon parameter validation."""

    def test_rejects_negative_epsilon(self):
        vec_env = _make_vec_env()
        with pytest.raises(ValueError, match="epsilon must be in"):
            EpsilonGreedyWrapper(vec_env, epsilon=-0.1)

    def test_rejects_epsilon_above_one(self):
        vec_env = _make_vec_env()
        with pytest.raises(ValueError, match="epsilon must be in"):
            EpsilonGreedyWrapper(vec_env, epsilon=1.5)

    def test_accepts_epsilon_zero(self):
        vec_env = _make_vec_env()
        wrapper = EpsilonGreedyWrapper(vec_env, epsilon=0.0)
        assert wrapper.epsilon == 0.0

    def test_accepts_epsilon_one(self):
        vec_env = _make_vec_env()
        wrapper = EpsilonGreedyWrapper(vec_env, epsilon=1.0)
        assert wrapper.epsilon == 1.0

    def test_accepts_epsilon_in_range(self):
        vec_env = _make_vec_env()
        wrapper = EpsilonGreedyWrapper(vec_env, epsilon=0.1)
        assert wrapper.epsilon == 0.1


class TestPassthrough:
    """Actions, observations, and info pass through correctly."""

    def test_epsilon_zero_passes_actions_through(self):
        """With epsilon=0, all actions should be passed through unchanged."""
        vec_env = _make_vec_env()
        wrapper = EpsilonGreedyWrapper(vec_env, epsilon=0.0)
        wrapper.reset()

        original_action = np.array([[0.42]], dtype=np.float32)
        obs, reward, done, info = wrapper.step(original_action)

        # The inner env received the exact action
        inner_env = vec_env.envs[0]
        np.testing.assert_allclose(inner_env._last_action, [0.42], atol=1e-6)

    def test_reset_passes_through(self):
        """reset() returns observations from the wrapped env."""
        vec_env = _make_vec_env()
        wrapper = EpsilonGreedyWrapper(vec_env, epsilon=0.5)
        obs = wrapper.reset()
        assert obs.shape == (1, 4)

    def test_step_returns_correct_shapes(self):
        """step() returns obs, reward, done, info with correct shapes."""
        vec_env = _make_vec_env()
        wrapper = EpsilonGreedyWrapper(vec_env, epsilon=0.1)
        wrapper.reset()

        actions = np.array([[0.0]], dtype=np.float32)
        obs, rewards, dones, infos = wrapper.step(actions)

        assert obs.shape == (1, 4)
        assert rewards.shape == (1,)
        assert dones.shape == (1,)
        assert len(infos) == 1


class TestActionReplacement:
    """Epsilon-greedy action replacement behaviour."""

    def test_epsilon_one_replaces_all_actions(self):
        """With epsilon=1.0, all actions should be replaced with random ones."""
        vec_env = _make_vec_env()
        wrapper = EpsilonGreedyWrapper(vec_env, epsilon=1.0, seed=42)
        wrapper.reset()

        # Send a distinctive action and verify it was replaced
        original_action = np.array([[0.999]], dtype=np.float32)
        n_replaced = 0
        for _ in range(20):
            wrapper.step(original_action.copy())
            inner_env = vec_env.envs[0]
            if abs(inner_env._last_action[0] - 0.999) > 1e-6:
                n_replaced += 1

        # All 20 should be replaced (epsilon=1.0)
        assert n_replaced == 20

    def test_epsilon_half_replaces_roughly_half(self):
        """With epsilon=0.5, roughly half the actions should be replaced."""
        vec_env = _make_vec_env()
        wrapper = EpsilonGreedyWrapper(vec_env, epsilon=0.5, seed=123)
        wrapper.reset()

        original_action = np.array([[0.999]], dtype=np.float32)
        n_replaced = 0
        n_trials = 1000
        for _ in range(n_trials):
            wrapper.step(original_action.copy())
            inner_env = vec_env.envs[0]
            if abs(inner_env._last_action[0] - 0.999) > 1e-6:
                n_replaced += 1

        # Should be roughly 50% (within 10% tolerance for 1000 trials)
        ratio = n_replaced / n_trials
        assert 0.35 < ratio < 0.65, f"Expected ~0.5, got {ratio}"

    def test_replaced_actions_are_in_action_space(self):
        """Replaced actions should be valid samples from the action space."""
        vec_env = _make_vec_env()
        wrapper = EpsilonGreedyWrapper(vec_env, epsilon=1.0, seed=42)
        wrapper.reset()

        for _ in range(50):
            wrapper.step(np.array([[0.0]], dtype=np.float32))
            inner_env = vec_env.envs[0]
            action_val = inner_env._last_action[0]
            assert -1.0 <= action_val <= 1.0, f"Action {action_val} out of bounds"


class TestMultiEnv:
    """Multi-environment behaviour."""

    def test_independent_per_env_replacement(self):
        """Each env in a VecEnv gets an independent coin flip."""
        vec_env = _make_vec_env(n_envs=4)
        wrapper = EpsilonGreedyWrapper(vec_env, epsilon=0.5, seed=42)
        wrapper.reset()

        # Send identical actions to all 4 envs
        actions = np.full((4, 1), 0.999, dtype=np.float32)
        wrapper.step(actions.copy())

        # With epsilon=0.5, it's unlikely all 4 envs get the same
        # replacement decision every time. Over many trials, we should
        # see mixed patterns.
        patterns = set()
        for _ in range(50):
            wrapper.step(actions.copy())
            received = tuple(
                abs(vec_env.envs[i]._last_action[0] - 0.999) > 1e-6 for i in range(4)
            )
            patterns.add(received)

        # Should have more than 1 unique pattern (not all same decision)
        assert len(patterns) > 1


class TestSeedReproducibility:
    """Seeded wrapper produces reproducible results."""

    def test_same_seed_same_sequence(self):
        """Two wrappers with the same seed produce identical replacement patterns."""
        results = []
        for _ in range(2):
            vec_env = _make_vec_env()
            wrapper = EpsilonGreedyWrapper(vec_env, epsilon=0.5, seed=42)
            wrapper.reset()

            sequence = []
            for _ in range(100):
                action = np.array([[0.999]], dtype=np.float32)
                wrapper.step(action.copy())
                inner_env = vec_env.envs[0]
                was_replaced = abs(inner_env._last_action[0] - 0.999) > 1e-6
                sequence.append(was_replaced)
            results.append(sequence)

        assert results[0] == results[1]

    def test_different_seeds_different_sequences(self):
        """Two wrappers with different seeds produce different patterns."""
        results = []
        for seed in [42, 99]:
            vec_env = _make_vec_env()
            wrapper = EpsilonGreedyWrapper(vec_env, epsilon=0.5, seed=seed)
            wrapper.reset()

            sequence = []
            for _ in range(100):
                action = np.array([[0.999]], dtype=np.float32)
                wrapper.step(action.copy())
                inner_env = vec_env.envs[0]
                was_replaced = abs(inner_env._last_action[0] - 0.999) > 1e-6
                sequence.append(was_replaced)
            results.append(sequence)

        # Very unlikely to be identical with different seeds
        assert results[0] != results[1]
