"""Tests for the env module (Breakout71Env)."""

import pytest


class TestBreakout71Env:
    """Placeholder tests for Breakout71Env."""

    def test_env_construction(self):
        """Breakout71Env can be constructed with default args."""
        from src.env.breakout71_env import Breakout71Env

        env = Breakout71Env()
        assert env.action_space.n == 4
        assert env.observation_space.shape[0] == 5 + 6 * 10  # 65

    def test_observation_space_shape(self):
        """Observation space should match paddle+ball+bricks layout."""
        from src.env.breakout71_env import Breakout71Env

        env = Breakout71Env(brick_grid_shape=(8, 12))
        expected = 5 + 8 * 12  # 101
        assert env.observation_space.shape[0] == expected

    def test_render_mode_rgb_array(self):
        """render() should return None when no frame captured yet."""
        from src.env.breakout71_env import Breakout71Env

        env = Breakout71Env(render_mode="rgb_array")
        assert env.render() is None

    def test_close_without_init(self):
        """close() should not raise if sub-components were never initialized."""
        from src.env.breakout71_env import Breakout71Env

        env = Breakout71Env()
        env.close()  # should not raise
