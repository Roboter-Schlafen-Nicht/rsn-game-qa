"""Tests for the env module (Breakout71Env)."""


class TestBreakout71Env:
    """Placeholder tests for Breakout71Env."""

    def test_env_construction(self):
        """Breakout71Env can be constructed with default args."""
        from src.env.breakout71_env import Breakout71Env

        env = Breakout71Env()
        assert env.action_space.n == 3
        assert env.observation_space.shape == (6,)

    def test_observation_space_bounds(self):
        """Observation space should have correct low/high bounds."""
        from src.env.breakout71_env import Breakout71Env

        env = Breakout71Env()
        # Positions [0,1], velocities [-1,1], bricks_norm [0,1]
        assert env.observation_space.low[0] == 0.0  # paddle_x low
        assert env.observation_space.low[3] == -1.0  # ball_vx low
        assert env.observation_space.high[5] == 1.0  # bricks_norm high

    def test_action_space_no_fire(self):
        """Action space should be Discrete(3) with no FIRE action."""
        from src.env.breakout71_env import Breakout71Env

        env = Breakout71Env()
        assert env.action_space.n == 3  # NOOP, LEFT, RIGHT only

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
