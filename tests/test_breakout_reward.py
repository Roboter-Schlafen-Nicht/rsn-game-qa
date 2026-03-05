"""Tests for the Breakout 71 reward module.

Tests the extracted ``compute_breakout_reward()`` function which
computes rewards based on brick destruction, score deltas, time
penalties, and terminal/level-clear bonuses.
"""

from games.breakout71.reward import RewardState, compute_breakout_reward

# -- Helpers ----------------------------------------------------------------


def _detections(bricks=None, **_kw):
    """Build minimal Breakout-specific detections dict for reward tests."""
    if bricks is None:
        bricks = [(0.1 * i, 0.1, 0.05, 0.03) for i in range(10)]
    return {"bricks": bricks}


# -- RewardState construction -----------------------------------------------


class TestRewardState:
    """Tests for the RewardState dataclass."""

    def test_default_construction(self):
        """Default RewardState should have expected initial values."""
        state = RewardState()
        assert state.prev_bricks_norm == 1.0
        assert state.bricks_total is None
        assert state.prev_score == 0

    def test_custom_construction(self):
        """RewardState should accept custom values."""
        state = RewardState(prev_bricks_norm=0.5, bricks_total=20, prev_score=100)
        assert state.prev_bricks_norm == 0.5
        assert state.bricks_total == 20
        assert state.prev_score == 100


# -- compute_breakout_reward -----------------------------------------------


class TestComputeBreakoutReward:
    """Tests for the pure compute_breakout_reward function."""

    def test_brick_destruction_reward(self):
        """Destroying bricks gives positive reward proportional to delta."""
        state = RewardState(prev_bricks_norm=1.0, bricks_total=10, prev_score=0)
        det = _detections(bricks=[(0.1 * i, 0.1, 0.05, 0.03) for i in range(8)])

        reward, new_state = compute_breakout_reward(
            detections=det,
            terminated=False,
            level_cleared=False,
            score=0,
            state=state,
        )

        # brick_delta = 1.0 - 0.8 = 0.2; reward = 0.2 * 10 - 0.01 = 1.99
        assert abs(reward - 1.99) < 1e-6

    def test_time_penalty_only(self):
        """No brick change gives only time penalty."""
        state = RewardState(prev_bricks_norm=0.5, bricks_total=10, prev_score=0)
        det = _detections(bricks=[(0.1 * i, 0.1, 0.05, 0.03) for i in range(5)])

        reward, new_state = compute_breakout_reward(
            detections=det,
            terminated=False,
            level_cleared=False,
            score=0,
            state=state,
        )

        assert abs(reward - (-0.01)) < 1e-6

    def test_game_over_penalty(self):
        """Game over adds -5.0 penalty on top of time penalty."""
        state = RewardState(prev_bricks_norm=0.5, bricks_total=10, prev_score=0)
        det = _detections(bricks=[(0.1 * i, 0.1, 0.05, 0.03) for i in range(5)])

        reward, new_state = compute_breakout_reward(
            detections=det,
            terminated=True,
            level_cleared=False,
            score=0,
            state=state,
        )

        # time penalty (-0.01) + game_over (-5.0) = -5.01
        assert abs(reward - (-5.01)) < 1e-6

    def test_level_cleared_bonus(self):
        """Level cleared adds +1.0 bonus."""
        state = RewardState(prev_bricks_norm=0.0, bricks_total=10, prev_score=0)
        det = _detections(bricks=[])

        reward, new_state = compute_breakout_reward(
            detections=det,
            terminated=False,
            level_cleared=True,
            score=0,
            state=state,
        )

        # brick_delta = 0.0; time penalty (-0.01) + level_clear (+1.0) = 0.99
        assert abs(reward - 0.99) < 1e-6

    def test_combined_brick_and_level_clear(self):
        """Last brick destroyed + level cleared gives both rewards."""
        state = RewardState(prev_bricks_norm=0.1, bricks_total=10, prev_score=0)
        det = _detections(bricks=[])

        reward, new_state = compute_breakout_reward(
            detections=det,
            terminated=False,
            level_cleared=True,
            score=0,
            state=state,
        )

        # brick_delta = 0.1 * 10 = 1.0; + time (-0.01) + level (+1.0) = 1.99
        assert abs(reward - 1.99) < 1e-6

    def test_score_delta_positive(self):
        """Score increase contributes positively to reward."""
        state = RewardState(prev_bricks_norm=1.0, bricks_total=10, prev_score=0)
        det = _detections(bricks=[(0.1 * i, 0.1, 0.05, 0.03) for i in range(10)])

        reward, new_state = compute_breakout_reward(
            detections=det,
            terminated=False,
            level_cleared=False,
            score=100,
            state=state,
        )

        # No brick change: score_delta = 100 * 0.01 = 1.0; time = -0.01
        assert abs(reward - 0.99) < 1e-6

    def test_state_updated_prev_bricks_norm(self):
        """Returned state should have updated prev_bricks_norm."""
        state = RewardState(prev_bricks_norm=1.0, bricks_total=10, prev_score=0)
        det = _detections(bricks=[(0.1 * i, 0.1, 0.05, 0.03) for i in range(7)])

        _, new_state = compute_breakout_reward(
            detections=det,
            terminated=False,
            level_cleared=False,
            score=0,
            state=state,
        )

        assert abs(new_state.prev_bricks_norm - 0.7) < 1e-6

    def test_state_updated_prev_score(self):
        """Returned state should have updated prev_score."""
        state = RewardState(prev_bricks_norm=1.0, bricks_total=10, prev_score=0)
        det = _detections(bricks=[(0.1 * i, 0.1, 0.05, 0.03) for i in range(10)])

        _, new_state = compute_breakout_reward(
            detections=det,
            terminated=False,
            level_cleared=False,
            score=42,
            state=state,
        )

        assert new_state.prev_score == 42

    def test_original_state_not_mutated(self):
        """The original state object should not be mutated."""
        state = RewardState(prev_bricks_norm=1.0, bricks_total=10, prev_score=0)
        det = _detections(bricks=[(0.1 * i, 0.1, 0.05, 0.03) for i in range(5)])

        _, new_state = compute_breakout_reward(
            detections=det,
            terminated=False,
            level_cleared=False,
            score=50,
            state=state,
        )

        # Original state should be unchanged
        assert state.prev_bricks_norm == 1.0
        assert state.prev_score == 0
        # New state should differ
        assert new_state.prev_bricks_norm != 1.0
        assert new_state.prev_score == 50

    def test_bricks_total_none_uses_one(self):
        """When bricks_total is None, should default to 1 for division."""
        state = RewardState(prev_bricks_norm=1.0, bricks_total=None, prev_score=0)
        det = _detections(bricks=[(0.5, 0.1, 0.05, 0.03)])

        reward, new_state = compute_breakout_reward(
            detections=det,
            terminated=False,
            level_cleared=False,
            score=0,
            state=state,
        )

        # bricks_total defaults to 1; bricks_left=1; norm=1.0
        # brick_delta = 1.0 - 1.0 = 0.0; reward = -0.01
        assert abs(reward - (-0.01)) < 1e-6

    def test_game_over_with_level_cleared_gives_no_death_penalty(self):
        """When terminated=True AND level_cleared=True, no death penalty."""
        state = RewardState(prev_bricks_norm=0.0, bricks_total=10, prev_score=0)
        det = _detections(bricks=[])

        reward, new_state = compute_breakout_reward(
            detections=det,
            terminated=True,
            level_cleared=True,
            score=0,
            state=state,
        )

        # Death penalty only applies when terminated and NOT level_cleared
        # So: time (-0.01) + level (+1.0) = 0.99
        assert abs(reward - 0.99) < 1e-6

    def test_constants_accessible(self):
        """Module-level reward constants should be importable."""
        from games.breakout71.reward import (
            BRICK_REWARD_SCALE,
            DEATH_PENALTY,
            LEVEL_CLEAR_BONUS,
            SCORE_REWARD_SCALE,
            TIME_PENALTY,
        )

        assert BRICK_REWARD_SCALE == 10.0
        assert SCORE_REWARD_SCALE == 0.01
        assert TIME_PENALTY == 0.01
        assert DEATH_PENALTY == 5.0
        assert LEVEL_CLEAR_BONUS == 1.0
