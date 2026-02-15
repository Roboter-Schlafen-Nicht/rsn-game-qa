"""Reward Consistency Oracle — detects reward/game-state mismatches.

Cross-references the reward signal with observable game state changes
(score delta, brick count, lives) to detect broken reward functions
or game logic bugs where the reward doesn't match what actually
happened in the game.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Oracle


class RewardConsistencyOracle(Oracle):
    """Detects inconsistencies between reward and game state changes.

    Detection strategy
    ------------------
    1. Track score changes and compare to reward.  If score increased
       but reward is zero/negative (or vice versa), flag a mismatch.
    2. Track lives — if lives decreased, reward should typically be
       negative (penalty).  Mismatch is flagged.
    3. Track brick count — if bricks decreased, reward should typically
       be positive.  Mismatch is flagged.

    Parameters
    ----------
    score_key : str
        Key in ``info`` for current score.  Default is ``"score"``.
    lives_key : str
        Key in ``info`` for current lives.  Default is ``"lives"``.
    brick_count_key : str
        Key in ``info`` for remaining brick count.
        Default is ``"brick_count"``.
    reward_tolerance : float
        Minimum absolute reward to consider as "non-zero".
        Default is 1e-6.
    check_lives : bool
        Whether to check reward vs lives changes.  Default is True.
    check_bricks : bool
        Whether to check reward vs brick count changes.
        Default is True.
    """

    def __init__(
        self,
        score_key: str = "score",
        lives_key: str = "lives",
        brick_count_key: str = "brick_count",
        reward_tolerance: float = 1e-6,
        check_lives: bool = True,
        check_bricks: bool = True,
    ) -> None:
        super().__init__(name="reward_consistency")
        self.score_key = score_key
        self.lives_key = lives_key
        self.brick_count_key = brick_count_key
        self.reward_tolerance = reward_tolerance
        self.check_lives = check_lives
        self.check_bricks = check_bricks

        self._prev_score: float | None = None
        self._prev_lives: int | None = None
        self._prev_brick_count: int | None = None
        self._step_count: int = 0
        self._mismatch_count: int = 0

    def on_reset(self, obs: np.ndarray, info: dict[str, Any]) -> None:
        """Reset tracking at episode start.

        Parameters
        ----------
        obs : np.ndarray
            Initial observation.
        info : dict[str, Any]
            Reset info dict.
        """
        self._prev_score = info.get(self.score_key)
        self._prev_lives = info.get(self.lives_key)
        self._prev_brick_count = info.get(self.brick_count_key)
        self._step_count = 0
        self._mismatch_count = 0

    def on_step(
        self,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        """Cross-reference reward with game state changes.

        Parameters
        ----------
        obs : np.ndarray
            Current observation.
        reward : float
            Step reward.
        terminated : bool
            Episode terminated flag.
        truncated : bool
            Episode truncated flag.
        info : dict[str, Any]
            Step info dict with game state data.
        """
        self._step_count += 1
        has_reward = abs(reward) > self.reward_tolerance

        # 1. Check score vs reward consistency
        score = info.get(self.score_key)
        if score is not None and self._prev_score is not None:
            score_delta = float(score) - float(self._prev_score)
            score_changed = abs(score_delta) > self.reward_tolerance

            # Score increased but no reward
            if score_delta > 0 and not has_reward:
                self._mismatch_count += 1
                self._add_finding(
                    severity="warning",
                    step=self._step_count,
                    description=(
                        f"Score increased by {score_delta:.1f} but reward "
                        f"is {reward:.6f} (expected positive reward)"
                    ),
                    data={
                        "type": "score_reward_mismatch",
                        "score_delta": score_delta,
                        "reward": reward,
                        "score": float(score),
                    },
                )

            # Reward given but score didn't change (and no life lost)
            lives = info.get(self.lives_key)
            lives_changed = (
                lives is not None
                and self._prev_lives is not None
                and lives != self._prev_lives
            )
            if has_reward and not score_changed and not lives_changed:
                self._mismatch_count += 1
                self._add_finding(
                    severity="info",
                    step=self._step_count,
                    description=(
                        f"Reward {reward:.6f} given but no score or lives "
                        f"change detected"
                    ),
                    data={
                        "type": "phantom_reward",
                        "reward": reward,
                        "score": float(score),
                        "score_delta": score_delta,
                    },
                )

        # 2. Check lives vs reward consistency
        lives = info.get(self.lives_key)
        if self.check_lives and lives is not None and self._prev_lives is not None:
            lives_delta = lives - self._prev_lives
            if lives_delta < 0 and reward > self.reward_tolerance:
                self._mismatch_count += 1
                self._add_finding(
                    severity="warning",
                    step=self._step_count,
                    description=(
                        f"Lost {-lives_delta} life(s) but reward is "
                        f"positive ({reward:.6f})"
                    ),
                    data={
                        "type": "lives_reward_mismatch",
                        "lives_delta": lives_delta,
                        "reward": reward,
                        "lives": lives,
                    },
                )

        # 3. Check brick count vs reward consistency
        brick_count = info.get(self.brick_count_key)
        if (
            self.check_bricks
            and brick_count is not None
            and self._prev_brick_count is not None
        ):
            bricks_destroyed = self._prev_brick_count - brick_count
            if bricks_destroyed > 0 and reward <= self.reward_tolerance:
                self._mismatch_count += 1
                self._add_finding(
                    severity="warning",
                    step=self._step_count,
                    description=(
                        f"Destroyed {bricks_destroyed} brick(s) but reward "
                        f"is {reward:.6f} (expected positive reward)"
                    ),
                    data={
                        "type": "brick_reward_mismatch",
                        "bricks_destroyed": bricks_destroyed,
                        "reward": reward,
                        "brick_count": brick_count,
                    },
                )

        # Update previous state
        if score is not None:
            self._prev_score = float(score)
        if lives is not None:
            self._prev_lives = lives
        if brick_count is not None:
            self._prev_brick_count = brick_count
