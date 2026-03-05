"""Breakout 71 reward computation.

Pure-function reward logic extracted from ``Breakout71Env.compute_reward()``.
All state is passed in via :class:`RewardState` and returned as a new
instance, making the function fully testable without environment
instantiation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BRICK_REWARD_SCALE: float = 10.0
"""Reward multiplier per normalised brick-destruction delta."""

SCORE_REWARD_SCALE: float = 0.01
"""Reward multiplier per JS score-delta point."""

TIME_PENALTY: float = 0.01
"""Small negative reward applied every step to discourage idling."""

DEATH_PENALTY: float = 5.0
"""Penalty subtracted when the episode terminates without clearing the level."""

LEVEL_CLEAR_BONUS: float = 1.0
"""Bonus added when the level is cleared (all bricks destroyed)."""


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardState:
    """Immutable snapshot of reward-relevant state between steps.

    Parameters
    ----------
    prev_bricks_norm : float
        Normalised brick count from the previous step (1.0 = all bricks).
    bricks_total : int | None
        Total brick count recorded at episode start.  ``None`` until the
        first observation sets it.
    prev_score : int
        Game score at the end of the previous step.
    """

    prev_bricks_norm: float = 1.0
    bricks_total: int | None = None
    prev_score: int = 0


# ---------------------------------------------------------------------------
# Pure reward function
# ---------------------------------------------------------------------------


def compute_breakout_reward(
    detections: dict[str, Any],
    terminated: bool,
    level_cleared: bool,
    score: int,
    state: RewardState,
) -> tuple[float, RewardState]:
    """Compute the reward for a single Breakout 71 step.

    Parameters
    ----------
    detections : dict[str, Any]
        Current game-specific detections dict (must contain ``"bricks"``).
    terminated : bool
        Whether the episode ended this step.
    level_cleared : bool
        Whether the level was cleared (all bricks destroyed).
    score : int
        Current game score (from JS bridge or explicit argument).
    state : RewardState
        Reward state from the previous step.

    Returns
    -------
    tuple[float, RewardState]
        ``(reward, new_state)`` where *new_state* carries updated
        ``prev_bricks_norm`` and ``prev_score`` for the next call.
    """
    # Brick destruction reward
    bricks_left = len(detections.get("bricks", []))
    bricks_total = state.bricks_total if state.bricks_total else 1
    bricks_norm = bricks_left / bricks_total
    brick_delta = state.prev_bricks_norm - bricks_norm
    reward = brick_delta * BRICK_REWARD_SCALE

    # Score delta reward
    score_delta = score - state.prev_score
    reward += score_delta * SCORE_REWARD_SCALE

    # Time penalty
    reward -= TIME_PENALTY

    # Terminal rewards
    if terminated and not level_cleared:
        reward -= DEATH_PENALTY

    # Level cleared bonus
    if level_cleared:
        reward += LEVEL_CLEAR_BONUS

    # Build new state (frozen dataclass -- use replace())
    new_state = replace(
        state,
        prev_bricks_norm=bricks_norm,
        prev_score=score,
    )

    return reward, new_state
