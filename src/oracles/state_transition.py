"""State Transition Oracle — detects invalid game state transitions.

Monitors game state variables (lives, level, game phase) across
consecutive steps and flags transitions that violate expected game
logic, such as losing multiple lives in a single step or skipping
levels.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Oracle


class StateTransitionOracle(Oracle):
    """Detects invalid game state transitions.

    Detection strategy
    ------------------
    1. Track ``lives``, ``level``, and ``game_state`` across steps.
    2. Lives should only decrease by at most 1 per step.
    3. Level should only increase by 1 (no skipping).
    4. Game state transitions should follow allowed paths (e.g.
       ``"playing"`` -> ``"game_over"`` is valid, but ``"menu"`` ->
       ``"game_over"`` without passing through ``"playing"`` is not).

    Parameters
    ----------
    lives_key : str
        Key in ``info`` for current lives.  Default is ``"lives"``.
    level_key : str
        Key in ``info`` for current level.  Default is ``"level"``.
    game_state_key : str
        Key in ``info`` for game state string.
        Default is ``"game_state"``.
    allowed_transitions : dict[str, list[str]], optional
        Mapping of ``state -> [allowed_next_states]``.  If not
        provided, all transitions are allowed (only lives/level
        checks are performed).
    max_lives_loss_per_step : int
        Maximum lives that can be lost in a single step.
        Default is 1.
    """

    def __init__(
        self,
        lives_key: str = "lives",
        level_key: str = "level",
        game_state_key: str = "game_state",
        allowed_transitions: dict[str, list[str]] | None = None,
        max_lives_loss_per_step: int = 1,
    ) -> None:
        super().__init__(name="state_transition")
        self.lives_key = lives_key
        self.level_key = level_key
        self.game_state_key = game_state_key
        self.allowed_transitions = allowed_transitions
        self.max_lives_loss_per_step = max_lives_loss_per_step

        self._prev_lives: int | None = None
        self._prev_level: int | None = None
        self._prev_game_state: str | None = None
        self._step_count: int = 0

    def on_reset(self, obs: np.ndarray, info: dict[str, Any]) -> None:
        """Reset tracking at episode start.

        Parameters
        ----------
        obs : np.ndarray
            Initial observation.
        info : dict[str, Any]
            Reset info dict.
        """
        self._prev_lives = info.get(self.lives_key)
        self._prev_level = info.get(self.level_key)
        self._prev_game_state = info.get(self.game_state_key)
        self._step_count = 0

    def on_step(
        self,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        """Validate state transitions after each step.

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

        # 1. Check lives transition
        lives = info.get(self.lives_key)
        if lives is not None and self._prev_lives is not None:
            lives_delta = self._prev_lives - lives  # positive = lost lives

            if lives_delta > self.max_lives_loss_per_step:
                self._add_finding(
                    severity="critical",
                    step=self._step_count,
                    description=(
                        f"Lost {lives_delta} lives in one step "
                        f"(max allowed: {self.max_lives_loss_per_step}): "
                        f"{self._prev_lives} -> {lives}"
                    ),
                    data={
                        "type": "excessive_lives_loss",
                        "prev_lives": self._prev_lives,
                        "curr_lives": lives,
                        "delta": lives_delta,
                    },
                )

            if lives_delta < 0:
                # Lives increased — might be valid (extra life pickup)
                # but worth noting
                self._add_finding(
                    severity="info",
                    step=self._step_count,
                    description=(
                        f"Lives increased: {self._prev_lives} -> {lives} "
                        f"(+{-lives_delta})"
                    ),
                    data={
                        "type": "lives_increase",
                        "prev_lives": self._prev_lives,
                        "curr_lives": lives,
                        "delta": lives_delta,
                    },
                )

        # 2. Check level transition
        level = info.get(self.level_key)
        if level is not None and self._prev_level is not None:
            level_delta = level - self._prev_level

            if level_delta > 1:
                self._add_finding(
                    severity="critical",
                    step=self._step_count,
                    description=(
                        f"Level skipped: {self._prev_level} -> {level} "
                        f"(delta={level_delta})"
                    ),
                    data={
                        "type": "level_skip",
                        "prev_level": self._prev_level,
                        "curr_level": level,
                        "delta": level_delta,
                    },
                )

            if level_delta < 0:
                self._add_finding(
                    severity="warning",
                    step=self._step_count,
                    description=(f"Level decreased: {self._prev_level} -> {level}"),
                    data={
                        "type": "level_decrease",
                        "prev_level": self._prev_level,
                        "curr_level": level,
                        "delta": level_delta,
                    },
                )

        # 3. Check game state transition
        game_state = info.get(self.game_state_key)
        if (
            game_state is not None
            and self._prev_game_state is not None
            and game_state != self._prev_game_state
            and self.allowed_transitions is not None
        ):
            allowed = self.allowed_transitions.get(self._prev_game_state, [])
            if game_state not in allowed:
                self._add_finding(
                    severity="critical",
                    step=self._step_count,
                    description=(
                        f"Invalid state transition: "
                        f"'{self._prev_game_state}' -> '{game_state}' "
                        f"(allowed: {allowed})"
                    ),
                    data={
                        "type": "invalid_transition",
                        "prev_state": self._prev_game_state,
                        "curr_state": game_state,
                        "allowed": allowed,
                    },
                )

        # Update previous state
        if lives is not None:
            self._prev_lives = lives
        if level is not None:
            self._prev_level = level
        if game_state is not None:
            self._prev_game_state = game_state
