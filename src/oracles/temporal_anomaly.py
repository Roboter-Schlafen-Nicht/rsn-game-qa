"""Temporal Anomaly Oracle — detects object teleportation and flickering.

Monitors tracked object positions across consecutive frames and flags
sudden position jumps (teleportation) or objects that appear and
disappear rapidly (flickering), indicating physics or rendering bugs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Oracle


class TemporalAnomalyOracle(Oracle):
    """Detects temporal anomalies in object motion.

    Detection strategy
    ------------------
    1. Track object positions across consecutive steps.
    2. If any object's position delta exceeds ``max_speed``, flag
       teleportation.
    3. If an object alternates between present and absent in the
       ``info`` dict, flag flickering.

    Parameters
    ----------
    tracked_keys : list[str]
        Keys in ``info`` whose values are ``[x, y]`` positions to
        monitor.  Default is ``["ball_pos", "paddle_pos"]``.
    max_speed : float
        Maximum expected displacement per step (normalised coords).
        Default is 0.15.
    flicker_window : int
        Number of steps to look back for presence/absence flickering.
        Default is 10.
    flicker_threshold : int
        Minimum number of presence/absence transitions within
        ``flicker_window`` to flag flickering.  Default is 4.
    """

    def __init__(
        self,
        tracked_keys: list[str] | None = None,
        max_speed: float = 0.15,
        flicker_window: int = 10,
        flicker_threshold: int = 4,
    ) -> None:
        super().__init__(name="temporal_anomaly")
        self.tracked_keys = tracked_keys or ["ball_pos", "paddle_pos"]
        self.max_speed = max_speed
        self.flicker_window = flicker_window
        self.flicker_threshold = flicker_threshold

        self._prev_positions: dict[str, np.ndarray | None] = {}
        # Track presence (True) / absence (False) history per key
        self._presence_history: dict[str, list[bool]] = {}
        self._step_count: int = 0
        # Cooldown: step at which flicker was last reported per key
        self._flicker_cooldown: dict[str, int] = {}

    def on_reset(self, obs: np.ndarray, info: dict[str, Any]) -> None:
        """Reset tracking at episode start.

        Parameters
        ----------
        obs : np.ndarray
            Initial observation.
        info : dict[str, Any]
            Reset info dict.
        """
        self._prev_positions = {}
        self._presence_history = {key: [] for key in self.tracked_keys}
        self._step_count = 0
        self._flicker_cooldown = {}

    def on_step(
        self,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        """Check for teleportation and flickering.

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
            Step info dict with position data.
        """
        self._step_count += 1

        for key in self.tracked_keys:
            pos = info.get(key)
            present = pos is not None

            # Track presence history for flicker detection
            history = self._presence_history.get(key, [])
            history.append(present)
            if len(history) > self.flicker_window:
                history.pop(0)
            self._presence_history[key] = history

            if present:
                pos_arr = np.asarray(pos, dtype=np.float64)

                # Check teleportation
                prev_pos = self._prev_positions.get(key)
                if prev_pos is not None:
                    displacement = float(np.linalg.norm(pos_arr - prev_pos))
                    if displacement > self.max_speed:
                        self._add_finding(
                            severity="warning",
                            step=self._step_count,
                            description=(
                                f"'{key}' teleported {displacement:.4f} units "
                                f"in one step (max: {self.max_speed}) — "
                                f"possible physics/rendering bug"
                            ),
                            data={
                                "type": "teleportation",
                                "key": key,
                                "displacement": displacement,
                                "max_speed": self.max_speed,
                                "prev_pos": prev_pos.tolist(),
                                "curr_pos": pos_arr.tolist(),
                            },
                        )

                self._prev_positions[key] = pos_arr.copy()
            else:
                self._prev_positions[key] = None

            # Check flickering
            self._check_flicker(key)

    def _check_flicker(self, key: str) -> None:
        """Check if an object is flickering (rapidly appearing/disappearing).

        Parameters
        ----------
        key : str
            The info dict key being monitored.
        """
        history = self._presence_history.get(key, [])
        if len(history) < self.flicker_window:
            return

        # Cooldown: suppress if we reported within the last flicker_window steps
        last_reported = self._flicker_cooldown.get(key, -self.flicker_window)
        if (self._step_count - last_reported) < self.flicker_window:
            return

        # Count transitions (present -> absent or absent -> present)
        transitions = sum(1 for i in range(1, len(history)) if history[i] != history[i - 1])

        if transitions >= self.flicker_threshold:
            self._add_finding(
                severity="warning",
                step=self._step_count,
                description=(
                    f"'{key}' is flickering — {transitions} presence "
                    f"transitions in last {self.flicker_window} steps"
                ),
                data={
                    "type": "flickering",
                    "key": key,
                    "transitions": transitions,
                    "window": self.flicker_window,
                    "history": history.copy(),
                },
            )
            # Record cooldown and reset history
            self._flicker_cooldown[key] = self._step_count
            self._presence_history[key] = [history[-1]]
