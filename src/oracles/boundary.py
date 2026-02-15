"""Boundary Oracle — detects objects leaving valid play area.

Monitors tracked object positions and flags when any object exits the
expected spatial bounds, which indicates out-of-bounds bugs, clipping,
or spatial calculation errors.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Oracle


class BoundaryOracle(Oracle):
    """Detects objects leaving the valid play area.

    Detection strategy
    ------------------
    1. Track named object positions from the ``info`` dict each step.
    2. If any position coordinate falls outside ``[min_bound, max_bound]``,
       flag an out-of-bounds violation.
    3. Distinguish between hard violations (object fully outside) and
       soft violations (object partially outside / at edge).

    Parameters
    ----------
    tracked_keys : list[str]
        Keys in the ``info`` dict whose values are ``[x, y]`` positions
        to monitor.  Default is ``["ball_pos", "paddle_pos"]``.
    min_bound : float
        Minimum valid coordinate value (inclusive).  Default is 0.0.
    max_bound : float
        Maximum valid coordinate value (inclusive).  Default is 1.0.
    margin : float
        Soft margin — positions within this distance of the boundary
        produce ``"info"`` severity, beyond it ``"warning"`` or
        ``"critical"``.  Default is 0.05.
    """

    def __init__(
        self,
        tracked_keys: list[str] | None = None,
        min_bound: float = 0.0,
        max_bound: float = 1.0,
        margin: float = 0.05,
    ) -> None:
        super().__init__(name="boundary")
        self.tracked_keys = tracked_keys or ["ball_pos", "paddle_pos"]
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.margin = margin
        self._step_count: int = 0

    def on_reset(self, obs: np.ndarray, info: dict[str, Any]) -> None:
        """Reset state at episode start.

        Parameters
        ----------
        obs : np.ndarray
            Initial observation.
        info : dict[str, Any]
            Reset info dict.
        """
        self._step_count = 0

    def on_step(
        self,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        """Check tracked positions for boundary violations.

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
            if pos is None:
                continue

            pos = np.asarray(pos, dtype=np.float64)

            for i, coord in enumerate(pos):
                coord_name = "x" if i == 0 else "y" if i == 1 else f"dim{i}"

                # Hard OOB: beyond boundary + margin
                hard_low = coord < self.min_bound - self.margin
                hard_high = coord > self.max_bound + self.margin

                # Soft OOB: between boundary and boundary + margin
                soft_low = self.min_bound - self.margin <= coord < self.min_bound
                soft_high = self.max_bound < coord <= self.max_bound + self.margin

                if hard_low or hard_high:
                    self._add_finding(
                        severity="critical",
                        step=self._step_count,
                        description=(
                            f"'{key}' {coord_name}={coord:.4f} is out of "
                            f"bounds [{self.min_bound}, {self.max_bound}]"
                        ),
                        data={
                            "type": "hard_oob",
                            "key": key,
                            "coordinate": coord_name,
                            "value": float(coord),
                            "min_bound": self.min_bound,
                            "max_bound": self.max_bound,
                        },
                    )
                elif soft_low or soft_high:
                    self._add_finding(
                        severity="info",
                        step=self._step_count,
                        description=(
                            f"'{key}' {coord_name}={coord:.4f} is near "
                            f"boundary [{self.min_bound}, {self.max_bound}]"
                        ),
                        data={
                            "type": "soft_oob",
                            "key": key,
                            "coordinate": coord_name,
                            "value": float(coord),
                            "min_bound": self.min_bound,
                            "max_bound": self.max_bound,
                        },
                    )
