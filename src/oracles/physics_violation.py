"""Physics Violation Oracle — detects collision and physics bugs.

Monitors object positions and velocities across consecutive steps to
detect tunneling (objects passing through each other), incorrect
reflections, and other physics inconsistencies.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Oracle


class PhysicsViolationOracle(Oracle):
    """Detects physics and collision bugs during gameplay.

    Detection strategy
    ------------------
    1. Track ball and paddle positions across consecutive steps.
    2. If the ball crosses the paddle's Y-line without a velocity
       reversal, flag a tunneling bug.
    3. If the ball overlaps a solid object (paddle/brick/wall) but its
       velocity doesn't reflect, flag a collision failure.
    4. If brick count doesn't decrease when the ball overlaps a brick
       region, flag a missed collision.

    Parameters
    ----------
    ball_pos_key : str
        Key in ``info`` dict for ball position ``[x, y]``.
        Default is ``"ball_pos"``.
    ball_vel_key : str
        Key in ``info`` dict for ball velocity ``[vx, vy]``.
        Default is ``"ball_velocity"``.
    paddle_pos_key : str
        Key in ``info`` dict for paddle position ``[x, y]``.
        Default is ``"paddle_pos"``.
    paddle_size_key : str
        Key in ``info`` dict for paddle size ``[width, height]``.
        Default is ``"paddle_size"``.
    brick_count_key : str
        Key in ``info`` dict for remaining brick count.
        Default is ``"brick_count"``.
    max_ball_speed : float
        Maximum expected ball speed per step (in normalised coordinates).
        If the ball moves faster than this, it may indicate tunneling.
        Default is 0.15.
    """

    def __init__(
        self,
        ball_pos_key: str = "ball_pos",
        ball_vel_key: str = "ball_velocity",
        paddle_pos_key: str = "paddle_pos",
        paddle_size_key: str = "paddle_size",
        brick_count_key: str = "brick_count",
        max_ball_speed: float = 0.15,
    ) -> None:
        super().__init__(name="physics_violation")
        self.ball_pos_key = ball_pos_key
        self.ball_vel_key = ball_vel_key
        self.paddle_pos_key = paddle_pos_key
        self.paddle_size_key = paddle_size_key
        self.brick_count_key = brick_count_key
        self.max_ball_speed = max_ball_speed

        self._prev_ball_pos: np.ndarray | None = None
        self._prev_ball_vel: np.ndarray | None = None
        self._prev_brick_count: int | None = None
        self._step_count: int = 0

    def on_reset(self, obs: np.ndarray, info: dict[str, Any]) -> None:
        """Reset tracking state at episode start.

        Parameters
        ----------
        obs : np.ndarray
            Initial observation.
        info : dict[str, Any]
            Reset info dict.
        """
        self._prev_ball_pos = None
        self._prev_ball_vel = None
        self._prev_brick_count = info.get(self.brick_count_key)
        self._step_count = 0

    def on_step(
        self,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        """Check for physics violations after each step.

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
            Step info dict with position/velocity data.
        """
        self._step_count += 1

        ball_pos = info.get(self.ball_pos_key)
        ball_vel = info.get(self.ball_vel_key)
        paddle_pos = info.get(self.paddle_pos_key)
        brick_count = info.get(self.brick_count_key)

        if ball_pos is not None:
            ball_pos = np.asarray(ball_pos, dtype=np.float64)

            # 1. Check for tunneling via speed
            if self._prev_ball_pos is not None:
                displacement = np.linalg.norm(ball_pos - self._prev_ball_pos)
                if displacement > self.max_ball_speed:
                    self._add_finding(
                        severity="warning",
                        step=self._step_count,
                        description=(
                            f"Ball moved {displacement:.4f} in one step "
                            f"(max expected: {self.max_ball_speed}) — "
                            f"possible tunneling"
                        ),
                        data={
                            "type": "tunneling",
                            "displacement": float(displacement),
                            "max_speed": self.max_ball_speed,
                            "prev_pos": self._prev_ball_pos.tolist(),
                            "curr_pos": ball_pos.tolist(),
                        },
                    )

            # 2. Check paddle collision — ball crosses paddle Y without
            #    velocity Y-component flipping
            if (
                self._prev_ball_pos is not None
                and paddle_pos is not None
                and ball_vel is not None
                and self._prev_ball_vel is not None
            ):
                paddle_y = float(np.asarray(paddle_pos)[1])
                prev_y = float(self._prev_ball_pos[1])
                curr_y = float(ball_pos[1])
                prev_vy = float(np.asarray(self._prev_ball_vel)[1])
                curr_vy = float(np.asarray(ball_vel)[1])

                # Ball crossed paddle Y-line (moving downward)
                crossed_paddle = prev_y < paddle_y <= curr_y or curr_y <= paddle_y < prev_y
                vy_flipped = (prev_vy > 0 and curr_vy < 0) or (prev_vy < 0 and curr_vy > 0)

                if crossed_paddle and not vy_flipped and not terminated:
                    self._add_finding(
                        severity="critical",
                        step=self._step_count,
                        description=(
                            "Ball crossed paddle Y-line without velocity "
                            "reversal — possible collision failure"
                        ),
                        data={
                            "type": "paddle_pass_through",
                            "paddle_y": paddle_y,
                            "prev_ball_y": prev_y,
                            "curr_ball_y": curr_y,
                            "prev_vy": prev_vy,
                            "curr_vy": curr_vy,
                        },
                    )

        # 3. Check brick collision — ball hit brick area but count didn't
        #    decrease (only if velocity reversed, implying collision detected
        #    by the game but brick not removed)
        if (
            brick_count is not None
            and self._prev_brick_count is not None
            and ball_vel is not None
            and self._prev_ball_vel is not None
        ):
            prev_vy = float(np.asarray(self._prev_ball_vel)[1])
            curr_vy = float(np.asarray(ball_vel)[1])
            vy_flipped = (prev_vy > 0 and curr_vy < 0) or (prev_vy < 0 and curr_vy > 0)

            # Velocity flipped (collision happened) but no brick removed
            # and ball is in the upper half (brick region)
            if ball_pos is not None:
                ball_y = float(ball_pos[1])
                in_brick_region = ball_y < 0.5  # Upper half of normalised space
                if (
                    vy_flipped
                    and in_brick_region
                    and brick_count >= self._prev_brick_count
                    and not terminated
                ):
                    self._add_finding(
                        severity="warning",
                        step=self._step_count,
                        description=(
                            "Ball velocity reversed in brick region but brick "
                            "count did not decrease — possible ghost collision"
                        ),
                        data={
                            "type": "ghost_collision",
                            "brick_count": brick_count,
                            "prev_brick_count": self._prev_brick_count,
                            "ball_y": ball_y,
                        },
                    )

        # Update previous state
        self._prev_ball_pos = ball_pos.copy() if ball_pos is not None else None
        if ball_vel is not None:
            self._prev_ball_vel = np.asarray(ball_vel, dtype=np.float64).copy()
        self._prev_brick_count = brick_count
