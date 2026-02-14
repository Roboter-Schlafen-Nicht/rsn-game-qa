# env_help.py
"""
Simulated RL environment for learning when to CLICK_HELP vs NOOP vs RANDOM_SWIPE.

This is a cheap digital twin for rapid iteration on:
- action space: {NOOP, CLICK_HELP, RANDOM_SWIPE}
- observation design: [has_help, help_conf, since_last_help_norm]
- reward shaping

Later, you can replace the internal simulator with the real game loop
(LastWarController + HelpOnlyPolicy) while keeping the same API.
"""

from __future__ import annotations

import random
from typing import Tuple, Dict, Any

import gymnasium as gym
import numpy as np


class LastWarHelpEnv(gym.Env):
    """
    A tiny MDP for the HELP button.

    State (hidden, internal):
        - help_visible: bool
        - help_conf: float in [0, 1] when visible, 0 otherwise
        - since_last_help: float (seconds since last successful help click)

    Observation (what the policy sees):
        obs = [has_help, help_conf, since_last_help_norm]
        where:
          has_help ∈ {0.0, 1.0}
          help_conf ∈ [0.0, 1.0]
          since_last_help_norm ∈ [0.0, 1.0], clipped

    Actions:
        0 = NOOP
        1 = CLICK_HELP
        2 = RANDOM_SWIPE
    """

    metadata = {"render_modes": []}

    ACTION_NOOP = 0
    ACTION_CLICK_HELP = 1
    ACTION_RANDOM_SWIPE = 2

    def __init__(
        self,
        max_episode_steps: int = 200,
        help_appear_prob: float = 0.05,
        help_disappear_prob: float = 0.05,
        swipe_find_help_prob: float = 0.3,
        since_last_clip: float = 30.0,
    ) -> None:
        super().__init__()

        self.max_episode_steps = max_episode_steps
        self.help_appear_prob = help_appear_prob
        self.help_disappear_prob = help_disappear_prob
        self.swipe_find_help_prob = swipe_find_help_prob
        self.since_last_clip = since_last_clip

        # Observation: 3 floats in [0, 1]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Action: 0=NOOP, 1=CLICK_HELP, 2=RANDOM_SWIPE
        self.action_space = gym.spaces.Discrete(3)

        # Internal state
        self._help_visible: bool = False
        self._help_conf: float = 0.0
        self._since_last_help: float = 0.0
        self._t: int = 0

    # --------------------------------------------------------------------- utils

    def _get_obs(self) -> np.ndarray:
        has_help = 1.0 if self._help_visible else 0.0
        conf = self._help_conf if self._help_visible else 0.0
        since_norm = (
            min(self._since_last_help, self.since_last_clip) / self.since_last_clip
        )
        return np.array([has_help, conf, since_norm], dtype=np.float32)

    def _transition_help_visibility(self, action: int) -> None:
        """
        Simple stochastic dynamics for help_visible / help_conf.
        """
        # Base "clock": HELP might appear spontaneously when not visible.
        if not self._help_visible:
            if random.random() < self.help_appear_prob:
                self._help_visible = True
                # When it appears, give it a reasonably high conf
                self._help_conf = random.uniform(0.7, 1.0)
        else:
            # HELP visible: chance it disappears on its own (e.g., timeout)
            if random.random() < self.help_disappear_prob:
                self._help_visible = False
                self._help_conf = 0.0

        # Effect of RANDOM_SWIPE: chance to reveal HELP
        if action == self.ACTION_RANDOM_SWIPE and not self._help_visible:
            if random.random() < self.swipe_find_help_prob:
                self._help_visible = True
                self._help_conf = random.uniform(0.6, 0.9)

    # ----------------------------------------------------------------- gym API

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._t = 0
        self._since_last_help = 0.0

        # Randomize initial HELP visibility a bit
        self._help_visible = random.random() < 0.2
        self._help_conf = random.uniform(0.6, 0.9) if self._help_visible else 0.0

        return self._get_obs(), {}

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action)

        self._t += 1
        done = False
        truncated = False
        info: Dict[str, Any] = {}

        # Time passes
        self._since_last_help += 1.0  # treat each step as 1 "second" for now

        reward = 0.0

        # --- Reward logic ---------------------------------------------------
        if action == self.ACTION_CLICK_HELP:
            if self._help_visible:
                # Successful help click
                reward += 1.0
                self._help_visible = False
                self._help_conf = 0.0
                self._since_last_help = 0.0
                info["event"] = "click_success"
            else:
                # Useless click
                reward -= 0.3
                info["event"] = "click_wasted"

        elif action == self.ACTION_NOOP:
            if self._help_visible:
                # Missing an opportunity
                reward -= 0.1
                info["event"] = "noop_missed"
            else:
                # Tiny time cost
                reward -= 0.01
                info["event"] = "noop_idle"

        elif action == self.ACTION_RANDOM_SWIPE:
            # Exploration is slightly costly
            reward -= 0.05
            info["event"] = "swipe"

        # --- State transition for HELP visibility ---------------------------
        self._transition_help_visibility(action)

        # Optional small step-wise time penalty to encourage efficiency
        reward -= 0.005

        if self._t >= self.max_episode_steps:
            truncated = True

        obs = self._get_obs()
        return obs, float(reward), done, truncated, info

    def render(self):
        # For now, just print state
        print(
            f"t={self._t} help_visible={self._help_visible} "
            f"conf={self._help_conf:.2f} since_last={self._since_last_help:.1f}"
        )
