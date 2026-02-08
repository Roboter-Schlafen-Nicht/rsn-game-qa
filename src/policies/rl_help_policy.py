# policies/rl_help_policy.py
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO


class RLHelpPolicy:
    """
    Thin wrapper around PPO help policy trained in train_help.py.
    Input obs: [has_help, help_conf, since_last_help_norm].
    """

    def __init__(self, model_path: str, since_last_clip: float = 60.0, device: str = "cpu"):
        self.model_path = Path(model_path)
        self.model = PPO.load(str(self.model_path), device=device)
        self.since_last_clip = since_last_clip

        self.last_help_ts: float | None = None  # wall-clock seconds when last click_help happened


    def build_obs(self, det: dict | None, now: float) -> np.ndarray:
        has_help = 1.0 if det is not None else 0.0
        help_conf = float(det["conf"]) if det is not None else 0.0

        if self.last_help_ts is None:
            since_last = 0.0
        else:
            since_last = max(0.0, now - self.last_help_ts)

        since_last_norm = min(since_last / self.since_last_clip, 1.0)
        return np.array([has_help, help_conf, since_last_norm], dtype=np.float32)


    def decide(self, det: dict | None, now: float) -> int:
        """
        Return discrete action id used in env:
          0 = NOOP, 1 = CLICK_HELP, 2 = RANDOM_SWIPE
        """
        obs = self.build_obs(det, now)
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


    def notify_help_clicked(self, now: float) -> None:
        self.last_help_ts = now
