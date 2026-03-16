"""Platform module — game-agnostic infrastructure for RL-driven game QA."""

from .base_env import BaseGameEnv
from .cnn_wrapper import CnnEvalWrapper, CnnObservationWrapper
from .epsilon_greedy_wrapper import EpsilonGreedyWrapper
from .event_recorder import EventRecorder
from .game_over_detector import (
    EntropyCollapseStrategy,
    GameOverDetector,
    GameOverStrategy,
    MotionCessationStrategy,
    ScreenFreezeStrategy,
    TextDetectionStrategy,
)
from .savegame_injector import SavegameInjector, SavegamePool
from .score_ocr import ScoreOCR

# RNDRewardWrapper requires torch; lazy-import to avoid pulling in torch
# unconditionally (CI/docs environments may not have it installed).
try:
    from .rnd_wrapper import RNDRewardWrapper
except Exception:  # pragma: no cover - torch may raise AttributeError in CI
    pass

# LTC (CfC) modules require torch + ncps + sb3-contrib; lazy-import.
try:
    from .ltc_policy import CnnCfCPolicy
    from .ltc_wrapper import LtcEvalWrapper
except Exception:  # pragma: no cover - torch/multiprocessing may raise AttributeError
    pass

__all__ = [
    "BaseGameEnv",
    "CnnCfCPolicy",
    "CnnEvalWrapper",
    "CnnObservationWrapper",
    "EntropyCollapseStrategy",
    "EpsilonGreedyWrapper",
    "EventRecorder",
    "GameOverDetector",
    "GameOverStrategy",
    "LtcEvalWrapper",
    "MotionCessationStrategy",
    "RNDRewardWrapper",
    "SavegameInjector",
    "SavegamePool",
    "ScoreOCR",
    "ScreenFreezeStrategy",
    "TextDetectionStrategy",
]
