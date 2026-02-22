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
except ImportError:  # pragma: no cover
    pass

__all__ = [
    "BaseGameEnv",
    "CnnEvalWrapper",
    "CnnObservationWrapper",
    "EntropyCollapseStrategy",
    "EpsilonGreedyWrapper",
    "EventRecorder",
    "GameOverDetector",
    "GameOverStrategy",
    "MotionCessationStrategy",
    "RNDRewardWrapper",
    "SavegameInjector",
    "SavegamePool",
    "ScoreOCR",
    "ScreenFreezeStrategy",
    "TextDetectionStrategy",
]
