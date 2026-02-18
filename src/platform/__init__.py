"""Platform module — game-agnostic infrastructure for RL-driven game QA."""

from .base_env import BaseGameEnv
from .cnn_wrapper import CnnEvalWrapper, CnnObservationWrapper

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
    "RNDRewardWrapper",
]
