"""Platform module — game-agnostic infrastructure for RL-driven game QA."""

from .base_env import BaseGameEnv
from .cnn_wrapper import CnnEvalWrapper, CnnObservationWrapper

__all__ = ["BaseGameEnv", "CnnEvalWrapper", "CnnObservationWrapper"]
