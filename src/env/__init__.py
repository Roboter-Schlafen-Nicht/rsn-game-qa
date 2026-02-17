"""Environment module — Gymnasium environments for game QA testing.

Re-exports platform classes from :mod:`src.platform` for backward
compatibility.  New code should import directly from
:mod:`src.platform`.
"""

from src.platform.base_env import BaseGameEnv
from src.platform.cnn_wrapper import CnnObservationWrapper

from .breakout71_env import Breakout71Env

__all__ = ["BaseGameEnv", "Breakout71Env", "CnnObservationWrapper"]
