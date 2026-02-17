"""Environment module — Gymnasium environments for game QA testing.

Re-exports selected platform classes from :mod:`src.platform` for
limited backward compatibility.  Legacy code using
``from src.env import BaseGameEnv`` or
``from src.env import CnnObservationWrapper`` will continue to work.
New code should import directly from :mod:`src.platform`, and the old
module paths :mod:`src.env.base_env` and :mod:`src.env.cnn_wrapper`
are no longer supported.
"""

from src.platform.base_env import BaseGameEnv
from src.platform.cnn_wrapper import CnnObservationWrapper

from .breakout71_env import Breakout71Env

__all__ = ["BaseGameEnv", "Breakout71Env", "CnnObservationWrapper"]
