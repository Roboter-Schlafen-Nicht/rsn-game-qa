"""Environment module — Gymnasium environments for game QA testing.

Re-exports selected platform classes from :mod:`src.platform` and
game-specific classes from :mod:`games.breakout71` for backward
compatibility.  Legacy code using ``from src.env import BaseGameEnv``,
``from src.env import CnnObservationWrapper``, or
``from src.env import Breakout71Env`` will continue to work.

New code should import directly from the canonical locations:

- Platform classes: :mod:`src.platform`
- Game-specific classes: :mod:`games.breakout71`
"""

from typing import Any

from src.platform.base_env import BaseGameEnv
from src.platform.cnn_wrapper import CnnObservationWrapper

__all__ = ["BaseGameEnv", "Breakout71Env", "CnnObservationWrapper"]


def __getattr__(name: str) -> Any:
    """Lazy import for backward compatibility with moved modules."""
    if name == "Breakout71Env":
        from games.breakout71.env import Breakout71Env

        return Breakout71Env
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
