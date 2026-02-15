"""Game loader subsystem — launch and manage game processes for QA testing.

Provides a configurable, extensible mechanism for starting games as a
prerequisite to RL training or automated QA.  Each game is described by
a :class:`GameLoaderConfig` and managed by a :class:`GameLoader`
subclass that knows how to build, serve, and health-check it.

Typical usage::

    from src.game_loader import load_game_config, create_loader

    config = load_game_config("breakout-71")
    loader = create_loader(config)
    loader.start()          # install deps, launch dev server, wait for ready
    # … run QA / RL against loader.url …
    loader.stop()
"""

from src.game_loader.config import GameLoaderConfig, load_game_config
from src.game_loader.base import GameLoader, GameLoaderError
from src.game_loader.browser_loader import BrowserGameLoader
from src.game_loader.breakout71_loader import Breakout71Loader
from src.game_loader.factory import create_loader

__all__ = [
    "GameLoaderConfig",
    "load_game_config",
    "GameLoader",
    "GameLoaderError",
    "BrowserGameLoader",
    "Breakout71Loader",
    "create_loader",
]
