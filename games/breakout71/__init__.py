"""Breakout 71 game plugin for the RSN Game QA platform.

Provides the game-specific implementation for Breakout 71, a roguelite
browser-based breakout game.  This plugin contains:

- :class:`Breakout71Env` -- Gymnasium environment subclass
- :class:`Breakout71Loader` -- Game loader for the Parcel dev server
- :mod:`~games.breakout71.modal_handler` -- JS snippets for DOM modal handling
- :mod:`~games.breakout71.perception` -- YOLO class names for Breakout 71
- Config files: ``config.yaml`` (loader) and ``training.yaml`` (YOLO training)

Usage::

    from games.breakout71 import Breakout71Env, Breakout71Loader
"""

from games.breakout71.env import Breakout71Env
from games.breakout71.loader import Breakout71Loader

__all__ = ["Breakout71Env", "Breakout71Loader"]
