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

Plugin metadata (used by ``games.load_game_plugin()``)::

    from games import load_game_plugin
    plugin = load_game_plugin("breakout71")
    env = plugin.env_class(...)
"""

from games.breakout71.env import Breakout71Env
from games.breakout71.loader import Breakout71Loader

__all__ = ["Breakout71Env", "Breakout71Loader"]

# -- Plugin metadata (required by games.load_game_plugin) ------------------
env_class = Breakout71Env
loader_class = Breakout71Loader
game_name = "breakout-71"
default_config = "configs/games/breakout-71.yaml"
default_weights = "weights/breakout71/best.pt"

# -- Optional plugin metadata ----------------------------------------------
# JS snippet to mute game audio (executed once before training starts).
# If not present, no mute action is taken.
mute_js = 'localStorage.setItem("breakout-settings-enable-sound", "false")'

# JS snippet to configure game settings for RL training.
# Disables mobile-mode (touch-hold UI) and delayed start (3-second countdown)
# so that a single mouseup event starts the game immediately.
# Executed once before training starts; requires a page refresh to take effect.
setup_js = "\n".join(
    [
        'localStorage.setItem("breakout-settings-enable-mobile-mode", "false");',
        'localStorage.setItem("breakout-settings-enable-touch_delayed_start", "false");',
    ]
)

# JS snippet to re-initialise the game after a page refresh.
# Calls the exposed window.restart() hook which re-creates the game state
# AND re-assigns window.gameState to the live module-scoped object (fixing
# the pointer mismatch caused by newGameState() / Object.assign in restart).
# Must be executed AFTER the page has fully loaded.
reinit_js = "window.restart({})"
