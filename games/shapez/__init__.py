"""shapez.io game plugin for the RSN Game QA platform.

Provides the game-specific implementation for shapez.io, a factory-builder
browser game where the player builds conveyor networks to produce and
deliver shapes to a central hub.

This plugin contains:

- :class:`ShapezEnv` -- Gymnasium environment subclass
- :class:`ShapezLoader` -- Game loader for BrowserSync dev server
- :mod:`~games.shapez.modal_handler` -- JS snippets for DOM interaction
- :mod:`~games.shapez.perception` -- Empty YOLO class list (CNN-only)

Usage::

    from games.shapez import ShapezEnv, ShapezLoader

Plugin metadata (used by ``games.load_game_plugin()``)::

    from games import load_game_plugin
    plugin = load_game_plugin("shapez")
    env = plugin.env_class(...)
"""

from games.shapez.env import ShapezEnv
from games.shapez.loader import ShapezLoader
from games.shapez.modal_handler import LOAD_SAVE_JS, MUTE_JS, SETUP_TRAINING_JS

__all__ = ["ShapezEnv", "ShapezLoader"]

# -- Plugin metadata (required by games.load_game_plugin) ------------------
env_class = ShapezEnv
loader_class = ShapezLoader
game_name = "shapez"
default_config = "configs/games/shapez.yaml"
default_weights = ""  # No YOLO weights — CNN-only

# -- Optional plugin metadata ----------------------------------------------
# JS snippet to mute game audio (executed once before training starts).
mute_js = MUTE_JS

# JS snippet to configure training settings (disable tutorials, etc.).
setup_js = SETUP_TRAINING_JS

# JS snippet to load a savegame from JSON data (arguments[0]).
# Used by SavegameInjector for starting episodes from mid-game states.
load_save_js = LOAD_SAVE_JS
