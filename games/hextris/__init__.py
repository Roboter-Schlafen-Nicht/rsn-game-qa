"""Hextris game plugin for the RSN Game QA platform.

Provides the game-specific implementation for Hextris, a hexagonal
puzzle browser game where colored blocks fall toward a central hexagon
and the player rotates it to match colors on each side.

This plugin contains:

- :class:`HextrisEnv` -- Gymnasium environment subclass
- :class:`HextrisLoader` -- Game loader for static HTTP serving
- :mod:`~games.hextris.modal_handler` -- JS snippets for DOM interaction
- :mod:`~games.hextris.perception` -- Empty YOLO class list (CNN-only)

Usage::

    from games.hextris import HextrisEnv, HextrisLoader

Plugin metadata (used by ``games.load_game_plugin()``)::

    from games import load_game_plugin
    plugin = load_game_plugin("hextris")
    env = plugin.env_class(...)
"""

from games.hextris.env import HextrisEnv
from games.hextris.loader import HextrisLoader

__all__ = ["HextrisEnv", "HextrisLoader"]

# -- Plugin metadata (required by games.load_game_plugin) ------------------
env_class = HextrisEnv
loader_class = HextrisLoader
game_name = "hextris"
default_config = "configs/games/hextris.yaml"
default_weights = ""  # No YOLO weights — CNN-only

# -- Optional plugin metadata ----------------------------------------------
# JS snippet to mute game audio (executed once before training starts).
mute_js = """(function() {
    Audio.prototype.play = function() { return Promise.resolve(); };
    if (typeof Howler !== 'undefined') { Howler.mute(true); }
})();"""

# No setup_js needed — Hextris has no training-specific settings to configure.
# No reinit_js needed — init(1) handles game restart (done in DISMISS_GAME_OVER_JS).
