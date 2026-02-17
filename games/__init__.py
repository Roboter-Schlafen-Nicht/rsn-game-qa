"""Game plugins for the RSN Game QA platform.

Each subdirectory under ``games/`` is a self-contained game plugin
providing the game-specific glue code: environment subclass, loader,
modal handler JS snippets, perception class list, and config files.

Current plugins:

- :mod:`games.breakout71` -- Breakout 71 browser game

Plugin Convention
-----------------
Each plugin's ``__init__.py`` must export these module-level attributes:

- ``env_class`` — the :class:`BaseGameEnv` subclass
- ``loader_class`` — the :class:`GameLoader` subclass (or ``None``)
- ``game_name`` — display / config name (e.g. ``"breakout-71"``)
- ``default_config`` — path to the game config YAML
- ``default_weights`` — path to the YOLO weights file
"""

from __future__ import annotations

import importlib
import types
from typing import Any


def load_game_plugin(name: str) -> types.ModuleType:
    """Dynamically load a game plugin by directory name.

    Parameters
    ----------
    name : str
        Plugin directory name under ``games/`` (e.g. ``"breakout71"``).

    Returns
    -------
    types.ModuleType
        The loaded plugin module.  Guaranteed to have the attributes
        ``env_class``, ``loader_class``, ``game_name``,
        ``default_config``, and ``default_weights``.

    Raises
    ------
    ImportError
        If the plugin module cannot be imported.
    AttributeError
        If the plugin module is missing required attributes.
    """
    module = importlib.import_module(f"games.{name}")

    required = (
        "env_class",
        "loader_class",
        "game_name",
        "default_config",
        "default_weights",
    )
    missing = [attr for attr in required if not hasattr(module, attr)]
    if missing:
        raise AttributeError(
            f"Game plugin 'games.{name}' is missing required attributes: "
            f"{', '.join(missing)}.  See games/__init__.py for the plugin convention."
        )

    return module


def get_env_class(game_name: str) -> type[Any]:
    """Load a game plugin and return its environment class.

    Convenience wrapper around :func:`load_game_plugin`.

    Parameters
    ----------
    game_name : str
        Plugin directory name (e.g. ``"breakout71"``).

    Returns
    -------
    type
        The :class:`BaseGameEnv` subclass for this game.
    """
    plugin = load_game_plugin(game_name)
    return plugin.env_class
