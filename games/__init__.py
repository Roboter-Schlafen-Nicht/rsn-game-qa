"""Game plugins for the RSN Game QA platform.

Each subdirectory under ``games/`` is a self-contained game plugin
providing the game-specific glue code: environment subclass, loader,
modal handler JS snippets, perception class list, and config files.

Current plugins:

- :mod:`games.breakout71` -- Breakout 71 browser game
- :mod:`games.hextris` -- Hextris hexagonal puzzle game
- :mod:`games.shapez` -- shapez.io factory builder game

Plugin Convention
-----------------
Each plugin's ``__init__.py`` must export these module-level attributes:

- ``env_class`` — the :class:`BaseGameEnv` subclass
- ``loader_class`` — the :class:`GameLoader` subclass (or ``None``)
- ``game_name`` — display / config name (e.g. ``"breakout-71"``)
- ``default_config`` — path to the game config YAML
- ``default_weights`` — path to the YOLO weights file

Registry
--------
Plugins can be discovered automatically via :func:`discover_plugins` or
registered explicitly via :func:`register_game`.  :func:`load_game_plugin`
checks the registry first and falls back to ``importlib`` for backward
compatibility.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REQUIRED_ATTRS = (
    "env_class",
    "loader_class",
    "game_name",
    "default_config",
    "default_weights",
)


@dataclass(frozen=True)
class PluginEntry:
    """Immutable record for a registered game plugin.

    Parameters
    ----------
    env_class : type
        The :class:`BaseGameEnv` subclass.
    loader_class : type | None
        The :class:`GameLoader` subclass, or ``None``.
    game_name : str
        Display / config name (e.g. ``"breakout-71"``).
    default_config : str
        Path to the game config YAML.
    default_weights : str
        Path to the YOLO weights file (empty string if none).
    extra : dict[str, Any]
        Additional optional metadata (e.g. ``mute_js``, ``setup_js``).
    """

    env_class: type
    loader_class: type | None
    game_name: str
    default_config: str
    default_weights: str
    extra: dict[str, Any] = field(default_factory=dict)


class GameRegistry:
    """Registry of game plugins.

    Stores :class:`PluginEntry` instances keyed by plugin directory name
    (e.g. ``"breakout71"``).  Supports explicit registration and automatic
    discovery via :func:`discover_plugins`.
    """

    def __init__(self) -> None:
        self._entries: dict[str, PluginEntry] = {}

    def register(
        self,
        name: str,
        *,
        env_class: type,
        loader_class: type | None,
        game_name: str,
        default_config: str,
        default_weights: str,
        **extra: Any,
    ) -> None:
        """Register a game plugin.

        Parameters
        ----------
        name : str
            Plugin directory name (e.g. ``"breakout71"``).
        env_class : type
            The :class:`BaseGameEnv` subclass.
        loader_class : type | None
            The :class:`GameLoader` subclass, or ``None``.
        game_name : str
            Display / config name.
        default_config : str
            Path to game config YAML.
        default_weights : str
            Path to YOLO weights file.
        **extra
            Additional optional metadata stored in
            :attr:`PluginEntry.extra`.

        Raises
        ------
        ValueError
            If *name* is already registered.
        """
        if name in self._entries:
            raise ValueError(
                f"Game plugin '{name}' is already registered.  "
                "Use a unique name or call clear() first."
            )
        self._entries[name] = PluginEntry(
            env_class=env_class,
            loader_class=loader_class,
            game_name=game_name,
            default_config=default_config,
            default_weights=default_weights,
            extra=dict(extra) if extra else {},
        )

    def get(self, name: str) -> PluginEntry | None:
        """Return the entry for *name*, or ``None`` if not registered."""
        return self._entries.get(name)

    def list(self) -> list[str]:
        """Return registered plugin names in sorted order."""
        return sorted(self._entries)

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()

    def __contains__(self, name: str) -> bool:
        return name in self._entries


# -- Module-level singleton registry ----------------------------------------

_registry = GameRegistry()


def register_game(
    name: str,
    *,
    env_class: type,
    loader_class: type | None,
    game_name: str,
    default_config: str,
    default_weights: str,
    **extra: Any,
) -> None:
    """Register a game plugin in the global registry.

    Convenience wrapper around ``_registry.register()``.

    Parameters
    ----------
    name : str
        Plugin directory name (e.g. ``"breakout71"``).
    env_class : type
        The :class:`BaseGameEnv` subclass.
    loader_class : type | None
        The :class:`GameLoader` subclass, or ``None``.
    game_name : str
        Display / config name.
    default_config : str
        Path to game config YAML.
    default_weights : str
        Path to YOLO weights file.
    **extra
        Additional optional metadata.
    """
    _registry.register(
        name,
        env_class=env_class,
        loader_class=loader_class,
        game_name=game_name,
        default_config=default_config,
        default_weights=default_weights,
        **extra,
    )


def list_games() -> list[str]:
    """Return the names of all registered game plugins (sorted)."""
    return _registry.list()


def discover_plugins(registry: GameRegistry | None = None) -> None:
    """Auto-discover game plugins under the ``games/`` package.

    Iterates over sub-packages of ``games/`` and imports each one.
    If the imported module exposes the required plugin attributes,
    it is registered in *registry* (defaults to the global
    ``_registry``).

    Already-registered plugins are silently skipped, making this
    function idempotent.

    Parameters
    ----------
    registry : GameRegistry | None
        Target registry.  Defaults to the module-level ``_registry``.
    """
    if registry is None:
        registry = _registry

    games_path = str(Path(__file__).parent)

    for _importer, module_name, is_pkg in pkgutil.iter_modules([games_path]):
        if not is_pkg:
            continue
        if module_name.startswith("_"):
            continue
        if module_name in registry:
            continue

        try:
            module = importlib.import_module(f"games.{module_name}")
        except Exception:
            logger.warning("Failed to import game plugin '%s'", module_name)
            continue

        missing = [a for a in _REQUIRED_ATTRS if not hasattr(module, a)]
        if missing:
            logger.warning(
                "Game plugin '%s' missing required attrs: %s — skipping",
                module_name,
                ", ".join(missing),
            )
            continue

        # Collect optional extra attributes
        extra: dict[str, Any] = {}
        for attr_name in dir(module):
            if attr_name.startswith("_") or attr_name in _REQUIRED_ATTRS:
                continue
            # Skip classes, modules, and common non-metadata names
            val = getattr(module, attr_name)
            if isinstance(val, (type, types.ModuleType)):
                continue
            if attr_name == "__all__":
                continue
            if attr_name in (
                "mute_js",
                "setup_js",
                "reinit_js",
                "load_save_js",
                "default_score_region",
            ):
                extra[attr_name] = val

        registry.register(
            module_name,
            env_class=module.env_class,
            loader_class=module.loader_class,
            game_name=module.game_name,
            default_config=module.default_config,
            default_weights=module.default_weights,
            **extra,
        )


def load_game_plugin(name: str) -> types.ModuleType:
    """Dynamically load a game plugin by directory name.

    Checks the global registry first.  If the plugin is registered,
    returns the actual module (imported via ``importlib``).  Otherwise
    falls back to direct ``importlib.import_module`` with attribute
    validation.

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
    # Fast path: if the module is already imported, return it
    module_key = f"games.{name}"
    if module_key in sys.modules:
        module = sys.modules[module_key]
        missing = [a for a in _REQUIRED_ATTRS if not hasattr(module, a)]
        if not missing:
            return module

    # Import the module (may trigger register_game() in plugin __init__)
    module = importlib.import_module(module_key)

    missing = [attr for attr in _REQUIRED_ATTRS if not hasattr(module, attr)]
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
