"""Factory function for creating game loaders from configuration.

The :func:`create_loader` function maps a :class:`GameLoaderConfig`
to the appropriate :class:`GameLoader` subclass based on the
``loader_type`` field.
"""

from __future__ import annotations

import logging

from src.game_loader.base import GameLoader, GameLoaderError
from src.game_loader.config import GameLoaderConfig

logger = logging.getLogger(__name__)

# Registry of loader_type → class.  Kept flat and simple; games that
# need truly custom loaders can register themselves here.
_LOADER_REGISTRY: dict[str, type[GameLoader]] = {}
_REGISTRY_INITIALIZED: bool = False


def _ensure_registry() -> None:
    """Lazily populate the registry to avoid circular imports."""
    global _REGISTRY_INITIALIZED  # noqa: PLW0603
    if _REGISTRY_INITIALIZED:
        return

    from src.game_loader.browser_loader import BrowserGameLoader
    from src.game_loader.breakout71_loader import Breakout71Loader

    _LOADER_REGISTRY.setdefault("browser", BrowserGameLoader)
    _LOADER_REGISTRY.setdefault("breakout-71", Breakout71Loader)
    _REGISTRY_INITIALIZED = True


def create_loader(config: GameLoaderConfig) -> GameLoader:
    """Instantiate the correct :class:`GameLoader` for ``config``.

    Parameters
    ----------
    config : GameLoaderConfig
        Game configuration.  The ``loader_type`` field selects which
        loader class to use.

    Returns
    -------
    GameLoader
        A fully-constructed (but not yet started) loader instance.

    Raises
    ------
    GameLoaderError
        If ``loader_type`` is not recognised.
    """
    _ensure_registry()

    loader_cls = _LOADER_REGISTRY.get(config.loader_type)
    if loader_cls is None:
        raise GameLoaderError(
            f"Unknown loader_type {config.loader_type!r}. "
            f"Available: {sorted(_LOADER_REGISTRY)}"
        )

    logger.info(
        "Creating %s for game %r",
        loader_cls.__name__,
        config.name,
    )
    return loader_cls(config)


def register_loader(name: str, loader_cls: type[GameLoader]) -> None:
    """Register a custom loader class for a given ``loader_type``.

    Parameters
    ----------
    name : str
        The ``loader_type`` value that should map to ``loader_cls``.
    loader_cls : type[GameLoader]
        The loader class.
    """
    _ensure_registry()
    _LOADER_REGISTRY[name] = loader_cls
    logger.info("Registered custom loader %r → %s", name, loader_cls.__name__)
