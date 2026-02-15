"""Abstract base class for game loaders.

A :class:`GameLoader` manages the full lifecycle of getting a game
running and reachable so that the QA / RL layer can interact with it.

Subclasses implement the concrete mechanics (e.g. starting a Node dev
server, launching an emulator, opening a native executable, etc.).
"""

from __future__ import annotations

import abc
import logging
from typing import Optional

from src.game_loader.config import GameLoaderConfig

logger = logging.getLogger(__name__)


class GameLoader(abc.ABC):
    """Base class for all game loaders.

    The loader lifecycle is:

    1. :meth:`setup`  — one-time preparation (install deps, build, etc.)
    2. :meth:`start`  — launch the game process and wait until reachable
    3. :meth:`is_ready` — check if the game is responding
    4. :meth:`stop`   — tear down the game process and free resources

    Parameters
    ----------
    config : GameLoaderConfig
        Declarative configuration for the game.
    """

    def __init__(self, config: GameLoaderConfig) -> None:
        self.config = config
        self._running: bool = False

    # -- Properties ----------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable game identifier."""
        return self.config.name

    @property
    def url(self) -> Optional[str]:
        """URL where the running game can be reached, or ``None``."""
        return self.config.url if self._running else None

    @property
    def running(self) -> bool:
        """Whether the game process is currently running."""
        return self._running

    # -- Lifecycle -----------------------------------------------------

    @abc.abstractmethod
    def setup(self) -> None:
        """One-time setup: install dependencies, compile, etc.

        Called before :meth:`start`.  Implementations should be
        idempotent (safe to call multiple times).
        """

    @abc.abstractmethod
    def start(self) -> None:
        """Launch the game and block until it is ready.

        After this method returns, :attr:`url` must point to a
        reachable game instance and :attr:`running` must be ``True``.

        Raises
        ------
        GameLoaderError
            If the game fails to start within the configured timeout.
        """

    @abc.abstractmethod
    def is_ready(self) -> bool:
        """Return ``True`` if the game is responding and playable."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Shut down the game process and release resources.

        After this method returns, :attr:`running` must be ``False``.
        Must be safe to call even if the game was never started.
        """

    # -- Context manager -----------------------------------------------

    def __enter__(self) -> "GameLoader":
        self.setup()
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:  # noqa: ANN001
        self.stop()
        return False

    # -- Repr -----------------------------------------------------------

    def __repr__(self) -> str:
        status = "running" if self._running else "stopped"
        return f"<{type(self).__name__}({self.config.name!r}, {status})>"


class GameLoaderError(Exception):
    """Raised when a game loader encounters an unrecoverable error."""
