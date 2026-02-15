"""Breakout 71 game loader — specialisation of :class:`BrowserGameLoader`.

Provides sensible defaults for loading the Breakout 71 browser game
from a local clone of the repository.  The game uses **Parcel** as its
bundler and serves on ``http://localhost:1234`` by default.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.game_loader.browser_loader import BrowserGameLoader
from src.game_loader.config import GameLoaderConfig

logger = logging.getLogger(__name__)

# Defaults that match the breakout71-testbed repository layout.
_DEFAULTS = dict(
    name="breakout-71",
    loader_type="breakout-71",
    install_command="npm install",
    serve_command="npx parcel src/index.html --no-cache",
    serve_port=1234,
    url="http://localhost:1234",
    readiness_endpoint="http://localhost:1234",
    readiness_timeout_s=120.0,
    readiness_poll_interval_s=2.0,
    window_title="Breakout",
)


class Breakout71Loader(BrowserGameLoader):
    """Loader tailored for the Breakout 71 browser game.

    Inherits all behaviour from :class:`BrowserGameLoader` and adds:

    - Sensible defaults for Parcel-based serving of Breakout 71.
    - Convenience constructor :meth:`from_repo_path` that only needs
      the path to the game repository.

    Parameters
    ----------
    config : GameLoaderConfig
        Full configuration.  Consider using :meth:`from_repo_path`
        for a simpler API.
    """

    @classmethod
    def from_repo_path(
        cls,
        game_dir: str | Path,
        *,
        serve_port: int = 1234,
        readiness_timeout_s: float = 120.0,
        window_title: Optional[str] = None,
    ) -> "Breakout71Loader":
        """Create a loader from just the repo path, using defaults.

        Parameters
        ----------
        game_dir : str or Path
            Path to the ``breakout71-testbed`` repository.
        serve_port : int
            Port for the Parcel dev server.  Default 1234.
        readiness_timeout_s : float
            How long to wait for the server to come up.
        window_title : str, optional
            Browser window title override.

        Returns
        -------
        Breakout71Loader
        """
        url = f"http://localhost:{serve_port}"
        config = GameLoaderConfig(
            **{
                **_DEFAULTS,
                "game_dir": str(game_dir),
                "serve_port": serve_port,
                "url": url,
                "readiness_endpoint": url,
                "readiness_timeout_s": readiness_timeout_s,
                "window_title": window_title or _DEFAULTS["window_title"],
            }
        )
        return cls(config)

    def setup(self) -> None:
        """Install npm dependencies and clear the Parcel cache."""
        logger.info("[%s] Clearing .parcel-cache", self.name)
        cache_dir = self.config.game_dir / ".parcel-cache"
        if cache_dir.is_dir():
            import shutil

            shutil.rmtree(cache_dir, ignore_errors=True)
            logger.info("[%s] Removed %s", self.name, cache_dir)

        super().setup()
