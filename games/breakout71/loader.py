"""Breakout 71 game loader — specialisation of :class:`BrowserGameLoader`.

Provides sensible defaults for loading the Breakout 71 browser game
from a local clone of the repository.  The game is built with
**Parcel** into a static ``dist/`` directory and served by Python's
built-in HTTP server.

The two-phase approach (build → serve) avoids Parcel dev-server issues
in headless environments (HMR race conditions, stale module caches)
and produces a deterministic bundle identical to production.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

from src.game_loader.browser_loader import BrowserGameLoader
from src.game_loader.config import GameLoaderConfig

logger = logging.getLogger(__name__)

# Defaults that match the breakout71-testbed repository layout.
_DEFAULTS = dict(
    name="breakout-71",
    loader_type="breakout-71",
    install_command="npm install",
    serve_command=(f"{sys.executable} -m http.server 1234 --directory dist"),
    serve_port=1234,
    url="http://localhost:1234",
    readiness_endpoint="http://localhost:1234",
    readiness_timeout_s=120.0,
    readiness_poll_interval_s=2.0,
    window_title="Breakout",
)

_BUILD_COMMAND = "npx parcel build src/index.html --no-cache"


class Breakout71Loader(BrowserGameLoader):
    """Loader tailored for the Breakout 71 browser game.

    Inherits all behaviour from :class:`BrowserGameLoader` and adds:

    - Sensible defaults for HTTP serving of pre-built Breakout 71 bundles.
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
        window_title: str | None = None,
    ) -> Breakout71Loader:
        """Create a loader from just the repo path, using defaults.

        Parameters
        ----------
        game_dir : str or Path
            Path to the ``breakout71-testbed`` repository.
        serve_port : int
            Port for the HTTP server serving the built game.  Default 1234.
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
        """Install npm dependencies, clear caches, and build the game.

        Runs ``npm install``, removes stale Parcel / dist caches, then
        executes ``npx parcel build`` to produce a fresh ``dist/``
        bundle.  The subsequent ``start()`` serves this static build.
        """
        logger.info("[%s] Clearing .parcel-cache", self.name)
        game_dir = Path(self.config.game_dir)
        cache_dir = game_dir / ".parcel-cache"
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir, ignore_errors=True)
            logger.info("[%s] Removed %s", self.name, cache_dir)

        dist_dir = game_dir / "dist"
        if dist_dir.is_dir():
            shutil.rmtree(dist_dir, ignore_errors=True)
            logger.info("[%s] Removed %s", self.name, dist_dir)

        super().setup()

        # Build the game into dist/
        logger.info("[%s] Building game: %s", self.name, _BUILD_COMMAND)
        result = subprocess.run(
            _BUILD_COMMAND,
            cwd=str(game_dir),
            shell=True,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error(
                "[%s] Build failed (exit %d): %s",
                self.name,
                result.returncode,
                result.stderr,
            )
            raise RuntimeError(
                f"Game build failed: {_BUILD_COMMAND!r} exited with code {result.returncode}"
            )
        logger.info("[%s] Build complete — dist/ ready", self.name)
