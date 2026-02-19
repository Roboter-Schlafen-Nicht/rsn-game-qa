"""Hextris game loader -- specialisation of :class:`BrowserGameLoader`.

Hextris is a pure static HTML5 game with no build step.  The loader
simply serves the cloned repository directory via Python's built-in
HTTP server.

Usage::

    git clone https://github.com/Hextris/hextris.git
    export HEXTRIS_DIR=/path/to/hextris

Then::

    from games.hextris.loader import HextrisLoader
    loader = HextrisLoader.from_repo_path("/path/to/hextris")
    loader.setup()   # no-op (no build step)
    loader.start()   # starts http.server on port 8271
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.game_loader.browser_loader import BrowserGameLoader
from src.game_loader.config import GameLoaderConfig

logger = logging.getLogger(__name__)

# Defaults that match the Hextris repository layout.
# Hextris is pure static HTML — serve the repo root directly.
_DEFAULTS = dict(
    name="hextris",
    loader_type="hextris",
    install_command=None,  # No build step needed
    serve_command=(f"{sys.executable} -m http.server 8271"),
    serve_port=8271,
    url="http://localhost:8271",
    readiness_endpoint="http://localhost:8271",
    readiness_timeout_s=30.0,
    readiness_poll_interval_s=1.0,
    window_title="HEXTRIS",
    window_width=1280,
    window_height=1024,
)


class HextrisLoader(BrowserGameLoader):
    """Loader tailored for the Hextris browser game.

    Inherits all behaviour from :class:`BrowserGameLoader`.  Since
    Hextris is a pure static site (no npm, no bundler), ``setup()``
    is a no-op and ``start()`` simply launches ``python -m http.server``
    in the repository root.

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
        serve_port: int = 8271,
        readiness_timeout_s: float = 30.0,
        window_title: str | None = None,
    ) -> HextrisLoader:
        """Create a loader from just the repo path, using defaults.

        Parameters
        ----------
        game_dir : str or Path
            Path to the cloned ``hextris`` repository.
        serve_port : int
            Port for the HTTP server.  Default 8271.
        readiness_timeout_s : float
            How long to wait for the server to come up.
        window_title : str, optional
            Browser window title override.

        Returns
        -------
        HextrisLoader
        """
        url = f"http://localhost:{serve_port}"
        serve_cmd = f"{sys.executable} -m http.server {serve_port}"
        config = GameLoaderConfig(
            **{
                **_DEFAULTS,
                "game_dir": str(game_dir),
                "serve_port": serve_port,
                "serve_command": serve_cmd,
                "url": url,
                "readiness_endpoint": url,
                "readiness_timeout_s": readiness_timeout_s,
                "window_title": window_title or _DEFAULTS["window_title"],
            }
        )
        return cls(config)

    def setup(self) -> None:
        """Validate game directory — Hextris has no build step.

        The game is pure static HTML/JS/CSS.  This method validates
        that the game directory exists and contains ``index.html``.
        """
        game_dir = Path(self.config.game_dir)
        if not game_dir.is_dir():
            raise FileNotFoundError(
                f"Hextris game directory does not exist: {game_dir}. "
                f"Clone it with: git clone https://github.com/Hextris/hextris.git"
            )

        index = game_dir / "index.html"
        if not index.exists():
            raise FileNotFoundError(
                f"No index.html found in {game_dir}. Expected the root of the Hextris repository."
            )

        logger.info("[%s] Game directory validated: %s", self.name, game_dir)
