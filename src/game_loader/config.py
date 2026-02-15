"""Game loader configuration data structures.

A :class:`GameLoaderConfig` describes everything the loader needs to
know about a game: where its source lives, how to install dependencies,
how to serve it, and how to tell when it is ready.

Configs can be loaded from YAML files via :func:`load_game_config`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# Default search path for game config YAML files.
_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "games"


@dataclass
class GameLoaderConfig:
    """Declarative description of how to load a game for QA.

    Parameters
    ----------
    name : str
        Human-readable identifier for the game (e.g. ``"breakout-71"``).
    game_dir : str or Path
        Absolute or relative path to the game's source repository.
    loader_type : str
        Which :class:`GameLoader` subclass to use.  Built-in values:
        ``"browser"`` and ``"breakout-71"``.
    install_command : str, optional
        Shell command to install dependencies (e.g. ``"npm install"``).
        Run once during :meth:`GameLoader.setup`.
    serve_command : str
        Shell command to start the game's dev server.
    serve_port : int
        Port the dev server listens on.
    url : str
        Full URL the game is reachable at once the server is up.
    readiness_endpoint : str
        URL to poll (HTTP GET) to determine when the server is ready.
        Defaults to the same value as ``url``.
    readiness_timeout_s : float
        Maximum seconds to wait for the readiness endpoint to respond.
    readiness_poll_interval_s : float
        Seconds between readiness polls.
    window_title : str, optional
        Expected browser window/tab title once the game is loaded.
        Used by ``WindowCapture`` to locate the game surface.
    env_vars : dict[str, str]
        Extra environment variables to set for the serve process.
    """

    name: str
    game_dir: str | Path
    loader_type: str = "browser"

    # Build / serve
    install_command: Optional[str] = "npm install"
    serve_command: str = "npx parcel src/index.html --no-cache"
    serve_port: int = 1234
    url: str = "http://localhost:1234"
    readiness_endpoint: str = ""
    readiness_timeout_s: float = 120.0
    readiness_poll_interval_s: float = 2.0

    # Window identification (for capture layer)
    window_title: Optional[str] = None

    # Extra env for the serve process
    env_vars: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.game_dir = Path(self.game_dir)
        if not self.readiness_endpoint:
            self.readiness_endpoint = self.url


def load_game_config(
    name: str,
    configs_dir: str | Path | None = None,
) -> GameLoaderConfig:
    """Load a :class:`GameLoaderConfig` from a YAML file.

    Searches ``configs_dir`` (default ``configs/games/``) for a file
    named ``<name>.yaml``.

    Parameters
    ----------
    name : str
        Game identifier matching the YAML filename (without extension).
    configs_dir : str or Path, optional
        Override the default config directory.

    Returns
    -------
    GameLoaderConfig

    Raises
    ------
    FileNotFoundError
        If no YAML file is found for ``name``.
    """
    search_dir = Path(configs_dir) if configs_dir else _CONFIGS_DIR
    config_path = search_dir / f"{name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"No game config found at {config_path}. "
            f"Available configs: {[p.stem for p in search_dir.glob('*.yaml')]}"
        )

    logger.info("Loading game config from %s", config_path)
    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    return GameLoaderConfig(**raw)
