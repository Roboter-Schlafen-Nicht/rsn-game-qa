"""Game loader configuration data structures.

A :class:`GameLoaderConfig` describes everything the loader needs to
know about a game: where its source lives, how to install dependencies,
how to serve it, and how to tell when it is ready.

Configs can be loaded from YAML files via :func:`load_game_config`.
String values in YAML configs support environment variable expansion
using ``$VAR`` or ``${VAR}`` syntax, as well as ``~`` for the user
home directory.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# Default search path for game config YAML files.
_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "games"

# Pattern matching $VAR or ${VAR} for environment variable expansion.
_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


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
    window_width : int
        Desired browser window width in pixels.  Used by
        ``BrowserInstance`` to set the initial window size.
    window_height : int
        Desired browser window height in pixels.
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
    window_width: int = 1280
    window_height: int = 720

    # Extra env for the serve process
    env_vars: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.game_dir = Path(os.path.expanduser(_expand_vars(str(self.game_dir))))
        if not self.readiness_endpoint:
            self.readiness_endpoint = self.url


def _expand_vars(value: str) -> str:
    """Expand ``$VAR`` and ``${VAR}`` references in a string.

    Undefined variables are left as-is (no error).

    Parameters
    ----------
    value : str
        String potentially containing environment variable references.

    Returns
    -------
    str
        String with known variables expanded.
    """

    def _replace(match: re.Match) -> str:
        braced = match.group(1)  # From ${...}
        bare = match.group(2)  # From $VAR
        original: str = match.group(0) or ""

        if braced is not None:
            # Support ${VAR:-default} syntax.
            if ":-" in braced:
                var_name, default = braced.split(":-", 1)
                return os.environ.get(var_name, default)
            return os.environ.get(braced, original)

        return os.environ.get(bare or "", original)

    return _ENV_VAR_RE.sub(_replace, value)


def _expand_vars_recursive(data: dict) -> dict:
    """Expand environment variables in all string values of *data*.

    Parameters
    ----------
    data : dict
        Raw YAML dict whose string values may contain ``$VAR`` or
        ``${VAR}`` references.

    Returns
    -------
    dict
        A new dict with all string values expanded.
    """
    expanded: dict = {}
    for key, value in data.items():
        if isinstance(value, str):
            expanded[key] = _expand_vars(value)
        else:
            expanded[key] = value
    return expanded


def load_game_config(
    name: str,
    configs_dir: str | Path | None = None,
) -> GameLoaderConfig:
    """Load a :class:`GameLoaderConfig` from a YAML file.

    Searches ``configs_dir`` (default ``configs/games/``) for a file
    named ``<name>.yaml``.  String values in the YAML undergo
    environment variable expansion (``$VAR`` / ``${VAR}``) and home
    directory expansion (``~``).

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
    ValueError
        If the YAML contains unknown or missing fields.
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

    if not isinstance(raw, dict):
        raise ValueError(
            f"Expected a YAML mapping in {config_path}, got {type(raw).__name__}"
        )

    # Expand environment variables in string values.
    raw = _expand_vars_recursive(raw)

    # Validate fields against the dataclass definition.
    valid_fields = {f.name for f in dataclasses.fields(GameLoaderConfig)}
    unknown = set(raw) - valid_fields
    if unknown:
        raise ValueError(
            f"Unknown fields in {config_path}: {sorted(unknown)}. "
            f"Valid fields: {sorted(valid_fields)}"
        )

    try:
        return GameLoaderConfig(**raw)
    except TypeError as exc:
        raise ValueError(
            f"Invalid config in {config_path}: {exc}. "
            f"Valid fields: {sorted(valid_fields)}"
        ) from exc
