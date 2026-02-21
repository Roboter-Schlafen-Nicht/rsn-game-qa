"""shapez.io game loader -- specialisation of :class:`BrowserGameLoader`.

shapez.io is a factory-builder browser game that requires a build
toolchain (Node.js 16, Yarn, Java, ffmpeg).  The build and dev server
are managed by gulp + BrowserSync in the ``gulp/`` subdirectory.

Usage::

    git clone https://github.com/tobspr-games/shapez.io
    export SHAPEZ_DIR=/path/to/shapez.io

Then::

    from games.shapez.loader import ShapezLoader
    loader = ShapezLoader.from_repo_path("/path/to/shapez.io")
    loader.setup()   # yarn install in root + gulp/
    loader.start()   # yarn gulp (BrowserSync on port 3005)
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from src.game_loader.browser_loader import BrowserGameLoader
from src.game_loader.config import GameLoaderConfig

logger = logging.getLogger(__name__)

# Shell preamble to activate nvm and switch to Node.js 16.
# Required because shapez.io needs Node 16 (incompatible with newer).
_NVM_PREAMBLE = (
    'export NVM_DIR="$HOME/.nvm" && '
    '[ -s "$NVM_DIR/nvm.sh" ] && \\. "$NVM_DIR/nvm.sh" && '
    "nvm use 16 --silent && "
)

# Defaults that match the shapez.io repository layout.
_DEFAULTS = dict(
    name="shapez",
    loader_type="shapez",
    install_command=None,  # Handled in setup() with nvm
    serve_command=None,  # Handled in start() override
    serve_port=3005,
    url="http://localhost:3005",
    readiness_endpoint="http://localhost:3005",
    readiness_timeout_s=120.0,
    readiness_poll_interval_s=2.0,
    window_title="shapez.io",
    window_width=1280,
    window_height=1024,
)


class ShapezLoader(BrowserGameLoader):
    """Loader tailored for the shapez.io browser game.

    Inherits all behaviour from :class:`BrowserGameLoader` and adds:

    - ``setup()`` installs dependencies via ``yarn`` in both the
      repository root and the ``gulp/`` subdirectory, using nvm to
      ensure Node.js 16 is active.
    - ``start()`` launches ``yarn gulp`` (BrowserSync dev server)
      in the ``gulp/`` subdirectory.

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
        serve_port: int = 3005,
        readiness_timeout_s: float = 120.0,
        window_title: str | None = None,
    ) -> ShapezLoader:
        """Create a loader from just the repo path, using defaults.

        Parameters
        ----------
        game_dir : str or Path
            Path to the cloned ``shapez.io`` repository.
        serve_port : int
            Port for BrowserSync.  Default 3005.
        readiness_timeout_s : float
            How long to wait for the server to come up.
        window_title : str, optional
            Browser window title override.

        Returns
        -------
        ShapezLoader
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
        """Install dependencies for shapez.io.

        Runs ``yarn`` in the repository root and in ``gulp/``,
        using nvm to ensure Node.js 16 is active.  Validates that
        the game directory and ``gulp/gulpfile.js`` exist.
        """
        game_dir = Path(self.config.game_dir)
        if not game_dir.is_dir():
            raise FileNotFoundError(
                f"shapez.io game directory does not exist: {game_dir}. "
                f"Clone it with: git clone https://github.com/tobspr-games/shapez.io"
            )

        gulp_dir = game_dir / "gulp"
        if not gulp_dir.is_dir():
            raise FileNotFoundError(
                f"gulp/ directory not found in {game_dir}. "
                f"Expected the root of the shapez.io repository."
            )

        gulpfile = gulp_dir / "gulpfile.js"
        if not gulpfile.exists():
            raise FileNotFoundError(
                f"No gulpfile.js found in {gulp_dir}. Expected the shapez.io gulp build directory."
            )

        # Install root dependencies
        logger.info("[%s] Installing root dependencies: yarn", self.name)
        self._run_nvm_command("yarn", cwd=game_dir)

        # Install gulp dependencies
        logger.info("[%s] Installing gulp dependencies: yarn", self.name)
        self._run_nvm_command("yarn", cwd=gulp_dir)

        logger.info("[%s] Setup complete", self.name)

    def start(self) -> None:
        """Start the BrowserSync dev server via ``yarn gulp``.

        Spawns the gulp process in the ``gulp/`` subdirectory and
        blocks until the server responds on the configured port.
        """
        if self._running:
            logger.warning("[%s] Already running, stop first", self.name)
            return

        game_dir = Path(self.config.game_dir)
        gulp_dir = game_dir / "gulp"
        if not gulp_dir.is_dir():
            from src.game_loader.base import GameLoaderError

            raise GameLoaderError(f"gulp/ directory does not exist: {gulp_dir}")

        serve_cmd = _NVM_PREAMBLE + "yarn gulp"
        logger.info(
            "[%s] Starting dev server: %s (in %s)",
            self.name,
            "yarn gulp",
            gulp_dir,
        )

        kwargs: dict = dict(
            cwd=str(gulp_dir),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, **self.config.env_vars},
        )
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True

        self._process = subprocess.Popen(serve_cmd, **kwargs)
        logger.info("[%s] Server process PID: %d", self.name, self._process.pid)

        # Poll readiness
        self._wait_until_ready()
        self._running = True
        logger.info("[%s] Game is ready at %s", self.name, self.config.url)

    def _run_nvm_command(self, command: str, cwd: Path) -> None:
        """Run a shell command with nvm Node.js 16 activated.

        Parameters
        ----------
        command : str
            Shell command to run (e.g. ``"yarn"``).
        cwd : Path
            Working directory.

        Raises
        ------
        RuntimeError
            If the command fails.
        """
        full_cmd = _NVM_PREAMBLE + command
        logger.info("[%s] Running: %s (in %s)", self.name, command, cwd)
        result = subprocess.run(
            full_cmd,
            cwd=str(cwd),
            shell=True,
            capture_output=True,
            text=True,
            env={**os.environ, **self.config.env_vars},
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed (exit {result.returncode}): {command}\n"
                f"stdout: {result.stdout[-500:]}\n"
                f"stderr: {result.stderr[-500:]}"
            )
        logger.info("[%s] Command succeeded: %s", self.name, command)
