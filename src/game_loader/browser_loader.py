"""Browser-based game loader — manages Node dev servers for browser games.

Handles the common pattern of: ``npm install`` → ``npx parcel …`` →
poll ``http://localhost:<port>`` until the dev server responds →
game is ready for capture / input injection.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from src.game_loader.base import GameLoader, GameLoaderError
from src.game_loader.config import GameLoaderConfig

logger = logging.getLogger(__name__)


class BrowserGameLoader(GameLoader):
    """Loader for browser-based games served by a local dev server.

    Manages the lifecycle of a Node.js dev server (e.g. Parcel, Vite,
    Webpack Dev Server) that serves a browser game.  The general flow:

    1. **setup** — run ``install_command`` (e.g. ``npm install``) in the
       game's repository directory.
    2. **start** — spawn ``serve_command`` as a background process, then
       poll ``readiness_endpoint`` until the server responds with
       HTTP 200 (or the timeout expires).
    3. **is_ready** — HTTP GET against the readiness endpoint.
    4. **stop** — terminate the server subprocess tree.

    Parameters
    ----------
    config : GameLoaderConfig
        Configuration describing the game.  Must include at minimum
        ``game_dir``, ``serve_command``, and ``serve_port``.
    """

    def __init__(self, config: GameLoaderConfig) -> None:
        super().__init__(config)
        self._process: subprocess.Popen | None = None

    # -- Lifecycle -----------------------------------------------------

    def setup(self) -> None:
        """Install dependencies via the configured install command.

        Runs ``config.install_command`` (default ``npm install``)
        inside ``config.game_dir``.  Skipped if ``install_command``
        is ``None`` or empty.
        """
        if not self.config.install_command:
            logger.info("[%s] No install command configured, skipping setup", self.name)
            return

        game_dir = self.config.game_dir
        if not game_dir.is_dir():
            raise GameLoaderError(f"Game directory does not exist: {game_dir}")

        logger.info(
            "[%s] Running install: %s (in %s)",
            self.name,
            self.config.install_command,
            game_dir,
        )

        result = subprocess.run(
            self.config.install_command,
            cwd=str(game_dir),
            shell=True,
            capture_output=True,
            text=True,
            env={**os.environ, **self.config.env_vars},
        )
        if result.returncode != 0:
            raise GameLoaderError(
                f"Install command failed (exit {result.returncode}):\n"
                f"stdout: {result.stdout[-500:]}\n"
                f"stderr: {result.stderr[-500:]}"
            )

        logger.info("[%s] Install completed successfully", self.name)

    def start(self) -> None:
        """Start the dev server and block until the game is reachable.

        Spawns ``config.serve_command`` as a background subprocess,
        then polls ``config.readiness_endpoint`` every
        ``readiness_poll_interval_s`` seconds until it responds with
        HTTP 200 or ``readiness_timeout_s`` expires.

        Raises
        ------
        GameLoaderError
            If the server process exits unexpectedly or the readiness
            timeout expires.
        """
        if self._running:
            logger.warning("[%s] Already running, stop first", self.name)
            return

        game_dir = self.config.game_dir
        if not game_dir.is_dir():
            raise GameLoaderError(f"Game directory does not exist: {game_dir}")

        # Kill any orphan process on the port from a previous run.
        self._kill_port_processes()

        logger.info(
            "[%s] Starting dev server: %s (in %s)",
            self.name,
            self.config.serve_command,
            game_dir,
        )

        # Use CREATE_NEW_PROCESS_GROUP on Windows so we can terminate
        # the whole tree; on POSIX use start_new_session.
        kwargs: dict = dict(
            cwd=str(game_dir),
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

        self._process = subprocess.Popen(self.config.serve_command, **kwargs)
        logger.info("[%s] Server process PID: %d", self.name, self._process.pid)

        # Poll readiness
        self._wait_until_ready()
        self._running = True
        logger.info("[%s] Game is ready at %s", self.name, self.config.url)

    def is_ready(self) -> bool:
        """Check if the dev server is responding.

        Uses a two-stage probe:

        1. **TCP connect** to ``serve_port`` — fast check that the
           process is listening.
        2. **HTTP GET** to the ``readiness_endpoint`` — confirms the
           application layer is serving content (status < 400).

        Returns
        -------
        bool
            ``True`` if both probes succeed.
        """
        # Stage 1: lightweight TCP socket probe.
        if not self._tcp_probe():
            return False

        # Stage 2: HTTP GET to confirm the application is ready.
        try:
            req = urllib.request.Request(
                self.config.readiness_endpoint,
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status < 400
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def _tcp_probe(self) -> bool:
        """Return ``True`` if ``serve_port`` accepts a TCP connection."""
        parsed = urlparse(self.config.readiness_endpoint)
        host = parsed.hostname or "localhost"
        port = parsed.port or self.config.serve_port
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except (OSError, ConnectionRefusedError):
            return False

    def stop(self) -> None:
        """Terminate the dev server process and clean up.

        On Windows, uses ``taskkill /F /T`` to kill the entire process
        tree.  On POSIX, sends ``SIGTERM`` to the process group.
        """
        if self._process is None:
            self._running = False
            return

        pid = self._process.pid
        logger.info("[%s] Stopping server (PID %d)", self.name, pid)

        try:
            if sys.platform == "win32":
                # Kill the whole process tree on Windows
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True,
                )
            else:
                os.killpg(os.getpgid(pid), signal.SIGTERM)

            self._process.wait(timeout=10)
        except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
            logger.warning("[%s] Forceful termination of PID %d", self.name, pid)
            try:
                self._process.kill()
                self._process.wait(timeout=5)
            except Exception:
                pass

        self._process = None
        self._running = False

        # Safety net: kill any orphan children that escaped the
        # process-group signal (common with deep nvm→node→yarn→gulp trees).
        self._kill_port_processes()

        logger.info("[%s] Server stopped", self.name)

    # -- Internal -------------------------------------------------------

    def _kill_port_processes(self) -> None:
        """Kill any processes listening on the configured serve port.

        Uses ``lsof`` (POSIX) or ``netstat`` + ``taskkill`` (Windows)
        to find and terminate processes bound to the port.  This
        prevents stale dev-server instances from a previous run from
        fooling the readiness check.

        Failures are logged but never raised — this is a best-effort
        cleanup step.
        """
        port = self.config.serve_port
        if sys.platform == "win32":
            try:
                result = subprocess.run(
                    f'netstat -ano | findstr ":{port} "',
                    shell=True,
                    capture_output=True,
                    text=True,
                )
                for line in result.stdout.strip().splitlines():
                    parts = line.split()
                    if len(parts) >= 5 and "LISTENING" in line:
                        pid = int(parts[-1])
                        logger.info(
                            "[%s] Killing stale process PID %d on port %d",
                            self.name,
                            pid,
                            port,
                        )
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(pid)],
                            capture_output=True,
                        )
            except Exception as exc:
                logger.debug("[%s] Port cleanup failed: %s", self.name, exc)
        else:
            try:
                result = subprocess.run(
                    ["lsof", "-ti", f":{port}"],
                    capture_output=True,
                    text=True,
                )
                pids = result.stdout.strip().splitlines()
                for pid_str in pids:
                    try:
                        pid = int(pid_str.strip())
                    except ValueError:
                        continue
                    # Don't kill our own managed process.
                    if self._process is not None and pid == self._process.pid:
                        continue
                    logger.info(
                        "[%s] Killing stale process PID %d on port %d",
                        self.name,
                        pid,
                        port,
                    )
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            except FileNotFoundError:
                logger.debug("[%s] lsof not found, skipping port cleanup", self.name)
            except Exception as exc:
                logger.debug("[%s] Port cleanup failed: %s", self.name, exc)

    def _wait_until_ready(self) -> None:
        """Poll the readiness endpoint until it responds or timeout.

        Raises
        ------
        GameLoaderError
            If the server process exits or the timeout expires before
            the endpoint responds.
        """
        deadline = time.monotonic() + self.config.readiness_timeout_s
        interval = self.config.readiness_poll_interval_s
        endpoint = self.config.readiness_endpoint

        logger.info(
            "[%s] Waiting for %s (timeout %.0fs)",
            self.name,
            endpoint,
            self.config.readiness_timeout_s,
        )

        while time.monotonic() < deadline:
            # Check if process died
            if self._process is not None and self._process.poll() is not None:
                stdout = self._process.stdout.read() if self._process.stdout else ""
                stderr = self._process.stderr.read() if self._process.stderr else ""
                raise GameLoaderError(
                    f"Server process exited with code "
                    f"{self._process.returncode} before becoming ready.\n"
                    f"stdout: {stdout[-500:]}\n"
                    f"stderr: {stderr[-500:]}"
                )

            if self.is_ready():
                return

            time.sleep(interval)

        # Timeout — kill the process so we don't leak it
        self.stop()
        raise GameLoaderError(
            f"Server did not become ready within {self.config.readiness_timeout_s}s at {endpoint}"
        )
