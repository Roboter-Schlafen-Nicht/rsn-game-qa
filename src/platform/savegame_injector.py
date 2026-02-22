"""Savegame injection for starting RL episodes from mid-game states.

Provides :class:`SavegamePool` for managing a directory of save files
and :class:`SavegameInjector` for loading them into browser games via
plugin-provided JS snippets.

This enables RL training to start from interesting mid-game states
(e.g. large factories in shapez.io) instead of always starting from
scratch -- critical for finding bugs that only manifest in late-game
states.

Parameters
----------
save_dir : str or Path
    Directory containing save files.  Each file is a potential
    starting state for an episode.
selection : str
    How to pick the next save file: ``"random"`` (default) or
    ``"sequential"``.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)


class SavegamePool:
    """Manage a pool of savegame files for episode injection.

    Scans a directory for save files and provides them one at a time,
    either randomly or sequentially.

    Parameters
    ----------
    save_dir : str or Path
        Directory containing save files.
    selection : str
        ``"random"`` (default) or ``"sequential"``.
    extensions : tuple[str, ...]
        File extensions to include.  Default ``(".json", ".sav",
        ".save")``.
    seed : int or None
        Random seed for ``"random"`` selection.  ``None`` for
        unseeded.
    """

    _VALID_SELECTIONS = ("random", "sequential")

    def __init__(
        self,
        save_dir: str | Path,
        selection: str = "random",
        extensions: tuple[str, ...] = (".json", ".sav", ".save"),
        seed: int | None = None,
    ) -> None:
        if selection not in self._VALID_SELECTIONS:
            raise ValueError(
                f"Invalid selection={selection!r}; expected one of {self._VALID_SELECTIONS}"
            )

        self._save_dir = Path(save_dir)
        self._selection = selection
        self._extensions = extensions
        self._index: int = 0

        # Seed-based RNG for reproducible random selection
        self._rng = random.Random(seed)

        # Discover save files
        self._files = self._discover_files()
        if not self._files:
            logger.warning(
                "SavegamePool: no save files found in %s with extensions %s",
                self._save_dir,
                self._extensions,
            )

    @property
    def save_dir(self) -> Path:
        """Return the save directory path."""
        return self._save_dir

    @property
    def files(self) -> list[Path]:
        """Return the list of discovered save files."""
        return list(self._files)

    @property
    def selection(self) -> str:
        """Return the selection mode."""
        return self._selection

    def __len__(self) -> int:
        """Return the number of save files in the pool."""
        return len(self._files)

    def next(self) -> Path | None:
        """Return the next save file path.

        Returns
        -------
        Path or None
            Path to the next save file, or ``None`` if the pool is
            empty.
        """
        if not self._files:
            return None

        if self._selection == "random":
            return self._rng.choice(self._files)

        # Sequential: cycle through files
        path = self._files[self._index % len(self._files)]
        self._index += 1
        return path

    def _discover_files(self) -> list[Path]:
        """Scan the save directory for matching files.

        Returns
        -------
        list[Path]
            Sorted list of save file paths.
        """
        if not self._save_dir.is_dir():
            return []

        files = []
        for ext in self._extensions:
            files.extend(self._save_dir.glob(f"*{ext}"))

        # Sort for deterministic ordering (sequential mode)
        return sorted(files, key=lambda p: p.name)


class SavegameInjector:
    """Inject savegame data into a browser game via JS.

    Uses a plugin-provided JS snippet (``load_save_js``) to load save
    data into the game.  The JS snippet receives the save data as
    ``arguments[0]`` (a string — typically JSON).

    Parameters
    ----------
    pool : SavegamePool
        Pool of save files to draw from.
    load_save_js : str
        JS snippet that accepts save data as ``arguments[0]`` and
        loads it into the game.  Should return a dict with an
        ``"action"`` key on success; may return ``null`` if the
        snippet has no meaningful result.
    """

    def __init__(
        self,
        pool: SavegamePool,
        load_save_js: str,
    ) -> None:
        self._pool = pool
        self._load_save_js = load_save_js

    @property
    def pool(self) -> SavegamePool:
        """Return the underlying SavegamePool."""
        return self._pool

    @property
    def load_save_js(self) -> str:
        """Return the JS snippet for loading saves."""
        return self._load_save_js

    def inject(self, driver: object) -> dict:
        """Load the next savegame into the browser.

        Reads the next save file from the pool, passes its content
        to the ``load_save_js`` snippet via ``execute_script``.

        Parameters
        ----------
        driver : WebDriver
            Selenium WebDriver instance.

        Returns
        -------
        dict
            Result from the JS snippet execution.  Always contains a
            ``"save_file"`` key.  May also contain ``"action"`` and
            other keys depending on the JS snippet.

        Raises
        ------
        RuntimeError
            If the pool is empty or the save file cannot be read.
        """
        save_path = self._pool.next()
        if save_path is None:
            raise RuntimeError("SavegamePool is empty — no save files available")

        try:
            save_data = save_path.read_text(encoding="utf-8")
        except Exception as exc:
            raise RuntimeError(f"Failed to read save file {save_path}: {exc}") from exc

        logger.info("Injecting savegame: %s", save_path.name)

        try:
            result = driver.execute_script(self._load_save_js, save_data)
        except Exception as exc:
            logger.error("Savegame injection JS failed: %s", exc)
            return {
                "action": "error",
                "error": str(exc),
                "save_file": str(save_path),
            }

        if result is None:
            result = {}
        elif not isinstance(result, dict):
            result = {"js_result": result}

        result["save_file"] = str(save_path)
        return result
