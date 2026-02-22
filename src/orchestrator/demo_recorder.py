"""Demo recorder -- enriched per-step recording for human demonstrations.

Records frames, input events, game state, reward, oracle findings, and
observation hashes during human play sessions.  Data is stored as JSONL
(one JSON object per step) with frames saved as PNG files.  A manifest
file summarises episode metadata after finalisation.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DemoRecorder:
    """Records enriched per-step data during human gameplay demonstrations.

    Captures frames as PNG files and writes per-step metadata to a JSONL
    file for post-processing by ``analyze_demo.py``.

    Parameters
    ----------
    output_dir : str or Path
        Base directory for recordings.  A timestamped subdirectory is
        created automatically.
    game_name : str
        Name of the game being recorded.  Stored in the manifest.
    frame_capture_interval : int
        Save a frame PNG every N steps.  Default is 1 (every step).
        Clamped to a minimum of 1.
    """

    def __init__(
        self,
        output_dir: str | Path = "output",
        game_name: str = "unknown",
        frame_capture_interval: int = 1,
    ) -> None:
        self._game_name = game_name
        self._frame_capture_interval = max(1, frame_capture_interval)
        self._step_count = 0
        self._episode_count = 0
        self._episode_in_progress = False
        self._current_episode_id: int | None = None
        self._current_episode_steps = 0
        self._current_episode_reward = 0.0
        self._episodes_meta: list[dict[str, Any]] = []

        # Create timestamped output directory
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        self._output_dir = Path(output_dir) / f"demo_{ts}"
        self._frames_dir = self._output_dir / "frames"
        self._frames_dir.mkdir(parents=True, exist_ok=True)

        # JSONL file handle (opened lazily on first record_step)
        self._jsonl_path = self._output_dir / "demo.jsonl"
        self._jsonl_fh = None

    # -- Properties -----------------------------------------------------------

    @property
    def output_dir(self) -> Path:
        """Return the output directory path.

        Returns
        -------
        Path
            The directory where demo data is saved.
        """
        return self._output_dir

    @property
    def step_count(self) -> int:
        """Return the total number of steps recorded across all episodes.

        Returns
        -------
        int
            Total steps recorded.
        """
        return self._step_count

    @property
    def episode_count(self) -> int:
        """Return the number of episodes started.

        Returns
        -------
        int
            Total episodes started.
        """
        return self._episode_count

    @property
    def game_name(self) -> str:
        """Return the game name.

        Returns
        -------
        str
            The game name stored in this recorder.
        """
        return self._game_name

    @property
    def frame_capture_interval(self) -> int:
        """Return the frame capture interval.

        Returns
        -------
        int
            Save a frame PNG every N steps.
        """
        return self._frame_capture_interval

    # -- Episode lifecycle ----------------------------------------------------

    def start_episode(self, episode_id: int) -> None:
        """Begin recording a new episode.

        Parameters
        ----------
        episode_id : int
            Unique identifier for this episode.

        Raises
        ------
        RuntimeError
            If an episode is already in progress.
        """
        if self._episode_in_progress:
            raise RuntimeError(
                f"Episode {self._current_episode_id} already in progress; call end_episode() first"
            )
        self._episode_in_progress = True
        self._current_episode_id = episode_id
        self._current_episode_steps = 0
        self._current_episode_reward = 0.0
        self._episode_count += 1
        logger.info("Demo recording started for episode %d", episode_id)

    def end_episode(
        self,
        *,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Finish recording the current episode.

        Parameters
        ----------
        terminated : bool
            Whether the episode ended via a terminal state.
        truncated : bool
            Whether the episode was truncated (e.g. max_steps).

        Raises
        ------
        RuntimeError
            If no episode is in progress.
        """
        if not self._episode_in_progress:
            raise RuntimeError("No episode in progress; call start_episode() first")

        self._episodes_meta.append(
            {
                "episode_id": self._current_episode_id,
                "steps": self._current_episode_steps,
                "total_reward": self._current_episode_reward,
                "terminated": terminated,
                "truncated": truncated,
            }
        )
        self._episode_in_progress = False
        logger.info(
            "Demo recording ended for episode %d: %d steps, reward=%.2f",
            self._current_episode_id,
            self._current_episode_steps,
            self._current_episode_reward,
        )

    # -- Per-step recording ---------------------------------------------------

    def record_step(
        self,
        *,
        step: int,
        frame: np.ndarray | None,
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        human_events: list[dict[str, Any]] | None = None,
        game_state: dict[str, Any] | None = None,
        observation: np.ndarray | None = None,
        oracle_findings: list[dict[str, Any]] | None = None,
    ) -> None:
        """Record one step of gameplay.

        Parameters
        ----------
        step : int
            Step number within the current episode.
        frame : np.ndarray or None
            BGR image captured at this step.  Saved as PNG if not None
            and the frame capture interval allows it.
        action : any
            The action taken (scalar, numpy array, or list).
        reward : float
            Reward received at this step.
        terminated : bool
            Whether this step is terminal.
        truncated : bool
            Whether this step is truncated.
        human_events : list[dict] or None
            Human input events captured by the EventRecorder.
        game_state : dict or None
            Game state from the JS bridge.
        observation : np.ndarray or None
            The observation vector/image.  Hashed for the record.
        oracle_findings : list[dict] or None
            Oracle findings from this step.

        Raises
        ------
        RuntimeError
            If no episode is in progress.
        """
        if not self._episode_in_progress:
            raise RuntimeError("No episode in progress; call start_episode() first")

        # Save frame PNG (respecting interval)
        frame_file: str | None = None
        if frame is not None and (self._current_episode_steps % self._frame_capture_interval == 0):
            frame_file = self._save_frame(frame, self._current_episode_id, step)

        # Compute observation hash
        obs_hash: str | None = None
        if observation is not None:
            obs_hash = hashlib.md5(observation.tobytes()).hexdigest()

        # Build JSONL record
        record: dict[str, Any] = {
            "step": step,
            "episode_id": self._current_episode_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "action": self._serialize_action(action),
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
            "human_events": human_events if human_events is not None else [],
            "game_state": game_state if game_state is not None else {},
            "obs_hash": obs_hash,
            "frame_file": frame_file,
            "oracle_findings": oracle_findings if oracle_findings is not None else [],
        }

        self._write_jsonl(record)
        self._step_count += 1
        self._current_episode_steps += 1
        self._current_episode_reward += reward

    # -- Finalization ---------------------------------------------------------

    def finalize(self) -> Path:
        """Write the manifest and close open file handles.

        Returns
        -------
        Path
            The output directory containing demo data and manifest.
        """
        # Close JSONL file handle
        if self._jsonl_fh is not None:
            self._jsonl_fh.close()
            self._jsonl_fh = None

        manifest = {
            "game_name": self._game_name,
            "total_steps": self._step_count,
            "total_episodes": self._episode_count,
            "frame_capture_interval": self._frame_capture_interval,
            "episodes": self._episodes_meta,
        }

        manifest_path = self._output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

        logger.info(
            "DemoRecorder finalized: %d steps, %d episodes in %s",
            self._step_count,
            self._episode_count,
            self._output_dir,
        )

        return self._output_dir

    # -- Internal helpers -----------------------------------------------------

    def _save_frame(
        self,
        frame: np.ndarray,
        episode_id: int | None,
        step: int,
    ) -> str:
        """Save a frame as PNG and return the relative filename.

        Parameters
        ----------
        frame : np.ndarray
            BGR image to save.
        episode_id : int or None
            Current episode ID.
        step : int
            Current step within the episode.

        Returns
        -------
        str
            The filename (relative to frames/ directory).
        """
        import cv2

        ep = episode_id if episode_id is not None else 0
        filename = f"ep{ep:03d}_step{step:05d}.png"
        filepath = self._frames_dir / filename
        cv2.imwrite(str(filepath), frame)
        return filename

    @staticmethod
    def _serialize_action(action: Any) -> int | float | list:
        """Convert an action to a JSON-serializable type.

        Parameters
        ----------
        action : any
            Scalar int, numpy scalar, numpy array, or list.

        Returns
        -------
        int or float or list
            JSON-serializable representation.
        """
        if isinstance(action, np.ndarray):
            return action.tolist()
        if isinstance(action, (np.integer,)):
            return int(action)
        if isinstance(action, (np.floating,)):
            return float(action)
        return action

    def _write_jsonl(self, record: dict[str, Any]) -> None:
        """Append a JSON record to the JSONL file.

        Parameters
        ----------
        record : dict
            The record to write as a single line.
        """
        if self._jsonl_fh is None:
            self._jsonl_fh = open(self._jsonl_path, "a", encoding="utf-8")  # noqa: SIM115
        self._jsonl_fh.write(json.dumps(record) + "\n")
        self._jsonl_fh.flush()
