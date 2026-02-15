"""Frame collector for hybrid YOLO retraining data.

Captures every Nth frame during episodes or RL training and saves them
to disk for later auto-annotation and Roboflow upload.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class FrameCollector:
    """Collects game frames during episodes/training for YOLO retraining.

    Saves every Nth frame as a PNG with metadata (step, episode, timestamp).
    After collection, frames can be batch-annotated via ``auto_annotate.py``
    and optionally uploaded to Roboflow.

    Parameters
    ----------
    output_dir : str or Path
        Base directory to save frames to.  A timestamped subdirectory
        will be created automatically.
    capture_interval : int
        Save every Nth frame (default 30 = ~1 per second at 30 FPS).
    """

    def __init__(
        self,
        output_dir: str | Path = "output",
        capture_interval: int = 30,
    ) -> None:
        self._capture_interval = max(1, capture_interval)
        self._frame_count = 0
        self._saved_count = 0
        self._step_counter = 0
        self._metadata: list[dict[str, Any]] = []

        # Create timestamped output directory
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._output_dir = Path(output_dir) / f"frames_{ts}"
        self._frames_dir = self._output_dir / "frames"
        self._frames_dir.mkdir(parents=True, exist_ok=True)

    @property
    def output_dir(self) -> Path:
        """Return the output directory path.

        Returns
        -------
        Path
            The directory where frames and metadata are saved.
        """
        return self._output_dir

    @property
    def frame_count(self) -> int:
        """Return the number of frames saved so far.

        Returns
        -------
        int
            Total frames saved to disk.
        """
        return self._saved_count

    def save_frame(
        self,
        frame: np.ndarray,
        step: int,
        episode_id: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> Path | None:
        """Conditionally save a frame based on the capture interval.

        Only saves every Nth call (controlled by ``capture_interval``).

        Parameters
        ----------
        frame : np.ndarray
            BGR image to save.
        step : int
            Current step number within the episode.
        episode_id : int
            Current episode number.
        metadata : dict, optional
            Additional metadata to store alongside the frame.

        Returns
        -------
        Path or None
            Path to the saved frame if it was captured, else None.
        """
        self._step_counter += 1

        if self._step_counter % self._capture_interval != 0:
            return None

        # Lazy import to avoid CI failures in Docker
        import cv2

        filename = f"frame_{self._saved_count:05d}.png"
        filepath = self._frames_dir / filename
        cv2.imwrite(str(filepath), frame)

        entry: dict[str, Any] = {
            "filename": filename,
            "step": step,
            "episode_id": episode_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            entry["metadata"] = metadata

        self._metadata.append(entry)
        self._saved_count += 1

        logger.debug(
            "Saved frame %d: ep=%d step=%d -> %s",
            self._saved_count,
            episode_id,
            step,
            filepath,
        )

        return filepath

    def finalize(self) -> Path:
        """Write metadata manifest and return the output directory.

        Writes a ``manifest.json`` file alongside the frames containing
        per-frame metadata (step, episode, timestamp).

        Returns
        -------
        Path
            The output directory containing frames and manifest.
        """
        manifest_path = self._output_dir / "manifest.json"
        manifest = {
            "total_frames": self._saved_count,
            "capture_interval": self._capture_interval,
            "total_steps_seen": self._step_counter,
            "frames": self._metadata,
        }

        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

        logger.info(
            "FrameCollector finalized: %d frames saved to %s",
            self._saved_count,
            self._output_dir,
        )

        return self._output_dir
