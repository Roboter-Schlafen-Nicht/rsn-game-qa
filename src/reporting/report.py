"""Episode report generation — structured JSON reports from QA runs.

Each episode produces an ``EpisodeReport`` dataclass which is serialised
to JSON.  The ``ReportGenerator`` aggregates multiple episode reports
into a session-level summary.

JSON schema::

    {
        "session_id": "uuid",
        "game": "breakout-71",
        "build_id": "local",
        "timestamp": "ISO-8601",
        "episodes": [
            {
                "episode_id": 1,
                "steps": 1234,
                "total_reward": 42.0,
                "terminated": true,
                "truncated": false,
                "findings": [ ... ],
                "metrics": {
                    "mean_fps": 30.2,
                    "min_fps": 18.5,
                    "max_reward_per_step": 3.0,
                    "total_duration_seconds": 41.1
                }
            }
        ],
        "summary": {
            "total_episodes": 10,
            "total_findings": 3,
            "critical_findings": 1,
            "warning_findings": 0,
            "info_findings": 2,
            "episodes_failed": 1,
            "mean_episode_reward": 38.5,
            "mean_episode_length": 1100
        }
    }

Severity levels align with the Oracle subsystem:
``"critical"`` (= spec ``"high"``), ``"warning"`` (= spec ``"medium"``),
``"info"`` (= spec ``"low"``).
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(obj: Any) -> Any:
    """Handle numpy types during JSON serialisation."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


@dataclass
class FindingReport:
    """Serialisable representation of a single oracle finding.

    Attributes
    ----------
    oracle_name : str
        Name of the oracle that detected the issue.
    severity : str
        ``"critical"``, ``"warning"``, or ``"info"``.
    step : int
        Environment step at which the finding occurred.
    description : str
        Human-readable description.
    data : dict[str, Any]
        Arbitrary metadata.
    screenshot_path : str | None
        Path to a saved screenshot, if available.
    """

    oracle_name: str
    severity: str
    step: int
    description: str
    data: dict[str, Any] = field(default_factory=dict)
    screenshot_path: str | None = None


@dataclass
class EpisodeMetrics:
    """Performance and gameplay metrics for a single episode.

    Attributes
    ----------
    mean_fps : float | None
        Average FPS during the episode.
    min_fps : float | None
        Minimum FPS observed.
    max_reward_per_step : float | None
        Highest single-step reward.
    total_duration_seconds : float | None
        Wall-clock duration of the episode.
    """

    mean_fps: float | None = None
    min_fps: float | None = None
    max_reward_per_step: float | None = None
    total_duration_seconds: float | None = None


@dataclass
class EpisodeReport:
    """Complete report for a single episode.

    Attributes
    ----------
    episode_id : int
        Sequential episode number within the session.
    steps : int
        Total steps taken.
    total_reward : float
        Cumulative reward.
    terminated : bool
        Whether the episode ended naturally.
    truncated : bool
        Whether the episode was truncated.
    findings : list[FindingReport]
        All oracle findings during this episode.
    metrics : EpisodeMetrics
        Performance and gameplay metrics.
    seed : int | None
        Random seed used for this episode, if any.
    """

    episode_id: int
    steps: int = 0
    total_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    findings: list[FindingReport] = field(default_factory=list)
    metrics: EpisodeMetrics = field(default_factory=EpisodeMetrics)
    seed: int | None = None


@dataclass
class SessionReport:
    """Aggregated report for an entire QA session (multiple episodes).

    Attributes
    ----------
    session_id : str
        UUID for this session.
    game : str
        Name of the game under test.
    build_id : str
        Git commit SHA or build number.  Reads from the
        ``CI_COMMIT_SHORT_SHA`` environment variable, falling back
        to ``"local"``.
    timestamp : str
        ISO-8601 timestamp of session start.
    episodes : list[EpisodeReport]
        Individual episode reports.
    summary : dict[str, Any]
        Aggregated summary statistics.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    game: str = "breakout-71"
    build_id: str = field(
        default_factory=lambda: os.getenv("CI_COMMIT_SHORT_SHA", "local")
    )
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    episodes: list[EpisodeReport] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


class ReportGenerator:
    """Generates and persists structured JSON reports from QA sessions.

    Parameters
    ----------
    output_dir : str or Path
        Directory to write report JSON files to.
    game_name : str
        Name of the game under test.
    """

    def __init__(
        self,
        output_dir: str | Path = "reports",
        game_name: str = "breakout-71",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.game_name = game_name
        self._session = SessionReport(game=game_name)

    @property
    def session(self) -> SessionReport:
        """Return the current session report (read-only access).

        Returns
        -------
        SessionReport
            The session being built.
        """
        return self._session

    def add_episode(self, episode: EpisodeReport) -> None:
        """Add a completed episode report to the session.

        Parameters
        ----------
        episode : EpisodeReport
            The completed episode report.
        """
        self._session.episodes.append(episode)

    def compute_summary(self) -> dict[str, Any]:
        """Compute aggregated summary statistics across all episodes.

        Severity counts use the Oracle naming convention:
        ``"critical"`` / ``"warning"`` / ``"info"``.

        An episode is considered *failed* if it contains at least one
        ``"critical"`` severity finding.

        Returns
        -------
        dict[str, Any]
            Summary dict with keys: ``total_episodes``,
            ``total_findings``, ``critical_findings``,
            ``warning_findings``, ``info_findings``,
            ``episodes_failed``, ``mean_episode_reward``,
            ``mean_episode_length``.
        """
        episodes = self._session.episodes
        n = len(episodes)

        if n == 0:
            return {
                "total_episodes": 0,
                "total_findings": 0,
                "critical_findings": 0,
                "warning_findings": 0,
                "info_findings": 0,
                "episodes_failed": 0,
                "mean_episode_reward": 0.0,
                "mean_episode_length": 0.0,
            }

        all_findings = [f for ep in episodes for f in ep.findings]
        critical = sum(1 for f in all_findings if f.severity == "critical")
        warning = sum(1 for f in all_findings if f.severity == "warning")
        info = sum(1 for f in all_findings if f.severity == "info")
        episodes_failed = sum(
            1 for ep in episodes if any(f.severity == "critical" for f in ep.findings)
        )

        return {
            "total_episodes": n,
            "total_findings": len(all_findings),
            "critical_findings": critical,
            "warning_findings": warning,
            "info_findings": info,
            "episodes_failed": episodes_failed,
            "mean_episode_reward": sum(ep.total_reward for ep in episodes) / n,
            "mean_episode_length": sum(ep.steps for ep in episodes) / n,
        }

    def save(self, filename: str | None = None) -> Path:
        """Serialise the session report to a JSON file.

        Creates the output directory if it does not exist.  The summary
        is computed automatically before writing.

        Parameters
        ----------
        filename : str, optional
            Output filename.  If ``None``, uses
            ``"{game}_{session_id}.json"``.

        Returns
        -------
        Path
            Path to the written JSON file.
        """
        self._session.summary = self.compute_summary()

        if filename is None:
            safe_id = self._session.session_id[:8]
            filename = f"{self.game_name}_{safe_id}.json"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / filename

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self._session), fh, indent=2, default=_json_default)

        return out_path

    def to_dict(self) -> dict[str, Any]:
        """Convert the session report to a plain dict.

        Returns
        -------
        dict[str, Any]
            The full session report as a nested dict.
        """
        self._session.summary = self.compute_summary()
        return asdict(self._session)
