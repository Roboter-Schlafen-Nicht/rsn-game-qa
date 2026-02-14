"""Episode report generation — structured JSON reports from QA runs.

Each episode produces an ``EpisodeReport`` dataclass which is serialised
to JSON.  The ``ReportGenerator`` aggregates multiple episode reports
into a session-level summary.

JSON schema (from session1.md spec)::

    {
        "session_id": "uuid",
        "game": "breakout-71",
        "timestamp": "ISO-8601",
        "episodes": [
            {
                "episode_id": 1,
                "steps": 1234,
                "total_reward": 42.0,
                "terminated": true,
                "truncated": false,
                "duration_seconds": 41.1,
                "findings": [ ... ],
                "metrics": {
                    "mean_fps": 30.2,
                    "min_fps": 18.5,
                    "max_reward_per_step": 3.0
                }
            }
        ],
        "summary": {
            "total_episodes": 10,
            "total_findings": 3,
            "critical_findings": 1,
            "mean_episode_reward": 38.5,
            "mean_episode_length": 1100
        }
    }
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
    """

    episode_id: int
    steps: int = 0
    total_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    findings: list[FindingReport] = field(default_factory=list)
    metrics: EpisodeMetrics = field(default_factory=EpisodeMetrics)


@dataclass
class SessionReport:
    """Aggregated report for an entire QA session (multiple episodes).

    Attributes
    ----------
    session_id : str
        UUID for this session.
    game : str
        Name of the game under test.
    timestamp : str
        ISO-8601 timestamp of session start.
    episodes : list[EpisodeReport]
        Individual episode reports.
    summary : dict[str, Any]
        Aggregated summary statistics.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    game: str = "breakout-71"
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

        Returns
        -------
        dict[str, Any]
            Summary dict with keys: ``total_episodes``,
            ``total_findings``, ``critical_findings``,
            ``mean_episode_reward``, ``mean_episode_length``.
        """
        raise NotImplementedError("Summary computation not yet implemented")

    def save(self, filename: str | None = None) -> Path:
        """Serialise the session report to a JSON file.

        Parameters
        ----------
        filename : str, optional
            Output filename.  If None, uses
            ``"{game}_{session_id}.json"``.

        Returns
        -------
        Path
            Path to the written JSON file.
        """
        raise NotImplementedError("Report saving not yet implemented")

    def to_dict(self) -> dict[str, Any]:
        """Convert the session report to a plain dict.

        Returns
        -------
        dict[str, Any]
            The full session report as a nested dict.
        """
        self._session.summary = self.compute_summary()
        return asdict(self._session)
