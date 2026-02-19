"""Base Oracle abstract class.

All bug-detection oracles inherit from this ABC and implement the three
lifecycle hooks: ``on_reset``, ``on_step``, and ``get_findings``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Finding:
    """A single bug/anomaly detected by an oracle.

    Attributes
    ----------
    oracle_name : str
        Name of the oracle that produced this finding.
    severity : str
        One of ``"critical"``, ``"warning"``, ``"info"``.
    step : int
        The environment step at which the anomaly was detected.
    description : str
        Human-readable description of the issue.
    data : dict[str, Any]
        Arbitrary metadata (e.g. frame hash, score delta, FPS value).
    frame : np.ndarray | None
        Optional screenshot at the time of detection.
    """

    oracle_name: str
    severity: str  # "critical" | "warning" | "info"
    step: int
    description: str
    data: dict[str, Any] = field(default_factory=dict)
    frame: np.ndarray | None = None


class Oracle(ABC):
    """Abstract base class for all bug-detection oracles.

    Oracles are attached to a Gymnasium environment and called on every
    reset and step. They accumulate ``Finding`` objects which are
    collected at the end of the episode for the test report.

    Parameters
    ----------
    name : str
        A short unique identifier for this oracle (e.g. ``"crash"``).
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._findings: list[Finding] = []

    @abstractmethod
    def on_reset(self, obs: np.ndarray, info: dict[str, Any]) -> None:
        """Called when the environment is reset at the start of an episode.

        Parameters
        ----------
        obs : np.ndarray
            The initial observation from the environment.
        info : dict[str, Any]
            The info dict returned by ``env.reset()``.
        """
        ...

    @abstractmethod
    def on_step(
        self,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        """Called after every environment step.

        Parameters
        ----------
        obs : np.ndarray
            The observation returned by ``env.step()``.
        reward : float
            The reward returned by ``env.step()``.
        terminated : bool
            Whether the episode ended naturally.
        truncated : bool
            Whether the episode was truncated (e.g. time limit).
        info : dict[str, Any]
            The info dict returned by ``env.step()``.
        """
        ...

    def get_findings(self) -> list[Finding]:
        """Return all findings accumulated during the current episode.

        Returns
        -------
        list[Finding]
            All findings detected so far. The list is NOT cleared —
            call ``clear()`` explicitly between episodes.
        """
        return list(self._findings)

    def clear(self) -> None:
        """Clear all accumulated findings (call between episodes)."""
        self._findings.clear()

    def _add_finding(
        self,
        severity: str,
        step: int,
        description: str,
        data: dict[str, Any] | None = None,
        frame: np.ndarray | None = None,
    ) -> None:
        """Helper to append a new finding.

        Parameters
        ----------
        severity : str
            One of ``"critical"``, ``"warning"``, ``"info"``.
        step : int
            Environment step number.
        description : str
            Human-readable issue description.
        data : dict, optional
            Extra metadata.
        frame : np.ndarray, optional
            Screenshot at the time of detection.
        """
        self._findings.append(
            Finding(
                oracle_name=self.name,
                severity=severity,
                step=step,
                description=description,
                data=data or {},
                frame=frame,
            )
        )
