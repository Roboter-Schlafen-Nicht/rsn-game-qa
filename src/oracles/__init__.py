"""Oracles module — bug detection oracles for automated game QA.

Each oracle monitors an episode in real time and produces structured
findings (potential bugs) that are aggregated into the episode report.
"""

from .base import Oracle
from .crash import CrashOracle
from .stuck import StuckOracle
from .score_anomaly import ScoreAnomalyOracle
from .visual_glitch import VisualGlitchOracle
from .performance import PerformanceOracle

__all__ = [
    "Oracle",
    "CrashOracle",
    "StuckOracle",
    "ScoreAnomalyOracle",
    "VisualGlitchOracle",
    "PerformanceOracle",
]
