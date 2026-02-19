"""Oracles module — bug detection oracles for automated game QA.

Each oracle monitors an episode in real time and produces structured
findings (potential bugs) that are aggregated into the episode report.
"""

from .base import Oracle
from .boundary import BoundaryOracle
from .crash import CrashOracle
from .episode_length import EpisodeLengthOracle
from .performance import PerformanceOracle
from .physics_violation import PhysicsViolationOracle
from .reward_consistency import RewardConsistencyOracle
from .score_anomaly import ScoreAnomalyOracle
from .soak import SoakOracle
from .state_transition import StateTransitionOracle
from .stuck import StuckOracle
from .temporal_anomaly import TemporalAnomalyOracle
from .visual_glitch import VisualGlitchOracle

__all__ = [
    "BoundaryOracle",
    "CrashOracle",
    "EpisodeLengthOracle",
    "Oracle",
    "PerformanceOracle",
    "PhysicsViolationOracle",
    "RewardConsistencyOracle",
    "ScoreAnomalyOracle",
    "SoakOracle",
    "StateTransitionOracle",
    "StuckOracle",
    "TemporalAnomalyOracle",
    "VisualGlitchOracle",
]
