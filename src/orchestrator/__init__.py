"""Orchestrator module -- episode runner and data collection for QA sessions.

Provides ``SessionRunner`` for full game lifecycle orchestration
(launch, run N episodes with oracles, generate reports) and
``FrameCollector`` for periodic frame capture during episodes/training
to feed the YOLO retraining pipeline.
"""

from .data_collector import FrameCollector
from .session_runner import SessionRunner

__all__ = [
    "FrameCollector",
    "SessionRunner",
]
