"""Orchestrator module -- episode runner and data collection for QA sessions.

Provides ``SessionRunner`` for full game lifecycle orchestration
(launch, run N episodes with oracles, generate reports),
``FrameCollector`` for periodic frame capture during episodes/training
to feed the YOLO retraining pipeline, and ``DemoRecorder`` for
enriched per-step recording during human gameplay demonstrations.
"""

from .data_collector import FrameCollector
from .demo_recorder import DemoRecorder
from .session_runner import SessionRunner

__all__ = [
    "DemoRecorder",
    "FrameCollector",
    "SessionRunner",
]
