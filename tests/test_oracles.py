"""Tests for the oracles module."""

import numpy as np
import pytest

from src.oracles.base import Finding, Oracle


class TestFinding:
    """Tests for the Finding dataclass."""

    def test_finding_creation(self):
        """Finding can be constructed with required fields."""
        f = Finding(
            oracle_name="test",
            severity="info",
            step=42,
            description="test finding",
        )
        assert f.oracle_name == "test"
        assert f.severity == "info"
        assert f.step == 42
        assert f.data == {}
        assert f.frame is None


class TestOracleBase:
    """Tests for the Oracle ABC."""

    def test_oracle_is_abstract(self):
        """Oracle cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Oracle(name="test")  # type: ignore[abstract]

    def test_clear_removes_findings(self):
        """clear() should empty the findings list."""
        # TODO: create a concrete subclass and test clear()
        pass


class TestCrashOracle:
    """Placeholder tests for CrashOracle."""

    def test_crash_oracle_init(self):
        """CrashOracle should initialise with default params."""
        from src.oracles.crash import CrashOracle

        oracle = CrashOracle()
        assert oracle.name == "crash"
        assert oracle.freeze_threshold == 30

    def test_on_reset_clears_state(self):
        """on_reset should reset internal counters."""
        from src.oracles.crash import CrashOracle

        oracle = CrashOracle()
        obs = np.zeros(10, dtype=np.float32)
        oracle.on_reset(obs, {})
        assert oracle._identical_count == 0


class TestStuckOracle:
    """Placeholder tests for StuckOracle."""

    def test_stuck_oracle_init(self):
        """StuckOracle should initialise with default params."""
        from src.oracles.stuck import StuckOracle

        oracle = StuckOracle()
        assert oracle.name == "stuck"
        assert oracle.patience == 300


class TestScoreAnomalyOracle:
    """Placeholder tests for ScoreAnomalyOracle."""

    def test_score_anomaly_oracle_init(self):
        """ScoreAnomalyOracle should initialise with default params."""
        from src.oracles.score_anomaly import ScoreAnomalyOracle

        oracle = ScoreAnomalyOracle()
        assert oracle.name == "score_anomaly"
        assert oracle.max_delta == 100.0


class TestVisualGlitchOracle:
    """Placeholder tests for VisualGlitchOracle."""

    def test_visual_glitch_oracle_init(self):
        """VisualGlitchOracle should initialise with default params."""
        from src.oracles.visual_glitch import VisualGlitchOracle

        oracle = VisualGlitchOracle()
        assert oracle.name == "visual_glitch"
        assert oracle.ssim_threshold == 0.5


class TestPerformanceOracle:
    """Placeholder tests for PerformanceOracle."""

    def test_performance_oracle_init(self):
        """PerformanceOracle should initialise with default params."""
        from src.oracles.performance import PerformanceOracle

        oracle = PerformanceOracle()
        assert oracle.name == "performance"
        assert oracle.min_fps == 20.0
