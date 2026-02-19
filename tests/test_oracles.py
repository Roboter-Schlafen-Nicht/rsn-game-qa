"""Tests for the oracles module.

Tests cover:

- Finding dataclass construction and defaults
- Oracle ABC (abstract enforcement, clear, get_findings, _add_finding)
- CrashOracle: process liveness detection, black frame detection,
  frozen frame detection, freeze threshold behaviour, reset lifecycle
- StuckOracle: reward stagnation, observation variance, patience
  threshold, sliding window, reset lifecycle
- ScoreAnomalyOracle: impossible jump detection, negative score
  detection, statistical anomaly detection, score_key customisation,
  reset lifecycle
- VisualGlitchOracle: SSIM drop detection, pHash distance detection,
  rate-limiting, no-frame graceful handling, reset lifecycle
- PerformanceOracle: low FPS sustained detection, CPU/RAM threshold
  detection, FPS summary statistics, reset lifecycle
- PhysicsViolationOracle: tunneling, paddle pass-through, ghost collision
- BoundaryOracle: hard OOB, soft OOB, configurable bounds
- StateTransitionOracle: lives/level/game_state validation
- EpisodeLengthOracle: short/long episodes, statistical outliers
- TemporalAnomalyOracle: teleportation, flickering detection
- RewardConsistencyOracle: score/reward, lives/reward, brick/reward mismatches
- SoakOracle: memory leak trend, FPS degradation across episodes
- CrashOracle dedup: black frame / frozen frame finding deduplication
- ScoreAnomalyOracle dedup: negative score finding deduplication
- TemporalAnomalyOracle cooldown: flicker finding cooldown
- VisualGlitchOracle cv2 guard: graceful cv2 import failure handling
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from src.oracles.base import Finding, Oracle

try:
    import cv2  # noqa: F401

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

_skip_no_cv2 = pytest.mark.skipif(not _CV2_AVAILABLE, reason="cv2 not available")


# ── Helpers ─────────────────────────────────────────────────────────


def _obs(n: int = 10, value: float = 0.0) -> np.ndarray:
    """Create a simple float32 observation vector."""
    return np.full(n, value, dtype=np.float32)


def _frame(
    h: int = 100,
    w: int = 100,
    color: tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    """Create a solid-colour BGR frame."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


def _random_frame(h: int = 100, w: int = 100, seed: int = 0) -> np.ndarray:
    """Create a random BGR frame with a given seed."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)


class ConcreteOracle(Oracle):
    """Minimal concrete Oracle for testing the base class."""

    def on_reset(self, obs: np.ndarray, info: dict) -> None:
        pass

    def on_step(self, obs, reward, terminated, truncated, info) -> None:
        pass


# ── Finding dataclass ───────────────────────────────────────────────


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

    def test_finding_with_data_and_frame(self):
        """Finding stores optional data dict and frame array."""
        frame = _frame()
        f = Finding(
            oracle_name="vis",
            severity="critical",
            step=10,
            description="glitch detected",
            data={"ssim": 0.3},
            frame=frame,
        )
        assert f.data == {"ssim": 0.3}
        assert np.array_equal(f.frame, frame)

    def test_finding_severity_values(self):
        """Finding accepts all valid severity levels."""
        for sev in ("critical", "warning", "info"):
            f = Finding(oracle_name="x", severity=sev, step=0, description="d")
            assert f.severity == sev


# ── Oracle ABC ──────────────────────────────────────────────────────


class TestOracleBase:
    """Tests for the Oracle ABC."""

    def test_oracle_is_abstract(self):
        """Oracle cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Oracle(name="test")  # type: ignore[abstract]

    def test_clear_removes_findings(self):
        """clear() should empty the findings list."""
        oracle = ConcreteOracle(name="test")
        oracle._add_finding("info", step=1, description="a")
        oracle._add_finding("warning", step=2, description="b")
        assert len(oracle.get_findings()) == 2
        oracle.clear()
        assert len(oracle.get_findings()) == 0

    def test_get_findings_returns_copy(self):
        """get_findings() returns a new list each time."""
        oracle = ConcreteOracle(name="test")
        oracle._add_finding("info", step=1, description="a")
        f1 = oracle.get_findings()
        f2 = oracle.get_findings()
        assert f1 == f2
        assert f1 is not f2

    def test_add_finding_creates_proper_finding(self):
        """_add_finding creates a Finding with correct oracle_name."""
        oracle = ConcreteOracle(name="myoracle")
        oracle._add_finding(
            severity="critical",
            step=5,
            description="test desc",
            data={"key": "val"},
        )
        findings = oracle.get_findings()
        assert len(findings) == 1
        assert findings[0].oracle_name == "myoracle"
        assert findings[0].severity == "critical"
        assert findings[0].step == 5
        assert findings[0].description == "test desc"
        assert findings[0].data == {"key": "val"}


# ── CrashOracle ─────────────────────────────────────────────────────


class TestCrashOracle:
    """Tests for CrashOracle on_step detection logic."""

    def test_crash_oracle_init(self):
        """CrashOracle should initialise with default params."""
        from src.oracles.crash import CrashOracle

        oracle = CrashOracle()
        assert oracle.name == "crash"
        assert oracle.freeze_threshold == 30
        assert oracle.process_name is None

    def test_on_reset_clears_state(self):
        """on_reset should reset internal counters."""
        from src.oracles.crash import CrashOracle

        oracle = CrashOracle()
        obs = _obs()
        # Simulate some state
        oracle._identical_count = 10
        oracle._step_count = 42
        oracle._add_finding("critical", step=1, description="old")
        oracle.on_reset(obs, {})
        assert oracle._identical_count == 0
        assert oracle._step_count == 0
        assert oracle._prev_frame is None

    def test_black_frame_detected(self):
        """A completely black frame should raise a critical finding."""
        from src.oracles.crash import CrashOracle

        oracle = CrashOracle()
        oracle.on_reset(_obs(), {})

        black = np.zeros((100, 100, 3), dtype=np.uint8)
        oracle.on_step(_obs(), 0.0, False, False, {"frame": black})

        findings = oracle.get_findings()
        assert len(findings) == 1
        assert findings[0].severity == "critical"
        assert "Black frame" in findings[0].description
        assert findings[0].data["type"] == "black_frame"

    def test_no_finding_for_normal_frame(self):
        """A normal (non-black, non-frozen) frame should produce no findings."""
        from src.oracles.crash import CrashOracle

        oracle = CrashOracle()
        oracle.on_reset(_obs(), {})

        # Two different frames
        oracle.on_step(_obs(), 0.0, False, False, {"frame": _frame(color=(100, 100, 100))})
        oracle.on_step(_obs(), 0.0, False, False, {"frame": _frame(color=(120, 120, 120))})

        assert len(oracle.get_findings()) == 0

    def test_frozen_frame_detected_at_threshold(self):
        """Identical frames for freeze_threshold steps should flag a freeze."""
        from src.oracles.crash import CrashOracle

        threshold = 5
        oracle = CrashOracle(freeze_threshold=threshold)
        oracle.on_reset(_obs(), {})

        frame = _frame(color=(50, 50, 50))
        for _ in range(threshold + 1):
            oracle.on_step(_obs(), 0.0, False, False, {"frame": frame.copy()})

        findings = oracle.get_findings()
        freeze_findings = [f for f in findings if f.data.get("type") == "frozen_frame"]
        assert len(freeze_findings) >= 1
        assert freeze_findings[0].severity == "critical"
        assert "frozen" in freeze_findings[0].description.lower()

    def test_frozen_frame_counter_resets_on_different_frame(self):
        """A new frame should reset the freeze counter."""
        from src.oracles.crash import CrashOracle

        oracle = CrashOracle(freeze_threshold=5)
        oracle.on_reset(_obs(), {})

        same = _frame(color=(50, 50, 50))
        # 3 identical frames, then a different one, then 3 more identical
        for _ in range(3):
            oracle.on_step(_obs(), 0.0, False, False, {"frame": same.copy()})
        oracle.on_step(_obs(), 0.0, False, False, {"frame": _frame(color=(200, 200, 200))})
        for _ in range(3):
            oracle.on_step(_obs(), 0.0, False, False, {"frame": same.copy()})

        # Should NOT reach threshold of 5 consecutive identical
        freeze_findings = [
            f for f in oracle.get_findings() if f.data.get("type") == "frozen_frame"
        ]
        assert len(freeze_findings) == 0

    def test_no_frame_in_info_is_safe(self):
        """If info has no 'frame' key, on_step should not crash."""
        from src.oracles.crash import CrashOracle

        oracle = CrashOracle()
        oracle.on_reset(_obs(), {})
        oracle.on_step(_obs(), 0.0, False, False, {})
        assert len(oracle.get_findings()) == 0

    def test_process_name_check_dead_process(self):
        """When process_name is set and process is dead, flag crash."""
        from src.oracles.crash import CrashOracle

        oracle = CrashOracle(process_name="nonexistent_game_12345.exe")
        oracle.on_reset(_obs(), {})

        # Mock psutil to return empty process list
        mock_psutil = mock.MagicMock()
        mock_psutil.process_iter.return_value = []

        with mock.patch.dict("sys.modules", {"psutil": mock_psutil}):
            oracle.on_step(_obs(), 0.0, False, False, {})

        findings = oracle.get_findings()
        assert len(findings) == 1
        assert findings[0].severity == "critical"
        assert "no longer running" in findings[0].description

    def test_process_name_check_alive_process(self):
        """When process_name is set and process is alive, no crash finding."""
        from src.oracles.crash import CrashOracle

        oracle = CrashOracle(process_name="chrome.exe")
        oracle.on_reset(_obs(), {})

        # Mock psutil with matching process
        mock_proc = mock.MagicMock()
        mock_proc.info = {"name": "chrome.exe"}
        mock_psutil = mock.MagicMock()
        mock_psutil.process_iter.return_value = [mock_proc]
        mock_psutil.NoSuchProcess = Exception
        mock_psutil.AccessDenied = Exception

        with mock.patch.dict("sys.modules", {"psutil": mock_psutil}):
            oracle.on_step(_obs(), 0.0, False, False, {})

        assert len(oracle.get_findings()) == 0

    def test_no_process_name_skips_process_check(self):
        """Without process_name, process check always passes."""
        from src.oracles.crash import CrashOracle

        oracle = CrashOracle(process_name=None)
        oracle.on_reset(_obs(), {})
        # Should not raise or produce findings for process check
        oracle.on_step(_obs(), 0.0, False, False, {"frame": _frame()})
        assert len(oracle.get_findings()) == 0


# ── StuckOracle ─────────────────────────────────────────────────────


class TestStuckOracle:
    """Tests for StuckOracle on_step detection logic."""

    def test_stuck_oracle_init(self):
        """StuckOracle should initialise with default params."""
        from src.oracles.stuck import StuckOracle

        oracle = StuckOracle()
        assert oracle.name == "stuck"
        assert oracle.patience == 300

    def test_on_reset_clears_state(self):
        """on_reset should reset all internal tracking."""
        from src.oracles.stuck import StuckOracle

        oracle = StuckOracle()
        oracle._reward_buffer = [1.0, 2.0]
        oracle._obs_buffer = [_obs()]
        oracle._no_progress_steps = 50
        oracle._step_count = 100
        oracle.on_reset(_obs(), {})
        assert oracle._reward_buffer == []
        assert oracle._obs_buffer == []
        assert oracle._no_progress_steps == 0
        assert oracle._step_count == 0

    def test_zero_reward_triggers_stuck_at_patience(self):
        """Sustained near-zero reward should trigger stuck finding at patience."""
        from src.oracles.stuck import StuckOracle

        patience = 10
        oracle = StuckOracle(patience=patience, reward_epsilon=1e-4, window_size=5)
        oracle.on_reset(_obs(), {})

        for _ in range(patience):
            oracle.on_step(_obs(), 0.0, False, False, {})

        findings = oracle.get_findings()
        assert len(findings) == 1
        assert findings[0].severity == "warning"
        assert "stuck" in findings[0].description.lower()
        assert findings[0].data["no_progress_steps"] == patience

    def test_positive_reward_prevents_stuck(self):
        """Steps with meaningful reward should reset the stuck counter."""
        from src.oracles.stuck import StuckOracle

        patience = 10
        oracle = StuckOracle(patience=patience, reward_epsilon=1e-4, window_size=5)
        oracle.on_reset(_obs(), {})

        # 5 zero-reward steps, then a positive reward, then 5 more zero
        for _ in range(5):
            oracle.on_step(_obs(), 0.0, False, False, {})
        oracle.on_step(_obs(), 1.0, False, False, {})  # resets counter
        for _ in range(5):
            oracle.on_step(_obs(), 0.0, False, False, {})

        # Should NOT have triggered: neither run reached patience of 10
        findings = oracle.get_findings()
        assert len(findings) == 0

    def test_stuck_fires_once_at_patience(self):
        """The stuck finding should fire exactly once at the patience threshold."""
        from src.oracles.stuck import StuckOracle

        patience = 5
        oracle = StuckOracle(patience=patience, reward_epsilon=1e-4, window_size=3)
        oracle.on_reset(_obs(), {})

        # Run well past patience
        for _ in range(patience * 3):
            oracle.on_step(_obs(), 0.0, False, False, {})

        # Should fire exactly once for the primary stuck detection
        stuck_findings = [
            f
            for f in oracle.get_findings()
            if "stuck" in f.description.lower() and f.severity == "warning"
        ]
        assert len(stuck_findings) == 1

    def test_observation_variance_check(self):
        """Low observation variance should be tracked in finding data."""
        from src.oracles.stuck import StuckOracle

        patience = 5
        oracle = StuckOracle(
            patience=patience,
            reward_epsilon=1e-4,
            obs_var_threshold=1e-6,
            window_size=3,
        )
        oracle.on_reset(_obs(), {})

        # All identical observations with zero reward
        for _ in range(patience):
            oracle.on_step(_obs(value=0.5), 0.0, False, False, {})

        findings = oracle.get_findings()
        assert len(findings) >= 1
        # The stuck finding should note low obs variance
        assert findings[0].data.get("obs_variance_low") is True

    def test_varying_observations_reduce_stuck_risk(self):
        """Changing observations with zero reward still triggers stuck (reward-based)."""
        from src.oracles.stuck import StuckOracle

        patience = 5
        oracle = StuckOracle(patience=patience, reward_epsilon=1e-4, window_size=3)
        oracle.on_reset(_obs(), {})

        # Zero reward but varying observations
        for i in range(patience):
            oracle.on_step(_obs(value=float(i)), 0.0, False, False, {})

        # Stuck is reward-based, so should still fire
        findings = oracle.get_findings()
        assert len(findings) >= 1


# ── ScoreAnomalyOracle ──────────────────────────────────────────────


class TestScoreAnomalyOracle:
    """Tests for ScoreAnomalyOracle on_step detection logic."""

    def test_score_anomaly_oracle_init(self):
        """ScoreAnomalyOracle should initialise with default params."""
        from src.oracles.score_anomaly import ScoreAnomalyOracle

        oracle = ScoreAnomalyOracle()
        assert oracle.name == "score_anomaly"
        assert oracle.max_delta == 100.0
        assert oracle.allow_negative is False

    def test_on_reset_clears_state(self):
        """on_reset should reset score tracking."""
        from src.oracles.score_anomaly import ScoreAnomalyOracle

        oracle = ScoreAnomalyOracle()
        oracle._score_history = [100.0, 200.0]
        oracle._step_count = 99
        oracle.on_reset(_obs(), {"score": 0})
        assert oracle._score_history == []
        assert oracle._step_count == 0
        assert oracle._prev_score == 0

    def test_impossible_jump_detected(self):
        """A score jump exceeding max_delta should raise critical finding."""
        from src.oracles.score_anomaly import ScoreAnomalyOracle

        oracle = ScoreAnomalyOracle(max_delta=50.0)
        oracle.on_reset(_obs(), {"score": 0})

        oracle.on_step(_obs(), 0.0, False, False, {"score": 10})
        assert len(oracle.get_findings()) == 0

        oracle.on_step(_obs(), 0.0, False, False, {"score": 200})

        findings = oracle.get_findings()
        jump_findings = [f for f in findings if f.severity == "critical"]
        assert len(jump_findings) >= 1
        assert "Impossible score jump" in jump_findings[0].description
        assert jump_findings[0].data["delta"] == 190.0

    def test_normal_score_progression_no_findings(self):
        """Gradual score increases within max_delta produce no findings."""
        from src.oracles.score_anomaly import ScoreAnomalyOracle

        oracle = ScoreAnomalyOracle(max_delta=100.0)
        oracle.on_reset(_obs(), {"score": 0})

        for i in range(10):
            oracle.on_step(_obs(), 0.0, False, False, {"score": i * 10})

        assert len(oracle.get_findings()) == 0

    def test_negative_score_detected(self):
        """Negative score should raise warning when allow_negative is False."""
        from src.oracles.score_anomaly import ScoreAnomalyOracle

        oracle = ScoreAnomalyOracle(allow_negative=False, max_delta=1000.0)
        oracle.on_reset(_obs(), {"score": 0})

        oracle.on_step(_obs(), 0.0, False, False, {"score": -5})

        findings = oracle.get_findings()
        neg_findings = [f for f in findings if "Negative score" in f.description]
        assert len(neg_findings) == 1
        assert neg_findings[0].severity == "warning"

    def test_negative_score_allowed(self):
        """Negative score should be ignored when allow_negative is True."""
        from src.oracles.score_anomaly import ScoreAnomalyOracle

        oracle = ScoreAnomalyOracle(allow_negative=True, max_delta=1000.0)
        oracle.on_reset(_obs(), {"score": 0})

        oracle.on_step(_obs(), 0.0, False, False, {"score": -5})

        neg_findings = [f for f in oracle.get_findings() if "Negative" in f.description]
        assert len(neg_findings) == 0

    def test_statistical_anomaly_detected(self):
        """A sudden score spike should be flagged as a statistical anomaly."""
        from src.oracles.score_anomaly import ScoreAnomalyOracle

        oracle = ScoreAnomalyOracle(max_delta=10000.0, z_threshold=2.0)
        oracle.on_reset(_obs(), {"score": 0})

        # Build a stable baseline
        for i in range(10):
            oracle.on_step(_obs(), 0.0, False, False, {"score": 10 + i})

        # Now a huge outlier (within max_delta but statistically anomalous)
        oracle.on_step(_obs(), 0.0, False, False, {"score": 5000})

        findings = oracle.get_findings()
        stat_findings = [f for f in findings if "Statistical" in f.description]
        assert len(stat_findings) >= 1
        assert stat_findings[0].data["z_score"] > 2.0

    def test_no_score_in_info_is_safe(self):
        """If score key is missing from info, on_step should not crash."""
        from src.oracles.score_anomaly import ScoreAnomalyOracle

        oracle = ScoreAnomalyOracle()
        oracle.on_reset(_obs(), {})
        oracle.on_step(_obs(), 0.0, False, False, {})
        assert len(oracle.get_findings()) == 0

    def test_custom_score_key(self):
        """ScoreAnomalyOracle should use the configured score_key."""
        from src.oracles.score_anomaly import ScoreAnomalyOracle

        oracle = ScoreAnomalyOracle(score_key="points", max_delta=10.0)
        oracle.on_reset(_obs(), {"points": 0})

        oracle.on_step(_obs(), 0.0, False, False, {"points": 5})
        assert len(oracle.get_findings()) == 0

        oracle.on_step(_obs(), 0.0, False, False, {"points": 500})

        findings = oracle.get_findings()
        assert any("Impossible score jump" in f.description for f in findings)

    def test_score_decrease_jump(self):
        """A large score decrease should also flag as impossible jump."""
        from src.oracles.score_anomaly import ScoreAnomalyOracle

        oracle = ScoreAnomalyOracle(max_delta=50.0)
        oracle.on_reset(_obs(), {"score": 100})

        oracle.on_step(_obs(), 0.0, False, False, {"score": 0})

        findings = oracle.get_findings()
        jump_findings = [f for f in findings if "Impossible score jump" in f.description]
        assert len(jump_findings) >= 1


# ── VisualGlitchOracle ──────────────────────────────────────────────


class TestVisualGlitchOracle:
    """Tests for VisualGlitchOracle on_step detection logic."""

    def test_visual_glitch_oracle_init(self):
        """VisualGlitchOracle should initialise with default params."""
        from src.oracles.visual_glitch import VisualGlitchOracle

        oracle = VisualGlitchOracle()
        assert oracle.name == "visual_glitch"
        assert oracle.ssim_threshold == 0.5
        assert oracle.hash_distance_threshold == 20

    def test_on_reset_clears_state(self):
        """on_reset should reset frame tracking."""
        from src.oracles.visual_glitch import VisualGlitchOracle

        oracle = VisualGlitchOracle()
        oracle._prev_frame = _frame()
        oracle._step_count = 42
        oracle._add_finding("warning", step=1, description="old")
        oracle.on_reset(_obs(), {})
        assert oracle._prev_frame is None
        assert oracle._step_count == 0
        # Note: on_reset doesn't call clear() — findings persist until
        # explicit clear() call.  But internal state is reset.

    def test_no_frame_in_info_is_safe(self):
        """If info has no 'frame' key, on_step should not crash."""
        from src.oracles.visual_glitch import VisualGlitchOracle

        oracle = VisualGlitchOracle()
        oracle.on_reset(_obs(), {})
        oracle.on_step(_obs(), 0.0, False, False, {})
        assert len(oracle.get_findings()) == 0

    @_skip_no_cv2
    def test_similar_frames_no_findings(self):
        """Very similar consecutive frames should not produce findings."""
        from src.oracles.visual_glitch import VisualGlitchOracle

        oracle = VisualGlitchOracle(ssim_threshold=0.5, min_interval=1)
        oracle.on_reset(_obs(), {})

        frame_a = _frame(color=(100, 100, 100))
        frame_b = _frame(color=(102, 102, 102))  # Nearly identical

        oracle.on_step(_obs(), 0.0, False, False, {"frame": frame_a})
        oracle.on_step(_obs(), 0.0, False, False, {"frame": frame_b})

        assert len(oracle.get_findings()) == 0

    @_skip_no_cv2
    def test_ssim_drop_detected(self):
        """A dramatic frame change should trigger an SSIM-based finding."""
        from src.oracles.visual_glitch import VisualGlitchOracle

        oracle = VisualGlitchOracle(
            ssim_threshold=0.8,
            hash_distance_threshold=100,  # High to avoid phash findings
            min_interval=1,
        )
        oracle.on_reset(_obs(), {})

        # First frame: solid grey
        frame_a = _frame(color=(128, 128, 128))
        oracle.on_step(_obs(), 0.0, False, False, {"frame": frame_a})

        # Second frame: radically different — use random noise
        frame_b = _random_frame(seed=42)
        oracle.on_step(_obs(), 0.0, False, False, {"frame": frame_b})

        findings = oracle.get_findings()
        ssim_findings = [f for f in findings if f.data.get("type") == "ssim_anomaly"]
        assert len(ssim_findings) >= 1
        assert ssim_findings[0].data["ssim"] < 0.8

    @_skip_no_cv2
    def test_phash_distance_detected(self):
        """A large pHash change should trigger a phash_anomaly finding."""
        from src.oracles.visual_glitch import VisualGlitchOracle

        oracle = VisualGlitchOracle(
            hash_distance_threshold=5,
            ssim_threshold=0.0,  # Disable SSIM findings
            min_interval=1,
        )
        oracle.on_reset(_obs(), {})

        # Solid frame vs random noise — very different pHash
        oracle.on_step(_obs(), 0.0, False, False, {"frame": _frame(color=(0, 0, 0))})
        oracle.on_step(_obs(), 0.0, False, False, {"frame": _random_frame(seed=99)})

        findings = oracle.get_findings()
        phash_findings = [f for f in findings if f.data.get("type") == "phash_anomaly"]
        # This depends on imagehash being installed
        try:
            import imagehash  # noqa: F401

            assert len(phash_findings) >= 1
        except ImportError:
            # If imagehash not installed, phash returns None, no finding
            assert len(phash_findings) == 0

    @_skip_no_cv2
    def test_rate_limiting(self):
        """Findings should be rate-limited by min_interval."""
        from src.oracles.visual_glitch import VisualGlitchOracle

        oracle = VisualGlitchOracle(
            ssim_threshold=0.99,  # Very strict — everything triggers
            hash_distance_threshold=100,  # Disable phash
            min_interval=5,
        )
        oracle.on_reset(_obs(), {})

        # Feed alternating very different frames
        for i in range(10):
            frame = _random_frame(seed=i)
            oracle.on_step(_obs(), 0.0, False, False, {"frame": frame})

        findings = oracle.get_findings()
        # With min_interval=5, at most 2 findings in 10 steps
        # (first possible at step 2, next at step 7 earliest)
        assert len(findings) <= 2

    @_skip_no_cv2
    def test_first_frame_no_comparison(self):
        """The first frame should not produce any findings (no previous frame)."""
        from src.oracles.visual_glitch import VisualGlitchOracle

        oracle = VisualGlitchOracle(min_interval=1)
        oracle.on_reset(_obs(), {})

        oracle.on_step(_obs(), 0.0, False, False, {"frame": _random_frame(seed=0)})
        assert len(oracle.get_findings()) == 0

    @_skip_no_cv2
    def test_compute_ssim_identical_frames(self):
        """SSIM of identical frames should be close to 1.0."""
        from src.oracles.visual_glitch import VisualGlitchOracle

        oracle = VisualGlitchOracle()
        frame = _frame(color=(100, 100, 100))
        ssim = oracle._compute_ssim(frame, frame.copy())
        assert ssim > 0.99


# ── PerformanceOracle ───────────────────────────────────────────────


class TestPerformanceOracle:
    """Tests for PerformanceOracle on_step detection logic."""

    def test_performance_oracle_init(self):
        """PerformanceOracle should initialise with default params."""
        from src.oracles.performance import PerformanceOracle

        oracle = PerformanceOracle()
        assert oracle.name == "performance"
        assert oracle.min_fps == 20.0
        assert oracle.sustained_frames == 30

    def test_on_reset_clears_state(self):
        """on_reset should reset all performance tracking."""
        from src.oracles.performance import PerformanceOracle

        oracle = PerformanceOracle()
        oracle._fps_history = [60.0, 30.0]
        oracle._low_fps_count = 10
        oracle._step_count = 100
        oracle.on_reset(_obs(), {})
        assert oracle._fps_history == []
        assert oracle._low_fps_count == 0
        assert oracle._step_count == 0
        assert oracle._last_step_time is None

    def test_sustained_low_fps_finding(self):
        """Low FPS for sustained_frames should trigger a warning."""
        from src.oracles.performance import PerformanceOracle

        oracle = PerformanceOracle(min_fps=30.0, sustained_frames=3)
        oracle.on_reset(_obs(), {})

        # Simulate slow steps by mocking time.perf_counter
        # Each step takes 100ms = 10 FPS (below min_fps of 30)
        times = [0.0]
        for i in range(1, 6):
            times.append(i * 0.1)  # 100ms per step

        with mock.patch("src.oracles.performance.time") as mock_time:
            mock_time.perf_counter = mock.Mock(side_effect=times)

            for _ in range(5):
                oracle.on_step(_obs(), 0.0, False, False, {})

        findings = oracle.get_findings()
        fps_findings = [f for f in findings if f.data.get("type") == "low_fps"]
        assert len(fps_findings) >= 1
        assert fps_findings[0].severity == "warning"
        assert fps_findings[0].data["current_fps"] < 30.0

    def test_normal_fps_no_findings(self):
        """High FPS should not produce any FPS findings."""
        from src.oracles.performance import PerformanceOracle

        oracle = PerformanceOracle(min_fps=20.0, sustained_frames=3)
        oracle.on_reset(_obs(), {})

        # Simulate fast steps: 1ms each = 1000 FPS
        times = [i * 0.001 for i in range(10)]

        with mock.patch("src.oracles.performance.time") as mock_time:
            mock_time.perf_counter = mock.Mock(side_effect=times)

            for _ in range(9):
                oracle.on_step(_obs(), 0.0, False, False, {})

        fps_findings = [f for f in oracle.get_findings() if f.data.get("type") == "low_fps"]
        assert len(fps_findings) == 0

    def test_fps_drop_resets_counter(self):
        """A fast frame after slow frames should reset the low FPS counter."""
        from src.oracles.performance import PerformanceOracle

        oracle = PerformanceOracle(min_fps=30.0, sustained_frames=5)
        oracle.on_reset(_obs(), {})

        # 3 slow steps (10 FPS), then 1 fast step (1000 FPS), then 3 slow
        # slow = 100ms apart, fast = 1ms apart
        times = [
            0.0,  # step 1 start
            0.1,  # step 1: 10 FPS (slow)
            0.2,  # step 2: 10 FPS (slow)
            0.3,  # step 3: 10 FPS (slow)
            0.301,  # step 4: 1000 FPS (fast) — resets counter
            0.401,  # step 5: 10 FPS (slow)
            0.501,  # step 6: 10 FPS (slow)
            0.601,  # step 7: 10 FPS (slow)
        ]

        with mock.patch("src.oracles.performance.time") as mock_time:
            mock_time.perf_counter = mock.Mock(side_effect=times)

            for _ in range(7):
                oracle.on_step(_obs(), 0.0, False, False, {})

        # Should NOT trigger: neither slow run reached sustained_frames of 5
        fps_findings = [f for f in oracle.get_findings() if f.data.get("type") == "low_fps"]
        assert len(fps_findings) == 0

    def test_cpu_threshold_finding(self):
        """CPU usage exceeding threshold should trigger a warning."""
        from src.oracles.performance import PerformanceOracle

        oracle = PerformanceOracle(cpu_threshold=80.0)
        oracle.on_reset(_obs(), {})

        # Mock psutil to return high CPU
        mock_psutil = mock.MagicMock()
        mock_psutil.cpu_percent.return_value = 95.0
        mock_vm = mock.MagicMock()
        mock_vm.used = 1024 * 1024 * 1024  # 1 GB
        mock_psutil.virtual_memory.return_value = mock_vm

        # Resource checks happen every 60 steps, so we need step 60
        # Use fast timing to avoid FPS findings
        times = [i * 0.001 for i in range(62)]

        with mock.patch("src.oracles.performance.time") as mock_time:
            mock_time.perf_counter = mock.Mock(side_effect=times)
            with mock.patch.dict("sys.modules", {"psutil": mock_psutil}):
                for _ in range(60):
                    oracle.on_step(_obs(), 0.0, False, False, {})

        cpu_findings = [f for f in oracle.get_findings() if f.data.get("type") == "high_cpu"]
        assert len(cpu_findings) >= 1
        assert cpu_findings[0].data["cpu_percent"] == 95.0

    def test_ram_threshold_finding(self):
        """RAM usage exceeding threshold should trigger a warning."""
        from src.oracles.performance import PerformanceOracle

        oracle = PerformanceOracle(ram_threshold_mb=500.0)
        oracle.on_reset(_obs(), {})

        # Mock psutil to return high RAM
        mock_psutil = mock.MagicMock()
        mock_psutil.cpu_percent.return_value = 10.0
        mock_vm = mock.MagicMock()
        mock_vm.used = 1024 * 1024 * 1024 * 2  # 2 GB = 2048 MB
        mock_psutil.virtual_memory.return_value = mock_vm

        times = [i * 0.001 for i in range(62)]

        with mock.patch("src.oracles.performance.time") as mock_time:
            mock_time.perf_counter = mock.Mock(side_effect=times)
            with mock.patch.dict("sys.modules", {"psutil": mock_psutil}):
                for _ in range(60):
                    oracle.on_step(_obs(), 0.0, False, False, {})

        ram_findings = [f for f in oracle.get_findings() if f.data.get("type") == "high_ram"]
        assert len(ram_findings) >= 1
        assert ram_findings[0].data["ram_mb"] > 500.0

    def test_get_fps_summary_empty(self):
        """FPS summary should return zeros when no data collected."""
        from src.oracles.performance import PerformanceOracle

        oracle = PerformanceOracle()
        summary = oracle.get_fps_summary()
        assert summary["mean_fps"] == 0.0
        assert summary["min_fps"] == 0.0

    def test_get_fps_summary_with_data(self):
        """FPS summary should compute correct statistics."""
        from src.oracles.performance import PerformanceOracle

        oracle = PerformanceOracle()
        oracle._fps_history = [30.0, 60.0, 60.0, 30.0]
        summary = oracle.get_fps_summary()
        assert summary["mean_fps"] == 45.0
        assert summary["min_fps"] == 30.0
        assert summary["max_fps"] == 60.0

    def test_first_step_no_fps(self):
        """First step has no previous time, so no FPS measurement."""
        from src.oracles.performance import PerformanceOracle

        oracle = PerformanceOracle()
        oracle.on_reset(_obs(), {})
        oracle.on_step(_obs(), 0.0, False, False, {})
        assert oracle._fps_history == []

    def test_sustained_finding_fires_once(self):
        """The sustained low FPS finding should fire exactly once."""
        from src.oracles.performance import PerformanceOracle

        oracle = PerformanceOracle(min_fps=30.0, sustained_frames=3)
        oracle.on_reset(_obs(), {})

        # 10 slow steps at 10 FPS
        times = [i * 0.1 for i in range(12)]

        with mock.patch("src.oracles.performance.time") as mock_time:
            mock_time.perf_counter = mock.Mock(side_effect=times)
            for _ in range(11):
                oracle.on_step(_obs(), 0.0, False, False, {})

        fps_findings = [f for f in oracle.get_findings() if f.data.get("type") == "low_fps"]
        # Should fire exactly once (at sustained_frames == count)
        assert len(fps_findings) == 1

    def test_no_psutil_graceful_fallback(self):
        """If psutil is not importable, resource check returns empty dict."""
        from src.oracles.performance import PerformanceOracle

        oracle = PerformanceOracle()
        # Patch psutil to raise ImportError
        with mock.patch.dict("sys.modules", {"psutil": None}):
            result = oracle._check_resource_usage()
        assert result == {}


# ── Integration / cross-oracle tests ────────────────────────────────


class TestOracleIntegration:
    """Integration tests that verify multiple oracles working together."""

    def test_all_oracles_importable(self):
        """All oracles should be importable from the package."""
        from src.oracles import (
            CrashOracle,
            Oracle,
            PerformanceOracle,
            ScoreAnomalyOracle,
            StuckOracle,
            VisualGlitchOracle,
        )

        assert issubclass(CrashOracle, Oracle)
        assert issubclass(StuckOracle, Oracle)
        assert issubclass(ScoreAnomalyOracle, Oracle)
        assert issubclass(VisualGlitchOracle, Oracle)
        assert issubclass(PerformanceOracle, Oracle)

    def test_oracle_lifecycle(self):
        """Verify the reset -> step -> get_findings -> clear lifecycle."""
        from src.oracles import ScoreAnomalyOracle

        oracle = ScoreAnomalyOracle(max_delta=10.0)

        # Episode 1: trigger a finding
        oracle.on_reset(_obs(), {"score": 0})
        oracle.on_step(_obs(), 0.0, False, False, {"score": 100})
        findings = oracle.get_findings()
        assert len(findings) >= 1

        # Clear and start episode 2
        oracle.clear()
        oracle.on_reset(_obs(), {"score": 0})
        oracle.on_step(_obs(), 0.0, False, False, {"score": 5})
        assert len(oracle.get_findings()) == 0

    def test_multiple_oracles_simultaneous(self):
        """Multiple oracles can process the same step without interference."""
        from src.oracles import CrashOracle, ScoreAnomalyOracle, StuckOracle

        oracles = [
            CrashOracle(),
            StuckOracle(patience=5, reward_epsilon=1e-4, window_size=3),
            ScoreAnomalyOracle(max_delta=10.0),
        ]

        obs = _obs()
        info: dict = {"score": 0}
        for o in oracles:
            o.on_reset(obs, info)

        # 5 steps of zero reward + big score jump at end
        for i in range(4):
            step_info: dict = {"score": i}
            for o in oracles:
                o.on_step(obs, 0.0, False, False, step_info)

        # Big score jump
        step_info = {"score": 500}
        for o in oracles:
            o.on_step(obs, 0.0, False, False, step_info)

        # Collect all findings
        all_findings = []
        for o in oracles:
            all_findings.extend(o.get_findings())

        # Should have stuck finding + score anomaly finding
        stuck = [f for f in all_findings if f.oracle_name == "stuck"]
        score = [f for f in all_findings if f.oracle_name == "score_anomaly"]
        assert len(stuck) >= 1  # 5 zero-reward steps
        assert len(score) >= 1  # score jump 3 -> 500

    def test_all_new_oracles_importable(self):
        """All new oracles should be importable from the package."""
        from src.oracles import (
            BoundaryOracle,
            EpisodeLengthOracle,
            Oracle,
            PhysicsViolationOracle,
            RewardConsistencyOracle,
            SoakOracle,
            StateTransitionOracle,
            TemporalAnomalyOracle,
        )

        assert issubclass(PhysicsViolationOracle, Oracle)
        assert issubclass(BoundaryOracle, Oracle)
        assert issubclass(StateTransitionOracle, Oracle)
        assert issubclass(EpisodeLengthOracle, Oracle)
        assert issubclass(TemporalAnomalyOracle, Oracle)
        assert issubclass(RewardConsistencyOracle, Oracle)
        assert issubclass(SoakOracle, Oracle)


# ── PhysicsViolationOracle ──────────────────────────────────────────


class TestPhysicsViolationOracle:
    """Tests for PhysicsViolationOracle on_step detection logic."""

    def test_init_defaults(self):
        """PhysicsViolationOracle should initialise with default params."""
        from src.oracles.physics_violation import PhysicsViolationOracle

        oracle = PhysicsViolationOracle()
        assert oracle.name == "physics_violation"
        assert oracle.max_ball_speed == 0.15

    def test_on_reset_clears_state(self):
        """on_reset should reset tracking state."""
        from src.oracles.physics_violation import PhysicsViolationOracle

        oracle = PhysicsViolationOracle()
        oracle._prev_ball_pos = np.array([0.5, 0.5])
        oracle._step_count = 10
        oracle.on_reset(_obs(), {"brick_count": 50})
        assert oracle._prev_ball_pos is None
        assert oracle._prev_ball_vel is None
        assert oracle._prev_brick_count == 50
        assert oracle._step_count == 0

    def test_tunneling_detected(self):
        """Ball moving faster than max_ball_speed should flag tunneling."""
        from src.oracles.physics_violation import PhysicsViolationOracle

        oracle = PhysicsViolationOracle(max_ball_speed=0.1)
        oracle.on_reset(_obs(), {})

        oracle.on_step(
            _obs(),
            0.0,
            False,
            False,
            {"ball_pos": [0.1, 0.1], "ball_velocity": [0.01, 0.01]},
        )
        oracle.on_step(
            _obs(),
            0.0,
            False,
            False,
            {"ball_pos": [0.5, 0.5], "ball_velocity": [0.01, 0.01]},
        )

        findings = oracle.get_findings()
        tunnel = [f for f in findings if f.data.get("type") == "tunneling"]
        assert len(tunnel) == 1
        assert tunnel[0].severity == "warning"
        assert tunnel[0].data["displacement"] > 0.1

    def test_no_tunneling_for_normal_speed(self):
        """Ball within max_ball_speed should not flag tunneling."""
        from src.oracles.physics_violation import PhysicsViolationOracle

        oracle = PhysicsViolationOracle(max_ball_speed=0.5)
        oracle.on_reset(_obs(), {})

        oracle.on_step(
            _obs(),
            0.0,
            False,
            False,
            {"ball_pos": [0.1, 0.1], "ball_velocity": [0.01, 0.01]},
        )
        oracle.on_step(
            _obs(),
            0.0,
            False,
            False,
            {"ball_pos": [0.15, 0.12], "ball_velocity": [0.01, 0.01]},
        )

        assert len(oracle.get_findings()) == 0

    def test_paddle_pass_through_detected(self):
        """Ball crossing paddle Y without velocity reversal flags critical."""
        from src.oracles.physics_violation import PhysicsViolationOracle

        oracle = PhysicsViolationOracle(max_ball_speed=1.0)
        oracle.on_reset(_obs(), {})

        # Step 1: ball above paddle, moving down
        oracle.on_step(
            _obs(),
            0.0,
            False,
            False,
            {
                "ball_pos": [0.5, 0.7],
                "ball_velocity": [0.0, 0.1],
                "paddle_pos": [0.5, 0.8],
            },
        )
        # Step 2: ball below paddle, still moving down (no bounce)
        oracle.on_step(
            _obs(),
            0.0,
            False,
            False,
            {
                "ball_pos": [0.5, 0.9],
                "ball_velocity": [0.0, 0.1],
                "paddle_pos": [0.5, 0.8],
            },
        )

        findings = oracle.get_findings()
        passthrough = [f for f in findings if f.data.get("type") == "paddle_pass_through"]
        assert len(passthrough) == 1
        assert passthrough[0].severity == "critical"

    def test_paddle_bounce_no_finding(self):
        """Ball crossing paddle Y with velocity reversal is normal."""
        from src.oracles.physics_violation import PhysicsViolationOracle

        oracle = PhysicsViolationOracle(max_ball_speed=1.0)
        oracle.on_reset(_obs(), {})

        # Step 1: ball above paddle, moving down
        oracle.on_step(
            _obs(),
            0.0,
            False,
            False,
            {
                "ball_pos": [0.5, 0.7],
                "ball_velocity": [0.0, 0.1],
                "paddle_pos": [0.5, 0.8],
            },
        )
        # Step 2: ball crosses paddle but velocity reversed (bounce)
        oracle.on_step(
            _obs(),
            0.0,
            False,
            False,
            {
                "ball_pos": [0.5, 0.85],
                "ball_velocity": [0.0, -0.1],
                "paddle_pos": [0.5, 0.8],
            },
        )

        passthrough = [
            f for f in oracle.get_findings() if f.data.get("type") == "paddle_pass_through"
        ]
        assert len(passthrough) == 0

    def test_ghost_collision_detected(self):
        """Velocity reversal in brick region without brick decrease flags ghost."""
        from src.oracles.physics_violation import PhysicsViolationOracle

        oracle = PhysicsViolationOracle(max_ball_speed=1.0)
        oracle.on_reset(_obs(), {"brick_count": 20})

        # Step 1: ball moving up in brick region
        oracle.on_step(
            _obs(),
            0.0,
            False,
            False,
            {
                "ball_pos": [0.5, 0.3],
                "ball_velocity": [0.0, -0.1],
                "brick_count": 20,
            },
        )
        # Step 2: velocity reversed (collision) but brick count unchanged
        oracle.on_step(
            _obs(),
            0.0,
            False,
            False,
            {
                "ball_pos": [0.5, 0.25],
                "ball_velocity": [0.0, 0.1],
                "brick_count": 20,
            },
        )

        findings = oracle.get_findings()
        ghost = [f for f in findings if f.data.get("type") == "ghost_collision"]
        assert len(ghost) == 1
        assert ghost[0].severity == "warning"

    def test_no_finding_without_ball_data(self):
        """Missing ball data in info should not crash."""
        from src.oracles.physics_violation import PhysicsViolationOracle

        oracle = PhysicsViolationOracle()
        oracle.on_reset(_obs(), {})
        oracle.on_step(_obs(), 0.0, False, False, {})
        assert len(oracle.get_findings()) == 0


# ── BoundaryOracle ──────────────────────────────────────────────────


class TestBoundaryOracle:
    """Tests for BoundaryOracle on_step detection logic."""

    def test_init_defaults(self):
        """BoundaryOracle should initialise with default params."""
        from src.oracles.boundary import BoundaryOracle

        oracle = BoundaryOracle()
        assert oracle.name == "boundary"
        assert oracle.min_bound == 0.0
        assert oracle.max_bound == 1.0
        assert oracle.margin == 0.05

    def test_on_reset_clears_state(self):
        """on_reset should reset step counter."""
        from src.oracles.boundary import BoundaryOracle

        oracle = BoundaryOracle()
        oracle._step_count = 42
        oracle.on_reset(_obs(), {})
        assert oracle._step_count == 0

    def test_hard_oob_detected(self):
        """Object far outside bounds should flag critical hard OOB."""
        from src.oracles.boundary import BoundaryOracle

        oracle = BoundaryOracle(min_bound=0.0, max_bound=1.0, margin=0.05)
        oracle.on_reset(_obs(), {})

        # Ball way outside bounds
        oracle.on_step(_obs(), 0.0, False, False, {"ball_pos": [-0.2, 0.5]})

        findings = oracle.get_findings()
        hard = [f for f in findings if f.data.get("type") == "hard_oob"]
        assert len(hard) == 1
        assert hard[0].severity == "critical"
        assert hard[0].data["key"] == "ball_pos"
        assert hard[0].data["coordinate"] == "x"

    def test_soft_oob_detected(self):
        """Object near boundary edge flags info soft OOB."""
        from src.oracles.boundary import BoundaryOracle

        oracle = BoundaryOracle(min_bound=0.0, max_bound=1.0, margin=0.05)
        oracle.on_reset(_obs(), {})

        # Ball just outside min_bound but within margin
        oracle.on_step(_obs(), 0.0, False, False, {"ball_pos": [-0.03, 0.5]})

        findings = oracle.get_findings()
        soft = [f for f in findings if f.data.get("type") == "soft_oob"]
        assert len(soft) == 1
        assert soft[0].severity == "info"

    def test_within_bounds_no_finding(self):
        """Object within valid bounds should produce no findings."""
        from src.oracles.boundary import BoundaryOracle

        oracle = BoundaryOracle()
        oracle.on_reset(_obs(), {})

        oracle.on_step(
            _obs(),
            0.0,
            False,
            False,
            {"ball_pos": [0.5, 0.5], "paddle_pos": [0.5, 0.9]},
        )

        assert len(oracle.get_findings()) == 0

    def test_custom_tracked_keys(self):
        """Custom tracked_keys should be monitored."""
        from src.oracles.boundary import BoundaryOracle

        oracle = BoundaryOracle(tracked_keys=["enemy_pos"])
        oracle.on_reset(_obs(), {})

        oracle.on_step(
            _obs(),
            0.0,
            False,
            False,
            {"enemy_pos": [1.5, 0.5]},  # Beyond max_bound + margin
        )

        findings = oracle.get_findings()
        assert len(findings) >= 1
        assert findings[0].data["key"] == "enemy_pos"

    def test_missing_key_safe(self):
        """Missing tracked key in info should not crash."""
        from src.oracles.boundary import BoundaryOracle

        oracle = BoundaryOracle()
        oracle.on_reset(_obs(), {})
        oracle.on_step(_obs(), 0.0, False, False, {})
        assert len(oracle.get_findings()) == 0

    def test_both_coords_checked(self):
        """Both x and y coordinates should be checked independently."""
        from src.oracles.boundary import BoundaryOracle

        oracle = BoundaryOracle(min_bound=0.0, max_bound=1.0, margin=0.05)
        oracle.on_reset(_obs(), {})

        # y is out of bounds, x is fine
        oracle.on_step(_obs(), 0.0, False, False, {"ball_pos": [0.5, 1.2]})

        findings = oracle.get_findings()
        assert len(findings) >= 1
        assert any(f.data["coordinate"] == "y" for f in findings)


# ── StateTransitionOracle ───────────────────────────────────────────


class TestStateTransitionOracle:
    """Tests for StateTransitionOracle on_step detection logic."""

    def test_init_defaults(self):
        """StateTransitionOracle should initialise with default params."""
        from src.oracles.state_transition import StateTransitionOracle

        oracle = StateTransitionOracle()
        assert oracle.name == "state_transition"
        assert oracle.max_lives_loss_per_step == 1

    def test_on_reset_clears_state(self):
        """on_reset should capture initial state."""
        from src.oracles.state_transition import StateTransitionOracle

        oracle = StateTransitionOracle()
        oracle.on_reset(_obs(), {"lives": 3, "level": 1, "game_state": "playing"})
        assert oracle._prev_lives == 3
        assert oracle._prev_level == 1
        assert oracle._prev_game_state == "playing"
        assert oracle._step_count == 0

    def test_excessive_lives_loss_detected(self):
        """Losing more than max_lives_loss_per_step flags critical."""
        from src.oracles.state_transition import StateTransitionOracle

        oracle = StateTransitionOracle(max_lives_loss_per_step=1)
        oracle.on_reset(_obs(), {"lives": 5})

        oracle.on_step(_obs(), 0.0, False, False, {"lives": 2})

        findings = oracle.get_findings()
        loss = [f for f in findings if f.data.get("type") == "excessive_lives_loss"]
        assert len(loss) == 1
        assert loss[0].severity == "critical"
        assert loss[0].data["delta"] == 3

    def test_normal_life_loss_no_finding(self):
        """Losing exactly 1 life (within max) should produce no error finding."""
        from src.oracles.state_transition import StateTransitionOracle

        oracle = StateTransitionOracle(max_lives_loss_per_step=1)
        oracle.on_reset(_obs(), {"lives": 3})

        oracle.on_step(_obs(), 0.0, False, False, {"lives": 2})

        findings = oracle.get_findings()
        loss = [f for f in findings if f.data.get("type") == "excessive_lives_loss"]
        assert len(loss) == 0

    def test_lives_increase_noted(self):
        """Lives increase should produce an info finding."""
        from src.oracles.state_transition import StateTransitionOracle

        oracle = StateTransitionOracle()
        oracle.on_reset(_obs(), {"lives": 3})

        oracle.on_step(_obs(), 0.0, False, False, {"lives": 4})

        findings = oracle.get_findings()
        inc = [f for f in findings if f.data.get("type") == "lives_increase"]
        assert len(inc) == 1
        assert inc[0].severity == "info"

    def test_level_skip_detected(self):
        """Skipping more than 1 level flags critical."""
        from src.oracles.state_transition import StateTransitionOracle

        oracle = StateTransitionOracle()
        oracle.on_reset(_obs(), {"level": 1})

        oracle.on_step(_obs(), 0.0, False, False, {"level": 4})

        findings = oracle.get_findings()
        skip = [f for f in findings if f.data.get("type") == "level_skip"]
        assert len(skip) == 1
        assert skip[0].severity == "critical"
        assert skip[0].data["delta"] == 3

    def test_level_decrease_detected(self):
        """Level decrease flags warning."""
        from src.oracles.state_transition import StateTransitionOracle

        oracle = StateTransitionOracle()
        oracle.on_reset(_obs(), {"level": 5})

        oracle.on_step(_obs(), 0.0, False, False, {"level": 3})

        findings = oracle.get_findings()
        dec = [f for f in findings if f.data.get("type") == "level_decrease"]
        assert len(dec) == 1
        assert dec[0].severity == "warning"

    def test_normal_level_advance_no_finding(self):
        """Advancing exactly 1 level produces no finding."""
        from src.oracles.state_transition import StateTransitionOracle

        oracle = StateTransitionOracle()
        oracle.on_reset(_obs(), {"level": 1})

        oracle.on_step(_obs(), 0.0, False, False, {"level": 2})

        findings = oracle.get_findings()
        level_findings = [
            f for f in findings if f.data.get("type") in ("level_skip", "level_decrease")
        ]
        assert len(level_findings) == 0

    def test_invalid_state_transition_detected(self):
        """Invalid game state transition flags critical."""
        from src.oracles.state_transition import StateTransitionOracle

        oracle = StateTransitionOracle(
            allowed_transitions={
                "menu": ["playing"],
                "playing": ["paused", "game_over"],
                "paused": ["playing"],
            }
        )
        oracle.on_reset(_obs(), {"game_state": "menu"})

        # menu -> game_over is NOT allowed
        oracle.on_step(_obs(), 0.0, False, False, {"game_state": "game_over"})

        findings = oracle.get_findings()
        trans = [f for f in findings if f.data.get("type") == "invalid_transition"]
        assert len(trans) == 1
        assert trans[0].severity == "critical"
        assert trans[0].data["prev_state"] == "menu"
        assert trans[0].data["curr_state"] == "game_over"

    def test_valid_state_transition_no_finding(self):
        """Valid game state transition produces no finding."""
        from src.oracles.state_transition import StateTransitionOracle

        oracle = StateTransitionOracle(
            allowed_transitions={
                "menu": ["playing"],
                "playing": ["game_over"],
            }
        )
        oracle.on_reset(_obs(), {"game_state": "menu"})

        oracle.on_step(_obs(), 0.0, False, False, {"game_state": "playing"})

        trans = [f for f in oracle.get_findings() if f.data.get("type") == "invalid_transition"]
        assert len(trans) == 0

    def test_no_transition_map_skips_state_check(self):
        """Without allowed_transitions, state transitions are not checked."""
        from src.oracles.state_transition import StateTransitionOracle

        oracle = StateTransitionOracle()  # No allowed_transitions
        oracle.on_reset(_obs(), {"game_state": "menu"})

        oracle.on_step(_obs(), 0.0, False, False, {"game_state": "game_over"})

        trans = [f for f in oracle.get_findings() if f.data.get("type") == "invalid_transition"]
        assert len(trans) == 0

    def test_missing_state_keys_safe(self):
        """Missing lives/level/game_state in info should not crash."""
        from src.oracles.state_transition import StateTransitionOracle

        oracle = StateTransitionOracle()
        oracle.on_reset(_obs(), {})
        oracle.on_step(_obs(), 0.0, False, False, {})
        assert len(oracle.get_findings()) == 0


# ── EpisodeLengthOracle ─────────────────────────────────────────────


class TestEpisodeLengthOracle:
    """Tests for EpisodeLengthOracle on_step detection logic."""

    def test_init_defaults(self):
        """EpisodeLengthOracle should initialise with default params."""
        from src.oracles.episode_length import EpisodeLengthOracle

        oracle = EpisodeLengthOracle()
        assert oracle.name == "episode_length"
        assert oracle.min_steps == 10
        assert oracle.max_steps == 10000

    def test_on_reset_records_previous_episode(self):
        """on_reset should record the previous episode length."""
        from src.oracles.episode_length import EpisodeLengthOracle

        oracle = EpisodeLengthOracle()
        oracle.on_reset(_obs(), {})
        # Run a short episode
        for _ in range(20):
            oracle.on_step(_obs(), 0.0, False, False, {})
        # Reset records the 20-step episode
        oracle.on_reset(_obs(), {})
        assert oracle._episode_lengths == [20]
        assert oracle._step_count == 0

    def test_short_episode_detected(self):
        """Episode shorter than min_steps at termination flags warning."""
        from src.oracles.episode_length import EpisodeLengthOracle

        oracle = EpisodeLengthOracle(min_steps=10)
        oracle.on_reset(_obs(), {})

        # Only 3 steps then terminate
        for _ in range(2):
            oracle.on_step(_obs(), 0.0, False, False, {})
        oracle.on_step(_obs(), 0.0, True, False, {})  # terminated

        findings = oracle.get_findings()
        short = [f for f in findings if f.data.get("type") == "short_episode"]
        assert len(short) == 1
        assert short[0].severity == "warning"
        assert short[0].data["step_count"] == 3

    def test_long_episode_detected(self):
        """Episode exceeding max_steps flags warning."""
        from src.oracles.episode_length import EpisodeLengthOracle

        oracle = EpisodeLengthOracle(max_steps=5)
        oracle.on_reset(_obs(), {})

        for _ in range(6):
            oracle.on_step(_obs(), 0.0, False, False, {})

        findings = oracle.get_findings()
        long_f = [f for f in findings if f.data.get("type") == "long_episode"]
        assert len(long_f) == 1
        assert long_f[0].severity == "warning"

    def test_long_episode_fires_once(self):
        """The long episode warning should fire at most once per episode."""
        from src.oracles.episode_length import EpisodeLengthOracle

        oracle = EpisodeLengthOracle(max_steps=5)
        oracle.on_reset(_obs(), {})

        for _ in range(20):
            oracle.on_step(_obs(), 0.0, False, False, {})

        long_f = [f for f in oracle.get_findings() if f.data.get("type") == "long_episode"]
        assert len(long_f) == 1

    def test_normal_episode_no_findings(self):
        """Episode within bounds should produce no findings."""
        from src.oracles.episode_length import EpisodeLengthOracle

        oracle = EpisodeLengthOracle(min_steps=5, max_steps=100)
        oracle.on_reset(_obs(), {})

        for _ in range(49):
            oracle.on_step(_obs(), 0.0, False, False, {})
        oracle.on_step(_obs(), 0.0, True, False, {})

        findings = oracle.get_findings()
        length_findings = [
            f for f in findings if f.data.get("type") in ("short_episode", "long_episode")
        ]
        assert len(length_findings) == 0

    def test_statistical_outlier_detected(self):
        """Episode length outlier across multiple episodes flags info."""
        from src.oracles.episode_length import EpisodeLengthOracle

        oracle = EpisodeLengthOracle(
            min_steps=1,
            max_steps=10000,
            z_threshold=2.0,
            min_episodes_for_stats=5,
        )

        # Directly seed episode_lengths with varied data to ensure
        # non-zero std (on_reset + terminated both record, so we seed
        # manually to avoid double-counting complexity).
        oracle._episode_lengths = [95, 100, 105, 98, 102]

        oracle.clear()  # Clear any old findings

        # Now run a very short episode (outlier vs mean ~100)
        oracle.on_reset(_obs(), {})
        for _ in range(4):
            oracle.on_step(_obs(), 0.0, False, False, {})
        oracle.on_step(_obs(), 0.0, True, False, {})

        findings = oracle.get_findings()
        outlier = [f for f in findings if f.data.get("type") == "statistical_outlier"]
        assert len(outlier) == 1
        assert outlier[0].severity == "info"

    def test_get_episode_stats_empty(self):
        """Episode stats should return zeros when no episodes recorded."""
        from src.oracles.episode_length import EpisodeLengthOracle

        oracle = EpisodeLengthOracle()
        stats = oracle.get_episode_stats()
        assert stats["mean_length"] == 0.0
        assert stats["num_episodes"] == 0

    def test_get_episode_stats_with_data(self):
        """Episode stats should compute correct statistics."""
        from src.oracles.episode_length import EpisodeLengthOracle

        oracle = EpisodeLengthOracle()
        oracle._episode_lengths = [100, 200, 300]
        stats = oracle.get_episode_stats()
        assert stats["mean_length"] == 200.0
        assert stats["min_length"] == 100.0
        assert stats["max_length"] == 300.0
        assert stats["num_episodes"] == 3


# ── TemporalAnomalyOracle ──────────────────────────────────────────


class TestTemporalAnomalyOracle:
    """Tests for TemporalAnomalyOracle on_step detection logic."""

    def test_init_defaults(self):
        """TemporalAnomalyOracle should initialise with default params."""
        from src.oracles.temporal_anomaly import TemporalAnomalyOracle

        oracle = TemporalAnomalyOracle()
        assert oracle.name == "temporal_anomaly"
        assert oracle.max_speed == 0.15
        assert oracle.flicker_window == 10
        assert oracle.flicker_threshold == 4

    def test_on_reset_clears_state(self):
        """on_reset should reset tracking."""
        from src.oracles.temporal_anomaly import TemporalAnomalyOracle

        oracle = TemporalAnomalyOracle()
        oracle._prev_positions = {"ball_pos": np.array([0.5, 0.5])}
        oracle._step_count = 10
        oracle.on_reset(_obs(), {})
        assert oracle._prev_positions == {}
        assert oracle._step_count == 0

    def test_teleportation_detected(self):
        """Object jumping farther than max_speed flags teleportation."""
        from src.oracles.temporal_anomaly import TemporalAnomalyOracle

        oracle = TemporalAnomalyOracle(max_speed=0.1)
        oracle.on_reset(_obs(), {})

        oracle.on_step(
            _obs(),
            0.0,
            False,
            False,
            {"ball_pos": [0.1, 0.1], "paddle_pos": [0.5, 0.9]},
        )
        oracle.on_step(
            _obs(),
            0.0,
            False,
            False,
            {"ball_pos": [0.8, 0.8], "paddle_pos": [0.5, 0.9]},
        )

        findings = oracle.get_findings()
        teleport = [f for f in findings if f.data.get("type") == "teleportation"]
        assert len(teleport) >= 1
        assert teleport[0].severity == "warning"
        assert teleport[0].data["key"] == "ball_pos"

    def test_no_teleportation_for_normal_movement(self):
        """Normal movement within max_speed should not flag."""
        from src.oracles.temporal_anomaly import TemporalAnomalyOracle

        oracle = TemporalAnomalyOracle(max_speed=0.5)
        oracle.on_reset(_obs(), {})

        oracle.on_step(
            _obs(),
            0.0,
            False,
            False,
            {"ball_pos": [0.1, 0.1]},
        )
        oracle.on_step(
            _obs(),
            0.0,
            False,
            False,
            {"ball_pos": [0.15, 0.12]},
        )

        teleport = [f for f in oracle.get_findings() if f.data.get("type") == "teleportation"]
        assert len(teleport) == 0

    def test_flickering_detected(self):
        """Rapid presence/absence pattern flags flickering."""
        from src.oracles.temporal_anomaly import TemporalAnomalyOracle

        oracle = TemporalAnomalyOracle(
            tracked_keys=["ball_pos"],
            max_speed=10.0,  # High so teleportation doesn't fire
            flicker_window=6,
            flicker_threshold=4,
        )
        oracle.on_reset(_obs(), {})

        # Alternate present/absent for ball_pos
        for i in range(8):
            if i % 2 == 0:
                oracle.on_step(
                    _obs(),
                    0.0,
                    False,
                    False,
                    {"ball_pos": [0.5, 0.5]},
                )
            else:
                oracle.on_step(_obs(), 0.0, False, False, {})

        findings = oracle.get_findings()
        flicker = [f for f in findings if f.data.get("type") == "flickering"]
        assert len(flicker) >= 1
        assert flicker[0].severity == "warning"

    def test_no_flickering_stable_presence(self):
        """Consistently present object should not flag flickering."""
        from src.oracles.temporal_anomaly import TemporalAnomalyOracle

        oracle = TemporalAnomalyOracle(
            tracked_keys=["ball_pos"],
            flicker_window=10,
            flicker_threshold=4,
        )
        oracle.on_reset(_obs(), {})

        for _ in range(15):
            oracle.on_step(
                _obs(),
                0.0,
                False,
                False,
                {"ball_pos": [0.5, 0.5]},
            )

        flicker = [f for f in oracle.get_findings() if f.data.get("type") == "flickering"]
        assert len(flicker) == 0

    def test_missing_key_safe(self):
        """Missing tracked key should not crash."""
        from src.oracles.temporal_anomaly import TemporalAnomalyOracle

        oracle = TemporalAnomalyOracle()
        oracle.on_reset(_obs(), {})
        oracle.on_step(_obs(), 0.0, False, False, {})
        assert len(oracle.get_findings()) == 0


# ── RewardConsistencyOracle ─────────────────────────────────────────


class TestRewardConsistencyOracle:
    """Tests for RewardConsistencyOracle on_step detection logic."""

    def test_init_defaults(self):
        """RewardConsistencyOracle should initialise with default params."""
        from src.oracles.reward_consistency import RewardConsistencyOracle

        oracle = RewardConsistencyOracle()
        assert oracle.name == "reward_consistency"
        assert oracle.check_lives is True
        assert oracle.check_bricks is True

    def test_on_reset_clears_state(self):
        """on_reset should capture initial state."""
        from src.oracles.reward_consistency import RewardConsistencyOracle

        oracle = RewardConsistencyOracle()
        oracle.on_reset(
            _obs(),
            {"score": 0, "lives": 3, "brick_count": 50},
        )
        assert oracle._prev_score == 0
        assert oracle._prev_lives == 3
        assert oracle._prev_brick_count == 50
        assert oracle._step_count == 0

    def test_score_increase_no_reward_mismatch(self):
        """Score increase with zero reward flags warning."""
        from src.oracles.reward_consistency import RewardConsistencyOracle

        oracle = RewardConsistencyOracle()
        oracle.on_reset(_obs(), {"score": 0, "lives": 3})

        oracle.on_step(
            _obs(),
            0.0,
            False,
            False,
            {"score": 100, "lives": 3},
        )

        findings = oracle.get_findings()
        mismatch = [f for f in findings if f.data.get("type") == "score_reward_mismatch"]
        assert len(mismatch) == 1
        assert mismatch[0].severity == "warning"

    def test_score_increase_with_reward_no_finding(self):
        """Score increase with positive reward should not flag."""
        from src.oracles.reward_consistency import RewardConsistencyOracle

        oracle = RewardConsistencyOracle()
        oracle.on_reset(_obs(), {"score": 0, "lives": 3})

        oracle.on_step(
            _obs(),
            10.0,
            False,
            False,
            {"score": 100, "lives": 3},
        )

        mismatch = [
            f for f in oracle.get_findings() if f.data.get("type") == "score_reward_mismatch"
        ]
        assert len(mismatch) == 0

    def test_phantom_reward_detected(self):
        """Reward given with no score/lives change flags phantom."""
        from src.oracles.reward_consistency import RewardConsistencyOracle

        oracle = RewardConsistencyOracle()
        oracle.on_reset(_obs(), {"score": 100, "lives": 3})

        oracle.on_step(
            _obs(),
            5.0,
            False,
            False,
            {"score": 100, "lives": 3},
        )

        findings = oracle.get_findings()
        phantom = [f for f in findings if f.data.get("type") == "phantom_reward"]
        assert len(phantom) == 1
        assert phantom[0].severity == "info"

    def test_lives_loss_positive_reward_mismatch(self):
        """Life lost with positive reward flags warning."""
        from src.oracles.reward_consistency import RewardConsistencyOracle

        oracle = RewardConsistencyOracle()
        oracle.on_reset(_obs(), {"score": 100, "lives": 3})

        oracle.on_step(
            _obs(),
            5.0,
            False,
            False,
            {"score": 100, "lives": 2},
        )

        findings = oracle.get_findings()
        lives_m = [f for f in findings if f.data.get("type") == "lives_reward_mismatch"]
        assert len(lives_m) == 1
        assert lives_m[0].severity == "warning"

    def test_bricks_destroyed_no_reward_mismatch(self):
        """Bricks destroyed with no reward flags warning."""
        from src.oracles.reward_consistency import RewardConsistencyOracle

        oracle = RewardConsistencyOracle()
        oracle.on_reset(
            _obs(),
            {"score": 0, "lives": 3, "brick_count": 50},
        )

        oracle.on_step(
            _obs(),
            0.0,
            False,
            False,
            {"score": 0, "lives": 3, "brick_count": 48},
        )

        findings = oracle.get_findings()
        brick_m = [f for f in findings if f.data.get("type") == "brick_reward_mismatch"]
        assert len(brick_m) == 1
        assert brick_m[0].severity == "warning"
        assert brick_m[0].data["bricks_destroyed"] == 2

    def test_consistent_step_no_findings(self):
        """Fully consistent reward/state step should produce no findings."""
        from src.oracles.reward_consistency import RewardConsistencyOracle

        oracle = RewardConsistencyOracle()
        oracle.on_reset(
            _obs(),
            {"score": 0, "lives": 3, "brick_count": 50},
        )

        # Brick destroyed + score increase + positive reward = consistent
        oracle.on_step(
            _obs(),
            10.0,
            False,
            False,
            {"score": 10, "lives": 3, "brick_count": 49},
        )

        findings = oracle.get_findings()
        # Should only have no mismatch findings (no score_reward_mismatch,
        # no brick_reward_mismatch)
        mismatches = [
            f
            for f in findings
            if f.data.get("type")
            in (
                "score_reward_mismatch",
                "brick_reward_mismatch",
                "lives_reward_mismatch",
            )
        ]
        assert len(mismatches) == 0

    def test_check_lives_disabled(self):
        """Disabling check_lives should skip lives/reward check."""
        from src.oracles.reward_consistency import RewardConsistencyOracle

        oracle = RewardConsistencyOracle(check_lives=False)
        oracle.on_reset(_obs(), {"score": 100, "lives": 3})

        oracle.on_step(
            _obs(),
            5.0,
            False,
            False,
            {"score": 100, "lives": 2},
        )

        lives_m = [
            f for f in oracle.get_findings() if f.data.get("type") == "lives_reward_mismatch"
        ]
        assert len(lives_m) == 0

    def test_missing_keys_safe(self):
        """Missing score/lives/brick_count in info should not crash."""
        from src.oracles.reward_consistency import RewardConsistencyOracle

        oracle = RewardConsistencyOracle()
        oracle.on_reset(_obs(), {})
        oracle.on_step(_obs(), 0.0, False, False, {})
        assert len(oracle.get_findings()) == 0


# ── SoakOracle ──────────────────────────────────────────────────────


class TestSoakOracle:
    """Tests for SoakOracle on_step detection logic."""

    def test_init_defaults(self):
        """SoakOracle should initialise with default params."""
        from src.oracles.soak import SoakOracle

        oracle = SoakOracle()
        assert oracle.name == "soak"
        assert oracle.sample_interval_steps == 300
        assert oracle.leak_threshold_mb_per_min == 10.0
        assert oracle.fps_degradation_threshold == 0.2

    def test_on_reset_records_fps(self):
        """on_reset should record per-episode FPS and reset step counter."""
        from src.oracles.soak import SoakOracle

        oracle = SoakOracle()
        oracle.on_reset(_obs(), {})

        # Simulate timing
        oracle._episode_step_times = [0.01, 0.01, 0.01]  # 100 FPS
        oracle.on_reset(_obs(), {})

        assert len(oracle._episode_fps_means) == 1
        assert oracle._episode_fps_means[0] == pytest.approx(100.0, rel=0.01)
        assert oracle._step_count == 0

    def test_memory_leak_detection(self):
        """Steadily increasing RAM should flag memory leak."""
        from src.oracles.soak import SoakOracle

        oracle = SoakOracle(
            sample_interval_steps=1,
            leak_threshold_mb_per_min=5.0,
            min_samples_for_trend=5,
        )
        oracle.on_reset(_obs(), {})

        # Simulate increasing RAM by mocking _get_ram_usage_mb and time
        ram_values = [100.0 + i * 10.0 for i in range(10)]
        oracle._get_ram_usage_mb = mock.Mock(side_effect=ram_values)

        # Mock time.perf_counter to advance 6 seconds per step (10 MB/min)
        start = oracle._start_time
        times = [start + i * 6.0 for i in range(10)]

        with mock.patch("src.oracles.soak.time") as mock_time:
            mock_time.perf_counter = mock.Mock(side_effect=times)
            for _ in range(10):
                oracle.on_step(_obs(), 0.0, False, False, {})

        findings = oracle.get_findings()
        leak = [f for f in findings if f.data.get("type") == "memory_leak"]
        assert len(leak) == 1
        assert leak[0].severity == "warning"
        assert leak[0].data["slope_mb_per_min"] > 5.0

    def test_stable_ram_no_leak(self):
        """Stable RAM usage should not flag memory leak."""
        from src.oracles.soak import SoakOracle

        oracle = SoakOracle(
            sample_interval_steps=1,
            leak_threshold_mb_per_min=5.0,
            min_samples_for_trend=5,
        )
        oracle.on_reset(_obs(), {})

        # Stable RAM
        oracle._get_ram_usage_mb = mock.Mock(return_value=500.0)

        start = oracle._start_time
        times = [start + i * 1.0 for i in range(10)]

        with mock.patch("src.oracles.soak.time") as mock_time:
            mock_time.perf_counter = mock.Mock(side_effect=times)
            for _ in range(10):
                oracle.on_step(_obs(), 0.0, False, False, {})

        leak = [f for f in oracle.get_findings() if f.data.get("type") == "memory_leak"]
        assert len(leak) == 0

    def test_fps_degradation_detection(self):
        """FPS dropping across episodes should flag degradation."""
        from src.oracles.soak import SoakOracle

        oracle = SoakOracle(fps_degradation_threshold=0.2)
        oracle.on_reset(_obs(), {})

        # Simulate 4 episodes with declining FPS
        fps_data = [100.0, 90.0, 80.0, 60.0]
        for fps in fps_data:
            oracle._episode_fps_means.append(fps)

        # Trigger check by ending an episode
        oracle._total_steps = 1
        oracle.on_step(_obs(), 0.0, True, False, {})

        findings = oracle.get_findings()
        degradation = [f for f in findings if f.data.get("type") == "fps_degradation"]
        assert len(degradation) == 1
        assert degradation[0].severity == "warning"

    def test_stable_fps_no_degradation(self):
        """Stable FPS across episodes should not flag."""
        from src.oracles.soak import SoakOracle

        oracle = SoakOracle(fps_degradation_threshold=0.2)
        oracle.on_reset(_obs(), {})

        # 4 episodes of stable FPS
        oracle._episode_fps_means = [60.0, 58.0, 59.0, 57.0]
        oracle._total_steps = 1
        oracle.on_step(_obs(), 0.0, True, False, {})

        degradation = [f for f in oracle.get_findings() if f.data.get("type") == "fps_degradation"]
        assert len(degradation) == 0

    def test_get_soak_summary_empty(self):
        """Soak summary should return base data when no samples collected."""
        from src.oracles.soak import SoakOracle

        oracle = SoakOracle()
        summary = oracle.get_soak_summary()
        assert summary["total_steps"] == 0
        assert summary["num_episodes"] == 0
        assert summary["num_ram_samples"] == 0

    def test_get_soak_summary_with_data(self):
        """Soak summary should include RAM and FPS data when available."""
        from src.oracles.soak import SoakOracle

        oracle = SoakOracle()
        oracle._ram_samples = [(0.0, 100.0), (60.0, 150.0)]
        oracle._episode_fps_means = [60.0, 55.0]
        oracle._total_steps = 1000

        summary = oracle.get_soak_summary()
        assert summary["total_steps"] == 1000
        assert summary["num_episodes"] == 2
        assert summary["ram_start_mb"] == 100.0
        assert summary["ram_end_mb"] == 150.0
        assert summary["ram_growth_mb"] == 50.0
        assert summary["fps_first_episode"] == 60.0
        assert summary["fps_latest_episode"] == 55.0

    def test_leak_warned_fires_once(self):
        """Memory leak warning should fire at most once."""
        from src.oracles.soak import SoakOracle

        oracle = SoakOracle(
            sample_interval_steps=1,
            leak_threshold_mb_per_min=1.0,
            min_samples_for_trend=3,
        )
        oracle.on_reset(_obs(), {})

        # Rapidly increasing RAM
        ram_values = [100.0 + i * 50.0 for i in range(20)]
        oracle._get_ram_usage_mb = mock.Mock(side_effect=ram_values)

        start = oracle._start_time
        times = [start + i * 6.0 for i in range(20)]

        with mock.patch("src.oracles.soak.time") as mock_time:
            mock_time.perf_counter = mock.Mock(side_effect=times)
            for _ in range(20):
                oracle.on_step(_obs(), 0.0, False, False, {})

        leak = [f for f in oracle.get_findings() if f.data.get("type") == "memory_leak"]
        assert len(leak) == 1  # Fires exactly once


# ── Dedup / Cooldown Tests (Copilot review fixes) ──────────────────


class TestCrashOracleDedup:
    """Verify CrashOracle dedup for black-frame and frozen-frame findings."""

    def test_consecutive_black_frames_single_finding(self):
        """Consecutive black frames should produce only one finding."""
        from src.oracles.crash import CrashOracle

        oracle = CrashOracle(freeze_threshold=100)
        oracle.on_reset(_obs(), {})

        black = np.zeros((100, 100, 3), dtype=np.uint8)
        for _ in range(5):
            oracle.on_step(_obs(), 0.0, False, False, {"frame": black})

        bf = [f for f in oracle.get_findings() if f.data.get("type") == "black_frame"]
        assert len(bf) == 1

    def test_black_frame_resets_after_normal(self):
        """After a non-black frame, the next black frame produces a new finding."""
        from src.oracles.crash import CrashOracle

        oracle = CrashOracle(freeze_threshold=100)
        oracle.on_reset(_obs(), {})

        black = np.zeros((100, 100, 3), dtype=np.uint8)
        normal = _frame()

        # First black streak
        oracle.on_step(_obs(), 0.0, False, False, {"frame": black})
        oracle.on_step(_obs(), 0.0, False, False, {"frame": black})
        # Non-black resets the flag
        oracle.on_step(_obs(), 0.0, False, False, {"frame": normal})
        # Second black streak
        oracle.on_step(_obs(), 0.0, False, False, {"frame": black})

        bf = [f for f in oracle.get_findings() if f.data.get("type") == "black_frame"]
        assert len(bf) == 2

    def test_frozen_frame_dedup(self):
        """Frozen frame detected at threshold fires only once per freeze."""
        from src.oracles.crash import CrashOracle

        threshold = 3
        oracle = CrashOracle(freeze_threshold=threshold)
        oracle.on_reset(_obs(), {})

        same = _frame(color=(50, 50, 50))
        # Send threshold + 5 identical frames to well exceed threshold
        for _ in range(threshold + 5):
            oracle.on_step(_obs(), 0.0, False, False, {"frame": same})

        ff = [f for f in oracle.get_findings() if f.data.get("type") == "frozen_frame"]
        assert len(ff) == 1

    def test_frozen_frame_re_fires_after_change(self):
        """After a different frame, a new freeze produces another finding."""
        from src.oracles.crash import CrashOracle

        threshold = 3
        oracle = CrashOracle(freeze_threshold=threshold)
        oracle.on_reset(_obs(), {})

        same1 = _frame(color=(50, 50, 50))
        diff = _frame(color=(200, 200, 200))
        same2 = _frame(color=(80, 80, 80))

        # First freeze
        for _ in range(threshold + 2):
            oracle.on_step(_obs(), 0.0, False, False, {"frame": same1})
        # Break the freeze
        oracle.on_step(_obs(), 0.0, False, False, {"frame": diff})
        # Second freeze
        for _ in range(threshold + 2):
            oracle.on_step(_obs(), 0.0, False, False, {"frame": same2})

        ff = [f for f in oracle.get_findings() if f.data.get("type") == "frozen_frame"]
        assert len(ff) == 2


class TestScoreAnomalyDedup:
    """Verify ScoreAnomalyOracle dedup for negative-score findings."""

    def test_consecutive_negative_scores_single_finding(self):
        """Consecutive negative scores should produce only one finding."""
        from src.oracles.score_anomaly import ScoreAnomalyOracle

        oracle = ScoreAnomalyOracle(max_delta=1000, allow_negative=False)
        oracle.on_reset(_obs(), {})

        for i in range(5):
            oracle.on_step(_obs(), 0.0, False, False, {"score": -10.0 - i})

        neg = [f for f in oracle.get_findings() if "Negative" in f.description]
        assert len(neg) == 1

    def test_negative_score_re_fires_after_positive(self):
        """After score goes positive, the next negative produces a new finding."""
        from src.oracles.score_anomaly import ScoreAnomalyOracle

        oracle = ScoreAnomalyOracle(max_delta=1000, allow_negative=False)
        oracle.on_reset(_obs(), {})

        # First negative streak
        oracle.on_step(_obs(), 0.0, False, False, {"score": -5.0})
        oracle.on_step(_obs(), 0.0, False, False, {"score": -3.0})
        # Back to positive
        oracle.on_step(_obs(), 0.0, False, False, {"score": 10.0})
        # Second negative
        oracle.on_step(_obs(), 0.0, False, False, {"score": -1.0})

        neg = [f for f in oracle.get_findings() if "Negative" in f.description]
        assert len(neg) == 2


class TestTemporalAnomalyCooldown:
    """Verify TemporalAnomalyOracle flicker cooldown."""

    def test_flicker_cooldown_suppresses_immediate_refire(self):
        """After reporting a flicker, another flicker within the window is suppressed."""
        from src.oracles.temporal_anomaly import TemporalAnomalyOracle

        oracle = TemporalAnomalyOracle(
            tracked_keys=["ball_pos"],
            flicker_window=6,
            flicker_threshold=4,
        )
        oracle.on_reset(_obs(), {})

        # Alternate present/absent to trigger flicker
        for i in range(6):
            info = {"ball_pos": [0.5, 0.5]} if i % 2 == 0 else {}
            oracle.on_step(_obs(), 0.0, False, False, info)

        # Should have fired once
        flicker1 = [f for f in oracle.get_findings() if f.data.get("type") == "flickering"]
        assert len(flicker1) == 1

        # Continue flickering immediately — still within cooldown
        for i in range(4):
            info = {"ball_pos": [0.5, 0.5]} if i % 2 == 0 else {}
            oracle.on_step(_obs(), 0.0, False, False, info)

        flicker2 = [f for f in oracle.get_findings() if f.data.get("type") == "flickering"]
        # Should still be just 1 — cooldown suppresses
        assert len(flicker2) == 1

    def test_flicker_re_fires_after_cooldown(self):
        """After cooldown expires, a new flicker sequence produces a new finding."""
        from src.oracles.temporal_anomaly import TemporalAnomalyOracle

        oracle = TemporalAnomalyOracle(
            tracked_keys=["ball_pos"],
            flicker_window=4,
            flicker_threshold=3,
        )
        oracle.on_reset(_obs(), {})

        # Alternate to trigger first flicker (4 steps)
        for i in range(4):
            info = {"ball_pos": [0.5, 0.5]} if i % 2 == 0 else {}
            oracle.on_step(_obs(), 0.0, False, False, info)

        f1 = [f for f in oracle.get_findings() if f.data.get("type") == "flickering"]
        assert len(f1) == 1

        # Wait out the cooldown (flicker_window = 4 more stable steps)
        for _ in range(4):
            oracle.on_step(_obs(), 0.0, False, False, {"ball_pos": [0.5, 0.5]})

        # New flicker sequence
        for i in range(4):
            info = {"ball_pos": [0.5, 0.5]} if i % 2 == 0 else {}
            oracle.on_step(_obs(), 0.0, False, False, info)

        f2 = [f for f in oracle.get_findings() if f.data.get("type") == "flickering"]
        assert len(f2) == 2


class TestVisualGlitchCv2Guard:
    """Verify cv2 import guards in VisualGlitchOracle."""

    def test_compute_phash_returns_none_without_cv2(self):
        """_compute_phash returns None when cv2 is unavailable."""
        from src.oracles.visual_glitch import VisualGlitchOracle

        oracle = VisualGlitchOracle()
        frame = _frame()

        with mock.patch.dict("sys.modules", {"cv2": None}):
            result = oracle._compute_phash(frame)
        # Should return None gracefully (cv2 import fails)
        assert result is None

    def test_compute_ssim_raises_runtime_error_without_cv2(self):
        """_compute_ssim raises RuntimeError when cv2 is unavailable."""
        from src.oracles.visual_glitch import VisualGlitchOracle

        oracle = VisualGlitchOracle()
        frame_a = _frame()
        frame_b = _frame(color=(200, 200, 200))

        with mock.patch.dict("sys.modules", {"cv2": None}):
            with pytest.raises(RuntimeError, match="OpenCV"):
                oracle._compute_ssim(frame_a, frame_b)
