"""Tests for the orchestrator module: FrameCollector, SessionRunner, and CLI scripts.

Covers:
- FrameCollector: construction, interval-based saving, finalize, manifest
- Finding-to-FindingReport bridge
- EpisodeMetrics computation
- Oracle factory (_create_oracles)
- SessionRunner: construction, setup, episode execution, cleanup
- CLI scripts: run_session.py arg parsing, train_rl.py arg parsing
- FrameCollectionCallback: construction, callback creation
- Env bug fixes: close() resets _initialized, step_count property,
  fragile reset retry, bricks_total corruption guard
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.orchestrator.data_collector import FrameCollector
from src.orchestrator.session_runner import (
    SessionRunner,
    _build_episode_metrics,
    _create_oracles,
    _finding_to_report,
)
from src.oracles.base import Finding
from src.reporting import EpisodeMetrics, FindingReport


# -- Helpers -------------------------------------------------------------------


def _fake_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Create a synthetic BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


@pytest.fixture(autouse=True)
def _mock_cv2_imwrite(monkeypatch):
    """Mock cv2.imwrite so tests don't depend on libGL (missing in CI Docker).

    Creates real 0-byte files so path-existence assertions still pass.
    """

    def _fake_imwrite(path: str, img, params=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"")
        return True

    # Build a lightweight mock cv2 module with imwrite
    mock_cv2 = mock.MagicMock()
    mock_cv2.imwrite = _fake_imwrite
    monkeypatch.setitem(sys.modules, "cv2", mock_cv2)


# ===========================================================================
# FrameCollector Tests
# ===========================================================================


class TestFrameCollector:
    """Tests for FrameCollector."""

    def test_construction_defaults(self, tmp_path):
        """FrameCollector creates output directory on construction."""
        collector = FrameCollector(output_dir=tmp_path, capture_interval=10)
        assert collector.frame_count == 0
        assert collector.output_dir.exists()

    def test_capture_interval_minimum(self, tmp_path):
        """Capture interval cannot be less than 1."""
        collector = FrameCollector(output_dir=tmp_path, capture_interval=0)
        # interval of 0 is clamped to 1, so every call saves
        frame = _fake_frame()
        result = collector.save_frame(frame, step=0, episode_id=0)
        assert result is not None
        assert collector.frame_count == 1

    def test_save_frame_respects_interval(self, tmp_path):
        """Only every Nth frame is saved."""
        collector = FrameCollector(output_dir=tmp_path, capture_interval=3)
        frame = _fake_frame()

        results = []
        for i in range(9):
            r = collector.save_frame(frame, step=i, episode_id=0)
            results.append(r)

        # Should save on calls 3, 6, 9 (1-indexed step_counter)
        saved = [r for r in results if r is not None]
        assert len(saved) == 3
        assert collector.frame_count == 3

    def test_save_frame_returns_path(self, tmp_path):
        """Saved frame returns a Path to the PNG file."""
        collector = FrameCollector(output_dir=tmp_path, capture_interval=1)
        frame = _fake_frame()
        result = collector.save_frame(frame, step=0, episode_id=0)
        assert result is not None
        assert result.suffix == ".png"
        assert result.exists()

    def test_save_frame_returns_none_when_skipped(self, tmp_path):
        """Non-interval frames return None."""
        collector = FrameCollector(output_dir=tmp_path, capture_interval=5)
        frame = _fake_frame()
        result = collector.save_frame(frame, step=0, episode_id=0)
        assert result is None

    def test_finalize_writes_manifest(self, tmp_path):
        """finalize() writes a manifest.json with correct metadata."""
        collector = FrameCollector(output_dir=tmp_path, capture_interval=1)
        frame = _fake_frame()
        collector.save_frame(frame, step=0, episode_id=0)
        collector.save_frame(frame, step=1, episode_id=0)

        output_dir = collector.finalize()
        manifest_path = output_dir / "manifest.json"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["total_frames"] == 2
        assert manifest["capture_interval"] == 1
        assert manifest["total_steps_seen"] == 2
        assert len(manifest["frames"]) == 2

    def test_finalize_manifest_frame_metadata(self, tmp_path):
        """Each frame entry in manifest has correct fields."""
        collector = FrameCollector(output_dir=tmp_path, capture_interval=1)
        frame = _fake_frame()
        collector.save_frame(frame, step=42, episode_id=3, metadata={"action": 1})

        output_dir = collector.finalize()
        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)

        entry = manifest["frames"][0]
        assert entry["step"] == 42
        assert entry["episode_id"] == 3
        assert "timestamp" in entry
        assert entry["metadata"]["action"] == 1

    def test_output_dir_property(self, tmp_path):
        """output_dir property returns the correct path."""
        collector = FrameCollector(output_dir=tmp_path, capture_interval=1)
        assert collector.output_dir.parent == tmp_path


# ===========================================================================
# Finding-to-FindingReport Bridge Tests
# ===========================================================================


class TestFindingBridge:
    """Tests for _finding_to_report conversion."""

    def test_basic_conversion(self, tmp_path):
        """Finding fields map to FindingReport correctly."""
        finding = Finding(
            oracle_name="test_oracle",
            severity="warning",
            step=10,
            description="Test issue",
            data={"key": "value"},
            frame=None,
        )
        report = _finding_to_report(finding, tmp_path / "screenshots")
        assert isinstance(report, FindingReport)
        assert report.oracle_name == "test_oracle"
        assert report.severity == "warning"
        assert report.step == 10
        assert report.description == "Test issue"
        assert report.data == {"key": "value"}
        assert report.screenshot_path is None

    def test_with_frame_saves_screenshot(self, tmp_path):
        """When finding has a frame, it is saved as PNG."""
        frame = _fake_frame()
        finding = Finding(
            oracle_name="crash",
            severity="critical",
            step=5,
            description="Crash detected",
            data={},
            frame=frame,
        )
        screenshots_dir = tmp_path / "screenshots"
        report = _finding_to_report(finding, screenshots_dir)
        assert report.screenshot_path is not None
        assert Path(report.screenshot_path).exists()
        assert screenshots_dir.exists()

    def test_without_frame_no_screenshot(self, tmp_path):
        """When finding has no frame, screenshot_path is None."""
        finding = Finding(
            oracle_name="stuck",
            severity="info",
            step=100,
            description="Stuck detected",
            data={},
            frame=None,
        )
        report = _finding_to_report(finding, tmp_path / "screenshots")
        assert report.screenshot_path is None

    def test_severity_preserved(self, tmp_path):
        """All severity levels are preserved in conversion."""
        for sev in ("critical", "warning", "info"):
            finding = Finding(
                oracle_name="test",
                severity=sev,
                step=0,
                description=f"{sev} finding",
                data={},
            )
            report = _finding_to_report(finding, tmp_path / "screenshots")
            assert report.severity == sev

    def test_data_dict_preserved(self, tmp_path):
        """Complex data dicts are preserved through conversion."""
        finding = Finding(
            oracle_name="test",
            severity="info",
            step=0,
            description="test",
            data={"nested": {"key": [1, 2, 3]}, "value": 42.0},
        )
        report = _finding_to_report(finding, tmp_path / "screenshots")
        assert report.data["nested"]["key"] == [1, 2, 3]
        assert report.data["value"] == 42.0


# ===========================================================================
# EpisodeMetrics Tests
# ===========================================================================


class TestBuildEpisodeMetrics:
    """Tests for _build_episode_metrics."""

    def test_basic_metrics(self):
        """Computes mean/min FPS and max reward from step data."""
        step_times = [0.033, 0.05, 0.02]  # ~30, 20, 50 FPS
        rewards = [1.0, -0.5, 2.0]
        start = time.perf_counter() - 1.0  # 1 second ago

        metrics = _build_episode_metrics(step_times, rewards, start)

        assert isinstance(metrics, EpisodeMetrics)
        assert metrics.mean_fps is not None
        assert metrics.mean_fps > 0
        assert metrics.min_fps is not None
        assert metrics.min_fps > 0
        assert metrics.max_reward_per_step == 2.0
        assert metrics.total_duration_seconds is not None
        assert metrics.total_duration_seconds >= 1.0

    def test_empty_step_times(self):
        """Empty step times yield None for FPS metrics."""
        metrics = _build_episode_metrics([], [], time.perf_counter())
        assert metrics.mean_fps is None
        assert metrics.min_fps is None
        assert metrics.max_reward_per_step is None

    def test_min_fps_is_minimum(self):
        """min_fps should be the minimum across all steps."""
        # 0.1s per step = 10 FPS, 0.01s = 100 FPS
        step_times = [0.1, 0.01, 0.05]
        metrics = _build_episode_metrics(
            step_times, [0.0, 0.0, 0.0], time.perf_counter()
        )
        assert metrics.min_fps is not None
        assert metrics.min_fps == pytest.approx(10.0, rel=0.01)

    def test_zero_step_time_handled(self):
        """Near-zero step times don't cause division errors."""
        step_times = [0.0, 0.033]
        rewards = [0.0, 1.0]
        metrics = _build_episode_metrics(step_times, rewards, time.perf_counter())
        assert metrics.mean_fps is not None
        # 0.0 is clamped to 1e-9, giving huge FPS — but no crash
        assert metrics.mean_fps > 0


# ===========================================================================
# Oracle Factory Tests
# ===========================================================================


class TestCreateOracles:
    """Tests for _create_oracles."""

    def test_creates_12_oracles(self):
        """Factory returns exactly 12 oracle instances."""
        oracles = _create_oracles()
        assert len(oracles) == 12

    def test_all_are_oracle_instances(self):
        """All returned objects are Oracle subclasses."""
        from src.oracles.base import Oracle

        oracles = _create_oracles()
        for o in oracles:
            assert isinstance(o, Oracle)

    def test_unique_names(self):
        """All oracles have unique names."""
        oracles = _create_oracles()
        names = [o.name for o in oracles]
        assert len(names) == len(set(names))


# ===========================================================================
# SessionRunner Construction Tests
# ===========================================================================


class TestSessionRunnerConstruction:
    """Tests for SessionRunner constructor."""

    def test_default_construction(self):
        """SessionRunner can be constructed with defaults."""
        runner = SessionRunner()
        assert runner.n_episodes == 3
        assert runner.max_steps_per_episode == 10_000
        assert runner.frame_capture_interval == 30
        assert runner.enable_data_collection is True

    def test_custom_construction(self):
        """SessionRunner accepts custom parameters."""
        runner = SessionRunner(
            game_config="configs/games/breakout-71.yaml",
            n_episodes=5,
            max_steps_per_episode=5000,
            output_dir="custom_output",
            browser="chrome",
            yolo_weights="custom/weights.pt",
            frame_capture_interval=10,
            enable_data_collection=False,
        )
        assert runner.n_episodes == 5
        assert runner.max_steps_per_episode == 5000
        assert runner.browser == "chrome"
        assert runner.enable_data_collection is False
        assert runner.yolo_weights == Path("custom/weights.pt")

    def test_paths_are_pathlib(self):
        """String paths are converted to Path objects."""
        runner = SessionRunner(
            game_config="configs/games/breakout-71.yaml",
            yolo_weights="weights/best.pt",
            output_dir="output",
        )
        assert isinstance(runner.game_config_path, Path)
        assert isinstance(runner.yolo_weights, Path)
        assert isinstance(runner.output_dir, Path)


# ===========================================================================
# SessionRunner._run_episode Tests (mocked subsystems)
# ===========================================================================


class TestSessionRunnerEpisode:
    """Tests for SessionRunner._run_episode with mocked subsystems."""

    def _make_runner_with_mock_env(self, steps_before_done=5, n_bricks=10):
        """Create a SessionRunner with a fully mocked environment."""
        runner = SessionRunner(
            n_episodes=1,
            max_steps_per_episode=100,
            enable_data_collection=False,
        )

        mock_env = mock.MagicMock()
        mock_env.action_space = mock.MagicMock()
        mock_env.action_space.sample.return_value = np.array([0.0], dtype=np.float32)
        mock_env.step_count = 0
        mock_env._oracles = []

        frame = _fake_frame()
        obs = np.zeros(8, dtype=np.float32)

        info = {"frame": frame, "detections": {}, "step": 0}

        mock_env.reset.return_value = (obs, info)

        call_count = 0

        def step_fn(action):
            nonlocal call_count
            call_count += 1
            mock_env.step_count = call_count
            done = call_count >= steps_before_done
            return obs, 1.0, done, False, info

        mock_env.step.side_effect = step_fn

        runner._env = mock_env
        return runner

    def test_episode_runs_to_termination(self):
        """Episode runs until env signals termination."""
        runner = self._make_runner_with_mock_env(steps_before_done=3)
        episode = runner._run_episode(0)

        assert episode.episode_id == 0
        assert episode.steps == 3
        assert episode.terminated is True
        assert episode.truncated is False

    def test_episode_accumulates_reward(self):
        """Episode correctly sums reward across steps."""
        runner = self._make_runner_with_mock_env(steps_before_done=5)
        episode = runner._run_episode(0)
        assert episode.total_reward == pytest.approx(5.0)

    def test_episode_collects_oracle_findings(self):
        """Oracle findings are collected and converted to FindingReports."""
        runner = self._make_runner_with_mock_env(steps_before_done=2)

        # Add mock oracle with a finding
        mock_oracle = mock.MagicMock()
        finding = Finding(
            oracle_name="test",
            severity="warning",
            step=1,
            description="Test finding",
        )
        mock_oracle.get_findings.return_value = [finding]
        runner._env._oracles = [mock_oracle]

        episode = runner._run_episode(0)
        assert len(episode.findings) == 1
        assert episode.findings[0].oracle_name == "test"
        assert episode.findings[0].severity == "warning"

    def test_episode_has_metrics(self):
        """Episode report includes performance metrics."""
        runner = self._make_runner_with_mock_env(steps_before_done=3)
        episode = runner._run_episode(0)

        assert episode.metrics is not None
        assert episode.metrics.total_duration_seconds is not None
        assert episode.metrics.total_duration_seconds >= 0

    def test_episode_with_data_collection(self, tmp_path):
        """FrameCollector is called during episode when enabled."""
        runner = self._make_runner_with_mock_env(steps_before_done=3)
        runner._collector = FrameCollector(output_dir=tmp_path, capture_interval=1)

        episode = runner._run_episode(0)  # noqa: F841
        assert runner._collector.frame_count > 0


# ===========================================================================
# SessionRunner._cleanup Tests
# ===========================================================================


class TestSessionRunnerCleanup:
    """Tests for SessionRunner._cleanup."""

    def test_cleanup_closes_env(self):
        """Cleanup calls env.close()."""
        runner = SessionRunner()
        mock_env = mock.MagicMock()
        runner._env = mock_env
        runner._browser_instance = None

        runner._cleanup()
        mock_env.close.assert_called_once()
        assert runner._env is None

    def test_cleanup_closes_browser(self):
        """Cleanup calls browser_instance.close()."""
        runner = SessionRunner()
        runner._env = None
        mock_browser = mock.MagicMock()
        runner._browser_instance = mock_browser

        runner._cleanup()
        mock_browser.close.assert_called_once()
        assert runner._browser_instance is None

    def test_cleanup_handles_env_error(self):
        """Cleanup continues even if env.close() raises."""
        runner = SessionRunner()
        mock_env = mock.MagicMock()
        mock_env.close.side_effect = RuntimeError("close failed")
        runner._env = mock_env
        runner._browser_instance = None

        runner._cleanup()  # Should not raise
        assert runner._env is None

    def test_cleanup_handles_browser_error(self):
        """Cleanup continues even if browser.close() raises."""
        runner = SessionRunner()
        runner._env = None
        mock_browser = mock.MagicMock()
        mock_browser.close.side_effect = RuntimeError("close failed")
        runner._browser_instance = mock_browser

        runner._cleanup()  # Should not raise
        assert runner._browser_instance is None

    def test_cleanup_stops_loader(self):
        """Cleanup calls loader.stop() and sets _loader to None."""
        runner = SessionRunner()
        runner._env = None
        runner._browser_instance = None
        mock_loader = mock.MagicMock()
        runner._loader = mock_loader

        runner._cleanup()
        mock_loader.stop.assert_called_once()
        assert runner._loader is None

    def test_cleanup_handles_loader_error(self):
        """Cleanup continues even if loader.stop() raises."""
        runner = SessionRunner()
        runner._env = None
        runner._browser_instance = None
        mock_loader = mock.MagicMock()
        mock_loader.stop.side_effect = RuntimeError("stop failed")
        runner._loader = mock_loader

        runner._cleanup()  # Should not raise
        assert runner._loader is None


# ===========================================================================
# Env Bug Fix Tests
# ===========================================================================


class TestEnvBugFixes:
    """Tests for the env bug fixes applied in this session."""

    def test_close_resets_initialized(self):
        """close() should set _initialized to False."""
        from src.env.breakout71_env import Breakout71Env

        env = Breakout71Env()
        env._initialized = True
        env._capture = mock.MagicMock()
        env._detector = mock.MagicMock()
        env._game_canvas = mock.MagicMock()

        env.close()
        assert env._initialized is False
        assert env._game_canvas is None

    def test_step_count_property(self):
        """step_count property returns _step_count."""
        from src.env.breakout71_env import Breakout71Env

        env = Breakout71Env()
        assert env.step_count == 0

        env._step_count = 42
        assert env.step_count == 42

    def test_step_count_is_readonly(self):
        """step_count property cannot be set directly."""
        from src.env.breakout71_env import Breakout71Env

        env = Breakout71Env()
        with pytest.raises(AttributeError):
            env.step_count = 10

    def test_window_title_default(self):
        """Default window_title should be 'Breakout' (not 'Breakout - 71')."""
        from src.env.breakout71_env import Breakout71Env

        env = Breakout71Env()
        assert env.window_title == "Breakout"

    def test_reset_retries_for_ball(self):
        """reset() retries up to 5 times to detect ball."""
        from src.env.breakout71_env import Breakout71Env

        mock_driver = mock.MagicMock()
        mock_canvas = mock.MagicMock()
        mock_driver.execute_script.return_value = {
            "state": "gameplay",
            "details": {},
        }

        env = Breakout71Env(driver=mock_driver)
        env._initialized = True
        env._game_canvas = mock_canvas
        env._canvas_dims = (0, 0, 1280, 1024)

        mock_capture = mock.MagicMock()
        mock_detector = mock.MagicMock()

        env._capture = mock_capture
        env._detector = mock_detector

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_capture.capture_frame.return_value = frame

        # First 2 detections: no ball. Third: ball found.
        no_ball = {
            "paddle": (0.5, 0.9, 0.1, 0.02),
            "ball": None,
            "bricks": [(0.1, 0.1, 0.05, 0.03)] * 10,
            "powerups": [],
            "raw_detections": [],
        }
        with_ball = {
            "paddle": (0.5, 0.9, 0.1, 0.02),
            "ball": (0.5, 0.5, 0.02, 0.02),
            "bricks": [(0.1, 0.1, 0.05, 0.03)] * 10,
            "powerups": [],
            "raw_detections": [],
        }
        mock_detector.detect_to_game_state.side_effect = [no_ball, no_ball, with_ball]

        with mock.patch("src.env.breakout71_env.time") as mock_time:
            mock_time.sleep = mock.MagicMock()
            obs, info = env.reset()

        # Ball found on third attempt
        assert obs[1] == pytest.approx(0.5)  # ball_x from with_ball

    def test_bricks_total_retry_on_zero(self):
        """_build_observation retries when 0 bricks detected on reset."""
        from src.env.breakout71_env import Breakout71Env

        env = Breakout71Env()
        env._initialized = True

        mock_capture = mock.MagicMock()
        mock_detector = mock.MagicMock()
        env._capture = mock_capture
        env._detector = mock_detector

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_capture.capture_frame.return_value = frame

        # Retry detection returns 10 bricks
        retry_dets = {
            "paddle": (0.5, 0.9, 0.1, 0.02),
            "ball": (0.5, 0.5, 0.02, 0.02),
            "bricks": [(0.1 * i, 0.1, 0.05, 0.03) for i in range(10)],
            "powerups": [],
            "raw_detections": [],
        }
        mock_detector.detect_to_game_state.return_value = retry_dets

        # Call _build_observation with 0 bricks on reset
        detections_no_bricks = {
            "paddle": (0.5, 0.9, 0.1, 0.02),
            "ball": (0.5, 0.5, 0.02, 0.02),
            "bricks": [],
            "powerups": [],
            "raw_detections": [],
        }

        with mock.patch("src.env.breakout71_env.time") as mock_time:
            mock_time.sleep = mock.MagicMock()
            env._build_observation(detections_no_bricks, reset=True)

        # Should have retried and found 10 bricks
        assert env._bricks_total == 10


# ===========================================================================
# CLI Arg Parsing Tests
# ===========================================================================


class TestRunSessionCLI:
    """Tests for scripts/run_session.py argument parsing."""

    def test_default_args(self):
        """Default arguments are set correctly."""
        from scripts.run_session import parse_args

        args = parse_args([])
        assert args.config == "configs/games/breakout-71.yaml"
        assert args.episodes == 3
        assert args.max_steps == 10_000
        assert args.output_dir == "output"
        assert args.browser is None
        assert args.verbose is False
        assert args.no_data_collection is False

    def test_custom_args(self):
        """Custom arguments are parsed correctly."""
        from scripts.run_session import parse_args

        args = parse_args(
            [
                "--config",
                "custom.yaml",
                "--episodes",
                "5",
                "--max-steps",
                "5000",
                "--browser",
                "chrome",
                "--frame-interval",
                "10",
                "--no-data-collection",
                "-v",
            ]
        )
        assert args.config == "custom.yaml"
        assert args.episodes == 5
        assert args.max_steps == 5000
        assert args.browser == "chrome"
        assert args.frame_interval == 10
        assert args.no_data_collection is True
        assert args.verbose is True

    def test_browser_choices(self):
        """Only valid browser choices are accepted."""
        from scripts.run_session import parse_args

        # Valid browsers should work
        for browser in ("chrome", "msedge", "firefox"):
            args = parse_args(["--browser", browser])
            assert args.browser == browser

        # Invalid browser should fail
        with pytest.raises(SystemExit):
            parse_args(["--browser", "safari"])


class TestTrainRLCLI:
    """Tests for scripts/train_rl.py argument parsing."""

    def test_default_args(self):
        """Default arguments are set correctly."""
        from scripts.train_rl import parse_args

        args = parse_args([])
        assert args.timesteps == 200_000
        assert args.checkpoint_interval == 50_000
        assert args.n_steps == 2048
        assert args.batch_size == 64
        assert args.gamma == 0.99
        assert args.lr == 3e-4
        assert args.clip_range == 0.2
        assert args.ent_coef == 0.01
        assert args.device == "cpu"

    def test_custom_ppo_args(self):
        """Custom PPO hyperparameters are parsed."""
        from scripts.train_rl import parse_args

        args = parse_args(
            [
                "--n-steps",
                "1024",
                "--batch-size",
                "32",
                "--gamma",
                "0.95",
                "--lr",
                "1e-3",
                "--ent-coef",
                "0.05",
                "--device",
                "cuda",
            ]
        )
        assert args.n_steps == 1024
        assert args.batch_size == 32
        assert args.gamma == 0.95
        assert args.lr == 1e-3
        assert args.ent_coef == 0.05
        assert args.device == "cuda"


# ===========================================================================
# FrameCollectionCallback Tests
# ===========================================================================


class TestFrameCollectionCallback:
    """Tests for the SB3 FrameCollectionCallback."""

    def test_construction(self, tmp_path):
        """Callback factory can be constructed."""
        from scripts.train_rl import FrameCollectionCallback

        cb = FrameCollectionCallback(
            collector=None,
            checkpoint_interval=10_000,
            checkpoint_dir=tmp_path / "checkpoints",
        )
        assert cb._checkpoint_interval == 10_000

    def test_create_returns_callback(self, tmp_path):
        """create() returns an SB3 BaseCallback instance."""
        from stable_baselines3.common.callbacks import BaseCallback

        from scripts.train_rl import FrameCollectionCallback

        cb = FrameCollectionCallback(
            collector=None,
            checkpoint_interval=10_000,
            checkpoint_dir=tmp_path / "checkpoints",
        )
        callback = cb.create()
        assert isinstance(callback, BaseCallback)

    def test_callback_checkpoint_dir_created(self, tmp_path):
        """Checkpoint directory is created on construction."""
        from scripts.train_rl import FrameCollectionCallback

        ckpt_dir = tmp_path / "checkpoints"
        FrameCollectionCallback(
            checkpoint_dir=ckpt_dir,
        )
        assert ckpt_dir.exists()


# ===========================================================================
# Integration: SessionRunner.run() with mocked subsystems
# ===========================================================================


# ===========================================================================
# SessionRunner._setup Tests (mocked imports)
# ===========================================================================


class TestSessionRunnerSetup:
    """Tests for SessionRunner._setup with mocked lazy imports."""

    def test_setup_creates_env_and_browser(self, tmp_path):
        """_setup passes correct args to BrowserInstance and Breakout71Env."""
        runner = SessionRunner(
            game_config="breakout-71",
            output_dir=tmp_path,
            enable_data_collection=True,
            frame_capture_interval=10,
            browser="chrome",
        )
        runner.game_config_path = Path("breakout-71")

        mock_config = mock.MagicMock()
        mock_config.url = "http://localhost:1234"
        mock_config.window_width = 1280
        mock_config.window_height = 1024
        mock_config.window_title = "Breakout"

        mock_loader = mock.MagicMock()
        mock_loader.url = "http://localhost:5678"

        mock_browser = mock.MagicMock()
        mock_browser.driver = mock.MagicMock()

        mock_env = mock.MagicMock()

        with (
            mock.patch(
                "scripts._smoke_utils.BrowserInstance",
                return_value=mock_browser,
            ) as MockBrowser,
            mock.patch(
                "src.env.breakout71_env.Breakout71Env",
                return_value=mock_env,
            ) as MockEnv,
            mock.patch(
                "src.game_loader.create_loader",
                return_value=mock_loader,
            ),
            mock.patch(
                "src.game_loader.config.load_game_config",
                return_value=mock_config,
            ),
        ):
            runner._setup()

        # Verify BrowserInstance received correct arguments
        MockBrowser.assert_called_once_with(
            url="http://localhost:5678",
            settle_seconds=8.0,
            window_size=(1280, 1024),
            browser="chrome",
        )

        # Verify Breakout71Env received correct arguments
        MockEnv.assert_called_once()
        env_kwargs = MockEnv.call_args[1]
        assert env_kwargs["window_title"] == "Breakout"
        assert env_kwargs["driver"] is mock_browser.driver

        assert runner._env is mock_env
        assert runner._browser_instance is mock_browser
        assert runner._loader is mock_loader
        assert runner._collector is not None

    def test_setup_full_integration(self, tmp_path):
        """_setup() calls all lazy imports and wires everything together."""
        runner = SessionRunner(
            game_config="breakout-71",
            output_dir=tmp_path,
            enable_data_collection=True,
        )
        runner.game_config_path = Path("breakout-71")

        mock_config = mock.MagicMock()
        mock_config.url = "http://localhost:1234"
        mock_config.window_width = 1280
        mock_config.window_height = 1024
        mock_config.window_title = "Breakout"

        mock_loader = mock.MagicMock()
        mock_loader.url = "http://localhost:1234"

        mock_browser = mock.MagicMock()
        mock_browser.driver = mock.MagicMock()

        mock_env = mock.MagicMock()

        # Patch all lazy imports at the point where _setup imports them
        with (
            mock.patch(
                "scripts._smoke_utils.BrowserInstance",
                return_value=mock_browser,
            ),
            mock.patch(
                "src.env.breakout71_env.Breakout71Env",
                return_value=mock_env,
            ),
            mock.patch(
                "src.game_loader.create_loader",
                return_value=mock_loader,
            ),
            mock.patch(
                "src.game_loader.config.load_game_config",
                return_value=mock_config,
            ),
        ):
            runner._setup()

        assert runner._env is not None
        assert runner._browser_instance is not None
        assert runner._loader is not None
        assert runner._collector is not None
        mock_loader.setup.assert_called_once()
        mock_loader.start.assert_called_once()

    def test_setup_without_data_collection(self, tmp_path):
        """_setup() does not create collector when data collection disabled."""
        runner = SessionRunner(
            game_config="breakout-71",
            output_dir=tmp_path,
            enable_data_collection=False,
        )
        runner.game_config_path = Path("breakout-71")

        mock_config = mock.MagicMock()
        mock_config.url = "http://localhost:1234"
        mock_config.window_width = 1280
        mock_config.window_height = 1024
        mock_config.window_title = "Breakout"

        mock_loader = mock.MagicMock()
        mock_loader.url = "http://localhost:1234"

        mock_browser = mock.MagicMock()
        mock_browser.driver = mock.MagicMock()

        with (
            mock.patch(
                "scripts._smoke_utils.BrowserInstance",
                return_value=mock_browser,
            ),
            mock.patch(
                "src.env.breakout71_env.Breakout71Env",
                return_value=mock.MagicMock(),
            ),
            mock.patch(
                "src.game_loader.create_loader",
                return_value=mock_loader,
            ),
            mock.patch(
                "src.game_loader.config.load_game_config",
                return_value=mock_config,
            ),
        ):
            runner._setup()

        assert runner._env is not None
        assert runner._collector is None

    def test_setup_uses_loader_url_over_config_url(self, tmp_path):
        """_setup() prefers loader.url over config.url when available."""
        runner = SessionRunner(
            game_config="breakout-71",
            output_dir=tmp_path,
            enable_data_collection=False,
        )
        runner.game_config_path = Path("breakout-71")

        mock_config = mock.MagicMock()
        mock_config.url = "http://localhost:9999"  # config URL
        mock_config.window_width = 1280
        mock_config.window_height = 1024
        mock_config.window_title = "Breakout"

        mock_loader = mock.MagicMock()
        mock_loader.url = "http://localhost:5555"  # loader URL (preferred)

        mock_browser_cls = mock.MagicMock()
        mock_browser = mock.MagicMock()
        mock_browser.driver = mock.MagicMock()
        mock_browser_cls.return_value = mock_browser

        with (
            mock.patch(
                "scripts._smoke_utils.BrowserInstance",
                mock_browser_cls,
            ),
            mock.patch(
                "src.env.breakout71_env.Breakout71Env",
                return_value=mock.MagicMock(),
            ),
            mock.patch(
                "src.game_loader.create_loader",
                return_value=mock_loader,
            ),
            mock.patch(
                "src.game_loader.config.load_game_config",
                return_value=mock_config,
            ),
        ):
            runner._setup()

        # BrowserInstance should have been called with the loader URL
        call_kwargs = mock_browser_cls.call_args
        assert call_kwargs[1]["url"] == "http://localhost:5555"


# ===========================================================================
# Integration: SessionRunner.run() with mocked subsystems
# ===========================================================================


class TestSessionRunnerRun:
    """Integration test for SessionRunner.run() with mocked game lifecycle."""

    def _make_runner_with_mock_setup(
        self,
        tmp_path,
        n_episodes=2,
        enable_data_collection=False,
        steps_per_episode=3,
    ):
        """Create a SessionRunner with _setup mocked to inject a fake env."""
        runner = SessionRunner(
            n_episodes=n_episodes,
            output_dir=tmp_path,
            enable_data_collection=enable_data_collection,
        )

        mock_env = mock.MagicMock()
        mock_env.action_space = mock.MagicMock()
        mock_env.action_space.sample.return_value = np.array([0.0], dtype=np.float32)
        mock_env._oracles = []

        frame = _fake_frame()
        obs = np.zeros(8, dtype=np.float32)
        info = {"frame": frame, "detections": {}, "step": 0}

        mock_env.reset.return_value = (obs, info)

        call_counter = {"count": 0}

        def step_fn(action):
            call_counter["count"] += 1
            step = call_counter["count"]
            mock_env.step_count = step
            done = step % steps_per_episode == 0
            return obs, 1.0, done, False, info

        mock_env.step.side_effect = step_fn

        def mock_setup():
            runner._env = mock_env
            runner._browser_instance = mock.MagicMock()

        runner._setup = mock_setup
        return runner

    def test_run_returns_session_report(self, tmp_path):
        """run() returns a SessionReport with correct episode count."""
        runner = self._make_runner_with_mock_setup(tmp_path, n_episodes=2)
        report = runner.run()

        from src.reporting import SessionReport

        assert isinstance(report, SessionReport)
        assert len(report.episodes) == 2

    def test_run_finalizes_data_collection(self, tmp_path):
        """run() calls collector.finalize() when data collection is enabled."""
        runner = self._make_runner_with_mock_setup(
            tmp_path,
            n_episodes=1,
            enable_data_collection=True,
            steps_per_episode=2,
        )

        # Override _setup to also attach a mock collector
        original_setup = runner._setup

        def setup_with_collector():
            original_setup()
            runner._collector = mock.MagicMock()
            runner._collector.finalize.return_value = tmp_path / "frames"
            runner._collector.frame_count = 5

        runner._setup = setup_with_collector

        report = runner.run()
        runner._collector.finalize.assert_called_once()
        assert len(report.episodes) == 1

    def test_run_generates_dashboard(self, tmp_path):
        """run() generates an HTML dashboard after the report."""
        runner = self._make_runner_with_mock_setup(
            tmp_path, n_episodes=1, steps_per_episode=2
        )
        runner.run()

        # Dashboard file should exist in reports dir
        reports_dir = tmp_path / "reports"
        assert reports_dir.exists()
        # The dashboard renderer writes an HTML file
        html_files = list(reports_dir.glob("*.html"))
        assert len(html_files) >= 1
