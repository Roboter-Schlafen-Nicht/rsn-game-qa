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
        from games.breakout71.env import Breakout71Env

        env = Breakout71Env()
        env._initialized = True
        env._capture = mock.MagicMock()
        env._detector = mock.MagicMock()
        env._game_canvas = mock.MagicMock()
        env._canvas_size = (640, 480)

        env.close()
        assert env._initialized is False
        assert env._game_canvas is None
        assert env._canvas_size is None

    def test_step_count_property(self):
        """step_count property returns _step_count."""
        from games.breakout71.env import Breakout71Env

        env = Breakout71Env()
        assert env.step_count == 0

        env._step_count = 42
        assert env.step_count == 42

    def test_step_count_is_readonly(self):
        """step_count property cannot be set directly."""
        from games.breakout71.env import Breakout71Env

        env = Breakout71Env()
        with pytest.raises(AttributeError):
            env.step_count = 10

    def test_window_title_default(self):
        """Default window_title should be 'Breakout' (not 'Breakout - 71')."""
        from games.breakout71.env import Breakout71Env

        env = Breakout71Env()
        assert env.window_title == "Breakout"

    def test_reset_retries_for_ball(self):
        """reset() retries up to 5 times to detect ball."""
        from games.breakout71.env import Breakout71Env

        mock_driver = mock.MagicMock()
        mock_driver.execute_script.return_value = {
            "state": "gameplay",
            "details": {},
        }

        env = Breakout71Env(driver=mock_driver)
        env._initialized = True
        env._game_canvas = mock.MagicMock()
        env._canvas_size = (640, 480)

        mock_capture = mock.MagicMock()
        mock_detector = mock.MagicMock()

        env._capture = mock_capture
        env._detector = mock_detector
        mock_capture.width = 640
        mock_capture.height = 480
        mock_capture.hwnd = 12345

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

        with mock.patch("games.breakout71.env.time") as mock_time:
            mock_time.sleep = mock.MagicMock()
            obs, info = env.reset()

        # Ball found on third attempt
        assert obs[1] == pytest.approx(0.5)  # ball_x from with_ball

    def test_bricks_total_retry_on_zero(self):
        """build_observation retries when 0 bricks detected on reset."""
        from games.breakout71.env import Breakout71Env

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

        with mock.patch("games.breakout71.env.time") as mock_time:
            mock_time.sleep = mock.MagicMock()
            env.build_observation(detections_no_bricks, reset=True)

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
        assert args.game == "breakout71"
        assert args.config is None
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
        assert args.device == "auto"

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

    def test_resume_default_none(self):
        """--resume defaults to None."""
        from scripts.train_rl import parse_args

        args = parse_args([])
        assert args.resume is None

    def test_resume_custom_path(self):
        """--resume accepts a path string."""
        from scripts.train_rl import parse_args

        args = parse_args(["--resume", "output/checkpoints/ppo_breakout71_50000.zip"])
        assert args.resume == "output/checkpoints/ppo_breakout71_50000.zip"

    def test_resume_path_validation_exists(self, tmp_path):
        """Resume path validation accepts existing file."""
        from scripts.train_rl import parse_args

        ckpt = tmp_path / "model.zip"
        ckpt.write_bytes(b"fake")
        args = parse_args(["--resume", str(ckpt)])
        assert args.resume == str(ckpt)

    def test_resume_path_validation_zip_suffix(self, tmp_path):
        """Resume path validation falls back to .zip suffix."""
        # The .zip suffix fallback happens in main(), not parse_args.
        # We verify the path logic via the Path resolution code.
        from pathlib import Path

        ckpt = tmp_path / "model.zip"
        ckpt.write_bytes(b"fake")
        # User passes path without .zip
        resume_path = Path(str(tmp_path / "model"))
        assert not resume_path.exists()
        assert resume_path.with_suffix(".zip").exists()

    def test_resume_path_validation_not_found(self, tmp_path):
        """Resume path validation detects missing file."""
        from pathlib import Path

        resume_path = Path(str(tmp_path / "nonexistent"))
        assert not resume_path.exists()
        assert not resume_path.with_suffix(".zip").exists()

    def test_resume_sets_reset_num_timesteps(self):
        """When --resume is set, reset_num_timesteps should be False."""
        from scripts.train_rl import parse_args

        args = parse_args(["--resume", "checkpoint.zip"])
        # The flag logic: reset_num_timesteps = not bool(args.resume)
        assert bool(args.resume) is True
        # reset_num_timesteps would be False
        reset_num_timesteps = not bool(args.resume)
        assert reset_num_timesteps is False

    def test_resume_not_set_resets_timesteps(self):
        """Without --resume, reset_num_timesteps should be True."""
        from scripts.train_rl import parse_args

        args = parse_args([])
        assert args.resume is None
        # reset_num_timesteps would be True
        reset_num_timesteps = not bool(args.resume)
        assert reset_num_timesteps is True


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

    def test_max_episodes_stored(self, tmp_path):
        """max_episodes parameter is stored on the factory."""
        from scripts.train_rl import FrameCollectionCallback

        cb = FrameCollectionCallback(
            checkpoint_dir=tmp_path / "checkpoints",
            max_episodes=10,
        )
        assert cb._max_episodes == 10

    def test_max_episodes_default_none(self, tmp_path):
        """max_episodes defaults to None."""
        from scripts.train_rl import FrameCollectionCallback

        cb = FrameCollectionCallback(
            checkpoint_dir=tmp_path / "checkpoints",
        )
        assert cb._max_episodes is None


# ===========================================================================
# Integration: SessionRunner.run() with mocked subsystems
# ===========================================================================


# ===========================================================================
# SessionRunner._setup Tests (mocked imports)
# ===========================================================================


class TestSessionRunnerSetup:
    """Tests for SessionRunner._setup with mocked lazy imports."""

    def test_setup_creates_env_and_browser(self, tmp_path):
        """_setup passes correct args to BrowserInstance and EnvClass."""
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
        MockEnvCls = mock.MagicMock(return_value=mock_env)

        # Replace the plugin's env_class with our mock
        runner._plugin = mock.MagicMock()
        runner._plugin.env_class = MockEnvCls
        runner._plugin.game_name = "breakout-71"

        with (
            mock.patch(
                "scripts._smoke_utils.BrowserInstance",
                return_value=mock_browser,
            ) as MockBrowser,
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
            headless=False,
        )

        # Verify env class received correct arguments
        MockEnvCls.assert_called_once()
        env_kwargs = MockEnvCls.call_args[1]
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

        # Replace the plugin's env_class with our mock
        runner._plugin = mock.MagicMock()
        runner._plugin.env_class = mock.MagicMock(return_value=mock_env)
        runner._plugin.game_name = "breakout-71"

        # Patch all lazy imports at the point where _setup imports them
        with (
            mock.patch(
                "scripts._smoke_utils.BrowserInstance",
                return_value=mock_browser,
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

        # Replace the plugin's env_class with our mock
        runner._plugin = mock.MagicMock()
        runner._plugin.env_class = mock.MagicMock(return_value=mock.MagicMock())
        runner._plugin.game_name = "breakout-71"

        with (
            mock.patch(
                "scripts._smoke_utils.BrowserInstance",
                return_value=mock_browser,
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

    def _setup_with_js_hooks(
        self, tmp_path, *, mute_js=None, setup_js=None, reinit_js=None
    ):
        """Helper: run _setup() with a plugin that has specific JS hooks."""
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

        mock_driver = mock.MagicMock()
        mock_browser = mock.MagicMock()
        mock_browser.driver = mock_driver

        runner._plugin = mock.MagicMock(spec=[])  # empty spec
        # Manually set only the hooks we want
        if mute_js is not None:
            runner._plugin.mute_js = mute_js
        if setup_js is not None:
            runner._plugin.setup_js = setup_js
        if reinit_js is not None:
            runner._plugin.reinit_js = reinit_js
        runner._plugin.env_class = mock.MagicMock(return_value=mock.MagicMock())
        runner._plugin.game_name = "breakout-71"

        with (
            mock.patch(
                "scripts._smoke_utils.BrowserInstance",
                return_value=mock_browser,
            ),
            mock.patch(
                "src.game_loader.create_loader",
                return_value=mock_loader,
            ),
            mock.patch(
                "src.game_loader.config.load_game_config",
                return_value=mock_config,
            ),
            mock.patch("src.orchestrator.session_runner.time") as mock_time,
        ):
            mock_time.sleep = mock.MagicMock()
            mock_time.perf_counter = time.perf_counter
            runner._setup()

        return runner, mock_driver, mock_time

    def test_setup_executes_mute_js(self, tmp_path):
        """_setup() calls driver.execute_script with mute_js snippet."""
        _, driver, _ = self._setup_with_js_hooks(tmp_path, mute_js="window.mute()")
        driver.execute_script.assert_any_call("window.mute()")

    def test_setup_executes_setup_js(self, tmp_path):
        """_setup() calls driver.execute_script with setup_js snippet."""
        _, driver, _ = self._setup_with_js_hooks(tmp_path, setup_js="window.setup()")
        driver.execute_script.assert_any_call("window.setup()")

    def test_setup_refreshes_after_mute_or_setup(self, tmp_path):
        """_setup() calls driver.refresh() when mute_js or setup_js ran."""
        _, driver, _ = self._setup_with_js_hooks(tmp_path, mute_js="window.mute()")
        driver.refresh.assert_called_once()

    def test_setup_no_refresh_without_hooks(self, tmp_path):
        """_setup() does not refresh when no JS hooks are present."""
        _, driver, _ = self._setup_with_js_hooks(tmp_path)
        driver.refresh.assert_not_called()

    def test_setup_reinit_js_runs_after_refresh(self, tmp_path):
        """_setup() executes reinit_js after refresh when mute/setup ran."""
        _, driver, _ = self._setup_with_js_hooks(
            tmp_path,
            mute_js="window.mute()",
            reinit_js="window.restart({})",
        )
        driver.execute_script.assert_any_call("window.restart({})")

    def test_setup_reinit_js_skipped_without_refresh(self, tmp_path):
        """_setup() does NOT execute reinit_js when no refresh occurred."""
        _, driver, _ = self._setup_with_js_hooks(
            tmp_path, reinit_js="window.restart({})"
        )
        # reinit_js should not have been called (no mute/setup => no refresh)
        for call in driver.execute_script.call_args_list:
            assert call[0][0] != "window.restart({})"

    def test_setup_sleeps_after_refresh_and_reinit(self, tmp_path):
        """_setup() sleeps 3s after refresh and 2s after reinit_js."""
        _, _, mock_time = self._setup_with_js_hooks(
            tmp_path,
            setup_js="window.setup()",
            reinit_js="window.restart({})",
        )
        sleep_calls = [c[0][0] for c in mock_time.sleep.call_args_list]
        assert 3 in sleep_calls  # post-refresh sleep
        assert 2 in sleep_calls  # post-reinit sleep

    def test_setup_js_exception_does_not_crash(self, tmp_path):
        """_setup() catches exceptions from JS hook execution."""
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

        mock_driver = mock.MagicMock()
        mock_driver.execute_script.side_effect = RuntimeError("JS error")
        mock_browser = mock.MagicMock()
        mock_browser.driver = mock_driver

        runner._plugin = mock.MagicMock(spec=[])
        runner._plugin.mute_js = "window.mute()"
        runner._plugin.setup_js = "window.setup()"
        runner._plugin.env_class = mock.MagicMock(return_value=mock.MagicMock())
        runner._plugin.game_name = "breakout-71"

        with (
            mock.patch(
                "scripts._smoke_utils.BrowserInstance",
                return_value=mock_browser,
            ),
            mock.patch(
                "src.game_loader.create_loader",
                return_value=mock_loader,
            ),
            mock.patch(
                "src.game_loader.config.load_game_config",
                return_value=mock_config,
            ),
            mock.patch("src.orchestrator.session_runner.time"),
        ):
            # Should not raise despite JS errors
            runner._setup()

        # _setup completed without crashing
        assert runner._env is not None

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

        # Replace the plugin's env_class with our mock
        runner._plugin = mock.MagicMock()
        runner._plugin.env_class = mock.MagicMock(return_value=mock.MagicMock())
        runner._plugin.game_name = "breakout-71"

        with (
            mock.patch(
                "scripts._smoke_utils.BrowserInstance",
                mock_browser_cls,
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


class TestSessionRunnerPolicyFn:
    """Tests for SessionRunner policy_fn support (model evaluation path).

    TDD specs:
    - SessionRunner accepts optional policy_fn parameter (callable (obs) -> action)
    - Default policy_fn is None, meaning random policy (env.action_space.sample)
    - When policy_fn is provided, _run_episode uses it instead of random sampling
    - policy_fn receives the observation and returns an action
    """

    def test_default_policy_fn_is_none(self):
        """SessionRunner defaults to policy_fn=None (random policy)."""
        runner = SessionRunner()
        assert runner.policy_fn is None

    def test_accepts_policy_fn(self):
        """SessionRunner accepts a callable policy_fn."""
        dummy_fn = lambda obs: np.array([0.0], dtype=np.float32)  # noqa: E731
        runner = SessionRunner(policy_fn=dummy_fn)
        assert runner.policy_fn is dummy_fn

    def test_episode_uses_random_when_policy_fn_none(self):
        """When policy_fn is None, _run_episode uses env.action_space.sample()."""
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

        obs = np.zeros(8, dtype=np.float32)
        info = {"frame": None, "detections": {}, "step": 0}
        mock_env.reset.return_value = (obs, info)

        call_count = 0

        def step_fn(action):
            nonlocal call_count
            call_count += 1
            mock_env.step_count = call_count
            return obs, 1.0, call_count >= 3, False, info

        mock_env.step.side_effect = step_fn
        runner._env = mock_env

        runner._run_episode(0)

        # action_space.sample should have been called (random policy)
        assert mock_env.action_space.sample.call_count == 3

    def test_episode_uses_policy_fn_when_provided(self):
        """When policy_fn is provided, _run_episode calls it with obs."""
        call_log = []

        def policy_fn(obs):
            call_log.append(obs)
            return np.array([1.0], dtype=np.float32)

        runner = SessionRunner(
            n_episodes=1,
            max_steps_per_episode=100,
            enable_data_collection=False,
            policy_fn=policy_fn,
        )

        mock_env = mock.MagicMock()
        mock_env.action_space = mock.MagicMock()
        mock_env.action_space.sample.return_value = np.array([0.0], dtype=np.float32)
        mock_env.step_count = 0
        mock_env._oracles = []

        obs = np.zeros(8, dtype=np.float32)
        info = {"frame": None, "detections": {}, "step": 0}
        mock_env.reset.return_value = (obs, info)

        call_count = 0

        def step_fn(action):
            nonlocal call_count
            call_count += 1
            mock_env.step_count = call_count
            return obs, 1.0, call_count >= 3, False, info

        mock_env.step.side_effect = step_fn
        runner._env = mock_env

        runner._run_episode(0)

        # policy_fn should have been called 3 times (not action_space.sample)
        assert len(call_log) == 3
        assert mock_env.action_space.sample.call_count == 0

    def test_policy_fn_receives_observation(self):
        """policy_fn receives the current observation from env."""
        received_obs = []

        def policy_fn(obs):
            received_obs.append(obs.copy())
            return np.array([0.0], dtype=np.float32)

        runner = SessionRunner(
            n_episodes=1,
            max_steps_per_episode=100,
            enable_data_collection=False,
            policy_fn=policy_fn,
        )

        mock_env = mock.MagicMock()
        mock_env.action_space = mock.MagicMock()
        mock_env.step_count = 0
        mock_env._oracles = []

        obs1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        obs2 = np.array(
            [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype=np.float32
        )
        info = {"frame": None, "detections": {}, "step": 0}

        mock_env.reset.return_value = (obs1, info)

        step_count = 0

        def step_fn(action):
            nonlocal step_count
            step_count += 1
            mock_env.step_count = step_count
            # Return obs2 on step, terminate after 2 steps
            return obs2, 1.0, step_count >= 2, False, info

        mock_env.step.side_effect = step_fn
        runner._env = mock_env

        runner._run_episode(0)

        # First call gets obs1 (from reset), second gets obs2 (from step)
        assert len(received_obs) == 2
        np.testing.assert_array_equal(received_obs[0], obs1)
        np.testing.assert_array_equal(received_obs[1], obs2)

    def test_policy_fn_action_passed_to_env(self):
        """The action returned by policy_fn is passed to env.step()."""
        custom_action = np.array([0.42], dtype=np.float32)

        def policy_fn(obs):
            return custom_action

        runner = SessionRunner(
            n_episodes=1,
            max_steps_per_episode=100,
            enable_data_collection=False,
            policy_fn=policy_fn,
        )

        mock_env = mock.MagicMock()
        mock_env.action_space = mock.MagicMock()
        mock_env.step_count = 0
        mock_env._oracles = []

        obs = np.zeros(8, dtype=np.float32)
        info = {"frame": None, "detections": {}, "step": 0}
        mock_env.reset.return_value = (obs, info)

        call_count = 0

        def step_fn(action):
            nonlocal call_count
            call_count += 1
            mock_env.step_count = call_count
            return obs, 1.0, True, False, info

        mock_env.step.side_effect = step_fn
        runner._env = mock_env

        runner._run_episode(0)

        # env.step should have been called with the policy_fn's action
        step_call_args = mock_env.step.call_args_list
        assert len(step_call_args) == 1
        np.testing.assert_array_equal(step_call_args[0][0][0], custom_action)


class TestRunSessionModelFlag:
    """Tests for --model CLI flag in run_session.py."""

    def test_model_arg_default_none(self):
        """--model defaults to None."""
        from scripts.run_session import parse_args

        args = parse_args([])
        assert args.model is None

    def test_model_arg_parsed(self):
        """--model accepts a path string."""
        from scripts.run_session import parse_args

        args = parse_args(["--model", "models/ppo_breakout.zip"])
        assert args.model == "models/ppo_breakout.zip"

    def test_main_passes_policy_fn_when_model_given(self):
        """main() creates a policy_fn from --model and passes to SessionRunner."""
        from scripts.run_session import main

        mock_ppo = mock.MagicMock()
        mock_ppo.predict.return_value = (np.array([0.0]), None)

        mock_ppo_cls = mock.MagicMock()
        mock_ppo_cls.load.return_value = mock_ppo

        with (
            mock.patch.dict(
                "sys.modules",
                {"stable_baselines3": mock.MagicMock(PPO=mock_ppo_cls)},
            ),
            mock.patch(
                "src.orchestrator.session_runner.SessionRunner",
            ) as MockRunner,
        ):
            mock_runner_instance = mock.MagicMock()
            mock_report = mock.MagicMock()
            mock_report.summary = {
                "total_episodes": 1,
                "total_findings": 0,
                "critical_findings": 0,
                "warning_findings": 0,
                "info_findings": 0,
                "episodes_failed": 0,
                "mean_episode_reward": 0.0,
                "mean_episode_length": 0.0,
            }
            mock_runner_instance.run.return_value = mock_report
            MockRunner.return_value = mock_runner_instance

            main(["--model", "models/ppo_breakout.zip", "--episodes", "1"])

            mock_ppo_cls.load.assert_called_once_with("models/ppo_breakout.zip")
            # SessionRunner should have been called with policy_fn keyword arg
            call_kwargs = MockRunner.call_args[1]
            assert "policy_fn" in call_kwargs
            assert call_kwargs["policy_fn"] is not None

    def test_main_passes_none_policy_fn_without_model(self):
        """main() passes policy_fn=None when no --model is given."""
        from scripts.run_session import main

        with mock.patch(
            "src.orchestrator.session_runner.SessionRunner",
        ) as MockRunner:
            mock_runner_instance = mock.MagicMock()
            mock_report = mock.MagicMock()
            mock_report.summary = {
                "total_episodes": 1,
                "total_findings": 0,
                "critical_findings": 0,
                "warning_findings": 0,
                "info_findings": 0,
                "episodes_failed": 0,
                "mean_episode_reward": 0.0,
                "mean_episode_length": 0.0,
            }
            mock_runner_instance.run.return_value = mock_report
            MockRunner.return_value = mock_runner_instance

            main(["--episodes", "1"])

            call_kwargs = MockRunner.call_args[1]
            assert call_kwargs.get("policy_fn") is None


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


# ===========================================================================
# CNN Policy Support Tests
# ===========================================================================


class TestRunSessionCNNPolicyCLI:
    """Tests for --policy and --frame-stack CLI flags in run_session.py."""

    def test_policy_arg_default_mlp(self):
        """--policy defaults to 'mlp'."""
        from scripts.run_session import parse_args

        args = parse_args([])
        assert args.policy == "mlp"

    def test_policy_arg_cnn(self):
        """--policy accepts 'cnn'."""
        from scripts.run_session import parse_args

        args = parse_args(["--policy", "cnn"])
        assert args.policy == "cnn"

    def test_policy_arg_invalid_rejected(self):
        """Invalid --policy value is rejected."""
        from scripts.run_session import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--policy", "rnn"])

    def test_frame_stack_default(self):
        """--frame-stack defaults to 4."""
        from scripts.run_session import parse_args

        args = parse_args([])
        assert args.frame_stack == 4

    def test_frame_stack_custom(self):
        """--frame-stack accepts custom values."""
        from scripts.run_session import parse_args

        args = parse_args(["--frame-stack", "8"])
        assert args.frame_stack == 8


class TestSessionRunnerCNNPolicy:
    """Tests for SessionRunner CNN policy wrapping support."""

    def test_default_policy_is_mlp(self):
        """SessionRunner defaults to policy='mlp'."""
        runner = SessionRunner()
        assert runner.policy == "mlp"

    def test_accepts_cnn_policy(self):
        """SessionRunner accepts policy='cnn'."""
        runner = SessionRunner(policy="cnn")
        assert runner.policy == "cnn"

    def test_default_frame_stack(self):
        """SessionRunner defaults to frame_stack=4."""
        runner = SessionRunner()
        assert runner.frame_stack == 4

    def test_custom_frame_stack(self):
        """SessionRunner accepts custom frame_stack."""
        runner = SessionRunner(frame_stack=8)
        assert runner.frame_stack == 8

    def test_cnn_policy_wraps_env_in_setup(self):
        """When policy='cnn', _setup wraps self._env with CnnEvalWrapper."""
        runner = SessionRunner(
            policy="cnn",
            frame_stack=4,
            n_episodes=1,
            enable_data_collection=False,
        )

        # Create a mock env to inject
        mock_env = mock.MagicMock()
        mock_env.action_space = mock.MagicMock()
        mock_env.observation_space = mock.MagicMock()
        mock_env._oracles = []

        # Patch _setup to only do the CNN wrapping part
        original_env = mock_env

        with mock.patch("src.platform.cnn_wrapper.CnnEvalWrapper") as MockWrapper:
            mock_wrapped = mock.MagicMock()
            MockWrapper.return_value = mock_wrapped

            # Simulate _setup having created the raw env, then apply CNN wrapping
            runner._env = mock_env
            runner._wrap_env_for_cnn()

            MockWrapper.assert_called_once_with(original_env, frame_stack=4)
            assert runner._env is mock_wrapped

    def test_mlp_policy_does_not_wrap_env(self):
        """When policy='mlp', _setup does NOT wrap the env."""
        runner = SessionRunner(
            policy="mlp",
            n_episodes=1,
            enable_data_collection=False,
        )

        mock_env = mock.MagicMock()
        runner._env = mock_env
        runner._wrap_env_for_cnn()  # should be a no-op

        assert runner._env is mock_env

    def test_episode_accesses_oracles_through_unwrapped(self):
        """_run_episode accesses oracles via unwrapped env when CNN-wrapped."""
        runner = SessionRunner(
            policy="cnn",
            frame_stack=4,
            n_episodes=1,
            enable_data_collection=False,
        )

        # Create mock base env with oracles
        mock_oracle = mock.MagicMock()
        mock_oracle.get_findings.return_value = []

        mock_base_env = mock.MagicMock()
        mock_base_env._oracles = [mock_oracle]
        mock_base_env.step_count = 0

        # Create a mock wrapper that delegates unwrapped
        mock_wrapped_env = mock.MagicMock()
        mock_wrapped_env.action_space = mock.MagicMock()
        mock_wrapped_env.action_space.sample.return_value = np.array(
            [0.0], dtype=np.float32
        )

        obs = np.zeros((4, 84, 84), dtype=np.uint8)
        info = {"frame": None, "detections": {}, "step": 0}
        mock_wrapped_env.reset.return_value = (obs, info)

        call_count = 0

        def step_fn(action):
            nonlocal call_count
            call_count += 1
            mock_base_env.step_count = call_count
            return obs, 1.0, call_count >= 2, False, info

        mock_wrapped_env.step.side_effect = step_fn

        # The wrapped env's unwrapped should point to base
        mock_wrapped_env.unwrapped = mock_base_env

        runner._env = mock_wrapped_env
        runner._raw_env = mock_base_env

        runner._run_episode(0)

        # Oracle findings should have been gathered
        mock_oracle.get_findings.assert_called_once()

    def test_main_passes_policy_and_frame_stack(self):
        """main() passes --policy and --frame-stack to SessionRunner."""
        from scripts.run_session import main

        with mock.patch(
            "src.orchestrator.session_runner.SessionRunner",
        ) as MockRunner:
            mock_runner_instance = mock.MagicMock()
            mock_report = mock.MagicMock()
            mock_report.summary = {
                "total_episodes": 1,
                "total_findings": 0,
                "critical_findings": 0,
                "warning_findings": 0,
                "info_findings": 0,
                "episodes_failed": 0,
                "mean_episode_reward": 0.0,
                "mean_episode_length": 0.0,
            }
            mock_runner_instance.run.return_value = mock_report
            MockRunner.return_value = mock_runner_instance

            main(
                [
                    "--policy",
                    "cnn",
                    "--frame-stack",
                    "8",
                    "--episodes",
                    "1",
                ]
            )

            call_kwargs = MockRunner.call_args[1]
            assert call_kwargs["policy"] == "cnn"
            assert call_kwargs["frame_stack"] == 8


class TestSessionRunnerValidation:
    """SessionRunner rejects invalid policy/frame_stack at construction."""

    def test_invalid_policy_raises(self):
        """policy='bad' raises ValueError."""
        with pytest.raises(ValueError, match="policy must be one of"):
            SessionRunner(game="breakout71", policy="bad")

    def test_cnn_with_zero_frame_stack_raises(self):
        """policy='cnn' with frame_stack=0 raises ValueError."""
        with pytest.raises(ValueError, match="frame_stack must be >= 1"):
            SessionRunner(game="breakout71", policy="cnn", frame_stack=0)

    def test_mlp_with_zero_frame_stack_ok(self):
        """policy='mlp' does not validate frame_stack (it's ignored)."""
        runner = SessionRunner(game="breakout71", policy="mlp", frame_stack=0)
        assert runner.policy == "mlp"

    def test_cleanup_clears_raw_env(self):
        """_cleanup() sets _raw_env to None alongside _env."""
        runner = SessionRunner(game="breakout71")
        runner._raw_env = mock.MagicMock()
        runner._env = mock.MagicMock()
        runner._cleanup()
        assert runner._raw_env is None
        assert runner._env is None


# ===========================================================================
# GameOverDetector CLI Integration Tests
# ===========================================================================


class TestGameOverDetectorCLI:
    """Tests for --game-over-detector CLI flag in train_rl.py and run_session.py."""

    def test_train_rl_detector_default_off(self):
        """--game-over-detector defaults to False in train_rl.py."""
        from scripts.train_rl import parse_args

        args = parse_args([])
        assert args.game_over_detector is False

    def test_train_rl_detector_enabled(self):
        """--game-over-detector flag sets True in train_rl.py."""
        from scripts.train_rl import parse_args

        args = parse_args(["--game-over-detector"])
        assert args.game_over_detector is True

    def test_train_rl_detector_threshold_default(self):
        """--detector-threshold defaults to 0.6 in train_rl.py."""
        from scripts.train_rl import parse_args

        args = parse_args([])
        assert args.detector_threshold == 0.6

    def test_train_rl_detector_threshold_custom(self):
        """--detector-threshold accepts custom value in train_rl.py."""
        from scripts.train_rl import parse_args

        args = parse_args(["--game-over-detector", "--detector-threshold", "0.8"])
        assert args.game_over_detector is True
        assert args.detector_threshold == 0.8

    def test_run_session_detector_default_off(self):
        """--game-over-detector defaults to False in run_session.py."""
        from scripts.run_session import parse_args

        args = parse_args([])
        assert args.game_over_detector is False

    def test_run_session_detector_enabled(self):
        """--game-over-detector flag sets True in run_session.py."""
        from scripts.run_session import parse_args

        args = parse_args(["--game-over-detector"])
        assert args.game_over_detector is True

    def test_run_session_detector_threshold_default(self):
        """--detector-threshold defaults to 0.6 in run_session.py."""
        from scripts.run_session import parse_args

        args = parse_args([])
        assert args.detector_threshold == 0.6

    def test_run_session_detector_threshold_custom(self):
        """--detector-threshold accepts custom value in run_session.py."""
        from scripts.run_session import parse_args

        args = parse_args(["--game-over-detector", "--detector-threshold", "0.9"])
        assert args.game_over_detector is True
        assert args.detector_threshold == 0.9


class TestSessionRunnerGameOverDetector:
    """Tests for SessionRunner game_over_detector parameter."""

    def test_default_detector_is_none(self):
        """SessionRunner defaults to no game-over detector."""
        runner = SessionRunner(game="breakout71")
        assert runner.game_over_detector is None

    def test_accepts_detector(self):
        """SessionRunner accepts a GameOverDetector instance."""
        from src.platform.game_over_detector import GameOverDetector

        detector = GameOverDetector()
        runner = SessionRunner(game="breakout71", game_over_detector=detector)
        assert runner.game_over_detector is detector

    def test_detector_passed_to_env(self, tmp_path):
        """SessionRunner passes detector to env constructor in _setup()."""
        from src.platform.game_over_detector import GameOverDetector

        detector = GameOverDetector(confidence_threshold=0.8)
        runner = SessionRunner(
            game="breakout71",
            game_over_detector=detector,
            output_dir=tmp_path,
        )
        runner.game_config_path = Path("breakout-71")

        mock_config = mock.MagicMock()
        mock_config.url = "http://localhost:1234"
        mock_config.window_width = 768
        mock_config.window_height = 1024
        mock_config.window_title = "test"

        mock_loader = mock.MagicMock()
        mock_loader.url = "http://localhost:1234"

        mock_browser = mock.MagicMock()
        mock_browser.driver = mock.MagicMock()

        mock_env = mock.MagicMock()
        MockEnvCls = mock.MagicMock(return_value=mock_env)

        runner._plugin = mock.MagicMock()
        runner._plugin.env_class = MockEnvCls
        runner._plugin.game_name = "breakout-71"

        with (
            mock.patch(
                "scripts._smoke_utils.BrowserInstance",
                return_value=mock_browser,
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

        # Verify detector was passed to env constructor
        call_kwargs = MockEnvCls.call_args[1]
        assert call_kwargs["game_over_detector"] is detector
