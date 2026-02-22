"""Tests for DemoRecorder -- enriched per-step recording for human demos.

Covers:
- DemoRecorder construction and directory setup
- Per-step recording (frame, events, game state, reward, obs hash)
- JSONL serialization format
- Frame saving as PNG
- Episode lifecycle (start, record steps, end)
- Manifest generation with episode metadata
- Multiple episode support
- Edge cases (no frame, no events, empty game state)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# -- Helpers -------------------------------------------------------------------


def _fake_frame(h: int = 480, w: int = 640, value: int = 0) -> np.ndarray:
    """Create a synthetic BGR frame."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _fake_obs(dim: int = 8) -> np.ndarray:
    """Create a synthetic observation vector."""
    return np.random.default_rng(42).random(dim).astype(np.float32)


@pytest.fixture(autouse=True)
def _mock_cv2_imwrite(monkeypatch):
    """Mock cv2.imwrite so tests don't depend on libGL (CI Docker).

    Creates real 0-byte files so path-existence assertions still pass.
    """

    def _fake_imwrite(path: str, img, params=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"")
        return True

    mock_cv2 = mock.MagicMock()
    mock_cv2.imwrite = _fake_imwrite
    monkeypatch.setitem(sys.modules, "cv2", mock_cv2)


# ===========================================================================
# DemoRecorder Tests
# ===========================================================================


class TestDemoRecorderConstruction:
    """Test DemoRecorder initialization and directory setup."""

    def test_creates_output_directory(self, tmp_path):
        """DemoRecorder creates a timestamped output directory."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        assert recorder.output_dir.exists()
        assert recorder.output_dir.parent == tmp_path

    def test_creates_frames_subdirectory(self, tmp_path):
        """DemoRecorder creates a frames/ subdirectory."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        frames_dir = recorder.output_dir / "frames"
        assert frames_dir.exists()

    def test_initial_step_count_zero(self, tmp_path):
        """DemoRecorder starts with zero steps recorded."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        assert recorder.step_count == 0

    def test_initial_episode_count_zero(self, tmp_path):
        """DemoRecorder starts with zero episodes."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        assert recorder.episode_count == 0

    def test_custom_game_name(self, tmp_path):
        """DemoRecorder stores the game name."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path, game_name="hextris")
        assert recorder.game_name == "hextris"

    def test_default_game_name(self, tmp_path):
        """DemoRecorder defaults to 'unknown' game name."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        assert recorder.game_name == "unknown"

    def test_capture_interval_default(self, tmp_path):
        """DemoRecorder defaults to saving every frame."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        assert recorder.frame_capture_interval == 1

    def test_capture_interval_custom(self, tmp_path):
        """DemoRecorder accepts custom frame capture interval."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path, frame_capture_interval=5)
        assert recorder.frame_capture_interval == 5

    def test_capture_interval_minimum_one(self, tmp_path):
        """Frame capture interval is clamped to minimum of 1."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path, frame_capture_interval=0)
        assert recorder.frame_capture_interval == 1


class TestDemoRecorderEpisodeLifecycle:
    """Test episode start/end lifecycle."""

    def test_start_episode_increments_count(self, tmp_path):
        """start_episode() increments the episode counter."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        recorder.start_episode(episode_id=0)
        assert recorder.episode_count == 1

    def test_start_multiple_episodes(self, tmp_path):
        """Multiple start_episode() calls increment counter."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        recorder.start_episode(episode_id=0)
        recorder.end_episode(terminated=True, truncated=False)
        recorder.start_episode(episode_id=1)
        assert recorder.episode_count == 2

    def test_end_episode_without_start_raises(self, tmp_path):
        """end_episode() without start_episode() raises RuntimeError."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        with pytest.raises(RuntimeError, match="No episode in progress"):
            recorder.end_episode(terminated=True, truncated=False)

    def test_double_start_episode_raises(self, tmp_path):
        """start_episode() while episode in progress raises RuntimeError."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        recorder.start_episode(episode_id=0)
        with pytest.raises(RuntimeError, match="already in progress"):
            recorder.start_episode(episode_id=1)

    def test_end_episode_resets_in_progress(self, tmp_path):
        """After end_episode(), a new episode can be started."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        recorder.start_episode(episode_id=0)
        recorder.end_episode(terminated=True, truncated=False)
        # Should not raise
        recorder.start_episode(episode_id=1)


class TestDemoRecorderRecordStep:
    """Test per-step recording."""

    def test_record_step_increments_count(self, tmp_path):
        """record_step() increments the total step counter."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        recorder.start_episode(episode_id=0)
        recorder.record_step(
            step=0,
            frame=_fake_frame(),
            action=np.array([0.5]),
            reward=0.01,
            terminated=False,
            truncated=False,
            human_events=[],
            game_state={},
            observation=_fake_obs(),
        )
        assert recorder.step_count == 1

    def test_record_step_without_episode_raises(self, tmp_path):
        """record_step() without start_episode() raises RuntimeError."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        with pytest.raises(RuntimeError, match="No episode in progress"):
            recorder.record_step(
                step=0,
                frame=_fake_frame(),
                action=np.array([0.5]),
                reward=0.01,
                terminated=False,
                truncated=False,
            )

    def test_record_step_saves_frame_png(self, tmp_path):
        """record_step() saves the frame as a PNG file."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        recorder.start_episode(episode_id=0)
        recorder.record_step(
            step=0,
            frame=_fake_frame(),
            action=np.array([0.5]),
            reward=0.01,
            terminated=False,
            truncated=False,
        )
        frames_dir = recorder.output_dir / "frames"
        png_files = list(frames_dir.glob("*.png"))
        assert len(png_files) == 1

    def test_record_step_frame_interval_skips(self, tmp_path):
        """record_step() respects frame_capture_interval for PNG saving."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path, frame_capture_interval=3)
        recorder.start_episode(episode_id=0)
        for i in range(6):
            recorder.record_step(
                step=i,
                frame=_fake_frame(),
                action=np.array([0.5]),
                reward=0.01,
                terminated=False,
                truncated=False,
            )
        frames_dir = recorder.output_dir / "frames"
        png_files = list(frames_dir.glob("*.png"))
        assert len(png_files) == 2  # step 0 (1st call) and step 3 (4th call)

    def test_record_step_none_frame_no_png(self, tmp_path):
        """record_step() with frame=None does not save a PNG."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        recorder.start_episode(episode_id=0)
        recorder.record_step(
            step=0,
            frame=None,
            action=np.array([0.5]),
            reward=0.01,
            terminated=False,
            truncated=False,
        )
        frames_dir = recorder.output_dir / "frames"
        png_files = list(frames_dir.glob("*.png"))
        assert len(png_files) == 0

    def test_record_step_writes_jsonl(self, tmp_path):
        """record_step() appends a line to the JSONL file."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        recorder.start_episode(episode_id=0)
        recorder.record_step(
            step=0,
            frame=_fake_frame(),
            action=np.array([0.5]),
            reward=0.01,
            terminated=False,
            truncated=False,
        )

        jsonl_path = recorder.output_dir / "demo.jsonl"
        assert jsonl_path.exists()

        with open(jsonl_path) as f:
            lines = f.readlines()
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["step"] == 0
        assert record["episode_id"] == 0

    def test_record_step_jsonl_contains_action(self, tmp_path):
        """JSONL record includes the action as a list."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        recorder.start_episode(episode_id=0)
        recorder.record_step(
            step=0,
            frame=_fake_frame(),
            action=np.array([0.5, -0.3]),
            reward=0.01,
            terminated=False,
            truncated=False,
        )

        with open(recorder.output_dir / "demo.jsonl") as f:
            record = json.loads(f.readline())

        assert record["action"] == [0.5, -0.3]

    def test_record_step_jsonl_contains_scalar_action(self, tmp_path):
        """JSONL record handles scalar (Discrete) actions."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        recorder.start_episode(episode_id=0)
        recorder.record_step(
            step=0,
            frame=_fake_frame(),
            action=2,
            reward=0.01,
            terminated=False,
            truncated=False,
        )

        with open(recorder.output_dir / "demo.jsonl") as f:
            record = json.loads(f.readline())

        assert record["action"] == 2

    def test_record_step_jsonl_contains_reward(self, tmp_path):
        """JSONL record includes the reward."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        recorder.start_episode(episode_id=0)
        recorder.record_step(
            step=0,
            frame=_fake_frame(),
            action=np.array([0.0]),
            reward=1.23,
            terminated=False,
            truncated=False,
        )

        with open(recorder.output_dir / "demo.jsonl") as f:
            record = json.loads(f.readline())

        assert record["reward"] == pytest.approx(1.23)

    def test_record_step_jsonl_contains_terminated_truncated(self, tmp_path):
        """JSONL record includes terminated and truncated flags."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        recorder.start_episode(episode_id=0)
        recorder.record_step(
            step=5,
            frame=_fake_frame(),
            action=np.array([0.0]),
            reward=-5.0,
            terminated=True,
            truncated=False,
        )

        with open(recorder.output_dir / "demo.jsonl") as f:
            record = json.loads(f.readline())

        assert record["terminated"] is True
        assert record["truncated"] is False

    def test_record_step_jsonl_contains_human_events(self, tmp_path):
        """JSONL record includes human input events when provided."""
        from src.orchestrator.demo_recorder import DemoRecorder

        events = [
            {"type": "mousemove", "timestamp": 1000, "x": 100, "y": 200},
            {"type": "click", "timestamp": 1010, "x": 100, "y": 200, "button": 0},
        ]
        recorder = DemoRecorder(output_dir=tmp_path)
        recorder.start_episode(episode_id=0)
        recorder.record_step(
            step=0,
            frame=_fake_frame(),
            action=np.array([0.0]),
            reward=0.01,
            terminated=False,
            truncated=False,
            human_events=events,
        )

        with open(recorder.output_dir / "demo.jsonl") as f:
            record = json.loads(f.readline())

        assert record["human_events"] == events

    def test_record_step_jsonl_empty_events_default(self, tmp_path):
        """JSONL record has empty human_events by default."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        recorder.start_episode(episode_id=0)
        recorder.record_step(
            step=0,
            frame=_fake_frame(),
            action=np.array([0.0]),
            reward=0.01,
            terminated=False,
            truncated=False,
        )

        with open(recorder.output_dir / "demo.jsonl") as f:
            record = json.loads(f.readline())

        assert record["human_events"] == []

    def test_record_step_jsonl_contains_game_state(self, tmp_path):
        """JSONL record includes game state dict when provided."""
        from src.orchestrator.demo_recorder import DemoRecorder

        state = {"score": 100, "level": 2, "lives": 3}
        recorder = DemoRecorder(output_dir=tmp_path)
        recorder.start_episode(episode_id=0)
        recorder.record_step(
            step=0,
            frame=_fake_frame(),
            action=np.array([0.0]),
            reward=0.01,
            terminated=False,
            truncated=False,
            game_state=state,
        )

        with open(recorder.output_dir / "demo.jsonl") as f:
            record = json.loads(f.readline())

        assert record["game_state"] == state

    def test_record_step_jsonl_contains_observation_hash(self, tmp_path):
        """JSONL record includes a hash of the observation."""
        from src.orchestrator.demo_recorder import DemoRecorder

        obs = _fake_obs()
        recorder = DemoRecorder(output_dir=tmp_path)
        recorder.start_episode(episode_id=0)
        recorder.record_step(
            step=0,
            frame=_fake_frame(),
            action=np.array([0.0]),
            reward=0.01,
            terminated=False,
            truncated=False,
            observation=obs,
        )

        with open(recorder.output_dir / "demo.jsonl") as f:
            record = json.loads(f.readline())

        assert "obs_hash" in record
        expected = hashlib.md5(obs.tobytes()).hexdigest()
        assert record["obs_hash"] == expected

    def test_record_step_jsonl_no_observation_no_hash(self, tmp_path):
        """JSONL record has null obs_hash when no observation provided."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        recorder.start_episode(episode_id=0)
        recorder.record_step(
            step=0,
            frame=_fake_frame(),
            action=np.array([0.0]),
            reward=0.01,
            terminated=False,
            truncated=False,
        )

        with open(recorder.output_dir / "demo.jsonl") as f:
            record = json.loads(f.readline())

        assert record["obs_hash"] is None

    def test_record_step_jsonl_contains_frame_path(self, tmp_path):
        """JSONL record references the saved frame filename."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        recorder.start_episode(episode_id=0)
        recorder.record_step(
            step=0,
            frame=_fake_frame(),
            action=np.array([0.0]),
            reward=0.01,
            terminated=False,
            truncated=False,
        )

        with open(recorder.output_dir / "demo.jsonl") as f:
            record = json.loads(f.readline())

        assert record["frame_file"] is not None
        assert record["frame_file"].endswith(".png")

    def test_record_step_jsonl_no_frame_null_path(self, tmp_path):
        """JSONL record has null frame_file when frame is None."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        recorder.start_episode(episode_id=0)
        recorder.record_step(
            step=0,
            frame=None,
            action=np.array([0.0]),
            reward=0.01,
            terminated=False,
            truncated=False,
        )

        with open(recorder.output_dir / "demo.jsonl") as f:
            record = json.loads(f.readline())

        assert record["frame_file"] is None

    def test_record_step_jsonl_contains_timestamp(self, tmp_path):
        """JSONL record includes an ISO timestamp."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        recorder.start_episode(episode_id=0)
        recorder.record_step(
            step=0,
            frame=_fake_frame(),
            action=np.array([0.0]),
            reward=0.01,
            terminated=False,
            truncated=False,
        )

        with open(recorder.output_dir / "demo.jsonl") as f:
            record = json.loads(f.readline())

        assert "timestamp" in record

    def test_record_step_jsonl_contains_oracle_findings(self, tmp_path):
        """JSONL record includes oracle findings when provided."""
        from src.orchestrator.demo_recorder import DemoRecorder

        findings = [
            {"oracle_name": "StuckOracle", "severity": "warning", "step": 0, "description": "test"}
        ]
        recorder = DemoRecorder(output_dir=tmp_path)
        recorder.start_episode(episode_id=0)
        recorder.record_step(
            step=0,
            frame=_fake_frame(),
            action=np.array([0.0]),
            reward=0.01,
            terminated=False,
            truncated=False,
            oracle_findings=findings,
        )

        with open(recorder.output_dir / "demo.jsonl") as f:
            record = json.loads(f.readline())

        assert record["oracle_findings"] == findings

    def test_record_step_jsonl_empty_findings_default(self, tmp_path):
        """JSONL record has empty oracle_findings by default."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        recorder.start_episode(episode_id=0)
        recorder.record_step(
            step=0,
            frame=_fake_frame(),
            action=np.array([0.0]),
            reward=0.01,
            terminated=False,
            truncated=False,
        )

        with open(recorder.output_dir / "demo.jsonl") as f:
            record = json.loads(f.readline())

        assert record["oracle_findings"] == []

    def test_multiple_steps_multiple_jsonl_lines(self, tmp_path):
        """Multiple record_step() calls produce multiple JSONL lines."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        recorder.start_episode(episode_id=0)
        for i in range(5):
            recorder.record_step(
                step=i,
                frame=_fake_frame(),
                action=np.array([float(i)]),
                reward=0.01,
                terminated=i == 4,
                truncated=False,
            )

        with open(recorder.output_dir / "demo.jsonl") as f:
            lines = f.readlines()
        assert len(lines) == 5

        # Verify sequential steps
        for i, line in enumerate(lines):
            record = json.loads(line)
            assert record["step"] == i


class TestDemoRecorderMultipleEpisodes:
    """Test recording across multiple episodes."""

    def test_records_across_episodes(self, tmp_path):
        """Records from multiple episodes all go to the same JSONL."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)

        # Episode 0
        recorder.start_episode(episode_id=0)
        for i in range(3):
            recorder.record_step(
                step=i,
                frame=_fake_frame(),
                action=np.array([0.0]),
                reward=0.01,
                terminated=i == 2,
                truncated=False,
            )
        recorder.end_episode(terminated=True, truncated=False)

        # Episode 1
        recorder.start_episode(episode_id=1)
        for i in range(2):
            recorder.record_step(
                step=i,
                frame=_fake_frame(),
                action=np.array([0.0]),
                reward=0.01,
                terminated=i == 1,
                truncated=False,
            )
        recorder.end_episode(terminated=True, truncated=False)

        with open(recorder.output_dir / "demo.jsonl") as f:
            lines = f.readlines()
        assert len(lines) == 5
        assert recorder.step_count == 5

        # Verify episode IDs
        ep_ids = [json.loads(line)["episode_id"] for line in lines]
        assert ep_ids == [0, 0, 0, 1, 1]

    def test_total_step_count_across_episodes(self, tmp_path):
        """step_count accumulates across episodes."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)

        recorder.start_episode(episode_id=0)
        recorder.record_step(
            step=0,
            frame=_fake_frame(),
            action=np.array([0.0]),
            reward=0.01,
            terminated=True,
            truncated=False,
        )
        recorder.end_episode(terminated=True, truncated=False)

        recorder.start_episode(episode_id=1)
        recorder.record_step(
            step=0,
            frame=_fake_frame(),
            action=np.array([0.0]),
            reward=0.01,
            terminated=True,
            truncated=False,
        )
        recorder.end_episode(terminated=True, truncated=False)

        assert recorder.step_count == 2


class TestDemoRecorderFinalize:
    """Test manifest generation on finalize()."""

    def test_finalize_creates_manifest(self, tmp_path):
        """finalize() writes a manifest.json file."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path, game_name="breakout71")
        recorder.start_episode(episode_id=0)
        recorder.record_step(
            step=0,
            frame=_fake_frame(),
            action=np.array([0.0]),
            reward=0.01,
            terminated=True,
            truncated=False,
        )
        recorder.end_episode(terminated=True, truncated=False)

        output_dir = recorder.finalize()
        manifest_path = output_dir / "manifest.json"
        assert manifest_path.exists()

    def test_finalize_manifest_contains_metadata(self, tmp_path):
        """Manifest contains game_name, total_steps, total_episodes."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path, game_name="hextris")
        recorder.start_episode(episode_id=0)
        for i in range(3):
            recorder.record_step(
                step=i,
                frame=_fake_frame(),
                action=np.array([0.0]),
                reward=0.01,
                terminated=i == 2,
                truncated=False,
            )
        recorder.end_episode(terminated=True, truncated=False)

        output_dir = recorder.finalize()
        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)

        assert manifest["game_name"] == "hextris"
        assert manifest["total_steps"] == 3
        assert manifest["total_episodes"] == 1

    def test_finalize_manifest_contains_episodes(self, tmp_path):
        """Manifest contains per-episode metadata."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path, game_name="breakout71")

        recorder.start_episode(episode_id=0)
        for i in range(5):
            recorder.record_step(
                step=i,
                frame=_fake_frame(),
                action=np.array([0.0]),
                reward=0.01 if i < 4 else -5.0,
                terminated=i == 4,
                truncated=False,
            )
        recorder.end_episode(terminated=True, truncated=False)

        recorder.start_episode(episode_id=1)
        for i in range(3):
            recorder.record_step(
                step=i,
                frame=_fake_frame(),
                action=np.array([0.0]),
                reward=0.01,
                terminated=False,
                truncated=i == 2,
            )
        recorder.end_episode(terminated=False, truncated=True)

        output_dir = recorder.finalize()
        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)

        assert len(manifest["episodes"]) == 2
        ep0 = manifest["episodes"][0]
        assert ep0["episode_id"] == 0
        assert ep0["steps"] == 5
        assert ep0["terminated"] is True
        assert ep0["truncated"] is False

        ep1 = manifest["episodes"][1]
        assert ep1["episode_id"] == 1
        assert ep1["steps"] == 3
        assert ep1["terminated"] is False
        assert ep1["truncated"] is True

    def test_finalize_manifest_contains_total_reward(self, tmp_path):
        """Manifest episode entries include total_reward."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        recorder.start_episode(episode_id=0)
        recorder.record_step(
            step=0,
            frame=_fake_frame(),
            action=np.array([0.0]),
            reward=1.5,
            terminated=False,
            truncated=False,
        )
        recorder.record_step(
            step=1,
            frame=_fake_frame(),
            action=np.array([0.0]),
            reward=2.5,
            terminated=True,
            truncated=False,
        )
        recorder.end_episode(terminated=True, truncated=False)
        output_dir = recorder.finalize()

        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)

        assert manifest["episodes"][0]["total_reward"] == pytest.approx(4.0)

    def test_finalize_returns_output_dir(self, tmp_path):
        """finalize() returns the output directory Path."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        recorder.start_episode(episode_id=0)
        recorder.record_step(
            step=0,
            frame=_fake_frame(),
            action=np.array([0.0]),
            reward=0.01,
            terminated=True,
            truncated=False,
        )
        recorder.end_episode(terminated=True, truncated=False)

        result = recorder.finalize()
        assert isinstance(result, Path)
        assert result == recorder.output_dir

    def test_finalize_manifest_contains_frame_capture_interval(self, tmp_path):
        """Manifest includes the frame_capture_interval setting."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path, frame_capture_interval=5)
        recorder.start_episode(episode_id=0)
        recorder.record_step(
            step=0,
            frame=_fake_frame(),
            action=np.array([0.0]),
            reward=0.01,
            terminated=True,
            truncated=False,
        )
        recorder.end_episode(terminated=True, truncated=False)
        output_dir = recorder.finalize()

        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)

        assert manifest["frame_capture_interval"] == 5


class TestDemoRecorderActionSerialization:
    """Test action serialization for different action space types."""

    def test_box_action_serialized_as_list(self, tmp_path):
        """Box action (numpy array) is serialized as a list of floats."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        recorder.start_episode(episode_id=0)
        recorder.record_step(
            step=0,
            frame=_fake_frame(),
            action=np.array([0.5, -0.3], dtype=np.float32),
            reward=0.01,
            terminated=False,
            truncated=False,
        )

        with open(recorder.output_dir / "demo.jsonl") as f:
            record = json.loads(f.readline())

        assert isinstance(record["action"], list)
        assert len(record["action"]) == 2

    def test_discrete_action_serialized_as_int(self, tmp_path):
        """Discrete action (int/np.int64) is serialized as int."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        recorder.start_episode(episode_id=0)
        recorder.record_step(
            step=0,
            frame=_fake_frame(),
            action=np.int64(2),
            reward=0.01,
            terminated=False,
            truncated=False,
        )

        with open(recorder.output_dir / "demo.jsonl") as f:
            record = json.loads(f.readline())

        assert isinstance(record["action"], int)
        assert record["action"] == 2

    def test_multidiscrete_action_serialized_as_list(self, tmp_path):
        """MultiDiscrete action (numpy array of ints) is serialized as list."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        recorder.start_episode(episode_id=0)
        recorder.record_step(
            step=0,
            frame=_fake_frame(),
            action=np.array([1, 3, 8, 8, 2], dtype=np.int64),
            reward=0.01,
            terminated=False,
            truncated=False,
        )

        with open(recorder.output_dir / "demo.jsonl") as f:
            record = json.loads(f.readline())

        assert isinstance(record["action"], list)
        assert record["action"] == [1, 3, 8, 8, 2]


class TestDemoRecorderFrameNaming:
    """Test frame file naming convention."""

    def test_frame_filename_includes_episode_and_step(self, tmp_path):
        """Frame PNG filename includes episode and step for uniqueness."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)
        recorder.start_episode(episode_id=0)
        recorder.record_step(
            step=5,
            frame=_fake_frame(),
            action=np.array([0.0]),
            reward=0.01,
            terminated=False,
            truncated=False,
        )

        with open(recorder.output_dir / "demo.jsonl") as f:
            record = json.loads(f.readline())

        # Frame file should contain episode and step info
        frame_file = record["frame_file"]
        assert "ep000" in frame_file
        assert "step00005" in frame_file

    def test_frame_files_unique_across_episodes(self, tmp_path):
        """Frame files from different episodes have unique names."""
        from src.orchestrator.demo_recorder import DemoRecorder

        recorder = DemoRecorder(output_dir=tmp_path)

        recorder.start_episode(episode_id=0)
        recorder.record_step(
            step=0,
            frame=_fake_frame(),
            action=np.array([0.0]),
            reward=0.01,
            terminated=True,
            truncated=False,
        )
        recorder.end_episode(terminated=True, truncated=False)

        recorder.start_episode(episode_id=1)
        recorder.record_step(
            step=0,
            frame=_fake_frame(),
            action=np.array([0.0]),
            reward=0.01,
            terminated=True,
            truncated=False,
        )
        recorder.end_episode(terminated=True, truncated=False)

        frames_dir = recorder.output_dir / "frames"
        png_files = list(frames_dir.glob("*.png"))
        assert len(png_files) == 2
        assert png_files[0].name != png_files[1].name
