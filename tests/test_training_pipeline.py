"""Tests for the YOLO training pipeline scripts.

Covers:
- Training config loading (load_training_config)
- Device resolution (resolve_device)
- Training function validation (dataset_path checks)
- Capture dataset manifest format
- Roboflow upload state management (resume support)
- Validation thresholds
"""

import argparse
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_model import (
    _TRAINING_CONFIG_DIR,
    load_training_config,
    resolve_device,
    train,
)
from scripts.upload_to_roboflow import (
    _build_labelmap,
    _load_upload_state,
    _save_upload_state,
)
from scripts.validate_model import validate_model


# ---------------------------------------------------------------------------
# Training Config Loading
# ---------------------------------------------------------------------------


class TestLoadTrainingConfig:
    """Tests for load_training_config."""

    def test_loads_breakout71_config(self):
        """Loading 'breakout-71' returns a dict with expected keys."""
        cfg = load_training_config("breakout-71")
        assert isinstance(cfg, dict)
        assert cfg["game"] == "breakout-71"
        assert cfg["classes"] == ["paddle", "ball", "brick", "powerup", "wall"]
        assert cfg["base_model"] == "yolov8n.pt"
        assert cfg["epochs"] == 100
        assert cfg["imgsz"] == 640

    def test_config_has_device_field(self):
        """Config includes a device field."""
        cfg = load_training_config("breakout-71")
        assert "device" in cfg
        assert cfg["device"] == "auto"

    def test_config_has_validation_thresholds(self):
        """Config includes mAP validation thresholds."""
        cfg = load_training_config("breakout-71")
        # Lowered from 0.80 to 0.65 for initial auto-annotated dataset
        assert cfg["min_map50"] == 0.65
        assert cfg["min_map50_95"] == 0.50

    def test_config_has_output_dir(self):
        """Config specifies an output directory for weights."""
        cfg = load_training_config("breakout-71")
        assert cfg["output_dir"] == "weights/breakout71"

    def test_config_has_batch_size(self):
        """Config specifies batch size."""
        cfg = load_training_config("breakout-71")
        assert cfg["batch"] == 16

    def test_config_has_amp_setting(self):
        """Config specifies AMP (automatic mixed precision) setting."""
        cfg = load_training_config("breakout-71")
        assert cfg["amp"] is False

    def test_config_dataset_path_initially_null(self):
        """Dataset path is null until user sets it after prepare_dataset.py."""
        cfg = load_training_config("breakout-71")
        assert cfg["dataset_path"] is None

    def test_missing_config_raises(self):
        """Loading a nonexistent config raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Training config not found"):
            load_training_config("nonexistent-game-xyz")

    def test_config_dir_exists(self):
        """The training config directory exists."""
        assert _TRAINING_CONFIG_DIR.is_dir()

    def test_breakout71_yaml_exists(self):
        """The breakout-71.yaml config file exists."""
        config_path = _TRAINING_CONFIG_DIR / "breakout-71.yaml"
        assert config_path.is_file()

    def test_config_classes_count(self):
        """Breakout 71 has exactly 5 YOLO classes."""
        cfg = load_training_config("breakout-71")
        assert len(cfg["classes"]) == 5

    def test_config_game_links_to_game_config(self):
        """The 'game' field matches a configs/games/*.yaml file."""
        cfg = load_training_config("breakout-71")
        game_config_path = (
            Path(__file__).resolve().parent.parent
            / "configs"
            / "games"
            / f"{cfg['game']}.yaml"
        )
        assert game_config_path.is_file()


# ---------------------------------------------------------------------------
# Device Resolution
# ---------------------------------------------------------------------------


class TestResolveDevice:
    """Tests for resolve_device."""

    def test_explicit_cpu(self):
        """Requesting 'cpu' returns 'cpu' without importing torch."""
        # torch import is now after the early return, so no mock needed
        assert resolve_device("cpu") == "cpu"

    def test_explicit_cuda(self):
        """Requesting 'cuda' returns 'cuda' without importing torch."""
        assert resolve_device("cuda") == "cuda"

    def test_explicit_xpu(self):
        """Requesting 'xpu' returns 'xpu' without importing torch."""
        assert resolve_device("xpu") == "xpu"

    def test_auto_falls_back_to_cpu(self):
        """Auto mode falls back to CPU when no GPU is available."""
        mock_torch = mock.MagicMock()
        mock_torch.xpu = mock.MagicMock()
        mock_torch.xpu.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False
        with mock.patch.dict("sys.modules", {"torch": mock_torch}):
            result = resolve_device("auto")
            assert result == "cpu"

    def test_auto_prefers_xpu(self):
        """Auto mode prefers XPU when available."""
        mock_torch = mock.MagicMock()
        mock_torch.xpu = mock.MagicMock()
        mock_torch.xpu.is_available.return_value = True
        with mock.patch.dict("sys.modules", {"torch": mock_torch}):
            result = resolve_device("auto")
            assert result == "xpu"

    def test_auto_falls_back_to_cuda(self):
        """Auto mode uses CUDA when XPU unavailable but CUDA available."""
        mock_torch = mock.MagicMock(spec=[])
        mock_torch.cuda = mock.MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with mock.patch.dict("sys.modules", {"torch": mock_torch}):
            result = resolve_device("auto")
            assert result == "cuda"


# ---------------------------------------------------------------------------
# Training Function Validation
# ---------------------------------------------------------------------------


class TestTrainValidation:
    """Tests for train() input validation."""

    def test_raises_without_dataset_path(self):
        """train() raises RuntimeError when dataset_path is null."""
        cfg = load_training_config("breakout-71")
        assert cfg["dataset_path"] is None

        with pytest.raises(RuntimeError, match="dataset_path is not set"):
            train(cfg)

    def test_raises_with_nonexistent_dataset_path(self):
        """train() raises ValueError for a missing dataset path."""
        cfg = load_training_config("breakout-71")
        cfg["dataset_path"] = "/nonexistent/path/data.yaml"

        with pytest.raises(ValueError, match="does not exist"):
            train(cfg)

    def test_overrides_merge(self):
        """Overrides are merged into the config."""
        cfg = load_training_config("breakout-71")
        cfg["dataset_path"] = "/nonexistent/path/data.yaml"

        # Override dataset_path — should fail at dataset path validation,
        # but with the overridden path in the error message
        overrides = {"dataset_path": "/other/nonexistent/data.yaml"}
        with pytest.raises(ValueError, match="does not exist"):
            train(cfg, overrides=overrides)

    def test_none_overrides_ignored(self):
        """None values in overrides dict are ignored."""
        cfg = load_training_config("breakout-71")
        overrides = {"epochs": None, "device": None, "dataset_path": None}

        # Should still fail because dataset_path remains None from config
        with pytest.raises(RuntimeError, match="dataset_path is not set"):
            train(cfg, overrides=overrides)


# ---------------------------------------------------------------------------
# Validation Function
# ---------------------------------------------------------------------------


class TestValidateModel:
    """Tests for validate_model input validation."""

    def test_raises_without_weights(self, tmp_path):
        """validate_model raises RuntimeError when weights not found."""
        cfg = load_training_config("breakout-71")
        # Point output_dir to an empty tmp dir so default weights path resolves
        # to a nonexistent file — test must not depend on local filesystem state.
        cfg["output_dir"] = str(tmp_path / "no_weights_here")

        with pytest.raises(RuntimeError, match="Weights file not found"):
            validate_model(cfg)

    def test_raises_with_explicit_missing_weights(self):
        """validate_model raises RuntimeError for explicitly missing weights."""
        cfg = load_training_config("breakout-71")

        with pytest.raises(RuntimeError, match="Weights file not found"):
            validate_model(cfg, weights_path=Path("/nonexistent/best.pt"))

    def test_raises_without_dataset_path(self, tmp_path):
        """validate_model raises RuntimeError when dataset_path not set."""
        cfg = load_training_config("breakout-71")

        # Create a fake weights file so we get past that check
        fake_weights = tmp_path / "best.pt"
        fake_weights.write_bytes(b"fake")

        with pytest.raises(RuntimeError, match="dataset_path is not set"):
            validate_model(cfg, weights_path=fake_weights)


# ---------------------------------------------------------------------------
# Upload State Management
# ---------------------------------------------------------------------------


class TestUploadState:
    """Tests for Roboflow upload state (resume support)."""

    def test_load_empty_state(self, tmp_path):
        """Loading from nonexistent file returns empty set."""
        state_path = tmp_path / ".upload_state.json"
        result = _load_upload_state(state_path)
        assert result == set()

    def test_save_and_load_state(self, tmp_path):
        """Saved state can be loaded back."""
        state_path = tmp_path / ".upload_state.json"
        uploaded = {"frame_00001.png", "frame_00002.png", "frame_00003.png"}

        _save_upload_state(state_path, uploaded)
        result = _load_upload_state(state_path)

        assert result == uploaded

    def test_state_file_is_valid_json(self, tmp_path):
        """State file is valid JSON with sorted filenames."""
        state_path = tmp_path / ".upload_state.json"
        uploaded = {"frame_00003.png", "frame_00001.png"}

        _save_upload_state(state_path, uploaded)

        with open(state_path) as f:
            data = json.load(f)

        assert data["uploaded"] == ["frame_00001.png", "frame_00003.png"]

    def test_incremental_state_update(self, tmp_path):
        """State can be incrementally updated."""
        state_path = tmp_path / ".upload_state.json"

        # First save
        uploaded = {"frame_00001.png"}
        _save_upload_state(state_path, uploaded)

        # Load and add more
        uploaded = _load_upload_state(state_path)
        uploaded.add("frame_00002.png")
        _save_upload_state(state_path, uploaded)

        result = _load_upload_state(state_path)
        assert result == {"frame_00001.png", "frame_00002.png"}

    def test_corrupted_state_file_returns_empty(self, tmp_path):
        """Corrupted state file gracefully returns empty set."""
        state_path = tmp_path / ".upload_state.json"
        state_path.write_text("not valid json{{{")

        result = _load_upload_state(state_path)
        assert result == set()


# ---------------------------------------------------------------------------
# Manifest Format
# ---------------------------------------------------------------------------


class TestManifestFormat:
    """Tests for the dataset capture manifest format."""

    def test_manifest_schema(self, tmp_path):
        """A manifest.json follows the expected schema."""
        manifest_data = {
            "dataset": "breakout71",
            "capture_timestamp": "20260215_120000",
            "total_frames": 3,
            "interval_seconds": 0.2,
            "action_interval_frames": 5,
            "window_size": [1280, 720],
            "classes": ["paddle", "ball", "brick", "powerup", "wall"],
            "frames": [
                {
                    "index": 0,
                    "filename": "frame_00000.png",
                    "timestamp": 1739000000.0,
                    "shape": [720, 1280, 3],
                    "capture_ms": 12.5,
                },
                {
                    "index": 1,
                    "filename": "frame_00001.png",
                    "timestamp": 1739000000.2,
                    "shape": [720, 1280, 3],
                    "capture_ms": 11.3,
                    "action": {"type": "mouse_move", "x_norm": 0.7234},
                },
                {
                    "index": 2,
                    "filename": "frame_00002.png",
                    "timestamp": 1739000000.4,
                    "shape": [720, 1280, 3],
                    "capture_ms": 13.1,
                },
            ],
        }

        # Verify required top-level keys
        assert "dataset" in manifest_data
        assert "classes" in manifest_data
        assert "frames" in manifest_data
        assert "total_frames" in manifest_data
        assert manifest_data["total_frames"] == len(manifest_data["frames"])

        # Verify frame entries
        for frame in manifest_data["frames"]:
            assert "index" in frame
            assert "filename" in frame
            assert "timestamp" in frame
            assert "shape" in frame
            assert "capture_ms" in frame
            assert frame["filename"].startswith("frame_")
            assert frame["filename"].endswith(".png")

    def test_manifest_classes_match_config(self):
        """Manifest classes should match training config classes."""
        cfg = load_training_config("breakout-71")
        expected_classes = ["paddle", "ball", "brick", "powerup", "wall"]
        assert cfg["classes"] == expected_classes

    def test_manifest_round_trip(self, tmp_path):
        """Manifest can be serialized and deserialized."""
        manifest = {
            "dataset": "breakout71",
            "capture_timestamp": "20260215_120000",
            "total_frames": 1,
            "interval_seconds": 0.2,
            "action_interval_frames": 5,
            "window_size": [1280, 720],
            "classes": ["paddle", "ball", "brick", "powerup", "wall"],
            "frames": [
                {
                    "index": 0,
                    "filename": "frame_00000.png",
                    "timestamp": 1739000000.0,
                    "shape": [720, 1280, 3],
                    "capture_ms": 12.5,
                }
            ],
        }

        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        with open(manifest_path) as f:
            loaded = json.load(f)

        assert loaded == manifest


# ---------------------------------------------------------------------------
# .env.example
# ---------------------------------------------------------------------------


class TestDotEnv:
    """Tests for .env.example template."""

    def test_env_example_exists(self):
        """The .env.example template file exists."""
        env_example = Path(__file__).resolve().parent.parent / ".env.example"
        assert env_example.is_file()

    def test_env_example_has_roboflow_keys(self):
        """The .env.example contains Roboflow configuration keys."""
        env_example = Path(__file__).resolve().parent.parent / ".env.example"
        content = env_example.read_text()
        assert "ROBOFLOW_API_KEY" in content
        assert "ROBOFLOW_WORKSPACE" in content
        assert "ROBOFLOW_PROJECT" in content

    def test_env_is_gitignored(self):
        """The .env file is listed in .gitignore."""
        gitignore = Path(__file__).resolve().parent.parent / ".gitignore"
        content = gitignore.read_text()
        # Check for .env as a standalone line (not .env.example)
        lines = [line.strip() for line in content.splitlines()]
        assert ".env" in lines


# ---------------------------------------------------------------------------
# Pipeline Integration (end-to-end config validation)
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    """Integration tests for the training pipeline configuration."""

    def test_training_config_links_to_valid_game(self):
        """Training config's game field matches an existing game config."""
        cfg = load_training_config("breakout-71")
        game_config = (
            Path(__file__).resolve().parent.parent
            / "configs"
            / "games"
            / f"{cfg['game']}.yaml"
        )
        assert game_config.is_file()

    def test_classes_match_yolo_detector_defaults(self):
        """Training config classes match YoloDetector.BREAKOUT71_CLASSES."""
        from src.perception.yolo_detector import YoloDetector

        cfg = load_training_config("breakout-71")
        assert cfg["classes"] == YoloDetector.BREAKOUT71_CLASSES

    def test_output_dir_is_gitignored(self):
        """Weights output directory pattern is gitignored (*.pt rule)."""
        gitignore = Path(__file__).resolve().parent.parent / ".gitignore"
        content = gitignore.read_text()
        # *.pt files are gitignored, preventing accidental weight commits
        assert "*.pt" in content

    def test_all_scripts_exist(self):
        """All pipeline scripts exist."""
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        expected = [
            "capture_dataset.py",
            "upload_to_roboflow.py",
            "train_model.py",
            "validate_model.py",
        ]
        for script in expected:
            assert (scripts_dir / script).is_file(), f"Missing: {script}"

    def test_training_config_reasonable_defaults(self):
        """Training config has reasonable default hyperparameters."""
        cfg = load_training_config("breakout-71")

        # Epochs: 50-200 is reasonable for fine-tuning
        assert 50 <= cfg["epochs"] <= 200

        # Image size: 640 is standard YOLOv8
        assert cfg["imgsz"] == 640

        # Batch size: 8-64 is reasonable
        assert 8 <= cfg["batch"] <= 64

        # mAP thresholds: must be in (0, 1)
        assert 0 < cfg["min_map50"] < 1
        assert 0 < cfg["min_map50_95"] < 1

        # mAP50 threshold should be >= mAP50-95 threshold
        assert cfg["min_map50"] >= cfg["min_map50_95"]


# ---------------------------------------------------------------------------
# Ball Head Detection (_find_ball_head)
# ---------------------------------------------------------------------------


class TestFindBallHead:
    """Tests for _find_ball_head in auto_annotate."""

    @pytest.fixture(autouse=True)
    def _import_deps(self):
        """Lazily import cv2/numpy — only available in yolo env."""
        np = pytest.importorskip("numpy")
        cv2 = pytest.importorskip("cv2")
        from scripts.auto_annotate import _find_ball_head

        self.np = np
        self.cv2 = cv2
        self._find_ball_head = _find_ball_head

    def test_picks_circular_blob_over_elongated(self):
        """Ball head (circular) should win over an elongated trail blob."""
        roi = self.np.zeros((80, 80), dtype=self.np.uint8)
        # Draw a circle (the ball) — radius 8
        self.cv2.circle(roi, (20, 40), 8, 255, -1)
        # Draw an elongated rectangle (the trail) — 40x6
        self.cv2.rectangle(roi, (35, 37), (75, 43), 255, -1)

        cx, cy = self._find_ball_head(roi)
        # Center should be near the circle (20, 40), not near the rectangle center (55, 40)
        assert abs(cx - 20) < 5, f"Expected cx near 20, got {cx}"
        assert abs(cy - 40) < 5, f"Expected cy near 40, got {cy}"

    def test_single_blob_returns_its_center(self):
        """A single circular blob should return its center."""
        roi = self.np.zeros((50, 50), dtype=self.np.uint8)
        self.cv2.circle(roi, (25, 25), 10, 255, -1)

        cx, cy = self._find_ball_head(roi)
        assert abs(cx - 25) < 5
        assert abs(cy - 25) < 5

    def test_empty_roi_returns_center(self):
        """An empty ROI should fall back to the ROI center."""
        roi = self.np.zeros((60, 40), dtype=self.np.uint8)

        cx, cy = self._find_ball_head(roi)
        assert cx == 20.0  # w/2
        assert cy == 30.0  # h/2

    def test_ignores_noise_pixels(self):
        """Very small contours (< 3px area) should be ignored."""
        roi = self.np.zeros((50, 50), dtype=self.np.uint8)
        # Single noise pixel
        roi[10, 10] = 255
        # Actual ball
        self.cv2.circle(roi, (30, 30), 7, 255, -1)

        cx, cy = self._find_ball_head(roi)
        assert abs(cx - 30) < 5
        assert abs(cy - 30) < 5


# ---------------------------------------------------------------------------
# Labelmap Builder (_build_labelmap)
# ---------------------------------------------------------------------------


class TestBuildLabelmap:
    """Tests for _build_labelmap in upload_to_roboflow."""

    def test_reads_classes_txt(self, tmp_path):
        """Prefers classes.txt when present in labels/ dir."""
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "classes.txt").write_text("paddle\nball\nbrick\npowerup\nwall\n")

        result = _build_labelmap(tmp_path)
        assert result == {0: "paddle", 1: "ball", 2: "brick", 3: "powerup", 4: "wall"}

    def test_falls_back_to_training_config(self, tmp_path):
        """Falls back to training config YAML when no classes.txt exists."""
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        # No classes.txt — should fall back to configs/training/breakout-71.yaml

        result = _build_labelmap(tmp_path)
        # The YAML config has the 5 classes
        assert result is not None
        assert result[0] == "paddle"
        assert result[1] == "ball"
        assert len(result) == 5

    def test_returns_none_when_no_source(self, tmp_path, monkeypatch):
        """Returns None when neither classes.txt nor config exists."""
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()

        # Monkeypatch __file__ in the module to point to a nonexistent dir
        # so the YAML fallback also fails
        import scripts.upload_to_roboflow as upload_mod

        monkeypatch.setattr(upload_mod, "__file__", str(tmp_path / "fake.py"))

        result = _build_labelmap(tmp_path)
        assert result is None

    def test_ignores_blank_lines_in_classes_txt(self, tmp_path):
        """Blank lines in classes.txt should be skipped."""
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "classes.txt").write_text("paddle\n\nball\n\n\nbrick\n")

        result = _build_labelmap(tmp_path)
        assert result == {0: "paddle", 1: "ball", 2: "brick"}


# ---------------------------------------------------------------------------
# train_rl.py — parse_args, resolve_window_size, TrainingLogger
# ---------------------------------------------------------------------------


class TestTrainRlParseArgs:
    """Tests for train_rl.py argument parsing."""

    def _parse(self, argv=None):
        from scripts.train_rl import parse_args

        return parse_args(argv or [])

    def test_defaults(self):
        """Default args should have sensible values."""
        args = self._parse([])
        assert args.timesteps == 200_000
        assert args.mute is True
        assert args.headless is False
        assert args.orientation == "portrait"
        assert args.window_size is None
        assert args.log_interval == 100
        assert args.max_time is None

    def test_mute_default_true(self):
        """--mute should be the default (True)."""
        args = self._parse([])
        assert args.mute is True

    def test_no_mute_flag(self):
        """--no-mute should set mute=False."""
        args = self._parse(["--no-mute"])
        assert args.mute is False

    def test_headless_flag(self):
        """--headless should set headless=True."""
        args = self._parse(["--headless"])
        assert args.headless is True

    def test_orientation_portrait(self):
        """--orientation portrait should be accepted."""
        args = self._parse(["--orientation", "portrait"])
        assert args.orientation == "portrait"

    def test_landscape_shorthand(self):
        """--landscape should set orientation=landscape."""
        args = self._parse(["--landscape"])
        assert args.orientation == "landscape"

    def test_window_size_override(self):
        """--window-size should be stored as a string."""
        args = self._parse(["--window-size", "800x600"])
        assert args.window_size == "800x600"

    def test_log_interval(self):
        """--log-interval should be stored as int."""
        args = self._parse(["--log-interval", "50"])
        assert args.log_interval == 50

    def test_max_time_flag(self):
        """--max-time should be stored as int."""
        args = self._parse(["--max-time", "180"])
        assert args.max_time == 180

    def test_max_time_default_none(self):
        """--max-time should default to None."""
        args = self._parse([])
        assert args.max_time is None


class TestResolveWindowSize:
    """Tests for resolve_window_size in train_rl.py."""

    def _resolve(self, argv=None, config=None):
        from scripts.train_rl import parse_args, resolve_window_size

        args = parse_args(argv or [])
        if config is None:
            config = mock.MagicMock()
            config.window_width = 1280
            config.window_height = 1024
        return resolve_window_size(args, config)

    def test_portrait_default(self):
        """Default orientation=portrait should give 768x1024."""
        w, h = self._resolve([])
        assert (w, h) == (768, 1024)

    def test_landscape_shorthand(self):
        """--landscape shorthand should give 1280x1024."""
        w, h = self._resolve(["--landscape"])
        assert (w, h) == (1280, 1024)

    def test_portrait_preset(self):
        """--orientation portrait should give 768x1024."""
        w, h = self._resolve(["--orientation", "portrait"])
        assert (w, h) == (768, 1024)

    def test_window_size_overrides_orientation(self):
        """--window-size should override --orientation."""
        w, h = self._resolve(["--orientation", "portrait", "--window-size", "800x600"])
        assert (w, h) == (800, 600)

    def test_window_size_parsing(self):
        """--window-size WxH should parse correctly."""
        w, h = self._resolve(["--window-size", "1920x1080"])
        assert (w, h) == (1920, 1080)

    def test_invalid_window_size_format(self):
        """Invalid --window-size format should raise ValueError."""
        from scripts.train_rl import parse_args, resolve_window_size

        args = parse_args(["--window-size", "badformat"])
        config = mock.MagicMock()
        with pytest.raises(ValueError, match="Invalid --window-size"):
            resolve_window_size(args, config)

    def test_non_integer_window_size_raises(self):
        """Non-integer --window-size values should raise ValueError."""
        from scripts.train_rl import parse_args, resolve_window_size

        args = parse_args(["--window-size", "abcxdef"])
        config = mock.MagicMock()
        with pytest.raises(ValueError, match="must be integers"):
            resolve_window_size(args, config)

    def test_negative_window_size_raises(self):
        """Negative --window-size values should raise ValueError."""
        from scripts.train_rl import resolve_window_size

        # Bypass argparse (which rejects -100x200 as a flag) by
        # constructing the Namespace directly.
        args = argparse.Namespace(window_size="100x-200", orientation="portrait")
        config = mock.MagicMock()
        with pytest.raises(ValueError, match="must be positive"):
            resolve_window_size(args, config)

    def test_zero_window_size_raises(self):
        """Zero --window-size values should raise ValueError."""
        from scripts.train_rl import parse_args, resolve_window_size

        args = parse_args(["--window-size", "0x1024"])
        config = mock.MagicMock()
        with pytest.raises(ValueError, match="must be positive"):
            resolve_window_size(args, config)

    def test_browser_choices_use_edge_not_msedge(self):
        """--browser should accept 'edge' not 'msedge'."""
        from scripts.train_rl import parse_args

        args = parse_args(["--browser", "edge"])
        assert args.browser == "edge"


class TestTrainingLogger:
    """Tests for TrainingLogger JSONL writer."""

    def test_creates_jsonl_file(self, tmp_path):
        """TrainingLogger should create the JSONL file."""
        from scripts.train_rl import TrainingLogger

        path = tmp_path / "test.jsonl"
        tlog = TrainingLogger(path)
        assert path.exists()
        tlog.close()

    def test_writes_json_lines(self, tmp_path):
        """log() should write valid JSON lines."""
        from scripts.train_rl import TrainingLogger

        path = tmp_path / "test.jsonl"
        tlog = TrainingLogger(path)
        tlog.log({"event": "test", "value": 42})
        tlog.log({"event": "test2", "value": 99})
        tlog.close()

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        obj1 = json.loads(lines[0])
        assert obj1["event"] == "test"
        assert obj1["value"] == 42
        assert "timestamp" in obj1
        obj2 = json.loads(lines[1])
        assert obj2["event"] == "test2"

    def test_adds_timestamp_if_missing(self, tmp_path):
        """log() should add timestamp when not in event dict."""
        from scripts.train_rl import TrainingLogger

        path = tmp_path / "test.jsonl"
        tlog = TrainingLogger(path)
        tlog.log({"event": "test"})
        tlog.close()

        line = json.loads(path.read_text().strip())
        assert "timestamp" in line

    def test_preserves_existing_timestamp(self, tmp_path):
        """log() should not overwrite an existing timestamp."""
        from scripts.train_rl import TrainingLogger

        path = tmp_path / "test.jsonl"
        tlog = TrainingLogger(path)
        tlog.log({"event": "test", "timestamp": "2026-01-01T00:00:00"})
        tlog.close()

        line = json.loads(path.read_text().strip())
        assert line["timestamp"] == "2026-01-01T00:00:00"

    def test_close_is_idempotent(self, tmp_path):
        """close() should not raise when called multiple times."""
        from scripts.train_rl import TrainingLogger

        path = tmp_path / "test.jsonl"
        tlog = TrainingLogger(path)
        tlog.close()
        tlog.close()  # should not raise

    def test_creates_parent_dirs(self, tmp_path):
        """TrainingLogger should create parent directories."""
        from scripts.train_rl import TrainingLogger

        path = tmp_path / "subdir" / "nested" / "test.jsonl"
        tlog = TrainingLogger(path)
        assert path.exists()
        tlog.close()
