"""Tests for ShapezLoader -- the game loader for shapez.io.

Covers:
- Construction from config and from_repo_path
- setup() validates game directory, gulp/, and gulpfile.js
- Default configuration values
- Port and serve settings
- nvm preamble usage
- start() process spawning
"""

from __future__ import annotations

from unittest import mock

import pytest

from games.shapez.loader import _NVM_PREAMBLE, ShapezLoader
from src.game_loader.config import GameLoaderConfig

# -- from_repo_path -----------------------------------------------------------


class TestShapezLoaderFromRepoPath:
    """Tests for ShapezLoader.from_repo_path() factory method."""

    def test_creates_loader_with_defaults(self, tmp_path):
        """from_repo_path creates a loader with correct defaults."""
        loader = ShapezLoader.from_repo_path(tmp_path)
        assert loader.config.serve_port == 3005
        assert "3005" in loader.config.url
        assert loader.config.name == "shapez"
        assert loader.config.window_title == "shapez.io"

    def test_custom_port(self, tmp_path):
        """from_repo_path accepts a custom port."""
        loader = ShapezLoader.from_repo_path(tmp_path, serve_port=9999)
        assert loader.config.serve_port == 9999
        assert "9999" in loader.config.url
        assert "9999" in loader.config.readiness_endpoint

    def test_custom_window_title(self, tmp_path):
        """from_repo_path accepts a custom window title."""
        loader = ShapezLoader.from_repo_path(tmp_path, window_title="Custom")
        assert loader.config.window_title == "Custom"

    def test_game_dir_stored(self, tmp_path):
        """Game directory path is stored in config."""
        loader = ShapezLoader.from_repo_path(tmp_path)
        assert str(loader.config.game_dir) == str(tmp_path)

    def test_no_install_command(self, tmp_path):
        """shapez.io has no install_command (setup() handles it)."""
        loader = ShapezLoader.from_repo_path(tmp_path)
        assert loader.config.install_command is None

    def test_no_serve_command(self, tmp_path):
        """shapez.io has no serve_command (start() handles it)."""
        loader = ShapezLoader.from_repo_path(tmp_path)
        assert loader.config.serve_command is None

    def test_readiness_timeout(self, tmp_path):
        """Default readiness timeout is 120 seconds."""
        loader = ShapezLoader.from_repo_path(tmp_path)
        assert loader.config.readiness_timeout_s == 120.0

    def test_custom_readiness_timeout(self, tmp_path):
        """from_repo_path accepts custom readiness timeout."""
        loader = ShapezLoader.from_repo_path(tmp_path, readiness_timeout_s=60.0)
        assert loader.config.readiness_timeout_s == 60.0

    def test_window_dimensions(self, tmp_path):
        """Default window dimensions are 1280x1024."""
        loader = ShapezLoader.from_repo_path(tmp_path)
        assert loader.config.window_width == 1280
        assert loader.config.window_height == 1024

    def test_loader_type_is_shapez(self, tmp_path):
        """Loader type is 'shapez'."""
        loader = ShapezLoader.from_repo_path(tmp_path)
        assert loader.config.loader_type == "shapez"


# -- setup() ------------------------------------------------------------------


class TestShapezLoaderSetup:
    """Tests for ShapezLoader.setup() validation and dependency install."""

    def test_setup_missing_directory_raises(self, tmp_path):
        """setup() raises FileNotFoundError if game directory doesn't exist."""
        missing_dir = tmp_path / "nonexistent"
        loader = ShapezLoader.from_repo_path(missing_dir)
        with pytest.raises(FileNotFoundError, match="does not exist"):
            loader.setup()

    def test_setup_missing_gulp_dir_raises(self, tmp_path):
        """setup() raises FileNotFoundError if gulp/ doesn't exist."""
        loader = ShapezLoader.from_repo_path(tmp_path)
        with pytest.raises(FileNotFoundError, match="gulp.*not found"):
            loader.setup()

    def test_setup_missing_gulpfile_raises(self, tmp_path):
        """setup() raises FileNotFoundError if gulpfile.js is missing."""
        (tmp_path / "gulp").mkdir()
        loader = ShapezLoader.from_repo_path(tmp_path)
        with pytest.raises(FileNotFoundError, match="gulpfile.js"):
            loader.setup()

    def test_setup_valid_directory_runs_yarn(self, tmp_path):
        """setup() runs yarn in root and gulp/ directories."""
        (tmp_path / "gulp").mkdir()
        (tmp_path / "gulp" / "gulpfile.js").write_text("// build")
        loader = ShapezLoader.from_repo_path(tmp_path)

        with mock.patch.object(loader, "_run_nvm_command") as mock_run:
            loader.setup()

        assert mock_run.call_count == 2
        # First call: yarn in root
        mock_run.assert_any_call("yarn", cwd=tmp_path)
        # Second call: yarn in gulp/
        mock_run.assert_any_call("yarn", cwd=tmp_path / "gulp")

    def test_setup_logs_completion(self, tmp_path, caplog):
        """setup() logs completion message."""
        (tmp_path / "gulp").mkdir()
        (tmp_path / "gulp" / "gulpfile.js").write_text("// build")
        loader = ShapezLoader.from_repo_path(tmp_path)

        import logging

        with (
            mock.patch.object(loader, "_run_nvm_command"),
            caplog.at_level(logging.INFO),
        ):
            loader.setup()
        assert any("complete" in msg.lower() for msg in caplog.messages)


# -- start() ------------------------------------------------------------------


class TestShapezLoaderStart:
    """Tests for ShapezLoader.start() server spawning."""

    def test_start_missing_gulp_dir_raises(self, tmp_path):
        """start() raises GameLoaderError if gulp/ doesn't exist."""
        loader = ShapezLoader.from_repo_path(tmp_path)
        from src.game_loader.base import GameLoaderError

        with pytest.raises(GameLoaderError, match="gulp"):
            loader.start()

    def test_start_already_running_warns(self, tmp_path, caplog):
        """start() warns if already running."""
        (tmp_path / "gulp").mkdir()
        loader = ShapezLoader.from_repo_path(tmp_path)
        loader._running = True

        import logging

        with caplog.at_level(logging.WARNING):
            loader.start()
        assert any("already running" in msg.lower() for msg in caplog.messages)

    def test_start_spawns_process(self, tmp_path):
        """start() spawns a subprocess with yarn gulp."""
        (tmp_path / "gulp").mkdir()
        loader = ShapezLoader.from_repo_path(tmp_path)

        mock_process = mock.MagicMock()
        mock_process.pid = 12345

        with (
            mock.patch("subprocess.Popen", return_value=mock_process) as mock_popen,
            mock.patch.object(loader, "_wait_until_ready"),
        ):
            loader.start()

        assert mock_popen.called
        call_args = mock_popen.call_args
        # Command should contain nvm preamble and yarn gulp
        cmd = call_args[0][0]
        assert "yarn gulp" in cmd
        assert "nvm" in cmd
        # cwd should be gulp/
        assert str(tmp_path / "gulp") in str(call_args[1].get("cwd", ""))

    def test_start_sets_running_flag(self, tmp_path):
        """start() sets _running to True after success."""
        (tmp_path / "gulp").mkdir()
        loader = ShapezLoader.from_repo_path(tmp_path)

        mock_process = mock.MagicMock()
        mock_process.pid = 12345

        with (
            mock.patch("subprocess.Popen", return_value=mock_process),
            mock.patch.object(loader, "_wait_until_ready"),
        ):
            loader.start()

        assert loader._running is True

    def test_start_stores_process(self, tmp_path):
        """start() stores the subprocess."""
        (tmp_path / "gulp").mkdir()
        loader = ShapezLoader.from_repo_path(tmp_path)

        mock_process = mock.MagicMock()
        mock_process.pid = 12345

        with (
            mock.patch("subprocess.Popen", return_value=mock_process),
            mock.patch.object(loader, "_wait_until_ready"),
        ):
            loader.start()

        assert loader._process is mock_process


# -- _run_nvm_command ---------------------------------------------------------


class TestRunNvmCommand:
    """Tests for _run_nvm_command() helper."""

    def test_nvm_preamble_in_command(self, tmp_path):
        """Command includes nvm preamble."""
        loader = ShapezLoader.from_repo_path(tmp_path)

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(returncode=0)
            loader._run_nvm_command("yarn", cwd=tmp_path)

        cmd = mock_run.call_args[0][0]
        assert "nvm" in cmd
        assert "yarn" in cmd

    def test_failed_command_raises_runtime_error(self, tmp_path):
        """Failed command raises RuntimeError."""
        loader = ShapezLoader.from_repo_path(tmp_path)

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(returncode=1, stdout="", stderr="error: ENOENT")
            with pytest.raises(RuntimeError, match="Command failed"):
                loader._run_nvm_command("yarn", cwd=tmp_path)

    def test_successful_command_no_exception(self, tmp_path):
        """Successful command does not raise."""
        loader = ShapezLoader.from_repo_path(tmp_path)

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(returncode=0)
            loader._run_nvm_command("yarn", cwd=tmp_path)  # Should not raise


# -- Config construction ------------------------------------------------------


class TestShapezLoaderConfig:
    """Tests for ShapezLoader created from a GameLoaderConfig."""

    def test_from_config_object(self, tmp_path):
        """Can create ShapezLoader from a GameLoaderConfig directly."""
        config = GameLoaderConfig(
            name="shapez",
            game_dir=str(tmp_path),
            loader_type="shapez",
            serve_port=3005,
            url="http://localhost:3005",
            readiness_endpoint="http://localhost:3005",
            readiness_timeout_s=120.0,
            readiness_poll_interval_s=2.0,
            window_title="shapez.io",
            window_width=1280,
            window_height=1024,
        )
        loader = ShapezLoader(config)
        assert loader.config.name == "shapez"
        assert loader.config.serve_port == 3005

    def test_inherits_browser_game_loader(self):
        """ShapezLoader inherits from BrowserGameLoader."""
        from src.game_loader.browser_loader import BrowserGameLoader

        assert issubclass(ShapezLoader, BrowserGameLoader)


# -- NVM preamble constant ----------------------------------------------------


class TestNvmPreamble:
    """Tests for the _NVM_PREAMBLE constant."""

    def test_contains_nvm_source(self):
        """Preamble sources nvm.sh."""
        assert "nvm.sh" in _NVM_PREAMBLE

    def test_contains_nvm_use_16(self):
        """Preamble switches to Node.js 16."""
        assert "nvm use 16" in _NVM_PREAMBLE

    def test_ends_with_separator(self):
        """Preamble ends with '&& ' for command chaining."""
        assert _NVM_PREAMBLE.rstrip().endswith("&&")
