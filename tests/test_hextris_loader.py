"""Tests for HextrisLoader -- the game loader for Hextris.

Covers:
- Construction from config and from_repo_path
- setup() validates game directory and index.html
- Default configuration values
- Port and serve command
"""

from __future__ import annotations

import pytest

from games.hextris.loader import HextrisLoader
from src.game_loader.config import GameLoaderConfig


class TestHextrisLoaderFromRepoPath:
    """Tests for HextrisLoader.from_repo_path() factory method."""

    def test_creates_loader_with_defaults(self, tmp_path):
        """from_repo_path creates a loader with correct defaults."""
        loader = HextrisLoader.from_repo_path(tmp_path)
        assert loader.config.serve_port == 8271
        assert "8271" in loader.config.url
        assert loader.config.name == "hextris"
        assert loader.config.window_title == "HEXTRIS"

    def test_custom_port(self, tmp_path):
        """from_repo_path accepts a custom port."""
        loader = HextrisLoader.from_repo_path(tmp_path, serve_port=9999)
        assert loader.config.serve_port == 9999
        assert "9999" in loader.config.url
        assert "9999" in loader.config.serve_command

    def test_custom_window_title(self, tmp_path):
        """from_repo_path accepts a custom window title."""
        loader = HextrisLoader.from_repo_path(tmp_path, window_title="Custom Title")
        assert loader.config.window_title == "Custom Title"

    def test_game_dir_stored(self, tmp_path):
        """Game directory path is stored in config."""
        loader = HextrisLoader.from_repo_path(tmp_path)
        assert str(loader.config.game_dir) == str(tmp_path)

    def test_no_install_command(self, tmp_path):
        """Hextris has no install command (pure static HTML)."""
        loader = HextrisLoader.from_repo_path(tmp_path)
        assert loader.config.install_command is None


class TestHextrisLoaderSetup:
    """Tests for HextrisLoader.setup() validation."""

    def test_setup_missing_directory_raises(self, tmp_path):
        """setup() raises FileNotFoundError if game directory doesn't exist."""
        missing_dir = tmp_path / "nonexistent"
        loader = HextrisLoader.from_repo_path(missing_dir)
        with pytest.raises(FileNotFoundError, match="does not exist"):
            loader.setup()

    def test_setup_missing_index_html_raises(self, tmp_path):
        """setup() raises FileNotFoundError if index.html is missing."""
        # Directory exists but no index.html
        loader = HextrisLoader.from_repo_path(tmp_path)
        with pytest.raises(FileNotFoundError, match="No index.html"):
            loader.setup()

    def test_setup_valid_directory_succeeds(self, tmp_path):
        """setup() succeeds when directory and index.html exist."""
        (tmp_path / "index.html").write_text("<html></html>")
        loader = HextrisLoader.from_repo_path(tmp_path)
        loader.setup()  # Should not raise

    def test_setup_logs_validation(self, tmp_path, caplog):
        """setup() logs validation success."""
        (tmp_path / "index.html").write_text("<html></html>")
        loader = HextrisLoader.from_repo_path(tmp_path)
        import logging

        with caplog.at_level(logging.INFO):
            loader.setup()
        assert any("validated" in msg.lower() for msg in caplog.messages)


class TestHextrisLoaderConfig:
    """Tests for HextrisLoader created from a GameLoaderConfig."""

    def test_from_config_object(self, tmp_path):
        """Can create HextrisLoader from a GameLoaderConfig directly."""
        config = GameLoaderConfig(
            name="hextris",
            game_dir=str(tmp_path),
            loader_type="hextris",
            serve_command="python3 -m http.server 8271",
            serve_port=8271,
            url="http://localhost:8271",
            readiness_endpoint="http://localhost:8271",
            readiness_timeout_s=30.0,
            readiness_poll_interval_s=1.0,
            window_title="HEXTRIS",
            window_width=1280,
            window_height=1024,
        )
        loader = HextrisLoader(config)
        assert loader.config.name == "hextris"
        assert loader.config.serve_port == 8271

    def test_inherits_browser_game_loader(self):
        """HextrisLoader inherits from BrowserGameLoader."""
        from src.game_loader.browser_loader import BrowserGameLoader

        assert issubclass(HextrisLoader, BrowserGameLoader)
