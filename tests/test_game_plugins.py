"""Tests for the game plugin loading system.

Verifies ``load_game_plugin()`` and ``get_env_class()`` from the
``games`` package, including validation of required attributes and
error handling for missing/incomplete plugins.
"""

from __future__ import annotations

import types
from unittest import mock

import pytest

from games import get_env_class, load_game_plugin


class TestLoadGamePlugin:
    """Tests for load_game_plugin()."""

    def test_loads_breakout71_plugin(self):
        """Loading 'breakout71' returns a module with all required attrs."""
        plugin = load_game_plugin("breakout71")

        assert isinstance(plugin, types.ModuleType)
        assert hasattr(plugin, "env_class")
        assert hasattr(plugin, "loader_class")
        assert hasattr(plugin, "game_name")
        assert hasattr(plugin, "default_config")
        assert hasattr(plugin, "default_weights")

    def test_breakout71_plugin_metadata(self):
        """Breakout71 plugin has correct metadata values."""
        plugin = load_game_plugin("breakout71")

        assert plugin.game_name == "breakout-71"
        assert "breakout-71" in plugin.default_config
        assert "breakout71" in plugin.default_weights

    def test_breakout71_env_class_is_class(self):
        """env_class is an actual class (not an instance)."""
        plugin = load_game_plugin("breakout71")

        assert isinstance(plugin.env_class, type)

    def test_nonexistent_plugin_raises_import_error(self):
        """Loading a plugin that doesn't exist raises ImportError."""
        with pytest.raises(ImportError):
            load_game_plugin("nonexistent_game_xyz")

    def test_incomplete_plugin_raises_attribute_error(self):
        """A plugin missing required attributes raises AttributeError."""
        # Create a fake module with only some attributes
        fake_module = types.ModuleType("games.fake_game")
        fake_module.env_class = object
        # Missing: loader_class, game_name, default_config, default_weights

        with mock.patch("games.importlib.import_module", return_value=fake_module):
            with pytest.raises(AttributeError, match="missing required attributes"):
                load_game_plugin("fake_game")

    def test_incomplete_plugin_lists_missing_attrs(self):
        """Error message lists which attributes are missing."""
        fake_module = types.ModuleType("games.fake_game")
        fake_module.env_class = object
        fake_module.game_name = "fake"

        with mock.patch("games.importlib.import_module", return_value=fake_module):
            with pytest.raises(
                AttributeError,
                match="loader_class.*default_config.*default_weights",
            ):
                load_game_plugin("fake_game")

    def test_complete_fake_plugin_succeeds(self):
        """A fake module with all required attrs loads successfully."""
        fake_module = types.ModuleType("games.complete_fake")
        fake_module.env_class = type("FakeEnv", (), {})
        fake_module.loader_class = None
        fake_module.game_name = "fake-game"
        fake_module.default_config = "configs/games/fake.yaml"
        fake_module.default_weights = "weights/fake/best.pt"

        with mock.patch("games.importlib.import_module", return_value=fake_module):
            plugin = load_game_plugin("complete_fake")

        assert plugin.game_name == "fake-game"
        assert plugin.env_class.__name__ == "FakeEnv"

    def test_optional_attrs_not_required(self):
        """Optional attrs like mute_js don't cause validation failure."""
        plugin = load_game_plugin("breakout71")

        # mute_js is optional — its presence shouldn't matter for loading
        assert hasattr(plugin, "mute_js")

    def test_breakout71_has_setup_js(self):
        """Breakout71 plugin should define setup_js for RL training settings."""
        plugin = load_game_plugin("breakout71")

        assert hasattr(plugin, "setup_js")
        setup_js = plugin.setup_js
        assert "mobile-mode" in setup_js
        assert "touch_delayed_start" in setup_js

    def test_breakout71_has_reinit_js(self):
        """Breakout71 plugin should define reinit_js for game re-init after refresh."""
        plugin = load_game_plugin("breakout71")

        assert hasattr(plugin, "reinit_js")
        reinit_js = plugin.reinit_js
        assert "restart" in reinit_js

    # -- Hextris plugin tests --------------------------------------------------

    def test_loads_hextris_plugin(self):
        """Loading 'hextris' returns a module with all required attrs."""
        plugin = load_game_plugin("hextris")

        assert isinstance(plugin, types.ModuleType)
        assert hasattr(plugin, "env_class")
        assert hasattr(plugin, "loader_class")
        assert hasattr(plugin, "game_name")
        assert hasattr(plugin, "default_config")
        assert hasattr(plugin, "default_weights")

    def test_hextris_plugin_metadata(self):
        """Hextris plugin has correct metadata values."""
        plugin = load_game_plugin("hextris")

        assert plugin.game_name == "hextris"
        assert "hextris" in plugin.default_config
        assert plugin.default_weights == ""  # CNN-only, no YOLO weights

    def test_hextris_env_class_is_class(self):
        """Hextris env_class is an actual class (not an instance)."""
        plugin = load_game_plugin("hextris")

        assert isinstance(plugin.env_class, type)

    def test_hextris_loader_class_is_class(self):
        """Hextris loader_class is an actual class."""
        plugin = load_game_plugin("hextris")

        assert isinstance(plugin.loader_class, type)

    def test_hextris_has_mute_js(self):
        """Hextris plugin defines mute_js for audio muting."""
        plugin = load_game_plugin("hextris")

        assert hasattr(plugin, "mute_js")
        assert "Audio" in plugin.mute_js

    def test_hextris_no_setup_js(self):
        """Hextris plugin does not define setup_js (not needed)."""
        plugin = load_game_plugin("hextris")

        assert not hasattr(plugin, "setup_js")

    def test_hextris_no_reinit_js(self):
        """Hextris plugin does not define reinit_js (not needed)."""
        plugin = load_game_plugin("hextris")

        assert not hasattr(plugin, "reinit_js")


class TestGetEnvClass:
    """Tests for get_env_class()."""

    def test_returns_breakout71_env_class(self):
        """get_env_class('breakout71') returns the Breakout71Env class."""
        env_cls = get_env_class("breakout71")

        from games.breakout71.env import Breakout71Env

        assert env_cls is Breakout71Env

    def test_returns_hextris_env_class(self):
        """get_env_class('hextris') returns the HextrisEnv class."""
        env_cls = get_env_class("hextris")

        from games.hextris.env import HextrisEnv

        assert env_cls is HextrisEnv

    def test_nonexistent_game_raises(self):
        """get_env_class for nonexistent game raises ImportError."""
        with pytest.raises(ImportError):
            get_env_class("nonexistent_game_xyz")

    def test_returns_a_class(self):
        """get_env_class returns a type, not an instance."""
        env_cls = get_env_class("breakout71")
        assert isinstance(env_cls, type)
