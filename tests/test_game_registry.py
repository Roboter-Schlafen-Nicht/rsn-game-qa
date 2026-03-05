"""Tests for the game plugin registry.

Tests the ``GameRegistry`` class and the ``register_game()`` /
``list_games()`` / ``discover_plugins()`` functions that provide
explicit plugin registration as an alternative to convention-based
importlib discovery.
"""

from __future__ import annotations

import types
from unittest import mock

import pytest

from games import GameRegistry, discover_plugins, list_games, load_game_plugin, register_game

# ---------------------------------------------------------------------------
# GameRegistry core behaviour
# ---------------------------------------------------------------------------


class TestGameRegistry:
    """Tests for the GameRegistry class."""

    def test_empty_registry(self):
        """A fresh registry has no entries."""
        reg = GameRegistry()
        assert reg.list() == []

    def test_register_and_retrieve(self):
        """A registered plugin can be retrieved by name."""
        reg = GameRegistry()
        FakeEnv = type("FakeEnv", (), {})
        reg.register(
            "testgame",
            env_class=FakeEnv,
            loader_class=None,
            game_name="test-game",
            default_config="configs/games/test.yaml",
            default_weights="",
        )
        entry = reg.get("testgame")
        assert entry is not None
        assert entry.env_class is FakeEnv
        assert entry.game_name == "test-game"

    def test_list_returns_sorted_names(self):
        """list() returns registered names in sorted order."""
        reg = GameRegistry()
        FakeEnv = type("FakeEnv", (), {})
        for name in ["charlie", "alpha", "bravo"]:
            reg.register(
                name,
                env_class=FakeEnv,
                loader_class=None,
                game_name=name,
                default_config=f"configs/games/{name}.yaml",
                default_weights="",
            )
        assert reg.list() == ["alpha", "bravo", "charlie"]

    def test_duplicate_registration_raises(self):
        """Registering the same name twice raises ValueError."""
        reg = GameRegistry()
        FakeEnv = type("FakeEnv", (), {})
        kwargs = dict(
            env_class=FakeEnv,
            loader_class=None,
            game_name="dup",
            default_config="c.yaml",
            default_weights="",
        )
        reg.register("dup", **kwargs)
        with pytest.raises(ValueError, match="already registered"):
            reg.register("dup", **kwargs)

    def test_get_nonexistent_returns_none(self):
        """get() for an unregistered name returns None."""
        reg = GameRegistry()
        assert reg.get("nonexistent") is None

    def test_contains(self):
        """'in' operator checks if a game is registered."""
        reg = GameRegistry()
        FakeEnv = type("FakeEnv", (), {})
        reg.register(
            "mygame",
            env_class=FakeEnv,
            loader_class=None,
            game_name="my-game",
            default_config="c.yaml",
            default_weights="",
        )
        assert "mygame" in reg
        assert "other" not in reg

    def test_extra_metadata_stored(self):
        """Extra keyword arguments are stored in the entry."""
        reg = GameRegistry()
        FakeEnv = type("FakeEnv", (), {})
        reg.register(
            "extra",
            env_class=FakeEnv,
            loader_class=None,
            game_name="extra",
            default_config="c.yaml",
            default_weights="",
            mute_js="some_js",
            setup_js="other_js",
        )
        entry = reg.get("extra")
        assert entry.extra["mute_js"] == "some_js"
        assert entry.extra["setup_js"] == "other_js"

    def test_clear(self):
        """clear() removes all entries."""
        reg = GameRegistry()
        FakeEnv = type("FakeEnv", (), {})
        reg.register(
            "game1",
            env_class=FakeEnv,
            loader_class=None,
            game_name="g1",
            default_config="c.yaml",
            default_weights="",
        )
        assert len(reg.list()) == 1
        reg.clear()
        assert reg.list() == []


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


class TestRegisterGameFunction:
    """Tests for the module-level register_game() function."""

    def test_register_game_adds_to_global_registry(self):
        """register_game() adds an entry to the module-level registry."""
        FakeEnv = type("FakeEnv", (), {})
        # Use a unique name that won't conflict with real plugins
        name = "_test_register_game_unique_xyz"
        try:
            register_game(
                name,
                env_class=FakeEnv,
                loader_class=None,
                game_name="test",
                default_config="c.yaml",
                default_weights="",
            )
            assert name in list_games()
        finally:
            # Cleanup: remove from global registry
            from games import _registry

            _registry._entries.pop(name, None)


class TestListGames:
    """Tests for the module-level list_games() function."""

    def test_list_games_includes_real_plugins(self):
        """list_games() includes the three built-in game plugins after discovery."""
        # After discover_plugins() has been called, all plugins should be listed
        discover_plugins()
        names = list_games()
        assert "breakout71" in names
        assert "hextris" in names
        assert "shapez" in names


class TestDiscoverPlugins:
    """Tests for the discover_plugins() function."""

    def test_discover_registers_all_plugins(self):
        """discover_plugins() finds and registers all game packages."""
        # Create a fresh registry to test discovery
        reg = GameRegistry()
        discover_plugins(registry=reg)
        names = reg.list()
        assert "breakout71" in names
        assert "hextris" in names
        assert "shapez" in names

    def test_discover_is_idempotent(self):
        """Calling discover_plugins() twice does not raise."""
        reg = GameRegistry()
        discover_plugins(registry=reg)
        # Second call should be a no-op (games already registered)
        discover_plugins(registry=reg)
        names = reg.list()
        assert "breakout71" in names

    def test_discover_skips_non_packages(self):
        """discover_plugins() ignores non-package modules in games/."""
        # This is implicitly tested by the fact that only the 3 game
        # packages are discovered (not __init__.py or other .py files)
        reg = GameRegistry()
        discover_plugins(registry=reg)
        # Should only have game packages, not utility modules
        for name in reg.list():
            assert not name.startswith("_")


# ---------------------------------------------------------------------------
# Integration with load_game_plugin
# ---------------------------------------------------------------------------


class TestLoadGamePluginWithRegistry:
    """Tests that load_game_plugin() uses the registry when available."""

    def test_load_from_registry(self):
        """load_game_plugin() returns a module-like object from registry."""
        # The existing tests already verify load_game_plugin works
        # This test verifies it also populates the registry
        discover_plugins()
        plugin = load_game_plugin("breakout71")
        assert hasattr(plugin, "env_class")
        assert hasattr(plugin, "game_name")

    def test_load_game_plugin_still_works_for_importlib(self):
        """load_game_plugin() falls back to importlib for unregistered games."""
        # Create a fake module that's importable but not registered
        fake_module = types.ModuleType("games.fake_importlib_test")
        fake_module.env_class = type("FakeEnv", (), {})
        fake_module.loader_class = None
        fake_module.game_name = "fake-importlib"
        fake_module.default_config = "c.yaml"
        fake_module.default_weights = ""

        with mock.patch("games.importlib.import_module", return_value=fake_module):
            plugin = load_game_plugin("fake_importlib_test")

        assert plugin.game_name == "fake-importlib"
