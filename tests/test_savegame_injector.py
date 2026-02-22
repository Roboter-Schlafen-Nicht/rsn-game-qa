"""Tests for the savegame injection platform module.

Verifies :class:`SavegamePool` file discovery and selection, and
:class:`SavegameInjector` JS execution via a mocked Selenium driver.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from src.platform.savegame_injector import SavegameInjector, SavegamePool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _populate_save_dir(tmp_path: Path, filenames: list[str]) -> Path:
    """Create dummy save files in a temp directory and return the dir."""
    for name in filenames:
        (tmp_path / name).write_text(f'{{"save": "{name}"}}', encoding="utf-8")
    return tmp_path


# ===========================================================================
# SavegamePool — Construction
# ===========================================================================


class TestSavegamePoolConstruction:
    """Test SavegamePool initialisation and parameter validation."""

    def test_default_construction_empty_dir(self, tmp_path: Path):
        """SavegamePool with an empty directory has length 0."""
        pool = SavegamePool(save_dir=tmp_path)
        assert len(pool) == 0
        assert pool.files == []
        assert pool.selection == "random"

    def test_construction_with_save_files(self, tmp_path: Path):
        """SavegamePool discovers files matching default extensions."""
        _populate_save_dir(tmp_path, ["a.json", "b.sav", "c.save", "d.bin"])
        pool = SavegamePool(save_dir=tmp_path)
        assert len(pool) == 4

    def test_ignores_non_matching_extensions(self, tmp_path: Path):
        """Files with non-matching extensions are excluded."""
        (tmp_path / "readme.txt").write_text("not a save")
        (tmp_path / "data.json").write_text("{}")
        pool = SavegamePool(save_dir=tmp_path)
        assert len(pool) == 1
        assert pool.files[0].name == "data.json"

    def test_custom_extensions(self, tmp_path: Path):
        """Custom extensions override the default set."""
        (tmp_path / "save.custom").write_text("{}")
        (tmp_path / "save.json").write_text("{}")
        pool = SavegamePool(save_dir=tmp_path, extensions=(".custom",))
        assert len(pool) == 1
        assert pool.files[0].name == "save.custom"

    def test_invalid_selection_raises(self, tmp_path: Path):
        """Invalid selection mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid selection"):
            SavegamePool(save_dir=tmp_path, selection="invalid")

    def test_sequential_selection_accepted(self, tmp_path: Path):
        """selection='sequential' is valid."""
        pool = SavegamePool(save_dir=tmp_path, selection="sequential")
        assert pool.selection == "sequential"

    def test_random_selection_accepted(self, tmp_path: Path):
        """selection='random' is valid (default)."""
        pool = SavegamePool(save_dir=tmp_path, selection="random")
        assert pool.selection == "random"

    def test_nonexistent_directory(self):
        """Non-existent directory results in empty pool (no crash)."""
        pool = SavegamePool(save_dir="/nonexistent/dir/12345")
        assert len(pool) == 0
        assert pool.files == []

    def test_save_dir_property(self, tmp_path: Path):
        """save_dir property returns the configured path."""
        pool = SavegamePool(save_dir=tmp_path)
        assert pool.save_dir == tmp_path

    def test_files_sorted_by_name(self, tmp_path: Path):
        """Discovered files are sorted by name for determinism."""
        _populate_save_dir(tmp_path, ["c.json", "a.json", "b.json"])
        pool = SavegamePool(save_dir=tmp_path)
        names = [f.name for f in pool.files]
        assert names == ["a.json", "b.json", "c.json"]

    def test_files_property_returns_copy(self, tmp_path: Path):
        """files property returns a copy, not the internal list."""
        _populate_save_dir(tmp_path, ["a.json"])
        pool = SavegamePool(save_dir=tmp_path)
        files = pool.files
        files.clear()
        assert len(pool) == 1  # internal list unchanged


# ===========================================================================
# SavegamePool — Selection
# ===========================================================================


class TestSavegamePoolSelection:
    """Test next() selection logic for random and sequential modes."""

    def test_next_returns_none_when_empty(self, tmp_path: Path):
        """next() returns None when pool is empty."""
        pool = SavegamePool(save_dir=tmp_path)
        assert pool.next() is None

    def test_sequential_cycles_through_files(self, tmp_path: Path):
        """Sequential mode cycles through files in sorted order."""
        _populate_save_dir(tmp_path, ["a.json", "b.json", "c.json"])
        pool = SavegamePool(save_dir=tmp_path, selection="sequential")

        names = [pool.next().name for _ in range(6)]
        assert names == ["a.json", "b.json", "c.json", "a.json", "b.json", "c.json"]

    def test_random_returns_valid_file(self, tmp_path: Path):
        """Random mode returns a file from the pool."""
        _populate_save_dir(tmp_path, ["a.json", "b.json"])
        pool = SavegamePool(save_dir=tmp_path, selection="random", seed=42)

        result = pool.next()
        assert result is not None
        assert result.name in ("a.json", "b.json")

    def test_random_with_seed_is_reproducible(self, tmp_path: Path):
        """Same seed produces same sequence of selections."""
        _populate_save_dir(tmp_path, ["a.json", "b.json", "c.json"])

        pool1 = SavegamePool(save_dir=tmp_path, selection="random", seed=123)
        pool2 = SavegamePool(save_dir=tmp_path, selection="random", seed=123)

        seq1 = [pool1.next().name for _ in range(10)]
        seq2 = [pool2.next().name for _ in range(10)]
        assert seq1 == seq2

    def test_random_different_seeds_differ(self, tmp_path: Path):
        """Different seeds produce different sequences (with high probability)."""
        _populate_save_dir(tmp_path, ["a.json", "b.json", "c.json", "d.json", "e.json"])

        pool1 = SavegamePool(save_dir=tmp_path, selection="random", seed=1)
        pool2 = SavegamePool(save_dir=tmp_path, selection="random", seed=999)

        seq1 = [pool1.next().name for _ in range(20)]
        seq2 = [pool2.next().name for _ in range(20)]
        # With 5 files and 20 picks, identical sequences are astronomically unlikely
        assert seq1 != seq2

    def test_sequential_single_file(self, tmp_path: Path):
        """Sequential mode with 1 file always returns that file."""
        _populate_save_dir(tmp_path, ["only.json"])
        pool = SavegamePool(save_dir=tmp_path, selection="sequential")

        for _ in range(5):
            assert pool.next().name == "only.json"


# ===========================================================================
# SavegameInjector — Construction
# ===========================================================================


class TestSavegameInjectorConstruction:
    """Test SavegameInjector initialisation."""

    def test_construction(self, tmp_path: Path):
        """SavegameInjector stores pool and JS snippet."""
        pool = SavegamePool(save_dir=tmp_path)
        js = "return arguments[0];"
        injector = SavegameInjector(pool=pool, load_save_js=js)

        assert injector.pool is pool
        assert injector.load_save_js == js


# ===========================================================================
# SavegameInjector — Injection
# ===========================================================================


class TestSavegameInjectorInject:
    """Test inject() method with mocked Selenium driver."""

    def test_inject_reads_file_and_executes_js(self, tmp_path: Path):
        """inject() reads save file content and passes to execute_script."""
        save_content = '{"level": 5, "score": 1000}'
        (tmp_path / "save1.json").write_text(save_content, encoding="utf-8")

        pool = SavegamePool(save_dir=tmp_path, selection="sequential")
        js = "return {action: 'loaded'};"
        injector = SavegameInjector(pool=pool, load_save_js=js)

        mock_driver = mock.MagicMock()
        mock_driver.execute_script.return_value = {"action": "loaded"}

        result = injector.inject(mock_driver)

        mock_driver.execute_script.assert_called_once_with(js, save_content)
        assert result["action"] == "loaded"
        assert "save_file" in result

    def test_inject_raises_on_empty_pool(self, tmp_path: Path):
        """inject() raises RuntimeError when pool is empty."""
        pool = SavegamePool(save_dir=tmp_path)  # empty
        injector = SavegameInjector(pool=pool, load_save_js="return;")

        mock_driver = mock.MagicMock()
        with pytest.raises(RuntimeError, match="empty"):
            injector.inject(mock_driver)

    def test_inject_raises_on_unreadable_file(self, tmp_path: Path):
        """inject() raises RuntimeError when save file cannot be read."""
        _populate_save_dir(tmp_path, ["save.json"])
        pool = SavegamePool(save_dir=tmp_path, selection="sequential")
        injector = SavegameInjector(pool=pool, load_save_js="return;")

        # Make the file unreadable by mocking Path.read_text
        mock_driver = mock.MagicMock()
        with mock.patch.object(Path, "read_text", side_effect=PermissionError("denied")):
            with pytest.raises(RuntimeError, match="Failed to read"):
                injector.inject(mock_driver)

    def test_inject_returns_error_dict_on_js_failure(self, tmp_path: Path):
        """inject() returns error dict when JS execution fails."""
        _populate_save_dir(tmp_path, ["save.json"])
        pool = SavegamePool(save_dir=tmp_path, selection="sequential")
        injector = SavegameInjector(pool=pool, load_save_js="throw Error('fail');")

        mock_driver = mock.MagicMock()
        mock_driver.execute_script.side_effect = RuntimeError("JS error")

        result = injector.inject(mock_driver)
        assert result["action"] == "error"
        assert "JS error" in result["error"]
        assert "save_file" in result

    def test_inject_handles_null_js_result(self, tmp_path: Path):
        """inject() handles null return from JS (converts to empty dict)."""
        _populate_save_dir(tmp_path, ["save.json"])
        pool = SavegamePool(save_dir=tmp_path, selection="sequential")
        injector = SavegameInjector(pool=pool, load_save_js="return null;")

        mock_driver = mock.MagicMock()
        mock_driver.execute_script.return_value = None

        result = injector.inject(mock_driver)
        assert result == {"save_file": str(tmp_path / "save.json")}

    def test_inject_sequential_cycles_saves(self, tmp_path: Path):
        """Multiple inject() calls cycle through saves sequentially."""
        _populate_save_dir(tmp_path, ["a.json", "b.json"])
        pool = SavegamePool(save_dir=tmp_path, selection="sequential")
        injector = SavegameInjector(pool=pool, load_save_js="return {};")

        mock_driver = mock.MagicMock()
        mock_driver.execute_script.side_effect = lambda *a: {}

        r1 = injector.inject(mock_driver)
        r2 = injector.inject(mock_driver)
        r3 = injector.inject(mock_driver)

        # a.json -> b.json -> a.json (cycles)
        assert r1["save_file"].endswith("a.json")
        assert r2["save_file"].endswith("b.json")
        assert r3["save_file"].endswith("a.json")

    def test_inject_passes_file_content_as_first_arg(self, tmp_path: Path):
        """The save file content is passed as arguments[0] to JS."""
        content = '{"factory": [1,2,3]}'
        (tmp_path / "factory.json").write_text(content, encoding="utf-8")

        pool = SavegamePool(save_dir=tmp_path, selection="sequential")
        js_snippet = "var data = JSON.parse(arguments[0]); return data;"
        injector = SavegameInjector(pool=pool, load_save_js=js_snippet)

        mock_driver = mock.MagicMock()
        mock_driver.execute_script.return_value = {"factory": [1, 2, 3]}

        injector.inject(mock_driver)

        # Verify the exact content string was passed
        call_args = mock_driver.execute_script.call_args
        assert call_args[0][0] == js_snippet
        assert call_args[0][1] == content
