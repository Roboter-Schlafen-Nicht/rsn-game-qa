"""Tests for the game_loader subsystem.

Tests cover:

- GameLoaderConfig construction and defaults
- YAML config loading (load_game_config)
- Environment variable expansion in configs
- YAML config validation (unknown / missing fields)
- GameLoader ABC contract
- BrowserGameLoader lifecycle (with mocked subprocess/HTTP)
- TCP socket readiness probe
- Breakout71Loader defaults and cache clearing
- Factory (create_loader) dispatch
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path
from unittest import mock

import pytest

from src.game_loader.base import GameLoader, GameLoaderError
from src.game_loader.browser_loader import BrowserGameLoader
from src.game_loader.breakout71_loader import Breakout71Loader
from src.game_loader.config import (
    GameLoaderConfig,
    _expand_vars,
    load_game_config,
)
from src.game_loader.factory import create_loader, register_loader


# ── Helpers ─────────────────────────────────────────────────────────


def _make_config(**overrides) -> GameLoaderConfig:
    """Build a minimal GameLoaderConfig with sensible test defaults."""
    defaults = dict(
        name="test-game",
        game_dir=".",
        loader_type="browser",
        install_command=None,
        serve_command="echo serving",
        serve_port=9999,
        url="http://localhost:9999",
    )
    defaults.update(overrides)
    return GameLoaderConfig(**defaults)


# ── GameLoaderConfig ────────────────────────────────────────────────


class TestGameLoaderConfig:
    """Tests for the configuration dataclass."""

    def test_minimal_construction(self):
        """Config can be built with just name and game_dir."""
        cfg = GameLoaderConfig(name="foo", game_dir="/tmp/foo")
        assert cfg.name == "foo"
        assert cfg.game_dir == Path("/tmp/foo")
        assert cfg.loader_type == "browser"

    def test_defaults(self):
        """Default values are applied correctly."""
        cfg = GameLoaderConfig(name="foo", game_dir=".")
        assert cfg.serve_port == 1234
        assert cfg.readiness_timeout_s == 120.0
        assert cfg.readiness_poll_interval_s == 2.0
        assert cfg.install_command == "npm install"

    def test_readiness_endpoint_defaults_to_url(self):
        """If readiness_endpoint is empty, it falls back to url."""
        cfg = GameLoaderConfig(name="foo", game_dir=".", url="http://localhost:5000")
        assert cfg.readiness_endpoint == "http://localhost:5000"

    def test_readiness_endpoint_override(self):
        """Explicit readiness_endpoint is preserved."""
        cfg = GameLoaderConfig(
            name="foo",
            game_dir=".",
            url="http://localhost:5000",
            readiness_endpoint="http://localhost:5000/health",
        )
        assert cfg.readiness_endpoint == "http://localhost:5000/health"

    def test_game_dir_converted_to_path(self):
        """game_dir string is converted to a Path."""
        cfg = GameLoaderConfig(name="foo", game_dir="/some/path")
        assert isinstance(cfg.game_dir, Path)
        assert cfg.game_dir == Path("/some/path")

    def test_env_vars_default_empty(self):
        """env_vars defaults to an empty dict."""
        cfg = GameLoaderConfig(name="foo", game_dir=".")
        assert cfg.env_vars == {}

    def test_custom_env_vars(self):
        """Custom env_vars are stored correctly."""
        cfg = GameLoaderConfig(name="foo", game_dir=".", env_vars={"NODE_ENV": "test"})
        assert cfg.env_vars == {"NODE_ENV": "test"}


# ── YAML loading ────────────────────────────────────────────────────


class TestLoadGameConfig:
    """Tests for load_game_config YAML parsing."""

    def test_load_from_yaml(self, tmp_path: Path):
        """A valid YAML file produces a correct GameLoaderConfig."""
        yaml_content = textwrap.dedent("""\
            name: my-game
            game_dir: /opt/games/my-game
            loader_type: browser
            serve_command: npm run dev
            serve_port: 3000
            url: http://localhost:3000
            readiness_timeout_s: 60.0
        """)
        config_file = tmp_path / "my-game.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        cfg = load_game_config("my-game", configs_dir=tmp_path)
        assert cfg.name == "my-game"
        assert cfg.serve_port == 3000
        assert cfg.readiness_timeout_s == 60.0
        assert cfg.loader_type == "browser"

    def test_missing_config_raises(self, tmp_path: Path):
        """FileNotFoundError is raised for a missing config."""
        with pytest.raises(FileNotFoundError, match="no-such-game"):
            load_game_config("no-such-game", configs_dir=tmp_path)

    def test_load_breakout71_config(self):
        """The shipped breakout-71.yaml config loads successfully."""
        configs_dir = Path(__file__).resolve().parent.parent / "configs" / "games"
        if not (configs_dir / "breakout-71.yaml").exists():
            pytest.skip("breakout-71.yaml not found in configs/games/")

        cfg = load_game_config("breakout-71", configs_dir=configs_dir)
        assert cfg.name == "breakout-71"
        assert cfg.loader_type == "breakout-71"
        assert cfg.serve_port == 1234


# ── Environment variable expansion ──────────────────────────────────


class TestEnvVarExpansion:
    """Tests for $VAR and ${VAR} expansion in config values."""

    def test_expand_dollar_var(self):
        """$VAR syntax is expanded from environment."""
        with mock.patch.dict(os.environ, {"MY_GAME_DIR": "/opt/games"}):
            result = _expand_vars("$MY_GAME_DIR/breakout")
        assert result == "/opt/games/breakout"

    def test_expand_braced_var(self):
        """${VAR} syntax is expanded from environment."""
        with mock.patch.dict(os.environ, {"MY_GAME_DIR": "/opt/games"}):
            result = _expand_vars("${MY_GAME_DIR}/breakout")
        assert result == "/opt/games/breakout"

    def test_expand_braced_with_default(self):
        """${VAR:-default} uses the default when VAR is unset."""
        env = os.environ.copy()
        env.pop("UNSET_VAR_12345", None)
        with mock.patch.dict(os.environ, env, clear=True):
            result = _expand_vars("${UNSET_VAR_12345:-/fallback/path}")
        assert result == "/fallback/path"

    def test_expand_braced_with_default_var_set(self):
        """${VAR:-default} uses the env value when VAR is set."""
        with mock.patch.dict(os.environ, {"MY_DIR": "/real/path"}):
            result = _expand_vars("${MY_DIR:-/fallback}")
        assert result == "/real/path"

    def test_undefined_var_left_as_is(self):
        """Undefined $VAR references are left unchanged."""
        env = os.environ.copy()
        env.pop("TOTALLY_UNDEFINED_XYZ", None)
        with mock.patch.dict(os.environ, env, clear=True):
            result = _expand_vars("$TOTALLY_UNDEFINED_XYZ")
        assert result == "$TOTALLY_UNDEFINED_XYZ"

    def test_expand_tilde_in_game_dir(self):
        """~ in game_dir is expanded to the home directory."""
        cfg = GameLoaderConfig(name="foo", game_dir="~/games/foo")
        assert "~" not in str(cfg.game_dir)
        assert cfg.game_dir == Path.home() / "games" / "foo"

    def test_expand_var_in_yaml(self, tmp_path: Path):
        """Environment variables are expanded when loading from YAML."""
        yaml_content = textwrap.dedent("""\
            name: env-game
            game_dir: $TEST_GAME_DIR_XYZ/sub
            loader_type: browser
            serve_command: echo hi
            serve_port: 8080
            url: http://localhost:8080
        """)
        (tmp_path / "env-game.yaml").write_text(yaml_content, encoding="utf-8")

        with mock.patch.dict(os.environ, {"TEST_GAME_DIR_XYZ": "/resolved"}):
            cfg = load_game_config("env-game", configs_dir=tmp_path)
        assert cfg.game_dir == Path("/resolved/sub")

    def test_expand_braced_default_in_yaml(self, tmp_path: Path):
        """${VAR:-default} in YAML falls back correctly."""
        yaml_content = textwrap.dedent("""\
            name: default-game
            game_dir: ${NONEXISTENT_VAR_ABC:-/default/path}
            loader_type: browser
            serve_command: echo hi
            serve_port: 8080
            url: http://localhost:8080
        """)
        (tmp_path / "default-game.yaml").write_text(yaml_content, encoding="utf-8")

        env = os.environ.copy()
        env.pop("NONEXISTENT_VAR_ABC", None)
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = load_game_config("default-game", configs_dir=tmp_path)
        assert cfg.game_dir == Path("/default/path")


# ── YAML config validation ──────────────────────────────────────────


class TestConfigValidation:
    """Tests for YAML config validation and helpful error messages."""

    def test_unknown_field_raises_valueerror(self, tmp_path: Path):
        """Unknown fields in YAML produce a ValueError listing valid fields."""
        yaml_content = textwrap.dedent("""\
            name: bad-game
            game_dir: /tmp/game
            serv_command: echo hi
        """)
        (tmp_path / "bad-game.yaml").write_text(yaml_content, encoding="utf-8")

        with pytest.raises(ValueError, match="Unknown fields.*serv_command"):
            load_game_config("bad-game", configs_dir=tmp_path)

    def test_unknown_field_lists_valid_fields(self, tmp_path: Path):
        """The error message includes the list of valid field names."""
        yaml_content = textwrap.dedent("""\
            name: bad-game
            game_dir: /tmp/game
            typo_field: oops
        """)
        (tmp_path / "bad-game.yaml").write_text(yaml_content, encoding="utf-8")

        with pytest.raises(ValueError, match="Valid fields:"):
            load_game_config("bad-game", configs_dir=tmp_path)

    def test_non_mapping_yaml_raises(self, tmp_path: Path):
        """A YAML file that isn't a mapping raises ValueError."""
        (tmp_path / "list-game.yaml").write_text("- item1\n- item2\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Expected a YAML mapping"):
            load_game_config("list-game", configs_dir=tmp_path)

    def test_valid_yaml_still_works(self, tmp_path: Path):
        """A correct YAML file still loads without error."""
        yaml_content = textwrap.dedent("""\
            name: ok-game
            game_dir: /tmp/game
            loader_type: browser
            serve_command: echo hi
            serve_port: 5000
            url: http://localhost:5000
        """)
        (tmp_path / "ok-game.yaml").write_text(yaml_content, encoding="utf-8")

        cfg = load_game_config("ok-game", configs_dir=tmp_path)
        assert cfg.name == "ok-game"
        assert cfg.serve_port == 5000


# ── GameLoader ABC ──────────────────────────────────────────────────


class TestGameLoaderABC:
    """Tests for the abstract base class contract."""

    def test_cannot_instantiate_directly(self):
        """GameLoader cannot be instantiated (it is abstract)."""
        cfg = _make_config()
        with pytest.raises(TypeError):
            GameLoader(cfg)  # type: ignore[abstract]

    def test_repr_stopped(self):
        """repr shows 'stopped' when not running."""
        cfg = _make_config()
        loader = BrowserGameLoader(cfg)
        assert "stopped" in repr(loader)
        assert "test-game" in repr(loader)

    def test_url_is_none_when_stopped(self):
        """url property returns None when the loader is not running."""
        cfg = _make_config()
        loader = BrowserGameLoader(cfg)
        assert loader.url is None

    def test_running_is_false_initially(self):
        """running is False right after construction."""
        cfg = _make_config()
        loader = BrowserGameLoader(cfg)
        assert loader.running is False

    def test_exit_returns_false(self):
        """__exit__ returns False so exceptions propagate."""
        cfg = _make_config(install_command=None)
        loader = BrowserGameLoader(cfg)
        result = loader.__exit__(None, None, None)
        assert result is False


# ── BrowserGameLoader ───────────────────────────────────────────────


class TestBrowserGameLoader:
    """Tests for the browser-based game loader."""

    def test_construction(self):
        """BrowserGameLoader can be constructed with a config."""
        cfg = _make_config()
        loader = BrowserGameLoader(cfg)
        assert loader.name == "test-game"
        assert not loader.running

    def test_setup_skips_when_no_install_command(self):
        """setup() does nothing when install_command is None."""
        cfg = _make_config(install_command=None)
        loader = BrowserGameLoader(cfg)
        # Should not raise
        loader.setup()

    def test_setup_raises_on_missing_dir(self, tmp_path: Path):
        """setup() raises GameLoaderError if game_dir doesn't exist."""
        cfg = _make_config(
            game_dir=str(tmp_path / "nonexistent"),
            install_command="npm install",
        )
        loader = BrowserGameLoader(cfg)
        with pytest.raises(GameLoaderError, match="does not exist"):
            loader.setup()

    @mock.patch("subprocess.run")
    def test_setup_runs_install_command(self, mock_run, tmp_path: Path):
        """setup() executes the install command in game_dir."""
        mock_run.return_value = subprocess.CompletedProcess(
            args="npm install", returncode=0, stdout="", stderr=""
        )
        cfg = _make_config(
            game_dir=str(tmp_path),
            install_command="npm install",
        )
        loader = BrowserGameLoader(cfg)
        loader.setup()

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        assert call_kwargs[0][0] == "npm install"
        assert call_kwargs[1]["cwd"] == str(tmp_path)

    @mock.patch("subprocess.run")
    def test_setup_raises_on_install_failure(self, mock_run, tmp_path: Path):
        """setup() raises GameLoaderError if install command fails."""
        mock_run.return_value = subprocess.CompletedProcess(
            args="npm install", returncode=1, stdout="", stderr="ERR!"
        )
        cfg = _make_config(
            game_dir=str(tmp_path),
            install_command="npm install",
        )
        loader = BrowserGameLoader(cfg)
        with pytest.raises(GameLoaderError, match="Install command failed"):
            loader.setup()

    def test_is_ready_returns_false_when_nothing_running(self):
        """is_ready() returns False when no server is running."""
        cfg = _make_config()
        loader = BrowserGameLoader(cfg)
        assert loader.is_ready() is False

    @mock.patch("src.game_loader.browser_loader.socket.create_connection")
    def test_tcp_probe_returns_true_on_connect(self, mock_conn):
        """_tcp_probe() returns True when the port accepts connections."""
        mock_conn.return_value.__enter__ = mock.Mock()
        mock_conn.return_value.__exit__ = mock.Mock(return_value=False)
        cfg = _make_config()
        loader = BrowserGameLoader(cfg)
        assert loader._tcp_probe() is True

    @mock.patch(
        "src.game_loader.browser_loader.socket.create_connection",
        side_effect=ConnectionRefusedError,
    )
    def test_tcp_probe_returns_false_on_refused(self, mock_conn):
        """_tcp_probe() returns False when connection is refused."""
        cfg = _make_config()
        loader = BrowserGameLoader(cfg)
        assert loader._tcp_probe() is False

    @mock.patch(
        "src.game_loader.browser_loader.socket.create_connection",
        side_effect=OSError("timeout"),
    )
    def test_tcp_probe_returns_false_on_oserror(self, mock_conn):
        """_tcp_probe() returns False on timeout/OS error."""
        cfg = _make_config()
        loader = BrowserGameLoader(cfg)
        assert loader._tcp_probe() is False

    def test_stop_is_safe_when_not_started(self):
        """stop() does not raise when called before start()."""
        cfg = _make_config()
        loader = BrowserGameLoader(cfg)
        loader.stop()
        assert not loader.running

    @mock.patch("subprocess.Popen")
    @mock.patch.object(BrowserGameLoader, "is_ready", return_value=True)
    def test_start_launches_process(self, mock_ready, mock_popen, tmp_path: Path):
        """start() spawns a subprocess and sets running=True."""
        mock_proc = mock.MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        cfg = _make_config(game_dir=str(tmp_path))
        loader = BrowserGameLoader(cfg)
        loader.start()

        assert loader.running is True
        assert loader.url == "http://localhost:9999"
        mock_popen.assert_called_once()

    @mock.patch("subprocess.Popen")
    @mock.patch.object(BrowserGameLoader, "is_ready", return_value=True)
    def test_start_then_stop(self, mock_ready, mock_popen, tmp_path: Path):
        """Full start→stop lifecycle completes cleanly."""
        mock_proc = mock.MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        cfg = _make_config(game_dir=str(tmp_path))
        loader = BrowserGameLoader(cfg)
        loader.start()
        assert loader.running

        with mock.patch("subprocess.run"):  # mock taskkill
            loader.stop()
        assert not loader.running
        assert loader.url is None

    @mock.patch("subprocess.Popen")
    @mock.patch.object(BrowserGameLoader, "is_ready", return_value=True)
    def test_context_manager(self, mock_ready, mock_popen, tmp_path: Path):
        """GameLoader works as a context manager."""
        mock_proc = mock.MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        cfg = _make_config(game_dir=str(tmp_path), install_command=None)

        with mock.patch("subprocess.run"):
            with BrowserGameLoader(cfg) as loader:
                assert loader.running
            assert not loader.running

    @mock.patch("subprocess.Popen")
    def test_start_raises_on_process_exit(self, mock_popen, tmp_path: Path):
        """start() raises if the server process exits immediately."""
        mock_proc = mock.MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = 1  # exited immediately
        mock_proc.returncode = 1
        mock_proc.stdout.read.return_value = ""
        mock_proc.stderr.read.return_value = "crash"
        mock_popen.return_value = mock_proc

        cfg = _make_config(
            game_dir=str(tmp_path),
            readiness_timeout_s=2.0,
            readiness_poll_interval_s=0.1,
        )
        loader = BrowserGameLoader(cfg)
        with pytest.raises(GameLoaderError, match="exited with code"):
            loader.start()

    @mock.patch("subprocess.Popen")
    @mock.patch.object(BrowserGameLoader, "is_ready", return_value=False)
    def test_start_raises_on_timeout(self, mock_ready, mock_popen, tmp_path: Path):
        """start() raises if readiness endpoint never responds."""
        mock_proc = mock.MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        cfg = _make_config(
            game_dir=str(tmp_path),
            readiness_timeout_s=0.5,
            readiness_poll_interval_s=0.1,
        )
        loader = BrowserGameLoader(cfg)
        with mock.patch("subprocess.run"):  # mock taskkill in stop()
            with pytest.raises(GameLoaderError, match="did not become ready"):
                loader.start()


# ── Breakout71Loader ────────────────────────────────────────────────


class TestBreakout71Loader:
    """Tests for the Breakout 71 specialisation."""

    def test_from_repo_path(self, tmp_path: Path):
        """from_repo_path creates a loader with correct defaults."""
        loader = Breakout71Loader.from_repo_path(tmp_path)
        assert loader.name == "breakout-71"
        assert loader.config.serve_port == 1234
        assert loader.config.game_dir == tmp_path
        assert "parcel" in loader.config.serve_command

    def test_from_repo_path_custom_port(self, tmp_path: Path):
        """from_repo_path respects a custom port."""
        loader = Breakout71Loader.from_repo_path(tmp_path, serve_port=3000)
        assert loader.config.serve_port == 3000
        assert "3000" in loader.config.url

    def test_setup_clears_parcel_cache(self, tmp_path: Path):
        """setup() removes .parcel-cache before installing."""
        cache_dir = tmp_path / ".parcel-cache"
        cache_dir.mkdir()
        (cache_dir / "dummy").write_text("x")

        loader = Breakout71Loader.from_repo_path(tmp_path)
        # Patch the parent setup to avoid actually running npm install
        with mock.patch.object(BrowserGameLoader, "setup"):
            loader.setup()

        assert not cache_dir.exists()

    def test_setup_works_without_cache_dir(self, tmp_path: Path):
        """setup() succeeds even if .parcel-cache doesn't exist."""
        loader = Breakout71Loader.from_repo_path(tmp_path)
        with mock.patch.object(BrowserGameLoader, "setup"):
            loader.setup()  # should not raise

    def test_window_title_default(self, tmp_path: Path):
        """Default window_title is set for Breakout."""
        loader = Breakout71Loader.from_repo_path(tmp_path)
        assert loader.config.window_title == "Breakout"


# ── Factory ─────────────────────────────────────────────────────────


class TestFactory:
    """Tests for the create_loader factory function."""

    def test_create_browser_loader(self):
        """loader_type='browser' creates a BrowserGameLoader."""
        cfg = _make_config(loader_type="browser")
        loader = create_loader(cfg)
        assert isinstance(loader, BrowserGameLoader)

    def test_create_breakout71_loader(self, tmp_path: Path):
        """loader_type='breakout-71' creates a Breakout71Loader."""
        cfg = _make_config(
            loader_type="breakout-71",
            game_dir=str(tmp_path),
        )
        loader = create_loader(cfg)
        assert isinstance(loader, Breakout71Loader)

    def test_unknown_loader_type_raises(self):
        """Unknown loader_type raises GameLoaderError."""
        cfg = _make_config(loader_type="alien-game-engine")
        with pytest.raises(GameLoaderError, match="Unknown loader_type"):
            create_loader(cfg)

    def test_register_custom_loader(self):
        """register_loader adds a custom loader type."""

        class DummyLoader(GameLoader):
            def setup(self):
                pass

            def start(self):
                pass

            def is_ready(self):
                return True

            def stop(self):
                pass

        register_loader("dummy", DummyLoader)
        cfg = _make_config(loader_type="dummy")
        loader = create_loader(cfg)
        assert isinstance(loader, DummyLoader)

    def test_end_to_end_yaml_to_loader(self, tmp_path: Path):
        """Full flow: YAML file → load_game_config → create_loader."""
        yaml_content = textwrap.dedent("""\
            name: e2e-game
            game_dir: .
            loader_type: browser
            serve_command: echo hi
            serve_port: 8080
            url: http://localhost:8080
        """)
        config_file = tmp_path / "e2e-game.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        cfg = load_game_config("e2e-game", configs_dir=tmp_path)
        loader = create_loader(cfg)
        assert isinstance(loader, BrowserGameLoader)
        assert loader.name == "e2e-game"
