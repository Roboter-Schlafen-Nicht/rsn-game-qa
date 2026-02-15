# Game Loader Spec

> Reference design for the game loader subsystem.  The game loader
> manages the full lifecycle of getting a game running and reachable
> so that the QA / RL layer can interact with it.

## Overview

Before any QA testing or RL training can happen, the game under test
must be running and reachable.  The game loader subsystem provides a
configurable, extensible mechanism for this:

1. **Install** dependencies (e.g. `npm install`)
2. **Serve** the game (e.g. start a Parcel dev server)
3. **Wait** until the game is ready (HTTP readiness polling)
4. **Expose** the URL for capture / input layers to connect to
5. **Tear down** the game process when testing is complete

The subsystem is designed to be game-agnostic.  Each game is described
by a declarative YAML configuration, and the loader class hierarchy
handles the mechanics.

## Architecture

```
                      ┌─────────────────┐
                      │ GameLoaderConfig │  ← YAML file (configs/games/*.yaml)
                      └────────┬────────┘
                               │
                      ┌────────▼────────┐
                      │   GameLoader    │  ← Abstract base class
                      │   (ABC)         │
                      └────────┬────────┘
                               │
                ┌──────────────┼──────────────┐
                │                             │
       ┌────────▼────────┐           ┌────────▼────────┐
       │ BrowserGame     │           │  (future)       │
       │ Loader          │           │ EmulatorLoader  │
       └────────┬────────┘           │ NativeLoader    │
                │                    └─────────────────┘
       ┌────────▼────────┐
       │ Breakout71      │
       │ Loader          │
       └─────────────────┘
```

The factory function `create_loader(config)` maps a config's
`loader_type` field to the correct class.

## Configuration

Each game is described by a `GameLoaderConfig` dataclass.  Configs can
be constructed directly in Python or loaded from YAML files stored in
`configs/games/`.

### YAML Schema

| Field                     | Type   | Default                                   | Description                                             |
|---------------------------|--------|-------------------------------------------|---------------------------------------------------------|
| `name`                    | str    | *(required)*                              | Human-readable game identifier                          |
| `game_dir`                | str    | *(required)*                              | Path to the game's source repository                    |
| `loader_type`             | str    | `"browser"`                               | Loader class to use (`"browser"`, `"breakout-71"`, ...) |
| `install_command`         | str    | `"npm install"`                           | Shell command to install dependencies                   |
| `serve_command`           | str    | `"npx parcel src/index.html --no-cache"`  | Shell command to start the dev server                   |
| `serve_port`              | int    | `1234`                                    | Port the dev server listens on                          |
| `url`                     | str    | `"http://localhost:1234"`                 | Full URL the game is reachable at                       |
| `readiness_endpoint`      | str    | *(defaults to `url`)*                     | URL to poll for readiness                               |
| `readiness_timeout_s`     | float  | `120.0`                                   | Max seconds to wait for readiness                       |
| `readiness_poll_interval_s` | float | `2.0`                                    | Seconds between readiness polls                         |
| `window_title`            | str    | `None`                                    | Browser window title (for `WindowCapture`)              |
| `env_vars`                | dict   | `{}`                                      | Extra environment variables for the serve process       |

### Example YAML (`configs/games/breakout-71.yaml`)

```yaml
name: breakout-71
game_dir: F:\work\breakout71-testbed
loader_type: breakout-71

install_command: npm install
serve_command: npx parcel src/index.html --no-cache
serve_port: 1234
url: http://localhost:1234
readiness_endpoint: http://localhost:1234

readiness_timeout_s: 120.0
readiness_poll_interval_s: 2.0

window_title: Breakout
```

### Loading a Config

```python
from src.game_loader import load_game_config

config = load_game_config("breakout-71")
# Searches configs/games/breakout-71.yaml
```

A custom search directory can be provided:

```python
config = load_game_config("my-game", configs_dir="/path/to/configs")
```

## GameLoader Lifecycle

The base `GameLoader` defines four lifecycle methods:

| Method       | Purpose                                          |
|--------------|--------------------------------------------------|
| `setup()`    | One-time preparation (install deps, clear caches) |
| `start()`    | Launch the game and block until ready             |
| `is_ready()` | Check if the game is responding                   |
| `stop()`     | Tear down the game process and free resources      |

### Context Manager

`GameLoader` implements `__enter__` / `__exit__` for convenient
lifecycle management:

```python
from src.game_loader import load_game_config, create_loader

config = load_game_config("breakout-71")
with create_loader(config) as loader:
    print(f"Game ready at {loader.url}")
    # … run QA / RL …
# loader.stop() called automatically
```

### Properties

| Property   | Type         | Description                                      |
|------------|--------------|--------------------------------------------------|
| `name`     | `str`        | Human-readable game identifier                    |
| `url`      | `str | None` | URL when running, `None` when stopped             |
| `running`  | `bool`       | Whether the game process is currently active      |

## BrowserGameLoader

Handles browser-based games served by a local dev server (Parcel, Vite,
Webpack Dev Server, etc.).

### setup()

1. Validate that `game_dir` exists
2. Run `install_command` via `subprocess.run()` in `game_dir`
3. Raise `GameLoaderError` if the command fails

Skipped if `install_command` is `None` or empty.

### start()

1. Spawn `serve_command` as a background subprocess via `subprocess.Popen`
2. On Windows: use `CREATE_NEW_PROCESS_GROUP` for clean process-tree
   termination
3. On POSIX: use `start_new_session=True`
4. Poll `readiness_endpoint` with HTTP GET every
   `readiness_poll_interval_s` seconds
5. If the server responds with HTTP < 400: mark as ready, set
   `_running = True`
6. If the process exits before becoming ready: raise `GameLoaderError`
7. If the timeout expires: kill the process and raise `GameLoaderError`

### is_ready()

Sends an HTTP GET to `readiness_endpoint` with a 5-second timeout.
Returns `True` if the response status is < 400.

### stop()

1. On Windows: `taskkill /F /T /PID <pid>` to kill the whole process tree
2. On POSIX: `os.killpg()` with `SIGTERM`
3. Wait up to 10 seconds for graceful shutdown, then force-kill
4. Set `_running = False`

Safe to call even if the game was never started.

## Breakout71Loader

Thin specialisation of `BrowserGameLoader` with Breakout-71-specific
defaults.

### Defaults

| Field            | Value                                    |
|------------------|------------------------------------------|
| `name`           | `"breakout-71"`                          |
| `install_command`| `"npm install"`                          |
| `serve_command`  | `"npx parcel src/index.html --no-cache"` |
| `serve_port`     | `1234`                                   |
| `url`            | `"http://localhost:1234"`                |
| `window_title`   | `"Breakout"`                             |

### Convenience Constructor

```python
from src.game_loader import Breakout71Loader

loader = Breakout71Loader.from_repo_path(
    r"F:\work\breakout71-testbed",
    serve_port=1234,           # optional
    readiness_timeout_s=120.0, # optional
    window_title="Breakout",   # optional
)
```

### setup() Override

Before calling the parent `setup()`, clears the `.parcel-cache`
directory if it exists. This prevents stale Parcel caches from
causing build failures.

## Factory

The `create_loader(config)` factory maps `loader_type` to the
correct class:

| `loader_type`  | Class               |
|----------------|---------------------|
| `"browser"`    | `BrowserGameLoader` |
| `"breakout-71"`| `Breakout71Loader`  |

### Custom Loader Registration

```python
from src.game_loader.factory import register_loader

class MyGameLoader(GameLoader):
    # …

register_loader("my-engine", MyGameLoader)
```

After registration, `create_loader(config)` will use `MyGameLoader`
for configs with `loader_type: my-engine`.

## Adding a New Game

1. **Create a YAML config** in `configs/games/<name>.yaml`
2. Set `loader_type: browser` if the game is a browser game served by
   a Node dev server — no code needed
3. If custom logic is required (e.g. emulator startup, native
   executable), subclass `GameLoader` or `BrowserGameLoader` and
   register it with `register_loader()`

### Example: Adding a Vite-based Game

```yaml
# configs/games/my-vite-game.yaml
name: my-vite-game
game_dir: /path/to/repo
loader_type: browser
install_command: npm install
serve_command: npx vite --port 5173
serve_port: 5173
url: http://localhost:5173
window_title: My Game
```

No Python code needed — `BrowserGameLoader` handles Vite the same way
it handles Parcel.

## Error Handling

| Scenario                     | Behaviour                                          |
|------------------------------|----------------------------------------------------|
| `game_dir` does not exist    | `GameLoaderError` raised in `setup()` or `start()` |
| Install command fails        | `GameLoaderError` with stdout/stderr tail           |
| Server process exits early   | `GameLoaderError` with exit code and output         |
| Readiness timeout expires    | Process killed, `GameLoaderError` raised            |
| Unknown `loader_type`        | `GameLoaderError` from `create_loader()`            |
| `stop()` before `start()`   | No-op, safe to call                                 |

## Integration with Existing Subsystems

The game loader is the entry point for the QA pipeline.  Once the
loader reports ready:

- `WindowCapture` can locate the game window via `config.window_title`
- `InputController` can inject actions into the focused window
- `Breakout71Env` can be constructed with the known window title and URL

```
GameLoader.start()
    │
    ▼
WindowCapture(window_title=config.window_title)
    │
    ▼
Breakout71Env / RL agent / oracles
    │
    ▼
GameLoader.stop()
```

## Source Files

- `src/game_loader/__init__.py` — Package init and public API
- `src/game_loader/config.py` — `GameLoaderConfig` dataclass and `load_game_config()`
- `src/game_loader/base.py` — `GameLoader` abstract base class
- `src/game_loader/browser_loader.py` — `BrowserGameLoader`
- `src/game_loader/breakout71_loader.py` — `Breakout71Loader`
- `src/game_loader/factory.py` — `create_loader()` and `register_loader()`
- `configs/games/breakout-71.yaml` — Breakout 71 configuration
- `tests/test_game_loader.py` — Test suite (36 tests)
