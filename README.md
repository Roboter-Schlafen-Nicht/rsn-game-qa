# rsn-game-qa

RL-driven autonomous game testing platform. YOLO object detection + reinforcement learning agents play games from pixels, explore bugs and balance issues, and generate human-readable test reports.

**RSN** = *Roboter Schlafen Nicht* ("Robots Don't Sleep")

## Architecture

The system uses a two-layer design:

1. **RL agents** play games and produce rich play traces (trajectories)
2. **Bug-detection oracles** inspect traces and flag anomalies (crashes, stuck states, score anomalies, visual glitches, performance drops)

```
 Game Loader      Capture          Perception        Policy / RL         Oracles           Reporting
+------------+   +-----------+    +------------+    +-------------+    +-------------+    +-----------+
| Config     |   | Window    |--->| YOLO       |--->| Help Policy |    | Crash       |    | Episode   |
| (YAML)     |-->| Capture   |    | Detector   |    | RL Policy   |--->| Stuck       |--->| Report    |
| Browser    |   | ADB       |    |            |    | PPO (SB3)   |    | ScoreAnom.  |    | Dashboard |
| Loader     |   +-----------+    +------------+    +-------------+    | VisGlitch   |    +-----------+
+------------+                                                         | Performance |
                                                                       +-------------+
```

## Target games

| Game | Interface | Status |
|------|-----------|--------|
| **Last War** (mobile) | Android emulator via ADB | Working -- YOLO detection + RL help-button policy + live bot loop |
| **Breakout 71** (browser) | Windows native window (GDI) | Stubbed -- env, capture, and perception interfaces defined |

## Project structure

```
src/
  agents/               Bot runners (help_bot, breakout controller)
  capture/              Window capture (GDI) + input controller (pydirectinput) [stub]
  controllers/          ADB emulator controller, game-specific controllers
  env/                  Gymnasium environments (Breakout71Env [stub])
  game_loader/          Configurable game loading (browser dev server lifecycle)
  oracles/              Bug-detection oracles (crash, stuck, score, visual, perf) [stub on_step]
  perception/           YOLO detector wrapper [stub]
  policies/             YOLO-based + RL-based policies for Last War
  reporting/            Episode/session reports + Jinja2 dashboard [stub]
  rl/                   RL training env + PPO training script for help policy
configs/
  games/                Game loader YAML configs (breakout-71.yaml, ...)
scripts/                YOLO training, dataset dedup, ADB test
tests/                  pytest suite (102 tests)
docs/                   Sphinx docs (Furo theme, MyST Markdown)
documentation/
  specs/                Design specs for env, oracles, capture, reporting, game loader
  sessions/             Development session notes
  business/             2026 plan
data/
  twin_dataset/         YOLO training images + labels
  live_logs/            Bot run logs (frames + JSON metadata)
runs/
  rl_help_v3/           Trained PPO model (ppo_lastwar_help.zip)
```

## Hardware

- AMD Ryzen 9 5900X
- 2x Intel Arc A770 GPU (PyTorch XPU backend via oneAPI/IPEX)

## Environment setup

The conda environment is named `yolo` and uses Python 3.12:

```bash
conda env create -f environment.yml
conda activate yolo
```

`environment.yml` installs PyTorch XPU wheels from `https://download.pytorch.org/whl/xpu` and uses PyPI as an extra index for non-torch packages.

Platform-specific dependencies (`pywin32`, `pydirectinput`) are guarded with `try/except ImportError` so the codebase loads on Linux as well.

## Running

### Load a game for testing

The game loader manages the lifecycle of getting a game running before QA or RL begins. Games are configured via YAML files in `configs/games/`.

```python
from src.game_loader import load_game_config, create_loader

config = load_game_config("breakout-71")
with create_loader(config) as loader:
    print(f"Game ready at {loader.url}")
    # … run QA / RL against the game …
```

Or use the Breakout 71 convenience constructor directly:

```python
from src.game_loader import Breakout71Loader

loader = Breakout71Loader.from_repo_path(r"F:\work\breakout71-testbed")
loader.setup()   # npm install
loader.start()   # parcel dev server → http://localhost:1234
# … game is running …
loader.stop()
```

To add a new browser game, drop a YAML file in `configs/games/` — no Python code needed if it follows the standard Node dev server pattern (npm install + serve command).

### Last War help bot (requires Android emulator + ADB)

```bash
python src/agents/run_help_bot.py
```

Captures frames via ADB, runs YOLO inference, applies RL shadow decisions, and logs events with screenshots + JSON metadata.

### Breakout 71 random controller (requires Android emulator + ADB)

```bash
python src/agents/run_breakout_controller.py
```

### Train the RL help policy

```bash
python src/rl/train_help.py
```

Trains a PPO agent in the simulated `LastWarHelpEnv` to decide NOOP / CLICK_HELP / RANDOM_SWIPE.

### Build YOLO training dataset from logs

```bash
python src/build_twin_dataset.py
```

## CI

GitHub Actions runs on every push to `main` or `big-rock-*` branches and on PRs to `main`:

| Job | What it does |
|-----|-------------|
| **Lint** | `ruff check` + `ruff format --check` |
| **Test** | `pytest` (102 passed) |
| **Build Check** | Verifies all module imports succeed |
| **Build Docs** | Sphinx HTML build with `-W` (warnings as errors) |

A pre-commit hook runs the full CI pipeline locally via [`act`](https://github.com/nektos/act) before each commit.

## Tests

```bash
python -m pytest tests/ -v
```

## Docs

Build the Sphinx documentation:

```bash
python -m sphinx.cmd.build -b html docs docs/_build/html
```

Design specs live in `documentation/specs/` and cover the Breakout71 env, oracle system, capture/input, reporting system, and game loader.

## License

Proprietary. All rights reserved.
