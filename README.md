# rsn-game-qa

[![CI](https://github.com/Roboter-Schlafen-Nicht/rsn-game-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/Roboter-Schlafen-Nicht/rsn-game-qa/actions/workflows/ci.yml)
![Version](https://img.shields.io/badge/version-0.1.0a1-blue)

RL-driven autonomous game testing platform. YOLO object detection + reinforcement learning agents play games from pixels, explore bugs and balance issues, and generate human-readable test reports.

**RSN** = *Roboter Schlafen Nicht* ("Robots Don't Sleep")

## Architecture

The system uses a two-layer design:

1. **RL agents** play games and produce rich play traces (trajectories)
2. **Bug-detection oracles** inspect traces and flag anomalies (crashes, stuck states, score anomalies, visual glitches, performance drops, physics violations, boundary violations, state transitions, episode length anomalies, temporal anomalies, reward inconsistencies, soak test degradation)

```
 Game Loader      Capture          Perception        Policy / RL       Oracles           Reporting
+------------+   +-----------+    +------------+    +-----------+    +-------------+    +-----------+
| Config     |   | Window    |--->| YOLO       |--->| PPO (SB3) |    | Crash       |    | Episode   |
| (YAML)     |-->| Capture   |    | Detector   |    |           |--->| Stuck       |--->| Report    |
| Browser    |   |           |    |            |    |           |    | ScoreAnom.  |    | Dashboard |
| Loader     |   +-----------+    +------------+    +-----------+    | VisGlitch   |    +-----------+
+------------+                                                       | Performance |
                                                                     | Physics     |
                                                                     | Boundary    |
                                                                     | StateTransn |
                                                                     | EpisodeLen  |
                                                                     | TemporalAn. |
                                                                     | RewardCons. |
                                                                     | Soak        |
                                                                     +-------------+
```

## Target games

| Game | Interface | Status |
|------|-----------|--------|
| **Breakout 71** (browser) | Windows native window (GDI) | In progress -- env v1, capture, perception, oracles implemented |

## Project structure

```
src/
  capture/              Window capture (GDI) + input controller (pydirectinput)
  env/                  Gymnasium environments (Breakout71Env)
  game_loader/          Configurable game loading (browser dev server lifecycle)
  oracles/              Bug-detection oracles (crash, stuck, score, visual, perf, physics, boundary, state, episode_length, temporal, reward, soak)
  orchestrator/         Session orchestration (FrameCollector, SessionRunner)
  perception/           YOLO detector wrapper + Breakout 71 capture helpers
  reporting/            Episode/session reports + Jinja2 HTML dashboard
configs/
  games/                Game loader YAML configs (breakout-71.yaml, ...)
  training/             YOLO training configs per game (breakout-71.yaml, ...)
scripts/                YOLO training, dataset capture, upload, smoke tests, RL training
tests/                  pytest suite (649 unit + 24 integration tests)
docs/                   Sphinx docs (Furo theme, MyST Markdown)
documentation/
  specs/                Design specs for env, oracles, capture, reporting, game loader
  sessions/             Development session notes
  business/             2026 plan
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

## CI

GitHub Actions runs on every push to `main` or `big-rock-*` branches and on PRs to `main`:

| Job | What it does |
|-----|-------------|
| **Lint** | `ruff check` + `ruff format --check` |
| **Test** | `pytest -m "not integration"` (649 passed) |
| **Build Check** | Verifies all module imports succeed |
| **Build Docs** | Sphinx HTML build with `-W` (warnings as errors) |

A pre-commit hook runs the full CI pipeline locally via [`act`](https://github.com/nektos/act) before each commit.

## Tests

```bash
# Unit tests (run in CI, no game required)
python -m pytest tests/ -v

# Integration tests (requires live Breakout 71 on Windows)
python -m pytest tests/ -m integration -v
```

## Smoke scripts

Standalone scripts for manual progress verification. Run from the project root
on Windows with the Breakout 71 testbed available:

| Script | What it does |
|--------|-------------|
| `scripts/smoke_launch.py` | Start game, verify readiness, capture proof screenshot, wait, shut down |
| `scripts/smoke_capture.py` | Capture N frames at configurable interval, save PNGs, log latency/FPS |
| `scripts/smoke_oracle.py` | Run oracles against live frames for M steps, save JSON report + findings |

```bash
# Quick launch test (30s wait, skip npm install)
python scripts/smoke_launch.py --wait 30 --skip-setup

# Capture 20 frames at 0.5s intervals
python scripts/smoke_capture.py --frames 20 --interval 0.5 --skip-setup

# Run oracles for 100 steps
python scripts/smoke_oracle.py --steps 100 --skip-setup
```

Artifacts are saved to `output/` (gitignored).

## YOLO training pipeline

Config-driven pipeline for training game-specific YOLO object detection models.
Each game gets its own training config in `configs/training/`.

```bash
# 1. Capture ~500 frames from a running game (random bot plays)
python scripts/capture_dataset.py --frames 500

# 2. Upload frames to Roboflow for annotation
python scripts/upload_to_roboflow.py output/dataset_<timestamp>

# 3. Annotate images in Roboflow UI (5 classes for Breakout 71)
#    Then export in YOLOv8 format and set dataset_path in configs/training/breakout-71.yaml

# 4. Train the model
python scripts/train_model.py --config breakout-71

# 5. Validate against quality thresholds
python scripts/validate_model.py --config breakout-71 --save-samples 10
```

API keys are stored in `.env` (gitignored). Copy `.env.example` to `.env` and fill in your values.

## Docs

Build the Sphinx documentation:

```bash
python -m sphinx.cmd.build -b html docs docs/_build/html
```

Design specs live in `documentation/specs/` and cover the Breakout71 env, oracle system, capture/input, reporting system, and game loader.

## License

Proprietary. All rights reserved.
