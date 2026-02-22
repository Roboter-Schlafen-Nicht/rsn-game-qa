# RSN Game QA

[![CI](https://github.com/Roboter-Schlafen-Nicht/rsn-game-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/Roboter-Schlafen-Nicht/rsn-game-qa/actions/workflows/ci.yml)
![Version](https://img.shields.io/badge/version-0.1.0a1-blue)

RL-driven autonomous game testing platform. Reinforcement learning agents
play games from pixels, explore bugs and balance issues, and generate
human-readable QA reports.

**RSN** = *Roboter Schlafen Nicht* ("Robots Don't Sleep")

---

## About This Project

RSN Game QA is a case study in autonomous software engineering. An AI
agent serves as project lead -- making architectural decisions, writing
code, managing CI, and driving the roadmap -- while the human founder
provides strategic direction and domain expertise.

The project demonstrates what this workflow produces:

- **6-8x calendar compression** vs a traditional engineering team
- **142+ PRs merged** with automated Copilot code review, zero review
  bottleneck
- **95%+ test coverage** not from a coverage sprint, but built into the
  workflow from day one
- **Thousands of lines of documentation** generated as a natural byproduct
  of working, not a separate effort
- **8 subsystems, 3 game plugins, full CI/CD** -- built across 62 sessions

The codebase is real, the tests pass, the CI is green. The methodology
behind it is the point.

### How It Was Built

An AI agent ([OpenCode](https://opencode.ai)) handles execution:
architecture, implementation, testing, PR creation, code review response,
and roadmap prioritization. The human founder sets goals, makes business
decisions, and provides course corrections. Structured documentation
(`AGENTS.md`, progress tracking, knowledge bases) gives the agent
continuity across sessions.

See [`documentation/ROADMAP.md`](documentation/ROADMAP.md) for the
development history and current phase.

---

## What It Does

The platform uses a two-layer approach to find bugs that human testers miss:

1. **RL agents** play games autonomously from raw pixels -- no source code
   access, no SDK integration, no game engine plugins required
2. **12 bug-detection oracles** watch every frame and flag anomalies:
   crashes, stuck states, score anomalies, visual glitches, performance
   drops, physics violations, boundary violations, invalid state
   transitions, episode length anomalies, temporal anomalies, reward
   inconsistencies, and soak test degradation

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

### Why Pixel-Based

- **Zero integration burden** -- no SDK, no engine plugin, no source code
- **Engine-agnostic** -- works on Unity, Unreal, Godot, HTML5, native, mobile
- **Tests the real player experience** -- sees exactly what a player sees
- **Scales to closed-source games** -- test titles you don't have source for

## Supported Games

| Game | Genre | Action Space | Interface | Status |
|------|-------|-------------|-----------|--------|
| **Breakout 71** | Brick-breaking arcade | Continuous paddle position | Headless Selenium | Complete |
| **Hextris** | Hexagonal puzzle | Discrete rotation (3 actions) | Headless Selenium | Complete |
| **shapez.io** | Factory builder | Mouse + keyboard (MultiDiscrete) | Headless Selenium | Complete |

New games are added as plugins in `games/` with zero changes to the
platform code. See [`documentation/ROADMAP.md`](documentation/ROADMAP.md)
for development history and training results.

## Project Structure

```
src/
  platform/             BaseGameEnv ABC, CnnObservationWrapper (game-agnostic)
  capture/              Window capture (GDI) + input controller (pydirectinput)
  env/                  Backward-compat re-exports for game environments
  game_loader/          Configurable game loading (browser dev server lifecycle)
  oracles/              12 bug-detection oracles
  orchestrator/         Session orchestration (FrameCollector, SessionRunner)
  perception/           YOLO detector wrapper + game-specific capture helpers
  reporting/            Episode/session reports + Jinja2 HTML dashboard
games/
  breakout71/           Breakout 71 game plugin (env, loader, modal handler)
  hextris/              Hextris game plugin (env, loader, modal handler)
  shapez/               shapez.io game plugin (env, loader, modal handler)
configs/
  games/                Game loader YAML configs
  training/             YOLO training configs per game
scripts/                CLI tools (train_rl, run_session, capture, analyze, etc.)
tests/                  pytest suite
docs/                   Sphinx docs (Furo theme, MyST Markdown)
documentation/
  specs/                Design specs (env, oracles, capture, reporting, loader)
  ROADMAP.md            Development plan and training results
```

## AI Agent System

This project uses [OpenCode](https://opencode.ai) for autonomous
development. The build agent follows `AGENTS.md` for project conventions.

## Hardware

- AMD Ryzen 9 5900X
- 2x Intel Arc A770 GPU (PyTorch XPU backend via oneAPI/IPEX)

## Setup

```bash
# Create conda environment (Python 3.12)
conda env create -f environment.yml
conda activate yolo

# Set environment variables
export PYTHONPATH=/path/to/rsn-game-qa
export BREAKOUT71_DIR=/path/to/breakout71-testbed
```

Platform-specific dependencies (`pywin32`, `pydirectinput`) are guarded with
`try/except ImportError` so the codebase loads on Linux/WSL2 as well.

## Running

### RL Training

```bash
# Train a PPO agent on Breakout 71 (headless, CNN policy)
python scripts/train_rl.py --game breakout71 --total-timesteps 200000 --headless

# Train with MLP policy (requires YOLO model)
python scripts/train_rl.py --game breakout71 --total-timesteps 200000 --headless --policy mlp
```

### Evaluation

```bash
# Run 10-episode evaluation with a trained model
python scripts/run_session.py --game breakout71 --model output/models/ppo_breakout71_final.zip --episodes 10 --headless

# Random baseline (no model)
python scripts/run_session.py --game breakout71 --episodes 10 --headless
```

### Game Loading

```python
from src.game_loader import load_game_config, create_loader

config = load_game_config("breakout-71")
with create_loader(config) as loader:
    print(f"Game ready at {loader.url}")
```

To add a new browser game, drop a YAML file in `configs/games/` -- no Python
code needed if it follows the standard Node dev server pattern.

## CI

GitHub Actions runs on every push to `main` and on PRs:

| Job | What it does |
|-----|-------------|
| **Lint** | `ruff check` + `ruff format --check` |
| **Test** | `pytest -m "not integration"` with 80% coverage threshold |
| **Build Check** | Verifies all module imports succeed |
| **Build Docs** | Sphinx HTML build with `-W` (warnings as errors) |

Automated Copilot code review is enabled via GitHub Ruleset on all PRs.

## Tests

```bash
# Unit tests (run in CI, no game required)
python -m pytest tests/ -v

# Integration tests (requires live game instance)
python -m pytest tests/ -m integration -v
```

## YOLO Training Pipeline

Config-driven pipeline for training game-specific YOLO object detection models:

```bash
# 1. Capture frames from a running game
python scripts/capture_dataset.py --frames 500

# 2. Auto-annotate using color segmentation + frame differencing
python scripts/auto_annotate.py output/dataset_<timestamp>

# 3. (Optional) Upload to Roboflow for manual review/correction
python scripts/upload_to_roboflow.py output/dataset_<timestamp>

# 4. Train
python scripts/train_model.py --config breakout-71

# 5. Validate
python scripts/validate_model.py --config breakout-71 --save-samples 10
```

## Documentation

```bash
# Build Sphinx docs
python -m sphinx.cmd.build -b html docs docs/_build/html
```

Design specs in `documentation/specs/` cover the env, oracle system,
capture/input, reporting, and game loader subsystems.

## License

MIT License. See [LICENSE](LICENSE) for details.
