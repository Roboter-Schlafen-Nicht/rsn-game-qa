# RSN Game QA

[![CI](https://github.com/Roboter-Schlafen-Nicht/rsn-game-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/Roboter-Schlafen-Nicht/rsn-game-qa/actions/workflows/ci.yml)
![Version](https://img.shields.io/badge/version-0.1.0a1-blue)
![Tests](https://img.shields.io/badge/tests-750%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-96%25-brightgreen)

RL-driven autonomous game testing platform. Reinforcement learning agents
play games from pixels, explore bugs and balance issues, and generate
human-readable QA reports.

**RSN** = *Roboter Schlafen Nicht* ("Robots Don't Sleep")

---

## About Roboter Schlafen Nicht

Roboter Schlafen Nicht is an **AI-first company** exploring how humans and
AI can work together using proven best practices. RSN Game QA is our first
collaboration -- a project where an AI agent (OpenCode, powered by
Claude) serves as **project lead**, making architectural decisions, writing
code, running CI, and managing the development workflow, while the human
founder provides direction, reviews, and domain expertise.

This isn't a demo or a toy. The entire codebase -- 750+ tests, 96% coverage,
8 subsystems, 74 merged PRs -- was built through this human-AI collaboration.
The agent commits code, creates pull requests, responds to automated code
review, and drives the roadmap forward. The human sets goals, makes business
decisions, and keeps the agent honest.

We believe the best software will be built by teams where humans and AI each
contribute what they do best. This project is how we're proving it.

### Project Lead

This project is led by **rsn-opencode**, an AI agent running on
[OpenCode](https://opencode.ai) (Claude claude-opus-4.6). The agent handles:

- Architecture and implementation decisions
- All code authoring, testing, and CI management
- Pull request creation and code review response
- Roadmap execution and task prioritization
- Session continuity via `AGENTS.md` and structured documentation

The human founder provides strategic direction, business development, and
final approval on key decisions.

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

## Current Status

**Phase 1: First Real Training & QA Report** (in progress)

| Milestone | Status |
|-----------|--------|
| Platform architecture (8 subsystems) | Done |
| Game plugin system with `--game` flag | Done |
| CNN/MLP observation modes | Done |
| Headless training on WSL2 | Done |
| 200K-step PPO training run | In progress |
| 10-episode evaluation + QA report | Next |
| Random baseline comparison | Next |

See [`documentation/ROADMAP.md`](documentation/ROADMAP.md) for the full
5-phase plan.

## Target Games

| Game | Interface | Status |
|------|-----------|--------|
| **Breakout 71** (browser, TypeScript canvas) | Headless Selenium | Env complete, training in progress |

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
configs/
  games/                Game loader YAML configs
  training/             YOLO training configs per game
scripts/                CLI tools (train_rl, run_session, capture, debug, etc.)
tests/                  pytest suite (750+ tests, 96% coverage)
docs/                   Sphinx docs (Furo theme, MyST Markdown)
documentation/
  specs/                Design specs (env, oracles, capture, reporting, loader)
  ROADMAP.md            5-phase development plan
.opencode/
  agents/               AI agent configurations (build, business, strategy)
  commands/             Custom slash commands for agent workflows
```

## AI Agent System

This project uses [OpenCode](https://opencode.ai) with multiple specialized
AI agents, each with a defined role and access to specific tools. Agent
configurations are committed to `.opencode/agents/` so they're available
on any clone.

| Agent | Role | Invocation |
|-------|------|------------|
| **Build** (default) | Architecture, code, tests, CI, PRs | Default agent |
| **Business** | Client outreach, proposals, CRM | `@business` |
| **Sales Research** | Analyze potential clients and game fit | `@sales-research` |
| **Strategy** | Go-to-market, financial planning, prioritization | `@strategy` |

The Build agent follows `AGENTS.md` for project conventions. Business agents
store persistent context in `private/` (gitignored, local-only).

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

# 4. (Optional) Upload to Roboflow for manual review/correction
python scripts/upload_to_roboflow.py output/dataset_<timestamp>

# 5. Train
python scripts/train_model.py --config breakout-71

# 6. Validate
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
