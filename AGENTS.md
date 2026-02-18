# RSN Game QA — Agent Instructions

RL-driven autonomous game testing platform. First target: **Breakout 71**
(browser-based TypeScript canvas game; local clone configured via
`$BREAKOUT71_DIR` or `configs/games/breakout-71.yaml`).

## Role & Workflow

- **Agent leads the project** — execute, don't ask for permission. Read
  `documentation/ROADMAP.md` for the current phase, pick the next task,
  do it.
- Work on feature branches from `main` (pattern: `feature/...`, `docs/...`)
- Implement -> commit -> push -> create PR -> **Copilot reviews automatically**
  (via GitHub Ruleset) -> evaluate review, create issues if needed -> merge
  with `--delete-branch` -> delete local branch -> **post-merge admin**
  (update checklist, create session log, update AGENTS.md if needed)
- **Post-merge admin is checklist-only** — no Copilot review needed
- Pre-commit hook runs CI via `act` (Docker). **Takes 5+ minutes.** Use
  `timeout` of 600000ms for commit commands; check `git log` afterward.

## Autonomous Operation

The agent operates autonomously without human supervision. After context
compaction, the agent continues by re-reading this file and the roadmap.

### Continuation after compaction

OpenCode auto-compacts when context is full. After compaction:

1. Re-read `AGENTS.md` (this file) for instructions and current state
2. Re-read `documentation/ROADMAP.md` for the current phase
3. Check `private/documentation/BigRocks/checklist.md` for task status
4. Run `git status`, `git log --oneline -5`, `gh pr list` to orient
5. Check for running training processes: `ps aux | grep train_rl`
6. Resume the next incomplete task from the roadmap

### Session continuity

To continue the last session after restart:
```
opencode -c
```

To continue a specific session:
```
opencode -s <session-id>
```

### Decision authority

The agent has full authority to:
- Create branches, PRs, merge PRs (after CI passes)
- Fix bugs, refactor code, add features per the roadmap
- Launch and monitor training runs
- Generate evaluation reports and dashboards
- Update documentation, checklist, and this file
- Create GitHub issues for problems that need human attention

The agent should NOT:
- Force-push to main
- Delete data or checkpoints without clear reason
- Change business strategy or pricing (use @business agent)
- Commit files from `private/` to git

### Training runs

All training uses headless mode on WSL2. Artifacts go to `/mnt/e/rsn-game-qa/`.
```bash
PYTHONPATH=/mnt/f/work/rsn-game-qa \
BREAKOUT71_DIR=/mnt/f/work/breakout71-testbed \
nohup /home/human/miniconda3/envs/yolo/bin/python scripts/train_rl.py \
  --game breakout71 --headless --policy mlp --timesteps 200000 \
  --orientation portrait --resume <checkpoint> \
  > /mnt/e/rsn-game-qa/training_logs/train_run_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Current status (updated by agent)

- **Phase:** 1 (First Real Training & QA Report)
- **Training:** 100K/200K steps complete, checkpoint at
  `output/checkpoints/ppo_breakout71_100001.zip`
- **Crashes:** 2 (Chrome OOM at 57K, JS alert at 115K — both fixed)
- **PR #79:** Alert handling fix — CI green, ready to merge
- **Next:** Merge PR #79, resume training from 100K, run to 200K,
  evaluate, generate QA report

## Conventions

- Python 3.12, conda env `yolo`
- NumPy-style docstrings, ruff for lint/format, pytest for tests, Sphinx
  (Furo theme) for docs
- Use conda env `yolo` tools directly (e.g. `ruff` on PATH, or the
  full conda env path if `conda run` times out)
- CI has 4 jobs: Lint, Test, Build Check, Build Docs
- Always run `ruff format` before committing
- Commit messages: imperative mood with type prefix (`feat:`, `fix:`, `ci:`,
  `docs:`)
- Delete feature branches after merging

## TDD Convention

**Applies to:** `src/env/`, `src/orchestrator/`, `src/platform/`, and any
module where the behavioral contract matters more than implementation.

| Change type | TDD required? |
|---|---|
| Public method or behavioral change to `step()`, `reset()`, `close()` | **Yes** |
| New game state transition | **Yes** |
| Reward function changes | **Yes** |
| Observation field changes | **Yes** |
| Orchestrator lifecycle behavior | **Yes** |
| Internal refactoring (same behavior) | No — existing tests must pass |
| Scripts, utilities, exploratory work | No |

**Process:** Write test names as specs (with `pytest.skip`) -> fill bodies
(red) -> implement (green) -> refactor. Naming:
`test_{method}_{expected_behavior}_when_{condition}`.

## Current State

- **Tests pass**, 96% coverage, 8 subsystems complete
- **Architecture done:** BaseGameEnv ABC, game plugin system (`games/`),
  `--game` flag, CNN/MLP observation modes, dynamic plugin loading
- **Phase 1 in progress** — 100K/200K training steps done, 2 crash bugs
  fixed (Chrome OOM, JS alert), resuming training from checkpoint
- **Human is OOO Feb 18–20, 2026** — agent operates fully autonomously
- See `documentation/ROADMAP.md` for the 5-phase plan
- See `private/documentation/BigRocks/checklist.md` for detailed task tracking

Note: Paths under `private/` refer to local, gitignored documentation
and are not expected to resolve in a clean public checkout or on GitHub.

## Critical Technical Pitfalls

These cause bugs if forgotten. Full knowledge base at
`private/documentation/reference/agent_knowledge_base.md`.

1. **`import cv2` at top level breaks CI** — Docker lacks `libGL.so.1`;
   use lazy imports inside methods
2. **Copilot review is automatic** — GitHub Ruleset triggers review on
   PR creation; no manual reviewer assignment needed
3. **`git pull` times out** — use
   `git fetch origin main && git reset --hard origin/main`
4. **`gh pr merge` times out** but often succeeds — verify with
   `gh pr view N --json state,mergedAt`
5. **Selenium `execute_script()` IIFE** — must use
   `return (function() { ... })();` for return value
6. **Game `mouseup` pauses gameplay** — click once to start, then
   `moveTo` only
7. **`_no_ball_count` ownership** — `_check_late_game_over()` exclusively
   owns updates; `check_termination()` only reads
8. **Episode boundary** — `step()` passes `dismiss_game_over=False`;
   `reset()` passes `True`
9. **Terminal penalty** — game-over uses fixed `-5.01`, not YOLO-computed
   reward (modal occludes game state)
10. **OpenVINO GPU routing** — requires `device="intel:GPU.0"` prefix;
    without it defaults to CPU
11. **`pip install ultralytics` overwrites XPU torch** — must reinstall
    XPU torch after
12. **Pixel-based only** — no JS injection for game state. Exception:
    Selenium JS for modal handling and one-time settings (mute)
13. **Keep BOTH `.pt` and OpenVINO models** — `.pt` is source of truth
14. **CNN is default observation mode** — game-agnostic, no YOLO needed.
    MLP is optional (requires game-specific YOLO model)
15. **wincam singleton** — only 1 DXCamera at a time; no CI/headless support

## Project Structure

```
src/
  platform/       # BaseGameEnv ABC, CnnObservationWrapper (game-agnostic)
  game_loader/    # YAML config, factory, loaders
  reporting/      # JSON reports, HTML dashboard
  capture/        # BitBlt window capture, wincam, pydirectinput input
  perception/     # YoloDetector, breakout_capture
  oracles/        # 12 oracles with on_step detection
  env/            # Backward-compat re-exports (actual envs in games/)
  orchestrator/   # FrameCollector, SessionRunner
games/
  breakout71/     # Breakout71Env, loader, modal handler, YOLO classes
configs/
  games/          # Per-game loader configs
  training/       # Per-game YOLO training configs
scripts/          # CLI tools (train_rl, run_session, capture, debug, etc.)
tests/            # Unit + integration tests
documentation/
  specs/          # Canonical specifications
  ROADMAP.md      # 5-phase plan
docs/             # Sphinx source
```

## Reference

- `documentation/ROADMAP.md` — what to do next (5-phase plan)
- `private/documentation/BigRocks/checklist.md` — detailed task tracking
- `private/documentation/reference/agent_knowledge_base.md` — all technical
  discoveries from sessions 1-30
- `documentation/specs/` — canonical specifications
