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
- Update README test count after adding tests

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

- **703 tests**, 96% coverage, 8 subsystems complete
- **Architecture done:** BaseGameEnv ABC, game plugin system (`games/`),
  `--game` flag, CNN/MLP observation modes, dynamic plugin loading
- **No real training results yet** — Phase 1 in roadmap
- See `documentation/ROADMAP.md` for the 5-phase plan
- See `private/documentation/BigRocks/checklist.md` for detailed task tracking

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
tests/            # 703 tests (679 unit + 24 integration)
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
