# Copilot Code Review Instructions

This is an RL-driven autonomous game testing platform (Python 3.12).
The agent plays browser games from pixels using YOLO object detection
and reinforcement learning, then generates QA reports via bug-detection
oracles.

## Review Priorities

1. **Security** — flag any hardcoded secrets, API keys, local filesystem
   paths that expose usernames, or credentials in code or configs.
2. **Correctness** — verify function signatures match their callers
   (especially `load_game_config()` which takes `(name, configs_dir=)`
   not a full path). Check for off-by-one errors in reward/penalty logic.
3. **CI compatibility** — `import cv2` at module top level breaks the
   Docker-based CI (missing `libGL.so.1`). Flag any top-level cv2 imports;
   they must be lazy (inside functions/methods).
4. **Test coverage** — public API changes to `step()`, `reset()`,
   `close()`, reward functions, observation fields, and orchestrator
   lifecycle must have corresponding tests.
5. **Pixel-based constraint** — the system must NOT inject JavaScript to
   read game state. Exception: Selenium JS is allowed for modal dismissal
   and one-time settings (e.g. mute audio).

## Conventions

- NumPy-style docstrings
- `ruff` for lint and format
- `pytest` for testing (693+ tests)
- Commit messages: imperative mood with type prefix (`feat:`, `fix:`,
  `ci:`, `docs:`)
- Game-specific code belongs in `games/<name>/`, not in `src/`
- CNN observation mode is the default (game-agnostic, no YOLO needed);
  MLP is optional and game-specific

## Known Patterns to Enforce

- `YoloDetector._DEFAULT_CLASSES` must be `[]` (game-agnostic); classes
  come from `game_classes()` on the env subclass
- `base_argparser()` sets `--config` default to `"breakout-71"` — scripts
  using the `--game` plugin system must call
  `parser.set_defaults(config=None)` so plugin defaults work
- `_no_ball_count` is exclusively owned by `_check_late_game_over()`;
  `check_termination()` only reads it
- Episode boundary: `step()` passes `dismiss_game_over=False`;
  `reset()` passes `True`
- Terminal penalty uses a fixed `-5.01`, not YOLO-computed reward
  (game-over modal occludes the game state)
