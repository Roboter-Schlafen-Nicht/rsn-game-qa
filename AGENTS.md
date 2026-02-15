# RSN Game QA — Agent Instructions

RL-driven autonomous game testing platform. First target: **Breakout 71** (browser-based TypeScript canvas game at `F:\work\breakout71-testbed`).

## Workflow

- Work on feature branches from `main` (pattern: `feature/...`, `docs/...`)
- Implement → commit → push → create PR → **request review from Copilot** → evaluate review, create issues if needed → merge with `--delete-branch` → delete local branch → **post-merge admin** (update `documentation/BigRocks/checklist.md`, create session log in `documentation/sessions/`, update `AGENTS.md`)
- **Post-merge admin is checklist-only** — no Copilot review needed; commit, push, PR, merge directly
- Pre-commit hook runs the full CI pipeline via `act` (Docker-based GitHub Actions). **This can take 5+ minutes.** Use `timeout` of 600000ms for commit commands; check `git log` afterward to verify.

## Conventions

- Python 3.12, conda env `yolo`
- NumPy-style docstrings, ruff for lint/format, pytest for tests, Sphinx (Furo theme) for docs
- `conda run -n yolo` for Python commands (frequently times out — use direct paths like `C:/Users/human/miniconda3/envs/yolo/Scripts/ruff.exe` for ruff, or generous timeouts)
- CI has 4 jobs: Lint (ruff check + format), Test (pytest), Build Check (verify imports), Build Docs (Sphinx -W)
- Always run `ruff format` before committing
- Commit messages: imperative mood with type prefix (`feat:`, `fix:`, `ci:`, `docs:`)
- Docs: canonical specs in `documentation/specs/`, thin `{include}` wrappers in `docs/specs/`, API autodoc in `docs/api/`
- Delete feature branches after merging
- Update README test count after adding tests

## Key Discoveries

- **Sphinx `-W` builds**: dataclass entries in RST need `:no-index:` to avoid duplicate warnings
- **Severity convention**: `"critical"` / `"warning"` / `"info"` throughout oracles
- **`pydirectinput` in `autodoc_mock_imports`** (`docs/conf.py`)
- **pywin32 + pydirectinput ARE installed** in conda env — tests simulating "missing" must use `importlib.reload` with `sys.modules` patched to `None`
- **Windows `NUL` file artifact**: `> NUL` redirect can create a literal file; delete with `rm -f NUL`
- **GitHub self-approval not allowed** — use Copilot as reviewer
- **Copilot code review is web-UI only** — `gh pr edit --add-reviewer copilot`, `gh pr create --reviewer copilot`, and `gh api .../requested_reviewers` all silently fail. Must request Copilot review manually via the GitHub web UI (PR page → Reviewers dropdown → select "Copilot"). The `gh agent-task` CLI commands are for the Copilot *coding agent*, not the code reviewer. Allow several minutes for Copilot to post its review after requesting.
- **`git pull` times out** — use `git fetch origin main && git reset --hard origin/main`
- **`gh pr merge` times out** but often succeeds — verify with `gh pr view N --json state,mergedAt`
- LSP unresolved import errors (cv2, gymnasium, pydirectinput, etc.) are pre-existing and harmless
- **`import cv2` at module top level breaks CI** — Docker container lacks `libGL.so.1`; must use lazy imports inside methods
- **Firefox subprocess cleanup impossible** — Firefox hands off to a separate main process; Selenium WebDriver solves this
- **Firefox first-run issues** — Fresh profile triggers Welcome tab; `-width`/`-height` CLI flags don't exist (Firefox interprets `1280` as a URL). Use Selenium Options API instead.
- **`BitBlt` all-black for Chromium** — GPU-accelerated compositing prevents capture; use `PrintWindow` with `PW_RENDERFULLCONTENT` (flag=2)
- **Selenium 4.40.0 has built-in Selenium Manager** — handles driver binaries automatically; no `webdriver-manager` needed
- **`selenium` is imported lazily** inside `BrowserInstance.__init__` only — no CI impact

### YOLO Training Pipeline (session 9)

- **`scripts/` needs `__init__.py`** — Without it, `from scripts.train_model import ...` fails in CI
- **ultralytics import triggers cv2 → libGL.so.1 failure in CI Docker** — Move all input validation BEFORE `from ultralytics import YOLO` import
- **`torch` import is local in `resolve_device()`** — Must mock via `mock.patch.dict("sys.modules", {"torch": mock_torch})`; moved after early return so explicit device selection doesn't require torch
- **Windows path backslashes in test assertions** — Use generic match strings like `"does not exist"` instead of literal paths in `pytest.raises(match=...)`
- **`roboflow` package uses `>=1.1.0`** — Pinned as minimum version in `environment.yml`
- **`python-dotenv==1.1.0`** — Added to `environment.yml` pip dependencies
- **`argparse` `action="store_true"` + `default=True` is a no-op** — Use `--no-X` with `action="store_false"` + `dest="X"` instead

### Data Collection & Auto-Annotation (session 10)

- **Selenium `execute_script()` IIFE bug** — Must use `return (function() { ... })();` — outer `return` required for Selenium to capture IIFE return value. Without it, returns `null`.
- **Browser chrome ~130px** — Tab bar + URL bar + "Chrome is being controlled" banner at top of captured frames. Must account for when mapping game coordinates.
- **21 palette colors → 11 HSV groups** — `palette.json` defines 21 brick colors; mapped to 11 HSV detection ranges (blue, yellow, red, orange, green, cyan, purple, pink, beige, white, gray). Adjacent same-color bricks merge — grid-splitting logic handles this.
- **Ball particle trail** — Ball emits multiple white fragments. Solution: dilate to merge nearby blobs, pick smallest confirmed candidate.
- **UI false positives** — "Level 1/7" button, "$0/$10" score, coin counter icon all trigger false detections. Eliminated via UI mask zones + frame differencing.
- **White/gray brick detection** — Must restrict to "brick zone" (upper 65% of game area) to avoid false-positiving on paddle, ball, or background.
- **Game zone boundaries** — Columns ~324 to ~956 (632px wide) at 1280x1024. Coins/ball must be within these boundaries.
- **Individual bricks ~89x89 pixels** at 1280x1024. Client area = 1264x1016 pixels.
- **Random bot dies quickly** — Never clears level 1, so dataset is 93.7% gameplay + 6.3% game-over (no perk picker states)
- **Game state detection via DOM** — `document.body.classList.contains('has-alert-open')` for modal detection, `#popup` for content, `#close-modale` for dismissible modals

### Breakout 71 Game Mechanics (session 8 — source study)

- **Scoring is coin-based, not brick-based** — breaking bricks spawns coins that fly with physics; catching coins with the paddle adds to the score. Combo system multiplies coin value. Combo resets if ball returns to paddle without hitting a brick.
- **No traditional lives** — the `extra_life` perk (max 7 levels) acts as expendable rescues. When last ball is lost with `extra_life > 0`, ball is rescued. When `extra_life == 0` and all balls lost → game over.
- **Multi-ball is a perk** — `multiball` perk spawns additional balls. Losing one ball isn't game over unless ALL are lost.
- **Level system** — 7 + `extra_levels` levels per run. Between levels: perk selection screen (`openUpgradesPicker()`). ~60+ perks available. `chill` perk = infinite levels.
- **No explicit state machine** — uses boolean flags: `running`, `isGameOver`, `ballStickToPuck`
- **Input** — mouse position sets paddle directly; keyboard `ArrowLeft`/`ArrowRight` move incrementally (`gameZoneWidth/50` per tick, 3x with Shift); `Space` toggles play/pause
- **Android version** — thin Kotlin WebView wrapper loading the same compiled `index.html`. 100% identical game logic. Only differences: touch input (hold-to-play), portrait lock, video/save export via Android intents.
- **Canvas** — `#game` element, fills `innerWidth x innerHeight` (x pixelRatio). Game zone width = `brickWidth * gridSize`, centered horizontally.
- **Level completion** — when `remainingBricks === 0 && !hasPendingBricks`, either instant win (no coins) or 5s delay (`winAt` timer) for coin collection.
- **Ball physics** — speed normalizes toward `baseSpeed * sqrt(2)` each tick; multiple sub-steps per frame to prevent tunneling; bounce angle depends on paddle hit position and `concave_puck` perk.
- **YOLO class `"powerup"` maps to coins** in the perception subsystem.

## Project Structure

```
src/
  game_loader/    # DONE — YAML config, factory, loaders (82 tests)
  reporting/      # DONE — JSON reports, HTML dashboard (26 tests)
  capture/        # DONE — BitBlt window capture, pydirectinput input (37 tests)
  perception/     # DONE — YoloDetector, breakout_capture (41 tests)
  oracles/        # DONE — 12 oracles with on_step detection (132 tests)
  env/            # DONE — Breakout71Env gymnasium wrapper (69 tests)
configs/
  games/                  # Per-game loader configs (breakout-71.yaml)
  training/               # Per-game YOLO training configs (breakout-71.yaml)
tests/
  conftest.py             # Integration test fixtures (Selenium browser parameterization)
  test_integration.py     # 12 integration tests × 2 browsers
  test_training_pipeline.py  # 41 training pipeline tests
scripts/
  _smoke_utils.py         # BrowserInstance (Selenium), get_available_browsers(), utilities
  smoke_launch.py         # Game launch + proof screenshot
  smoke_capture.py        # Multi-frame capture verification
  smoke_oracle.py         # Oracle run + JSON report
  capture_dataset.py      # Frame capture with random bot + game state detection
  auto_annotate.py        # OpenCV auto-annotation (HSV segmentation, frame differencing)
  upload_to_roboflow.py   # Roboflow API upload with resume support
  train_model.py          # Config-driven YOLO training
  validate_model.py       # mAP threshold validation
documentation/
  BigRocks/checklist.md   # Master checklist — the source of truth for what's done and what's next
  specs/                  # Canonical spec files
  sessions/               # Session logs (gitignored)
docs/                     # Sphinx source (conf.py, api/, specs/)
```

## What's Done (sessions 1-10)

1. **Session 1** — Perplexity research (capture, input, RL, market analysis)
2. **Session 2** — Project scaffolding, game loader subsystem, CI pipeline (PR #4, #6)
3. **Session 3** — Reporting subsystem (PR #7, #8)
4. **Session 4** — Capture & Input subsystem (PR #9, #10, #11)
5. **Session 5** — Perception subsystem (PR #12)
6. **Session 6** — Oracle `on_step` detection logic, 12 oracles (PR #13)
7. **Session 7** — Smoke scripts, integration tests, Selenium browser management (PR #15)
8. **Session 8** — Game source study, revised env design, Breakout71Env v1 implementation (PR #17)
9. **Session 9** — Config-driven YOLO training pipeline: capture, upload, train, validate (PR #19)
10. **Session 10** — Data collection pipeline: 300-frame capture with game state detection, auto-annotation with OpenCV (PR #21)

Total: **398 tests** (398 unit + 24 integration), 6 subsystems + training pipeline complete.

## What's Next

Read `documentation/BigRocks/checklist.md` for the full breakdown. In order:

1. **Roboflow upload + review** — upload 300 auto-annotated frames, human review/correction
2. **Train + validate YOLO model** — train on reviewed annotations, validate mAP thresholds
3. **Integration & E2E** — wire all subsystems, run episodes, generate reports

## Reference Files

When starting a new feature, read these first:
- `documentation/BigRocks/checklist.md` — what to do next and detailed subtasks
- The relevant spec in `documentation/specs/`
- The existing stub in `src/` for the subsystem being implemented
- A completed subsystem (e.g., `src/capture/`) for pattern reference
