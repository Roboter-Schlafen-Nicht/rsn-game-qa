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
- **Ball particle trail** — Ball emits multiple white fragments. Solution: dilate to merge nearby blobs, use `_find_ball_head()` to locate the most circular sub-contour within the merged blob.
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

### XPU YOLO Training (session 12)

- **Ultralytics 8.4.14 XPU support requires 3 monkey patches** in `_patch_ultralytics_xpu()`:
  - `select_device()` — rejects `"xpu"` as invalid CUDA device; must intercept before CUDA validation. Patch all import sites (torch_utils, trainer, validator, predictor, exporter).
  - `GradScaler("cuda")` — `_setup_train()` hard-codes CUDA; must replace with `GradScaler("xpu")` after original setup.
  - `_get_memory()` — falls through to `torch.cuda.memory_reserved()` returning 0; must add XPU branch.
- **XPU patches must be idempotent** — `_applied` flag prevents re-wrapping on multiple calls (Copilot review)
- **`pip install ultralytics` overwrites XPU torch** — ultralytics pulls CPU-only torch from PyPI; must reinstall XPU torch after every ultralytics install
- **4 additional ultralytics XPU issues** (training works without): `_clear_memory()`, `autocast()`, `check_amp()`, OOM handler — all hard-code CUDA
- **Posted on ultralytics #16930** — detailed comment with all 7 XPU fixes: https://github.com/ultralytics/ultralytics/issues/16930#issuecomment-3905263741
- **Training results** — 100 epochs on XPU (~23 min, ~14s/epoch). Best model epoch 91. mAP50=0.679, mAP50-95=0.578. brick=0.995, paddle=0.976, ball=0.922, powerup=0.502, wall=0.000.
- **mAP threshold lowered to 0.65** from 0.80 for initial auto-annotated dataset. Raise after human-reviewed annotations.
- **`prepare_dataset.py`** — restructures flat dataset into YOLO train/val format. Val ratio validation and train set non-empty guard added per Copilot review.
- **Config `dataset_path` should be `null`** — prepared dataset is under gitignored `output/`; users override via CLI or edit locally (Copilot review)
- **Default device is `cpu`** — user found CPU fastest for training on this hardware

### Annotation Pipeline Improvements (session 11)

- **Ball-in-brick false positive** — White/gray bricks (~79x79px, circularity ~0.81) merge into giant blobs during dilation. `_find_ball_head` then picks a brick sub-contour as the "ball head." Fix: pass `brick_detections` to `_detect_ball()` and zero out brick bounding boxes (with 4px padding) from the white mask before dilation.
- **`_find_ball_head` scoring** — Changed from `circularity * area` to `circularity² * area` to prevent large elongated trail blobs from outscoring the smaller but circular ball.
- **Paddle zone exclusion killed near-paddle balls** — Fixed bottom-15% cut (`y > 0.85 * img_h`) masked out balls near the paddle. Fix: `_detect_ball` now accepts `paddle_detections` and masks only the paddle's bounding box (6px padding). Fallback when no paddle detected: conservative 5% bottom cut. Ball detection went from 288/297 to 297/297 (100%).
- **`_detect_game_zone` left-wall bug** — Returned `left=0` because browser chrome columns 0-7 had brightness ~32, above the relative threshold. Fix: primarily detect wall peaks (brightness > 200), fallback skips first/last 10 columns.
- **Roboflow annotation upload** — `project.single_upload(annotation_path=..., annotation_labelmap=...)` works for YOLO `.txt` pre-labels. `search_all()` returns a generator of batches (lists of dicts), not a flat list.
- **`classes.txt` convention** — `auto_annotate.py` now writes `labels/classes.txt` (standard YOLO class-name file). `upload_to_roboflow.py` reads it via `_build_labelmap()` with YAML config fallback.

### Integration & E2E + RL Scaffold (session 13)

- **CI Docker libGL fix for cv2.imwrite** — FrameCollector and FindingBridge tests call `cv2.imwrite()` which fails in CI Docker (missing `libGL.so.1`). Fixed with `autouse` fixture `_mock_cv2_imwrite` that injects a mock cv2 module creating 0-byte files.
- **test_raises_without_weights fragile** — Test assumed no weights file existed, but `weights/breakout71/best.pt` exists locally from training. Fixed by overriding `cfg["output_dir"]` to `tmp_path`.
- **pytest-cov configured** — coverage runs automatically on `src/` only (scripts excluded). Config in `pyproject.toml` under `[tool.pytest.ini_options]` addopts, `[tool.coverage.run]`, `[tool.coverage.report]`. 96% coverage.
- **Env bug: close/reset lifecycle** — `close()` didn't reset `_initialized` → crash if `reset()` called after `close()`. Fixed.
- **Env bug: fragile reset timing** — 500ms sleep after Space assumes instant game restart. Fixed with retry loop.
- **Env bug: _bricks_total corruption** — If first frame after reset shows transition screen (0 bricks), `_bricks_total` becomes 1. Fixed with retry logic guarded by `self._initialized`.
- **Env bug: reset() ball retry** — Now raises RuntimeError if ball not detected after 5 retries (Copilot review fix).
- **SB3 2.7.1 compatible** — PPO with `MlpPolicy`, `Discrete(3)` action, `Box(8,)` obs, `DummyVecEnv` wrapper. Hyperparams: `n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, lr=3e-4, clip_range=0.2, ent_coef=0.01`.
- **Hybrid data generation** — During RL training, capture every Nth frame (default 30) via FrameCollectionCallback. Batch auto-annotate after session. Human reviews on Roboflow before YOLO retraining.
- **Empty ReportGenerator removed from train_rl.py** — Copilot review caught instantiation with no episodes added. Removed entirely.
- **`coins_norm` and `score_norm` hardcoded 0.0** — 2 of 8 observation dimensions carry no info. SB3 will learn to ignore them.
- **Real-time training bottleneck** — Env runs at ~30 FPS with `time.sleep(1/30)` per step. 200k steps ≈ 1.85 hours.

### Selenium-Based Env Control (session 14)

- **pydirectinput keyboard control doesn't work** — The game uses **mouse position** to control the paddle directly; keyboard arrows move incrementally (`gameZoneWidth/50` per tick) — too slow for RL. `Space` via pydirectinput didn't reliably start the game either.
- **Env refactored from pydirectinput to Selenium** — `Breakout71Env` now accepts a Selenium `WebDriver` via `driver=` parameter. Paddle control uses `ActionChains` mouse movement on the `#game` canvas. Modal handling (game over, perk picker, menu) uses `driver.execute_script()` with JS snippets from `capture_dataset.py`.
- **`InputController` no longer used by env** — `_input` attribute replaced with `_driver`, `_game_canvas`, `_canvas_dims`. The `src/capture/input_controller.py` module still exists for other use cases but env doesn't import it.
- **`train_rl.py` needs GameLoader** — Original version only launched `BrowserInstance` (Selenium) without starting the Parcel dev server. Fixed by adding `create_loader(config)` → `loader.setup()` → `loader.start()` before browser launch, and `loader.stop()` in teardown.
- **JS game state detection snippets** — `DETECT_STATE_JS` returns `"gameplay"`, `"game_over"`, `"perk_picker"`, or `"menu"` via DOM inspection (`document.body.classList.contains('has-alert-open')`, `#popup` content). `CLICK_PERK_JS` picks first available perk, `DISMISS_GAME_OVER_JS` clicks `#close-modale`, `DISMISS_MENU_JS` clicks `#game`.
- **`close()` does NOT close the WebDriver** — Caller owns the driver lifecycle (typically `BrowserInstance`). Env only nulls its `_game_canvas` and `_canvas_dims` references.
- **Canvas element lookup** — `_lazy_init()` finds `#game` canvas via `driver.find_element(By.ID, "game")`, falls back to `body` if not found. Canvas dimensions read via `element.size`.
- **`pydirectinput` still in conda env and `autodoc_mock_imports`** — Not removed since `src/capture/input_controller.py` still uses it. Just no longer imported by env.
- **`_apply_action()` must track absolute paddle position** — Selenium `move_to_element_with_offset` positions the mouse at an offset from the element's *centre*, not from current position. Using a fixed ±step_size causes the paddle to oscillate between two positions instead of moving incrementally. Fix: track `_paddle_target_x` (pixels from canvas left edge) and convert to centre-relative offset for each call.
- **`step()` must handle modals mid-episode** — Without modal handling in `step()`, the game-over/perk-picker modal overlay blocks YOLO detection, causing `_no_ball_count` to reach threshold and the episode to terminate incorrectly. Fix: call `_handle_game_state()` in `step()` before frame capture.

### First RL Training Attempt & Action Space Redesign (session 15)

- **Pipeline validated** — GameLoader → Selenium → YOLO → Env → PPO loop runs end-to-end. Game-over modals are dismissed correctly.
- **Discrete(3) action space is wrong for this game** — The game uses mouse position (continuous) to control the paddle directly. `Discrete(3)` with fixed increments produces jerky, rarely-moving paddle because: (a) ~33% NOOP, (b) LEFT/RIGHT cancel out with random policy, (c) step_size=10% canvas width = ~128px discrete jumps.
- **Decision: switch to continuous action space** — `Box(low=-1, high=1, shape=(1,))` where the action value maps to absolute paddle position. SB3 PPO with `MlpPolicy` auto-detects continuous vs discrete from the action space.
- **JS `puckPosition` injection** — Instead of Selenium ActionChains mouse events, set paddle position directly via `driver.execute_script()`. The game's `setMousePos()` just does `puckPosition = Math.round(x)` with no transformation. More reliable than synthetic mouse events.
- **Game mouse input pipeline** — `mousemove` on `#game` canvas → `e.clientX * getPixelRatio()` → `setMousePos()` → `puckPosition = Math.round(x)` → `normalizeGameState()` clamps to game zone. No pointer lock by default. `getPixelRatio()` returns 1 on standard displays.
- **`puckPosition` coordinate system** — Canvas pixels from left edge. Game zone centered with `offsetX`. Clamp bounds: `offsetX + puckWidth/2` to `offsetX + gameZoneWidth - puckWidth/2`.

### Continuous Action Space & Coverage (session 16)

- **Continuous action space implemented** — `Discrete(3)` → `Box(-1, 1, shape=(1,), dtype=float32)`. Action value linearly maps to absolute paddle pixel position via `SET_PUCK_POSITION_JS`. Removed `_paddle_target_x`, replaced with `_game_zone_left`/`_game_zone_right` queried from JS globals (`offsetX`, `gameZoneWidth`, `puckWidth`).
- **`DashboardRenderer` bug in `session_runner.py`** — `run()` called `DashboardRenderer(output_dir=...)` but the constructor only accepts `template_dir`/`template_name`. Also called `.render()` (returns HTML string) instead of `.render_to_file()`. Fixed to use `DashboardRenderer()` + `render_to_file()`.
- **Coverage threshold enforced** — Added `fail_under = 80` to `[tool.coverage.report]` in `pyproject.toml`. All source files now ≥82% coverage.
- **`session_runner.py` coverage: 79% → 98%** — Added 8 new tests: `_setup()` with mocked lazy imports (4 tests), `_cleanup()` loader branch (2 tests), `run()` data collection finalize + dashboard generation (2 tests). Only uncovered: exception handler in dashboard try/except.
- **`_setup()` testing pattern** — Must mock 4 lazy imports: `scripts._smoke_utils.BrowserInstance`, `src.env.breakout71_env.Breakout71Env`, `src.game_loader.create_loader`, `src.game_loader.config.load_game_config`. The imports happen inside `_setup()` so patches must target the original module paths.
- **`scripts/train_rl.py` unchanged** — SB3 PPO auto-detects continuous `Box` action space; no code changes needed.
- **`scripts/capture_dataset.py` unchanged** — Random bot uses its own ActionChains; separate concern from env action space.

### Pixel-Based Debug Loop (session 19)

- **Architecture decision: pixel-based only** — As a QA platform product, we won't always have game source code. The universal interface is pixels in, mouse/keyboard out. YOLO for observations, pydirectinput for input, JS injection only for modal handling (DOM overlays).
- **Game `mouseup` pauses** — `mouseup` on `#game` canvas while `running=true` pauses the game (game.ts:247-256). Must click only once to start, then use `moveTo` only.
- **`ballStickToPuck=true` at start** — Ball sticks to paddle and follows it until first click. Key insight for safe phase 2 testing (pre-start phases).
- **YOLO CPU performance** — ~130-190ms per frame inference on CPU (~5-7 FPS). GPU/XPU will easily exceed 30 FPS.
- **Selenium modal handling throttling** — Calling `_ensure_gameplay` every frame drops FPS from ~7 to ~4 due to HTTP round-trip (~100ms). Throttling to every 2s restores FPS.
- **Phase 2 paddle tracking accuracy** — Edge positions (0.20, 0.80) have ~0.11 error due to game zone clamping; center positions (0.35, 0.50, 0.65) have near-perfect accuracy (~0.006 error).
- **Copilot review: 7 fixes** — Inverted `modal_recoveries` logic (phases 3 & 4), pixel clamping in `_norm_to_screen`, retry resilience in `_ensure_gameplay`, double YOLO inference in phase 2, docstring accuracy (FPS threshold, output path).

### XPU Auto-Detection & OpenVINO Research (session 20)

- **`resolve_device()` canonical implementation** — Module-level function in `src/perception/yolo_detector.py` (lines 28-54). Priority: xpu > cuda > cpu. `"auto"` triggers detection; explicit strings pass through unchanged.
- **`YoloDetector` default changed to `"auto"`** — Copilot review caught `device="xpu"` hard-coded default; changed to `device="auto"` with `resolve_device()` called inside `__init__`.
- **All scripts/configs updated** — `debug_pixel_loop.py`, `train_rl.py`, `train_model.py`, `configs/training/breakout-71.yaml` all default to `"auto"` instead of `"cpu"`.
- **SB3 PPO doesn't support XPU** — `train_rl.py` maps `"xpu"` → `"cpu"` for the policy network.
- **OpenVINO research** — Intel toolkit converting PyTorch/ONNX to optimized IR. Ultralytics benchmarks on Arc A770: YOLO11n 16.29ms (PyTorch) → 4.84ms (OpenVINO FP32) → 3.34ms (INT8). ~5x speedup. Inference-only (no training). `model.export(format="openvino")` then `YOLO("model_openvino_model/")`. No XPU monkey patches needed for inference.
- **DXGI Desktop Duplication research** — DXcam library: 238 FPS at 2560x1440 vs PrintWindow ~30 FPS. Captures entire monitor (not window-specific). Returns numpy arrays directly. Requires real GPU/display for CI.
- **Combined projection** — OpenVINO + DXcam together: ~10-15ms/frame (60-100 FPS) vs current ~140ms/frame (6-8 FPS).
- **XPU vs CPU benchmark** — Intel Arc A770: XPU 6-8 FPS vs CPU 5 FPS (+30-60% improvement, but OpenVINO will be much better).

## Project Structure

```
src/
  game_loader/    # DONE — YAML config, factory, loaders (82 tests)
  reporting/      # DONE — JSON reports, HTML dashboard (26 tests)
  capture/        # DONE — BitBlt window capture, pydirectinput input (37 tests)
  perception/     # DONE — YoloDetector, breakout_capture (41 tests)
  oracles/        # DONE — 12 oracles with on_step detection (132 tests)
  env/            # DONE — Breakout71Env gymnasium wrapper (94 tests)
  orchestrator/   # DONE — FrameCollector, SessionRunner (55 tests)
configs/
  games/                  # Per-game loader configs (breakout-71.yaml)
  training/               # Per-game YOLO training configs (breakout-71.yaml)
tests/
  conftest.py             # Integration test fixtures (Selenium browser parameterization)
  test_integration.py     # 12 integration tests × 2 browsers
  test_training_pipeline.py  # 49 training pipeline tests
  test_orchestrator.py    # 55 orchestrator tests (FrameCollector, SessionRunner)
scripts/
  _smoke_utils.py         # BrowserInstance (Selenium), get_available_browsers(), utilities
  smoke_launch.py         # Game launch + proof screenshot
  smoke_capture.py        # Multi-frame capture verification
  smoke_oracle.py         # Oracle run + JSON report
  capture_dataset.py      # Frame capture with random bot + game state detection
  auto_annotate.py        # OpenCV auto-annotation (HSV segmentation, frame differencing)
  upload_to_roboflow.py   # Roboflow API upload with resume support
  train_model.py            # Config-driven YOLO training (with XPU monkey patches)
  validate_model.py         # mAP threshold validation (with XPU support)
  prepare_dataset.py        # Restructure flat dataset into YOLO train/val format
  run_session.py            # CLI for N-episode QA evaluation runs
  train_rl.py               # CLI for PPO training with SB3
  debug_game_state.py       # Diagnostic script for JS game state detection
  debug_pixel_loop.py       # 4-phase pixel-based pipeline validation (capture→YOLO→pydirectinput)
documentation/
  BigRocks/checklist.md   # Master checklist — the source of truth for what's done and what's next
  specs/                  # Canonical spec files
  sessions/               # Session logs (private, gitignored — never commit)
docs/                     # Sphinx source (conf.py, api/, specs/)
```

## What's Done (sessions 1-20)

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
11. **Session 11** — Annotation pipeline improvements: ball-in-brick fix, paddle-zone fix, game-zone wall detection, Roboflow annotation upload, 100% ball detection (PR #23)
12. **Session 12** — XPU YOLO training: 3 ultralytics monkey patches, dataset preparation, 100-epoch training (mAP50=0.679), open-source contribution to ultralytics #16930 (PR #26)
13. **Session 13** — Integration & E2E + RL scaffold: orchestrator (FrameCollector, SessionRunner), run_session.py, train_rl.py (SB3 PPO), 6 env bug fixes, legacy code cleanup, pytest-cov (96% coverage) (PR #28)
14. **Session 14** — Selenium-based env control: replaced pydirectinput with Selenium ActionChains for paddle control and JS execution for modal handling, GameLoader integration in train_rl.py, Copilot review fixes (public JS constants, body fallback handling) (PR #30)
15. **Session 15** — First RL training attempt: pipeline validated end-to-end, diagnosed Discrete(3) action space as wrong for continuous paddle control, decided to switch to Box(-1,1) continuous action with JS puckPosition injection (no PR — interrupted before implementation)
16. **Session 16** — Continuous action space & coverage: Box(-1,1) with JS puckPosition injection, robust _query_game_zone, DashboardRenderer fix, coverage enforcement (fail_under=80), 21 new tests (PR #32)
17. **Session 17** — Deep source code analysis & game spec: read all 11 critical testbed source files, TypeDoc generation (PR #34), comprehensive game technical specification covering ~80 GameState fields, 63 perks, physics, scoring, combo, level progression, Selenium integration points (PR #35)
18. **Session 18** — Player behavior specification: abstracted technical spec into player-perspective behavioral contracts (80+ B-IDs), covering observable states, actions, cause-effect behaviors, and RL training implications (PR #37)
19. **Session 19** — Pixel-based debug loop: 4-phase validation script (capture→YOLO→pydirectinput→modal handling), validated live — Phase 1 (static detection), Phase 2 (paddle tracking, max error 0.119), Phase 3 (100% ball detection), Phase 4 (gameplay loop, FPS=4 on CPU YOLO). Copilot review: 7 fixes (inverted modal logic, pixel clamping, retry resilience, double inference, docstrings) (PR #39)
20. **Session 20** — XPU auto-detection: `resolve_device()` function (xpu>cuda>cpu priority), all scripts/configs default to `"auto"`, OpenVINO & DXGI Desktop Duplication research (PR #43)

Total: **510 tests** (486 unit + 24 integration), 7 subsystems + training pipeline complete.

## What's Next

Read `documentation/BigRocks/checklist.md` for the full breakdown. In order:

1. **RL Training & Iteration** — run first real PPO training, evaluate vs random baseline, iterate on reward shaping, retrain YOLO with human-reviewed annotations

## Reference Files

When starting a new feature, read these first:
- `documentation/BigRocks/checklist.md` — what to do next and detailed subtasks
- The relevant spec in `documentation/specs/`
- The existing stub in `src/` for the subsystem being implemented
- A completed subsystem (e.g., `src/capture/`) for pattern reference
