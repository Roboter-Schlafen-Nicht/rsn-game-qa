# Agent Knowledge Base

Accumulated technical discoveries and session history from sessions 1-29.
Organized by topic for searchability. For the current operational guide,
see `AGENTS.md` at the project root.

---

## CI / Docker Pitfalls

- **`import cv2` at module top level breaks CI** -- Docker container lacks
  `libGL.so.1`. Must use lazy imports inside methods.
- **CI Docker libGL fix for `cv2.imwrite`** -- FrameCollector and
  FindingBridge tests call `cv2.imwrite()` which fails in CI Docker. Fixed
  with `autouse` fixture `_mock_cv2_imwrite` that injects a mock cv2 module
  creating 0-byte files.
- **`scripts/` needs `__init__.py`** -- Without it,
  `from scripts.train_model import ...` fails in CI.
- **ultralytics import triggers cv2 failure in CI Docker** -- Move all
  input validation BEFORE `from ultralytics import YOLO` import.
- **Sphinx `-W` builds**: dataclass entries in RST need `:no-index:` to
  avoid duplicate warnings.
- **Windows `NUL` file artifact**: `> NUL` redirect can create a literal
  file; delete with `rm -f NUL`.
- **openvino 2025.4.1 requires `numpy<2.4.0`** -- Both `environment.yml`
  AND `ci.yml` must use `numpy>=2.3.0,<2.4.0`.

## GitHub / Git Workflow

- **GitHub self-approval not allowed** -- use Copilot as reviewer.
- **Copilot code review is web-UI only** -- `gh pr edit --add-reviewer
  copilot`, `gh pr create --reviewer copilot`, and
  `gh api .../requested_reviewers` all silently fail. Must request Copilot
  review manually via the GitHub web UI (PR page -> Reviewers dropdown ->
  select "Copilot"). Allow several minutes for Copilot to post its review.
- **`git pull` times out** -- use
  `git fetch origin main && git reset --hard origin/main`.
- **`gh pr merge` times out** but often succeeds -- verify with
  `gh pr view N --json state,mergedAt`.

## Selenium / Browser

- **Firefox subprocess cleanup impossible** -- Firefox hands off to a
  separate main process; Selenium WebDriver solves this.
- **Firefox first-run issues** -- Fresh profile triggers Welcome tab;
  `-width`/`-height` CLI flags don't exist (Firefox interprets `1280` as a
  URL). Use Selenium Options API instead.
- **`BitBlt` all-black for Chromium** -- GPU-accelerated compositing
  prevents capture; use `PrintWindow` with `PW_RENDERFULLCONTENT` (flag=2).
- **Selenium 4.40.0 has built-in Selenium Manager** -- handles driver
  binaries automatically; no `webdriver-manager` needed.
- **`selenium` is imported lazily** inside `BrowserInstance.__init__` only
  -- no CI impact.
- **Selenium `execute_script()` IIFE bug** -- Must use
  `return (function() { ... })();` -- outer `return` required for Selenium
  to capture IIFE return value. Without it, returns `null`.
- **Browser chrome ~130px** -- Tab bar + URL bar + "Chrome is being
  controlled" banner at top of captured frames. Must account for when
  mapping game coordinates.
- **BrowserInstance `_SUPPORTED_BROWSERS`** -- Uses `"edge"` not
  `"msedge"`.
- **Game `mouseup` pauses** -- `mouseup` on `#game` canvas while
  `running=true` pauses the game. Must click only once to start, then use
  `moveTo` only.

## Python / Conda Environment

- **pywin32 + pydirectinput ARE installed** in conda env -- tests
  simulating "missing" must use `importlib.reload` with `sys.modules`
  patched to `None`.
- **`conda run -n yolo` frequently times out** -- use the full conda
  env path for the tool (e.g. `<conda_envs>/yolo/Scripts/ruff.exe`),
  or generous timeouts.
- **`argparse` `action="store_true"` + `default=True` is a no-op** -- Use
  `--no-X` with `action="store_false"` + `dest="X"` instead.
- **`torch` import is local in `resolve_device()`** -- Must mock via
  `mock.patch.dict("sys.modules", {"torch": mock_torch})`.
- **Windows path backslashes in test assertions** -- Use generic match
  strings like `"does not exist"` instead of literal paths in
  `pytest.raises(match=...)`.
- **Lazy import mocking pattern** -- `pydirectinput` and `win32gui`
  imported lazily inside functions; `mock.patch.dict(sys.modules, {...})`
  intercepts the lazy import.
- **LSP unresolved import errors** (cv2, gymnasium, pydirectinput, etc.)
  are pre-existing and harmless.

## YOLO / Perception

- **YOLO class `"powerup"` maps to coins** in the perception subsystem.
- **mAP threshold lowered to 0.65** from 0.80 for initial auto-annotated
  dataset. Raise after human-reviewed annotations.
- **Training results (session 12)** -- 100 epochs on XPU (~23 min,
  ~14s/epoch). Best model epoch 91. mAP50=0.679, mAP50-95=0.578.
  brick=0.995, paddle=0.976, ball=0.922, powerup=0.502, wall=0.000.
- **Ball particle trail** -- Ball emits multiple white fragments. Solution:
  dilate to merge nearby blobs, use `_find_ball_head()` to locate the most
  circular sub-contour within the merged blob.
- **`_find_ball_head` scoring** -- `circularity^2 * area` to prevent large
  elongated trail blobs from outscoring the smaller but circular ball.
- **Ball-in-brick false positive** -- White/gray bricks merge into giant
  blobs during dilation. Fix: pass `brick_detections` to `_detect_ball()`
  and zero out brick bounding boxes (with 4px padding) from the white mask.
- **UI false positives** -- "Level 1/7" button, "$0/$10" score, coin
  counter icon all trigger false detections. Eliminated via UI mask zones +
  frame differencing.
- **Game zone boundaries** -- Columns ~324 to ~956 (632px wide) at
  1280x1024. Individual bricks ~89x89 pixels. Client area = 1264x1016
  pixels.
- **`detect_to_game_state()` has Breakout-specific grouping logic** --
  groups detections into `"paddle"`, `"ball"`, `"bricks"`, `"powerups"`.
  Each game's env interprets the returned dict in its own
  `build_observation()`.
- **`_DEFAULT_CLASSES` is now `[]`** (empty list) -- game_classes() from
  the plugin provides them via BaseGameEnv during `_lazy_init()`.

## XPU / OpenVINO / Hardware Acceleration

- **Ultralytics 8.4.14 XPU support requires 3 monkey patches** in
  `_patch_ultralytics_xpu()`: `select_device()`, `GradScaler("cuda")`,
  `_get_memory()`. XPU patches must be idempotent (`_applied` flag).
- **`pip install ultralytics` overwrites XPU torch** -- ultralytics pulls
  CPU-only torch from PyPI; must reinstall XPU torch after every
  ultralytics install.
- **4 additional ultralytics XPU issues** (training works without):
  `_clear_memory()`, `autocast()`, `check_amp()`, OOM handler -- all
  hard-code CUDA.
- **Posted on ultralytics #16930** -- detailed comment with all 7 XPU
  fixes.
- **OpenVINO export paradigm** -- `YOLO("best.pt").export(format="openvino")`
  creates `best_openvino_model/` directory. Load with
  `YOLO("best_openvino_model/")`. Inference-only (no training).
- **Ultralytics OpenVINO GPU routing** -- Requires
  `device="intel:<OV_DEVICE>"` prefix (e.g., `"intel:GPU.0"`). Without
  prefix, OpenVINO defaults to CPU.
- **`_resolve_openvino_device()`** -- Queries
  `ov.Core().available_devices` at runtime. Available devices on dev
  machine: `['CPU', 'GPU.0', 'GPU.1']`.
- **OpenVINO warmup issue** -- Ultralytics runs internal warmup on CPU,
  then recompiles for GPU.0 on first real call. Fix: explicit warmup
  inference in `load()` with correct `device` kwarg.
- **`pydirectinput.PAUSE` bottleneck** -- Default `PAUSE = 0.1` adds
  100ms sleep to every input call. Fix: `pydirectinput.PAUSE = 0`.
- **wincam library** -- Direct3D11-based screen capture (<1ms async frame
  reads). Singleton constraint (1 DXCamera at a time). No CI/headless
  support (requires GPU + display). Falls back to `WindowCapture` in CI.
- **wincam dependency conflict** -- Pulls `opencv-contrib-python` which
  conflicts with `opencv-python`. Fixed by reinstalling.
- **SB3 PPO doesn't support XPU** -- `train_rl.py` maps `"xpu"` ->
  `"cpu"` for the policy network.
- **Multi-GPU strategy** -- Dedicate GPU.0 to YOLO inference, GPU.1 to RL
  policy/other compute.

## Breakout 71 Game Mechanics

- **Scoring is coin-based, not brick-based** -- breaking bricks spawns
  coins that fly with physics; catching coins with the paddle adds to the
  score. Combo system multiplies coin value. Combo resets if ball returns
  to paddle without hitting a brick.
- **No traditional lives** -- the `extra_life` perk (max 7 levels) acts as
  expendable rescues. When `extra_life == 0` and all balls lost -> game
  over.
- **Multi-ball is a perk** -- `multiball` perk spawns additional balls.
  Losing one ball isn't game over unless ALL are lost.
- **Level system** -- 7 + `extra_levels` levels per run. Between levels:
  perk selection screen (`openUpgradesPicker()`). ~60+ perks available.
  `chill` perk = infinite levels.
- **No explicit state machine** -- uses boolean flags: `running`,
  `isGameOver`, `ballStickToPuck`.
- **Input** -- mouse position sets paddle directly; keyboard
  `ArrowLeft`/`ArrowRight` move incrementally (`gameZoneWidth/50` per tick,
  3x with Shift); `Space` toggles play/pause.
- **Canvas** -- `#game` element, fills `innerWidth x innerHeight` (x
  pixelRatio). Game zone width = `brickWidth * gridSize`, centered
  horizontally.
- **Level completion** -- when `remainingBricks === 0 && !hasPendingBricks`,
  either instant win (no coins) or 5s delay (`winAt` timer) for coin
  collection.
- **Ball physics** -- speed normalizes toward `baseSpeed * sqrt(2)` each
  tick; multiple sub-steps per frame to prevent tunneling; bounce angle
  depends on paddle hit position and `concave_puck` perk.
- **`ballStickToPuck=true` at start** -- Ball sticks to paddle and follows
  it until first click.
- **`puckPosition` coordinate system** -- Canvas pixels from left edge.
  Game zone centered with `offsetX`. Clamp bounds:
  `offsetX + puckWidth/2` to `offsetX + gameZoneWidth - puckWidth/2`.
- **Game state detection via DOM** --
  `document.body.classList.contains('has-alert-open')` for modal detection,
  `#popup` for content, `#close-modale` for dismissible modals.
- **Random bot dies quickly** -- Never clears level 1, so dataset is 93.7%
  gameplay + 6.3% game-over (no perk picker states).

## Environment Architecture

- **Base step() flow**: `apply_action()` -> `_should_check_modals()` ->
  `handle_modals()` -> Capture + detect -> `build_observation()` ->
  `_check_late_game_over()` (updates `_no_ball_count`) ->
  `check_termination()` (updates `_no_bricks_count`, reads
  `_no_ball_count`) -> `compute_reward()` -> `_make_info()` ->
  `build_info()` -> `_run_oracles()`.
- **`_no_ball_count` ownership** -- `_check_late_game_over()` exclusively
  owns updates (increment when missing, reset when found).
  `check_termination()` only reads. Fixed in PR #59.
- **`_should_check_modals` defaults to `True`** -- Safe for new games.
  Breakout71 overrides to throttle based on ball visibility
  (`_no_ball_count > 0`).
- **Episode boundary bug (fixed PR #51)** -- `step()` called
  `_handle_game_state()` which dismissed game-over modals before frame
  capture. Multiple game lives merged into one infinite episode. Fixed:
  `step()` passes `dismiss_game_over=False`, `reset()` passes `True`.
- **Terminal penalty for modal-occluded frames** -- Game-over early-return
  uses fixed `-5.01` instead of computing reward from YOLO detections
  (which see modal overlay, not game state).
- **Modal check throttling** -- Only check for modals when
  `_no_ball_count > 0` (ball already missing). Saves ~100-150ms Selenium
  HTTP round-trip per step during normal gameplay.
- **Late check on 0->1 ball-miss transition** -- Catches game-over modals
  appearing in the same frame the ball vanishes.
- **`close()` does NOT close the WebDriver** -- Caller owns the driver
  lifecycle (typically `BrowserInstance`). Env only nulls its references.
- **Continuous action space** -- `Box(-1, 1, shape=(1,), dtype=float32)`.
  Action value linearly maps to absolute paddle pixel position via
  `SET_PUCK_POSITION_JS`.
- **`coins_norm` and `score_norm` hardcoded 0.0** -- 2 of 8 MLP
  observation dimensions carry no info. SB3 learns to ignore them.

## Plugin System

- **Plugin location: `games/` (top-level)** -- Games are not part of the
  platform package. Minimal plugin ~40 lines for a new game.
- **`load_game_plugin(name)`** -- uses
  `importlib.import_module(f"games.{name}")`, validates 5 required
  attributes (`env_class`, `loader_class`, `game_name`, `default_config`,
  `default_weights`).
- **Circular import fix** -- Eager re-exports in
  `src/game_loader/__init__.py` created a cycle. Fixed with lazy
  `__getattr__` in both `src/env/__init__.py` and
  `src/game_loader/__init__.py`.
- **Config files stay in `configs/`** -- `load_game_config()` and
  `load_training_config()` search `configs/games/` and
  `configs/training/` by default. ~72 call sites depend on current paths.
- **JS snippet deduplication** -- 4 JS constants consolidated into
  `games/breakout71/modal_handler.py` as single source of truth.

## Testing Patterns

- **`test_raises_without_weights` fragile** -- Test assumed no weights
  file existed, but `weights/breakout71/best.pt` exists locally. Fixed by
  overriding `cfg["output_dir"]` to `tmp_path`.
- **pytest-cov configured** -- coverage runs automatically on `src/` only
  (scripts excluded). Config in `pyproject.toml`. `fail_under = 80`.
- **SessionRunner `_setup()` mock pattern** -- `_setup()` uses
  `self._plugin.env_class`, so mocking the import path no longer works.
  Fix: set `runner._plugin = mock.MagicMock()` with
  `runner._plugin.env_class = MockEnvCls` directly on the runner instance.
- **Roboflow API** -- `project.single_upload(annotation_path=...,
  annotation_labelmap=...)` works for YOLO `.txt` pre-labels.
  `search_all()` returns a generator of batches (lists of dicts), not a
  flat list.

## Performance Benchmarks

- **52 FPS end-to-end (session 21)** -- Capture 4.4ms (wincam) + YOLO
  14.4ms (OpenVINO GPU.0) + Input 0.1ms = ~19ms/frame.
- **55 FPS raw pipeline (session 24)** -- Capture 3.3ms + YOLO 13.9ms +
  Input 0.4ms = ~18ms/frame. SB3 PPO overhead reduces to ~8-10 FPS.
- **YOLO CPU performance** -- ~130-190ms per frame (~5-7 FPS).
- **Headless mode** -- ~2.3 FPS due to Selenium screenshot overhead
  (~400ms/call).
- **MLP 9.6 FPS / CNN 8.4 FPS** on xpu:1 (session 25).

## RL Training Features

- **`--no-mute` flag** -- Game audio muted by default via localStorage.
- **`--headless` mode** -- Selenium screenshots instead of Win32 capture.
- **`--orientation portrait|landscape`** + `--window-size WxH` -- Portrait
  768x1024 is default.
- **Rich structured logging (`TrainingLogger`)** -- JSONL event log +
  human-readable console log + `training_summary.json`.
- **`--max-time SECONDS` / `--max-episodes N`** -- Clean shutdown via SB3
  callback.
- **`--data-collection`** -- Opt-in frame collection for YOLO retraining.
- **`--policy mlp|cnn`** -- MLP uses 8-dim YOLO features, CNN uses 84x84
  grayscale with VecFrameStack(4) + VecTransposeImage.
- **`--game` flag** -- Dynamic plugin loading (defaults to `"breakout71"`).

---

## Session History

### Sessions 1-7: Foundation (PR #4 - #15)
1. Perplexity research (capture, input, RL, market analysis)
2. Project scaffolding, game loader subsystem, CI pipeline
3. Reporting subsystem
4. Capture & Input subsystem
5. Perception subsystem (YoloDetector)
6. Oracle `on_step` detection logic, 12 oracles
7. Smoke scripts, integration tests, Selenium browser management

### Sessions 8-12: Game Study & YOLO Pipeline (PR #17 - #26)
8. Game source study, revised env design, Breakout71Env v1
9. Config-driven YOLO training pipeline: capture, upload, train, validate
10. Data collection pipeline: 300-frame capture, auto-annotation with OpenCV
11. Annotation improvements: 100% ball detection, Roboflow upload
12. XPU YOLO training: 3 ultralytics monkey patches, 100-epoch training (mAP50=0.679)

### Sessions 13-16: Integration & RL Scaffold (PR #28 - #32)
13. Orchestrator (FrameCollector, SessionRunner), train_rl.py, 6 env bug fixes, pytest-cov (96%)
14. Selenium-based env control replacing pydirectinput
15. First RL training attempt: pipeline validated, diagnosed Discrete(3) action space issue
16. Continuous action space Box(-1,1), coverage enforcement (fail_under=80)

### Sessions 17-18: Game Specifications (PR #34 - #37)
17. Deep source analysis, TypeDoc, game technical spec (~80 fields, 63 perks)
18. Player behavior spec: 80+ behavioral contract IDs

### Sessions 19-25: Performance & Training Features (PR #39 - #55)
19. Pixel-based debug loop: 4-phase pipeline validation
20. XPU auto-detection, OpenVINO & DXcam research
21. OpenVINO + wincam: 52 FPS end-to-end
22. RL training features: mute, headless, orientation, logging, max-time
23. Episode boundary bug fix, TDD convention
24. Modal check throttling, 55 FPS pipeline, alpha release v0.1.0a1
25. CNN policy pipeline (`--policy cnn|mlp`, VecFrameStack)

### Sessions 26-29: Platform Architecture (PR #57 - #64)
26. Selenium-only input, platform architecture planning, reward strategy spec
27. BaseGameEnv ABC extraction (13 abstract methods + 2 hooks)
28. Platform package creation (`src/platform/`)
29. Game plugin directory (`games/breakout71/`)

### Session 30: Documentation Restructure
30. AGENTS.md rewrite (lean operational guide), knowledge base extraction,
    ROADMAP.md creation, checklist update with Phase 1 training tasks,
    agent leadership transition.
