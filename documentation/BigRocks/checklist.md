# RSN Game QA -- Master Checklist

## Prerequisites (every feature)

- [ ] Activate conda env `yolo` (`conda activate yolo`)
- [ ] Verify required packages installed; add missing ones to `environment.yml`
- [ ] Create feature branch from `main`

## Completed

- [x] **Game Loader subsystem** (PR #4, PR #6)
  - [x] YAML config loading with env-var expansion
  - [x] `GameLoader` ABC, `BrowserGameLoader`, `Breakout71Loader`
  - [x] Factory pattern with loader registry
  - [x] TCP + HTTP readiness probing
  - [x] 82 tests, Sphinx docs, spec
- [x] **Reporting subsystem** (PR #7)
  - [x] `FindingReport`, `EpisodeMetrics`, `EpisodeReport`, `SessionReport` dataclasses
  - [x] `ReportGenerator` -- `compute_summary`, `save` (JSON), `to_dict`, `session` property
  - [x] `DashboardRenderer` -- Jinja2 HTML dashboard with built-in Bootstrap 5 template
  - [x] 26 tests, Sphinx docs
- [x] **Reporting spec alignment** (PR #8)
  - [x] Updated spec to match implementation (severity naming, `build_id`, `seed`)
  - [x] README updated to remove `[stub]` from reporting
- [x] **Capture & Input subsystem** (PR #9)
  - [x] `WindowCapture` -- BitBlt/GDI via pywin32 (`_find_window`, `capture_frame`, `is_window_visible`, `release`)
  - [x] `InputController` -- pydirectinput (`apply_action`, `move_mouse_to`, `click`, `press_key`, `hold_key`, `release_key`)
  - [x] 37 tests (133 total), Sphinx docs, `pydirectinput` added to `autodoc_mock_imports`
- [x] **Perception subsystem** (PR #12)
  - [x] `YoloDetector` -- `load` (weights validation, XPU→CPU fallback), `detect` (inference + result parsing), `detect_to_game_state` (grouping, normalization)
  - [x] `breakout_capture.py` -- `grab_frame` (visibility check), `detect_objects` (auto-infer dimensions)
  - [x] 41 tests (170 total), Sphinx docs updated
- [x] **Oracle `on_step` detection logic** (PR #13)
  - [x] Implemented `on_step` for 5 original oracles: CrashOracle, StuckOracle, ScoreAnomalyOracle, VisualGlitchOracle, PerformanceOracle
  - [x] Added 7 new research-backed oracles: PhysicsViolationOracle, BoundaryOracle, StateTransitionOracle, EpisodeLengthOracle, TemporalAnomalyOracle, RewardConsistencyOracle, SoakOracle
  - [x] Finding dedup (black frame, frozen frame, negative score), flicker cooldown, cv2 import guards (Copilot review)
  - [x] 132 oracle tests (293 total), Sphinx docs, README updated
- [x] **Smoke scripts & integration tests** (PR #15)
  - [x] Selenium-based `BrowserInstance` (Chrome/Edge/Firefox) replacing subprocess approach
  - [x] Smoke scripts: `smoke_launch.py`, `smoke_capture.py`, `smoke_oracle.py`
  - [x] 12 integration tests × 2 browsers (Chrome, Firefox) — lifecycle, capture, oracle
  - [x] PrintWindow with `PW_RENDERFULLCONTENT` for GPU-composited windows + BitBlt fallback
  - [x] `window_width`/`window_height` config fields, `pyproject.toml` pytest config
  - [x] selenium added to `environment.yml`, guarded import (Copilot review)
  - [x] 23 integration tests pass + 293 unit tests (316 total)

## To Do

### 1. Data collection & YOLO training pipeline
- [ ] Automated frame capture during gameplay (save frames + metadata)
- [ ] Annotation tooling integration (Roboflow upload / labeling)
- [ ] YOLO training script for Breakout 71 classes (`paddle`, `ball`, `brick`, `powerup`, `wall`)
- [ ] Trained weights validation (mAP threshold)

### 2. Breakout71 Gymnasium environment v1 (`src/env/`) — session 8
- [x] Check `environment.yml` has `gymnasium` (already present)
- [x] Study Breakout 71 game source to understand actual mechanics
  - [x] Discovered: coin-based scoring (not brick-based), no traditional lives, perk system, level transitions
  - [x] Android version is identical game logic in Kotlin WebView wrapper
- [x] Rewrite spec (`documentation/specs/breakout71_env_spec.md`) with actual game mechanics
- [x] Update observation space from 6 to 8 elements (add `coins_norm`, `score_norm` placeholders)
- [x] Implement `Breakout71Env` core methods
  - [x] `_lazy_init` -- wire capture + perception + input (lazy imports for CI safety)
  - [x] `_capture_frame` -- delegate to `WindowCapture.capture_frame()`
  - [x] `_detect_objects` -- delegate to `YoloDetector.detect_to_game_state()`
  - [x] `_build_observation(detections, *, reset=False)` -- extract positions, compute velocity, normalize bricks
  - [x] `_compute_reward(detections, terminated, level_cleared)` -- brick delta + time penalty + terminal rewards + score_delta placeholder
  - [x] `_apply_action` -- delegate to `InputController.apply_action()`
  - [x] `_run_oracles` -- call `on_step` on all oracles, collect findings
  - [x] `reset` -- lazy init, fire space, capture, build obs (reset=True), clear oracles
  - [x] `step` -- apply action, sleep, capture, detect, build obs, compute reward, check termination (ball-lost / level-cleared / max_steps), run oracles
- [x] Termination logic: ball not detected for N frames = game over; 0 bricks for M frames = level cleared
- [x] Replace placeholder tests with comprehensive unit tests (~50 tests, all mocked)
- [x] Update Sphinx docs if needed
- [x] Update README (remove `[stub]`, update test count)
- [ ] Commit, push, create PR, request review from Copilot, evaluate review and create issues if necessary, merge (`--delete-branch`), delete local branch
- [ ] Post-merge admin (no Copilot review needed):
  - [ ] Update this checklist (move item to Completed, record PR number)
  - [ ] Create session log in `documentation/sessions/session8-env.md`
  - [ ] Update `AGENTS.md` (session list, project structure, discoveries, what's next)

#### v1 scoping decisions
- **Episode = single level** (perk selection between levels not part of action space)
- **8-element observation**: paddle_x, ball_x, ball_y, ball_vx, ball_vy, bricks_norm, coins_norm (placeholder 0.0), score_norm (placeholder 0.0)
- **Reward**: brick-based for v1, score_delta slot prepared for future OCR/JS bridge
- **Action space**: Discrete(3) — NOOP, LEFT, RIGHT (keyboard arrow keys via InputController)
- **Out of scope for v1**: coin observation, score observation, perk selection, multi-level episodes, continuous action space, multiple ball tracking

### 3. Integration & end-to-end
- [x] Add integration tests (PR #15 — 12 tests × 2 browsers)
- [x] Add smoke scripts for manual verification (PR #15)
- [ ] Wire all subsystems: loader -> capture -> perception -> env -> oracles -> reporting
- [ ] Run Breakout 71 env for N episodes, generate session report + dashboard
- [ ] Update README and docs
- [ ] Commit, push, create PR, request review from Copilot, evaluate review and create issues if necessary, merge (`--delete-branch`), delete local branch
- [ ] Post-merge admin (no Copilot review needed):
  - [ ] Update this checklist (move item to Completed, record PR number)
  - [ ] Create session log in `documentation/sessions/sessionN-<topic>.md`
  - [ ] Update `AGENTS.md` (session list, project structure, discoveries, what's next)
