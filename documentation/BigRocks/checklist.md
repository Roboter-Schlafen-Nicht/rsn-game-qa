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
- [x] **Breakout71 Gymnasium environment v1** (PR #17)
  - [x] Game source study: coin-based scoring, no traditional lives, perk system, level transitions
  - [x] Full spec rewrite with actual game mechanics
  - [x] 8-element observation: paddle_x, ball_x, ball_y, ball_vx, ball_vy, bricks_norm, coins_norm, score_norm
  - [x] Reward: brick_delta × 10.0, time penalty, game_over/level_clear terminal rewards, score_delta placeholder
  - [x] Termination: ball-lost (5 frames), level-cleared (3 frames), max_steps truncation
  - [x] Lazy imports for CI safety (Docker lacks pywin32/pydirectinput)
  - [x] 69 unit tests (357 total unit + 24 integration)
  - [x] Copilot review: fixed step counter increment ordering (truncation bug)

- [x] **YOLO training pipeline** (PR #19)
  - [x] Config-driven architecture (`configs/training/<game>.yaml`)
  - [x] Frame capture with random bot (`scripts/capture_dataset.py`)
  - [x] Roboflow upload with resume support (`scripts/upload_to_roboflow.py`)
  - [x] YOLO training script (`scripts/train_model.py`)
  - [x] Trained weights validation (`scripts/validate_model.py`)
  - [x] `.env` / `python-dotenv` for API keys
  - [x] 41 unit tests (398 total unit + 24 integration)
  - [x] Copilot review: 8 fixes (flag semantics, zero checks, dead code, docstring, error recovery)
- [x] **Data collection pipeline & auto-annotation** (PR #21)
  - [x] Enhanced `capture_dataset.py` — JS-based game state detection, auto-dismiss modals, random paddle actions
  - [x] `auto_annotate.py` — HSV color segmentation (11 groups, 21 palette colors), frame differencing, UI masks, deduplication
  - [x] Captured 300 frames, auto-annotated 297 with 222 paddles, 278 balls, 4445 bricks, 987 coins, 566 walls
  - [x] Window height changed to 1024 for maximum game area
  - [x] Copilot review: 6 fixes (constants, docstrings, dead code, comment accuracy)
- [x] **Annotation pipeline improvements** (PR #23)
  - [x] Ball-in-brick false positive fix (brick bbox exclusion from white mask before dilation)
  - [x] `_find_ball_head` scoring changed to `circularity² * area`
  - [x] Paddle-zone exclusion fix (paddle bbox masking instead of fixed 15% cut) — 288/297 → 297/297 ball detection
  - [x] Game-zone wall detection fix (brightness > 200 peaks, edge column margin)
  - [x] Roboflow annotation upload support (`single_upload` with `annotation_path` + `_build_labelmap`)
  - [x] `classes.txt` generation in auto_annotate.py
  - [x] 8 new tests (49 training pipeline tests, 430 total)
  - [x] Copilot review: 8 fixes (docstrings, error handling, test tolerance, progress tracking, test count)

## To Do

### 1. Data collection & YOLO training pipeline
- [x] Automated frame capture during gameplay (PR #19)
- [x] Annotation tooling integration (PR #19 — Roboflow upload)
- [x] YOLO training script for Breakout 71 classes (PR #19)
- [x] Trained weights validation / mAP threshold (PR #19)
- [x] Config-driven architecture (PR #19)
- [x] `.env` / `python-dotenv` for API keys (PR #19)
- [x] 41 unit tests (PR #19)
- [x] Capture ~300 frames from live game (PR #21)
- [x] Auto-annotate frames with OpenCV (PR #21)
- [x] Upload annotated frames to Roboflow for human review/correction (PR #23)
- [x] Train and validate model (PR #26)

### 2. Integration & end-to-end
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
