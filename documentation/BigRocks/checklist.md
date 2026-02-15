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
- [x] **Perception subsystem** (PR #TBD)
  - [x] `YoloDetector` -- `load` (weights validation, XPU→CPU fallback), `detect` (inference + result parsing), `detect_to_game_state` (grouping, normalization)
  - [x] `breakout_capture.py` -- `grab_frame` (visibility check), `detect_objects` (auto-infer dimensions)
  - [x] 40 tests (169 total), Sphinx docs updated

## To Do

### 1. Data collection & YOLO training pipeline
- [ ] Automated frame capture during gameplay (save frames + metadata)
- [ ] Annotation tooling integration (Roboflow upload / labeling)
- [ ] YOLO training script for Breakout 71 classes (`paddle`, `ball`, `brick`, `powerup`, `wall`)
- [ ] Trained weights validation (mAP threshold)

### 2. Oracle `on_step` detection logic (`src/oracles/`)
- [ ] Check `environment.yml` for any new deps (imagehash, numpy -- already present)
- [ ] Implement `CrashOracle.on_step`
- [ ] Implement `StuckOracle.on_step`
- [ ] Implement `ScoreAnomalyOracle.on_step`
- [ ] Implement `VisualGlitchOracle.on_step`
- [ ] Implement `PerformanceOracle.on_step`
- [ ] Replace placeholder tests with real tests
- [ ] Update Sphinx docs if needed
- [ ] Update README (remove `[stub on_step]`, update test count)
- [ ] Commit, push, create PR, request review from Copilot, evaluate review and create issues if necessary, merge (`--delete-branch`), delete local branch
- [ ] Update this checklist (move item to Completed, record PR number)

### 3. Breakout71 Gymnasium environment (`src/env/`)
- [ ] Check `environment.yml` has `gymnasium` (already present)
- [ ] Implement `Breakout71Env` core methods
  - [ ] `_lazy_init` -- wire capture + perception + game loader
  - [ ] `reset` -- start episode, return initial observation
  - [ ] `step` -- apply action, capture, detect, build obs, compute reward, run oracles
  - [ ] `_capture_frame`, `_detect_objects`, `_build_observation`
  - [ ] `_compute_reward`, `_apply_action`, `_run_oracles`
- [ ] Replace placeholder tests with real tests
- [ ] Update Sphinx docs if needed
- [ ] Update README (remove `[stub]`, update test count)
- [ ] Commit, push, create PR, request review from Copilot, evaluate review and create issues if necessary, merge (`--delete-branch`), delete local branch
- [ ] Update this checklist (move item to Completed, record PR number)

### 4. Integration & end-to-end
- [ ] Wire all subsystems: loader -> capture -> perception -> env -> oracles -> reporting
- [ ] Run Breakout 71 env for N episodes, generate session report + dashboard
- [ ] Add integration tests
- [ ] Update README and docs
- [ ] Commit, push, create PR, request review from Copilot, evaluate review and create issues if necessary, merge (`--delete-branch`), delete local branch
- [ ] Update this checklist (move item to Completed, record PR number)
