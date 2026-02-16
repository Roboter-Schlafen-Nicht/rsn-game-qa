# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0a1] — 2026-02-16

First alpha release. All core subsystems implemented and tested (594 tests).

### Game Loader
- YAML-driven game configuration (`configs/games/`)
- Browser lifecycle management via Selenium (Chrome, Edge, Firefox)
- Parcel dev server integration for browser-based games
- Breakout 71 as first supported target

### Capture & Input
- `WindowCapture` — GDI `PrintWindow` with `PW_RENDERFULLCONTENT` for Chromium
- `WinCamCapture` — Direct3D11 desktop duplication (<1ms async frame reads)
- `InputController` — pydirectinput wrapper with `PAUSE=0` optimization

### Perception
- `YoloDetector` — Ultralytics YOLO wrapper with auto device detection (XPU > CUDA > CPU)
- OpenVINO inference acceleration (~14ms/frame on Intel Arc A770 GPU)
- OpenVINO model export script (`scripts/export_openvino.py`)
- 5-class detection for Breakout 71: brick, paddle, ball, powerup (coin), wall

### Oracles
- 12 bug-detection oracles with `on_step` interface: Crash, Stuck, ScoreAnomaly, VisualGlitch, Performance, Physics, Boundary, StateTransition, EpisodeLength, Temporal, Reward, Soak

### Gymnasium Environment
- `Breakout71Env` — continuous `Box(-1, 1)` action space, `Box(8,)` observation
- Selenium-based paddle control via JS `puckPosition` injection
- Modal handling (game over, perk picker, menu) via DOM inspection
- Headless mode support (Selenium screenshots + ActionChains)
- Portrait (768x1024) and landscape orientation support

### RL Training
- SB3 PPO integration (`scripts/train_rl.py`)
- `TrainingLogger` — structured JSONL events + human-readable console log
- `FrameCollectionCallback` — capture frames during training for YOLO dataset
- Clean shutdown via `--max-time` with SB3 callback
- Audio mute by default, `--no-mute` flag

### YOLO Training Pipeline
- Config-driven pipeline: capture → auto-annotate → upload → train → validate
- OpenCV auto-annotation (HSV segmentation, frame differencing)
- Roboflow integration for annotation management
- XPU training support with 3 ultralytics monkey patches
- Dataset preparation script (`scripts/prepare_dataset.py`)

### Orchestration
- `FrameCollector` — periodic frame capture with auto-annotation bridge
- `SessionRunner` — multi-episode QA evaluation sessions
- `run_session.py` CLI for N-episode runs with JSON/HTML reporting

### Reporting
- `EpisodeReport` / `ReportGenerator` — structured JSON reports
- `DashboardRenderer` — Jinja2 HTML dashboard

### CI
- GitHub Actions: Lint (ruff), Test (pytest), Build Check, Build Docs (Sphinx)
- Pre-commit hook via `act` (Docker-based local CI)
- pytest-cov with 80% minimum threshold enforced

### Performance
- 52 FPS end-to-end (wincam + OpenVINO GPU + pydirectinput)
- ~6 FPS with Selenium modal handling every step
- ~2.3 FPS in headless mode

[0.1.0a1]: https://github.com/Roboter-Schlafen-Nicht/rsn-game-qa/releases/tag/v0.1.0a1
