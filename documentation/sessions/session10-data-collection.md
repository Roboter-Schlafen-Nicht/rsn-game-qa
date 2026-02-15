# Session 10 — Data Collection Pipeline & Auto-Annotation

**Date**: 2026-02-15
**PR**: #21
**Branch**: `feature/data-collection-pipeline`

## Goal

Capture ~300 frames from the live Breakout 71 game and auto-annotate them using OpenCV color segmentation + frame differencing, preparing for human review in Roboflow before YOLO training.

## What Was Done

### New Files (1)
- `scripts/auto_annotate.py` — Auto-annotation script: HSV color segmentation (11 color groups covering all 21 palette colors), frame differencing for motion-based ball/coin detection, UI mask zones, ball trail handling, brick-zone restriction for white/gray, deduplication of overlapping detections

### Modified Files (1)
- `configs/games/breakout-71.yaml` — Changed `window_height: 720` → `window_height: 1024` (taller window maximizes game area)

### Enhanced Files (1)
- `scripts/capture_dataset.py` — Major rewrite: JS-based game state detection (modal detection via DOM), auto-dismiss game-over and perk-picker screens, random paddle actions during gameplay, state-tagged frame metadata in manifest

### Captured Data (gitignored)
- `output/dataset_20260215_175739/` — 300 frames + 297 YOLO label files + 297 visualization overlays + manifest.json
- Detection stats: 222 paddles, 278 balls, 4445 bricks, 987 coins, 566 walls

### Design Decisions
- **Hybrid annotation**: Auto-annotate programmatically first, then upload to Roboflow for human review/correction
- **Window height 1024**: Maximizes game area between paddle and bricks; must be consistent for capture, annotation, and future RL training
- **11 HSV color groups**: Cover all 21 palette colors defined in `palette.json` (blue, yellow, red, orange, green, cyan, purple, pink, beige, white, gray)
- **Brick-zone restriction**: White/gray bricks only detected in upper 65% of game area to avoid false-positiving on paddle, ball, or background
- **Deduplication pass**: Removes overlapping detections when a brick's hue falls on boundary between two HSV ranges

### Game Source Study Findings
- Game UI uses `asyncAlert()` modal system — all screens (game-over, perk selection, menus) are DOM-based popups
- `document.body.classList.contains('has-alert-open')` detects ANY modal
- Selenium `execute_script()` requires `return` before IIFE — without it, returns `null`
- Browser chrome is ~130px at top of captured frames
- Individual bricks are ~89x89 pixels at 1280x1024
- Game zone columns ~324 to ~956 (632px wide)
- Client area at 1280x1024 window = 1264x1016 pixels

### Copilot Review (6 comments, all addressed)
1. Replaced hard-coded 0.5 with `MOTION_OVERLAP_BALL_CONFIRMED` constant
2. Fixed docstring for motion mask threshold description
3. Fixed paddle detection docstring accuracy
4. Removed unreachable `STATE_PAUSED` branch from capture bot
5. Fixed comment describing game-over dismiss strategy
6. Fixed comment about modal capture timing

## Key Discoveries
- Selenium `execute_script()` IIFE bug: Must use `return (function() { ... })();`
- Browser chrome ~130px affects frame coordinates
- All 21 game palette colors from `palette.json` mapped to 11 HSV detection ranges
- Ball has particle trail effects — dilate to merge, pick smallest confirmed candidate
- UI false positives (level menu, score display, coin counter) eliminated via mask zones + frame differencing
- Random bot dies quickly (never clears level 1) — only gameplay (93.7%) and game-over (6.3%) states captured

## Test Count
- Before: 398 unit + 24 integration = 422 total
- After: 398 unit + 24 integration = 422 total (no new tests — scripts-only, manual pipeline)
- Delta: +0 tests

## Next Steps
1. **Upload 300 annotated frames to Roboflow** for human review/correction
2. **Train YOLO model** on reviewed annotations
3. **Integration & E2E** — wire all subsystems end-to-end
