# Technical Intake Workflow: Adding a New Game

How RSN Game QA onboards a new game title. This workflow is pixel-based --
no source code access required.

## Prerequisites

- RSN platform installed (conda env `yolo`, all dependencies)
- Game runs in a window on Windows (browser, native, or emulated mobile)
- YOLO base weights available (`yolov8n.pt`)
- Roboflow account for annotation review

## Phase 1: Game Study & Setup (8-16 hours)

**Goal:** Understand the game well enough to define what to detect and how
to control it.

| Step | Task | Output |
|------|------|--------|
| 1.1 | Play the game manually (1-2 hours) | Notes on mechanics, UI states, win/lose conditions, input method |
| 1.2 | Define YOLO classes | `configs/training/<game>.yaml` with class list (typically 5-15 classes) |
| 1.3 | Create game loader config | `configs/games/<game>.yaml` (launch command, window title, resolution) |
| 1.4 | Verify capture pipeline | Run `smoke_launch.py` -- confirm PrintWindow captures the game window correctly |
| 1.5 | Document game mechanics | Game spec: scoring, lives/health, level progression, input method, key UI states |

### Key decisions in Phase 1

- **YOLO class list:** Start minimal. For Breakout 71 we use 5 classes
  (paddle, ball, brick, powerup, wall). Resist the urge to add classes for
  every UI element -- each class needs training data.
- **Input method:** Determine whether the game uses mouse position
  (continuous), keyboard (discrete), or both. This dictates action space
  design in Phase 3.
- **Modal/overlay detection:** Identify any DOM-based overlays (menus,
  game-over screens, perk pickers) that block gameplay. These need
  Selenium-based dismissal logic.
- **Genre template:** Check if a genre template exists (see issue #41).
  If so, inherit defaults for observation space, action space, reward
  structure, and oracle configuration.

### Lesson learned

> Always study the actual game before designing. Our original Breakout 71
> env spec was based on assumptions about how breakout games work. After
> reading the source, we discovered scoring is coin-based (not brick-based),
> there are no traditional lives, and mouse is the primary input. Budget
> research time explicitly -- it prevents building the wrong thing.

## Phase 2: Perception Training (16-24 hours)

**Goal:** Train a YOLO model that reliably detects game objects from
screenshots.

| Step | Task | Tooling | Output |
|------|------|---------|--------|
| 2.1 | Capture 300-500 frames with random bot | `capture_dataset.py` | Raw frames covering gameplay, menus, transitions |
| 2.2 | Auto-annotate with OpenCV | `auto_annotate.py` | YOLO `.txt` labels (game-specific HSV ranges needed) |
| 2.3 | Upload to Roboflow | `upload_to_roboflow.py` | Pre-labeled dataset for human review |
| 2.4 | Human review & correct annotations | Roboflow web UI | Corrected bounding boxes |
| 2.5 | Prepare dataset | `prepare_dataset.py` | Train/val split in YOLO format |
| 2.6 | Train YOLO model | `train_model.py` | `weights/<game>/best.pt` |
| 2.7 | Validate mAP thresholds | `validate_model.py` | mAP50 >= 0.65 (first pass) |

### The annotation bottleneck

Human review on Roboflow (step 2.4) is the biggest time sink: 8-12 hours
for 300 frames on the first game. This is where genre templates and
improved auto-annotation will have the highest ROI.

Current auto-annotation uses OpenCV HSV segmentation tailored to
Breakout 71's palette. For new games, the HSV ranges and object detection
heuristics must be adapted. Future work: train a lightweight "universal
game object" pre-annotator that bootstraps labels across genres.

### Quality thresholds

| Metric | First pass (auto-annotated) | After human review |
|--------|----------------------------|--------------------|
| mAP@0.5 | >= 0.65 | >= 0.80 |
| mAP@0.5:0.95 | >= 0.50 | >= 0.60 |

## Phase 3: Environment & Oracles (8-16 hours)

**Goal:** Build the Gymnasium environment and configure bug-detection
oracles.

| Step | Task | Output |
|------|------|--------|
| 3.1 | Define observation space | Mapping from YOLO detections to observation vector |
| 3.2 | Define action space | Mouse/keyboard mapping (continuous `Box` vs `Discrete`) |
| 3.3 | Define reward function | Based on YOLO-observable state changes only |
| 3.4 | Implement game-specific Env | `src/env/<game>_env.py` (Gymnasium wrapper) |
| 3.5 | Configure oracles | Select from 12 oracles, set game-specific thresholds |
| 3.6 | Validate with debug loop | 4-phase: capture, detect, control, gameplay |

### Observation space design principles

- **Only use what YOLO can see.** No JS injection for game state.
  If YOLO can't detect it, the agent can't observe it.
- **Normalize to [0, 1].** All positions relative to game zone dimensions.
- **Include velocity estimates.** Compute from consecutive frame
  detections (ball_vx, ball_vy).
- **Placeholder dimensions are OK.** Breakout 71 has `coins_norm` and
  `score_norm` hardcoded to 0.0 because we can't yet read those from
  pixels. SB3 learns to ignore them.

### Reward function design principles

- **Observable changes only.** Reward brick count decreasing (YOLO
  detects fewer bricks), not internal score changes.
- **Time penalty.** Small negative reward per step to encourage progress.
- **Terminal rewards.** Large negative for game-over, large positive for
  level-clear.
- **Iterate empirically.** The first reward function will be wrong. Plan
  for 2-3 rounds of adjustment after watching agent behavior.

### Oracle configuration

All 12 oracles are parameterized. Per-game tuning:

| Oracle | What to tune |
|--------|-------------|
| StuckOracle | Frozen-frame threshold (depends on game speed) |
| ScoreAnomalyOracle | Score observation index, anomaly threshold |
| PerformanceOracle | FPS threshold (depends on YOLO speed) |
| PhysicsViolationOracle | Object speed/acceleration limits |
| BoundaryOracle | Game zone boundaries |
| EpisodeLengthOracle | Expected episode duration range |

## Phase 4: RL Training & QA (16-40 hours)

**Goal:** Train an RL agent and generate QA reports.

| Step | Task | Tooling | Output |
|------|------|---------|--------|
| 4.1 | Train PPO baseline (~200k steps) | `train_rl.py` | Initial policy |
| 4.2 | Evaluate vs random baseline | `run_session.py` | Session report with oracle findings |
| 4.3 | Iterate reward shaping (2-3 rounds) | Developer | Adjusted reward function |
| 4.4 | Retrain YOLO with RL-collected frames | Full pipeline | Improved model for agent-discovered states |
| 4.5 | Generate QA reports | SessionRunner | HTML dashboard with findings |

### The YOLO-RL feedback loop

RL training generates frames from states the random bot never reaches
(later levels, rare game states). These frames improve YOLO training data,
which improves observations, which improves RL performance. Plan for at
least one full retrain cycle.

## Total Time Estimate

| Phase | Hours | Notes |
|-------|-------|-------|
| Phase 1: Game Study & Setup | 8-16 | Less if genre template exists |
| Phase 2: Perception Training | 16-24 | Annotation review is the bottleneck |
| Phase 3: Environment & Oracles | 8-16 | Less if genre template provides defaults |
| Phase 4: RL Training & QA | 16-40 | Depends on game complexity and iteration |
| **Total** | **48-96** | First game; subsequent similar games ~50-60% of this |

## What's Reusable vs. Game-Specific

| Reusable (platform) | Game-specific (per intake) |
|---------------------|---------------------------|
| WindowCapture pipeline | YOLO class list & training data |
| YoloDetector inference | HSV ranges for auto-annotation |
| All 12 oracles (parameterized) | Observation space mapping |
| FrameCollector, SessionRunner | Action space & input mapping |
| Reporting (JSON, HTML dashboard) | Reward function |
| Training pipeline scripts | Game loader config |
| Roboflow integration | Env wrapper |
| Debug loop pattern | Oracle threshold tuning |
