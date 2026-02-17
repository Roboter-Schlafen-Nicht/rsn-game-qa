# Breakout71 Gymnasium Environment Spec (v1)

> Revised in session 8 after studying the Breakout 71 source code.
> Replaces the session 1 spec which was based on assumptions about
> standard breakout games.

## Overview

`Breakout71Env` wraps the [Breakout 71](https://github.com/lecarore/breakout71)
browser game running in a native Windows window. It captures frames via
`WindowCapture` (PrintWindow/BitBlt GDI), runs YOLOv8 inference to extract
game-object positions, converts them to a structured observation vector, and
injects actions via `InputController` (pydirectinput).

## Actual Game Mechanics (from source study)

Understanding these mechanics is critical for correct env design. The original
spec assumed a standard Atari-style breakout; the real game differs
significantly.

### Scoring is coin-based

Breaking bricks does NOT directly add to the score. Instead:

1. Breaking a brick spawns **coins** (quantity based on combo multiplier)
2. Coins have physics — gravity, velocity, bouncing off bricks/walls
3. **Catching coins with the paddle** adds their value to the score
4. Coins lost off-screen are wasted points

The combo system multiplies coin value. Combo resets if the ball returns to
the paddle without hitting a brick.

### No traditional lives

There is no lives counter. Instead:

- The `extra_life` perk (max 7 levels) acts as expendable rescues
- When the last ball is lost with `extra_life > 0`, the ball is rescued and
  placed on the paddle; `extra_life` decrements
- When `extra_life == 0` and all balls are lost → game over
- The `multiball` perk spawns additional balls; losing one isn't game over
  unless ALL are lost

### Level system with perk selection

- A run consists of 7 + `extra_levels` levels (default 7)
- Between levels: a perk selection screen (`openUpgradesPicker()`) offers
  3+ choices from ~60 available perks
- Level completes when `remainingBricks === 0 && !hasPendingBricks`, with
  an optional 5-second delay for coin collection
- The `chill` perk = infinite levels

### Input

- **Mouse** (primary): `mousemove` sets `puckPosition` directly
- **Keyboard** (secondary): `ArrowLeft`/`ArrowRight` move paddle incrementally
  (`gameZoneWidth/50` per tick, 3x with Shift); `Space` toggles play/pause
- **Touch** (Android/mobile): hold to play, drag for paddle, release to pause

### Canvas

- `#game` element, fills `innerWidth x innerHeight` (times `pixelRatio`)
- Game zone width = `brickWidth * gridSize`, centered horizontally
- `puckWidth = (gameZoneWidth/12) * min(12, 3 - smaller_puck + bigger_puck)`

## v1 Scoping Decisions

| Decision | Rationale |
|---|---|
| **Episode = single level** | Perk selection requires a different action space; defer to v2 |
| **Keyboard input only** | `InputController` uses `pydirectinput`; mouse control is a v2 option |
| **Brick-based reward** | Coins can't be reliably tracked without trained YOLO; bricks can |
| **Single ball tracking** | Uses highest-confidence YOLO detection; multi-ball is a v2 extension |
| **Score/coin observation placeholders** | Slots reserved in the observation vector but defaulting to 0.0 |

## Observation Space

**Type:** `gymnasium.spaces.Box` (continuous, float32)

**Layout:** 8-element vector

| Index | Name          | Range       | Description                                        | v1 Status   |
|-------|---------------|-------------|----------------------------------------------------|-------------|
| 0     | `paddle_x`    | [0.0, 1.0]  | Normalised x-position of paddle centre             | Active      |
| 1     | `ball_x`      | [0.0, 1.0]  | Normalised x-position of ball centre               | Active      |
| 2     | `ball_y`      | [0.0, 1.0]  | Normalised y-position of ball centre               | Active      |
| 3     | `ball_vx`     | [-1.0, 1.0] | Estimated horizontal velocity (frame delta)        | Active      |
| 4     | `ball_vy`     | [-1.0, 1.0] | Estimated vertical velocity (frame delta)          | Active      |
| 5     | `bricks_norm` | [0.0, 1.0]  | Fraction of bricks remaining (count / initial)     | Active      |
| 6     | `coins_norm`  | [0.0, 1.0]  | Normalised coin count on screen (placeholder: 0.0) | Placeholder |
| 7     | `score_norm`  | [0.0, 1.0]  | Normalised score delta since last step (placeholder: 0.0) | Placeholder |

**Normalisation rules:**

- Positions: centre coordinate / frame dimension (already normalised by YOLO)
- Velocity: `delta = pos_t - pos_{t-1}`, clipped to [-1, 1]
- Bricks: `bricks_left / bricks_total` (total set on reset)
- Coins/score: placeholder at 0.0 for v1; will be populated when YOLO coin
  tracking or OCR/JS bridge is available

**Missing detections:** Default to 0.5 for positions, 0.0 for velocities.

## Action Space

**Type:** `gymnasium.spaces.Box(low=-1, high=1, shape=(1,), dtype=float32)`

The action is a single continuous value in ``[-1, 1]`` that maps directly to
the absolute paddle position via JavaScript ``puckPosition`` injection:

| Value | Paddle position                              |
|-------|----------------------------------------------|
| -1.0  | Left edge of the game zone                   |
|  0.0  | Centre of the game zone                      |
| +1.0  | Right edge of the game zone                  |

There is no FIRE action in the action space. Canvas clicks via Selenium
ActionChains are used during ``reset()`` to start a new game and may also
be triggered from ``step()`` when modal dialogs (e.g., game over,
perk picker, menu) appear mid-episode.

**Implementation:** The env queries the game's ``offsetX``,
``gameZoneWidth``, and ``puckWidth`` JavaScript globals to determine the
valid paddle range, then sets ``puckPosition = Math.round(pixel_x)``
directly.  This bypasses Selenium mouse events entirely for paddle
control, giving frame-accurate positioning.

## Reward Function

| Signal                | Value    | Condition                                | v1 Status |
|-----------------------|----------|------------------------------------------|-----------|
| Brick destroyed       | +scaled  | `(prev_bricks_norm - bricks_norm) * 10.0` | Active |
| Score delta           | +scaled  | `score_delta * 0.01`                     | Placeholder (0.0) |
| Time penalty          | -0.01    | Every step                               | Active |
| Game over (ball lost) | -5.0     | `terminated` and not level cleared       | Active |
| Level cleared         | +5.0     | All bricks destroyed                     | Active |

**Future enhancement (v2+):**

- Activate `score_delta` reward when OCR or JS bridge can read the score
- Curiosity bonus for visiting rarely-seen states
- Coin-catching reward when YOLO reliably detects coins near the paddle

## Episode Termination

| Condition              | Flag         | Detection                                            |
|------------------------|--------------|------------------------------------------------------|
| Game over (ball lost)  | `terminated` | Ball not detected for N consecutive frames (default 5) AND brick count unchanged |
| Level cleared          | `terminated` | Zero bricks detected for M consecutive frames (default 3) |
| Max steps reached      | `truncated`  | `_step_count >= max_steps`                           |

**Why "ball not detected for N frames" instead of ball y-position threshold:**

1. Ball can legitimately be near the bottom (bouncing off paddle)
2. When truly lost, the ball disappears from the frame entirely
3. Combined with "brick count unchanged" to distinguish from brief YOLO misses
4. Handles multi-ball edge cases (ball detection picks highest-confidence)

## YOLO Classes

| Class      | Description                             | Env usage                    |
|------------|-----------------------------------------|------------------------------|
| `paddle`   | The player paddle                       | `paddle_x` in observation    |
| `ball`     | The ball (highest-confidence if >1)     | `ball_x`, `ball_y`, velocity |
| `brick`    | All bricks (single class)               | `bricks_norm`, reward delta  |
| `powerup`  | Coins spawned by broken bricks          | `coins_norm` (v2)            |
| `wall`     | Optional: wall/ceiling bounds           | Not used in v1               |

## Environment Lifecycle

### Constructor

```python
Breakout71Env(
    window_title="Breakout - 71",
    yolo_weights="weights/best.pt",
    max_steps=10_000,
    render_mode=None,        # "human" or "rgb_array"
    oracles=None,            # list[Oracle] or None for empty
)
```

Sub-components (`WindowCapture`, `YoloDetector`) are NOT created in the
constructor — they are lazily initialised on first `reset()`.
This allows the env to be constructed and inspected (spaces, metadata) without
requiring a live game window (useful for testing, config validation, and CI).

The Selenium ``WebDriver`` is passed via the ``driver`` parameter and is
used for JavaScript-based paddle control (``puckPosition`` injection) and
game state detection/modal dismissal.

### `_lazy_init()`

Called once on first `reset()`. Imports and creates:

1. `WindowCapture(window_title=self.window_title)` — frame capture
2. `YoloDetector(weights_path=self.yolo_weights)` + `.load()` — object detection
3. Canvas element lookup via Selenium (`#game` or `<body>` fallback)
4. Game zone boundary query from JavaScript globals

All imports are performed inside the method body (not at module level)
so that the env module loads cleanly in Docker CI where `pywin32` is
unavailable.

### `reset(seed, options)`

1. Call `super().reset(seed=seed)` (Gymnasium API)
2. If not initialised: call `_lazy_init()`
3. Handle game state modals (game over, perk picker, menu) via JS execution
4. Click canvas via Selenium ActionChains to start/unpause
5. Brief sleep for game to start
6. Capture first frame via `_capture_frame()`
7. Detect objects via `_detect_objects(frame)`
8. Build observation with `reset=True` (sets `_bricks_total`, zeroes velocity)
9. Reset counters: `_step_count`, `_prev_bricks_norm`, `_no_ball_count`, `_no_bricks_count`
10. Re-query game zone boundaries from JavaScript globals
11. Clear and notify oracles: `oracle.clear()` + `oracle.on_reset(obs, info)`
12. Return `(obs, info)`

### `step(action)`

1. Apply action via `_apply_action(action)` — sets `puckPosition` via JS
2. Wait fixed interval: `time.sleep(1.0 / 30.0)` (~30 FPS)
3. Handle any modals that appeared mid-episode (perk picker, game over, menu)
4. Capture new frame
5. Detect objects
6. Build observation (velocity from frame delta)
7. Update termination counters:
   - Ball not detected → `_no_ball_count += 1`; else reset to 0
   - No bricks remaining → `_no_bricks_count += 1`; else reset to 0
7. Determine termination:
   - `level_cleared = _no_bricks_count >= _LEVEL_CLEAR_THRESHOLD`
   - `game_over = _no_ball_count >= _BALL_LOST_THRESHOLD`
   - `terminated = level_cleared or game_over`
   - `truncated = _step_count >= max_steps`
8. Compute reward
9. Increment `_step_count`
10. Build info dict
11. Run all attached oracles
12. Return `(obs, reward, terminated, truncated, info)`

### `render()`

Returns `self._last_frame` when `render_mode="rgb_array"`, else `None`.

### `close()`

Releases capture resources, sets sub-components to `None`.

## Internal State

```python
# Episode tracking
_step_count: int = 0
_prev_ball_pos: tuple[float, float] | None = None
_bricks_total: int | None = None           # set on first reset
_prev_bricks_norm: float = 1.0             # for reward delta

# Termination counters
_no_ball_count: int = 0                    # consecutive frames with no ball detection
_no_bricks_count: int = 0                  # consecutive frames with 0 bricks

# Constants
_BALL_LOST_THRESHOLD: int = 5             # frames without ball → game over
_LEVEL_CLEAR_THRESHOLD: int = 3           # frames with 0 bricks → level cleared

# Sub-components (lazy)
_capture = None                            # WindowCapture
_detector = None                           # YoloDetector
_input = None                              # InputController
_initialized: bool = False

# Oracle + render
_oracles: list[Oracle]
_last_frame: np.ndarray | None = None
```

## Info Dict

The `info` dict returned by `reset()` and `step()` contains:

| Key                | Type               | Description                              |
|--------------------|--------------------|------------------------------------------|
| `frame`            | `np.ndarray`       | BGR screenshot of the game               |
| `detections`       | `dict[str, Any]`   | Raw YOLO game state dict                 |
| `ball_pos`         | `list[float, float]` | Normalised [x, y] ball position or None |
| `paddle_pos`       | `list[float, float]` | Normalised [x, y] paddle position or None |
| `brick_count`      | `int`              | Number of bricks currently detected      |
| `step`             | `int`              | Current step count                       |
| `score`            | `float`            | Placeholder (0.0) — future: actual score |
| `oracle_findings`  | `list[Finding]`    | Findings from all oracles this step      |

## RL Training

- **Framework:** stable-baselines3 (DQN or PPO)
- **Hardware:** Intel Arc A770 GPU via XPU backend
- **Hyperparameters:** Start with defaults, tune reward scaling, action
  frequency, and episode length
- **Success criteria:** Clear improvement over random baseline in reward and
  episode length

## v2 Roadmap

When the data collection pipeline (YOLO training, OCR, or JS bridge) is ready:

1. **Populate `coins_norm`** — count YOLO `powerup` detections on screen
2. **Populate `score_norm`** — read score via OCR on the score text region,
   or inject JavaScript into the browser to read `gameState.score` directly
3. **Activate score_delta reward** — replace `0.0` with actual score changes
4. **Multi-level episodes** — handle perk selection screen (possibly via
   separate hierarchical action space or automated selection)
5. **Multiple ball tracking** — extend observation or use attention mechanism
6. **Continuous action space** — mouse-based paddle control for finer movement

## Source File

`src/env/breakout71_env.py`
