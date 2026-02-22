# Hextris -- Environment Specification

Design decisions and technical details for the RSN Game QA Hextris
environment plugin (`games/hextris/`). Companion to
`hextris_game_spec.md`.

---

## Table of Contents

1. [Plugin Structure](#1-plugin-structure)
2. [Action Space](#2-action-space)
3. [Observation Space](#3-observation-space)
4. [Reward Design](#4-reward-design)
5. [Termination Conditions](#5-termination-conditions)
6. [State Detection](#6-state-detection)
7. [Modal Handling](#7-modal-handling)
8. [Game Loader](#8-game-loader)
9. [YOLO Bypass](#9-yolo-bypass)
10. [Training Configuration](#10-training-configuration)
11. [Key Design Decisions](#11-key-design-decisions)

---

## 1. Plugin Structure

```
games/hextris/
├── __init__.py          # Plugin metadata (5 required attrs + mute_js)
├── env.py               # HextrisEnv(BaseGameEnv), 557 lines
├── loader.py            # HextrisLoader(BrowserGameLoader), 125 lines
├── modal_handler.py     # 7 JS snippets, 185 lines
└── perception.py        # Empty YOLO class list (CNN-only)

configs/games/
└── hextris.yaml         # Game loader config (port 8271, portrait)
```

### Plugin Metadata (`__init__.py`)

| Attribute | Value |
|---|---|
| `game_name` | `"hextris"` |
| `env_class` | `HextrisEnv` |
| `loader_class` | `HextrisLoader` |
| `default_config` | `"hextris.yaml"` |
| `default_weights` | `None` (CNN-only, no YOLO) |
| `mute_js` | JS snippet to mute all `<audio>` elements |

---

## 2. Action Space

**Type:** `gymnasium.spaces.Discrete(3)`

| Value | Action | JS Implementation |
|---|---|---|
| 0 | No-op | (no script executed) |
| 1 | Rotate left | `MainHex.rotate(-1)` (counterclockwise) |
| 2 | Rotate right | `MainHex.rotate(1)` (clockwise) |

### Design Rationale

- Hextris has exactly two meaningful inputs: rotate left and rotate
  right. Adding no-op gives the agent the option to wait, which is a
  valid strategy (let blocks settle before rotating).
- The rush mechanic (Down/S key) is intentionally excluded. Rush only
  speeds up block descent and is a convenience feature, not a strategic
  input. For QA purposes, the agent should experience normal-speed
  gameplay to detect timing-related bugs.
- `Discrete(3)` is simpler than `Box` (continuous) and matches the
  discrete nature of hexagonal rotation (6 positions, not continuous
  angles).

### Action Execution

Actions are executed via Selenium `execute_script()`:

```javascript
// Rotate left
return (function() { MainHex.rotate(-1); return true; })();

// Rotate right
return (function() { MainHex.rotate(1); return true; })();
```

The 75ms rotation cooldown in the game source is not enforced for
JS-injected rotations. At ~15 FPS headless capture rate, steps are
~66ms apart, which is close to the cooldown anyway.

---

## 3. Observation Space

### CNN Mode (Primary)

**Type:** `gymnasium.spaces.Box(0, 255, shape=(84, 84, 1), dtype=uint8)`

The raw browser screenshot is converted to 84x84 grayscale. With
`--frame-stack 4`, the effective observation is `(84, 84, 4)` via
SB3's `VecFrameStack`.

CNN mode is the only supported observation mode. There is no MLP mode
because Hextris has no YOLO model and the game state globals (block
positions, lane heights) would require complex extraction logic with
minimal benefit over pixel observations.

### What the CNN Sees

- Central hexagon with colored blocks stacked on each side
- Falling blocks approaching from edges
- Score text at top center (rendered on canvas)
- Combo timer arc around hex (when active)
- Background color (`#ecf0f1`, light gray)
- Game-over screen blur overlay (when `gameState == 2`)

### Frame Capture

Frames are captured via Selenium's `toDataURL()` on the `#canvas`
element, consistent with the platform's headless capture pipeline.
The score is rendered on the canvas itself (via `ctx.fillText()`),
so it is visible in CNN observations.

---

## 4. Reward Design

### Survival Mode (Default)

```
reward = +0.01 per step (survival bonus)
       + (-5.01) on game over (terminal penalty)
```

- `survival_bonus`: Configurable via `--survival-bonus` CLI flag
  (default 0.01)
- Terminal penalty: Fixed at `-(5.0 + survival_bonus)` to ensure net
  negative lifetime reward for trivially short episodes

### Score Mode (`--reward-mode score`)

```
reward = +0.01 per step (survival base)
       + score_delta × score_reward_coeff
       + (-5.01) on game over
```

- Score is read from `window.score` via JS bridge
- `score_reward_coeff`: Configurable via `--score-reward-coeff`
  (default 0.01)
- `score_ocr_interval`: How often to read score (default 10 steps)
- Score region for OCR: `"540,402,200,80"` (validated in session 61)

Note: Hextris renders the score on canvas via `ctx.fillText()`, so
both OCR and JS bridge approaches work. The JS bridge
(`window.score`) is more reliable.

### Why Not YOLO-Based Reward

Hextris has no YOLO model trained for its visual elements. The game's
hexagonal geometry and overlapping block colors make YOLO detection
impractical compared to the readily available JS globals. The survival
and score reward modes provide sufficient training signal.

---

## 5. Termination Conditions

### Game Over (Terminal)

- **Primary:** `window.gameState == 2` detected via JS bridge
- **Confirmation:** Game-over state must persist for 3 consecutive
  frames to avoid false positives from transient state changes
- **Reward:** Terminal penalty `-(5.0 + survival_bonus)`
- **Info key:** `terminated = True`, `game_over = True`

### Max Steps (Truncation)

- Episode ends when `step_count >= max_steps`
- Default `max_steps`: 2000 (configurable via `--max-steps`)
- **Reward:** No additional penalty (just the final step's survival
  bonus)
- **Info key:** `truncated = True`

### Platform-Level Detectors

The `GameOverDetector` (pixel-based) can also trigger termination if
enabled via `--game-over-detector`. In practice, the JS-based detection
fires instantly, so the pixel detector rarely activates for Hextris.
Its value is as a fallback for non-DOM games.

---

## 6. State Detection

### JS Bridge (`modal_handler.py`)

**`DETECT_STATE_JS`:**

```javascript
return (function() {
    var state = window.gameState;
    if (state === undefined || state === null) return 'loading';
    if (state === 2) return 'game_over';
    if (state === -1) return 'paused';
    if (state === 0) return 'start_screen';
    if (state === 1) return 'playing';
    return 'unknown';
})();
```

Returns one of: `'loading'`, `'game_over'`, `'paused'`,
`'start_screen'`, `'playing'`, `'unknown'`.

### Game State Reading (`READ_GAME_STATE_JS`)

```javascript
return (function() {
    return {
        score: window.score || 0,
        gameState: window.gameState,
        blockCount: (window.blocks || []).length
    };
})();
```

Used by the score reward mode to read the current score each step.

### Confirmation Logic

Game-over detection requires 3 consecutive frames returning
`'game_over'` before the episode terminates. This prevents false
positives from:
- Momentary state transitions during `init()` calls
- Brief `gameState` value changes during restart sequences

---

## 7. Modal Handling

Hextris has minimal modal states compared to Breakout 71:

| State | Detection | Action |
|---|---|---|
| Start screen | `gameState == 0` | Call `init(1)` |
| Paused | `gameState == -1` | Call `resumeGame()` |
| Game over | `gameState == 2` | Terminate episode |

### Start Game (`START_GAME_JS`)

```javascript
return (function() { init(1); return true; })();
```

Called during `reset()` to start a new game. `init(1)` resets all
game state (score, blocks, difficulty) and sets `gameState = 1`.

### Resume from Pause

```javascript
return (function() { resumeGame(); return true; })();
```

If the game enters paused state during an episode (e.g., from
accidental Escape key injection), the env detects it and calls
`resumeGame()`.

### No Perk Picker / Level Transitions

Unlike Breakout 71, Hextris has no level transitions, perk selection
screens, or upgrade modals. The game is a single continuous session.
The `_handle_level_transition()` hook returns `False` (default
behavior from `BaseGameEnv`), meaning any "level cleared" signal
would terminate the episode. In practice, this never fires because
Hextris has no level system.

---

## 8. Game Loader

### `HextrisLoader(BrowserGameLoader)` (`loader.py`)

| Property | Value |
|---|---|
| Server type | Static HTTP server (`python -m http.server`) |
| Port | 8271 |
| Build step | None (plain HTML/JS, no compilation) |
| Health check | HTTP GET to `http://localhost:8271/` |
| Source directory | `$HEXTRIS_DIR` environment variable |

### Server Lifecycle

1. `start()`: Launches `python -m http.server 8271` in `$HEXTRIS_DIR`
2. Waits for HTTP readiness probe (polls `localhost:8271`)
3. Returns server URL for Selenium to navigate to
4. `stop()`: Kills the HTTP server process

### Configuration (`hextris.yaml`)

```yaml
game_name: hextris
server:
  command: "python -m http.server 8271"
  port: 8271
  health_check_url: "http://localhost:8271/"
  startup_timeout: 10
browser:
  orientation: portrait
  window_width: 768
  window_height: 1024
```

---

## 9. YOLO Bypass

Hextris is a CNN-only game with no trained YOLO model. The plugin
overrides two methods to skip YOLO entirely:

### `_lazy_init()` Override

The default `BaseGameEnv._lazy_init()` initializes the `YoloDetector`
with game-specific classes and weights. `HextrisEnv` overrides this
to skip YOLO initialization:

```python
def _lazy_init(self):
    """Skip YOLO initialization — CNN-only game."""
    self._initialized = True
```

### `_detect_objects()` Override

The default `BaseGameEnv._detect_objects()` runs YOLO inference on
captured frames. `HextrisEnv` overrides this to return an empty
detection result:

```python
def _detect_objects(self, frame):
    """No YOLO detection — return empty game state."""
    return {"raw_detections": []}
```

This pattern enables any game plugin to operate without YOLO by
overriding these two methods. No changes to `src/platform/` are
required.

### Empty Class List (`perception.py`)

```python
HEXTRIS_CLASSES = []
```

The empty class list signals to the platform that this game does not
use YOLO detection. It is referenced by the plugin metadata but not
used at runtime due to the overrides above.

---

## 10. Training Configuration

### Recommended Training Command

```bash
HEXTRIS_DIR=/mnt/f/work/hextris \
  /home/human/miniconda3/envs/yolo/bin/python scripts/train_rl.py \
  --game hextris \
  --policy cnn \
  --reward-mode survival \
  --headless \
  --orientation portrait \
  --max-steps 2000 \
  --frame-stack 4 \
  --n-steps 2048 \
  --batch-size 64 \
  --n-epochs 10 \
  --gamma 0.99 \
  --lr 0.0003 \
  --clip-range 0.2 \
  --ent-coef 0.01 \
  --timesteps 200000 \
  --output-dir /mnt/e/rsn-game-qa/output
```

### Training Results (200K steps, session 52)

| Metric | Value |
|---|---|
| Total timesteps | 200,704 |
| Duration | 3h 6m |
| FPS | ~19 it/s (~20 FPS) |
| Episodes | 323 |
| Termination | ALL natural game_over (no truncation) |
| Mean episode length | 620 |
| Mean reward | 1.18 |
| Best reward | 9.10 |
| Unique visual states | 184,314 |
| Explained variance | 0.883 |
| Browser crashes | 0 |

### Evaluation Results (200K model, 10 episodes)

| Metric | Random | Trained |
|---|---|---|
| Mean episode length | 374 | 404 |
| Mean reward | -1.28 | -0.98 |
| Critical findings | 0 | 3 |
| Total findings | 4,186 | 4,741 |
| Game over rate | 10/10 | 10/10 |

---

## 11. Key Design Decisions

### D1: Discrete(3) over Box(-1,1)

Hextris rotation is inherently discrete (6 positions). A continuous
action space would need arbitrary thresholding to map to
left/right/noop. `Discrete(3)` directly represents the three possible
actions with no information loss.

### D2: CNN-Only (No MLP)

Hextris game state (block positions, colors, lane heights) is complex
to extract via JS bridge into a fixed-size observation vector. The
hexagonal geometry means block positions are best understood visually.
CNN observation captures all relevant information without custom
extraction logic.

### D3: No Rush Action

The Down/S rush mechanic speeds up block descent. Excluding it from
the action space means:
- The agent plays at "normal" speed, matching typical player experience
- Timing-related bugs are more likely to manifest at natural game speed
- The action space stays minimal (3 actions vs 5+)

### D4: 3-Frame Game-Over Confirmation

A single `gameState == 2` reading could be a transient state. The
3-frame confirmation adds ~200ms latency to game-over detection but
eliminates false positives from state transitions during `init()` or
frame capture timing.

### D5: Static HTTP Server (No Build Step)

Hextris is plain HTML/JS with no build toolchain. Using Python's
built-in `http.server` module avoids any Node.js dependency. The
server starts in <1 second and serves files directly from the cloned
repository.

### D6: Portrait Orientation

The hexagonal game area is roughly square, but the game-over screen
and score display benefit from vertical space. Portrait mode (768x1024)
provides the best layout for both gameplay and UI elements.
