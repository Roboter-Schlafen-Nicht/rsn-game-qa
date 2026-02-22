# shapez.io -- Environment Specification

Design decisions and technical details for the RSN Game QA shapez.io
environment plugin (`games/shapez/`). Companion to
`shapez_game_spec.md`.

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
11. [Savegame Injection](#11-savegame-injection)
12. [Key Design Decisions](#12-key-design-decisions)

---

## 1. Plugin Structure

```
games/shapez/
├── __init__.py          # Plugin metadata (5 required attrs + mute_js, setup_js, load_save_js)
├── env.py               # ShapezEnv(BaseGameEnv), 897 lines
├── loader.py            # ShapezLoader(BrowserGameLoader), 230 lines
├── modal_handler.py     # 15 JS snippets, 533 lines
└── perception.py        # Empty YOLO class list (CNN-only)

configs/games/
└── shapez.yaml          # Game loader config (port 3005, landscape)
```

### Plugin Metadata (`__init__.py`)

| Attribute | Value |
|---|---|
| `game_name` | `"shapez"` |
| `env_class` | `ShapezEnv` |
| `loader_class` | `ShapezLoader` |
| `default_config` | `"shapez.yaml"` |
| `default_weights` | `None` (CNN-only, no YOLO) |
| `mute_js` | JS snippet to set music and sound volume to 0 |
| `setup_js` | `SETUP_TRAINING_JS` — disables tutorials, sets training flags |
| `load_save_js` | `LOAD_SAVE_JS` — injects savegame data and transitions to InGameState |

---

## 2. Action Space

**Type:** `gymnasium.spaces.MultiDiscrete([7, 10, 16, 16, 4])`

Total combinatorial size: 7 x 10 x 16 x 16 x 4 = 71,680 unique actions.

### Action Dimensions

| Index | Name | Range | Description |
|---|---|---|---|
| 0 | `action_type` | 0-6 | What to do |
| 1 | `building_id` | 0-9 | Which building to select |
| 2 | `grid_x` | 0-15 | Horizontal grid position |
| 3 | `grid_y` | 0-15 | Vertical grid position |
| 4 | `pan_direction` | 0-3 | Camera pan direction |

### Action Type Mapping

| Value | Action | Uses dimensions |
|---|---|---|
| 0 | No-op | None |
| 1 | Select building | `building_id` |
| 2 | Place building | `grid_x`, `grid_y` |
| 3 | Delete building | `grid_x`, `grid_y` |
| 4 | Rotate building | None |
| 5 | Pan camera | `pan_direction` |
| 6 | Center on Hub | None |

### Building ID Mapping

| ID | Building | Keyboard shortcut |
|---|---|---|
| 0 | Tunnel | `0` |
| 1 | Belt | `1` |
| 2 | Extractor | `2` |
| 3 | Cutter | `3` |
| 4 | Rotator | `4` |
| 5 | Stacker | `5` |
| 6 | Mixer | `6` |
| 7 | Painter | `7` |
| 8 | Trash | `8` |
| 9 | Balancer | `9` |

### Pan Direction Mapping

| ID | Direction | Key |
|---|---|---|
| 0 | Up | `w` |
| 1 | Down | `s` |
| 2 | Left | `a` |
| 3 | Right | `d` |

### Grid-to-Canvas Mapping

The 16x16 grid is mapped to canvas pixel coordinates:

```python
cell_w = canvas_width / grid_size   # e.g., 1280 / 16 = 80
cell_h = canvas_height / grid_size  # e.g., 885 / 16 ≈ 55
pixel_x = grid_x * cell_w + cell_w / 2   # center of cell
pixel_y = grid_y * cell_h + cell_h / 2
```

The grid covers the visible canvas area. Camera pan actions shift
what part of the infinite map is visible, making different grid
positions correspond to different world coordinates.

### Action Execution

Each action type maps to a JS snippet executed via Selenium:

```javascript
// Select building: dispatch keydown with digit key
document.dispatchEvent(new KeyboardEvent('keydown',
    {key: building_id.toString(), keyCode: 48 + building_id}));

// Place building: click at canvas position
var canvas = document.getElementById('ingame_Canvas');
canvas.dispatchEvent(new MouseEvent('click',
    {clientX: px, clientY: py, bubbles: true}));

// Delete building: right-click at canvas position
canvas.dispatchEvent(new MouseEvent('contextmenu',
    {clientX: px, clientY: py, bubbles: true}));

// Rotate: dispatch 'r' keydown
document.dispatchEvent(new KeyboardEvent('keydown',
    {key: 'r', keyCode: 82}));

// Pan: dispatch WASD keydown + keyup
document.dispatchEvent(new KeyboardEvent('keydown',
    {key: direction, keyCode: code}));
// ... short delay ...
document.dispatchEvent(new KeyboardEvent('keyup',
    {key: direction, keyCode: code}));

// Center on Hub: dispatch space key
document.dispatchEvent(new KeyboardEvent('keydown',
    {key: ' ', keyCode: 32}));
```

---

## 3. Observation Space

### CNN Mode (Primary)

**Type:** `gymnasium.spaces.Box(0, 255, shape=(84, 84, 1), dtype=uint8)`

The raw browser screenshot of `#ingame_Canvas` is converted to 84x84
grayscale. With `--frame-stack 4`, the effective observation is
`(84, 84, 4)`.

### MLP Mode (Secondary)

**Type:** `gymnasium.spaces.Box(-inf, inf, shape=(8,), dtype=float32)`

| Index | Name | Normalization | Description |
|---|---|---|---|
| 0 | `level_norm` | level / 30 | Current level progress |
| 1 | `goal_progress` | delivered / required | Goal completion ratio |
| 2 | `entity_norm` | entities / 1000 | Factory size |
| 3 | `running` | 0.0 or 1.0 | Game is running |
| 4-7 | (unused) | 0.0 | Reserved |

### Frame Capture

Frames are captured via Selenium's `toDataURL()` on `#ingame_Canvas`.
The canvas only exists during `InGameState` — capture attempts during
other states use a cached black frame.

### What the CNN Sees

- Infinite 2D grid with placed buildings
- Conveyor belts with moving shapes
- The central Hub
- Shape deposits (colored patches on the grid)
- HUD overlays (toolbar at bottom, level indicator)
- Building ghost preview (if a building is selected)

---

## 4. Reward Design

### Survival Mode (Default)

```
reward = +0.01 per step (survival bonus)
       + shape_delivery_delta × 0.01
       + entity_placement_delta × 0.005
       + level_completion × 1.0
       + (-5.001) on idle termination
       + time_penalty × -0.001 per step
```

### Reward Components

| Component | Value | Source | Rationale |
|---|---|---|---|
| Survival bonus | +0.01 / step | Platform default | Base signal |
| Shape delivery | delta × 0.01 | `hubGoals` JS bridge | Reward for delivering shapes to Hub |
| Entity placement | delta × 0.005 | `entityMap_` JS bridge | Reward for building things |
| Level completion | +1.0 | Level change detected | Major milestone |
| Time penalty | -0.001 / step | Fixed | Mild pressure to act |
| Terminal penalty | -5.001 | On episode end | Standard platform penalty |

### Game State Reading for Reward

```javascript
return (function() {
    if (!window.globalRoot) return null;
    var goals = globalRoot.hubGoals;
    var entityCount = 0;
    var entityMap = globalRoot.map.entityMap_;
    // ... count entities ...
    return {
        level: goals.level,
        shapesDelivered: goals.storedShapes,
        goalRequired: goals.currentGoal ? goals.currentGoal.required : 0,
        entityCount: entityCount,
        running: true
    };
})();
```

### Reward Design Challenges

shapez.io's reward landscape is fundamentally different from
arcade/puzzle games:

1. **Survival is trivial.** The player cannot die, so survival
   reward provides no learning signal.
2. **Building requires multi-step sequences.** Placing a functional
   factory requires: select building → click position → confirm. Random
   actions rarely produce this sequence.
3. **Reward is extremely sparse.** Shape deliveries only happen after
   a complete extraction → transport → Hub pipeline is built.
4. **Long horizon.** Even a simple factory takes 50+ actions to build.

These challenges make pure RL training ineffective for shapez.io.
The survival reward mode produces identical results for trained and
random agents (both score 25.00 mean reward at 200K steps).

---

## 5. Termination Conditions

### Idle Detection (Primary)

shapez.io has no natural game-over. Episodes terminate via idle
detection:

- **Threshold:** 3000 steps without meaningful state change
- **State change:** Any of: shape delivery delta > 0, entity count
  delta > 0, level change
- **Counter:** `_idle_steps` increments when no change, resets on change
- **Terminal:** When `_idle_steps >= idle_threshold`

This is the primary termination mechanism. All 67 training episodes
(200K steps) terminated via idle detection.

### Max Steps (Truncation)

- Episode ends when `step_count >= max_steps`
- Default `max_steps`: 3000 (matches idle threshold)
- **Info key:** `truncated = True`

### Terminal Penalty

On either idle or max_steps termination:
```
terminal_reward = -(5.0 + survival_bonus) = -5.001 (with default 0.001 time penalty)
```

### No Game-Over Detection

The `GameOverDetector` (pixel-based) is available but irrelevant for
shapez.io since the game has no game-over state. The JS-based game
state detection (`document.body.id`) is used only to confirm the
game is in `InGameState`.

---

## 6. State Detection

### Primary State Detection (`modal_handler.py`)

**`DETECT_STATE_JS`:**

```javascript
return (function() {
    var bodyId = document.body.id;
    if (bodyId === 'state_InGameState') {
        // Check sub-states within InGameState
        if (globalRoot && globalRoot.app.stateMgr.currentState) {
            var state = globalRoot.app.stateMgr.currentState;
            if (state.getIsIngame && state.getIsIngame()) return 'playing';
        }
        return 'ingame';
    }
    if (bodyId === 'state_MainMenuState') return 'main_menu';
    if (bodyId === 'state_PreloadState') return 'loading';
    return 'unknown';
})();
```

### Game State Reading (`READ_GAME_STATE_JS`)

```javascript
return (function() {
    if (!window.globalRoot) return null;
    var goals = globalRoot.hubGoals;
    var entityCount = /* ... count entities ... */;
    return {
        level: goals.level,
        shapesDelivered: /* current delivery count */,
        goalRequired: goals.currentGoal ? goals.currentGoal.required : 0,
        entityCount: entityCount,
        upgradeLevels: goals.upgradeLevels,
        running: true
    };
})();
```

Returns `null` if `globalRoot` is not available (not in InGameState).

### InGameState Polling

After calling `start_game()`, the env polls for InGameState readiness:

```python
for _ in range(max_retries):
    state = driver.execute_script(DETECT_STATE_JS)
    if state == 'playing':
        break
    time.sleep(poll_interval)
```

This handles the asynchronous state transition from MainMenuState
through savegame creation to InGameState entry.

---

## 7. Modal Handling

### Start Game (`START_NEW_GAME_JS`)

Creates a new savegame and transitions to InGameState:

```javascript
return (function() {
    var app = window.shapez.GLOBAL_APP;
    if (!app) return false;
    var savegame = app.savegameMgr.createNewSavegame({});
    app.stateMgr.moveToState("InGameState", { savegame: savegame });
    return true;
})();
```

### Setup Training (`SETUP_TRAINING_JS`)

Disables tutorials and configures training-friendly settings:

```javascript
return (function() {
    var app = window.shapez.GLOBAL_APP;
    if (!app || !app.settings) return false;
    app.settings.updateSetting("offerHints", false);
    return true;
})();
```

### Mute Audio (`MUTE_AUDIO_JS`)

```javascript
return (function() {
    var app = window.shapez.GLOBAL_APP;
    if (!app || !app.settings) return false;
    app.settings.updateSetting("musicVolume", 0);
    app.settings.updateSetting("soundVolume", 0);
    return true;
})();
```

### No Perk Picker / Level Transitions

shapez.io has no perk selection between levels. Level completion
triggers a brief notification overlay that auto-dismisses. The env
does not need to handle it — the game continues automatically.

---

## 8. Game Loader

### `ShapezLoader(BrowserGameLoader)` (`loader.py`)

| Property | Value |
|---|---|
| Server type | BrowserSync dev server (via Gulp) |
| Port | 3005 |
| Build step | `yarn gulp` in `gulp/` subdirectory |
| Health check | HTTP GET to `http://localhost:3005/` |
| Source directory | `$SHAPEZ_DIR` environment variable |
| Node.js version | 16 (via nvm) |

### Server Lifecycle

1. `start()`:
   a. Kills any existing processes on port 3005
      (`_kill_port_processes()`)
   b. Sets up nvm Node 16 environment
   c. Runs `yarn gulp` in the `gulp/` subdirectory
   d. Waits for HTTP readiness probe (polls `localhost:3005`)
2. Returns server URL for Selenium
3. `stop()`: Kills the Gulp/BrowserSync process and orphan port
   processes

### nvm Preamble

The loader prepends an nvm activation preamble to the build command:

```bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
nvm use 16
cd gulp && yarn gulp
```

This ensures Node 16 is used regardless of the system's default
Node version.

### Port Cleanup

`_kill_port_processes()` uses `lsof -ti :PORT | xargs kill -9` to
clean up orphan server processes from previous runs. This prevents
"port already in use" errors.

### Configuration (`shapez.yaml`)

```yaml
game_name: shapez
server:
  command: "bash -c 'export NVM_DIR=...; nvm use 16; cd gulp && yarn gulp'"
  port: 3005
  health_check_url: "http://localhost:3005/"
  startup_timeout: 60
browser:
  orientation: landscape
  window_width: 1280
  window_height: 1024
```

### BrowserSync Configuration

The `gulpfile.js` has `open: false` set to prevent BrowserSync from
auto-opening a browser window on startup (which would interfere with
headless operation).

---

## 9. YOLO Bypass

Same pattern as Hextris — shapez.io is CNN-only with no YOLO model.

### `_lazy_init()` Override

```python
def _lazy_init(self):
    """Skip YOLO initialization — CNN-only game."""
    self._initialized = True
```

### `_detect_objects()` Override

```python
def _detect_objects(self, frame):
    """No YOLO detection — return empty game state."""
    return {"raw_detections": []}
```

### Empty Class List (`perception.py`)

```python
SHAPEZ_CLASSES = []
```

---

## 10. Training Configuration

### Recommended Training Command

```bash
SHAPEZ_DIR=/home/human/games/shapez.io \
  /home/human/miniconda3/envs/yolo/bin/python scripts/train_rl.py \
  --game shapez \
  --policy cnn \
  --reward-mode survival \
  --headless \
  --orientation landscape \
  --max-steps 3000 \
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

### Training Results (200K steps, session 59)

| Metric | Value |
|---|---|
| Total timesteps | 200,704 |
| Duration | 4h 41m |
| FPS | ~13 |
| Episodes | 67 |
| Termination | ALL idle detection at max_steps=3000 |
| Mean episode length | 2,979 |
| Mean reward | 24.79 |
| Best reward | 25.00 |
| Unique visual states | 12 (8x8 fingerprint) |
| Explained variance | ~0.0 |
| Browser crashes | 0 |

### Evaluation Results (200K model, 10 episodes)

| Metric | Random | Trained |
|---|---|---|
| Mean episode length | 3000 | 3000 |
| Mean reward | 25.00 | 25.00 |
| Critical findings | 10 | 10 |
| Warning findings | 496 | 510 |
| Total findings | 506 | 520 |
| Game over rate | 10/10 | 10/10 |

Trained and random agents produce identical results. Survival reward
is genre-limited for factory builders where survival is trivial.

---

## 11. Savegame Injection

### Purpose

shapez.io bugs often manifest in late-game states with large
factories (performance bugs, throughput inconsistencies, rendering
glitches at scale). Savegame injection enables starting episodes from
these states without requiring the agent to build factories.

### Plugin Hook

The `load_save_js` attribute in `__init__.py` provides the
`LOAD_SAVE_JS` snippet from `modal_handler.py`.

### `LOAD_SAVE_JS` Implementation

```javascript
return (function() {
    var _sel_args = arguments;
    var app = window.shapez.GLOBAL_APP;
    if (!app) return { success: false, error: 'No app' };
    var data;
    try { data = JSON.parse(_sel_args[0]); }
    catch (e) { return { success: false, error: e.message }; }
    var savegame = app.savegameMgr.createNewSavegame({});
    savegame.currentData = data;
    savegame.migrate(data);
    app.stateMgr.moveToState("InGameState", { savegame: savegame });
    return { success: true };
})();
```

### Integration

1. `SavegamePool` scans `--savegame-dir` for `.json`/`.sav`/`.save` files
2. Each episode, `SavegamePool.next()` selects a save file
   (random or sequential)
3. `SavegameInjector` reads the file and calls `LOAD_SAVE_JS` with
   the file contents as `arguments[0]`
4. The game transitions to `InGameState` with the loaded factory
5. Normal episode proceeds from the loaded state

---

## 12. Key Design Decisions

### D1: MultiDiscrete over Box or Discrete

shapez.io requires structured multi-dimensional actions (what to do +
which building + where). Options considered:

| Option | Size | Issue |
|---|---|---|
| `Discrete(N)` | ~71,680 | Flat enumeration, no structure for PPO |
| `Box(low, high, shape=(5,))` | Continuous | Arbitrary thresholding needed |
| `MultiDiscrete([7,10,16,16,4])` | 71,680 | Structured, SB3 native support |

`MultiDiscrete` preserves the semantic structure of each dimension
while allowing SB3's PPO to learn independent policies for each
action dimension.

### D2: 16x16 Grid Resolution

The 16x16 grid balances precision and action space size:
- 8x8: Too coarse — buildings at wrong positions
- 16x16: Good balance — 256 positions, ~5 tile precision
- 32x32: 1024 positions, action space explodes to 286K

### D3: Idle Detection vs Max Steps

Factory builders have no natural endpoint. Two termination mechanisms
work together:

- **Idle detection (3000 steps):** Terminates when nothing changes.
  This is the primary mechanism — the agent isn't doing anything
  useful if the factory state is static.
- **Max steps (3000 steps, matching):** Hard cap as a safety net.
  Set equal to idle threshold, so in practice idle always fires
  first (unless the agent is actively building but not delivering).

### D4: Landscape Orientation

shapez.io's factory layout is naturally horizontal (belts flow
left-to-right or right-to-left by convention). Landscape mode
(1280x1024) provides the widest view of the factory. This differs
from Breakout 71 and Hextris which use portrait mode.

### D5: CNN-Only (No YOLO)

shapez.io's visual complexity (hundreds of buildings, conveyor items,
terrain features) makes YOLO impractical without extensive annotation.
The CNN observation captures the full visual state at 84x84 resolution.

### D6: nvm Node 16 Preamble

shapez.io requires Node.js 16 (newer versions break the build).
The nvm preamble in the loader ensures the correct version regardless
of the system default. This is more robust than requiring users to
globally switch Node versions.

### D7: Port Cleanup on Start

Previous training runs can leave orphan BrowserSync processes. The
`_kill_port_processes()` method in the loader ensures a clean start.
This prevents cascading failures from "port already in use" errors.

### D8: Canvas Re-initialization

The `#ingame_Canvas` element is created dynamically when entering
`InGameState`. The env must re-find the canvas element after each
game start, as the DOM reference from a previous episode is stale.
This was discovered during live validation (PR #132) and required
re-querying the canvas selector after confirming InGameState entry.

### D9: globalRoot vs GLOBAL_APP

Early plugin development used `window.globalRoot` which is only
available during `InGameState`. The migration to
`window.shapez.GLOBAL_APP` (PR #133) enables access to application
services (settings, savegame manager) from any state, including
during `start_game()` before `InGameState` is entered.

### D10: Savegame Injection Architecture

Rather than building a curriculum system to teach the agent to
construct factories, savegame injection bypasses the "agent can't
build" problem entirely. This is a pragmatic engineering decision:
the agent's value for QA is in exploring and stress-testing existing
factories, not in learning to build them from scratch. Human demos
(Phase 7) provide the building knowledge; savegame injection provides
immediate access to interesting game states.
