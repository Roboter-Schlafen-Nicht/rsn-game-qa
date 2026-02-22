# Hextris -- Game Technical Specification

Comprehensive reference for the Hextris game mechanics, derived from
the [Hextris source code](https://github.com/Hextris/hextris). Used by
the RSN Game QA platform to inform environment design, oracle
configuration, and reward engineering.

**Game version:** GitHub HEAD (GPL-3.0)
**Genre:** Hexagonal puzzle / arcade
**Platform:** Browser (HTML5 Canvas, vanilla JavaScript)
**Input:** Keyboard (desktop), touch (mobile)

---

## Table of Contents

1. [Key Source Files](#1-key-source-files)
2. [Game Overview](#2-game-overview)
3. [Game States](#3-game-states)
4. [Central Hex](#4-central-hex)
5. [Block System](#5-block-system)
6. [Wave Generator](#6-wave-generator)
7. [Scoring and Combo System](#7-scoring-and-combo-system)
8. [Game Over Conditions](#8-game-over-conditions)
9. [Input Model](#9-input-model)
10. [Rendering and Canvas](#10-rendering-and-canvas)
11. [Settings and Configuration](#11-settings-and-configuration)
12. [Audio](#12-audio)
13. [Persistence](#13-persistence)
14. [Game Loop](#14-game-loop)
15. [Difficulty Progression](#15-difficulty-progression)
16. [Selenium Integration Points](#16-selenium-integration-points)

---

## 1. Key Source Files

| File | Purpose | Key exports / globals |
|---|---|---|
| `js/main.js` | Game loop, lifecycle, game-over check | `init()`, `animLoop()`, `checkGameOver()`, `gameState` |
| `js/Hex.js` | Central hexagon class | `Hex()`, `.rotate()`, `.addBlock()`, `.doesBlockCollide()` |
| `js/Block.js` | Falling block class | `Block()`, `.draw()`, `.incrementOpacity()` |
| `js/initialization.js` | Settings, global vars, `initialize()` | `settings`, `colors`, `gdx`, `gdy`, `trueCanvas` |
| `js/checking.js` | Match detection, score calculation | `floodFill()`, `consolidateBlocks()` |
| `js/update.js` | Per-frame game tick | `update()` |
| `js/wavegen.js` | Block spawn patterns and difficulty | `waveGen()`, 6 generation patterns |
| `js/input.js` | Keyboard and touch handlers | Event listeners, `keyPressedDown[]` |
| `js/render.js` | Canvas drawing | `render()`, `drawTimer()` |
| `js/comboTimer.js` | Combo timer visual | `drawComboTimer()` |
| `index.html` | DOM structure, game-over screen | `#gameoverscreen`, `#게임결과` |

---

## 2. Game Overview

Hextris is a hexagonal puzzle game inspired by Tetris. Colored blocks
fall toward a central hexagon from six lanes (sides). The player rotates
the hexagon left or right to catch and stack blocks. When three or more
adjacent blocks of the same color connect, they are cleared and the
player scores points. The game ends when any lane's block stack exceeds
the maximum allowed height.

**Core loop:**
1. Blocks spawn at the edges and fall toward the center hex
2. Player rotates the hex to position lanes under incoming blocks
3. Blocks settle on the hex surface or on top of previously settled blocks
4. Connected same-color groups of 3+ blocks are cleared (scored)
5. Difficulty increases over time (faster spawns, faster blocks)
6. Game ends when a lane overflows

There are no levels, no transitions, and no win condition. The game is
a single continuous session until death. The only objective is to
maximize the score.

---

## 3. Game States

The global variable `window.gameState` tracks the current state:

| Value | Constant | Description |
|---|---|---|
| `0` | Start screen | Initial state before first game |
| `1` | Playing | Active gameplay |
| `2` | Game over | Game-over screen displayed |
| `-1` | Paused | Game paused (Escape or P key) |

Additional visual states exist but are not tracked by `gameState`:

- **Fade-out transition:** When game over triggers, blocks briefly
  fade before the game-over screen appears. Controlled by block
  `opacity` property and `incrementOpacity()`.
- **Game-over screen:** DOM overlay `#gameoverscreen` with blur
  filter (`-webkit-filter: blur(5px)`) applied to the canvas.

State transitions:

```
Start (0) --[init(1)]--> Playing (1)
Playing (1) --[checkGameOver()]--> Game Over (2)
Playing (1) --[Escape/P]--> Paused (-1)
Paused (-1) --[resumeGame()]--> Playing (1)
Game Over (2) --[init(1)]--> Playing (1)
```

---

## 4. Central Hex

The `Hex` class (`js/Hex.js`) represents the central hexagon that the
player rotates.

### Properties

| Property | Type | Description |
|---|---|---|
| `x`, `y` | number | Center position on canvas (typically `trueCanvas.width/2`, `trueCanvas.height/2`) |
| `position` | number | Current rotation position (0-5, represents which side faces "up") |
| `sideLength` | number | Length of each hex side in pixels (from `settings.hexWidth`) |
| `height` | number | Hex height = `sideLength * Math.sqrt(3)` |
| `width` | number | Hex width = `sideLength * 2` |
| `blocks` | Block[][] | 2D array: `blocks[lane][depth]` — 6 lanes, variable depth |
| `sides` | number | Always 6 |
| `ct` | number | Internal counter (unused in core logic) |
| `fillColors` | string[] | Current fill colors per side |
| `strokeColors` | string[] | Current stroke colors per side |
| `targetAngles` | number[] | Target rotation angles for animation |
| `angularVelocity` | number | Rotation animation speed |

### Key Methods

**`rotate(steps)`**
- Rotates the hex by `steps` positions (positive = counterclockwise,
  negative = clockwise)
- Desktop: 75ms cooldown between rotations (not enforced for JS calls)
- Mobile: no cooldown
- Updates `position = (position + steps) % 6`
- Sets `targetAngles` for smooth animation
- After rotation completes, calls `consolidateBlocks()` to check for
  matches

**`addBlock(block)`**
- Attaches a settled block to `blocks[lane][]`
- Block's `attachedLane` and `height` are set based on stack position
- Distance from hex center = `sideLength + height * blockHeight`

**`doesBlockCollide(block, position)`**
- Checks if a block at `position` in its lane would overlap with
  existing blocks
- Used by `update()` to determine when a falling block settles

### Coordinate System

The hex uses a rotated coordinate system where each of the 6 sides
corresponds to a lane (0-5). Lane 0 is at the top, proceeding clockwise:

```
      Lane 0
       ___
  5  /     \  1
    /       \
    \       /
  4  \_____/  2
      Lane 3
```

Block positions are calculated using trigonometric functions based on
lane index and the hex's current rotation angle.

---

## 5. Block System

The `Block` class (`js/Block.js`) represents individual colored blocks
that fall toward the hex.

### Properties

| Property | Type | Description |
|---|---|---|
| `fallingLane` | number | Lane (0-5) the block falls toward |
| `color` | string | Hex color string (one of 4 colors) |
| `iter` | number | Fall speed (pixels per frame) |
| `distFromHex` | number | Current distance from hex center |
| `settled` | boolean | True when block has landed on hex |
| `deleted` | boolean | True when block is being removed (match) |
| `attachedLane` | number | Lane the block settled on (set on landing) |
| `height` | number | Stack position within the lane (0 = bottom) |
| `opacity` | number | Visual opacity (1.0 = fully visible) |
| `tint` | number | Color tint intensity |
| `tintDir` | number | Tint animation direction |

### Colors

Four block colors are used, defined in `initialization.js`:

```javascript
var colors = ["#e74c3c", "#f1c40f", "#3498db", "#2ecc71"];
//             red        yellow     blue       green
```

### Block Lifecycle

1. **Spawning:** `waveGen` creates a new `Block` at maximum distance
   from the hex center, assigned to a random lane with a random color
2. **Falling:** Each frame, `distFromHex -= iter` (moves toward center)
3. **Collision check:** `doesBlockCollide()` tests if the block would
   overlap with existing settled blocks or the hex surface
4. **Settling:** When collision detected, `settled = true`, block is
   added to `MainHex.blocks[lane][]`
5. **Match check:** `consolidateBlocks()` runs flood-fill to find
   connected same-color groups
6. **Deletion:** If group size >= 3, blocks are marked `deleted = true`
   and fade out via `incrementOpacity()`
7. **Cleanup:** Deleted blocks are removed from the array, remaining
   blocks shift down

### Block Dimensions

- `blockHeight`: 20px (desktop), configured in `settings`
- Block width matches the hex side length
- Visual rendering uses trapezoidal shapes that conform to the hex geometry

---

## 6. Wave Generator

The `waveGen` class (`js/wavegen.js`) controls when and how blocks
spawn. It is the primary driver of difficulty progression.

### Properties

| Property | Type | Description |
|---|---|---|
| `nextGen` | number | Frames until next spawn (decreases over time) |
| `difficulty` | number | Current difficulty level (starts 1, caps at 35) |
| `dt` | number | Internal timer for spawn interval |

### Spawn Patterns

Six generation patterns are randomly selected:

| Pattern | Description |
|---|---|
| `randomGeneration()` | 1 block in a random lane |
| `doubleGeneration()` | 2 blocks in two random non-adjacent lanes |
| `crosswiseGeneration()` | 2 blocks in opposite lanes (180 degrees apart) |
| `spiralGeneration()` | 3 blocks in consecutive lanes |
| `circleGeneration()` | 6 blocks, one in each lane |
| `halfCircleGeneration()` | 3 blocks in three adjacent lanes |

### Difficulty Curve

```
difficulty: starts at 1, increments by 1 each spawn cycle, caps at 35
nextGen (spawn interval): starts at 2700, decreases over time
  floor: 600 (minimum spawn interval)
block speed (iter): 1.5 + random(0, 0.1) + (difficulty / 15) * 3
  at difficulty 1:  ~1.7 pixels/frame
  at difficulty 35: ~8.5 pixels/frame
```

The spawn interval formula uses a complex decay that starts slow and
accelerates:

```javascript
// Approximate: nextGen decreases by ~30-50 per difficulty level
// with randomization, reaching floor of 600 around difficulty 15-20
```

### Spawn Position

New blocks spawn at a distance of `gdx` (global canvas diagonal) from
the hex center, ensuring they appear off-screen and fall into view.

---

## 7. Scoring and Combo System

### Match Detection (`checking.js`)

**`consolidateBlocks()`** runs after every rotation and block settling:

1. Iterates all settled, non-deleted blocks
2. For each block, runs `floodFill()` to find connected same-color
   neighbors
3. Adjacency: blocks in the same lane that are vertically adjacent,
   or blocks in neighboring lanes at matching heights
4. If connected group size >= 3: all blocks in the group are deleted

**`floodFill(index, lane, color, result)`:**
- Recursive flood fill starting from `blocks[lane][index]`
- Checks same-lane vertical neighbors (index +/- 1)
- Checks adjacent lanes (lane +/- 1, mod 6) at matching height
- Returns array of `{index, lane}` pairs for all connected blocks

### Score Calculation

```javascript
score += count * count * comboMultiplier;
// count = number of blocks cleared in one match
// comboMultiplier starts at 1, increments on consecutive clears
```

Score is quadratic in match size:
- 3-block match: 9 * comboMultiplier
- 4-block match: 16 * comboMultiplier
- 6-block match (full circle): 36 * comboMultiplier

### Combo System (`comboTimer.js`)

- `comboTime`: Duration of the combo window (starts at 240 frames on
  desktop, 310 on some configs)
- When a match occurs, the combo timer resets to `comboTime`
- While the combo timer is active and another match occurs:
  `comboMultiplier` increments by 1
- When the combo timer expires: `comboMultiplier` resets to 1
- Visual: combo timer drawn as a circular progress indicator around the
  hex

The combo system rewards rapid consecutive clears. A skilled player
can chain matches by rotating to create multiple match opportunities
before the combo window expires.

---

## 8. Game Over Conditions

### Primary Check (`main.js`)

**`checkGameOver()`** is called every frame during `animLoop()`:

```javascript
function checkGameOver() {
    if (MainHex.isInfringing()) {
        // trigger game over
        gameState = 2;
        // show game-over screen
    }
}
```

**`isInfringing()`** (`Hex.js`):
- Iterates all 6 lanes
- Counts non-deleted blocks in each lane
- If any lane has more blocks than `settings.rows`: return `true`
- `settings.rows = 8` on desktop, `7` on mobile

### Game-Over Visual Sequence

1. `gameState` set to `2`
2. All blocks begin opacity fade-out (`incrementOpacity()`)
3. DOM element `#gameoverscreen` becomes visible
4. Canvas gets CSS blur filter applied
5. Final score displayed in `#게임결과` element
6. Share buttons (Twitter, Facebook) become active

### Restart

Pressing any key or clicking during game-over state calls `init(1)`:
- Resets `gameState = 1`
- Clears all blocks
- Resets score, combo, difficulty
- Hides game-over screen
- Starts new game immediately

---

## 9. Input Model

### Desktop Keyboard (`input.js`)

| Key | Action | Notes |
|---|---|---|
| Left Arrow / A | Rotate left (counterclockwise) | `MainHex.rotate(-1)` |
| Right Arrow / D | Rotate right (clockwise) | `MainHex.rotate(1)` |
| Down Arrow / S | Speed up blocks | `rush = rush * 4` (while held) |
| Escape / P | Pause / unpause | Toggles `gameState` between 1 and -1 |
| Space | (not mapped in Hextris) | |

### Key Repeat

- `keydown` events are tracked in `keyPressedDown[]` array
- Rotation fires on every `keydown` event (key repeat applies)
- The game uses `requestAnimationFrame` for its loop, so key processing
  is frame-rate dependent
- 75ms rotation cooldown on desktop prevents excessively rapid rotation

### Touch Controls (`input.js`)

- Left half of screen: rotate left
- Right half of screen: rotate right
- Swipe detection for rotation direction
- Touch-and-hold for continuous rotation

### Rush Mechanic

When Down/S is held:
- `rush` variable is multiplied by 4 each frame
- All falling blocks' `iter` (speed) is multiplied by `rush`
- On key release: `rush` resets to 1
- This allows the player to speed up slow-falling blocks when they
  know where they want them to land

---

## 10. Rendering and Canvas

### Canvas Setup

```javascript
var canvas = document.getElementById("canvas");
var ctx = canvas.getContext("2d");
var trueCanvas = { width: window.innerWidth, height: window.innerHeight };
```

The canvas fills the entire browser window and resizes on
`window.resize`. The hex is centered at `(trueCanvas.width/2,
trueCanvas.height/2)`.

### Render Pipeline (`render.js`)

Each frame in `animLoop()`:
1. Clear canvas with background color (`#ecf0f1`)
2. Draw the central hexagon (filled with gradient colors per side)
3. Draw all settled blocks (trapezoidal shapes conforming to hex geometry)
4. Draw all falling blocks
5. Draw deleted blocks (fading out)
6. Draw combo timer indicator (circular arc around hex)
7. Draw score text (top-center, via `ctx.fillText()`)
8. Draw any debug overlays

### Visual Elements

- **Hex fill:** Each side has independent fill and stroke colors
- **Blocks:** Drawn as trapezoids whose inner edge conforms to the hex
  side and outer edge is parallel but at `blockHeight` distance
- **Score:** Rendered on canvas via `ctx.fillText()` (not DOM) at
  top-center position
- **Combo timer:** Circular arc drawn around the hex perimeter
- **Game-over overlay:** DOM-based (`#gameoverscreen`), not canvas

### Canvas Selector

The game canvas element has `id="canvas"` in the DOM:
```html
<canvas id="canvas"></canvas>
```

---

## 11. Settings and Configuration

### Desktop Settings (`initialization.js`)

| Setting | Value | Description |
|---|---|---|
| `rows` | 8 | Max blocks per lane before game over |
| `hexWidth` | 87 | Hex side length in pixels |
| `blockHeight` | 20 | Block height in pixels |
| `speedModifier` | 0.65 | Global speed multiplier |
| `comboTime` | 310 | Combo window duration (frames) |
| `comboAmount` | 1 | Initial combo multiplier |
| `baseScore` | 0 | Starting score |

### Mobile Settings (overrides)

| Setting | Value | Description |
|---|---|---|
| `rows` | 7 | Reduced stack height for smaller screens |
| `speedModifier` | 0.73 | Slightly faster on mobile |

### Difficulty Settings (in `wavegen.js`)

| Setting | Value | Description |
|---|---|---|
| `startDifficulty` | 1 | Initial difficulty |
| `maxDifficulty` | 35 | Difficulty cap |
| `startSpawnInterval` | 2700 | Initial frames between spawns |
| `minSpawnInterval` | 600 | Fastest spawn rate (floor) |

---

## 12. Audio

Hextris has optional sound effects:

- **Block clear:** Short chime on successful match
- **Game over:** Distinct game-over sound
- **Rotation:** Subtle click on hex rotation

Audio is loaded via HTML `<audio>` elements and played via
`HTMLAudioElement.play()`. Audio can be muted by setting the audio
element `volume` to 0 or `muted` to `true`.

The RSN platform mutes audio during training via the plugin's
`mute_js` attribute:
```javascript
var audios = document.querySelectorAll('audio');
audios.forEach(function(a) { a.muted = true; a.volume = 0; });
```

---

## 13. Persistence

### localStorage

Hextris uses `localStorage` for:
- **High score:** `localStorage.getItem("highscore")`
- **Save state:** `exportSaveState()` serializes current game state
- **Restore:** `importSaveState()` resumes from saved state
- **Clear:** `clearSaveState()` removes saved data

### Save State Format

The save state captures:
- `gameState` value
- `score`
- `MainHex.blocks[][]` (all settled blocks with positions and colors)
- `blocks[]` (all falling blocks with positions, speeds, lanes)
- `waveGen` state (difficulty, spawn timer)
- `comboMultiplier` and `comboTime` remaining

---

## 14. Game Loop

### `animLoop()` (`main.js`)

The main game loop uses `requestAnimationFrame`:

```
animLoop():
  1. requestAnimationFrame(animLoop)  // schedule next frame
  2. if gameState != 1: return        // skip if not playing
  3. update()                         // game logic tick
  4. render()                         // draw frame
  5. checkGameOver()                  // check termination
```

### `update()` (`update.js`)

Per-frame game logic:

```
update():
  1. waveGen.dt++                     // increment spawn timer
  2. if waveGen.dt >= waveGen.nextGen:
     a. spawn new blocks (random pattern)
     b. increase difficulty
     c. reset spawn timer
  3. for each falling block:
     a. move toward center (distFromHex -= iter * rush)
     b. if collision: settle block, add to hex
     c. consolidateBlocks()           // check for matches
  4. remove fully faded deleted blocks
  5. update combo timer
```

### Frame Rate

The game runs at the browser's `requestAnimationFrame` rate, typically
60 FPS on modern displays. All game speeds are frame-rate dependent
(not time-based), so slower frame rates result in slower gameplay.

---

## 15. Difficulty Progression

Hextris has a single continuous difficulty curve with no discrete levels
or transitions.

### Progression Variables

| Variable | Start | End | Effect |
|---|---|---|---|
| `difficulty` | 1 | 35 (cap) | Controls spawn interval and block speed |
| `nextGen` (spawn interval) | 2700 | 600 (floor) | Frames between block spawns |
| Block speed (`iter`) | ~1.7 | ~8.5 | Pixels per frame |
| Spawn pattern complexity | Random | All patterns | More complex patterns at higher difficulty |

### Difficulty Curve Shape

```
Time (seconds)    Difficulty    Spawn Rate    Block Speed
0                 1             ~45s          ~1.7 px/f
30                5             ~30s          ~2.5 px/f
60                10            ~20s          ~3.5 px/f
120               20            ~12s          ~5.5 px/f
180               30            ~10s          ~7.5 px/f
240+              35 (cap)      ~10s (floor)  ~8.5 px/f
```

The difficulty increases each time a new wave of blocks spawns.
Early game is slow and forgiving; late game has rapid spawns with
fast-falling blocks that require quick rotational decisions.

---

## 16. Selenium Integration Points

These are the JavaScript globals and methods used by the RSN platform
to control and observe Hextris via Selenium `execute_script()`.

### Readable Globals

| Global | Type | Description |
|---|---|---|
| `window.gameState` | number | Current game state (0/1/2/-1) |
| `window.score` | number | Current score |
| `window.blocks` | Block[] | Array of all blocks (falling + settled) |
| `window.MainHex` | Hex | The central hexagon object |
| `window.MainHex.blocks` | Block[][] | Settled blocks by lane |
| `window.MainHex.position` | number | Current rotation position (0-5) |
| `window.settings` | object | Game settings (rows, hexWidth, etc.) |
| `window.rush` | number | Current rush multiplier |

### Control Methods

| Method | Effect | Used for |
|---|---|---|
| `init(1)` | Start new game / restart | `start_game()`, `reset()` |
| `MainHex.rotate(-1)` | Rotate left (counterclockwise) | Action: rotate_left |
| `MainHex.rotate(1)` | Rotate right (clockwise) | Action: rotate_right |
| `resumeGame()` | Resume from pause | Handling paused state |

### DOM Elements

| Selector | Purpose |
|---|---|
| `#canvas` | Game canvas element |
| `#gameoverscreen` | Game-over overlay (visibility = game over state) |
| `#게임결과` | Score display on game-over screen |

### State Detection Pattern

```javascript
// Detect current game state
var state = window.gameState;
var score = window.score || 0;
var isGameOver = document.getElementById('gameoverscreen')
    .style.webkitFilter.indexOf('blur') > -1;
```

The platform uses a combination of `window.gameState` and DOM element
inspection (blur filter on game-over screen) to reliably detect the
current state.
