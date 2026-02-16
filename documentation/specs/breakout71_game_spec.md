# Breakout 71 -- Comprehensive Game Technical Specification

> Extracted from the Breakout 71 TypeScript source code at
> `F:\work\breakout71-testbed\src\` during session 17. This document is the
> canonical reference for building, debugging, and extending the Gymnasium
> environment (`Breakout71Env`). Future agents should NOT need to re-read
> the testbed source.

**Source repository:** <https://github.com/nicogs/breakout71> (or local
`F:\work\breakout71-testbed`)

**Key source files:**

| File | Lines | Purpose |
|------|-------|---------|
| `types.d.ts` | 314 | All TypeScript type definitions |
| `game.ts` | 1036+ | Game loop, input handlers, canvas layout |
| `gameStateMutators.ts` | 2497 | Physics, collisions, scoring, level transitions |
| `newGameState.ts` | 163 | State factory, level generation |
| `upgrades.ts` | 996 | All 63 perks with metadata |
| `gameOver.ts` | 327+ | Game over logic, run history |
| `openUpgradesPicker.ts` | 311 | Perk selection modal, medal system |
| `pure_functions.ts` | 156 | Utility functions, medal thresholds |
| `game_utils.ts` | 375 | Brick coordinates, max_levels, zone borders |
| `settings.ts` | 85 | localStorage settings, totalScore |
| `options.ts` | 124 | Game options, pixelRatio |

---

## Table of Contents

1. [Architecture & Game Loop](#1-architecture--game-loop)
2. [GameState Model](#2-gamestate-model)
3. [Entity Types](#3-entity-types)
4. [Canvas Layout & Coordinate System](#4-canvas-layout--coordinate-system)
5. [Input & Control Model](#5-input--control-model)
6. [Ball Physics](#6-ball-physics)
7. [Brick System & HP](#7-brick-system--hp)
8. [Scoring & Combo System](#8-scoring--combo-system)
9. [Coin Physics & Catching](#9-coin-physics--catching)
10. [Level Progression & Win Conditions](#10-level-progression--win-conditions)
11. [Game Over & Lives](#11-game-over--lives)
12. [Perk Selection System](#12-perk-selection-system)
13. [All 63 Perks Reference](#13-all-63-perks-reference)
14. [Game Initialization](#14-game-initialization)
15. [Selenium Integration Points](#15-selenium-integration-points)
16. [Gymnasium Env Design Implications](#16-gymnasium-env-design-implications)

---

## 1. Architecture & Game Loop

### No State Machine

The game has **no explicit state machine**. Instead, it uses boolean flags on
the `GameState` object:

| Flag | Meaning |
|------|---------|
| `running` | Game is actively ticking (false = paused/menu) |
| `isGameOver` | Game has ended (win or loss) |
| `ballStickToPuck` | Ball is attached to paddle (pre-launch or after rescue) |

Modal overlays (perk picker, game over screen) are handled by the
`asyncAlert()` system which adds `has-alert-open` to `document.body.classList`.

### Main Loop (`game.ts` `tick()`, line 351)

The game loop is `requestAnimationFrame`-driven:

```
tick(elapsed):
  timeDelta = elapsed - lastTick
  lastTick = elapsed
  frames = min(4, timeDelta / (1000/60))   # lag compensation, capped at 4x

  if superhot perk:
    frames *= (lastPuckMove > 0.05) ? 1 : 0.01   # time slows when paddle still

  maxBallSpeed = max(ball.vx, ball.vy for all balls)
  steps = ceil(maxBallSpeed / 8)           # sub-stepping to prevent tunneling

  for i in 0..steps:
    gameStateTick(gameState, frames / steps)

  render(gameState)
  requestAnimationFrame(tick)
```

**Key properties:**
- Frame time normalized to 60 FPS (`frames = 1.0` at exactly 16.67ms)
- Sub-stepping divides each frame into multiple physics steps when ball is fast
- `superhot` perk ties game speed to paddle movement (nearly frozen when idle)
- Maximum `frames` cap of 4 prevents extreme catch-up after tab switches

### `gameStateTick(gameState, frames)` Overview

Called once per sub-step. Executes in order:

1. `normalizeGameState()` -- clamps puck, computes derived values
2. Ball physics (`ballTick()` for each ball)
3. Coin physics (gravity, wall bounces, puck catching)
4. Particle/text/light tick (visual only)
5. Win condition check (remaining bricks, `winAt` timer)
6. Ball-lost / game-over check
7. Level time update (`levelTime += frames * 1000/60`)
8. Statistics update

---

## 2. GameState Model

The `GameState` type (`types.d.ts` line 170-290) contains ~80 fields. Here is
every field, grouped by subsystem.

### Canvas & Layout

| Field | Type | Description |
|-------|------|-------------|
| `canvasWidth` | `number` | Total canvas width (pixels) |
| `canvasHeight` | `number` | Total canvas height (pixels) |
| `offsetX` | `number` | Left margin before game zone (pixels) |
| `offsetXRoundedDown` | `number` | `offsetX - unbounded * brickWidth` (or 0) |
| `gameZoneWidth` | `number` | `brickWidth * gridSize` |
| `gameZoneWidthRoundedUp` | `number` | `canvasWidth - 2 * offsetXRoundedDown` |
| `gameZoneHeight` | `number` | Full height (desktop) or 80% (mobile) |
| `brickWidth` | `number` | Single brick side length (pixels) |
| `ballSize` | `number` | `ceil(20 * pixelRatio)` |
| `coinSize` | `number` | `ceil(14 * pixelRatio)` |
| `puckHeight` | `number` | `ceil(20 * pixelRatio)` |

### Grid & Bricks

| Field | Type | Description |
|-------|------|-------------|
| `gridSize` | `number` | Grid dimension (N x N) |
| `currentLevel` | `number` | 0-indexed current level number |
| `bricks` | `colorString[]` | Color of each brick (empty string = no brick) |
| `brickHP` | `number[]` | Hit points remaining per brick |
| `remainingBricks` | `number` | Count of non-empty bricks |
| `hasPendingBricks` | `boolean` | True if respawn queue has bricks waiting |

### Puck (Paddle)

| Field | Type | Description |
|-------|------|-------------|
| `puckPosition` | `number` | Centre x-coordinate of paddle (canvas pixels) |
| `puckWidth` | `number` | Width of paddle (pixels) |
| `puckHeight` | `number` | Height of paddle (pixels) |
| `lastPuckPosition` | `number` | Previous frame's puck position |
| `lastPuckMove` | `number` | `abs(puckPosition - lastPuckPosition)` |

### Balls

| Field | Type | Description |
|-------|------|-------------|
| `balls` | `Ball[]` | All active balls (see Ball type below) |

### Coins

| Field | Type | Description |
|-------|------|-------------|
| `coins` | `ReusableArray<Coin>` | Pool of active/recycled coins |

### State Flags

| Field | Type | Description |
|-------|------|-------------|
| `running` | `boolean` | True = game is actively ticking |
| `isGameOver` | `boolean` | True = game has ended |
| `ballStickToPuck` | `boolean` | True = ball attached to paddle |
| `needsRender` | `boolean` | Dirty flag for rendering |

### Scoring & Combo

| Field | Type | Description |
|-------|------|-------------|
| `score` | `number` | Current run score (coins caught) |
| `combo` | `number` | Current combo multiplier |
| `lastCombo` | `number` | Combo value before last reset |
| `baseSpeed` | `number` | Current ball base speed |

### Perks

| Field | Type | Description |
|-------|------|-------------|
| `perks` | `PerksMap` | Map of perk_id -> level (0 = not owned) |

### Level & Progression

| Field | Type | Description |
|-------|------|-------------|
| `runLevels` | `Level[]` | All levels selected for this run |
| `level` | `Level` | Current level object |
| `levelTime` | `number` | Milliseconds elapsed in current level |
| `winAt` | `number \| null` | If set, delayed win fires at this levelTime |

### Run Statistics (`runStatistics: RunStats`)

| Field | Type | Description |
|-------|------|-------------|
| `started` | `number` | Timestamp when run started |
| `levelsPlayed` | `number` | Levels completed so far |
| `runTime` | `number` | Total run time (ms) |
| `coins_spawned` | `number` | Total coins spawned |
| `score` | `number` | Total score |
| `bricks_broken` | `number` | Total bricks destroyed |
| `misses` | `number` | Ball returns to paddle without hitting brick |
| `balls_lost` | `number` | Balls that went off-screen |
| `puck_bounces` | `number` | Ball-paddle bounces |
| `wall_bounces` | `number` | Ball-wall bounces |
| `upgrades_picked` | `number` | Perks selected |
| `max_combo` | `number` | Highest combo achieved |

### Level-Scoped Statistics

| Field | Type | Description |
|-------|------|-------------|
| `levelMisses` | `number` | Misses in current level |
| `levelSpawnedCoins` | `number` | Coins spawned in current level |
| `levelCoughtCoins` | `number` | Coins caught in current level |
| `levelLostCoins` | `number` | Coins lost in current level |

### Parameters (`startParams: RunParams`)

| Field | Type | Description |
|-------|------|-------------|
| `computer_controlled` | `boolean` | If true, built-in AI plays |
| (other params) | various | Level selection, custom settings |

### Visual/Audio (not relevant to env)

`particles`, `texts`, `lights`, `respawns`, `flashes`, `lastFlash`,
`lastExplosion`, `combo_broken_steak` -- purely cosmetic.

---

## 3. Entity Types

### Ball

```typescript
interface Ball {
  x: number;             // current x position (canvas pixels)
  y: number;             // current y position
  previousX: number;     // position last frame
  previousY: number;
  vx: number;            // x velocity (pixels per frame-unit)
  vy: number;            // y velocity
  previousVX: number;    // velocity last frame
  previousVY: number;
  piercePoints: number;  // remaining pierce-through uses
  hitSinceBounce: number;        // bricks hit since last paddle bounce
  brokenSinceBounce: number;     // bricks broken since last paddle bounce
  sidesHitsSinceBounce: number;  // wall hits since last paddle bounce
  wrapsSinceBounce: number;      // wrap-arounds since last paddle bounce
  bouncedToEmptyLevel: boolean;  // ball bounced from paddle when no bricks
  sapperUses: number;    // sapper perk uses remaining
  destroyed: boolean;    // true = ball is dead
}
```

### Coin

```typescript
interface Coin {
  points: number;        // score value
  color: string;         // CSS color
  x: number;             // current x (canvas pixels)
  y: number;             // current y
  previousX: number;
  previousY: number;
  vx: number;            // x velocity
  vy: number;            // y velocity
  a: number;             // rotation angle
  sa: number;            // spin angular velocity
  size: number;          // radius
  weight: number;        // gravity multiplier
  destroyed: boolean;    // true = collected or off-screen
  collidedLastFrame: boolean;
  metamorphosisPoints: number;   // points added by metamorphosis perk
  floatingTime: number;          // time coin has been floating
}
```

### Level

```typescript
interface Level {
  name: string;          // display name
  size: number;          // grid dimension (N x N square)
  bricks: colorString[]; // flat array of brick colors (length = size * size)
  bricksCount: number;   // non-empty bricks count
  svg: string;           // optional SVG icon
  color: string;         // theme color
  sortKey: number;
  credit: string;        // attribution
}
```

The grid is always **square** (N x N). Brick at position `(row, col)` is
`bricks[row * size + col]`. An empty string means no brick at that position.

---

## 4. Canvas Layout & Coordinate System

### `fitSize()` (`game.ts` line 146-233)

Called on resize and initialization. Computes all layout constants:

```
canvasWidth  = window.innerWidth * pixelRatio
canvasHeight = window.innerHeight * pixelRatio
gameZoneHeight = mobile ? floor(height * 0.8) : canvasHeight

baseWidth = min(canvasWidth, gameZoneHeight * 0.73 * (gridSize + unbounded*2) / gridSize)
brickWidth = floor(baseWidth / (gridSize + unbounded*2) / 2) * 2   # always even
gameZoneWidth = brickWidth * gridSize

offsetX = floor((canvasWidth - gameZoneWidth) / 2)
offsetXRoundedDown = offsetX - unbounded * brickWidth
  (clamped to 0 if too small)
gameZoneWidthRoundedUp = canvasWidth - 2 * offsetXRoundedDown

ballSize  = ceil(20 * pixelRatio)
coinSize  = ceil(14 * pixelRatio)
puckHeight = ceil(20 * pixelRatio)
```

### Coordinate System

- **Origin:** Top-left corner of the canvas
- **X axis:** Left to right, in canvas pixels
- **Y axis:** Top to bottom, in canvas pixels
- **Game zone:** Horizontally centred within canvas, starting at `offsetX`
- **Puck operates at:** `y = gameZoneHeight` (bottom of game zone)
- **Bricks start at:** `y = 0` (top of game zone), each brick is
  `brickWidth x brickWidth` pixels

### Brick Coordinates (`game_utils.ts`)

```
brickCenterX(index, gameState) = offsetX + (index % gridSize) * brickWidth + brickWidth / 2
brickCenterY(index, gameState) = floor(index / gridSize) * brickWidth + brickWidth / 2
```

### Zone Border Functions (`game_utils.ts`)

```
zoneLeftBorderX(gameState)  = offsetXRoundedDown
zoneRightBorderX(gameState) = offsetXRoundedDown + gameZoneWidthRoundedUp
```

### At 1280x1024 Window (Standard Test Config)

With `pixelRatio = 1`, client area 1264x1016:
- Columns ~324 to ~956 (632px wide game zone)
- Individual bricks ~89x89 pixels
- `ballSize = 20`, `coinSize = 14`, `puckHeight = 20`

---

## 5. Input & Control Model

### Mouse (Primary)

`mousemove` on `#game` canvas triggers:

```javascript
setMousePos(gameState, e.clientX * getPixelRatio())
```

Where `setMousePos()` (`gameStateMutators.ts` line 64):

```javascript
function setMousePos(gameState, x) {
  if (gameState.startParams.computer_controlled) return;  // BLOCKED!
  gameState.puckPosition = Math.round(x);
}
```

**CRITICAL:** `setMousePos()` returns early when `computer_controlled` is true.
External paddle control MUST NOT use `computer_controlled` mode.

### Keyboard (Secondary)

| Key | Action |
|-----|--------|
| `ArrowLeft` | `puckPosition -= gameZoneWidth / 50` |
| `ArrowRight` | `puckPosition += gameZoneWidth / 50` |
| `Shift + Arrow` | 3x movement speed |
| `Space` | Toggle `play()` / `pause()` |

Keyboard movement is incremental and too slow for effective RL control.

### Touch (Mobile/Android)

Hold to play, drag for paddle, release to pause. Not relevant for desktop RL.

### Computer Control (Built-in AI)

`computerControl()` (`gameStateMutators.ts` line 79-119):

```
if ball.y > gameZoneHeight/2 and ball.vy > 0:
  target = closest ball to puck
elif coins exist below gameZoneHeight/2:
  target = closest falling coin
else:
  target = game zone center

puckPosition += clamp((target - puckPosition) / 10, -10, 10)
```

The built-in AI:
- Tracks the closest ball when it's in the lower half moving downward
- Otherwise chases coins or centres the paddle
- Smooth movement (approaches target at 1/10 speed, max 10px/frame)
- Auto-restarts after 30 seconds per level
- **Bypasses `setMousePos()`** -- directly modifies `puckPosition`

---

## 6. Ball Physics

### Speed Regulation (`ballTick()`, `gameStateMutators.ts` line 1693+)

The ball's speed is continuously regulated toward a target:

```
targetSpeed = baseSpeed * sqrt(2)
currentSpeed = sqrt(vx^2 + vy^2)

if currentSpeed < targetSpeed:
  multiplier = 1 + 0.02 / dampener    # speed up
else:
  multiplier = 1 - 0.02 / dampener    # slow down

vx *= multiplier
vy *= multiplier
```

Where `dampener = 1` normally (modified by certain perks).

### Base Speed Calculation

```
baseSpeed = max(3,
  gameZoneWidth / 120
  + currentLevel / 3 / (1 + chill * 10)
  + levelTime / 30000
  - slow_down * 2
)
```

Base speed increases with:
- Higher levels
- Time spent in current level
- Larger game zones

And decreases with `slow_down` and `chill` perks.

### Sub-Stepping (Anti-Tunneling)

The game loop computes:

```
maxBallSpeed = max(abs(ball.vx), abs(ball.vy)) for all balls
steps = ceil(maxBallSpeed / 8)
```

Each frame is divided into `steps` sub-frames, with `frames/steps` passed to
`gameStateTick()`. This prevents the ball from tunneling through bricks at
high speed.

### Puck Bounce Angle

When the ball hits the paddle:

```javascript
angle = atan2(
  -puckWidth / 2,
  (ball.x - puckPosition) * (concave ? -1 / (1 + concave) : 1)
)
speed = sqrt(vx^2 + vy^2)
vx = speed * cos(angle)
vy = speed * sin(angle)   // always negative (upward)
```

The bounce angle depends on where the ball hits relative to paddle centre:
- Centre hit: nearly vertical
- Edge hit: steep angle
- `concave_puck` perk inverts the angle mapping (concave vs convex paddle)

### Ball Loss Detection

A ball is considered lost when:

```
ball.y > gameZoneHeight + ballSize / 2
```

Or when extreme out-of-bounds (off canvas entirely).

### Ball Rescue (Extra Life)

When the **last** ball is lost and `perks.extra_life > 0`:

1. Ball position reset to puck centre
2. `ballStickToPuck = true` (ball attached, waiting for click/space)
3. Game pauses (`running = false`)
4. `perks.extra_life -= 1`

The ball is NOT immediately relaunched -- the player must click/press space.

---

## 7. Brick System & HP

### Brick Colors

The game defines 21 palette colors in `palette.json`, mapped to 11 HSV
detection groups: blue, yellow, red, orange, green, cyan, purple, pink,
beige, white, gray. There is also a special "black" color for bomb bricks.

### Hit Points

```
Normal brick: HP = 1 + perks.sturdy_bricks
Black brick (bomb): HP = 1 (always)
```

### Brick Destruction

When a ball hits a brick:

```
damage = min(ball.piercePoints, max(1, brick.HP + 1))
brick.HP -= damage
ball.piercePoints -= damage

if brick.HP <= 0:
  explodeBrick(gameState, index)
```

`explodeBrick()` performs:
1. Remove brick from grid (`bricks[index] = ""`)
2. Spawn coins (quantity from combo calculation)
3. Spawn visual particles
4. Update `remainingBricks` count
5. If bomb brick: chain-explosion to adjacent bricks
6. Update combo tracking

### Adjacent Bricks & Grid

Grid is N x N (square). Adjacent cells for brick at index `i`:

```
row = floor(i / gridSize)
col = i % gridSize
```

Neighbours: `(row-1, col)`, `(row+1, col)`, `(row, col-1)`, `(row, col+1)`.

**Visual note:** Adjacent bricks of the same color merge visually (no gap),
which makes HSV-based detection see them as one large blob. The auto-annotator
uses grid-splitting logic to handle this.

---

## 8. Scoring & Combo System

### Coin-Based Scoring (NOT Brick-Based)

**Breaking bricks does NOT directly add to the score.** Instead:

1. Breaking a brick spawns coins
2. Coins have physics (fly with gravity, bounce off walls/bricks)
3. Catching coins with the paddle adds their `points` value to `score`
4. Coins that fall off-screen or float too long are lost

### Combo System

The combo multiplier tracks consecutive brick hits:

- `combo` increments when a ball breaks a brick
- `combo` resets to `base_combo` floor when a ball returns to the paddle
  without having hit any brick (a "miss")
- `lastCombo` stores the combo value before the last reset

The number of coins spawned per brick:

```javascript
coinsBoostedCombo(gameState) = ceil(max(combo, lastCombo) * boost)
```

Where `boost` is a multiplier from difficulty-increasing perks:

```
boost = 1
  + sturdy_bricks / 2
  + smaller_puck / 2
  + transparency / 2
  + (black_brick_count * 0.1 * minefield)
```

### `base_combo` Perk Values

The `base_combo` perk (max level 7) sets the combo floor:

| Level | 0 | 1 | 2  | 3  | 4  | 5  | 6  | 7  |
|-------|---|---|----|----|----|----|----|----|
| Floor | 1 | 4 | 8  | 13 | 19 | 26 | 34 | 43 |

### `addToScore(gameState, coin)`

Called when a coin is caught by the paddle. Adds `coin.points` to `score`.
Additional modifiers: `compound_interest` perk adds percentage bonus,
`trickledown` redistributes to other coins.

---

## 9. Coin Physics & Catching

### Coin Spawning

When a brick is destroyed, `coinsBoostedCombo()` coins are spawned at the
brick's position with:

- Random velocity (upward bias)
- `weight = 1` (gravity multiplier)
- `points` based on coin color and modifiers

### Coin Movement (per frame)

```
vy += 0.8 * weight * frames    # gravity
x += vx * frames
y += vy * frames

# Wall bounces
if x < zoneLeft or x > zoneRight:
  vx *= -0.9                    # damped bounce

# Viscosity (from perk)
vx *= (1 - viscosity * 0.01)
vy *= (1 - viscosity * 0.01)
```

### Coin Catching (Puck Collision)

A coin is caught when ALL conditions are true:

```
coin.y > gameZoneHeight - coinRadius - puckHeight
coin.y < gameZoneHeight + puckHeight + coin.vy
abs(coin.x - puckPosition) < coinRadius + puckWidth/2 + puckHeight * (coin.points ? 1 : -1)
NOT isMovingWhilePassiveIncome
```

The catch zone extends slightly above and below the paddle, with the width
depending on whether the coin has points (generous) or not (tighter).

### Coin Loss

Coins are destroyed (lost) when:
- `y > canvasHeight + coinSize` (fell below canvas)
- `floatingTime` exceeds threshold (coin_magnet perk duration)

### Coin Magnets

The `coin_magnet` perk attracts coins toward the paddle:

```
dx = puckPosition - coin.x
coin.vx += dx * 0.01 * coin_magnet
```

The `ball_attracts_coins` perk similarly attracts coins toward balls.

---

## 10. Level Progression & Win Conditions

### Level Structure

A run consists of `max_levels(gameState)` levels:

```javascript
max_levels(gameState) =
  creative ? 1 :
  chill ? currentLevel + 2 :    // infinite (always 2 more)
  7 + perks.extra_levels
```

Levels are selected randomly from a pool at game start (`getRunLevels()`).

### Win Conditions (Checked in `gameStateTick()`)

A level is won when one of these triggers:

1. **Instant win:** `remainingBricks === 0 && !hasPendingBricks && liveCoins === 0`
   - All bricks gone, no respawns pending, no coins on screen
2. **Delayed win:** When `remainingBricks === 0 && !hasPendingBricks` but coins
   still exist:
   - Sets `winAt = levelTime + 5000` (5-second timer)
   - When `levelTime >= winAt`: level advances (any remaining coins are lost)
3. **Ball-lost while winning:** If `winAt` is set and all balls are destroyed,
   the level still advances (player already cleared all bricks)

### Level Transition

When a level is won:

1. Increment `currentLevel`
2. Check if run is complete (`currentLevel >= max_levels(gameState)`)
3. If complete: `gameOver()` with win title
4. If not complete: `openUpgradesPicker(gameState)` -- perk selection modal
5. After perk selection: `setLevel(gameState, currentLevel)` -- load next level

### `setLevel(gameState, levelIndex)` (`gameStateMutators.ts` line 675+)

1. Load level data from `runLevels[levelIndex]`
2. Set `gridSize = level.size`
3. Populate `bricks[]` and `brickHP[]` arrays
4. Reset `levelTime = 0`, `winAt = null`
5. Reset level statistics
6. Place ball on paddle (`ballStickToPuck = true`)
7. `fitSize()` to recalculate layout for new grid size

---

## 11. Game Over & Lives

### No Traditional Lives

There is no lives counter. The survival system is:

1. `extra_life` perk (max 7 levels) provides rescues
2. `multiball` perk spawns additional balls

### Ball Loss Sequence

When `ball.y > gameZoneHeight + ballSize/2`:

```
ball.destroyed = true

if no balls remain (all destroyed):
  if perks.extra_life > 0:
    // RESCUE
    perks.extra_life -= 1
    reset ball to puck center
    ballStickToPuck = true
    running = false (pause)
  else:
    if running AND NOT winAt:
      gameOver()             // true game over
    elif startParams.computer_controlled:
      restart()              // AI auto-restarts
```

### `gameOver()` (`gameOver.ts`)

1. Set `isGameOver = true`
2. Record run statistics
3. Update total score (`settings.ts` localStorage)
4. Display game-over modal via `asyncAlert()`
5. Modal contains: score, level reached, perks collected, run history

### Multi-Ball

When `multiball` perk is active, losing one ball is NOT game over. Only when
**all** balls are destroyed does the rescue/game-over check trigger.

---

## 12. Perk Selection System

### `openUpgradesPicker(gameState)` (`openUpgradesPicker.ts`)

Called between levels. **Skipped if `chill` perk is active.**

### Medal System

Performance on the completed level is evaluated:

| Metric | Gold | Silver |
|--------|------|--------|
| `levelTime` | < 25,000ms | < 45,000ms |
| `catchRate` (coins caught / spawned) | > 98% | > 90% |
| `levelMisses` (ball returns without hit) | < 1 | < 6 |

Each medal provides:
- **Gold:** +1 upgrade point, +3 extra choices
- **Silver:** +1 upgrade point, +1 extra choice

### Upgrade Points & Choices

```
upgradePoints = 1 + gold_medals + silver_medals
numChoices = 3 + perks.one_more_choice + gold_bonuses + silver_bonuses
```

The player picks `upgradePoints` perks from `numChoices` randomly offered.

### Perk Filtering

Offered perks are filtered by:
1. `totalScore >= perk.threshold` (player has earned enough lifetime score)
2. `perk.requires` dependency is owned (e.g., `happy_family` requires `multiball`)
3. Perk is not already at max level

---

## 13. All 63 Perks Reference

Perks are defined in `upgrades.ts`. Each has:
- `id`: string identifier (used as key in `PerksMap`)
- `max`: maximum level
- `threshold` (`t`): minimum total score to unlock
- `category`: beginner, combo, combo_boost, simple, advanced
- `requires`: optional perk dependency

### Beginner Category

| ID | Max | Threshold | Description |
|----|-----|-----------|-------------|
| `slow_down` | 2 | 0 | Reduces ball base speed by 2 per level |
| `extra_life` | 7 | 0 | Ball rescue when last ball lost |
| `bigger_puck` | 2 | 0 | Increases paddle width |
| `skip_last` | 7 | 50 | Auto-destroys last N bricks in level |
| `telekinesis` | 1 | 500 | Coins attracted to paddle |
| `yoyo` | 1 | 600 | Ball returns to paddle after going past |
| `one_more_choice` | 3 | 750 | +1 perk choice per level |
| `concave_puck` | 1 | 950 | Inverts paddle bounce angle mapping |
| `chill` | 1 | 5000 | Infinite levels, skips perk picker |

### Combo Category

| ID | Max | Threshold | Requires | Description |
|----|-----|-----------|----------|-------------|
| `streak_shots` | 1 | 100 | | Combo bonus for consecutive hits |
| `left_is_lava` | 1 | 200 | | Left wall hit breaks combo |
| `right_is_lava` | 1 | 300 | | Right wall hit breaks combo |
| `top_is_lava` | 1 | 400 | | Top wall hit breaks combo |
| `hot_start` | 3 | 4000 | | Start level with higher combo |
| `picky_eater` | 1 | 2000 | | Same-color hits increase combo faster |
| `compound_interest` | 1 | 3000 | | Score bonus based on current score |
| `side_kick` | 3 | 150000 | | Side-wall hit spawns bonus coin |
| `side_flip` | 3 | 150000 | | Side-wall hit changes ball direction |
| `reach` | 1 | 135000 | | Extended paddle hit zone |
| `happy_family` | 1 | 245000 | `multiball` | Multi-ball combo bonus |
| `addiction` | 7 | 165000 | | Combo grows faster |
| `nbricks` | 3 | 90000 | | Multi-brick hit combo bonus |
| `three_cushion` | 1 | 230000 | | Wall-bounce combo bonus |
| `trampoline` | 1 | 115000 | | Paddle-bounce combo bonus |
| `zen` | 1 | 105000 | | Combo stability modifier |
| `asceticism` | 1 | 70000 | | Combo bonus when few perks owned |
| `passive_income` | 4 | 140000 | | Coins spawn over time |

### Combo Boost Category

| ID | Max | Threshold | Description |
|----|-----|-----------|-------------|
| `base_combo` | 7 | 0 | Sets combo floor (1,4,8,13,19,26,34,43) |
| `smaller_puck` | 2 | 1000 | Smaller paddle, higher coin multiplier |
| `soft_reset` | 3 | 18000 | Combo resets less on miss |
| `shunt` | 3 | 80000 | Combo preservation modifier |
| `fountain_toss` | 7 | 170000 | Coins tossed higher (more catch time) |
| `minefield` | 3 | 180000 | Black bricks boost coin multiplier |
| `transparency` | 3 | 190000 | Ball transparency, higher multiplier |
| `sturdy_bricks` | 4 | 40000 | Bricks need more hits, higher multiplier |
| `forgiving` | 1 | 125000 | Reduced combo penalty on miss |

### Simple Category

| ID | Max | Threshold | Requires | Description |
|----|-----|-----------|----------|-------------|
| `pierce_color` | 4 | 15000 | | Pierce through same-color bricks |
| `pierce` | 3 | 1500 | | Pierce through any brick |
| `multiball` | 6 | 800 | | Spawn additional balls |
| `respawn` | 4 | 45000 | | Destroyed bricks respawn after delay |
| `viscosity` | 3 | 0 | | Air resistance on coins (slower drift) |
| `coin_magnet` | 3 | 700 | | Coins attracted toward paddle |
| `metamorphosis` | 1 | 2500 | | Coins transform during flight |
| `sapper` | 7 | 6000 | | Ball destroys adjacent bricks |
| `bigger_explosions` | 1 | 9000 | | Larger explosion radius |
| `extra_levels` | 3 | 13000 | | +1 level per perk level |
| `ball_attracts_coins` | 3 | 130000 | | Coins attracted to ball |
| `clairvoyant` | 1 | 145000 | | Shows ball trajectory |
| `corner_shot` | 1 | 160000 | | Extends puck catch zone into corners |
| `superhot` | 3 | 195000 | | Time tied to paddle movement |
| `bricks_attract_ball` | 1 | 215000 | | Ball curves toward bricks |
| `buoy` | 3 | 220000 | | Ball buoyancy (floats upward) |

### Advanced Category

| ID | Max | Threshold | Requires | Description |
|----|-----|-----------|----------|-------------|
| `ball_repulse_ball` | 3 | 21000 | `multiball` | Balls repel each other |
| `ball_attract_ball` | 3 | 25000 | `multiball` | Balls attract each other |
| `puck_repulse_ball` | 2 | 30000 | | Paddle repels ball |
| `wind` | 3 | 35000 | | Horizontal wind force |
| `helium` | 3 | 65000 | | Upward force on ball |
| `bricks_attract_coins` | 3 | 200000 | | Coins attracted to bricks |
| `wrap_left` | 1 | 240000 | | Ball wraps left edge |
| `wrap_right` | 1 | 245000 | | Ball wraps right edge |
| `double_or_nothing` | 3 | 55000 | | Double score or zero on catch |
| `unbounded` | 3 | 75000 | | Expands game zone beyond grid |
| `etherealcoins` | 1 | 95000 | | Coins pass through bricks |
| `shocks` | 1 | 100000 | `multiball` | Ball collision creates shockwave |
| `sacrifice` | 1 | 110000 | | Sacrifice ball for score bonus |
| `ghost_coins` | 3 | 120000 | | Ghost coins (visual decoys) |
| `implosions` | 1 | 155000 | | Brick destruction creates implosion |
| `limitless` | 1 | 175000 | | No max combo |
| `trickledown` | 1 | 185000 | | Score redistributed to coins |
| `rainbow` | 7 | 205000 | | Color-cycling bricks |
| `golden_goose` | 1 | 210000 | | Gold coins worth more |
| `ottawa_treaty` | 1 | 225000 | | No bomb bricks |
| `sticky_coins` | 1 | 235000 | | Coins stick to paddle |
| `steering` | 4 | 250000 | | Control ball direction |
| `wrap_up` | 1 | 255000 | | Ball wraps at top |

---

## 14. Game Initialization

### `newGameState(params)` (`newGameState.ts`)

Creates a fresh `GameState` with:

1. All fields initialized to defaults
2. Random starting perk selected (filtered by `isStartingPerk()`)
3. Run levels selected via `getRunLevels()`
4. `creative` flag set when:
   - `computer_controlled === true`, OR
   - `sumOfValues(perks) > 1` (multiple starting perks), OR
   - Custom level selected
5. **`window.gameState = gameState`** -- globally exposed for JS access

### Run Level Selection

`getRunLevels()` selects `max_levels()` levels randomly from the available
pool, shuffled. Each level has a different grid size and brick layout.

### Starting the Game

1. `newGameState()` creates state
2. `setLevel(gameState, 0)` loads first level
3. `fitSize()` computes layout
4. `ballStickToPuck = true` (ball attached to paddle)
5. Player clicks or presses Space to launch ball
6. `play()` sets `running = true`

---

## 15. Selenium Integration Points

These are the key hooks for controlling the game from Python via Selenium
`driver.execute_script()`.

### Global State Access

```javascript
window.gameState    // the entire GameState object
```

All fields documented in Section 2 are accessible. Examples:

```javascript
// Read current score
return window.gameState.score;

// Read paddle position
return window.gameState.puckPosition;

// Read remaining bricks
return window.gameState.remainingBricks;

// Read all ball positions
return window.gameState.balls.map(b => ({x: b.x, y: b.y, vx: b.vx, vy: b.vy, destroyed: b.destroyed}));

// Read combo
return window.gameState.combo;
```

### Paddle Control (JS Injection)

**Do NOT use `computer_controlled` mode.** Instead, directly set `puckPosition`:

```javascript
// Set paddle position (canvas pixels from left edge)
window.gameState.puckPosition = Math.round(pixelX);
```

To convert from normalized [-1, 1] action to pixel position:

```javascript
var gs = window.gameState;
var left = gs.offsetX + gs.puckWidth / 2;
var right = gs.offsetX + gs.gameZoneWidth - gs.puckWidth / 2;
var action = {action_value};  // -1 to 1
var pixelX = left + (action + 1) / 2 * (right - left);
gs.puckPosition = Math.round(pixelX);
```

### Game Zone Query

```javascript
return {
  offsetX: window.gameState.offsetX,
  gameZoneWidth: window.gameState.gameZoneWidth,
  puckWidth: window.gameState.puckWidth,
  gameZoneHeight: window.gameState.gameZoneHeight,
  gridSize: window.gameState.gridSize,
  brickWidth: window.gameState.brickWidth
};
```

### Modal Detection

```javascript
// Check if any modal is open
document.body.classList.contains('has-alert-open')

// Get popup content
document.querySelector('#popup')?.innerHTML

// Dismiss modal (game over, alerts)
document.querySelector('#close-modale')?.click()

// Click on game canvas (start/unpause)
document.querySelector('#game')?.click()
```

### Game State Detection (Composite)

```javascript
return (function() {
  if (window.gameState && window.gameState.isGameOver) return "game_over";
  var hasAlert = document.body.classList.contains('has-alert-open');
  if (!hasAlert) return "gameplay";
  var popup = document.querySelector('#popup');
  if (!popup) return "gameplay";
  var content = popup.innerHTML || "";
  if (content.includes("Pick") || content.includes("upgrade")) return "perk_picker";
  return "menu";
})();
```

**Note:** The outer `return` is REQUIRED for Selenium `execute_script()` to
capture the IIFE return value. Without it, returns `null`.

### Perk Selection (Auto-Pick)

```javascript
// Click first available perk button
var btns = document.querySelectorAll('#popup button');
if (btns.length > 0) btns[0].click();
```

### Game Restart

```javascript
// After game over: dismiss modal then click canvas
document.querySelector('#close-modale')?.click();
setTimeout(function() { document.querySelector('#game')?.click(); }, 200);
```

### Key JavaScript Globals

| Global | Type | Notes |
|--------|------|-------|
| `window.gameState` | `GameState` | Full game state object |
| `getPixelRatio()` | `function` | Returns 1 on standard displays |

---

## 16. Gymnasium Env Design Implications

### Current Implementation (v1)

The current `Breakout71Env` uses:
- **Observation:** YOLO-based detection (paddle, ball, bricks, coins)
- **Action:** Continuous `Box(-1, 1)` mapped to `puckPosition` via JS
- **Reward:** Brick-based (not coin-based) due to YOLO coin tracking limitations
- **Episode:** Single level (perk selection = episode boundary)

### Critical Design Considerations

1. **Never set `computer_controlled`** -- `setMousePos()` is blocked in that
   mode. Instead, inject `puckPosition` directly via JS.

2. **The game has no pause-on-step** -- it runs in real-time. The env captures
   frames at ~30 FPS via `time.sleep(1/30)`. This means:
   - Actions take effect at the next render frame
   - YOLO inference adds latency
   - Frame capture adds latency
   - Total step time ~ 33ms + inference + capture

3. **Modal handling is mandatory** -- without it, modals overlay the game and
   YOLO detections fail, causing false termination. The env must check for
   and dismiss modals in `step()`.

4. **Score can be read via JS** -- `window.gameState.score` is directly
   accessible. This enables the `score_delta` reward signal without OCR.

5. **Combo and other state readable via JS** -- ALL game state is accessible
   through `window.gameState`. Future observation extensions can read:
   - `score` (actual score)
   - `combo` (current combo multiplier)
   - `balls` (all ball positions and velocities, not just highest-confidence)
   - `remainingBricks` (exact count, no YOLO noise)
   - `perks` (which perks are active)
   - `levelTime` (time in current level)
   - `extra_life` (remaining rescues)

6. **`unbounded` perk changes the coordinate system** -- when active, the game
   zone expands beyond the grid. `offsetXRoundedDown` shifts and
   `gameZoneWidthRoundedUp` changes. The env must re-query zone boundaries
   after perk selection.

7. **Grid size varies between levels** -- each level can have a different
   `gridSize`, changing `brickWidth` and `gameZoneWidth`. The env should
   re-query layout after level transitions.

### v2 Enhancement Opportunities

| Enhancement | Source | Impact |
|-------------|--------|--------|
| JS-based observation | `window.gameState` | Exact positions, no YOLO noise |
| Coin-based reward | `gameState.score` delta | Correct reward signal |
| Multi-ball tracking | `gameState.balls[]` | Handle multiball perk |
| Combo tracking | `gameState.combo` | Reward combo maintenance |
| Perk-aware episodes | `openUpgradesPicker` | Multi-level training |
| Exact brick count | `gameState.remainingBricks` | No YOLO counting errors |
| Level-time reward | `gameState.levelTime` | Reward faster completion |

### Hybrid Approach (Recommended for v2)

Combine YOLO visual detection with JS state reading:
- **YOLO:** Frame capture for training data, visual debugging, oracle analysis
- **JS bridge:** Exact positions for observation, score for reward, state for
  modal detection

This gives the best of both worlds: visual QA coverage plus precise RL signals.
