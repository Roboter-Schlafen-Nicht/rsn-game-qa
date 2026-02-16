# Breakout 71 -- Player Behavior Specification

> Abstracted from the [Technical Specification](breakout71_game_spec.md)
> during session 18. Describes the game **from a player's perspective** --
> what is observable, what the player can do, and what happens in response.
> This document is the primary reference for designing the RL training
> environment, reward functions, and QA oracles.

---

## Table of Contents

1. [Game Overview](#1-game-overview)
2. [Observable Game States](#2-observable-game-states)
3. [Player Actions](#3-player-actions)
4. [Core Gameplay Loop](#4-core-gameplay-loop)
5. [Paddle Behaviors](#5-paddle-behaviors)
6. [Ball Behaviors](#6-ball-behaviors)
7. [Brick Behaviors](#7-brick-behaviors)
8. [Coin Behaviors](#8-coin-behaviors)
9. [Scoring Behaviors](#9-scoring-behaviors)
10. [Combo Behaviors](#10-combo-behaviors)
11. [Level Progression Behaviors](#11-level-progression-behaviors)
12. [Game Over & Survival Behaviors](#12-game-over--survival-behaviors)
13. [Perk Selection Behaviors](#13-perk-selection-behaviors)
14. [Perk Effects on Gameplay](#14-perk-effects-on-gameplay)
15. [Modal & UI Behaviors](#15-modal--ui-behaviors)
16. [Timing & Performance Behaviors](#16-timing--performance-behaviors)
17. [Env Training Implications](#17-env-training-implications)

---

## 1. Game Overview

Breakout 71 is a brick-breaking game where the player controls a horizontal
paddle at the bottom of the screen. A ball bounces around the play area,
destroying bricks on contact. Destroyed bricks spawn coins that the player
must catch with the paddle to earn points.

A **run** consists of multiple levels. Between levels, the player selects
perks that modify gameplay. The run ends when the player either completes
all levels (win) or loses all balls with no rescues remaining (loss).

### What Makes This Game Different from Classic Breakout

| Classic Breakout | Breakout 71 |
|------------------|-------------|
| Score += points when brick breaks | Score += coins caught by paddle |
| Fixed number of lives (3) | No lives; `extra_life` perk provides rescues |
| Single level or fixed progression | 7+ randomized levels with perk selection between |
| Fixed ball speed | Ball speed increases with level, time, and game zone size |
| Fixed paddle size | Paddle size modified by perks (`bigger_puck`, `smaller_puck`) |
| No combo system | Combo multiplier affects coin spawn quantity |

---

## 2. Observable Game States

The game cycles through distinct states visible to the player. There is no
explicit state machine -- states are determined by combinations of visual
cues and UI elements.

### State: Pre-Launch

- **Visual:** Ball sits on top of the paddle, motionless. Bricks visible above.
- **Trigger:** Game just started, or ball was rescued after loss.
- **Player action required:** Click or press Space to launch the ball.
- **Transitions to:** Gameplay (on click/Space).
- **Observable via JS:** `gameState.ballStickToPuck === true`

### State: Gameplay

- **Visual:** Ball bouncing around the play area. Bricks breaking. Coins
  flying. Score updating.
- **Trigger:** Ball launched from paddle.
- **Player action available:** Move paddle left/right.
- **Transitions to:**
  - Level Complete (all bricks destroyed)
  - Ball Lost (ball falls below paddle)
  - Perk Picker (between levels)
  - Game Over (last ball lost, no rescues)
  - Paused (Space pressed)
- **Observable via JS:** `gameState.running === true && !gameState.isGameOver && !gameState.ballStickToPuck`

### State: Level Complete

- **Visual:** All bricks gone. Remaining coins may still be in flight for
  up to 5 seconds.
- **Trigger:** `remainingBricks === 0` and no pending brick respawns.
- **Player action available:** Continue catching coins during the 5-second
  delay window (if coins are on screen).
- **Transitions to:**
  - Perk Picker (if more levels remain)
  - Game Over / Win (if final level)
- **Observable via JS:** `gameState.remainingBricks === 0 && !gameState.hasPendingBricks`

### State: Perk Picker

- **Visual:** Modal overlay with perk choices (icons, names, descriptions).
- **Trigger:** Level completed and more levels remain.
- **Player action required:** Select one or more perks from the offered choices.
- **Transitions to:** Pre-Launch (next level loaded, ball on paddle).
- **Observable via JS:** `document.body.classList.contains('has-alert-open')` and popup contains "Pick"/"upgrade"

### State: Game Over

- **Visual:** Modal overlay showing final score, level reached, perks
  collected, and run history.
- **Trigger:** Last ball lost with no `extra_life` remaining, OR all levels
  completed.
- **Player action required:** Dismiss the modal to return to the menu.
- **Transitions to:** Menu (on modal dismiss).
- **Observable via JS:** `gameState.isGameOver === true`

### State: Paused

- **Visual:** Game frozen. All entities stop moving.
- **Trigger:** Space pressed during gameplay.
- **Player action required:** Space or click to resume.
- **Transitions to:** Gameplay (on Space/click).
- **Observable via JS:** `gameState.running === false && !gameState.isGameOver && !gameState.ballStickToPuck`

---

## 3. Player Actions

The player has a limited set of actions. The primary action -- paddle
movement -- is the only one relevant during active gameplay.

### Primary: Paddle Movement

- **Input:** Mouse position (horizontal) or arrow keys.
- **Effect:** Paddle moves horizontally to follow the mouse cursor.
- **Constraints:**
  - Paddle cannot move vertically.
  - Paddle is clamped to the game zone boundaries.
  - Mouse input sets absolute position; keyboard moves incrementally.
- **Responsiveness:** Immediate (same frame as input).

### Secondary: Launch Ball

- **Input:** Click on canvas or press Space.
- **Effect:** Ball detaches from paddle and moves upward.
- **When available:** Pre-Launch state only.

### Secondary: Pause / Resume

- **Input:** Press Space during gameplay.
- **Effect:** Toggles between Gameplay and Paused states.

### Secondary: Perk Selection

- **Input:** Click on a perk button in the perk picker modal.
- **Effect:** Selected perk is added to the player's perk set.
- **When available:** Perk Picker state only.

### Secondary: Dismiss Modal

- **Input:** Click the close button on game-over or menu modals.
- **Effect:** Returns to menu / starts a new game.

---

## 4. Core Gameplay Loop

The moment-to-moment gameplay follows this cycle:

```
1. Ball bounces around the play area
2. Ball hits a brick → brick takes damage
3. If brick HP reaches 0 → brick is destroyed
4. Destroyed brick spawns coins
5. Coins fly with physics (gravity, bouncing)
6. Player positions paddle to catch coins → score increases
7. Player positions paddle to bounce the ball → ball continues
8. Repeat until all bricks destroyed or ball lost
```

### Behavioral Contracts (Core Loop)

| ID | Behavior | Precondition | Expected Outcome |
|----|----------|--------------|------------------|
| B-CORE-01 | Ball bounces off paddle | Ball reaches paddle height, paddle is under ball | Ball reverses vertical direction, angle depends on hit position |
| B-CORE-02 | Ball bounces off walls | Ball reaches left, right, or top boundary | Ball reverses direction along the hit axis |
| B-CORE-03 | Ball damages brick | Ball position overlaps a brick | Brick HP decreases by at least 1 |
| B-CORE-04 | Brick destroyed spawns coins | Brick HP reaches 0 | One or more coins appear at the brick's position |
| B-CORE-05 | Coin caught increases score | Coin reaches paddle height and is within paddle width | Score increases by the coin's point value |
| B-CORE-06 | Coin lost off-screen | Coin falls below the bottom of the play area | Coin disappears; no score added |
| B-CORE-07 | Ball lost below paddle | Ball falls below paddle with no bounce | Ball disappears from play area |

---

## 5. Paddle Behaviors

### Movement

| ID | Behavior | Precondition | Expected Outcome |
|----|----------|--------------|------------------|
| B-PAD-01 | Paddle follows mouse | Mouse moves horizontally over canvas | Paddle center moves to mouse x-position |
| B-PAD-02 | Paddle clamped to zone | Mouse moves beyond game zone edges | Paddle stops at zone boundary (half paddle width from edge) |
| B-PAD-03 | Paddle ignores vertical input | Mouse moves vertically | Paddle stays at fixed y-position (bottom of game zone) |
| B-PAD-04 | Keyboard moves paddle | Arrow key pressed | Paddle moves ~2% of game zone width per tick (6% with Shift) |

### Ball Interaction

| ID | Behavior | Precondition | Expected Outcome |
|----|----------|--------------|------------------|
| B-PAD-05 | Centre hit → vertical bounce | Ball hits paddle near centre | Ball bounces nearly straight up |
| B-PAD-06 | Edge hit → angled bounce | Ball hits paddle near edge | Ball bounces at a steep angle away from centre |
| B-PAD-07 | Ball sticks on rescue | Extra life used, ball rescued | Ball placed on paddle centre, waits for launch input |

### Coin Interaction

| ID | Behavior | Precondition | Expected Outcome |
|----|----------|--------------|------------------|
| B-PAD-08 | Paddle catches coin | Coin overlaps paddle's catch zone | Coin disappears, score increases |
| B-PAD-09 | Catch zone is generous | Coin has point value > 0 | Catch zone extends slightly beyond visual paddle edges |

---

## 6. Ball Behaviors

### Movement

| ID | Behavior | Precondition | Expected Outcome |
|----|----------|--------------|------------------|
| B-BALL-01 | Constant speed regulation | Every frame | Ball speed converges toward a target speed |
| B-BALL-02 | Speed increases over time | Time passes within a level | Target speed gradually increases |
| B-BALL-03 | Speed increases per level | Higher level number | Base target speed is higher |
| B-BALL-04 | Ball never stops | Any gameplay state | Speed never drops to zero (minimum enforced) |

### Collisions

| ID | Behavior | Precondition | Expected Outcome |
|----|----------|--------------|------------------|
| B-BALL-05 | Wall bounce | Ball reaches left/right/top wall | Ball reflects off the wall surface |
| B-BALL-06 | Brick collision | Ball overlaps a brick | Ball bounces, brick takes damage |
| B-BALL-07 | No tunneling | Ball moves very fast | Ball never passes through a brick without hitting it |
| B-BALL-08 | Paddle bounce | Ball reaches paddle and is within paddle width | Ball bounces upward at angle determined by hit position |

### Loss

| ID | Behavior | Precondition | Expected Outcome |
|----|----------|--------------|------------------|
| B-BALL-09 | Ball lost below screen | Ball y-position exceeds bottom boundary | Ball is destroyed |
| B-BALL-10 | Last ball lost triggers check | All balls destroyed | Rescue check (extra_life) or game over |

### Visual

| ID | Behavior | Precondition | Expected Outcome |
|----|----------|--------------|------------------|
| B-BALL-11 | Ball has particle trail | Ball is in motion | White particle fragments trail behind ball |
| B-BALL-12 | Ball visually distinct | Any gameplay state | Ball is smaller than bricks, roughly circular, light-colored |

---

## 7. Brick Behaviors

### Appearance

| ID | Behavior | Precondition | Expected Outcome |
|----|----------|--------------|------------------|
| B-BRK-01 | Bricks arranged in grid | Level loaded | Bricks form an N x N square grid (N varies per level) |
| B-BRK-02 | Same-color bricks merge | Adjacent bricks share a color | No visible gap between them (appear as one block) |
| B-BRK-03 | Bricks have distinct colors | Level loaded | Up to 21 different colors from the palette |
| B-BRK-04 | Grid size varies per level | Different levels | Grid dimension N changes, altering brick size and game zone width |

### Destruction

| ID | Behavior | Precondition | Expected Outcome |
|----|----------|--------------|------------------|
| B-BRK-05 | Normal brick: 1 hit to break | Ball hits normal brick (no `sturdy_bricks` perk) | Brick destroyed in one hit |
| B-BRK-06 | Sturdy brick: multiple hits | `sturdy_bricks` perk active | Brick requires 1 + perk_level hits to destroy |
| B-BRK-07 | Bomb brick chain reaction | Ball destroys a black/bomb brick | Adjacent bricks also destroyed in chain |
| B-BRK-08 | Destruction spawns particles | Brick destroyed | Visual explosion/particle effect at brick position |
| B-BRK-09 | Destruction spawns coins | Brick destroyed | Coins appear at brick position, quantity depends on combo |

### Respawn (Perk-Dependent)

| ID | Behavior | Precondition | Expected Outcome |
|----|----------|--------------|------------------|
| B-BRK-10 | Bricks respawn after delay | `respawn` perk active | Previously destroyed bricks reappear after a time |
| B-BRK-11 | Respawned bricks delay level win | Bricks pending respawn | Level cannot be won until respawn queue is empty |

---

## 8. Coin Behaviors

### Spawning

| ID | Behavior | Precondition | Expected Outcome |
|----|----------|--------------|------------------|
| B-COIN-01 | Coins spawn from bricks | Brick destroyed | Coins appear at the brick's former position |
| B-COIN-02 | Coin quantity scales with combo | Higher combo | More coins spawned per brick destruction |
| B-COIN-03 | Coins spawn with random velocity | Coin created | Each coin flies in a slightly different direction (upward bias) |

### Physics

| ID | Behavior | Precondition | Expected Outcome |
|----|----------|--------------|------------------|
| B-COIN-04 | Coins fall with gravity | Coins in flight | Coins accelerate downward over time |
| B-COIN-05 | Coins bounce off walls | Coin reaches left/right boundary | Coin reverses horizontal direction (slightly damped) |
| B-COIN-06 | Coins lost off bottom | Coin falls below canvas | Coin disappears; points not earned |
| B-COIN-07 | Coins have limited lifetime | Coin floats too long | Coin eventually disappears if not caught |

### Catching

| ID | Behavior | Precondition | Expected Outcome |
|----|----------|--------------|------------------|
| B-COIN-08 | Catching coin adds to score | Coin contacts paddle catch zone | Coin value added to score |
| B-COIN-09 | Coin magnet attraction | `coin_magnet` perk active | Coins drift toward paddle position |
| B-COIN-10 | Ball attracts coins | `ball_attracts_coins` perk active | Coins drift toward ball position |

---

## 9. Scoring Behaviors

### Score Accumulation

| ID | Behavior | Precondition | Expected Outcome |
|----|----------|--------------|------------------|
| B-SCORE-01 | Score only from coins | Any gameplay | Breaking bricks alone does NOT increase score |
| B-SCORE-02 | Score increases on catch | Paddle catches a coin | Score increases by the coin's point value |
| B-SCORE-03 | Score never decreases | Any gameplay | Score is monotonically non-decreasing within a run |
| B-SCORE-04 | Score visible on screen | Any gameplay state | Current score displayed in the UI |

### Coin Value

| ID | Behavior | Precondition | Expected Outcome |
|----|----------|--------------|------------------|
| B-SCORE-05 | Coin value from combo | Higher combo at time of brick destruction | Individual coins worth more points |
| B-SCORE-06 | Difficulty perks boost value | `sturdy_bricks`, `smaller_puck`, `transparency`, or `minefield` active | Coin multiplier increases |
| B-SCORE-07 | Compound interest bonus | `compound_interest` perk active | Coin value scales with current score |

---

## 10. Combo Behaviors

The combo system is the central scoring mechanic. Understanding it is
critical for reward shaping.

### Combo Building

| ID | Behavior | Precondition | Expected Outcome |
|----|----------|--------------|------------------|
| B-COMBO-01 | Combo increments on hit | Ball breaks a brick | Combo counter increases |
| B-COMBO-02 | Higher combo → more coins | Combo is high when brick breaks | More coins spawn per brick |
| B-COMBO-03 | Combo persists across bounces | Ball bounces off walls/bricks between paddle bounces | Combo continues to build |

### Combo Breaking

| ID | Behavior | Precondition | Expected Outcome |
|----|----------|--------------|------------------|
| B-COMBO-04 | Miss resets combo | Ball returns to paddle without hitting any brick | Combo resets to floor value |
| B-COMBO-05 | Combo floor from perk | `base_combo` perk active | Combo resets to floor (1/4/8/13/19/26/34/43) instead of 1 |
| B-COMBO-06 | Soft reset reduces penalty | `soft_reset` perk active | Combo resets to a value higher than the floor |
| B-COMBO-07 | Recent combo remembered | Combo just reset | `lastCombo` preserves the pre-reset value briefly for coin spawning |

### Combo Modification by Perks

| ID | Behavior | Precondition | Expected Outcome |
|----|----------|--------------|------------------|
| B-COMBO-08 | Lava walls break combo | Ball hits a lava-designated wall | Combo resets (penalty for hitting that wall) |
| B-COMBO-09 | Picky eater bonus | `picky_eater` perk + ball hits same-color brick consecutively | Combo increases faster |
| B-COMBO-10 | Wall bounce combo bonus | `three_cushion` perk active | Wall bounces contribute to combo |
| B-COMBO-11 | Streak bonus | `streak_shots` perk active | Consecutive brick hits without wall bounce give bonus |

### RL Implications

For the training environment, combo is the most important intermediate
signal:

- **High combo = exponentially more coins = higher score**
- **Missing (ball returns without hitting brick) = combo reset = scoring
  penalty**
- The optimal strategy is to keep the ball hitting bricks continuously
  without returning to the paddle empty-handed.
- However, the player must also catch falling coins, creating a tension
  between keeping the ball in play and positioning for coin collection.

---

## 11. Level Progression Behaviors

### Level Structure

| ID | Behavior | Precondition | Expected Outcome |
|----|----------|--------------|------------------|
| B-LVL-01 | Run has 7+ levels | New game started | Player must complete 7 + `extra_levels` levels to win |
| B-LVL-02 | Levels are randomized | New game started | Level order and selection differ between runs |
| B-LVL-03 | Grid size varies | Different level loaded | Brick grid dimension changes (affects brick size and zone width) |

### Level Completion

| ID | Behavior | Precondition | Expected Outcome |
|----|----------|--------------|------------------|
| B-LVL-04 | Level won when bricks gone | All bricks destroyed, no respawns pending | Level completion triggers |
| B-LVL-05 | Instant win if no coins | Level won and no coins on screen | Immediate transition to next level |
| B-LVL-06 | Delayed win with coins | Level won but coins still on screen | 5-second window for catching remaining coins |
| B-LVL-07 | Ball lost during win → still win | All bricks gone but ball falls off | Level still counts as won (player cleared all bricks) |
| B-LVL-08 | Skip last bricks | `skip_last` perk active | Last N bricks auto-destroyed (player doesn't need to hit them all) |

### Level Transition

| ID | Behavior | Precondition | Expected Outcome |
|----|----------|--------------|------------------|
| B-LVL-09 | Perk picker between levels | Level won, more levels remain | Perk selection modal appears |
| B-LVL-10 | New level resets ball | Next level loaded | Ball placed on paddle, waiting for launch |
| B-LVL-11 | New level resets layout | Next level loaded | Grid size may change, game zone width recalculated |
| B-LVL-12 | Speed resets per level | Next level loaded | Level timer resets (affects ball speed calculation) |

---

## 12. Game Over & Survival Behaviors

### Ball Loss Sequence

| ID | Behavior | Precondition | Expected Outcome |
|----|----------|--------------|------------------|
| B-SURV-01 | Single ball lost → no effect | Multiple balls active (`multiball` perk) | Other balls continue; game proceeds |
| B-SURV-02 | Last ball lost with rescue | All balls destroyed, `extra_life > 0` | Ball rescued: placed on paddle, game pauses |
| B-SURV-03 | Rescue consumes extra life | Rescue triggered | `extra_life` decrements by 1 |
| B-SURV-04 | Rescue requires re-launch | Ball rescued | Player must click/Space to relaunch |
| B-SURV-05 | Last ball lost, no rescue | All balls destroyed, `extra_life === 0` | Game over (loss) |

### Game Over

| ID | Behavior | Precondition | Expected Outcome |
|----|----------|--------------|------------------|
| B-SURV-06 | Game over on loss | Last ball lost, no rescue available | Game over modal appears with final stats |
| B-SURV-07 | Game over on win | All levels completed | Game over modal appears with winning title |
| B-SURV-08 | Score persists | Game over | Total score updated in persistent storage |
| B-SURV-09 | Modal must be dismissed | Game over modal visible | Player must click close to proceed |

---

## 13. Perk Selection Behaviors

### Perk Picker

| ID | Behavior | Precondition | Expected Outcome |
|----|----------|--------------|------------------|
| B-PERK-01 | Perk picker appears between levels | Level completed, more levels remain | Modal with perk choices shown |
| B-PERK-02 | Choices are random subset | Perk picker opened | 3+ perks offered from available pool |
| B-PERK-03 | Player selects N perks | Upgrade points available | Player clicks to select (1 + medal bonuses) perks |
| B-PERK-04 | Selected perks take effect immediately | Perk selected | Perk modifies gameplay for all subsequent levels |
| B-PERK-05 | Perks can stack | Same perk offered again (if not at max) | Perk level increases, effect strengthens |
| B-PERK-06 | Chill perk skips picker | `chill` perk active | No perk selection between levels |

### Medal System

| ID | Behavior | Precondition | Expected Outcome |
|----|----------|--------------|------------------|
| B-PERK-07 | Fast level → gold medal | Level completed in < 25 seconds | Gold time medal (+1 upgrade point, +3 choices) |
| B-PERK-08 | Fast level → silver medal | Level completed in 25-45 seconds | Silver time medal (+1 upgrade point, +1 choice) |
| B-PERK-09 | High catch rate → gold medal | Caught > 98% of spawned coins | Gold catch medal |
| B-PERK-10 | High catch rate → silver medal | Caught 90-98% of spawned coins | Silver catch medal |
| B-PERK-11 | Few misses → gold medal | < 1 paddle miss (ball return without hit) | Gold accuracy medal |
| B-PERK-12 | Few misses → silver medal | 1-5 paddle misses | Silver accuracy medal |
| B-PERK-13 | Medals grant choices | Medals earned | More perks offered and more upgrade points |

---

## 14. Perk Effects on Gameplay

Perks significantly alter gameplay behavior. This section groups perks by
the type of behavioral change they create, relevant to how an RL agent
must adapt.

### Perks That Change Paddle Behavior

| Perk | Effect on Player Experience |
|------|----------------------------|
| `bigger_puck` | Paddle wider → easier to catch coins and bounce ball |
| `smaller_puck` | Paddle narrower → harder to catch, but higher coin multiplier |
| `concave_puck` | Bounce angle mapping inverted → centre hits angle outward, edge hits go straight |
| `corner_shot` | Extended catch zone into corners → more forgiving at edges |
| `reach` | Extended hit zone → ball bounces from wider area |
| `sticky_coins` | Coins stick to paddle → automatic catching |

### Perks That Change Ball Behavior

| Perk | Effect on Player Experience |
|------|----------------------------|
| `slow_down` | Ball moves slower → more reaction time |
| `multiball` | Multiple balls → harder to track, but more brick hits |
| `yoyo` | Ball returns to paddle after going past → no ball loss |
| `pierce` | Ball passes through bricks → hits multiple bricks per pass |
| `pierce_color` | Ball passes through same-color bricks → combo chains |
| `sapper` | Ball destroys adjacent bricks on hit → area damage |
| `bricks_attract_ball` | Ball curves toward bricks → more hits |
| `buoy` | Ball floats upward → stays in play longer |
| `superhot` | Time tied to paddle movement → game freezes when paddle is still |
| `clairvoyant` | Ball trajectory shown → player can predict bounces |
| `steering` | Player can influence ball direction → more control |
| `helium` | Upward force on ball → counteracts gravity |
| `wind` | Horizontal drift → ball path is less predictable |

### Perks That Change Coin Behavior

| Perk | Effect on Player Experience |
|------|----------------------------|
| `coin_magnet` | Coins drift toward paddle → easier to catch |
| `ball_attracts_coins` | Coins drift toward ball → coins follow ball path |
| `viscosity` | Coins move slower → more catch time |
| `metamorphosis` | Coins transform mid-flight → value changes |
| `fountain_toss` | Coins tossed higher → longer catch window |
| `etherealcoins` | Coins pass through bricks → coins don't get stuck |
| `bricks_attract_coins` | Coins drift toward bricks → harder to catch |
| `ghost_coins` | Decoy coins appear → player must distinguish real from fake |
| `golden_goose` | Gold coins worth more → higher value targets |
| `passive_income` | Coins spawn over time → score without breaking bricks |

### Perks That Change Level Structure

| Perk | Effect on Player Experience |
|------|----------------------------|
| `extra_levels` | More levels per run → longer game |
| `chill` | Infinite levels → game never ends (no perk picker) |
| `respawn` | Bricks reappear → level takes longer to clear |
| `sturdy_bricks` | Bricks need multiple hits → more time per level, but higher coin multiplier |
| `minefield` | Black/bomb bricks boost coin multiplier → risk/reward from chain explosions |
| `unbounded` | Game zone expands → coordinate system changes, more play area |
| `rainbow` | Bricks cycle colors → visual complexity |

### Perks That Change Combo Rules

| Perk | Effect on Player Experience |
|------|----------------------------|
| `base_combo` | Higher combo floor → combo never drops below threshold |
| `soft_reset` | Combo drops less on miss → more forgiving |
| `forgiving` | Reduced combo penalty → similar to soft_reset |
| `hot_start` | Start levels with combo → immediate high scoring |
| `streak_shots` | Consecutive hits without wall bounce → faster combo growth |
| `picky_eater` | Same-color consecutive hits → faster combo growth |
| `addiction` | General combo growth speed → faster combo buildup |
| `limitless` | No combo cap → unlimited combo potential |
| `zen` | Combo stability → less variance in combo changes |
| `asceticism` | Fewer perks = better combo → incentivizes minimal perk selection |

### Perks That Change Scoring

| Perk | Effect on Player Experience |
|------|----------------------------|
| `compound_interest` | Coin value scales with score → exponential growth |
| `double_or_nothing` | 50/50 chance of double value or zero → high variance |
| `trickledown` | Score redistributed to coins → score spread |
| `sacrifice` | Lose a ball for score bonus → intentional ball sacrifice |

---

## 15. Modal & UI Behaviors

Modals interrupt gameplay and must be handled by the RL environment.

### Modal Types

| ID | Behavior | Trigger | Player Action | Auto-Dismissable |
|----|----------|---------|---------------|------------------|
| B-UI-01 | Perk picker modal | Level completed | Select perk(s) | Yes (click first button) |
| B-UI-02 | Game over modal | Win or loss | Click close | Yes (click `#close-modale`) |
| B-UI-03 | Menu/start screen | Game loaded or after game over | Click canvas | Yes (click `#game`) |
| B-UI-04 | Pause overlay | Space pressed | Space or click | Yes (click canvas) |

### Modal Detection

| ID | Behavior | Precondition | Expected Outcome |
|----|----------|--------------|------------------|
| B-UI-05 | Modals block gameplay | Any modal open | Ball/coins frozen, input redirected to modal |
| B-UI-06 | Modal has CSS indicator | Modal opens | `document.body.classList` contains `has-alert-open` |
| B-UI-07 | Modal overlays game | Modal open | Game canvas partially or fully obscured |
| B-UI-08 | YOLO detections unreliable | Modal open | Object detection fails due to overlay → env must detect and dismiss modals before relying on YOLO |

---

## 16. Timing & Performance Behaviors

### Real-Time Execution

| ID | Behavior | Precondition | Expected Outcome |
|----|----------|--------------|------------------|
| B-TIME-01 | Game runs at 60 FPS target | Browser rendering | Physics scaled to maintain consistent speed |
| B-TIME-02 | Lag compensation | Frame takes longer than 16.67ms | Game catches up (capped at 4x normal speed) |
| B-TIME-03 | No step-by-step mode | RL env controls game | Game runs in real-time; env must keep up |
| B-TIME-04 | Action-to-effect delay | Action injected via JS | Effect visible on next render frame |

### Observable Timing

| ID | Behavior | Precondition | Expected Outcome |
|----|----------|--------------|------------------|
| B-TIME-05 | Ball speed increases within level | Time passes | Ball gradually accelerates (levelTime term in speed formula) |
| B-TIME-06 | Level timer visible | Gameplay active | `levelTime` accessible via JS (not visually displayed) |
| B-TIME-07 | 5-second win delay | Level cleared with coins remaining | Coins have 5 seconds to be caught before forced transition |

---

## 17. Env Training Implications

This section maps player behaviors to concrete RL training considerations.

### Reward Signal Design

Based on the behavioral contracts above, the ideal reward function should
capture these player-observable outcomes:

| Signal | Behavior Source | Why It Matters |
|--------|----------------|----------------|
| Coin caught (score delta) | B-SCORE-02 | Primary scoring mechanic; aligns with game objective |
| Brick destroyed | B-BRK-09 | Intermediate signal; leads to coin spawning |
| Combo maintained | B-COMBO-01 to B-COMBO-03 | Multiplier effect on future rewards |
| Combo broken | B-COMBO-04 | Penalty signal; miss = wasted opportunity |
| Ball lost | B-BALL-09 | Catastrophic failure (potential episode end) |
| Level cleared | B-LVL-04 | Episode milestone |
| Medal earned | B-PERK-07 to B-PERK-12 | Performance quality indicator |

### Episode Boundaries

| Boundary | Relevant Behaviors | Detection Method |
|----------|-------------------|------------------|
| Episode start | B-BALL pre-launch → gameplay | Ball launches from paddle |
| Episode end (win) | B-LVL-04, B-LVL-05, B-LVL-06 | No bricks remaining for N frames |
| Episode end (loss) | B-SURV-05, B-SURV-06 | No ball detected for N frames, bricks unchanged |
| Episode end (rescue) | B-SURV-02, B-SURV-04 | Ball sticks to paddle (optional: treat as same episode) |
| Truncation | B-TIME-03 | Max steps exceeded |

### Observation Priorities

Based on which behaviors matter most for the player:

| Priority | Observable | Behaviors | Current Status |
|----------|-----------|-----------|----------------|
| Critical | Paddle position | B-PAD-01, B-PAD-02 | Active (YOLO) |
| Critical | Ball position | B-BALL-01 to B-BALL-08 | Active (YOLO) |
| Critical | Ball velocity | B-BALL-01, B-PAD-05 | Active (frame delta) |
| High | Brick count | B-BRK-05, B-LVL-04 | Active (YOLO count) |
| High | Score | B-SCORE-01 to B-SCORE-04 | Available via JS |
| High | Combo | B-COMBO-01 to B-COMBO-07 | Available via JS |
| Medium | Coin positions | B-COIN-04 to B-COIN-08 | YOLO `powerup` class |
| Medium | Extra lives | B-SURV-02, B-SURV-03 | Available via JS |
| Low | Active perks | B-PERK-04 | Available via JS |
| Low | Level time | B-TIME-05, B-TIME-06 | Available via JS |

### Key Strategic Tensions

The game creates several competing objectives that the RL agent must
balance:

1. **Ball control vs. coin catching** -- The paddle must both bounce the
   ball AND catch coins. These require being in different positions.

2. **Combo vs. safety** -- Keeping combo high means the ball should hit
   bricks continuously. But risky angles increase ball-loss probability.

3. **Speed vs. accuracy** -- Faster level completion earns medals (more
   perks). But rushing increases miss rate and coin loss.

4. **Perk synergies** -- Some perks fundamentally change optimal strategy
   (e.g., `superhot` rewards stillness between moves; `yoyo` eliminates
   ball-loss risk; `coin_magnet` reduces the catch positioning burden).

---

*This specification is derived from the
[Technical Specification](breakout71_game_spec.md) and the
[Env Specification](breakout71_env_spec.md). For implementation details,
internal data structures, and source code references, see those documents.*
