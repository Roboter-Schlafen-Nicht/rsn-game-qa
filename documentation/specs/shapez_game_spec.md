# shapez.io -- Game Technical Specification

Comprehensive reference for the shapez.io game mechanics, derived from
the [shapez.io source code](https://github.com/tobspr-games/shapez.io)
and the RSN Game QA plugin implementation. Used by the platform to
inform environment design, oracle configuration, and reward engineering.

**Game version:** GitHub HEAD (GPL-3.0)
**Genre:** Factory builder / automation
**Platform:** Browser (HTML5 Canvas, custom JS engine)
**Input:** Mouse + keyboard
**Build toolchain:** Node.js 16, Yarn, Java, ffmpeg

---

## Table of Contents

1. [Key Source Files](#1-key-source-files)
2. [Game Overview](#2-game-overview)
3. [Application Architecture](#3-application-architecture)
4. [Game States](#4-game-states)
5. [Level System and Goals](#5-level-system-and-goals)
6. [Building System](#6-building-system)
7. [Shape System](#7-shape-system)
8. [Input Model](#8-input-model)
9. [Rendering and Canvas](#9-rendering-and-canvas)
10. [Upgrade System](#10-upgrade-system)
11. [Map and Entity Model](#11-map-and-entity-model)
12. [UI and Modals](#12-ui-and-modals)
13. [Audio](#13-audio)
14. [Persistence and Savegames](#14-persistence-and-savegames)
15. [Build Toolchain](#15-build-toolchain)
16. [Selenium Integration Points](#16-selenium-integration-points)

---

## 1. Key Source Files

shapez.io uses a custom engine with ES5-style classes. The source is
organized under `src/js/`:

| Path | Purpose | Key exports |
|---|---|---|
| `src/js/application.js` | Main application class | `Application` |
| `src/js/states/main_menu.js` | Main menu state | `MainMenuState` |
| `src/js/states/ingame.js` | In-game state | `InGameState` |
| `src/js/states/preload.js` | Asset preloading | `PreloadState` |
| `src/js/game/hud/` | HUD components | Building toolbar, notifications |
| `src/js/game/buildings/` | Building definitions | All building types |
| `src/js/game/shape_definition.js` | Shape grammar | Shape parsing/rendering |
| `src/js/game/hub_goals.js` | Level goals | `HubGoals` class |
| `src/js/savegame/` | Save/load system | `SavegameManager`, `Savegame` |
| `src/js/game/map.js` | World map | `GameMap`, entity management |
| `src/js/game/root.js` | Game root | `GameRoot` (accessible as `globalRoot`) |
| `gulp/gulpfile.js` | Build configuration | Gulp tasks, BrowserSync |

---

## 2. Game Overview

shapez.io is a factory-building automation game. The player constructs
factories on an infinite 2D grid by placing buildings (extractors,
conveyors, processors) to mine, transport, and process geometric shapes.
Processed shapes are delivered to a central Hub to complete level goals.

**Core loop:**
1. A level goal specifies a required shape and quantity
2. The player builds extractors on shape deposits to mine raw shapes
3. Conveyors transport shapes toward the Hub
4. Processing buildings (cutters, rotators, painters, stackers) modify
   shapes to match the goal
5. Shapes delivered to the Hub count toward the goal
6. When the goal is met, the next level unlocks with a harder shape
7. After level 26, the game enters freeplay mode with procedurally
   generated goals

**Key differences from arcade games:**
- **No natural game-over.** The player cannot die or lose. The game is
  purely creative and goal-oriented.
- **Persistent state.** Factories remain between levels. The world is
  an ever-growing construction.
- **Mouse-primary input.** Building placement, selection, and camera
  control are all mouse-driven with keyboard shortcuts.
- **Time-independent.** The game runs continuously; factories produce
  shapes even when the player is idle.

---

## 3. Application Architecture

### Global Access Points

shapez.io exposes its application via `window.shapez.GLOBAL_APP`:

```javascript
// Available from boot (PreloadState onward)
window.shapez.GLOBAL_APP  // Application instance
window.shapez.GLOBAL_APP.stateMgr  // State manager
window.shapez.GLOBAL_APP.settings  // Settings manager
window.shapez.GLOBAL_APP.savegameMgr  // Savegame manager

// Available only during InGameState
window.globalRoot  // GameRoot instance (game logic)
window.globalRoot.hubGoals  // Level goals
window.globalRoot.map  // World map
window.globalRoot.camera  // Camera controller
window.globalRoot.entityMgr  // Entity manager
```

### Application Class (`Application`)

The `Application` class manages:
- State machine (`stateMgr`) for transitioning between PreloadState,
  MainMenuState, and InGameState
- Settings persistence
- Savegame management
- Audio engine
- Analytics (disabled in open-source version)

### State Manager

States are managed by `ApplicationStateMgr`:
- `moveToState(stateClass, payload)` transitions to a new state
- States have `onEnter()`, `onLeave()`, `update()`, `render()` methods
- Only one state is active at a time

---

## 4. Game States

shapez.io has a formal state machine. States are identified by the
`document.body.id` attribute:

| State | `document.body.id` | Description |
|---|---|---|
| Preload | `state_PreloadState` | Asset loading, splash screen |
| Main Menu | `state_MainMenuState` | New game, continue, settings |
| In Game | `state_InGameState` | Active gameplay |

### State Transitions

```
PreloadState --[assets loaded]--> MainMenuState
MainMenuState --[New Game]--> InGameState (new savegame)
MainMenuState --[Continue]--> InGameState (loaded savegame)
InGameState --[Back to Menu]--> MainMenuState
```

### In-Game Sub-States

During `InGameState`, the game has several overlay states:

| Sub-State | Trigger | Description |
|---|---|---|
| Playing | Default | Normal gameplay |
| Level complete | Shape goal met | Unlock notification + reward |
| Settings | Escape key | Settings overlay |
| Modal dialog | Various | Confirmation dialogs |

### State Detection via DOM

```javascript
// Read current state from body element
var bodyId = document.body.id;
// "state_PreloadState", "state_MainMenuState", "state_InGameState"
```

---

## 5. Level System and Goals

### Level Structure

shapez.io has 26 hand-crafted levels plus infinite freeplay:

| Level | Required Shape | Approximate Count |
|---|---|---|
| 1 | Circle (CuCuCuCu) | 30 |
| 2 | Rectangle (RuRuRuRu) | 40 |
| 3-5 | Simple single-color shapes | 50-100 |
| 6-10 | Two-layer shapes | 100-300 |
| 11-20 | Multi-color, multi-layer | 300-1000 |
| 21-26 | Complex stacked shapes | 1000-5000 |
| 27+ | Freeplay (procedural) | Increasing |

### Hub Goals (`HubGoals`)

The `HubGoals` class tracks level progression:

| Property | Type | Description |
|---|---|---|
| `level` | number | Current level (1-based) |
| `currentGoal` | object | `{definition, required, reward}` |
| `storedShapes` | object | Shapes delivered but not yet counted |
| `upgradeLevels` | object | Upgrade tiers per category |
| `isEndOfDemoReached` | boolean | Demo mode flag |

### Goal Completion

When `shapesDelivered >= goalRequired`:
1. Level increments
2. A notification overlay appears briefly
3. Next goal activates immediately
4. Player's factory continues running (no reset)

### Goal Progress Reading

```javascript
var goals = globalRoot.hubGoals;
var level = goals.level;
var required = goals.currentGoal.required;
var definition = goals.currentGoal.definition;
```

---

## 6. Building System

### Building Types

shapez.io has approximately 10 core building types:

| Building | Key | Function |
|---|---|---|
| Belt (conveyor) | `1` | Transports shapes in one direction |
| Extractor/Miner | `2` | Mines shapes from deposits |
| Cutter | `3` | Cuts shapes in half (left/right) |
| Rotator | `4` | Rotates shapes 90 degrees |
| Stacker | `5` | Stacks two shapes on top of each other |
| Mixer | `6` | Mixes two colors |
| Painter | `7` | Paints shapes a specific color |
| Trash | `8` | Destroys items |
| Balancer | `9` | Splits/merges belt flows |
| Tunnel | `0` | Underground belt passage |

### Building Placement

1. Player selects a building type (keyboard shortcut or toolbar click)
2. A ghost preview follows the mouse cursor
3. Left-click places the building at the grid position
4. Buildings snap to the grid (each cell is one building)
5. Right-click deletes a building at the cursor position
6. `R` key rotates the building before placement (4 orientations)

### Building Properties

Each building has:
- Grid position (x, y)
- Rotation (0, 90, 180, 270 degrees)
- Input/output slots (directional)
- Processing logic (building-type-specific)

---

## 7. Shape System

### Shape Grammar

Shapes in shapez.io are defined by a string notation:

```
Format: [quad1][quad2][quad3][quad4]
Each quad: [shape_type][color]

Shape types: C=circle, R=rectangle, S=star, W=windmill, -=empty
Colors: r=red, g=green, b=blue, u=uncolored, y=yellow,
        p=purple, c=cyan, w=white

Examples:
  CuCuCuCu  = full uncolored circle (4 quadrants)
  RuRu----  = half uncolored rectangle
  CrCgCbCy  = circle with 4 different colored quadrants
  Sr--Sb--  = star with 2 quadrants (red and blue)
```

### Multi-Layer Shapes

Shapes can be stacked (up to 4 layers), separated by `:`:

```
CuCuCuCu:RrRrRrRr  = circle layer + rectangle layer on top
```

### Shape Deposits

The infinite map has procedurally generated deposits of basic shapes:
- Circles, rectangles, stars, windmills
- Uncolored by default
- Color deposits appear near the Hub area
- Deposits are infinite (never depleted)

---

## 8. Input Model

### Mouse Controls

| Action | Input | Description |
|---|---|---|
| Pan camera | Middle-click drag / WASD | Move the view |
| Zoom | Scroll wheel | Zoom in/out |
| Place building | Left-click | Place selected building at cursor |
| Delete building | Right-click | Remove building at cursor |
| Select building | Left-click on building | Show info overlay |
| Drag-place | Left-click + drag | Place buildings along a line |

### Keyboard Shortcuts

| Key | Action | Description |
|---|---|---|
| `1`-`0` | Select building | Quick-select building type |
| `R` | Rotate | Rotate selected building 90 degrees |
| `Q` | Previous variant | Cycle building variant backward |
| `E` | Next variant | Cycle building variant forward |
| `W`/`A`/`S`/`D` | Pan camera | Move camera up/left/down/right |
| `Space` | Center on Hub | Reset camera to Hub position |
| `Escape` | Settings/cancel | Open settings or cancel placement |
| `Delete` | Mass delete | Enter mass-delete mode |
| `Ctrl+Z` | Undo | Undo last placement |

### Input Processing

shapez.io uses custom event handling through its HUD system. Input
events on the canvas are processed by the active game state and
routed to the appropriate handler (camera controller, building
placer, HUD elements).

---

## 9. Rendering and Canvas

### Canvas Setup

shapez.io creates its own canvas element dynamically:

```html
<canvas id="ingame_Canvas"></canvas>
```

The canvas is created when `InGameState` is entered and destroyed
when leaving. It is NOT present during `MainMenuState` or
`PreloadState`.

### Render Pipeline

1. Clear canvas
2. Draw map background (grid, terrain)
3. Draw all entities (buildings, belts, items on belts)
4. Draw shape deposits
5. Draw the Hub (central delivery point)
6. Draw HUD overlays (toolbar, notifications, tooltips)
7. Draw ghost building preview (if placing)

### Resolution

The canvas resizes to fill the browser window. The game uses a
camera system with zoom levels to show more or less of the factory.
Default zoom shows approximately a 20x20 grid area.

### Canvas Selector

```javascript
document.getElementById("ingame_Canvas")
```

Note: This canvas only exists during `InGameState`. Attempts to
capture it during other states will fail.

---

## 10. Upgrade System

### Upgrade Categories

shapez.io has upgrade categories that improve building performance:

| Category | Effect |
|---|---|
| Belt speed | Increases conveyor belt speed |
| Miner speed | Increases extractor throughput |
| Processing speed | Increases building processing speed |
| Painting speed | Increases painter throughput |

### Upgrade Tiers

Each category has multiple tiers. Each tier requires delivering a
specific shape in a specific quantity. Upgrades are persistent within
a save file.

### Upgrade Levels Reading

```javascript
var upgrades = globalRoot.hubGoals.upgradeLevels;
// Returns: { belt: N, miner: N, processors: N, painting: N }
```

---

## 11. Map and Entity Model

### Infinite Grid

The map is an infinite 2D grid. Each cell can contain one building.
The player can scroll and zoom to any position.

### Entity System

All placed buildings are entities managed by `GameRoot.entityMgr`:

| Property | Description |
|---|---|
| `entityMgr.entities` | Array of all entities |
| `map.entityMap_` | Spatial lookup map (position → entity) |

### Entity Count

```javascript
var count = 0;
var entityMap = globalRoot.map.entityMap_;
if (entityMap && entityMap.size) count = entityMap.size;
else if (entityMap) {
    for (var key in entityMap) { if (entityMap.hasOwnProperty(key)) count++; }
}
```

### Hub Position

The Hub is always at the center of the map (coordinates 0,0).
New games start with the camera centered on the Hub.

---

## 12. UI and Modals

### HUD Components

During `InGameState`, the HUD displays:
- **Building toolbar:** Bottom of screen, shows available buildings
- **Level indicator:** Current level and goal shape
- **Resource counters:** Shapes delivered, upgrade progress
- **Notifications:** Level completion, achievements
- **Tooltips:** Building descriptions on hover

### Modal Dialogs

shapez.io uses modal dialogs for:
- Level completion rewards
- Settings menu
- Confirmation dialogs (delete save, etc.)
- Tutorial hints (can be disabled)

### Tutorial System

Tutorial hints appear during early gameplay. They can be disabled via
settings:

```javascript
app.settings.updateSetting("offerHints", false);
```

The RSN platform disables tutorials during training to avoid
interfering with the agent.

---

## 13. Audio

shapez.io has background music and sound effects:
- **Music:** Ambient background tracks
- **SFX:** Building placement, shape delivery, level completion

Audio can be muted via settings:

```javascript
app.settings.updateSetting("musicVolume", 0);
app.settings.updateSetting("soundVolume", 0);
```

---

## 14. Persistence and Savegames

### Savegame Manager

`Application.savegameMgr` manages save files:

| Method | Description |
|---|---|
| `createNewSavegame(metadata)` | Creates a new save slot |
| `readSavegameData(savegame)` | Reads save data |
| `deleteSavegame(savegame)` | Deletes a save file |

### Savegame Format

Save files are JSON objects containing:
- Map state (all entities and their positions)
- Hub goals progress (level, shapes delivered)
- Upgrade levels
- Camera position and zoom
- Game time
- Version information for migration

### Savegame Migration

Old save files are migrated to the current version via
`savegame.migrate(data)`. This handles schema changes between versions.

### Savegame Injection

The RSN platform can inject save files to start episodes from
interesting mid-game states:

```javascript
// Create new savegame slot
var savegame = app.savegameMgr.createNewSavegame({
    name: "RSN_injected",
    date: Date.now()
});
// Write save data
savegame.currentData = data;
savegame.migrate(data);
// Transition to InGameState
app.stateMgr.moveToState("InGameState", { savegame: savegame });
```

This enables RL training from late-game states with large factories,
bypassing the "agent can't build" problem.

---

## 15. Build Toolchain

### Requirements

| Tool | Version | Purpose |
|---|---|---|
| Node.js | 16 (via nvm) | JavaScript runtime |
| Yarn | 1.22+ | Package manager |
| Java | 21+ (OpenJDK) | Build tool dependency |
| ffmpeg | 6.1+ | Audio processing |

### Build Commands

```bash
# Install dependencies (from repo root)
yarn

# Start dev server (from gulp/ directory)
cd gulp && yarn && yarn gulp
```

### Dev Server

The dev server uses BrowserSync and runs on port 3005:
- Hot reload on source changes
- Source maps for debugging
- Serves from `build/` directory
- BrowserSync GUI auto-open disabled via `open: false` in gulpfile.js

### Source Structure

```
shapez.io/
├── src/
│   ├── js/          # Game source (ES5 classes)
│   ├── css/         # Stylesheets (SCSS)
│   └── html/        # HTML templates
├── gulp/
│   ├── gulpfile.js  # Build configuration
│   └── package.json # Build dependencies
├── res_raw/         # Raw assets (sprites, sounds)
└── package.json     # Root dependencies
```

---

## 16. Selenium Integration Points

These are the JavaScript globals, methods, and DOM elements used by the
RSN platform to control and observe shapez.io via Selenium
`execute_script()`.

### Readable Globals

| Global | Type | Available | Description |
|---|---|---|---|
| `window.shapez.GLOBAL_APP` | Application | From boot | Main application instance |
| `window.shapez.GLOBAL_APP.stateMgr` | StateMgr | From boot | State machine |
| `window.shapez.GLOBAL_APP.settings` | Settings | From boot | Settings manager |
| `window.shapez.GLOBAL_APP.savegameMgr` | SavegameMgr | From boot | Save file manager |
| `window.globalRoot` | GameRoot | InGameState only | Game logic root |
| `window.globalRoot.hubGoals` | HubGoals | InGameState only | Level goals |
| `window.globalRoot.map` | GameMap | InGameState only | World map + entities |
| `window.globalRoot.camera` | Camera | InGameState only | Camera controller |
| `document.body.id` | string | Always | Current state identifier |

### Control Methods

| Method | Effect | Used for |
|---|---|---|
| `app.stateMgr.moveToState("InGameState", {savegame})` | Enter game | `start_game()` |
| `app.savegameMgr.createNewSavegame(meta)` | Create save slot | New game, savegame injection |
| `app.settings.updateSetting(key, value)` | Change settings | Mute audio, disable tutorials |

### Input Simulation (via JS)

| Action | JS Implementation |
|---|---|
| Select building | `document.dispatchEvent(new KeyboardEvent('keydown', {key: digit}))` |
| Place building | `canvas.dispatchEvent(new MouseEvent('click', {clientX, clientY}))` |
| Delete building | `canvas.dispatchEvent(new MouseEvent('contextmenu', {clientX, clientY}))` |
| Rotate building | `document.dispatchEvent(new KeyboardEvent('keydown', {key: 'r'}))` |
| Pan camera | `document.dispatchEvent(new KeyboardEvent('keydown', {key: wasd}))` |
| Center on Hub | `document.dispatchEvent(new KeyboardEvent('keydown', {key: ' '}))` |

### DOM Elements

| Selector | Purpose | Availability |
|---|---|---|
| `#ingame_Canvas` | Game canvas | InGameState only |
| `document.body` | State detection via `id` attribute | Always |

### State Detection Patterns

```javascript
// Check if in game
var inGame = document.body.id === "state_InGameState";

// Check if playing (not in menu/loading)
var isPlaying = inGame && globalRoot &&
    globalRoot.app.stateMgr.currentState &&
    globalRoot.app.stateMgr.currentState.getIsIngame &&
    globalRoot.app.stateMgr.currentState.getIsIngame();

// Read game progress
var level = globalRoot.hubGoals.level;
var entities = globalRoot.map.entityMap_.size || 0;
```
