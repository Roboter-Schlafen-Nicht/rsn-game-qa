# Platform Architecture: Game-Agnostic Plugin System

> Authored in session 26. Defines the separation between platform
> infrastructure and game-specific plugins, enabling the system to
> support multiple games without per-game engineering.

## Motivation

The RSN Game QA platform currently works with one game (Breakout 71).
Game-specific code is interleaved with platform infrastructure throughout
the codebase: JS modal handling snippets, YOLO class names, observation
vector layout, reward logic, and termination conditions are all embedded
in `Breakout71Env`.

To scale to new games, we need a clean separation:

- **Platform code** handles capture, perception, RL training, oracles,
  reporting, and session orchestration — game-agnostic infrastructure
  that works for any game.
- **Game plugins** provide the game-specific glue: how to detect game
  state, how to translate actions, what constitutes "game over", and
  optionally how to compute game-aware rewards.

The goal: onboarding a new game should take **minutes** (write a config
file and a thin plugin), not **days** (retrain YOLO, write reward logic,
capture datasets, annotate frames).

## Current State: What Is Game-Specific vs Generic

Based on a full codebase analysis (session 26), the codebase is already
~80% game-agnostic. Game-specific code is concentrated in a few files:

### Fully Game-Agnostic (no changes needed)

| Module | Files | Notes |
|---|---|---|
| Capture | `window_capture.py`, `wincam_capture.py` | Generic Windows frame capture |
| Game Loader | `base.py`, `browser_loader.py`, `config.py`, `factory.py` | ABC + factory pattern |
| Oracles | All 12 oracle files | Parameterised via constructor args |
| Orchestrator | `data_collector.py` | Generic frame collector |
| Reporting | `report.py`, `dashboard.py` | Generic JSON/HTML reporting |
| CNN Wrapper | `cnn_wrapper.py` | Works with any env returning `info["frame"]` |

### Breakout-71-Specific (must move to plugin)

| File | Game-Specific Elements |
|---|---|
| `breakout71_env.py` | JS modal snippets, observation vector, reward function, termination logic, `#game` canvas ID, `puckPosition` injection, action mapping |
| `breakout71_loader.py` | Parcel defaults, `.parcel-cache` cleanup |
| `session_runner.py` | Hardcoded `Breakout71Env` import, default config/weights paths |
| `train_rl.py` | Hardcoded env class, checkpoint names, mute JS, config paths |
| `capture_dataset.py` | Duplicated JS modal handling, YOLO class names |

### Mixed (needs refactoring)

| File | Generic Part | Game-Specific Part |
|---|---|---|
| `yolo_detector.py` | Inference engine, device resolution | `BREAKOUT71_CLASSES` default, `detect_to_game_state()` class mapping |
| `input_controller.py` | Key/mouse injection | `ACTION_NOOP/LEFT/RIGHT/FIRE` constants |
| `_smoke_utils.py` | `BrowserInstance`, `Timer` | Default config `"breakout-71"` |

## Target Architecture

### Directory Structure

```
src/
  platform/                     # Game-agnostic platform code
    __init__.py
    base_env.py                 # BaseGameEnv ABC
    rnd_wrapper.py              # RND VecEnv wrapper (Tier 1 reward)
    cnn_wrapper.py              # Moved from src/env/
    game_over_detector.py       # Ensemble game-over detection
  capture/                      # Unchanged
    window_capture.py
    wincam_capture.py
    input_controller.py         # Generic input (remove Breakout constants)
  perception/
    yolo_detector.py            # Generic (class mapping from config)
  oracles/                      # Unchanged (already parameterised)
  orchestrator/
    data_collector.py           # Unchanged
    session_runner.py           # Accepts any BaseGameEnv (no hardcoded import)
  reporting/                    # Unchanged
  game_loader/                  # Unchanged (already has ABC + factory)

games/                          # Game plugins (top-level, not in src/)
  __init__.py
  breakout71/
    __init__.py
    env.py                      # Breakout71Env(BaseGameEnv)
    loader.py                   # Breakout71Loader (moved from src/game_loader/)
    modal_handler.py            # JS snippets for Breakout 71 DOM
    perception.py               # YOLO class names + detection mapping
    config.yaml                 # Game config (moved from configs/games/)
    training.yaml               # YOLO training config (moved from configs/training/)
    reward.py                   # Optional: YOLO-based brick reward
  # future games:
  # tetris/
  #   env.py
  #   ...
  # platformer/
  #   ...

configs/                        # Platform-level configs only
  platform.yaml                 # Default platform settings

scripts/                        # Game-agnostic CLI tools
  train_rl.py                   # --game breakout71, loads plugin dynamically
  run_session.py                # --game breakout71
  capture_dataset.py            # --game breakout71
  # game-specific scripts stay in games/<name>/scripts/ if needed
```

### The Test Horseshoe

The platform wraps around the game in a horseshoe pattern:

```
                         ┌──────────────────────┐
                         │     Game Plugin       │
                         │                       │
                         │  env.py (actions,      │
                         │   obs, reward,         │
                         │   termination,         │
                         │   modal handling)      │
                         │                       │
                         │  perception.py         │
                         │   (YOLO classes,       │
                         │    detection mapping)  │
                         │                       │
                         │  config.yaml           │
                         │   (window title,       │
                         │    loader settings)    │
                         └──────────┬────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         │                          │                          │
   ┌─────▼──────┐          ┌───────▼───────┐          ┌───────▼──────┐
   │   Capture   │          │  Perception   │          │    Input     │
   │             │          │               │          │              │
   │ wincam /    │────────▶ │  YOLO         │────────▶ │  Selenium    │
   │ WindowCap / │          │  (configurable│          │  ActionChains│
   │ screenshot  │          │   classes)    │          │              │
   └─────┬──────┘          └───────┬───────┘          └───────┬──────┘
         │                          │                          │
         └──────────────────────────┼──────────────────────────┘
                                    │
                   ┌────────────────▼────────────────────┐
                   │         Platform Core               │
                   │                                     │
                   │  BaseGameEnv    (lifecycle, capture) │
                   │  RND Wrapper    (exploration reward) │
                   │  CNN Wrapper    (pixel observations) │
                   │  Oracles        (passive detection)  │
                   │  Orchestrator   (session management) │
                   │  Reporting      (findings → reports) │
                   │  Game Loader    (process lifecycle)  │
                   └─────────────────────────────────────┘
```

## The Game Plugin Interface

### BaseGameEnv ABC

```python
class BaseGameEnv(gymnasium.Env, ABC):
    """Platform base class for all game environments.

    Handles the common lifecycle: lazy initialisation, frame capture,
    YOLO detection, oracle wiring, and headless/native mode branching.
    Game plugins implement the abstract methods to provide game-specific
    behaviour.
    """

    # ── Game plugin MUST implement these ──────────────────────────

    @abstractmethod
    def game_classes(self) -> list[str]:
        """YOLO class names for this game.

        Returns
        -------
        list[str]
            e.g. ["paddle", "ball", "brick", "powerup", "wall"]
        """

    @abstractmethod
    def build_observation(
        self, detections: dict, reset: bool = False
    ) -> np.ndarray:
        """Convert raw YOLO detections to the observation vector.

        Parameters
        ----------
        detections : dict
            Grouped detections from YoloDetector.detect_to_game_state().
        reset : bool
            True on first call after reset (initialise baselines).
        """

    @abstractmethod
    def compute_reward(
        self, detections: dict, terminated: bool, info: dict
    ) -> float:
        """Compute the game-specific reward for this step.

        This is the YOLO-based reward (e.g., brick counting for Breakout).
        In survival/RND mode, the platform ignores this and uses a
        fixed survival bonus instead.
        """

    @abstractmethod
    def check_termination(
        self, detections: dict
    ) -> tuple[bool, bool, dict]:
        """Determine if the episode has ended.

        Returns
        -------
        tuple[bool, bool, dict]
            (terminated, truncated, info_updates)
        """

    @abstractmethod
    def handle_modals(self, driver, dismiss_game_over: bool = True) -> str:
        """Detect and handle game-specific modal dialogs.

        Parameters
        ----------
        driver : WebDriver
            Selenium WebDriver instance.
        dismiss_game_over : bool
            Whether to dismiss game-over modals (True during reset,
            False during step to preserve episode boundaries).

        Returns
        -------
        str
            Current game state: "gameplay", "game_over", "perk_picker",
            "menu", or game-specific states.
        """

    @abstractmethod
    def apply_action(
        self, action: np.ndarray, driver, canvas, canvas_size: dict
    ) -> None:
        """Translate the RL action to game input.

        Parameters
        ----------
        action : np.ndarray
            Action from the policy (shape and semantics defined by
            the game plugin's action_space).
        driver : WebDriver
            Selenium WebDriver.
        canvas : WebElement
            The game's canvas/container element.
        canvas_size : dict
            Canvas dimensions {"width": int, "height": int}.
        """

    @abstractmethod
    def start_game(self, driver, canvas) -> None:
        """Game-specific logic to start or restart a game.

        Called during reset() after modals are handled. Typically
        clicks the canvas, presses a key, or triggers a JS function.
        """

    @abstractmethod
    def canvas_selector(self) -> str:
        """CSS selector for the game's main canvas/container element.

        Returns
        -------
        str
            e.g. "#game", "canvas", "#game-container"
        """

    # ── Game plugin MAY override these ────────────────────────────

    def on_reset_complete(self, obs: np.ndarray, info: dict) -> None:
        """Hook called after reset() completes. Optional."""
        pass

    def on_lazy_init(self, driver) -> None:
        """Hook called during lazy initialisation. Optional.

        Use for one-time setup like muting audio, injecting CSS, etc.
        """
        pass

    # ── Platform provides these (not overridden) ──────────────────

    def reset(self, seed=None, options=None):
        """Standard lifecycle."""
        # 1. super().reset(seed=seed)
        # 2. _lazy_init() if needed
        # 3. handle_modals(dismiss_game_over=True)
        # 4. start_game()
        # 5. _capture_frame()
        # 6. _detect_objects()
        # 7. build_observation(reset=True)
        # 8. on_reset_complete()
        # 9. oracle.on_reset()
        ...

    def step(self, action):
        """Standard lifecycle."""
        # 1. apply_action()
        # 2. handle_modals(dismiss_game_over=False) [throttled]
        # 3. _capture_frame()
        # 4. _detect_objects()
        # 5. build_observation()
        # 6. check_termination()
        # 7. compute_reward() or survival_reward() [based on mode]
        # 8. oracle.on_step()
        ...
```

### Game Plugin Registration

Games are discovered via a registry pattern, similar to the existing
`GameLoader` factory:

```python
# games/breakout71/__init__.py
from src.platform.base_env import register_game

@register_game("breakout71")
class Breakout71Plugin:
    env_class = Breakout71Env
    loader_class = Breakout71Loader
    config_path = "games/breakout71/config.yaml"
```

Scripts load games by name:

```bash
# Train RL on Breakout 71
python scripts/train_rl.py --game breakout71 --reward-mode rnd --policy cnn

# Train RL on a future game
python scripts/train_rl.py --game tetris --reward-mode rnd --policy cnn
```

### Game Plugin Minimal Example

To onboard a new game with **zero YOLO training**, using only CNN
observations and survival + RND reward:

```python
# games/simple_platformer/env.py

class SimplePlatformerEnv(BaseGameEnv):
    observation_space = spaces.Box(0, 255, (84, 84, 1), dtype=np.uint8)
    action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)  # x, y movement

    def game_classes(self) -> list[str]:
        return []  # No YOLO — CNN only

    def build_observation(self, detections, reset=False):
        return np.zeros(0)  # Not used in CNN mode

    def compute_reward(self, detections, terminated, info):
        return 0.0  # Not used in RND mode — survival reward from platform

    def check_termination(self, detections):
        # Use the platform's ensemble game-over detector
        game_over = self._game_over_detector.is_game_over(self._last_frame)
        return game_over, self._step_count >= self.max_steps, {}

    def handle_modals(self, driver, dismiss_game_over=True):
        return "gameplay"  # Simple game, no modals

    def apply_action(self, action, driver, canvas, canvas_size):
        # Map 2D action to mouse position
        x = int((action[0] + 1) / 2 * canvas_size["width"])
        y = int((action[1] + 1) / 2 * canvas_size["height"])
        ActionChains(driver).move_to_element_with_offset(
            canvas, x - canvas_size["width"] // 2,
            y - canvas_size["height"] // 2
        ).perform()

    def start_game(self, driver, canvas):
        ActionChains(driver).click(canvas).perform()

    def canvas_selector(self):
        return "canvas"
```

This plugin is ~40 lines. No YOLO model, no training data, no
annotation pipeline. The platform handles everything else.

## Observation Modes

### CNN (Default — Game-Agnostic)

```
Raw frame → CnnObservationWrapper → 84x84 grayscale → VecFrameStack(4) → Policy
```

- No YOLO required
- Works for any game immediately
- The CNN policy learns visual features from reward signal (+ RND)
- Slower to train but universally applicable

### MLP (Optional — Requires Game-Specific YOLO)

```
Raw frame → YoloDetector → game_classes() mapping → build_observation() → Policy
```

- Requires a trained YOLO model with game-specific classes
- Plugin must implement `build_observation()` with meaningful features
- Faster training (pre-extracted features) but high onboarding cost
- Available as `--policy mlp` when the game plugin provides YOLO support

### Decision

CNN is the **default** observation mode. It requires zero game-specific
setup and aligns with the platform's game-agnostic mission. MLP is an
**optional enhancement** for games where a YOLO model is available
(e.g., Breakout 71).

## YOLO in the Platform

`YoloDetector` remains a platform-level service. Game plugins configure
it rather than owning it:

```python
# Platform: YoloDetector
class YoloDetector:
    def __init__(self, weights_path, classes=None, device="auto"):
        self._classes = classes  # From game plugin, or inferred from model

    def detect_to_game_state(self, frame, class_mapping=None):
        """Group detections by class.

        Parameters
        ----------
        class_mapping : dict[int, str] | None
            Override class name mapping. If None, use model's own names.
        """
        ...
```

The game plugin provides class names via `game_classes()`. If the plugin
returns an empty list (no YOLO), the detector is not initialised and
CNN mode is used exclusively.

## Reward Mode: Platform-Level Override

The reward mode is a **platform decision**, not a game plugin decision.
The `--reward-mode` CLI flag applies to any game:

| Mode | Source | Game Plugin Involvement |
|---|---|---|
| `yolo` | `plugin.compute_reward()` | Plugin computes reward from YOLO detections |
| `survival` | Platform fixed formula | Plugin not consulted; +0.01 survival, -5.0 game over |
| `rnd` | Platform survival + RND wrapper | Plugin not consulted; exploration bonus from wrapper |

The platform's `step()` method selects the reward source:

```python
# In BaseGameEnv.step()
if self._reward_mode == "yolo":
    reward = self.compute_reward(detections, terminated, info)
elif self._reward_mode in ("survival", "rnd"):
    reward = 0.01  # survival bonus
    if terminated and not level_cleared:
        reward = -5.0  # game-over penalty
# RND bonus is added externally by the VecEnv wrapper, not here
```

## Migration Plan

The refactoring is a **behaviour-preserving** change. The goal is to
move code, not change logic. All 640+ existing tests must continue to
pass after refactoring.

### Phase 1: Extract BaseGameEnv

1. Create `src/platform/base_env.py` with the ABC
2. Make `Breakout71Env` inherit from `BaseGameEnv`
3. Move generic lifecycle code to `BaseGameEnv`
4. Move Breakout-specific code to the abstract method implementations
5. All tests pass unchanged

### Phase 2: Create games/ directory

1. Create `games/breakout71/` with plugin files
2. Move `breakout71_loader.py` → `games/breakout71/loader.py`
3. Move JS modal snippets → `games/breakout71/modal_handler.py`
4. Move YOLO class constants → `games/breakout71/perception.py`
5. Move config YAML → `games/breakout71/config.yaml`
6. Update imports throughout

### Phase 3: Refactor scripts

1. `train_rl.py` — accept `--game` flag, load plugin dynamically
2. `run_session.py` — same
3. `session_runner.py` — accept any `BaseGameEnv` subclass
4. `capture_dataset.py` — parameterise by game

### Phase 4: Move CnnObservationWrapper

1. Move `src/env/cnn_wrapper.py` → `src/platform/cnn_wrapper.py`
2. Update imports

### Validation

After each phase, run the full test suite to verify no regressions:

```bash
conda run -n yolo pytest tests/ -x -q
```

## What This Enables

Once the plugin architecture is in place:

1. **New game in minutes:** Write a ~40-line plugin, point at the game's
   URL, run `python scripts/train_rl.py --game my_game --reward-mode rnd`
2. **A/B reward strategies:** Compare RND vs YOLO-based reward on the
   same game without code changes (`--reward-mode rnd` vs `--reward-mode yolo`)
3. **Shared improvements:** Better RND, better oracles, better capture —
   all games benefit automatically
4. **Community contributions:** Game plugins are self-contained; third
   parties can contribute new game support without touching platform code
5. **Testing isolation:** Platform tests are game-agnostic; game plugin
   tests are isolated to `games/<name>/tests/`

## Architectural Decisions Record

| Decision | Choice | Rationale |
|---|---|---|
| Plugin location | `games/` (top-level) | Games are not part of the platform package |
| Default observation | CNN (84x84 grayscale) | Game-agnostic, no YOLO required |
| MLP observation | Optional plugin feature | Requires game-specific YOLO model |
| YOLO ownership | Platform with configurable classes | Avoids duplicating inference infrastructure |
| Reward mode | Platform-level override | `--reward-mode` works for any game |
| Implementation order | Architecture first, then RND | Clean foundation before adding features |
| Migration strategy | Behaviour-preserving refactor | All existing tests must pass |

## Source Files (After Implementation)

### Platform (new)
- `src/platform/__init__.py`
- `src/platform/base_env.py` — BaseGameEnv ABC
- `src/platform/rnd_wrapper.py` — RND VecEnv wrapper
- `src/platform/cnn_wrapper.py` — moved from `src/env/`
- `src/platform/game_over_detector.py` — ensemble detector

### Game Plugin (Breakout 71)
- `games/__init__.py`
- `games/breakout71/__init__.py` — plugin registration
- `games/breakout71/env.py` — Breakout71Env(BaseGameEnv)
- `games/breakout71/loader.py` — moved from `src/game_loader/`
- `games/breakout71/modal_handler.py` — JS snippets
- `games/breakout71/perception.py` — YOLO classes + mapping
- `games/breakout71/reward.py` — YOLO brick-counting reward
- `games/breakout71/config.yaml` — game config
- `games/breakout71/training.yaml` — YOLO training config

### Refactored (existing files, updated)
- `src/perception/yolo_detector.py` — remove `BREAKOUT71_CLASSES` default
- `src/orchestrator/session_runner.py` — accept any `BaseGameEnv`
- `scripts/train_rl.py` — `--game` flag, dynamic plugin loading
- `scripts/run_session.py` — `--game` flag
