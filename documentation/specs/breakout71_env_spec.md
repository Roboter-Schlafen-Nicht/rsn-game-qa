# Breakout71 Gymnasium Environment Spec

> Extracted from `documentation/sessions/session1.md`. This is the reference
> design for the `Breakout71Env` Gymnasium environment.

## Overview

`Breakout71Env` wraps the [Breakout 71](https://github.com/lecarore/breakout71)
browser game running in a native Windows window. It captures frames via
BitBlt/GDI, runs YOLOv8 inference to extract game-object positions, converts
them to a structured observation vector, and injects actions via pydirectinput.

## Observation Space

**Type:** `gymnasium.spaces.Box` (continuous, float32)

**Layout:** 6-element vector

| Index | Name          | Range      | Description                                    |
|-------|---------------|------------|------------------------------------------------|
| 0     | `paddle_x`    | [0.0, 1.0] | Normalised x-position of paddle centre        |
| 1     | `ball_x`      | [0.0, 1.0] | Normalised x-position of ball centre          |
| 2     | `ball_y`      | [0.0, 1.0] | Normalised y-position of ball centre          |
| 3     | `ball_vx`     | [-1.0, 1.0]| Estimated horizontal velocity (frame delta)   |
| 4     | `ball_vy`     | [-1.0, 1.0]| Estimated vertical velocity (frame delta)     |
| 5     | `bricks_norm` | [0.0, 1.0] | Fraction of bricks remaining (count/initial)  |

**Normalisation rules:**
- Positions: `x_center / frame_width`, `y_center / frame_height`
- Velocity: `delta_x = x_t - x_{t-1}`, clipped to [-1, 1]
- Bricks: `bricks_left / bricks_total` (set on reset)

**Missing detections:** Default to 0.5 for positions, 0.0 for velocities.

> **Note:** An alternative "Option B" is discussed for the future: downsampled
> 84x84 grayscale image + YOLO features as extra channels. The vector approach
> (Option A) is the implemented design.

## Action Space

**Type:** `gymnasium.spaces.Discrete(3)`

| Value | Action     | Implementation            |
|-------|------------|---------------------------|
| 0     | No-op      | Do nothing                |
| 1     | Move left  | `pydirectinput.keyDown('left')` / `keyUp('left')` |
| 2     | Move right | `pydirectinput.keyDown('right')` / `keyUp('right')` |

There is no FIRE action in the action space. `Space` is only used during
`reset()` to start a new game.

**Smoother motion variant:** Hold key for ~30ms instead of instant tap:

```python
def _apply_action(self, action: int):
    self._focus_window()
    key = None
    if action == 1:
        key = 'left'
    elif action == 2:
        key = 'right'
    if key is not None:
        pydirectinput.keyDown(key)
        time.sleep(0.03)  # ~1-2 frames of input
        pydirectinput.keyUp(key)
```

## Reward Function

| Signal                | Value    | Condition                                |
|-----------------------|----------|------------------------------------------|
| Brick destroyed       | +scaled  | `(prev_bricks_norm - bricks_norm) * 10.0` |
| Time penalty          | -0.01    | Every step                               |
| Life lost / game over | -1.0     | `_is_life_lost()` returns True           |

**Future enhancement:** Curiosity bonus — maintain a state-visit table over
coarse (paddle, ball) positions or brick configurations. Give bonus reward for
visiting rarely-seen states (novelty).

## Episode Termination

| Condition          | Flag           | Trigger                                      |
|--------------------|----------------|----------------------------------------------|
| Life lost          | `terminated`   | Ball below paddle region / game-over screen   |
| Max steps reached  | `truncated`    | `_step_count >= _max_steps`                  |

**Life-loss detection options:**
1. Ball y-position below a threshold for N consecutive frames
2. "Game Over" text detected via OCR / template matching

## YOLO Classes

| Class    | Description                   |
|----------|-------------------------------|
| `ball`   | The ball                      |
| `paddle` | The player paddle             |
| `brick`  | All bricks (single class)     |
| `wall`   | Optional: wall/ceiling bounds |

**Training data:** 200-500 diverse frames with bounding box labels.
Augmentation: brightness, slight scaling. Labeling tools: CVAT, LabelImg, or
Roboflow. Model: YOLOv8 small variant.

## Observation Extraction (`_extract_observation`)

```python
def _extract_observation(self, frame, reset=False):
    detections = self.yolo_model(frame)

    paddle_x = self._find_paddle_x(detections, frame.shape)
    ball_x, ball_y = self._find_ball_xy(detections, frame.shape)
    bricks_left = self._count_bricks(detections)

    # Velocity from frame delta
    if reset or self.prev_ball_pos is None or ball_x is None:
        ball_vx, ball_vy = 0.0, 0.0
    else:
        ball_vx = ball_x - self.prev_ball_pos[0]
        ball_vy = ball_y - self.prev_ball_pos[1]
    self.prev_ball_pos = (ball_x, ball_y)

    # Brick normalisation
    if self.bricks_total is None or reset:
        self.bricks_total = max(bricks_left, 1)
    bricks_norm = bricks_left / self.bricks_total

    # Handle missing detections
    paddle_x = 0.5 if paddle_x is None else paddle_x
    ball_x = 0.5 if ball_x is None else ball_x
    ball_y = 0.5 if ball_y is None else ball_y

    obs = np.array([
        paddle_x,
        ball_x, ball_y,
        np.clip(ball_vx, -1, 1),
        np.clip(ball_vy, -1, 1),
        np.clip(bricks_norm, 0, 1),
    ], dtype=np.float32)

    return obs
```

## Environment Lifecycle

### `reset()`

1. Ensure browser + game running (`_ensure_game_ready()`)
2. Start new game (`pydirectinput.press('space')`)
3. Capture first frame (`_capture_frame()`)
4. Extract observation with `reset=True`
5. Clear oracle state and episode findings
6. Return `(obs, info)`

### `step(action)`

1. Apply action via `_apply_action(action)`
2. Wait fixed interval (`time.sleep(1.0 / 30.0)`)
3. Capture new frame
4. Build observation from YOLO detections
5. Compute reward and termination
6. Run all attached oracles
7. Return `(obs, reward, terminated, truncated, info)`

### `render()`

Returns `self._last_frame` when `render_mode="rgb_array"`, else `None`.

## Frame Capture (`_capture_frame`)

Uses Win32 GDI BitBlt to capture the game window's client area. See
[`capture_and_input_spec.md`](capture_and_input_spec.md) for the full
implementation.

**Game region cropping:** If the browser UI is visible in the capture, measure
the game canvas sub-rectangle manually and crop:

```python
img = img[y0:y1, x0:x1]  # hardcoded offsets after manual inspection
```

## Constructor Parameters

```python
class Breakout71Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, window_title="Breakout 71"):
        # observation_space: Box(low=[0,0,0,-1,-1,0], high=[1,1,1,1,1,1])
        # action_space: Discrete(3)
        # Internal: prev_ball_pos, bricks_total, window_info, oracles list
```

## RL Training

- **Framework:** stable-baselines3 (DQN or PPO)
- **Hardware:** Intel Arc A770 GPU via XPU backend
- **Hyperparameters:** Start with defaults, tune reward scaling, action
  frequency, and episode length
- **Success criteria:** Clear improvement over random baseline in reward and
  episode length

## Source File

`src/env/breakout71_env.py`
