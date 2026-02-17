# Oracle System Spec

> Extracted from session 1 notes. This is the reference
> design for the bug-detection oracle system.

## Architecture

Two-layer design:

1. **RL agent = data generator** — creates rich play traces (trajectories)
2. **Bug detectors ("oracles") = anomaly detectors** — inspect those traces
   and flag abnormal patterns

The RL agent does NOT determine what is a bug. It maximises coverage and variety.
Oracles turn trajectories into actionable findings.

This pattern follows frameworks like Wuji: one oracle checks "is app alive?",
another checks "is state changing?", etc.

## Base Oracle ABC

```python
from abc import ABC, abstractmethod

class Oracle(ABC):
    @abstractmethod
    def on_reset(self, env, obs):
        """Called at the beginning of each episode."""
        ...

    @abstractmethod
    def on_step(self, env, obs, reward, done, info):
        """Called after each step. Can append findings."""
        ...

    @abstractmethod
    def get_findings(self):
        """Return a list of finding dicts for this episode."""
        ...
```

## Finding Data Structure

Each finding is a dict with consistent fields:

```python
{
    "type": str,        # e.g. "crash", "stuck", "score_anomaly", "visual_glitch", "perf_frame_time"
    "severity": str,    # "high", "medium", or "low"
    "message": str,     # human-readable description
    # ... additional type-specific fields
}
```

## Environment Integration

### `__init__`

```python
def __init__(self, ..., oracles=None):
    self.oracles = oracles or []
    self._episode_findings = []
```

### `reset()`

```python
self._episode_findings = []
for oracle in self.oracles:
    oracle.on_reset(self, obs)
```

### `step()`

```python
for oracle in self.oracles:
    oracle.on_step(self, obs, reward, terminated or truncated, info)
```

### `get_episode_findings()`

```python
def get_episode_findings(self):
    all_findings = []
    for oracle in self.oracles:
        all_findings.extend(oracle.get_findings())
    return all_findings
```

## Usage Example

```python
env = Breakout71Env(
    render_mode="rgb_array",
    oracles=[
        CrashOracle(process_name="chrome.exe"),
        StuckOracle(max_stuck_steps=120),
        ScoreAnomalyOracle(),
        # VisualGlitchOracle(model=your_cnn),
    ],
)

obs, info = env.reset()
done = truncated = False

while not (done or truncated):
    action = policy(obs)
    obs, reward, done, truncated, info = env.step(action)

findings = env.get_episode_findings()
```

In CI: fail the job if any `severity == "high"` finding exists.

---

## CrashOracle

**Purpose:** Detect game crashes, process death, and window disappearance.

**Detection strategy:**
- Monitor game process liveness via `psutil`
- Monitor window handle validity via `win32gui.IsWindow`
- If either check fails mid-episode, flag as crash

**Parameters:**
- `process_name` (str): Game process name (e.g. `"chrome.exe"`)
- `hwnd` (int): Window handle (alternative to process name)

**Finding:**
```python
{"type": "crash", "severity": "high", "message": "Game process/window died during episode."}
```

**Reference implementation:**

```python
import psutil

class CrashOracle(Oracle):
    def __init__(self, process_name=None, hwnd=None):
        self.process_name = process_name
        self.hwnd = hwnd
        self._crashed = False

    def on_reset(self, env, obs):
        self._crashed = False

    def on_step(self, env, obs, reward, done, info):
        if self._crashed:
            return
        alive = True
        if self.process_name:
            alive = any(p.name() == self.process_name for p in psutil.process_iter())
        elif self.hwnd:
            alive = bool(win32gui.IsWindow(env.hwnd))
        if not alive:
            self._crashed = True

    def get_findings(self):
        if self._crashed:
            return [{"type": "crash", "severity": "high",
                     "message": "Game process/window died during episode."}]
        return []
```

---

## StuckOracle

**Purpose:** Detect when the agent/game is stuck with no progress.

**Detection strategy:**
- Track paddle_x, ball_x, ball_y positions (obs indices 0-2)
- If all positions change less than `pos_tol` for `max_stuck_steps`
  consecutive steps, flag as stuck

**Parameters:**
- `max_stuck_steps` (int): Steps before declaring stuck. Default: 120
- `pos_tol` (float): Position change tolerance. Default: 0.01

**Finding:**
```python
{"type": "stuck", "severity": "medium",
 "message": f"Agent/game state unchanged for {max_stuck_steps} steps."}
```

**Reference implementation:**

```python
class StuckOracle(Oracle):
    def __init__(self, max_stuck_steps=120, pos_tol=0.01):
        self.max_stuck_steps = max_stuck_steps
        self.pos_tol = pos_tol
        self._last_obs = None
        self._stuck_counter = 0
        self._stuck = False

    def on_reset(self, env, obs):
        self._last_obs = obs.copy()
        self._stuck_counter = 0
        self._stuck = False

    def on_step(self, env, obs, reward, done, info):
        if self._stuck:
            return
        cur = obs[:3]   # paddle_x, ball_x, ball_y
        prev = self._last_obs[:3]
        if np.all(np.abs(cur - prev) < self.pos_tol):
            self._stuck_counter += 1
        else:
            self._stuck_counter = 0
        self._last_obs = obs.copy()
        if self._stuck_counter >= self.max_stuck_steps:
            self._stuck = True

    def get_findings(self):
        if self._stuck:
            return [{"type": "stuck", "severity": "medium",
                     "message": f"Agent/game state unchanged for "
                                f"{self.max_stuck_steps} steps."}]
        return []
```

---

## ScoreAnomalyOracle

**Purpose:** Detect score increases that occur without brick destruction.

**Detection strategy:**
- Monitor `env.current_score` and `bricks_norm` (obs index 5)
- If score increases but no bricks were destroyed (bricks_delta <= 0),
  flag as anomaly
- Future: LSTM/autoencoder over play metrics, or check ball-brick contact

**Parameters:**
- `suspicious_ratio` (float): Reserved for future threshold. Default: 5.0

**Finding:**
```python
{"type": "score_anomaly", "severity": "medium",
 "message": f"Score increased by {score_delta} with no bricks destroyed."}
```

**Reference implementation:**

```python
class ScoreAnomalyOracle(Oracle):
    def __init__(self, suspicious_ratio=5.0):
        self.suspicious_ratio = suspicious_ratio
        self._findings = []
        self._prev_score = 0
        self._prev_bricks_norm = None

    def on_reset(self, env, obs):
        self._findings = []
        self._prev_score = getattr(env, 'current_score', 0)
        self._prev_bricks_norm = obs[5]

    def on_step(self, env, obs, reward, done, info):
        score = getattr(env, 'current_score', 0)
        bricks_norm = obs[5]
        score_delta = score - self._prev_score
        bricks_delta = self._prev_bricks_norm - bricks_norm

        if score_delta > 0 and bricks_delta <= 0:
            self._findings.append({
                "type": "score_anomaly", "severity": "medium",
                "message": f"Score increased by {score_delta} with no "
                           f"bricks destroyed."
            })

        self._prev_score = score
        self._prev_bricks_norm = bricks_norm

    def get_findings(self):
        return self._findings
```

---

## VisualGlitchOracle

**Purpose:** Detect rendering artifacts (missing textures, stretched sprites,
placeholder magenta fills, checkerboard patterns).

**Detection strategy:**
- CNN classifier (ResNet-18/50) trained on "normal" vs "glitched" frames
- Input: full game frame, optionally cropped/resized
- Output: dict of class probabilities `{"ok": 0.8, "missing_tex": 0.1, ...}`
- Flag if `label != "ok"` and `confidence >= threshold`
- EA SEED approach: full-image classification, ~86-87% accuracy, ~88% glitch
  recall, ~6-7% false positive rate

**Parameters:**
- `model`: CNN model with a `.predict(frame)` method
- `threshold` (float): Confidence threshold. Default: 0.9

**Finding:**
```python
{"type": "visual_glitch", "severity": "high",
 "label": label, "confidence": float(conf), "step": step_idx}
```

**Reference implementation:**

```python
class VisualGlitchOracle(Oracle):
    def __init__(self, model, threshold=0.9):
        self.model = model
        self.threshold = threshold
        self._findings = []
        self._step_idx = 0

    def on_reset(self, env, obs):
        self._findings = []
        self._step_idx = 0

    def on_step(self, env, obs, reward, done, info):
        frame = env._last_frame
        if frame is None:
            self._step_idx += 1
            return
        probs = self.model.predict(frame)
        label = max(probs, key=probs.get)
        conf = probs[label]
        if label != "ok" and conf >= self.threshold:
            self._findings.append({
                "type": "visual_glitch", "severity": "high",
                "label": label, "confidence": float(conf),
                "step": self._step_idx,
            })
        self._step_idx += 1

    def get_findings(self):
        return self._findings
```

**Training approach:** Synthetic data generation — inject visual bugs into
Breakout 71 JS code (missing textures, stretched sprites, color corruption),
capture frames, train small CNN classifier.

---

## PerformanceOracle

**Purpose:** Monitor FPS, frame timing, CPU usage, and memory consumption.

**Detection strategy:**
- `time.perf_counter()` measures wall-clock step duration
- `psutil` monitors per-process CPU% and RSS memory
- Flag if avg step time, p99 step time, CPU, or memory increase exceed
  thresholds

**Environment-side setup** (in `step()`):

```python
t0 = time.perf_counter()
# ... apply action, capture, compute obs/reward ...
step_duration = time.perf_counter() - t0

info["perf"] = {
    "step_duration_s": step_duration,
    "fps": 1.0 / step_duration if step_duration > 0 else 0,
    "cpu_sys_percent": psutil.cpu_percent(interval=None),
    "mem_sys_percent": psutil.virtual_memory().percent,
    "cpu_proc_percent": process.cpu_percent(interval=None),
    "mem_proc_bytes": process.memory_info().rss,
}
```

**Parameters:**
- `max_avg_step_ms` (float): Max average step time in ms. Default: 40.0 (~25 FPS)
- `max_p99_step_ms` (float): Max p99 step time. Default: 80.0
- `max_cpu_proc_percent` (float): Max process CPU%. Default: 90.0
- `max_mem_proc_mb_increase` (float): Max memory increase in MB. Default: 500.0

**Findings:**
```python
{"type": "perf_frame_time", "severity": "medium",
 "message": f"Step times high: avg={avg_ms:.1f} ms, p99={p99_ms:.1f} ms"}

{"type": "perf_cpu", "severity": "medium",
 "message": f"Game process CPU peaked at {max_cpu:.1f}%"}

{"type": "perf_memory_leak", "severity": "medium",
 "message": f"Game memory increased by {mem_inc:.1f} MB during episode"}
```

**Reference implementation:**

```python
class PerformanceOracle(Oracle):
    def __init__(self, max_avg_step_ms=40.0, max_p99_step_ms=80.0,
                 max_cpu_proc_percent=90.0, max_mem_proc_mb_increase=500.0):
        self.max_avg_step_ms = max_avg_step_ms
        self.max_p99_step_ms = max_p99_step_ms
        self.max_cpu_proc_percent = max_cpu_proc_percent
        self.max_mem_proc_mb_increase = max_mem_proc_mb_increase
        self._step_times = []
        self._cpu_proc = []
        self._mem_proc = []
        self._mem_start = None
        self._findings = []

    def on_reset(self, env, obs):
        self._step_times = []
        self._cpu_proc = []
        self._mem_proc = []
        self._mem_start = None
        self._findings = []

    def on_step(self, env, obs, reward, done, info):
        perf = info.get("perf", {})
        dt = perf.get("step_duration_s")
        cpu_p = perf.get("cpu_proc_percent")
        mem_b = perf.get("mem_proc_bytes")

        if dt is not None:
            self._step_times.append(dt * 1000.0)
        if cpu_p is not None:
            self._cpu_proc.append(cpu_p)
        if mem_b is not None:
            mb = mem_b / (1024 * 1024)
            if self._mem_start is None:
                self._mem_start = mb
            self._mem_proc.append(mb)

    def get_findings(self):
        if not self._step_times:
            return []
        avg_ms = float(np.mean(self._step_times))
        p99_ms = float(np.percentile(self._step_times, 99))
        findings = []

        if avg_ms > self.max_avg_step_ms or p99_ms > self.max_p99_step_ms:
            findings.append({"type": "perf_frame_time", "severity": "medium",
                "message": f"Step times high: avg={avg_ms:.1f} ms, p99={p99_ms:.1f} ms"})

        if self._cpu_proc and max(self._cpu_proc) > self.max_cpu_proc_percent:
            findings.append({"type": "perf_cpu", "severity": "medium",
                "message": f"Game process CPU peaked at {max(self._cpu_proc):.1f}%"})

        if self._mem_start is not None and self._mem_proc:
            mem_inc = max(self._mem_proc) - self._mem_start
            if mem_inc > self.max_mem_proc_mb_increase:
                findings.append({"type": "perf_memory_leak", "severity": "medium",
                    "message": f"Game memory increased by {mem_inc:.1f} MB during episode"})

        return findings
```

---

## Bug Detection Scenarios for Breakout 71

| Bug Type              | Detection Method                                                 |
|-----------------------|------------------------------------------------------------------|
| Physics/collision     | Ball passes through paddle/brick without bouncing — compare expected vs observed ball direction |
| Score anomaly         | Score increases without brick collision — cross-check score delta with bricks_norm delta |
| Visual glitches       | Missing sprites, broken tiles, stretched textures — CNN classifier |
| Frame skipping/timing | Ball "teleports" (huge delta) without matching time passage — step_duration spike detection |
| Crash                 | Process dies / window disappears — psutil + win32gui monitoring |
| Stuck/freeze          | No position changes for N steps — observation comparison |

## Design Principles

1. **Composable:** Small, independent detectors. Start simple, add more later.
2. **RL is the driver, not the judge:** RL maximises coverage; oracles detect bugs.
3. **Reward is "bug-friendly":** Curiosity reward for novel states, stress
   reward for aggressive play. Oracles catch what breaks under stress.
4. **Thresholds are per-game:** The pattern (sample in step, feed to oracle,
   aggregate findings) is game-agnostic; thresholds are tuned per game.
5. **Visual oracle is pluggable:** Stub with `model.predict(frame) -> dict`,
   replace with real CNN later.

## Source Files

- `src/oracles/base.py` — `Oracle` ABC + `Finding` dataclass
- `src/oracles/crash.py` — `CrashOracle`
- `src/oracles/stuck.py` — `StuckOracle`
- `src/oracles/score_anomaly.py` — `ScoreAnomalyOracle`
- `src/oracles/visual_glitch.py` — `VisualGlitchOracle`
- `src/oracles/performance.py` — `PerformanceOracle`
