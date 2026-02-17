# Reward Strategy: Exploration-Driven QA

> Authored in session 26. Defines the three-tier reward architecture for
> game-agnostic QA exploration. This spec replaces the v2 reward roadmap
> in `breakout71_env_spec.md`.

## Why QA Testers Are Not Game Players

A game player maximises score. A QA tester maximises **coverage** of game
states. These are fundamentally different objectives, and the reward
function must reflect this.

A score-maximising RL agent will:

- Learn one dominant strategy and repeat it every episode
- Avoid risky or unusual actions (they lower expected reward)
- Never pick "bad" perks (even though those code paths need testing)
- Never explore game states that don't lead to high scores
- Be a poor QA tester

An exploration-maximising RL agent will:

- Visit many different game states across episodes
- Try different perk combinations, even suboptimal ones
- Play in unexpected ways — exactly how real users find bugs
- Sometimes die quickly, sometimes survive a long time
- Exercise edge cases that a score-optimal policy would never reach

The platform's 12 oracles are passive observers. They can only detect bugs
in states the agent actually visits. An agent that always plays the same
optimal strategy will never trigger the oracles on the vast majority of
game states where bugs are most likely to hide.

**The agent's job is to generate rich, diverse play traces.
The oracles' job is to inspect those traces for anomalies.**

## Game-Agnostic Reward Requirements

As a QA platform, we must support new games without rewriting reward
functions. The current YOLO-based brick-counting reward requires:

1. A trained YOLO model for the specific game
2. Game-specific detection-to-reward mapping logic
3. Knowledge of what constitutes "progress" in that game

This does not scale. For a new game, the onboarding cost is prohibitive:
capture frames, annotate objects, train YOLO, write reward logic, tune
hyperparameters — all before the agent takes its first step.

The reward function must work with **minimal game-specific knowledge**.
The only truly universal signal across all games is:

- **The game is still running** (survival)
- **The current state is different from previously seen states** (novelty)
- **The game ended** (game-over detection)

Everything else — score, lives, level progression, power-ups — is
game-specific and should be treated as optional enrichment, not a
requirement.

## Three-Tier Reward Architecture

### Tier 1: Survival + Novelty (RND) — Current Priority

**Reward formula:**

```
reward = survival_bonus + intrinsic_novelty_bonus
       = 0.01 + normalized_rnd_prediction_error
```

**Game knowledge required:** Game-over detection only.

**How RND (Random Network Distillation) works:**

RND (Burda et al., 2018) provides an intrinsic motivation signal without
any game-specific knowledge:

1. A **target network** is randomly initialised and **never trained**.
   It maps observations to a 512-dimensional embedding.
2. A **predictor network** (same architecture + deeper head) is trained
   to match the target network's output via MSE loss.
3. For frequently-visited states, the predictor learns to match the
   target accurately — low prediction error, low novelty bonus.
4. For rarely-visited states, the predictor has high error — high
   novelty bonus, encouraging the agent to revisit.

The novelty bonus decays naturally as the agent explores: states that
were once novel become familiar, pushing the agent toward unexplored
regions of the state space.

#### RND Network Architecture

**Target network** (fixed, randomly initialised):

| Layer | Type | Details |
|---|---|---|
| Conv1 | Conv2d | 32 filters, 8x8, stride 4, LeakyReLU |
| Conv2 | Conv2d | 64 filters, 4x4, stride 2, LeakyReLU |
| Conv3 | Conv2d | 64 filters, 3x3, stride 1, LeakyReLU |
| Flatten | — | — |
| FC | Linear | → 512 output |

**Predictor network** (trainable):

Same convolutional backbone as target, plus:

| Layer | Type | Details |
|---|---|---|
| FC1 | Linear | 512 → 512, ReLU |
| FC2 | Linear | 512 → 512, ReLU |
| FC3 | Linear | 512 → 512 output |

The deeper head gives the predictor more capacity, but it still cannot
perfectly match the target on novel states because the target's random
weights create an unpredictable mapping.

**Input:** Single frame (not stacked), normalised by running mean/std,
clipped to [-5, 5]. RND operates on individual observations, not
temporal sequences — the frame stack is only for the policy network.

#### Intrinsic Reward Computation

```python
# Computed per step
target_embedding = stop_gradient(target_network(normalize(obs)))
predicted_embedding = predictor_network(normalize(obs))
intrinsic_reward = mse_loss(predicted_embedding, target_embedding)
```

#### Intrinsic Reward Normalisation (Critical)

Raw intrinsic rewards can be orders of magnitude larger than extrinsic
rewards. Without normalisation, intrinsic rewards dominate and training
becomes unstable.

1. Apply a **reward forward filter** (discounted running sum with gamma)
   to the stream of intrinsic rewards
2. Divide by `sqrt(running_variance)` — **no mean subtraction**, only
   standard deviation scaling
3. The result is a normalised bonus with roughly unit variance

**Important:** Intrinsic rewards are **non-episodic**. Do not zero them
out on episode boundaries. The agent should remain curious about novel
states regardless of episode structure.

#### SB3 Integration: VecEnv Wrapper

The simplest integration with Stable Baselines 3:

```python
class RNDRewardWrapper(VecEnvWrapper):
    """Add RND intrinsic reward bonus to environment rewards."""

    def __init__(self, venv, int_coeff=1.0, ext_coeff=2.0, ...):
        self.target_network = RNDTargetNetwork(obs_shape)  # fixed
        self.predictor_network = RNDPredictorNetwork(obs_shape)  # trainable
        self.reward_normalizer = RunningMeanStd()
        self.obs_normalizer = RunningMeanStd()

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        intrinsic = self._compute_intrinsic_reward(obs)
        normalized_intrinsic = self._normalize_reward(intrinsic)
        combined = self.ext_coeff * reward + self.int_coeff * normalized_intrinsic
        self._update_predictor(obs)  # train predictor on this batch
        return obs, combined, done, info
```

This approach uses SB3's single value head with combined rewards. A
faithful RND implementation would use dual value heads (separate
extrinsic and intrinsic advantage estimates), but the single-head
wrapper is sufficient for initial validation. Upgrade to dual heads
only if training shows instability.

#### Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `int_coeff` | 1.0 | Intrinsic reward weight |
| `ext_coeff` | 2.0 | Extrinsic reward weight |
| `gamma_int` | 0.99 | Intrinsic discount factor (non-episodic) |
| `gamma_ext` | 0.999 | Extrinsic discount factor |
| `predictor_lr` | 1e-3 | Predictor network learning rate |
| `embedding_dim` | 512 | RND embedding dimension |
| `update_proportion` | 0.25 | Fraction of experience for predictor updates |

#### CNN vs MLP Considerations

- **CNN (84x84 grayscale, 4-frame stack):** RND is most valuable here.
  Pixel observations have rich visual structure — different brick layouts,
  ball positions, perk effects, modal states all produce distinct
  embeddings. RND naturally rewards visiting visually distinct states.
- **MLP (8-dim YOLO vector):** RND offers limited benefit. The state
  space is compact (8 floats) and PPO can explore it adequately without
  intrinsic motivation. The 8-dim space may not have enough structure
  for RND prediction errors to be meaningful.

**Decision:** CNN is the default observation mode for game-agnostic
operation. MLP is an optional enhancement when a game-specific YOLO
model is available (see `platform_architecture_spec.md`).

#### What Changes in the Env

- `_compute_reward()` gains a `survival` mode: only returns the small
  survival bonus (+0.01) and terminal penalty (-5.0 on game over).
  No brick counting, no YOLO dependency for reward.
- The RND bonus is added **externally** by the VecEnv wrapper, not
  inside the env. This keeps the env clean and the RND logic reusable.
- CLI flag `--reward-mode yolo|survival|rnd` selects the mode:
  - `yolo`: current brick-counting reward (requires YOLO, game-specific)
  - `survival`: survival bonus + terminal penalty only (game-agnostic)
  - `rnd`: survival reward + RND wrapper (game-agnostic, exploration)

#### What Stays Unchanged

- Observation spaces (CNN 84x84, MLP 8-dim)
- Action space (Box(-1, 1))
- Episode lifecycle (reset, step, close)
- Oracle system (passive, decoupled from reward)
- Capture pipeline (wincam, WindowCapture, headless)
- Modal handling (game-specific, in game plugin — see architecture spec)

### Tier 2: Score-Aware Exploration (OCR)

**Reward formula:**

```
reward = survival_bonus + novelty_bonus + score_delta_bonus
```

**Game knowledge required:** A visible score on screen (most games).

**Implementation:**

1. Define a score region in the game frame (configurable per game, or
   auto-detected via OCR scanning)
2. Run OCR (Tesseract or EasyOCR) on the score region each frame
3. Compute `score_delta = current_score - previous_score`
4. Add `score_delta * scale_factor` to the reward

**Why this helps:** Pure survival + novelty can lead to aimless
wandering. The score signal provides a gentle nudge toward "making
progress" — which helps the agent reach later game states (level 2,
level 3) where different bugs may hide. But it's secondary to
exploration: the agent should not sacrifice novelty for score.

**Game-agnostic properties:** Most games display a visible score.
OCR is a universal technique. The only game-specific element is the
score region location, which can be provided via game config or
auto-detected.

**Timing:** Implement after Tier 1 is validated. Score OCR adds
latency (~50-100ms per frame with Tesseract) and complexity.

### Tier 3: Oracle-Guided Exploration (Future Research)

**Reward formula:**

```
reward = survival_bonus + novelty_bonus + oracle_proximity_bonus
```

**Game knowledge required:** None beyond Tier 1.

**Concept:** When an oracle detects an anomaly (visual glitch, physics
violation, stuck state), boost the exploration reward for states
*near* that anomaly. The agent learns to revisit suspicious areas,
essentially performing **directed fuzzing**.

**Implementation sketch:**

1. Oracle findings include a state fingerprint (frame hash or
   observation vector at the time of detection)
2. Maintain a buffer of recent findings with their state fingerprints
3. For each new observation, compute distance to the nearest finding
4. If close to a known anomaly, add a proximity bonus to the reward
5. The agent is drawn back to anomalous regions, increasing the
   chance of reproducing bugs

**Open questions:**

- How to define "proximity" in observation space (pixel space? latent
  space? RND embedding space?)
- How to decay the proximity bonus (avoid the agent getting stuck in
  a loop around one anomaly)
- How to balance directed exploration with undirected novelty-seeking

**Timing:** Research phase. Implement after Tiers 1 and 2 are proven.

## Game-Over Detection Generalisation

Game-over detection is the **only** game-specific knowledge required
for Tier 1. Currently, Breakout 71 uses DOM modal inspection
(`document.body.classList.contains('has-alert-open')`). For unknown
games, we need generic detection strategies.

### Detection Strategies

| Strategy | How It Works | Strengths | Weaknesses |
|---|---|---|---|
| **Screen freeze** | Pixel diff < threshold for N frames | Works for any game | False positive on pause menus, cutscenes |
| **OCR text matching** | Scan for "Game Over", "You Died", "Continue?" | High confidence when matched | Language-dependent, font-dependent |
| **Entropy collapse** | Frame entropy drops below threshold (static screen) | Detects modal overlays | False positive on loading screens |
| **Input responsiveness** | Send actions, check if state changes | Detects stuck/frozen states | Slow (requires multiple frames) |
| **Audio silence** | Game audio stops (if available) | Independent of visual state | Not all games have audio cues |
| **Frame classification** | Train a small CNN to classify "playing" vs "not playing" | Accurate once trained | Requires labelled data per game |

### Recommended Approach: Ensemble Detector

No single strategy is reliable enough alone. An ensemble detector
combines multiple signals with configurable weights:

```python
class GameOverDetector:
    """Ensemble game-over detector with configurable strategies."""

    def __init__(self, strategies: list[GameOverStrategy], threshold: float = 0.6):
        self.strategies = strategies
        self.threshold = threshold

    def is_game_over(self, frame, prev_frame, ...) -> tuple[bool, float]:
        """Return (is_game_over, confidence) based on weighted vote."""
        ...
```

Each game config can specify which strategies to use and their weights.
For Breakout 71, the DOM modal check provides a high-confidence signal
and can be the sole strategy. For unknown games, the ensemble provides
robustness.

**Timing:** Implement incrementally. Start with screen freeze + OCR
text matching as the first generic strategies. Add others as needed
when onboarding new games.

## Comparison: Current vs Tier 1

| Aspect | Current (YOLO + brick reward) | Tier 1 (survival + RND) |
|---|---|---|
| Game knowledge | YOLO model + brick counting | Game-over detection only |
| Scales to new games | No (retrain YOLO, rewrite reward) | Yes (only need game-over signal) |
| Agent behaviour | Score-maximising (repetitive) | Exploration-maximising (diverse) |
| Training speed | Faster (dense reward signal) | Slower (sparse extrinsic + intrinsic) |
| QA value | Low (one strategy, misses edge cases) | High (diverse states, exercises perks) |
| Observation mode | MLP (requires YOLO) or CNN | CNN (game-agnostic, no YOLO needed) |
| Onboarding a new game | Days (data collection, annotation, training) | Minutes (config + game-over detector) |

## References

- Burda et al., "Exploration by Random Network Distillation", ICLR 2019
- Laskin et al., "CURL: Contrastive Unsupervised Representations for RL", ICML 2020
- Laskin et al., "Reinforcement Learning with Augmented Data (RAD)", NeurIPS 2020
- Yarats et al., "Mastering Visual Continuous Control (DrQ-v2)", 2021
- Schwarzer et al., "Self-Predictive Representations (SPR)", ICLR 2021

## Source Files (After Implementation)

- `src/platform/rnd_wrapper.py` — RND VecEnv wrapper
- `src/platform/base_env.py` — BaseGameEnv with survival reward mode
- `games/breakout71/reward.py` — Breakout-specific YOLO reward (optional)
- `scripts/train_rl.py` — `--reward-mode` CLI flag
