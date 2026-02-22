# RSN Game QA — Roadmap

Six-phase plan for delivering the platform's core value: autonomous
RL-driven game testing that finds bugs humans miss.

**Current state (session 55):** Phases 1-4 complete. Phase 5 (shapez.io
onboarding) is the current focus — third game plugin to validate the
platform handles complex factory-builder games with mouse+keyboard input,
build toolchain, and rich state spaces. 1137 tests, 95.26% coverage.

---

## Phase 1: First Real Training & QA Report ✓

**Goal:** Prove the platform works end-to-end by producing a trained agent,
evaluating it, and generating a QA report with real oracle findings.

| Task | Details |
|---|---|
| Merge PR #65 | `--game` flag and dynamic plugin loading — **DONE** |
| Run 200K-step PPO training | CNN policy, survival mode, 189K steps — **DONE** |
| Run 10-episode evaluation | Mean length 402.6, 4 critical findings — **DONE** |
| Generate QA report | Oracle findings, HTML dashboard — **DONE** |
| Random baseline comparison | Mean length 5, 0 critical findings — **DONE** |
| Analyze results | 80x survival, 63x findings vs random — **DONE** |

**Success criteria:** All met. Trained model 80x longer survival than random,
4 critical frame-freeze bugs found that random baseline missed.

---

## Phase 2: Exploration-Driven Reward (RND) ✓

**Goal:** Make the agent explore diverse game states instead of optimizing
one strategy. This is the key differentiator for QA vs gameplay.

| Task | Details |
|---|---|
| `src/platform/rnd_wrapper.py` | VecEnv wrapper with RND intrinsic reward — **DONE** (PR #86) |
| Survival reward mode | `+0.01` per step, `-5.0` on game over — **DONE** (PR #83) |
| `--reward-mode` CLI flag | `yolo\|survival\|rnd` for `train_rl.py` — **DONE** (PR #83, #86) |
| `_reward_mode` in BaseGameEnv | Platform-level override — **DONE** (PR #83) |
| CNN + RND training run | 200K steps (3 runs due to Chrome OOM) — **DONE** |
| Coverage measurement | 98 unique visual states at 200K steps — **DONE** (PR #88) |
| 10-episode evaluation | Mean length 3003, 3 critical findings — **DONE** |
| Baseline comparison | 589x vs random, 7.5x vs Phase 1 — **DONE** |

**Success criteria:** Partially met. RND agent survives much longer than
random (589x) and longer than Phase 1 CNN (7.5x when comparing max
episode length). However, RND intrinsic reward collapsed to near-zero
during training, meaning exploration was not meaningfully driven by
novelty. The agent learned survival but not diverse exploration.

**Key findings:**
- RND intrinsic reward decays to ~0.0000 within each episode (predictor
  learns target too quickly for low-diversity visual observations)
- Training episodes always hit max_steps (10K) — ball never exits in
  training, survival reward is trivially maximized
- Eval episodes are bimodal: 7/10 die instantly (3-6 steps), 3/10
  survive to max_steps — same pattern as Phase 1
- 3 critical "frozen frame" bugs found (same class as Phase 1's 4)
- 270 performance warnings (RAM > 4GB threshold) — new finding class
- State coverage: 98 unique visual fingerprints (8x8 quantized) over
  50K training steps — modest diversity

**RND architecture:**
- Fixed random CNN target: 3 conv layers -> 512-dim embedding
- Trainable predictor: same backbone + deeper head, MSE loss
- Non-episodic intrinsic rewards with running variance normalization
- Single value head with combined extrinsic + intrinsic reward

**Phase 2 vs Phase 1 vs Random comparison:**

| Metric | Random | Phase 1 CNN | Phase 2 RND |
|---|---|---|---|
| Mean episode length | 5.1 | 402.6 | 3003.0 |
| Mean reward | -4.96 | 1.01 | 26.52 |
| Survival rate | 0/10 | 4/10 (1K cap) | 3/10 (10K cap) |
| Critical findings | 0 | 4 | 3 |
| Performance warnings | 0 | 4 | 270 |
| Total findings | 64 | 4,045 | 30,316 |

---

## Phase 2b: Multi-Level Play & RND Rescue

**Goal:** Fix RND intrinsic reward collapse by enabling multi-level play
with random perk selection, creating diverse visual states across levels.

| Task | Details |
|---|---|
| Multi-level episode semantics | Level clear no longer terminates — **DONE** |
| `_handle_level_transition()` hook | BaseGameEnv optional hook, Breakout71 override — **DONE** |
| Perk picker loop | Click random perks until modal closes — **DONE** |
| JS score bridge | `READ_GAME_STATE_JS` reads score/level/lives — **DONE** |
| Score-delta reward | `compute_reward()` uses JS score delta — **DONE** |
| Level clear bonus | +1.0 per level cleared (not terminal) — **DONE** |
| TDD tests | 20 new tests, 909 total, 96% coverage — **DONE** |
| Bug fix: perk_picker routing | Route perk_picker modals through `_handle_level_transition()` instead of `start_game()` — **DONE** (PR #96) |
| Bug fix: survival level clear | Add modal-based `level_cleared` signal (`modal_level_cleared`) bypassing YOLO suppression, add non-terminal level clear bonus (+1.0) — **DONE** (PR #96) |
| Bug fix tests | 3 new tests (912 total, 96.31% coverage) — **DONE** (PR #96) |
| RND training validation | Run 200K+ with multi-level, verify RND stays alive — **FAILED** (100K steps, RND collapsed, degenerate policy) |
| Evaluation comparison | Compare vs Phase 2 single-level results — **DONE** (identical: mean 3003, 3 critical) |

**Success criteria:** NOT MET. RND intrinsic reward collapsed to zero
within first 100 steps. Agent learned degenerate survival policy (paddle
frozen at x=0.618), never cleared bricks, never triggered multi-level
play. Evaluation results identical to Phase 2.

**Phase 2b Evaluation Results:**

| Metric | Random | Phase 1 CNN | Phase 2 RND | Phase 2b RND |
|---|---|---|---|---|
| Mean episode length | 5.1 | 402.6 | 3003.0 | 3003.0 |
| Mean reward | -4.96 | 1.01 | 26.52 | 26.51 |
| Survival rate | 0/10 | 4/10 (1K cap) | 3/10 (10K cap) | 3/10 (10K cap) |
| Critical findings | 0 | 4 | 3 | 3 |
| Performance warnings | 0 | 4 | 270 | 297 |
| Total findings | 64 | 4,045 | 30,316 | 30,328 |

**Root cause analysis:** The survival reward (+0.01/step) creates a
trivially exploitable local optimum. The agent learns to park the paddle
and collect survival reward indefinitely. RND intrinsic reward, which
should drive exploration, collapses because the predictor network learns
the target network's output for the static frame within ~100 gradient
updates. Multi-level play cannot rescue this because the agent never
clears bricks to trigger a level transition.

---

## Phase 2c: Paddle Fix & First Real Training

**Goal:** Complete 200K training with working paddle movement (PR #107
fixed the critical IIFE arguments shadowing bug that prevented the paddle
from ever moving) and crash recovery (PR #117 fixed stale cached frames
preventing browser crash recovery from triggering).

| Task | Details |
|---|---|
| MOVE_MOUSE_JS bug fix | IIFE `arguments` shadowing — paddle never moved — **DONE** (PR #107) |
| RND exploration fix | Configurable `survival_bonus`, `EpsilonGreedyWrapper` — **DONE** (PR #105) |
| Browser crash recovery | `is_alive()` + `restart()`, crash detection in headless capture — **DONE** (PR #114, #117) |
| Training log analysis tool | JSONL parser with episode/coverage/paddle/RND analysis — **DONE** (PR #111) |
| 200K CNN+RND training | 4 runs (crashes at 64K, 114K, 178K; final 150K→229K successful) — **DONE** |
| 10-episode evaluation | Mean length 204, 1 critical finding, 9/10 instant deaths — **DONE** |
| Random baseline comparison | Mean length 608, 0 critical findings, 3/10 truncated — **DONE** |

**Training results (200K steps, post-paddle-fix):**
- 259 episodes total (across 4 runs, final run 150K→229K)
- 89.2% terminate via game_over (natural deaths — paddle is moving!)
- Episode length: mean=305, median=35, min=6, max=2000
- 15,521 unique visual states (214/1K growth rate — 158x Phase 2's 98)
- 28 degenerate episodes (10.8%, all truncated at max_steps=2000)
- Paddle movement: HEALTHY (9 unique positions, 55.3% most common)
- FPS: 14.97 mean, stable throughout

**Evaluation results (200K model, max_steps=2000):**

| Metric | Random | Phase 2c Trained |
|---|---|---|
| Mean episode length | 608.4 | 203.8 |
| Median episode length | 15 | 4 |
| Mean reward | 2.57 | -2.48 |
| Game over rate | 7/10 | 9/10 |
| Truncated (max_steps) | 3/10 | 1/10 |
| Critical findings | 0 | 1 |
| Total findings | 11,804 | 2,055 |

**Success criteria:** NOT MET. The trained model performs **worse** than
random baseline (mean 204 vs 608 steps). 9/10 episodes die in 4 steps
(ball lost immediately after release). The training data showed genuine
learning (diverse episodes, paddle movement, 15K+ unique states), but the
policy does not generalize from training to evaluation.

**Root cause analysis:** The bimodal eval pattern (90% instant death,
10% degenerate survival) suggests the model learned two modes during
training: (1) active play that occasionally catches the ball but mostly
fails at the critical first bounce, and (2) a degenerate parking strategy.
During evaluation, mode (1) dominates but fails because the initial ball
trajectory after release is highly variable and the 4-frame CNN observation
provides insufficient temporal context for reliable first-bounce
interception. The 200K training budget may also be insufficient — the
model saw ~259 episodes with mean 305 steps, which is modest experience.

**Full cross-phase comparison:**

| Metric | Random | Phase 1 | Phase 2 | Phase 2b | Phase 2c |
|---|---|---|---|---|---|
| Mean episode length | 608 | 403 | 3003 | 3003 | 204 |
| Mean reward | 2.57 | 1.01 | 26.52 | 26.51 | -2.48 |
| Survival rate | 3/10 | 4/10 | 3/10 | 3/10 | 1/10 |
| Critical findings | 0 | 4 | 3 | 3 | 1 |
| Total findings | 11,804 | 4,045 | 30,316 | 30,328 | 2,055 |

*Note: Phases 1-2b had the IIFE bug (paddle never moved). Phase 2c is the
first evaluation with working paddle movement. Prior phases' survival came
from degenerate paddle-parking, not active gameplay. The random baseline
also benefits from degenerate trajectories where the ball enters infinite
bounce loops without paddle interaction.*

---

## Phase 3: Game-Over Detection Generalization ✓

**Goal:** Detect game-over without DOM access, enabling the platform to
work with any game (not just web games with inspectable DOM).

| Task | Details |
|---|---|
| Screen freeze detector | Pixel diff < threshold for N consecutive frames — **DONE** (PR #91) |
| Entropy collapse detector | Static/uniform screen detection — **DONE** (PR #91) |
| Motion cessation detector | Replaces input responsiveness — passive motion fraction tracking — **DONE** (PR #91) |
| OCR terminal text detector | `TextDetectionStrategy` with pytesseract, 7 default patterns — **DONE** (PR #91) |
| Ensemble `GameOverDetector` | Configurable strategies with per-game weights — **DONE** (PR #91) |
| Integrate into BaseGameEnv | `step()` and `reset()` lifecycle — **DONE** (PR #92) |
| Wire into CLI scripts | `--game-over-detector`, `--detector-threshold` flags — **DONE** (PR #103) |
| Live validation on Breakout 71 | 0% false positive rate (10 episodes, random policy) — **DONE** (session 50) |

**Live validation results (session 50):**
- 10 episodes with `--game-over-detector --detector-threshold 0.6`
- Random policy, headless mode, max_steps=2000
- Mean episode length: 610.1 steps (virtually identical to baseline 608)
- Game over rate: 7/10 (all via DOM detection), 3/10 truncated
- Detector false positive rate: **0%** (never fired during active gameplay)
- Detector true positive rate: 0% (DOM detection always fires first)
- Key finding: DOM-based detection is instantaneous; pixel detector needs
  ~16 consecutive frozen frames (~1.1s at 15 FPS) before reaching
  confidence threshold. The detector's value is for non-DOM games.

**Success criteria:** MET. Pixel-based game-over detection was validated on
Breakout 71 to have 0% false positive rate without any JS injection
(criterion was <5%); DOM-based detection handled all true positives in
this DOM-enabled game.

---

## Phase 4: Second Game Onboarding ✓

**Goal:** Validate the plugin architecture by onboarding a completely
different game with zero platform code changes.

| Task | Details |
|---|---|
| Select second game | Hextris (hextris.io) — hexagonal puzzle, keyboard rotation, different genre — **DONE** |
| Create `games/hextris/` plugin | env.py, loader.py, modal_handler.py, perception.py, `__init__.py` — **DONE** (PR #120) |
| CNN-only YOLO bypass | Override `_lazy_init()` and `_detect_objects()` to skip YOLO for CNN-only games — **DONE** (PR #120) |
| Plugin tests | 91 tests (73 env + 11 loader + 7 plugin loading), zero regressions — **DONE** (PR #120) |
| Clone Hextris repo | Clone to local directory, set `$HEXTRIS_DIR` — **DONE** |
| Auto-discover plugin loaders | Factory scans `games/` for plugin loader classes — **DONE** |
| Live validation | `--game hextris --headless --episodes 3` with random policy — **DONE** (session 51) |
| Discrete action logging fix | `Discrete` action spaces crash callback — **DONE** (PR #122) |
| CNN + survival training | 200K steps, 323 episodes, 184K unique visual states, mean reward 1.18 — **DONE** (session 52) |
| 10-episode evaluation | Trained mean 404 steps, 3 critical findings vs random mean 374, 0 critical — **DONE** (session 52) |
| QA report comparison | Cross-game oracle findings — **DONE** (session 52) |

**Hextris plugin architecture (PR #120):**
- `Discrete(3)` action space: noop / rotate_left / rotate_right (via JS
  `MainHex.rotate()` injection)
- CNN-only pixel observation (no YOLO model required)
- Game state detection via `window.gameState` (0=start, 1=playing,
  2=game_over, -1=paused) — gray-box hook for DOM-enabled web games;
  Phase 3's pixel-only GameOverDetector is available for non-DOM games
- Game-over confirmed over 3 consecutive frames
- Static HTTP serving on port 8271 (no build step)
- 0 changes to `src/` or `src/platform/` — plugin architecture validated

**Live validation results (session 51):**
- 3 episodes with random policy, headless mode, max_steps=500
- Mean episode length: 256 steps (natural game overs)
- Mean reward: -5.23 (survival mode)
- 837 total findings (0 critical, 71 warnings, 766 info)
- Full lifecycle validated: start, play, game-over detection, restart
- QA report and HTML dashboard generated successfully

**Onboarding & validation success criteria:** MET. Hextris running with
`--game hextris`, producing QA reports. Plugin code complete (PR #120),
live validation successful. Auto-discover plugin loaders added to factory
for seamless multi-game support.

**CNN training results (200K steps, session 52):**
- 200,704 timesteps in 3h 6m (~19 it/s, ~20 FPS)
- 323 episodes, ALL terminated via natural game_over (no degenerate
  truncation at max_steps — unlike Breakout71's 10.8% truncation rate)
- Episode length: mean=620, diverse range (no bimodal degenerate pattern)
- Reward: mean=1.18, std=1.78, best=9.10, worst=-2.45
- **184,314 unique visual states** (12x more than Breakout71's 15,521)
- Explained variance: 0.883 (strong value function learning)
- 4 checkpoints saved (50K, 100K, 150K, 200K)
- Zero browser crashes (crash recovery from PR #114 working)
- Discrete action logging fix required (PR #122) — `Discrete(3)` actions
  are scalars, not arrays

**Hextris evaluation results (200K model, session 52):**

| Metric | Random | Trained (200K) |
|---|---|---|
| Mean episode length | 374 | 404 |
| Mean reward | -1.28 | -0.98 |
| Critical findings | 0 | **3** |
| Warning findings | 462 | 735 |
| Total findings | 4,186 | 4,741 |
| Game over rate | 10/10 | 10/10 |

**Cross-game QA comparison (Breakout71 vs Hextris, trained models):**

| Metric | Breakout71 Trained | Hextris Trained |
|---|---|---|
| Training steps | 200K | 200K |
| Training episodes | 259 | 323 |
| Unique visual states | 15,521 | 184,314 |
| Eval mean length | 204 | 404 |
| Eval mean reward | -2.48 | -0.98 |
| Eval critical findings | 1 | 3 |
| Eval total findings | 2,055 | 4,741 |
| vs random (length) | 0.34x (worse) | 1.08x (better) |
| vs random (critical) | 1 vs 0 | 3 vs 0 |
| Plugin code changes to `src/` | N/A (reference) | 0 files |

**Phase 4 success criteria:** MET. The plugin architecture is validated:
- A completely different game (hexagonal puzzle vs brick-breaking arcade)
  was onboarded with **zero changes to `src/` or `src/platform/`** (PR #120)
- CNN training required a separate bug fix (PR #122) in the training
  script (`scripts/train_rl.py`) — a pre-existing `Discrete` action space
  handling bug, not a plugin architecture limitation
- CNN training ran successfully with a different action space (`Discrete(3)`
  vs `Box(-1,1)`) after that fix
- Trained model found **3 critical findings** that random baseline missed
  (same pattern as Breakout71)
- 184K unique visual states demonstrate the platform generalizes to
  diverse game types
- The Hextris trained model performs **better** than random (unlike
  Breakout71 where it performed worse), suggesting the simpler
  rotation-based action space is easier to learn than paddle positioning

---

## Phase 5: Third Game Onboarding — shapez.io ✓

**Goal:** Validate the platform handles complex factory-builder games with
mouse+keyboard input, a build toolchain, and rich state spaces. This is
the hardest onboarding yet — shapez.io is a commercial indie game with a
real build pipeline, complex UI, and a fundamentally different interaction
model from the arcade/puzzle games in Phases 1-4.

**Why shapez.io:**
- **Genre jump:** Factory builder vs arcade/puzzle — proves the platform
  is truly game-agnostic, not just "works on simple canvas games"
- **Input complexity:** Mouse click + drag + keyboard shortcuts (not just
  position or rotation)
- **Build toolchain:** Node.js 16, Yarn, Java, ffmpeg — tests that the
  game loader handles real build steps
- **Rich state space:** Factory layouts, conveyor belts, shape processing,
  research tree — orders of magnitude more visual diversity than Breakout
  or Hextris
- **Commercial game:** 6.8K GitHub stars, available on Steam — finding
  bugs here is a credible QA demo
- **Open issues:** 112 open issues, 96 open PRs, not actively maintained
  (team on shapez 2) — high chance of rediscovering known bugs

| Task | Details |
|---|---|
| Clone shapez.io repo | Clone to `/home/human/games/shapez.io`, set `$SHAPEZ_DIR` — **DONE** |
| Build toolchain setup | nvm Node.js 16, Yarn 1.22.22, Java 21, ffmpeg 6.1.1 — **DONE** |
| Verify local build | Dev server on port 3005 via `yarn gulp` — **DONE** |
| Study game source | 12 game states, `window.shapez.GLOBAL_APP` API, 26 levels + freeplay — **DONE** |
| Create `configs/games/shapez.yaml` | Port 3005, landscape, window 1280×1024 — **DONE** (PR #128) |
| Create `games/shapez/` plugin | 755-line env, 464-line modal handler (15 JS snippets), 230-line loader — **DONE** (PR #128) |
| Design action space | `MultiDiscrete([7,10,16,16,4])`: noop, select_building, place, delete, rotate, pan, center_hub — **DONE** (PR #128) |
| Implement game state detection | `window.shapez.GLOBAL_APP` JS bridge for level, shapes, entities — **DONE** (PRs #128, #133) |
| Implement session boundaries | `max_steps` truncation + idle detection (3000 steps) — **DONE** (PR #128) |
| CNN-only observation | 84×84 grayscale, `_lazy_init`/`_detect_objects` overrides — **DONE** (PR #128) |
| Plugin tests | 160 tests (143 env + 9 loader + 8 plugin), 1289 total, 94.88% coverage — **DONE** (PR #128) |
| Bug fixes | 6 PRs (#131-#136): JS API migration, canvas re-init, InGameState polling, port cleanup — **DONE** |
| MultiDiscrete action logging | `_serialize_action()` with `isinstance` type detection — **DONE** (PR #137) |
| Live validation | 3 episodes × 500 steps, full lifecycle validated — **DONE** (session 58) |
| CNN training | 200,704 steps, 67 episodes, 12 unique visual states, ~13 FPS — **DONE** (session 59) |
| 10-episode evaluation | Trained vs random comparison — **DONE** (session 59) |
| QA report | Reports + HTML dashboards generated for both trained and random — **DONE** (session 59) |

**Key challenges (all resolved):**
- **No natural game-over:** Solved with idle detection (3000-step threshold)
  and `max_steps` truncation. All episodes terminate via idle detection.
- **Tutorial flow:** Disabled via `offerHints = false` in `SETUP_TRAINING_JS`.
- **Complex input model:** `MultiDiscrete([7,10,16,16,4])` maps 7 action
  types to JS function calls. SB3 PPO natively supports `MultiDiscrete`.
- **Build step:** `ShapezLoader` uses nvm Node 16 preamble with yarn/gulp.
  `_kill_port_processes()` prevents orphan process accumulation.
- **JS API migration:** shapez.io uses `window.shapez.GLOBAL_APP` (not
  `window.globalRoot`), available from main menu onward.

**shapez.io plugin architecture (PR #128):**
- `MultiDiscrete([7,10,16,16,4])` action space: noop, select_building(10),
  place(16×16), delete, rotate, pan(4), center_hub — mapped to JS calls
- CNN-only pixel observation (no YOLO model required)
- Game state detection via `window.shapez.GLOBAL_APP` JS bridge (gray-box)
- Idle detection termination (3000 steps without meaningful state change)
- nvm Node 16 build toolchain with yarn/gulp dev server (port 3005)
- 0 changes to `src/` or `src/platform/` — plugin architecture validated

**CNN training results (200K steps, session 59):**
- 200,704 timesteps in 4h 41m (~13 FPS)
- 67 episodes, ALL terminated via idle detection at max_steps=3000
- Episode length: mean=2979, min=1817, max=3000
- Reward: mean=24.79, std=1.46, best=25.00, worst=13.15
- **12 unique visual states** (8×8 fingerprint — very low diversity, random
  actions in a factory builder rarely produce meaningful layout changes)
- Explained variance: ~0.0 (survival reward is constant, no learning signal)
- Entropy: -11.1 (unchanged — `MultiDiscrete([7,10,16,16,4])` = 71,680
  possible action combinations, policy remains near-uniform)
- 4 checkpoints saved (50K, 100K, 150K, 200K)
- Zero browser crashes

**shapez.io evaluation results (200K model, session 59):**

| Metric | Random | Trained (200K) |
|---|---|---|
| Mean episode length | 3000 | 3000 |
| Mean reward | 25.00 | 25.00 |
| Critical findings | 10 | 10 |
| Warning findings | 496 | 510 |
| Total findings | 506 | 520 |
| Game over rate | 10/10 | 10/10 |

**Cross-game QA comparison (Breakout 71 vs Hextris vs shapez.io, trained models):**

| Metric | Breakout 71 | Hextris | shapez.io |
|---|---|---|---|
| Training steps | 200K | 200K | 200K |
| Training episodes | 259 | 323 | 67 |
| Unique visual states | 15,521 | 184,314 | 12 |
| Eval mean length | 204 | 404 | 3000 |
| Eval mean reward | -2.48 | -0.98 | 25.00 |
| Eval critical findings | 1 | 3 | 10 |
| Eval total findings | 2,055 | 4,741 | 520 |
| vs random (length) | 0.34x (worse) | 1.08x (better) | 1.00x (same) |
| vs random (critical) | 1 vs 0 | 3 vs 0 | 10 vs 10 |
| Plugin code changes to `src/` | N/A (reference) | 0 files | 0 files |
| Action space | `Box(-1,1)` | `Discrete(3)` | `MultiDiscrete([7,10,16,16,4])` |

**Success criteria:**
- ✅ shapez.io running with `--game shapez`, producing QA reports
- ✅ Zero changes to `src/` or `src/platform/` (plugin-only, PRs #128, #131-#137)
- ❌ Trained model explores more diverse states than random baseline —
  NOT MET (both produce 12 unique states; survival reward provides no
  exploration incentive in a factory builder where staying alive is trivial)
- ❌ At least 1 finding that random baseline misses — NOT MET (both
  produce identical finding patterns: 1 critical frozen-frame per episode
  + performance warnings)

**Root cause analysis:** The survival-only reward mode (+0.01/step) is
appropriate for arcade games where survival requires skill (Breakout,
Hextris) but provides zero learning signal for factory builders where
the agent cannot die. The `MultiDiscrete([7,10,16,16,4])` action space
(71,680 combinations) is too large for the policy to learn meaningful
building sequences from 200K steps of constant reward. RND exploration
reward (not used here) might help by rewarding novel screen states, but
the fundamental challenge is that meaningful factory-building actions
require multi-step sequences (select tool → click location → confirm
placement) that random sampling almost never produces. Future work should
investigate curriculum-based reward (e.g., reward for placing any building,
then for connecting conveyors, then for delivering shapes) or
demonstration-guided exploration.

**Phase 5 takeaways:**
1. **Plugin architecture validated at scale:** Three games across three
   genres (arcade → puzzle → factory builder) with zero platform code
   changes. `MultiDiscrete`, `Discrete`, and `Box` action spaces all work.
2. **Build toolchain handling works:** nvm Node 16, yarn, gulp, Java —
   the loader infrastructure handles real build pipelines.
3. **Survival reward is genre-limited:** Works for games where survival
   is hard (arcade/puzzle), fails for games where survival is trivial
   (strategy/builder/sim). Game-specific or OCR-based reward signals
   are needed for these genres (Phase 6 research).

---

## Phase 6: Advanced Reward Strategies (Research)

**Goal:** Investigate whether domain knowledge improves exploration beyond
pure novelty-seeking.

| Task | Details |
|---|---|
| **Tier 2: OCR score delta** | Game-agnostic score reading via Tesseract/EasyOCR |
| Score region auto-detection | Or per-game config for score location |
| **Tier 3: Oracle-guided exploration** | Oracle findings -> state fingerprint -> exploration bonus |
| State proximity metric | Pixel space, latent space, or RND embedding space |
| Proximity bonus with decay | Avoid getting stuck near one anomaly |
| Directed vs undirected comparison | Measure coverage improvement over RND alone |

**Success criteria:** Measurable coverage improvement over RND alone.

---

## Deprioritized

These are nice-to-haves that don't block the core value proposition:

- `register_game()` decorator — current `load_game_plugin()` works fine
- `detect_to_game_state()` refactoring — works, not blocking anything
- `games/breakout71/reward.py` extraction — reward logic is small, stays in env
- Retrain YOLO with human-reviewed Roboflow annotations — CNN is default now
- DXGI Desktop Duplication (DXcam) for capture — wincam already <1ms
