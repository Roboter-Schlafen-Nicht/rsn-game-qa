# RSN Game QA — Roadmap

Five-phase plan for delivering the platform's core value: autonomous
RL-driven game testing that finds bugs humans miss.

**Current state (session 50):** Phase 1 complete. Phase 2 complete.
Phase 2b complete (RND rescue FAILED). Phase 2c complete — first 200K
training with working paddle movement (PR #107 IIFE fix) and crash
recovery (PR #117). Training showed real gameplay: 89% game_over
episodes, 15K+ unique visual states, diverse episode lengths. However,
**evaluation regressed** — trained model (mean 204 steps) performs worse
than random baseline (mean 608 steps). The model fails to generalize
from training to evaluation. Phase 3 complete — GameOverDetector achieves
0% false positive rate on Breakout 71, meeting <5% criterion. 1037 tests,
95.47% coverage. Phase 4 starting: second game onboarding (Hextris).

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

**Success criteria:** MET. Pixel-based game-over detection works on
Breakout 71 without any JS injection, with 0% false positive rate
(criterion was <5%).

---

## Phase 4: Second Game Onboarding

**Goal:** Validate the plugin architecture by onboarding a completely
different game with zero platform code changes.

| Task | Details |
|---|---|
| Select second game | Different genre, different input modality |
| Create `games/<game>/` plugin | env.py, loader.py, `__init__.py` |
| CNN + survival training | No YOLO model needed |
| QA report comparison | Cross-game oracle findings |

**Success criteria:** New game running with `--game <name>`, producing QA
reports, with zero changes to `src/` or `src/platform/`.

---

## Phase 5: Advanced Reward Strategies (Research)

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
