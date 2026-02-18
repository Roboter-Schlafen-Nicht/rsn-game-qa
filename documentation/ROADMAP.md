# RSN Game QA — Roadmap

Five-phase plan for delivering the platform's core value: autonomous
RL-driven game testing that finds bugs humans miss.

**Current state (session 38):** Phase 1 complete. Phase 2 complete.
Phase 2b (multi-level play) merged (PR #94). Bug fix for survival/RND
mode (PR #96) — routes perk_picker modals through level transition,
adds modal-based level clear signal. 912 tests, 96% coverage.
Next: Merge bug fix, then run RND training with multi-level play.

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
| RND training validation | Run 200K+ with multi-level, verify RND stays alive — TODO |
| Evaluation comparison | Compare vs Phase 2 single-level results — TODO |

**Success criteria:** RND intrinsic reward stays above zero through
multi-level episodes. State coverage (unique fingerprints) significantly
exceeds Phase 2's 98.

---

## Phase 3: Game-Over Detection Generalization

**Goal:** Detect game-over without DOM access, enabling the platform to
work with any game (not just web games with inspectable DOM).

| Task | Details |
|---|---|
| Screen freeze detector | Pixel diff < threshold for N consecutive frames |
| Entropy collapse detector | Static/uniform screen detection |
| Input responsiveness detector | Send actions, check if state changes |
| OCR terminal text detector | "Game Over", "You Died", "Continue?" patterns |
| Ensemble `GameOverDetector` | Configurable strategies with per-game weights |

**Success criteria:** Pixel-based game-over detection works on Breakout 71
without any JS injection, with <5% false positive rate.

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
