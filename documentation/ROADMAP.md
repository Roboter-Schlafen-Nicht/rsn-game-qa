# RSN Game QA — Roadmap

Five-phase plan for delivering the platform's core value: autonomous
RL-driven game testing that finds bugs humans miss.

**Current state (session 33):** Phase 1 complete. Phase 2 in progress.
RND wrapper implemented (PR #86 merged, 37 tests, 9 Copilot review comments
addressed). 834 tests, 96% coverage. Next: CNN+RND training run.

---

## Phase 1: First Real Training & QA Report ✓

**Goal:** Prove the platform works end-to-end by producing a trained agent,
evaluating it, and generating a QA report with real oracle findings.

| Task | Details |
|---|---|
| Merge PR #65 | `--game` flag and dynamic plugin loading — **DONE** |
| Run 200K-step PPO training | CNN policy, survival mode, 189K steps — **DONE** |
| Run 10-episode evaluation | Mean length 403, 4 critical findings — **DONE** |
| Generate QA report | Oracle findings, HTML dashboard — **DONE** |
| Random baseline comparison | Mean length 5, 0 critical findings — **DONE** |
| Analyze results | 80x survival, 63x findings vs random — **DONE** |

**Success criteria:** All met. Trained model 80x longer survival than random,
4 critical frame-freeze bugs found that random baseline missed.

---

## Phase 2: Exploration-Driven Reward (RND)

**Goal:** Make the agent explore diverse game states instead of optimizing
one strategy. This is the key differentiator for QA vs gameplay.

| Task | Details |
|---|---|
| `src/platform/rnd_wrapper.py` | VecEnv wrapper with RND intrinsic reward — **DONE** (PR #86) |
| Survival reward mode | `+0.01` per step, `-5.0` on game over — **DONE** (PR #83) |
| `--reward-mode` CLI flag | `yolo\|survival\|rnd` for `train_rl.py` — **DONE** (PR #83, #86) |
| `_reward_mode` in BaseGameEnv | Platform-level override — **DONE** (PR #83) |
| CNN + RND training run | 200K steps, compare state coverage — **IN PROGRESS** (PID 31583) |
| Coverage measurement | Unique visual states, perk encounters, level progression — **DONE** (logging merged; coverage analysis TBD, PR #88) |

**Success criteria:** CNN + RND agent visits more diverse states than
score-maximizing agent, measured by state coverage metrics.

**RND architecture:**
- Fixed random CNN target: 3 conv layers -> 512-dim embedding
- Trainable predictor: same backbone + deeper head, MSE loss
- Non-episodic intrinsic rewards with running variance normalization
- Single value head with combined extrinsic + intrinsic reward

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
