# RSN Game QA — Roadmap

Five-phase plan for delivering the platform's core value: autonomous
RL-driven game testing that finds bugs humans miss.

**Current state (session 31):** 703 tests, 96% coverage, 8 subsystems
complete, game plugin architecture working. Zero meaningful RL training
results and zero QA reports produced. The architecture is solid — now
we execute.

---

## Phase 1: First Real Training & QA Report

**Goal:** Prove the platform works end-to-end by producing a trained agent,
evaluating it, and generating a QA report with real oracle findings.

| Task | Details |
|---|---|
| Merge PR #65 | `--game` flag and dynamic plugin loading — **DONE** |
| Run 200K-step PPO training | CNN policy, portrait mode, `--max-time 7200` |
| Run 10-episode evaluation | `run_session.py --game breakout71 --episodes 10` |
| Generate QA report | Oracle findings, episode metrics, HTML dashboard |
| Random baseline comparison | 10-episode random policy evaluation for comparison |
| Analyze results | Mean episode length, reward, oracle findings frequency |

**Success criteria:** Trained model file, 10+ episode eval, QA report with
oracle findings, measurable difference vs random baseline (even if small).

**Why first:** Everything else is optimization. If the platform can't
produce a QA report from a trained agent, nothing else matters.

---

## Phase 2: Exploration-Driven Reward (RND)

**Goal:** Make the agent explore diverse game states instead of optimizing
one strategy. This is the key differentiator for QA vs gameplay.

| Task | Details |
|---|---|
| `src/platform/rnd_wrapper.py` | VecEnv wrapper with RND intrinsic reward |
| Survival reward mode | `+0.01` per step, `-5.0` on game over |
| `--reward-mode` CLI flag | `yolo\|survival\|rnd` for `train_rl.py` |
| `_reward_mode` in BaseGameEnv | Platform-level override |
| CNN + RND training run | 200K steps, compare state coverage |
| Coverage measurement | Unique visual states, perk encounters, level progression |

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
