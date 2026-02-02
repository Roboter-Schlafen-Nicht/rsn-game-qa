# Help Policy Roadmap (v1–v3)

## v1 – Heuristic YOLO HELP Bot

**Goal:** Single‑class HELP clicker with human‑like timing, dry‑run + live mode.

- [x] Wire inference + controller
  - [x] Implement `HelpOnlyPolicy` (load `best.pt`, infer on `screencapnp`, return best `BTN_HELP` center).
  - [x] Implement `run_help_bot(dry_run: bool)` loop:
    - Capture frame → YOLO → best HELP.
    - Apply confidence threshold, rate limiting, miss probability, jittered tap.

- [x] Human‑like behavior
  - [x] Add random observe delay and click delay.
  - [x] Add pixel jitter around the detected center.
  - [x] Add max actions per minute to prevent spam.
  - [x] Add occasional BACK press if no HELP is seen for a while.

- [x] Shadow vs live modes
  - [x] `dry_run=True`: only log “WOULD CLICK (x,y) conf=…”, no taps.
  - [x] `dry_run=False`: actually tap via `LastWarController.tap`.

- [x] Sanity check
  - [x] Run shadow mode for 10–15 minutes and inspect logs.
  - [x] Run live mode and confirm reliable HELP clicks without spamming.

---

## v2 – Twin‑Aware HELP Bot (Improved YOLO via Real Data)

**Goal:** Use live experience to improve detection; keep heuristic policy.

- [ ] Logging pipeline
  - [ ] Extend help loop to save:
    - `data/live_logs/frame_*.png` from `screencapnp`.
    - Matching `frame_*.json` with detections, `will_click`, `reason`, timestamps.
  - [ ] Log only when HELP is visible or at a low fixed frequency to control disk usage.

- [ ] Build `twin_dataset`
  - [ ] Implement `build_twin_dataset.py`:
    - Load logged frames and metadata.
    - Rerun YOLO on each image.
    - For high‑confidence HELP (e.g. `conf ≥ 0.9`), write YOLO `.txt` labels.
    - Copy selected images + labels into `data/twin_dataset/images` and `data/twin_dataset/labels`.

- [ ] Train YOLO v2
  - [ ] Merge manual dataset + `twin_dataset` into a new training config (Roboflow or local YAML).
  - [ ] Train a new model (e.g. `runs/detect/train_help_v2/weights/best.pt`) on XPU.
  - [ ] Validate on a held‑out manual set and a few manual screenshots.

- [ ] Deploy v2 model
  - [ ] Switch `HelpOnlyPolicy` to use v2 weights.
  - [ ] Shadow‑run v1 vs v2 (log both, let v1 control) and verify v2:
    - Misses fewer real HELP buttons.
    - Doesn’t introduce obvious false positives.

---

## v3 – RL HELP Policy in Digital Twin

**Goal:** Learn *when* to click HELP via RL on top of YOLO features.

- [ ] Define Gym environment
  - [ ] Implement `LastWarHelpEnv(gym.Env)`:
    - Observation: small vector, e.g. `[has_help, help_conf, since_last_help]`.
    - Actions: `{NOOP, CLICK_HELP, BACK}`.
    - Uses `LastWarController` + `HelpOnlyPolicy` in each `step()`.

- [ ] Reward shaping
  - [ ] Reward +1.0 when clicking HELP makes HELP disappear (success).
  - [ ] Penalize useless clicks (no HELP visible or HELP remains), e.g. −0.2/−0.3.
  - [ ] Small negative for NOOP when HELP is visible; tiny negative otherwise.
  - [ ] Small negative for BACK to avoid spam.

- [ ] Train RL agent
  - [ ] Plug env into an RL algorithm (e.g. PPO with MLP policy).
  - [ ] Train for N episodes in shadow mode (optionally not sending taps).
  - [ ] Inspect trajectories and rewards to ensure it learns sensible behavior.

- [ ] Integrate RL policy
  - [ ] Export the trained RL policy.
  - [ ] Replace heuristic decision logic in the help loop with `policy.predict(obs)` while keeping:
    - Same YOLO perception.
    - Same human‑like timing layer (jitter, delays, rate limits).
  - [ ] Run limited live sessions and compare against v2 heuristic:
    - HELP successes per hour.
    - Number of wasted actions / back presses.
