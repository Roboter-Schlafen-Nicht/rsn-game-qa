# Business Proposition: Pixel-Based Game QA

Rough draft -- hours-based, pre-revenue. Pricing TBD based on investor
expectations and customer willingness to pay.

## The Pitch

**"We play your game 24/7 and find bugs you can't."**

RSN Game QA deploys AI agents that play your game from pixels -- no source
code access needed, no SDK integration. Our agents learn to play, and 12
specialized oracles watch every frame for crashes, stuck states, physics
violations, visual glitches, score anomalies, and balance issues. You get
an HTML dashboard with timestamped findings and replay-ready frame
sequences.

## Why Pixel-Based Is the Moat

The core architectural decision: **pixels in, mouse/keyboard out**. This
is not a limitation -- it is the product.

1. **No integration burden on the client.** They don't ship source code,
   don't add SDK hooks, don't modify their build. We just need the game
   running in a window.

2. **Works across engines.** Unity, Unreal, Godot, custom HTML5, native
   Win32, mobile via emulator -- all the same to us. We see pixels.

3. **Tests the real player experience.** If a bug is invisible to the
   player (internal state inconsistency that never manifests visually), we
   don't flag it. If it's visible, we catch it. This matches what players
   actually report.

4. **Scales to games we can't read.** Obfuscated code, closed-source
   engines, mobile games via emulator -- all fair game.

5. **Honest testing.** We have no privileged access to game state. Our
   agents face the same information constraints as real players. Bugs we
   find are bugs players will find.

## Engagement Tiers (Hours-Based)

### Tier 1: Assessment (~1 week)

**Question answered:** "Can we test this game?"

| Activity | Hours |
|----------|-------|
| Play & study the game | 4 |
| Set up capture pipeline, verify screenshot quality | 4 |
| Capture 300 frames, auto-annotate, train initial YOLO | 12 |
| Human review annotations on Roboflow | 8 |
| Build debug loop, validate detection accuracy | 8 |
| Write feasibility report | 4 |
| **Total** | **40** |

**Client gets:** Feasibility report, sample detection screenshots with
bounding boxes, mAP scores per class, honest assessment of what can and
cannot be detected, estimate for full engagement.

**Go/no-go decision point.** If detection quality is insufficient (e.g.
the game's visual style defeats YOLO at reasonable training cost), we say
so. No point proceeding to Tier 2 with bad perception.

### Tier 2: Full QA Pipeline (~3-4 weeks)

**Deliverable:** Working RL agent that plays the game + QA oracle dashboard.

| Activity | Hours |
|----------|-------|
| Tier 1 work (included) | 40 |
| Implement game-specific Env (obs, action, reward) | 16 |
| Configure oracles with game-specific thresholds | 8 |
| Train RL agent (PPO, ~200k steps) | 8 |
| Iterate reward shaping (2-3 rounds) | 16 |
| Retrain YOLO with RL-discovered states | 8 |
| Generate & deliver QA reports | 4 |
| Documentation & handoff | 8 |
| **Total** | **108** |

**Client gets:** Trained RL agent, QA dashboard with bug findings, trained
YOLO model for their game, all configs to re-run sessions independently.

### Tier 3: Ongoing QA Retainer (monthly)

| Activity | Hours/month |
|----------|-------------|
| Run nightly QA sessions (automated, review results) | 4 |
| Review oracle findings, triage bugs | 8 |
| Retrain YOLO/RL when game updates | 8 |
| Monthly QA report with trends | 4 |
| **Total** | **24/month** |

## What Makes It Expensive vs. Cheap

### The expensive part (human hours)

- **Annotation review:** 8-12 hours per game on Roboflow (manual bounding
  box correction). This is the single biggest cost driver.
- **Reward function design:** Requires deeply understanding game mechanics.
  Cannot be fully automated -- someone must decide what "good play" means.
- **Iteration cycles:** Reward shaping and YOLO retraining are empirical.
  Each round takes 4-8 hours.

### The cheap part (automated, near-zero marginal cost)

- Frame capture: fully automated random bot
- Auto-annotation: OpenCV pipeline generates first-pass labels
- YOLO training: ~23 min on GPU, fully scripted
- RL training: ~2 hours for 200k steps, fully scripted
- QA sessions: fully unattended, can run overnight/continuously
- Report generation: fully automated HTML dashboard

### How costs decrease over time

| Investment | Effect |
|------------|--------|
| Better auto-annotation | Annotation review drops from 8-12h to 2-3h |
| Genre templates (issue #41) | Phase 1 + 3 drop by ~50% for similar games |
| Transfer learning | YOLO converges faster on games with similar visual style |
| Reusable oracle configs | Less per-game threshold tuning |
| Platform maturity | Bug fixes and edge cases already handled |

**First game:** ~108 hours (Tier 2). **Second similar game:** ~60-70 hours.
**Nth game in same genre:** ~40-50 hours.

## Cost Structure for a Sustainable Business

### Fixed costs (platform development & maintenance)

- Platform engineering (ongoing): ~20 hours/month
- Infrastructure (GPU compute for training): ~50-100 EUR/month
- Roboflow subscription: ~50 EUR/month (free tier may suffice early)
- CI/CD (GitHub Actions): free tier

### Variable costs per game (one-time setup)

- Tier 1 assessment: ~40 hours
- Tier 2 full pipeline: ~68 additional hours
- Total one-time: ~108 hours

### Variable costs per game (ongoing)

- Monthly retainer: ~24 hours/month
- Compute for nightly runs: negligible (runs on existing hardware)

## Competitive Landscape

### What exists today

| Approach | Pros | Cons |
|----------|------|------|
| Manual QA teams | Human intuition, flexible | Expensive, slow, doesn't scale, humans get bored |
| Scripted test automation (Selenium, Appium) | Deterministic, repeatable | Brittle, breaks on UI changes, can't explore |
| Game-engine-integrated testing (Unity Test Framework) | Deep access, fast | Engine-locked, requires source code, misses visual bugs |
| AI-assisted QA (emerging) | Adaptive, can explore | Most require engine integration or API hooks |

### Where RSN fits

We're the **only approach that works without source code or engine
integration**. This makes us the natural choice for:

- **Publishers** testing third-party titles they didn't develop
- **Platform holders** (Steam, console manufacturers) doing certification testing
- **Studios** that want QA on builds before source is available to QA teams
- **Mobile game companies** testing competitor products for market research
- **Indie developers** who can't afford dedicated QA teams

## Honest Limitations (as of session 19)

- **YOLO CPU inference is ~5 FPS.** Agents play slower than humans. GPU
  inference will solve this (>30 FPS expected).
- **Complex UI is harder.** Nested menus, text-heavy interfaces, and
  small UI elements are harder to detect than large game objects.
- **No OCR yet.** We can't reliably read score, currency, or text from
  pixels. This limits reward function design to visual object counting.
- **First-game setup is labor-intensive.** ~108 hours is a real
  investment. The economics improve on subsequent games.
- **RL agents are not human-level players.** They explore differently
  (more random, less strategic). This is actually useful for QA (they
  find edge cases humans avoid) but means they may miss bugs that require
  skilled play to trigger.

## What We've Proven (19 Sessions)

| Capability | Evidence |
|------------|----------|
| Pixel capture from any windowed game | PrintWindow with PW_RENDERFULLCONTENT works for GPU-composited browsers |
| YOLO detection of game objects | mAP50=0.679 overall, ball=0.922, paddle=0.976, brick=0.995 |
| Automated frame collection | 300 frames captured with random bot + game state detection |
| Auto-annotation pipeline | OpenCV HSV segmentation generates first-pass YOLO labels |
| Human-in-the-loop annotation | Roboflow integration for review and correction |
| 12 bug-detection oracles | All implemented with on_step detection, parameterized thresholds |
| Gymnasium RL environment | Continuous action space, modal handling, YOLO observations |
| End-to-end pipeline validation | 4-phase debug loop: capture -> detect -> control -> gameplay |
| 510 tests, 96% coverage | CI pipeline with ruff, pytest, Sphinx, coverage enforcement |

## Next Milestones Before Going to Market

1. **Complete first RL training loop** -- prove the agent learns to play
   better than random
2. **Generate first real QA report** from trained agent episodes
3. **Build second game intake** to validate reusability and measure actual
   time savings (validates the "platform" claim)
4. **Genre templates** -- reduce per-game setup cost for common game types
5. **GPU inference** -- achieve real-time FPS for practical training speed
