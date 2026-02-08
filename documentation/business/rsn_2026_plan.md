# RSN 2026 Plan – Breakout‑71 QA Stack

## Identity & Discipline (all year)

- In 2026, I am the kind of person who:
  - Shows up for **Roboter Schlafen Nicht** every weekend.
  - Ships at least one meaningful thing for RSN each week.
  - Takes boring glue work seriously because it turns research into impact.

- Non‑negotiable rules
  - Weekend presence (Fri–Sun): at least 15 minutes at the RSN machine every day.
  - No‑guilt swap: if I skip one RSN day, I schedule a compensating 2‑hour block within 7 days.

- Weekly structure (Fri–Sun)
  - Friday: freelance in the morning, RSN deep work in the afternoon (hardest technical task).
  - Saturday: RSN deep work (core build/research).
  - Sunday: RSN light block (docs, tests, planning, long runs) + weekly review.

- Big Rock system
  - Friday Big Rock: hardest technical task (RL, envs, pipelines).
  - Saturday Big Rock: substantial build task (features, infra, data).
  - Sunday Big Rock: boring but necessary task (docs, packaging, tests).

---

## Q1 2026 (Feb–Mar) – Foundation & Clarity

**Financial baseline**

- Fix monthly personal budget (target 3,000 EUR net) and compute minimum freelance hours.
- Decide how much of 35,000 EUR savings to risk on RSN in 2026.

**Technical core – Breakout‑71 cognition vertical slice**

- Perception
  - Implement YOLO‑based detection for Breakout‑71 (paddle, ball, bricks, coins, combo, upgrades).
  - Validate on both Android capture and web version.

- Control
  - Implement heuristic agent that plays full levels and chooses upgrades.
  - Ensure the agent can run unattended for multi‑hour sessions.

- Logging & replay

  - Log frames + JSON world‑state for all sessions.
  - Add basic replay tooling (step through runs with overlays).

**Infrastructure hygiene**

- Clean repo structure: perception / control / agents / rl / data / scripts.
- Add minimal README explaining how to run:
  - Breakout‑71 agent.
  - Training scripts (if any).
- Simple experiment tracking:
  - Per‑run folders with configs, logs, metrics.

---

## Q2 2026 (Apr–Jun) – Productize Breakout‑71 Stack

**Robust grounded agent**

- RL integration
  - Build a simple Breakout‑like sim environment.
  - Train at least one PPO (or similar) policy.
  - Demonstrate the policy beats the heuristic on:
    - Coins caught.
    - Bricks missed.
    - Level clear time.

- Live loop integration
  - Integrate RL policy into the real Breakout‑71 loop (shadow mode first).
  - Switch to RL as primary control after it’s stable.

- Safety rails
  - Implement resets, “stuck” detection, and fallback to heuristics.

**Twin dataset loop for perception**

- Automate log collection for all runs into per‑run directories.
- Build scripts to generate YOLO datasets (images + bounding boxes) from logs.
- Train at least one improved YOLO model from twin data and deploy it.
- Validate detection quality on your own play and agent runs.

**Usability & demoability**

- Provide simple config (file or minimal UI) for capture + input setup.
- Create “fresh install” docs:
  - From clean machine to running agent + training.
- Record an internal demo video:
  - Perception overlays.
  - Agent behaviour.
  - Example LLM explanations.

**Planning for H2**

- Review freelance hours vs contract targets.
- Decide July–Dec freelance load to protect RSN time.

---

## Q3 2026 (Jul–Sep) – Early Users & Feedback

**Early tester program – “Grounded Game Agent Lab”**

- Set up a small community space (Discord/Telegram).
- Recruit 5–10 alpha testers:
  - Mix of game devs, AI/game‑AI researchers, technical friends.
- Prepare onboarding:
  - Prerequisites.
  - Install steps.
  - Known limitations.
  - Example test plans.

**Feedback & iteration**

- Collect structured feedback:
  - Stability.
  - Detection errors.
  - UX of config and reports.
  - Clarity/usefulness of LLM explanations.
- Prioritize and fix top issues (stability, catastrophic failures, report clarity).
- Add basic telemetry:
  - How often tests run.
  - Win/lose rates.
  - Failure clusters.

**Business story (lightweight)**

- Compress funding memo into:
  - 1–2 page “problem → solution → roadmap” for automated QA.
  - One architecture slide (perception, control/RL, planning, LLM strategy).
- Draft initial pricing for:
  - Per‑title SaaS.
  - Setup fees.
  - Consulting engagements.

---

## Q4 2026 (Oct–Dec) – Pre‑Launch & Company Structure

**From alpha to small paid beta**

- Define “beta ready” criteria:
  - Stability thresholds (e.g. can run N hours without crash).
  - Minimum feature set (config, test suites, reports).
  - Documentation quality.
- Offer early testers a discounted paid beta plan.
- Implement simple license/account check (scripted/manual is fine).
- Document support expectations:
  - Channels.
  - Response time.

**Company & legal basics**

- Decide legal form for 2027 (e.g. Einzelunternehmen).
- Talk to Steuerberater about:
  - Freelance + RSN income.
  - VAT for subscriptions.
- Register “Roboter Schlafen Nicht”.
- Set up:
  - Business bank account.
  - Basic accounting workflow (invoicing, expenses).

**2027 working model & launch readiness**

- Decide 2027 freelance hours/month (e.g. 70–85h).
- Lock Fri–Sun RSN routine into calendar.
- Prepare “Jan 2027 beta” build:
  - Stable installer/setup.
  - Versioning and changelog.
- Prepare public‑facing material:
  - Landing page describing automated QA offering.
  - At least one polished demo video.

---

## Continuous (All 2026)

- Technical
  - Regularly review logs to refine RL rewards and edge cases.
  - Maintain an experiment logbook (what you tried, what worked, what broke).

- Personal
  - Respect Fri–Sun RSN blocks as default.
  - Use social events as exceptions and re‑plan consciously.
  - Recheck finances quarterly (savings, freelance hours, risk comfort).

- Vision
  - Keep a living roadmap:
    - From Breakout‑71 vertical slice.
    - To additional games.
    - Toward a general QA automation platform.
