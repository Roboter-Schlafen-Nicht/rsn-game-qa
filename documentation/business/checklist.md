# 2026 Checklist – Roboter Schlafen Nicht

## Q1 2026 (Feb–Mar) – Foundation & clarity

- [ ] **Financial baseline**
  - [ ] Fix your monthly personal budget (confirm 3,000 EUR net target).
  - [ ] Estimate effective tax/insurance rate and derive minimum freelance hours.
  - [ ] Decide how much of the 35,000 EUR savings you are willing to risk in 2026.

- [ ] **Technical core – HELP vertical slice**
  - [ ] Stabilize V1 HELP bot (heuristic) so it runs unattended for multiple hours.
  - [ ] Ensure logging of frames + JSON metadata works reliably for HELP sessions.
  - [ ] Make the simulated RL HELP environment (LastWarHelpEnv) stable and reproducible.
  - [ ] Train at least one working PPO policy for HELP in the simulated env.
  - [ ] Integrate PPO policy into the existing HELP loop (shadow mode first).

- [ ] **Infrastructure hygiene**
  - [ ] Clean repo structure (controllers / policies / agents / rl / data / scripts).
  - [ ] Add minimal README explaining how to run V1 HELP bot and RL training.
  - [ ] Set up basic experiment tracking (simple folder structure + logs is enough).

---

## Q2 2026 (Apr–Jun) – Productize HELP bot

- [ ] **Robust HELP agent**
  - [ ] Prove real‑world RL HELP policy beats heuristic on clear metrics (e.g. helps/hour, misclicks).
  - [ ] Implement safety rails: timeouts, BACK/reset flows, “stuck” detection.
  - [ ] Run long (2–4h) test sessions and log failure modes; fix top 3 recurrent issues.

- [ ] **Twin dataset loop**
  - [ ] Automate collection of live logs to a per‑run directory.
  - [ ] Build a repeatable script to convert logs to a twin YOLO dataset (images + labels).
  - [ ] Train at least one improved YOLO HELP model from twin data and deploy it into the bot.
  - [ ] Validate improved detection quality on your own gameplay.

- [ ] **Usability**
  - [ ] Provide a simple configuration file or minimal UI (e.g. `.env` or small config dialog).
  - [ ] Create one “fresh install” script / doc from zero to running HELP bot.
  - [ ] Record a short internal demo video of the HELP bot running end‑to‑end.

- [ ] **Planning for H2**
  - [ ] Review freelance hours used vs 1,750h contract; adjust H2 planning if behind.
  - [ ] Decide preliminary July–Dec pattern (how much you want to reduce freelance load).

---

## Q3 2026 (Jul–Sep) – Users & feedback

- [ ] **Early tester program**
  - [ ] Set up a small community space (Discord/Telegram) for “Last War Automation Lab”.
  - [ ] Recruit 5–10 alpha testers (mix of Last War players + technical friends).
  - [ ] Prepare an “alpha onboarding” guide: prerequisites, install steps, known limitations.
  - [ ] Support testers through at least one full weekend of use each.

- [ ] **Feedback & iteration**
  - [ ] Collect systematic feedback: stability, detection errors, UX pain points.
  - [ ] Log and prioritize issues; fix the top stability and UX problems.
  - [ ] Add basic telemetry (even manual log parsing) to see how often HELP runs, how many helps/hour, error cases.

- [ ] **Business story (lightweight)**
  - [ ] Compress your funding memo into:
    - [ ] A 1–2 page “problem → solution → roadmap” doc for Last War only.
    - [ ] One slide / diagram showing the 4‑layer architecture (perception, control/RL, planning, strategy).
  - [ ] Draft initial pricing idea for HELP automation (e.g. 10–20 EUR/month).

---

## Q4 2026 (Oct–Dec) – Pre‑launch & structure

- [ ] **From alpha to small paid beta**
  - [ ] Decide clear “beta” conditions (minimum stability / features).
  - [ ] Offer early testers a discounted paid beta plan and see who converts.
  - [ ] Implement simple license key or account check (can be manual at first).
  - [ ] Document support expectations (reaction times, channels).

- [ ] **Company and legal basics**
  - [ ] Decide legal form for 2027 start (e.g. Einzelunternehmen first).
  - [ ] Talk to a Steuerberater about:
    - [ ] Freelance + startup income structure.
    - [ ] VAT handling for subscriptions.
  - [ ] Register “Roboter Schlafen Nicht” officially (name, address in Essen).
  - [ ] Set up:
    - [ ] Business bank account.
    - [ ] Basic accounting workflow (invoicing, expense tracking).

- [ ] **2027 working model**
  - [ ] Decide final 2027 freelance hours/month (e.g. 70–85h).
  - [ ] Confirm with main client if you want a smaller 2027 contract or more flexible arrangement.
  - [ ] Lock in your Fri–Sun RSN routine in the calendar and treat it as default.

- [ ] **Launch readiness**
  - [ ] Prepare a “Jan 2027 beta” build:
    - [ ] Stable installer / setup path.
    - [ ] Clear versioning and changelog.
  - [ ] Prepare public‑facing material:
    - [ ] Landing page or simple site section describing HELP bot.
    - [ ] At least one polished demo video.

---

## Continuous (all year)

- [ ] **Technical**
  - [ ] Regularly review logs to refine RL rewards, edge cases, and YOLO training.
  - [ ] Keep an experiment logbook: what you tried, what worked, what broke.

- [ ] **Personal**
  - [ ] Respect the Fri–Sun RSN blocks as your “default” commitment.
  - [ ] Use social events as exceptions, not the default, and consciously re‑plan that week.
  - [ ] Recheck finances quarterly: savings level, freelance hours, and how comfortable you feel with risk.

- [ ] **Vision**
  - [ ] Keep one doc up to date with the longer‑term roadmap: from HELP vertical slice to more actions (Daily, Alliance, etc.) and eventually world‑model / LLM strategy layer.
