# RSN Discipline Protocol 2026

## Identity

> In 2026, I am the kind of person who:
>
> - Shows up for **Roboter Schlafen Nicht** every weekend,
> - Ships at least one meaningful thing for RSN each week,
> - And takes boring glue work seriously because it turns research into impact.

---

## 1. Non‑negotiable Rules

1. **Weekend Presence Rule**
   - Every **Friday, Saturday, and Sunday**, I sit down at the RSN machine and open the project for **at least 15 minutes**.
   - I may stop after 15 minutes, but I may **not skip** the sit‑down (except illness or no‑laptop travel).

2. **No‑Guilt Swap Rule**
   - If I skip one RSN day for a social event, I schedule a **compensating 2‑hour RSN block** within the next 7 days.
   - Social events are allowed, but skipped time is consciously recovered.

---

## 2. Weekly Structure (Fri–Sun)

**Friday**

- 09:00–12:00: Freelance.
- 13:00–17:00: **RSN deep‑work block** (hardest technical task of the week).

**Saturday**

- 10:00–14:00: RSN deep‑work block (core build/research).
- 14:00–18:00: Optional second RSN block or buffer for errands/social.

**Sunday**

- 10:00–13:00: RSN **light block** (docs, tests, planning, long runs).
- 13:30–14:00/14:15: Weekly review ritual (see below).

I protect **11:00–15:00** as prime deep‑work time on Fri/Sat whenever possible.

---

## 3. Daily RSN Focus: “Big Rock” System

For each RSN day I define **one Big Rock** in a Markdown checklist (e.g. `rsn-week-X.md`):

- **Friday Big Rock:** hardest technical task (RL, envs, pipelines).
- **Saturday Big Rock:** substantial build task (features, infra, data).
- **Sunday Big Rock:** boring but necessary task
  (docs, packaging, testing, install scripts, small refactors).

Rules:

- During the main RSN block, I work **only on the Big Rock** (no task hopping).
- If a task feels too big or vague, I break it into **small, mechanical sub‑steps**.

Example sub‑steps:

- “Write `INSTALL.md` – prerequisites section.”
- “Write `INSTALL.md` – Windows steps.”
- “Add `run_integration_tests.py` with 3 hard‑coded scenarios.”

---

## 4. Weekly Review Ritual (Sunday)

Every **Sunday after the RSN block** (13:30–14:00), I open `2026-weekly-review.md` and append a new section with this template:

```md
## Week YYYY‑WW

**Shipped for RSN this week**
- …

**What blocked or fatigued me**
- …

**Weekend presence**
- Fri: [ ] Sat: [ ] Sun: [ ]
(Checked = I did at least 15 minutes.)

**Big Rocks next week**
- Fri: …
- Sat: …
- Sun: …

**One boring but necessary task I will complete next weekend**
- …

**Pride score (1–10):** _
