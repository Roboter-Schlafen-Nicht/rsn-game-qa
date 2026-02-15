# Client Workflow Notes

Lessons learned from the RSN Game QA project that apply to real-life client
engagements.

## Study the domain before coding

**Lesson (session 8):** The original Breakout71 environment spec was written
based on Perplexity research and assumptions about how breakout games typically
work. When we later studied the actual game source code, we discovered the spec
had several wrong assumptions:

| Assumption | Reality |
|---|---|
| Breaking bricks = direct points | Breaking bricks spawns coins; catching coins = points |
| Traditional lives counter | No lives -- `extra_life` perk provides expendable rescues |
| Single ball always | `multiball` perk can spawn additional balls |
| Simple level progression | 7+ levels with perk selection screens between them |
| Keyboard is primary input | Mouse position is primary; keyboard is secondary |

**Impact:** If we had implemented the original spec without studying the source,
we would have built the wrong reward function (brick-based instead of
coin-based), the wrong termination logic (lives counter instead of ball-loss
detection), and missed critical game elements (coins, perks, level transitions).

**Takeaway for client projects:**

1. **Always study the actual system before designing the solution.** Read the
   source code, play the game, use the product. Design docs and specs written
   from second-hand knowledge will have wrong assumptions.

2. **Budget research time explicitly.** Tell the client: "Before I estimate or
   design, I need N hours to study the existing system." This is not wasted
   time -- it prevents building the wrong thing.

3. **Document what you discovered vs. what was assumed.** Create a clear
   "assumptions vs. reality" table (like above) to justify design changes.
   Clients respect evidence-based course corrections far more than discovering
   mid-project that the foundation was wrong.

4. **Prototype with mocks, but validate against reality early.** The env can be
   fully unit-tested with mocks, but the reward function design and observation
   space must be grounded in how the real system actually works.

5. **Scope v1 tightly based on what you actually know.** After studying the
   game, we scoped v1 to single-level episodes with brick-based reward and
   placeholder slots for coin/score tracking. This is honest: we know what we
   can observe (bricks via YOLO) and what we can't yet (score, coins reliably).
   Don't pretend to support features you haven't validated.
