# RSN Game QA -- Agent Instructions

RL-driven autonomous game testing platform. First target: **Breakout 71**
(browser-based TypeScript canvas game; local clone configured via
`$BREAKOUT71_DIR` or `configs/games/breakout-71.yaml`).

## Role

You lead this project. Execute autonomously -- read the roadmap, pick the
next task, do it. Don't ask for permission on engineering decisions.

## Workflow

1. Work on feature branches from `main` (`feature/...`, `fix/...`,
   `docs/...`).
2. Implement -> commit -> push -> create PR -> verify CI and review ->
   merge with `--delete-branch` -> delete local branch.
3. Post-merge admin: update `private/documentation/progress.md` and
   `private/documentation/BigRocks/checklist.md`.
4. Pre-commit hook runs CI via `act` (Docker). Takes 5+ minutes.
   Use `timeout` of 600000ms for commit commands; verify with
   `git log` afterward.

## Continuation After Compaction

OpenCode auto-compacts when context is full. After compaction:

1. Re-read this file (`AGENTS.md`).
2. The following are auto-injected via `opencode.json` instructions:
   - `documentation/ROADMAP.md` -- the 5-phase plan
   - `private/documentation/progress.md` -- current status
   - `private/documentation/BigRocks/checklist.md` -- task tracking
   - `private/documentation/reference/agent_knowledge_base.md` -- pitfalls
3. Run `git status`, `git log --oneline -5`, `gh pr list` to orient.
4. Check for running training: `ps aux | grep train_rl`.
5. Resume the next incomplete task.

## Decision Authority

The agent has full authority to:
- Create branches, PRs, merge PRs (after CI passes)
- Fix bugs, refactor code, add features per the roadmap
- Launch and monitor training runs
- Generate evaluation reports and dashboards
- Update documentation and checklists
- Create GitHub issues for problems needing human attention

The agent must NOT:
- Force-push to main
- Delete data or checkpoints without clear reason
- Commit files from `private/` to git

## Conventions

- **Hardware:** Intel Arc GPU (XPU, not CUDA). `torch.xpu` for GPU,
  SB3 PPO on CPU. See `agent_knowledge_base.md` for details.
- Python 3.12, conda env `yolo`
- NumPy-style docstrings
- `ruff` for lint/format -- always run `ruff format` before committing
- `pytest` for tests
- Sphinx (Furo theme) for docs
- CI has 4 jobs: Lint, Test, Build Check, Build Docs
- Commit messages: imperative mood with type prefix (`feat:`, `fix:`,
  `ci:`, `docs:`)
- Delete feature branches after merging
- Use conda env tools directly (e.g. `ruff` on PATH, or full conda
  path if `conda run` times out)

## TDD Convention

Applies to: `src/env/`, `src/orchestrator/`, `src/platform/`, and any
module where the behavioral contract matters more than implementation.

| Change type | TDD required? |
|---|---|
| Public method or behavioral change to `step()`, `reset()`, `close()` | Yes |
| New game state transition | Yes |
| Reward function changes | Yes |
| Observation field changes | Yes |
| Orchestrator lifecycle behavior | Yes |
| Internal refactoring (same behavior) | No (existing tests must pass) |
| Scripts, utilities, exploratory work | No |

Process: Write test names as specs (with `pytest.skip`) -> fill bodies
(red) -> implement (green) -> refactor. Naming:
`test_{method}_{expected_behavior}_when_{condition}`.

## Reference

Technical pitfalls, project structure, and training run templates are
in the auto-injected private docs (see `agent_knowledge_base.md` and
`progress.md`). Canonical specs live in `documentation/specs/`.
