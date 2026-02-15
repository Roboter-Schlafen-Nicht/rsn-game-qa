# RSN Game QA — Agent Instructions

RL-driven autonomous game testing platform. First target: **Breakout 71** (browser-based TypeScript canvas game at `F:\work\breakout71-testbed`).

## Workflow

- Work on feature branches from `main` (pattern: `feature/...`, `docs/...`)
- Implement → commit → push → create PR → **request review from Copilot** → evaluate review, create issues if needed → merge with `--delete-branch` → delete local branch → **update `documentation/BigRocks/checklist.md`** (move item to Completed, record PR number)
- **Checklist updates are admin** — no Copilot review needed; commit, push, PR, merge directly
- Pre-commit hook runs the full CI pipeline via `act` (Docker-based GitHub Actions). **This can take 5+ minutes.** Use `timeout` of 600000ms for commit commands; check `git log` afterward to verify.

## Conventions

- Python 3.12, conda env `yolo`
- NumPy-style docstrings, ruff for lint/format, pytest for tests, Sphinx (Furo theme) for docs
- `conda run -n yolo` for Python commands (frequently times out — use direct paths like `C:/Users/human/miniconda3/envs/yolo/Scripts/ruff.exe` for ruff, or generous timeouts)
- CI has 4 jobs: Lint (ruff check + format), Test (pytest), Build Check (verify imports), Build Docs (Sphinx -W)
- Always run `ruff format` before committing
- Commit messages: imperative mood with type prefix (`feat:`, `fix:`, `ci:`, `docs:`)
- Docs: canonical specs in `documentation/specs/`, thin `{include}` wrappers in `docs/specs/`, API autodoc in `docs/api/`
- Delete feature branches after merging
- Update README test count after adding tests

## Key Discoveries

- **Sphinx `-W` builds**: dataclass entries in RST need `:no-index:` to avoid duplicate warnings
- **Severity convention**: `"critical"` / `"warning"` / `"info"` throughout oracles
- **`pydirectinput` in `autodoc_mock_imports`** (`docs/conf.py`)
- **pywin32 + pydirectinput ARE installed** in conda env — tests simulating "missing" must use `importlib.reload` with `sys.modules` patched to `None`
- **Windows `NUL` file artifact**: `> NUL` redirect can create a literal file; delete with `rm -f NUL`
- **GitHub self-approval not allowed** — use Copilot as reviewer
- **Copilot code review is web-UI only** — `gh pr edit --add-reviewer copilot`, `gh pr create --reviewer copilot`, and `gh api .../requested_reviewers` all silently fail. Must request Copilot review manually via the GitHub web UI (PR page → Reviewers dropdown → select "Copilot"). The `gh agent-task` CLI commands are for the Copilot *coding agent*, not the code reviewer. Allow several minutes for Copilot to post its review after requesting.
- **`git pull` times out** — use `git fetch origin main && git reset --hard origin/main`
- **`gh pr merge` times out** but often succeeds — verify with `gh pr view N --json state,mergedAt`
- LSP unresolved import errors (cv2, gymnasium, pydirectinput, etc.) are pre-existing and harmless

## Project Structure

```
src/
  game_loader/    # DONE — YAML config, factory, loaders (82 tests)
  reporting/      # DONE — JSON reports, HTML dashboard (26 tests)
  capture/        # DONE — BitBlt window capture, pydirectinput input (37 tests)
  perception/     # DONE — YoloDetector, breakout_capture (41 tests)
  oracles/        # STUB on_step — base.py complete, 5 oracle stubs
  env/            # STUB — Breakout71Env gymnasium wrapper
tests/
documentation/
  BigRocks/checklist.md   # Master checklist — the source of truth for what's done and what's next
  specs/                  # Canonical spec files
  sessions/               # Local-only session logs (gitignored)
docs/                     # Sphinx source (conf.py, api/, specs/)
```

## What's Done (sessions 1-5)

1. **Session 1** — Perplexity research (capture, input, RL, market analysis)
2. **Session 2** — Project scaffolding, game loader subsystem, CI pipeline (PR #4, #6)
3. **Session 3** — Reporting subsystem (PR #7, #8)
4. **Session 4** — Capture & Input subsystem (PR #9, #10, #11)
5. **Session 5** — Perception subsystem (PR #12)

Total: **170 tests passing**, 4 subsystems complete.

## What's Next

Read `documentation/BigRocks/checklist.md` for the full breakdown. In order:

1. **Oracle `on_step`** (`src/oracles/`) — detection logic for all 5 oracles
2. **Breakout71 Env** (`src/env/`) — Gymnasium env core methods
3. **Integration & E2E** — wire all subsystems, run episodes, generate reports

## Reference Files

When starting a new feature, read these first:
- `documentation/BigRocks/checklist.md` — what to do next and detailed subtasks
- The relevant spec in `documentation/specs/`
- The existing stub in `src/` for the subsystem being implemented
- A completed subsystem (e.g., `src/capture/`) for pattern reference
