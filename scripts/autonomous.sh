#!/usr/bin/env bash
# scripts/autonomous.sh — Run OpenCode in continuous autonomous mode
#
# Usage:
#   ./scripts/autonomous.sh              # new session
#   ./scripts/autonomous.sh --continue   # continue last session
#   ./scripts/autonomous.sh --session ID # continue specific session
#
# Stop: Ctrl+C (waits for current iteration to finish gracefully)
#
# Environment:
#   LOOP_DELAY   — seconds between iterations (default: 10)
#   MAX_ITERS    — max iterations, 0 = infinite (default: 0)
#   OPENCODE_BIN — path to opencode binary (default: opencode)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# --- Config ---
LOOP_DELAY="${LOOP_DELAY:-10}"
MAX_ITERS="${MAX_ITERS:-0}"
OPENCODE_BIN="${OPENCODE_BIN:-opencode}"
LOG_DIR="${PROJECT_DIR}/output/autonomous_logs"
mkdir -p "$LOG_DIR"

# --- Session handling ---
SESSION_FLAG=""
CONTINUE_FLAG=""
if [[ "${1:-}" == "--continue" || "${1:-}" == "-c" ]]; then
    CONTINUE_FLAG="--continue"
    shift || true
elif [[ "${1:-}" == "--session" || "${1:-}" == "-s" ]]; then
    shift
    SESSION_FLAG="--session ${1:-}"
    shift || true
fi

# --- Prompt ---
# The agent reads AGENTS.md on every iteration, which contains full
# instructions for autonomous operation. This prompt just kicks it off.
PROMPT='You are running in autonomous continuous mode. Read AGENTS.md for your full instructions and current state. Then:

1. Run `git status` and `git log --oneline -5` to orient
2. Check `documentation/ROADMAP.md` for current phase
3. Check `private/documentation/BigRocks/checklist.md` for task status
4. Check for running training: `ps aux | grep train_rl`
5. Pick the next incomplete task and execute it fully
6. When done, summarize what you accomplished

Do NOT ask for user input. Execute autonomously. You have full permissions.'

# --- Trap Ctrl+C ---
RUNNING=true
trap 'echo -e "\n[autonomous] Caught SIGINT — finishing after current iteration..."; RUNNING=false' INT

# --- Main loop ---
ITER=0
LOGFILE="$LOG_DIR/autonomous_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo " OpenCode Autonomous Mode"
echo "=========================================="
echo " Project:    $PROJECT_DIR"
echo " Log:        $LOGFILE"
echo " Delay:      ${LOOP_DELAY}s between iterations"
echo " Max iters:  $([ "$MAX_ITERS" -eq 0 ] && echo 'infinite' || echo "$MAX_ITERS")"
echo " Session:    ${CONTINUE_FLAG:-${SESSION_FLAG:-new}}"
echo "=========================================="
echo ""

{
    echo "[$(date)] Autonomous mode started"
    echo "[$(date)] Config: delay=${LOOP_DELAY}s max_iters=${MAX_ITERS}"
} >> "$LOGFILE"

while $RUNNING; do
    ITER=$((ITER + 1))

    if [[ "$MAX_ITERS" -gt 0 && "$ITER" -gt "$MAX_ITERS" ]]; then
        echo "[autonomous] Reached max iterations ($MAX_ITERS). Stopping."
        break
    fi

    echo ""
    echo "--- Iteration $ITER | $(date) ---"
    echo "[$(date)] === Iteration $ITER ===" >> "$LOGFILE"

    # Build command
    CMD=("$OPENCODE_BIN" "run")

    # First iteration: use session flags if provided
    # Subsequent iterations: always continue last session
    if [[ "$ITER" -eq 1 ]]; then
        if [[ -n "$CONTINUE_FLAG" ]]; then
            CMD+=("$CONTINUE_FLAG")
        elif [[ -n "$SESSION_FLAG" ]]; then
            # shellcheck disable=SC2206
            CMD+=($SESSION_FLAG)
        fi
    else
        CMD+=("--continue")
    fi

    CMD+=("$PROMPT")

    # Run opencode
    echo "[autonomous] Running: ${CMD[*]:0:4}..."
    if "${CMD[@]}" 2>&1 | tee -a "$LOGFILE"; then
        echo "[$(date)] Iteration $ITER completed successfully" >> "$LOGFILE"
    else
        EXIT_CODE=$?
        echo "[$(date)] Iteration $ITER exited with code $EXIT_CODE" >> "$LOGFILE"
        echo "[autonomous] opencode exited with code $EXIT_CODE"

        # If opencode crashes, wait longer before retry
        if [[ "$EXIT_CODE" -ne 0 ]]; then
            echo "[autonomous] Waiting 30s before retry..."
            sleep 30
        fi
    fi

    if $RUNNING; then
        echo "[autonomous] Sleeping ${LOOP_DELAY}s before next iteration..."
        sleep "$LOOP_DELAY"
    fi
done

echo ""
echo "[autonomous] Stopped after $ITER iterations."
echo "[$(date)] Autonomous mode stopped after $ITER iterations" >> "$LOGFILE"
