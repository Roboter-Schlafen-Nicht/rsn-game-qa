#!/usr/bin/env python
"""Verify ball release AND random paddle movement.

Connects to an already-running game server, applies settings, starts
the game, moves the paddle randomly via MOVE_MOUSE_JS (the same
mechanism the env uses), and confirms both the ball and paddle move.

Usage::

    # Start server first:
    python -m http.server 1234 --directory /mnt/f/work/breakout71-testbed/dist

    # Then run:
    python scripts/verify_ball_release.py --browser chrome --headless
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._smoke_utils import BrowserInstance


# JS to read ball + game state from gameState (gray-box)
READ_BALL_STATE_JS = """
return (function() {
    if (typeof gameState === 'undefined') {
        return {error: 'no gameState'};
    }
    var balls = gameState.balls;
    var ballData = [];
    if (Array.isArray(balls)) {
        for (var i = 0; i < balls.length; i++) {
            var b = balls[i];
            if (b && typeof b === 'object') {
                ballData.push({
                    x: b.x || 0,
                    y: b.y || 0,
                    vx: b.vx || 0,
                    vy: b.vy || 0
                });
            }
        }
    }
    return {
        running: !!gameState.running,
        ballStickToPuck: !!gameState.ballStickToPuck,
        puckPosition: gameState.puckPosition || 0,
        balls: ballData,
        ballsType: typeof balls,
        ballsIsArray: Array.isArray(balls),
        ballsLength: Array.isArray(balls) ? balls.length : -1,
        levelTime: gameState.levelTime || 0,
        score: gameState.score || 0,
        currentLevel: gameState.currentLevel || 0,
        startCountDown: gameState.startCountDown || 0,
        isGameOver: !!gameState.isGameOver,
        gameZoneWidth: gameState.gameZoneWidth || 0,
        offsetX: gameState.offsetX || 0
    };
})();
"""

# Same JS the env uses to move the paddle (from modal_handler.py)
MOVE_MOUSE_JS = """
var _sel_args = arguments;
return (function(id, cx, cy) {
    var canvas = document.getElementById(id);
    if (!canvas) return {ok: false, error: "canvas not found"};
    var evt = new MouseEvent('mousemove', {
        clientX: cx,
        clientY: cy,
        bubbles: true,
        cancelable: true
    });
    canvas.dispatchEvent(evt);
    return {ok: true};
})(_sel_args[0], _sel_args[1], _sel_args[2]);
"""


def _action_to_client_x(action: float, canvas_left: float, canvas_w: float) -> float:
    """Map action in [-1, 1] to clientX pixel, same formula as env."""
    value = max(-1.0, min(1.0, action))
    return canvas_left + (value + 1.0) / 2.0 * canvas_w


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify ball release AND random paddle movement"
    )
    parser.add_argument("--browser", default="chrome")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--url", default="http://localhost:1234")
    parser.add_argument(
        "--output-dir",
        default="/mnt/e/rsn-game-qa/output/verify_ball_release",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of action steps to run (default 30)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help="Seconds between steps (default 0.2)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from games import load_game_plugin

    plugin = load_game_plugin("breakout71")

    browser = BrowserInstance(
        url=args.url,
        settle_seconds=5.0,
        window_size=(768, 1024),
        browser=args.browser,
        headless=args.headless,
    )

    driver = browser.driver
    canvas_selector = "game"  # Breakout 71 canvas element ID

    try:
        # Step 1: Apply settings
        mute_js = getattr(plugin, "mute_js", None)
        setup_js = getattr(plugin, "setup_js", None)

        if mute_js:
            driver.execute_script(mute_js)
            print("[OK] mute_js applied")
        if setup_js:
            driver.execute_script(setup_js)
            print("[OK] setup_js applied")

        driver.refresh()
        time.sleep(3)
        print("[OK] Page refreshed")

        reinit_js = getattr(plugin, "reinit_js", None)
        if reinit_js:
            driver.execute_script(reinit_js)
            time.sleep(2)
            print("[OK] reinit_js applied")

        # Step 2: Get canvas geometry
        rect = driver.execute_script(
            "var c = document.getElementById(arguments[0]);"
            "if (!c) return null;"
            "var r = c.getBoundingClientRect();"
            "return {left: r.left, top: r.top, width: r.width, height: r.height};",
            canvas_selector,
        )
        if rect is None:
            print("ERROR: Canvas not found")
            return
        canvas_left = rect["left"]
        canvas_top = rect["top"]
        canvas_w = rect["width"]
        canvas_h = rect["height"]
        client_y = canvas_top + 0.9 * canvas_h  # paddle Y
        print(
            f"[OK] Canvas: left={canvas_left}, top={canvas_top}, "
            f"w={canvas_w}, h={canvas_h}"
        )

        # Step 3: Pre-start state
        pre_state = driver.execute_script(READ_BALL_STATE_JS)
        print(
            f"\n[PRE-START] running={pre_state.get('running')} "
            f"stick={pre_state.get('ballStickToPuck')} "
            f"puck={pre_state.get('puckPosition', 0):.0f}"
        )
        driver.save_screenshot(str(output_dir / "00_pre_start.png"))

        # Step 4: Start game
        driver.execute_script(
            "gameState.running = true; gameState.ballStickToPuck = false;"
        )
        print("[START] running=true, ballStickToPuck=false\n")

        # Step 5: Run random actions + observe
        print(
            f"{'step':>4}  {'action':>7}  {'puck':>6}  {'ball_x':>6}  "
            f"{'ball_y':>6}  {'run':>4}  {'over':>4}"
        )
        print("-" * 55)

        puck_positions = []
        observations = []
        rng = random.Random(42)

        for i in range(args.steps):
            # Random action in [-1, 1]
            action = rng.uniform(-1.0, 1.0)
            client_x = _action_to_client_x(action, canvas_left, canvas_w)

            # Move paddle via JS (same as env)
            move_result = driver.execute_script(
                MOVE_MOUSE_JS, canvas_selector, client_x, client_y
            )
            if not move_result.get("ok"):
                print(f"[WARN] mousemove failed: {move_result}")

            time.sleep(args.interval)

            # Read state
            state = driver.execute_script(READ_BALL_STATE_JS)
            observations.append({"step": i, "action": action, "state": state})

            puck_pos = state.get("puckPosition", 0)
            puck_positions.append(puck_pos)

            ball_str = ""
            if state.get("balls"):
                b = state["balls"][0]
                ball_str = f"{b['x']:6.0f}  {b['y']:6.0f}"
            else:
                ball_str = "  gone    gone"

            print(
                f"{i:4d}  {action:+7.3f}  {puck_pos:6.0f}  {ball_str}  "
                f"{'Y' if state.get('running') else 'N':>4}  "
                f"{'Y' if state.get('isGameOver') else 'N':>4}"
            )

            # Save a few screenshots
            if i < 5 or i % 10 == 0 or i == args.steps - 1:
                fname = f"{i + 1:02d}_step{i}.png"
                driver.save_screenshot(str(output_dir / fname))

        # Step 6: Analysis
        print("\n" + "=" * 60)
        print("ANALYSIS")
        print("=" * 60)

        # Paddle movement analysis
        unique_puck = len(set(round(p, 1) for p in puck_positions))
        puck_min = min(puck_positions)
        puck_max = max(puck_positions)
        puck_range = puck_max - puck_min
        print(f"\nPaddle positions: {unique_puck} unique values")
        print(
            f"Paddle range: {puck_min:.0f} - {puck_max:.0f} (span={puck_range:.0f}px)"
        )

        if puck_range > 50:
            print("PADDLE: MOVING (random actions working)")
        elif puck_range > 10:
            print("PADDLE: BARELY MOVING (actions may not propagate)")
        else:
            print("PADDLE: FROZEN (actions NOT working)")

        # Ball movement analysis
        ball_obs = [o for o in observations if o["state"].get("balls")]
        if len(ball_obs) >= 2:
            first_b = ball_obs[0]["state"]["balls"][0]
            last_b = ball_obs[-1]["state"]["balls"][0]
            dx = abs(last_b["x"] - first_b["x"])
            dy = abs(last_b["y"] - first_b["y"])
            print(f"\nBall displacement: dx={dx:.1f}, dy={dy:.1f}")
            if dx > 5 or dy > 5:
                print("BALL: MOVING")
            else:
                print("BALL: FROZEN")
        elif ball_obs:
            print("\nBALL: Only one observation with ball data")
        else:
            print("\nBALL: Never seen (game over before first read?)")

        # Game over timing
        game_over_step = None
        for o in observations:
            if o["state"].get("isGameOver"):
                game_over_step = o["step"]
                break
        if game_over_step is not None:
            print(
                f"\nGame over at step {game_over_step} "
                f"(t={game_over_step * args.interval:.1f}s)"
            )
        else:
            print(f"\nGame still running after {args.steps} steps")

        # Save log
        log_path = output_dir / "ball_state_log.json"
        with open(log_path, "w") as f:
            json.dump(
                {"pre_start": pre_state, "observations": observations},
                f,
                indent=2,
            )
        print(f"\nLog: {log_path}")
        print(f"Screenshots: {output_dir}")

    except Exception as exc:
        print(f"\nERROR: {exc}")
        import traceback

        traceback.print_exc()
    finally:
        browser.close()


if __name__ == "__main__":
    main()
