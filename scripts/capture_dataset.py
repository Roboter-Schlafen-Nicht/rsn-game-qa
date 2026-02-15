#!/usr/bin/env python
"""Capture frames from Breakout 71 for YOLO training dataset.

Launches the game, plays it with a random bot (random paddle movements
via mouse input), and captures frames at configurable intervals.  Saves
PNGs and a JSON manifest with metadata for each frame.

The bot automatically detects and handles game UI states:
- **Gameplay**: random paddle movements via Selenium ActionChains
- **Perk selection**: picks a random perk from the upgrade modal
- **Game over**: clicks "New Run" to restart
- **Paused**: clicks the canvas or presses Space to resume

Detection uses ``driver.execute_script()`` to query the DOM directly
(``body.classList.contains('has-alert-open')``, ``#close-modale``
visibility, ``#popup`` content, etc.).

Usage::

    python scripts/capture_dataset.py
    python scripts/capture_dataset.py --frames 300 --interval 0.5
    python scripts/capture_dataset.py --skip-setup --browser chrome -v
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from scripts._smoke_utils import (
    BrowserInstance,
    Timer,
    base_argparser,
    ensure_output_dir,
    save_frame_png,
    setup_logging,
    timestamp_str,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Game state detection via Selenium JS execution
# ---------------------------------------------------------------------------

# Game UI states returned by _detect_game_state()
STATE_GAMEPLAY = "gameplay"
STATE_PAUSED = "paused"
STATE_PERK_PICKER = "perk_picker"
STATE_GAME_OVER = "game_over"
STATE_MENU = "menu"
STATE_UNKNOWN = "unknown"

# JavaScript snippet that queries the game DOM to determine current state.
# Returns a JSON-serializable dict with state info.
_DETECT_STATE_JS = """
return (function() {
    var result = {state: "unknown", details: {}};

    // Check if any modal/alert is open
    var hasAlert = document.body.classList.contains('has-alert-open');
    var popup = document.getElementById('popup');
    var closeBtn = document.getElementById('close-modale');

    if (!hasAlert) {
        // No modal open — either gameplay or paused
        // We can't easily check gameState.running from the DOM, but
        // if the game is running the ball is moving.  For our purposes,
        // "no modal = gameplay or paused" is fine — we click to unpause.
        result.state = "gameplay";
        return result;
    }

    // Modal is open — determine which type
    if (!popup) {
        result.state = "unknown";
        return result;
    }

    var popupText = popup.innerText || "";
    var buttons = popup.querySelectorAll('button');
    var buttonTexts = [];
    for (var i = 0; i < buttons.length; i++) {
        buttonTexts.push(buttons[i].innerText.trim());
    }
    result.details.buttonTexts = buttonTexts;
    result.details.popupTextSnippet = popupText.substring(0, 200);

    // Check if close button is visible (game-over and menu modals have it;
    // perk picker does NOT)
    var closeBtnVisible = false;
    if (closeBtn) {
        var style = window.getComputedStyle(closeBtn);
        closeBtnVisible = (style.display !== 'none' && style.visibility !== 'hidden');
    }
    result.details.closeBtnVisible = closeBtnVisible;

    // Perk picker: required modal (no close button), has upgrade buttons
    if (!closeBtnVisible && buttons.length >= 2) {
        result.state = "perk_picker";
        result.details.numPerks = buttons.length;
        return result;
    }

    // Game over: has close button, popup text usually contains score info
    // and a "New Run" / restart button
    if (closeBtnVisible) {
        // Check for restart-like button text
        var hasRestart = false;
        for (var j = 0; j < buttonTexts.length; j++) {
            var t = buttonTexts[j].toLowerCase();
            if (t.indexOf("new") >= 0 || t.indexOf("restart") >= 0 ||
                t.indexOf("run") >= 0 || t.indexOf("yes") >= 0 ||
                t.indexOf("again") >= 0) {
                hasRestart = true;
                break;
            }
        }

        if (hasRestart || popupText.toLowerCase().indexOf("game over") >= 0 ||
            popupText.toLowerCase().indexOf("score") >= 0) {
            result.state = "game_over";
        } else {
            result.state = "menu";
        }
        return result;
    }

    // Fallback: unknown modal
    result.state = "unknown";
    return result;
})();
"""

# JavaScript snippet to click a random perk button in the upgrade picker.
# Returns the index of the button clicked, or -1 if none found.
_CLICK_PERK_JS = """
return (function() {
    var popup = document.getElementById('popup');
    if (!popup) return {clicked: -1, text: ""};

    var buttons = popup.querySelectorAll('button');
    if (buttons.length === 0) return {clicked: -1, text: ""};

    // Pick a random button
    var idx = Math.floor(Math.random() * buttons.length);
    var text = buttons[idx].innerText.trim();
    buttons[idx].click();
    return {clicked: idx, text: text};
})();
"""

# JavaScript snippet to dismiss game-over modal.
# Strategy: click the close button first, or find a restart/new-run button.
_DISMISS_GAME_OVER_JS = """
return (function() {
    var popup = document.getElementById('popup');
    var closeBtn = document.getElementById('close-modale');

    // Try clicking any restart-like button first
    if (popup) {
        var buttons = popup.querySelectorAll('button');
        for (var i = 0; i < buttons.length; i++) {
            var t = buttons[i].innerText.trim().toLowerCase();
            if (t.indexOf("new") >= 0 || t.indexOf("restart") >= 0 ||
                t.indexOf("run") >= 0 || t.indexOf("yes") >= 0 ||
                t.indexOf("again") >= 0) {
                buttons[i].click();
                return {action: "restart_button", text: buttons[i].innerText.trim()};
            }
        }
    }

    // Fall back to close button
    if (closeBtn) {
        closeBtn.click();
        return {action: "close_button", text: ""};
    }

    return {action: "none", text: ""};
})();
"""

# JavaScript snippet to dismiss a generic menu modal.
_DISMISS_MENU_JS = """
return (function() {
    var closeBtn = document.getElementById('close-modale');
    if (closeBtn) {
        closeBtn.click();
        return {action: "close_button"};
    }

    // Try pressing Escape as fallback
    document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', code: 'Escape'}));
    return {action: "escape_key"};
})();
"""


def _detect_game_state(driver) -> dict:
    """Detect the current game UI state via JavaScript execution.

    Parameters
    ----------
    driver : selenium.webdriver.Remote
        The Selenium WebDriver instance.

    Returns
    -------
    dict
        ``{"state": str, "details": dict}`` where state is one of
        ``STATE_*`` constants.
    """
    try:
        result = driver.execute_script(_DETECT_STATE_JS)
        if result is None:
            return {"state": STATE_UNKNOWN, "details": {}}
        return result
    except Exception as exc:
        logger.debug("State detection failed: %s", exc)
        return {"state": STATE_UNKNOWN, "details": {}}


def _handle_modal(driver, state_info: dict) -> dict:
    """Handle a detected modal by clicking the appropriate button.

    Parameters
    ----------
    driver : selenium.webdriver.Remote
        The Selenium WebDriver instance.
    state_info : dict
        Result from ``_detect_game_state()``.

    Returns
    -------
    dict
        Action metadata describing what was done.
    """
    state = state_info["state"]

    if state == STATE_PERK_PICKER:
        result = driver.execute_script(_CLICK_PERK_JS)
        logger.info(
            "Perk picker: clicked button %d (%s)",
            result.get("clicked", -1),
            result.get("text", "?"),
        )
        return {
            "type": "perk_pick",
            "button_index": result.get("clicked", -1),
            "perk_text": result.get("text", ""),
        }

    elif state == STATE_GAME_OVER:
        result = driver.execute_script(_DISMISS_GAME_OVER_JS)
        logger.info("Game over: %s (%s)", result.get("action"), result.get("text", ""))
        return {
            "type": "game_over_dismiss",
            "action": result.get("action", "none"),
        }

    elif state == STATE_MENU:
        result = driver.execute_script(_DISMISS_MENU_JS)
        logger.info("Menu dismissed: %s", result.get("action"))
        return {"type": "menu_dismiss", "action": result.get("action", "none")}

    return {"type": "no_action"}


# ---------------------------------------------------------------------------
# Random bot actions
# ---------------------------------------------------------------------------


def _random_paddle_action(
    driver,
    game_element,
    window_rect: tuple[int, int, int, int],
) -> dict:
    """Move the mouse to a random horizontal position over the paddle area.

    Uses Selenium ActionChains for reliable input delivery to the game
    canvas, rather than pydirectinput which requires the window to be
    foreground.

    Parameters
    ----------
    driver : selenium.webdriver.Remote
        The Selenium WebDriver instance.
    game_element : selenium.webdriver.remote.webelement.WebElement
        The game canvas element.
    window_rect : tuple[int, int, int, int]
        ``(left, top, width, height)`` of the game canvas element.

    Returns
    -------
    dict
        Action metadata: ``{"type": "mouse_move", "x_norm": float}``.
    """
    from selenium.webdriver.common.action_chains import ActionChains

    _left, _top, canvas_w, canvas_h = window_rect

    # Random horizontal position, paddle is near the bottom
    x_norm = random.random()
    x_offset = int(x_norm * canvas_w) - (canvas_w // 2)  # relative to center
    y_offset = int(canvas_h * 0.85) - (canvas_h // 2)  # paddle zone

    ActionChains(driver).move_to_element_with_offset(
        game_element, x_offset, y_offset
    ).perform()

    return {"type": "mouse_move", "x_norm": round(x_norm, 4)}


def _click_to_start(driver, game_element, window_rect):
    """Click the canvas center to start/unpause the game."""
    from selenium.webdriver.common.action_chains import ActionChains

    ActionChains(driver).move_to_element(game_element).click().perform()
    time.sleep(0.5)


def main() -> int:
    parser = base_argparser("Capture Breakout 71 frames for YOLO training dataset.")
    parser.add_argument(
        "--frames",
        type=int,
        default=300,
        help="Number of frames to capture (default: %(default)s)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Seconds between captures (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip npm install step",
    )
    parser.add_argument(
        "--browser",
        type=str,
        default=None,
        help="Browser to use (chrome, edge, firefox). Default: auto-detect.",
    )
    parser.add_argument(
        "--action-interval",
        type=int,
        default=3,
        help="Perform a random action every N frames (default: %(default)s, min: 1)",
    )
    parser.add_argument(
        "--no-click-start",
        action="store_false",
        dest="click_start",
        help="Skip clicking the canvas to start the game",
    )
    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.action_interval < 1:
        parser.error("--action-interval must be >= 1")

    from dotenv import load_dotenv

    load_dotenv()

    from src.game_loader import load_game_config, create_loader
    from src.capture import WindowCapture

    config = load_game_config(args.config)
    loader = create_loader(config)

    # ── Start game server ────────────────────────────────────────────
    if not args.skip_setup:
        logger.info("Running setup (npm install) ...")
        loader.setup()

    logger.info("Starting game server ...")
    with Timer("start") as t:
        loader.start()
    logger.info("Game ready at %s (%.1fs)", loader.url, t.elapsed)

    # ── Launch browser ───────────────────────────────────────────────
    url = loader.url or config.url
    logger.info("Launching browser: %s", url)
    browser = BrowserInstance(
        url,
        settle_seconds=5,
        window_size=(config.window_width, config.window_height),
        browser=args.browser,
    )

    # ── Set up capture ───────────────────────────────────────────────
    window_title = config.window_title or "Breakout"
    cap = WindowCapture(window_title=window_title)
    logger.info("Window: HWND=%s, %dx%d", cap.hwnd, cap.width, cap.height)

    # Find the game canvas for Selenium actions
    driver = browser.driver
    try:
        game_canvas = driver.find_element("css selector", "#game")
        canvas_rect = game_canvas.rect  # {x, y, width, height}
        canvas_dims = (
            int(canvas_rect["x"]),
            int(canvas_rect["y"]),
            int(canvas_rect["width"]),
            int(canvas_rect["height"]),
        )
        logger.info("Game canvas: %s", canvas_dims)
    except Exception:
        logger.warning("Could not find #game canvas — using full window for actions")
        game_canvas = driver.find_element("css selector", "body")
        canvas_dims = (0, 0, config.window_width, config.window_height)

    # ── Click to start the game ──────────────────────────────────────
    if args.click_start:
        logger.info("Clicking canvas to start game ...")
        _click_to_start(driver, game_canvas, canvas_dims)
        time.sleep(1.0)

    # ── Capture loop ─────────────────────────────────────────────────
    ts = timestamp_str()
    out_dir = (
        ensure_output_dir(f"dataset_{ts}")
        if args.output_dir is None
        else args.output_dir
    )
    logger.info("Saving frames to: %s", out_dir)

    manifest: list[dict] = []
    capture_times: list[float] = []
    state_counts: dict[str, int] = {}
    modal_actions: list[dict] = []

    for i in range(args.frames):
        # ── Detect game state and handle modals ──────────────────────
        state_info = _detect_game_state(driver)
        game_state = state_info["state"]
        state_counts[game_state] = state_counts.get(game_state, 0) + 1

        modal_action = None
        if game_state in (STATE_PERK_PICKER, STATE_GAME_OVER, STATE_MENU):
            # Capture the modal frame BEFORE dismissing it — these are
            # still useful for training (the model should learn to
            # recognize non-gameplay states).
            modal_action = _handle_modal(driver, state_info)
            modal_actions.append({"frame": i, "state": game_state, **modal_action})
            # Wait for the modal animation to finish
            time.sleep(1.0)

            # After dismissing game-over, click canvas to start playing
            if game_state == STATE_GAME_OVER:
                try:
                    _click_to_start(driver, game_canvas, canvas_dims)
                except Exception as exc:
                    logger.debug("Post-restart click failed: %s", exc)

            # After perk pick, the game auto-resumes — but sometimes
            # it pauses briefly.  Click canvas to be safe.
            if game_state == STATE_PERK_PICKER:
                time.sleep(0.5)
                try:
                    _click_to_start(driver, game_canvas, canvas_dims)
                except Exception as exc:
                    logger.debug("Post-perk click failed: %s", exc)

        elif game_state == STATE_PAUSED:
            # Click to unpause
            try:
                _click_to_start(driver, game_canvas, canvas_dims)
            except Exception as exc:
                logger.debug("Unpause click failed: %s", exc)

        # ── Periodic random paddle action during gameplay ────────────
        action_meta = None
        if (
            game_state in (STATE_GAMEPLAY, STATE_UNKNOWN)
            and i % args.action_interval == 0
        ):
            try:
                action_meta = _random_paddle_action(driver, game_canvas, canvas_dims)
            except Exception as exc:
                logger.debug("Action failed (frame %d): %s", i, exc)

        # ── Capture frame ────────────────────────────────────────────
        with Timer(f"frame_{i}") as ft:
            frame = cap.capture_frame()

        capture_times.append(ft.elapsed)

        # Save frame
        frame_name = f"frame_{i:05d}.png"
        frame_path = out_dir / frame_name
        save_frame_png(frame, frame_path)

        # Record metadata
        entry = {
            "index": i,
            "filename": frame_name,
            "timestamp": time.time(),
            "shape": list(frame.shape),
            "capture_ms": round(ft.elapsed * 1000, 1),
            "game_state": game_state,
        }
        if action_meta:
            entry["action"] = action_meta
        if modal_action:
            entry["modal_action"] = modal_action
        manifest.append(entry)

        if i % 50 == 0 or i == args.frames - 1:
            logger.info(
                "Frame %4d/%d  state=%-12s shape=%s  latency=%.1fms",
                i + 1,
                args.frames,
                game_state,
                frame.shape,
                ft.elapsed * 1000,
            )

        if i < args.frames - 1:
            time.sleep(args.interval)

    # ── Save manifest ────────────────────────────────────────────────
    manifest_path = out_dir / "manifest.json"
    manifest_data = {
        "dataset": "breakout71",
        "capture_timestamp": ts,
        "total_frames": len(manifest),
        "interval_seconds": args.interval,
        "action_interval_frames": args.action_interval,
        "window_size": [config.window_width, config.window_height],
        "classes": ["paddle", "ball", "brick", "powerup", "wall"],
        "state_counts": state_counts,
        "modal_actions": modal_actions,
        "frames": manifest,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    logger.info("Manifest saved: %s", manifest_path)

    # ── Summary ──────────────────────────────────────────────────────
    import numpy as np

    times = np.array(capture_times)
    logger.info("--- Capture Summary ---")
    logger.info("Frames captured : %d", len(times))
    logger.info("Avg latency     : %.1f ms", times.mean() * 1000)
    logger.info("Min latency     : %.1f ms", times.min() * 1000)
    logger.info("Max latency     : %.1f ms", times.max() * 1000)
    logger.info(
        "Avg FPS         : %.0f",
        1.0 / times.mean() if times.mean() > 0 else 0,
    )
    logger.info("Output dir      : %s", out_dir)
    logger.info(
        "Total duration  : %.1f s",
        args.frames * args.interval,
    )
    logger.info("--- State Distribution ---")
    for state_name, count in sorted(state_counts.items()):
        logger.info(
            "  %-15s : %d (%.1f%%)", state_name, count, count / len(times) * 100
        )
    if modal_actions:
        logger.info("Modal interactions: %d", len(modal_actions))

    # ── Cleanup ──────────────────────────────────────────────────────
    cap.release()
    logger.info("Shutting down ...")
    browser.close()
    loader.stop()
    logger.info("Done — %d frames ready for annotation", len(manifest))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as exc:
        logger.critical("FAILED: %s", exc, exc_info=True)
        sys.exit(1)
