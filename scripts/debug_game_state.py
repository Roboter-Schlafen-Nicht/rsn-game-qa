#!/usr/bin/env python
"""Diagnostic script: test game-state JS detection on a live game.

Usage::

    python scripts/debug_game_state.py --browser chrome

Opens the game and polls the game-state JS every 2 seconds, printing
what it detects.  Let the ball die to see if game-over is detected.
Press Ctrl+C to stop.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._smoke_utils import BrowserInstance
from games.breakout71.modal_handler import (
    CLICK_PERK_JS,
    DETECT_STATE_JS,
    DISMISS_GAME_OVER_JS,
    DISMISS_MENU_JS,
)
from src.game_loader import create_loader
from src.game_loader.config import load_game_config


def main() -> None:
    """Run the diagnostic loop."""
    import argparse

    parser = argparse.ArgumentParser(description="Debug game-state JS detection")
    parser.add_argument("--browser", default="chrome")
    parser.add_argument("--config", default="configs/games/breakout-71.yaml")
    parser.add_argument(
        "--auto-dismiss",
        action="store_true",
        help="Automatically dismiss detected modals",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_game_config(config_path.stem, configs_dir=config_path.parent)

    loader = create_loader(config)
    print("Setting up game...")
    loader.setup()
    print("Starting dev server...")
    loader.start()
    url = loader.url or config.url
    print(f"Dev server ready at {url}")

    browser = BrowserInstance(
        url=url,
        settle_seconds=5.0,
        window_size=(config.window_width, config.window_height),
        browser=args.browser,
    )

    driver = browser.driver
    print("\nPolling game state every 2 seconds. Ctrl+C to stop.\n")

    try:
        while True:
            try:
                result = driver.execute_script(DETECT_STATE_JS)
                print(
                    f"[{time.strftime('%H:%M:%S')}] State: {json.dumps(result, indent=2)}"
                )

                if args.auto_dismiss and result:
                    state = result.get("state", "unknown")
                    if state == "game_over":
                        print("  -> Attempting dismiss...")
                        dismiss_result = driver.execute_script(DISMISS_GAME_OVER_JS)
                        print(f"  -> Dismiss result: {json.dumps(dismiss_result)}")
                    elif state == "perk_picker":
                        print("  -> Clicking perk...")
                        perk_result = driver.execute_script(CLICK_PERK_JS)
                        print(f"  -> Perk result: {json.dumps(perk_result)}")
                    elif state == "menu":
                        print("  -> Dismissing menu...")
                        menu_result = driver.execute_script(DISMISS_MENU_JS)
                        print(f"  -> Menu result: {json.dumps(menu_result)}")

            except Exception as exc:
                print(f"[{time.strftime('%H:%M:%S')}] Error: {exc}")

            time.sleep(2.0)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        browser.close()
        loader.stop()


if __name__ == "__main__":
    main()
