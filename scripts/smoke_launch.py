#!/usr/bin/env python
"""Smoke test: launch Breakout 71, verify readiness, capture a proof frame, shut down.

Usage::

    python scripts/smoke_launch.py
    python scripts/smoke_launch.py --wait 30        # wait 30s instead of 60
    python scripts/smoke_launch.py --skip-setup      # skip npm install
    python scripts/smoke_launch.py -v                # debug logging
"""

from __future__ import annotations

import logging
import sys
import time

# Allow running from project root
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


def main() -> int:
    parser = base_argparser("Launch Breakout 71, verify readiness, and shut down.")
    parser.add_argument(
        "--wait",
        type=int,
        default=60,
        help="Seconds to keep the game running (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip npm install (faster if already installed)",
    )
    args = parser.parse_args()
    setup_logging(args.verbose)

    # ── Load config ──────────────────────────────────────────────────
    from src.game_loader import create_loader, load_game_config

    logger.info("Loading config: %s", args.config)
    config = load_game_config(args.config)
    loader = create_loader(config)

    # ── Setup ────────────────────────────────────────────────────────
    if not args.skip_setup:
        logger.info("Running setup (npm install) ...")
        with Timer("setup") as t:
            loader.setup()
        logger.info("Setup complete (%.1fs)", t.elapsed)
    else:
        logger.info("Skipping setup (--skip-setup)")

    # ── Start ────────────────────────────────────────────────────────
    logger.info("Starting game server ...")
    with Timer("start") as t:
        loader.start()
    logger.info(
        "Game is READY at %s (PID %s, took %.1fs)",
        loader.url,
        getattr(loader, "_process", None) and loader._process.pid,
        t.elapsed,
    )

    # ── Verify readiness ─────────────────────────────────────────────
    assert loader.running, "Loader reports not running!"
    assert loader.is_ready(), "Loader readiness check failed!"
    logger.info("Readiness check: PASSED")

    # ── Open browser ─────────────────────────────────────────────────
    url = loader.url or config.url
    logger.info("Launching dedicated browser window: %s", url)
    browser = BrowserInstance(
        url,
        settle_seconds=5,
        window_size=(config.window_width, config.window_height),
    )

    # ── Try to capture a proof frame ─────────────────────────────────
    try:
        from src.capture import WindowCapture

        window_title = config.window_title or "Breakout"
        logger.info("Looking for window: %r", window_title)
        cap = WindowCapture(window_title=window_title)
        logger.info(
            "Window found: HWND=%s, size=%dx%d",
            cap.hwnd,
            cap.width,
            cap.height,
        )

        frame = cap.capture_frame()
        logger.info("Frame captured: shape=%s, dtype=%s", frame.shape, frame.dtype)

        # Save proof screenshot
        out_dir = ensure_output_dir() if args.output_dir is None else args.output_dir
        proof_path = out_dir / f"proof_{timestamp_str()}.png"
        save_frame_png(frame, proof_path)
        logger.info("Proof screenshot saved: %s", proof_path)

        cap.release()
    except Exception as exc:
        logger.warning("Frame capture skipped: %s", exc)

    # ── Wait ─────────────────────────────────────────────────────────
    logger.info("Keeping game alive for %ds ...", args.wait)
    time.sleep(args.wait)

    # ── Stop ─────────────────────────────────────────────────────────
    logger.info("Shutting down ...")
    browser.close()
    with Timer("stop") as t:
        loader.stop()
    logger.info("Shutdown complete (%.1fs)", t.elapsed)

    assert not loader.running, "Loader still reports running after stop!"
    logger.info("All checks PASSED")
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
