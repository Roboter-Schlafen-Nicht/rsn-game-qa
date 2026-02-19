#!/usr/bin/env python
"""Smoke test: capture multiple frames from a running Breakout 71 instance.

Starts the game, captures N frames at a configurable interval, saves
them as PNGs, and logs per-frame stats (dimensions, capture latency,
effective FPS).

Usage::

    python scripts/smoke_capture.py
    python scripts/smoke_capture.py --frames 20 --interval 0.5
    python scripts/smoke_capture.py --skip-setup -v
"""

from __future__ import annotations

import logging
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


def main() -> int:
    parser = base_argparser("Capture multiple frames from Breakout 71.")
    parser.add_argument(
        "--frames",
        type=int,
        default=10,
        help="Number of frames to capture (default: %(default)s)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between captures (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip npm install",
    )
    args = parser.parse_args()
    setup_logging(args.verbose)

    from src.capture import WindowCapture
    from src.game_loader import create_loader, load_game_config

    config = load_game_config(args.config)
    loader = create_loader(config)

    # ── Lifecycle ────────────────────────────────────────────────────
    if not args.skip_setup:
        logger.info("Running setup ...")
        loader.setup()

    logger.info("Starting game ...")
    with Timer("start") as t:
        loader.start()
    logger.info("Game ready at %s (%.1fs)", loader.url, t.elapsed)

    # Open a dedicated browser window so WindowCapture can find the game
    url = loader.url or config.url
    logger.info("Launching dedicated browser window: %s", url)
    browser = BrowserInstance(
        url,
        settle_seconds=5,
        window_size=(config.window_width, config.window_height),
    )

    window_title = config.window_title or "Breakout"
    cap = WindowCapture(window_title=window_title)
    logger.info("Window: HWND=%s, %dx%d", cap.hwnd, cap.width, cap.height)

    # ── Capture loop ─────────────────────────────────────────────────
    ts = timestamp_str()
    out_dir = ensure_output_dir(f"frames_{ts}") if args.output_dir is None else args.output_dir
    logger.info("Saving frames to: %s", out_dir)

    capture_times: list[float] = []

    for i in range(args.frames):
        with Timer(f"frame_{i}") as ft:
            frame = cap.capture_frame()

        capture_times.append(ft.elapsed)
        fps = 1.0 / ft.elapsed if ft.elapsed > 0 else float("inf")

        frame_path = out_dir / f"frame_{i:04d}.png"
        save_frame_png(frame, frame_path)

        logger.info(
            "Frame %3d/%d  shape=%s  latency=%.1fms  fps=%.0f  -> %s",
            i + 1,
            args.frames,
            frame.shape,
            ft.elapsed * 1000,
            fps,
            frame_path.name,
        )

        if i < args.frames - 1:
            time.sleep(args.interval)

    # ── Summary ──────────────────────────────────────────────────────
    import numpy as np

    times = np.array(capture_times)
    logger.info("--- Capture Summary ---")
    logger.info("Frames captured : %d", len(times))
    logger.info("Avg latency     : %.1f ms", times.mean() * 1000)
    logger.info("Min latency     : %.1f ms", times.min() * 1000)
    logger.info("Max latency     : %.1f ms", times.max() * 1000)
    logger.info("Avg FPS         : %.0f", 1.0 / times.mean() if times.mean() > 0 else 0)
    logger.info("Output dir      : %s", out_dir)

    # ── Cleanup ──────────────────────────────────────────────────────
    cap.release()
    logger.info("Shutting down ...")
    browser.close()
    loader.stop()
    logger.info("Done")
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
