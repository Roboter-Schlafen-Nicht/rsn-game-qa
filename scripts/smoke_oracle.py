#!/usr/bin/env python
"""Smoke test: run oracles against a live Breakout 71 session.

Starts the game, captures frames in a step loop, feeds each frame to a
set of oracles, then saves the oracle findings as a JSON report.

Usage::

    python scripts/smoke_oracle.py
    python scripts/smoke_oracle.py --steps 200 --step-interval 0.1
    python scripts/smoke_oracle.py --skip-setup -v
"""

from __future__ import annotations

import json
import logging
import sys
import time

import numpy as np

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


def _build_oracles() -> list:
    """Instantiate a representative set of oracles for smoke testing."""
    from src.oracles import (
        CrashOracle,
        EpisodeLengthOracle,
        PerformanceOracle,
        StuckOracle,
        TemporalAnomalyOracle,
        VisualGlitchOracle,
    )

    return [
        CrashOracle(),
        StuckOracle(patience=100),
        PerformanceOracle(min_fps=10.0, sustained_frames=10),
        VisualGlitchOracle(),
        EpisodeLengthOracle(min_steps=5, max_steps=5000),
        TemporalAnomalyOracle(),
    ]


def _finding_to_dict(finding) -> dict:
    """Serialize a Finding to a JSON-safe dict."""
    return {
        "oracle": finding.oracle_name,
        "severity": finding.severity,
        "step": finding.step,
        "description": finding.description,
        "data": finding.data,
    }


def main() -> int:
    parser = base_argparser("Run oracles against a live Breakout 71 session.")
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of observation steps (default: %(default)s)",
    )
    parser.add_argument(
        "--step-interval",
        type=float,
        default=0.2,
        help="Seconds between steps (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip npm install",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Also save every captured frame as PNG",
    )
    args = parser.parse_args()
    setup_logging(args.verbose)

    from src.capture import WindowCapture
    from src.game_loader import create_loader, load_game_config

    config = load_game_config(args.config)
    loader = create_loader(config)

    # ── Start game ───────────────────────────────────────────────────
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

    # ── Setup oracles ────────────────────────────────────────────────
    oracles = _build_oracles()
    oracle_names = [o.name for o in oracles]
    logger.info("Oracles active: %s", ", ".join(oracle_names))

    ts = timestamp_str()
    out_dir = ensure_output_dir(f"oracle_run_{ts}") if args.output_dir is None else args.output_dir

    # Initial observation (simulates env.reset)
    frame = cap.capture_frame()
    obs = frame.astype(np.float32) / 255.0
    info = {"frame": frame}
    for oracle in oracles:
        oracle.on_reset(obs, info)

    # ── Step loop ────────────────────────────────────────────────────
    logger.info("Running %d steps (interval=%.2fs) ...", args.steps, args.step_interval)
    step_times: list[float] = []

    for step in range(args.steps):
        time.sleep(args.step_interval)

        with Timer("step") as st:
            frame = cap.capture_frame()

        step_times.append(st.elapsed)
        obs = frame.astype(np.float32) / 255.0

        # Synthetic info dict — a real env would provide game state
        info = {"frame": frame}

        # Synthetic reward/terminated (we're just observing, not playing)
        reward = 0.0
        terminated = False
        truncated = False

        for oracle in oracles:
            oracle.on_step(obs, reward, terminated, truncated, info)

        if args.save_frames:
            save_frame_png(frame, out_dir / f"frame_{step:04d}.png")

        if (step + 1) % 25 == 0:
            total_findings = sum(len(o.get_findings()) for o in oracles)
            logger.info(
                "Step %d/%d  latency=%.1fms  findings_so_far=%d",
                step + 1,
                args.steps,
                st.elapsed * 1000,
                total_findings,
            )

    # ── Collect findings ─────────────────────────────────────────────
    all_findings = []
    for oracle in oracles:
        findings = oracle.get_findings()
        all_findings.extend(findings)

    # ── Summary ──────────────────────────────────────────────────────
    logger.info("--- Oracle Run Summary ---")
    logger.info("Steps completed  : %d", args.steps)
    logger.info("Total findings   : %d", len(all_findings))

    severity_counts = {"critical": 0, "warning": 0, "info": 0}
    for f in all_findings:
        severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1

    for sev, count in severity_counts.items():
        if count > 0:
            logger.info("  %-12s: %d", sev, count)

    # Per-oracle breakdown
    logger.info("--- Per-Oracle Breakdown ---")
    for oracle in oracles:
        findings = oracle.get_findings()
        logger.info("  %-25s: %d findings", oracle.name, len(findings))
        for f in findings:
            logger.info("    [%s] step %d: %s", f.severity, f.step, f.description)

    # ── Save JSON report ─────────────────────────────────────────────
    times_arr = np.array(step_times)
    report = {
        "timestamp": ts,
        "config": args.config,
        "steps": args.steps,
        "step_interval_s": args.step_interval,
        "oracles": oracle_names,
        "capture_stats": {
            "avg_latency_ms": float(times_arr.mean() * 1000),
            "min_latency_ms": float(times_arr.min() * 1000),
            "max_latency_ms": float(times_arr.max() * 1000),
        },
        "summary": {
            "total_findings": len(all_findings),
            "by_severity": severity_counts,
            "by_oracle": {o.name: len(o.get_findings()) for o in oracles},
        },
        "findings": [_finding_to_dict(f) for f in all_findings],
    }

    report_path = out_dir / "oracle_report.json"
    with open(report_path, "w") as fp:
        json.dump(report, fp, indent=2)
    logger.info("Report saved: %s", report_path)

    # ── Save proof frame ─────────────────────────────────────────────
    save_frame_png(frame, out_dir / "final_frame.png")
    logger.info("Final frame saved: %s", out_dir / "final_frame.png")

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
