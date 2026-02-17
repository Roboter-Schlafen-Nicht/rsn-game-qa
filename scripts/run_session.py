#!/usr/bin/env python
"""CLI entry point -- run N episodes with random policy and generate a QA report.

Usage::

    python scripts/run_session.py --game breakout71 \\
        --episodes 3 --max-steps 10000 --browser chrome

    # Override config / weights (optional):
    python scripts/run_session.py --game breakout71 \\
        --config configs/games/breakout-71.yaml \\
        --yolo-weights weights/breakout71/best.pt

Requires a running game server (see ``smoke_launch.py`` or start manually).
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments.  If None, uses ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run N QA episodes with random policy and generate report.",
    )
    parser.add_argument(
        "--game",
        type=str,
        default="breakout71",
        help=("Game plugin name (directory under games/). Default: breakout71"),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to game config YAML.  If omitted, uses the plugin's default config."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run (default: 3)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10_000,
        help="Max steps per episode before truncation (default: 10000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for reports and data (default: output)",
    )
    parser.add_argument(
        "--browser",
        type=str,
        default=None,
        choices=["chrome", "msedge", "firefox"],
        help="Browser to use (default: auto-select)",
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default=None,
        help=("Path to YOLO weights.  If omitted, uses the plugin's default weights."),
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=30,
        help="Capture every Nth frame for data collection (default: 30)",
    )
    parser.add_argument(
        "--no-data-collection",
        action="store_true",
        help="Disable frame collection for YOLO retraining",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the QA session.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    from src.orchestrator.session_runner import SessionRunner

    runner = SessionRunner(
        game=args.game,
        game_config=args.config,
        n_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        output_dir=args.output_dir,
        browser=args.browser,
        yolo_weights=args.yolo_weights,
        frame_capture_interval=args.frame_interval,
        enable_data_collection=not args.no_data_collection,
    )

    report = runner.run()

    # Print summary
    summary = report.summary
    print("\n--- Session Summary ---")
    print(f"Episodes:        {summary.get('total_episodes', 0)}")
    print(f"Total findings:  {summary.get('total_findings', 0)}")
    print(f"  Critical:      {summary.get('critical_findings', 0)}")
    print(f"  Warning:       {summary.get('warning_findings', 0)}")
    print(f"  Info:          {summary.get('info_findings', 0)}")
    print(f"Episodes failed: {summary.get('episodes_failed', 0)}")
    print(f"Mean reward:     {summary.get('mean_episode_reward', 0.0):.2f}")
    print(f"Mean length:     {summary.get('mean_episode_length', 0.0):.0f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
