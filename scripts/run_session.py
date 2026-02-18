#!/usr/bin/env python
"""CLI entry point -- run N episodes and generate a QA report.

Supports both random policy (default) and trained model evaluation::

    # Random policy (baseline):
    python scripts/run_session.py --game breakout71 \\
        --episodes 3 --max-steps 10000 --browser chrome

    # Evaluate a trained MLP model:
    python scripts/run_session.py --game breakout71 \\
        --model models/ppo_breakout71.zip --episodes 10

    # Evaluate a trained CNN model:
    python scripts/run_session.py --game breakout71 \\
        --model models/ppo_breakout71_cnn.zip --episodes 10 \\
        --policy cnn --frame-stack 4

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
        description="Run N QA episodes and generate report (random or trained policy).",
    )
    parser.add_argument(
        "--game",
        type=str,
        default="breakout71",
        help=("Game plugin name (directory under games/). Default: breakout71"),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Path to a trained SB3 PPO model (.zip).  When provided, the "
            "model's policy is used instead of random action sampling."
        ),
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
        "--headless",
        action="store_true",
        help="Run browser in headless mode (no GUI, Selenium frame capture)",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="mlp",
        choices=["mlp", "cnn"],
        help=(
            "Observation policy type.  'mlp' (default) uses the raw feature "
            "vector; 'cnn' wraps the env to produce stacked grayscale image "
            "observations matching the CNN training pipeline."
        ),
    )

    def _positive_int(value: str) -> int:
        """Argparse type that enforces a positive integer (>= 1)."""
        ival = int(value)
        if ival < 1:
            raise argparse.ArgumentTypeError(f"--frame-stack must be >= 1, got {ival}")
        return ival

    parser.add_argument(
        "--frame-stack",
        type=_positive_int,
        default=4,
        help=(
            "Number of frames to stack when --policy=cnn (default: 4).  "
            "Ignored when --policy=mlp."
        ),
    )
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="yolo",
        choices=["yolo", "survival"],
        help=(
            "Reward signal strategy.  'yolo' (default) uses game-specific "
            "YOLO-based reward.  'survival' uses +0.01 per step, -5.0 on "
            "game over, +5.0 on level clear."
        ),
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

    # Silence noisy third-party loggers
    for noisy in (
        "selenium",
        "urllib3",
        "asyncio",
        "PIL",
        "ultralytics",
        "matplotlib",
        "parso",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from src.orchestrator.session_runner import SessionRunner

    # Build policy_fn from trained model if provided
    policy_fn = None
    if args.model:
        from stable_baselines3 import PPO

        logger.info("Loading trained model from %s", args.model)
        model = PPO.load(args.model)

        def policy_fn(obs):
            action, _ = model.predict(obs, deterministic=True)
            return action

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
        policy_fn=policy_fn,
        headless=args.headless,
        policy=args.policy,
        frame_stack=args.frame_stack,
        reward_mode=args.reward_mode,
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
