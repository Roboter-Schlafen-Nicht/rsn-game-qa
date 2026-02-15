#!/usr/bin/env python
"""CLI entry point -- PPO training with Stable Baselines3 on Breakout71Env.

Usage::

    python scripts/train_rl.py --config configs/games/breakout-71.yaml \\
        --timesteps 200000 --browser chrome

Requires a running game server.  Training runs in real-time (~30 FPS),
so 200k steps takes ~1.85 hours.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

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
        description="Train PPO agent on Breakout 71 with data collection.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/games/breakout-71.yaml",
        help="Path to game config YAML (default: configs/games/breakout-71.yaml)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Total training timesteps (default: 200000)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50_000,
        help="Save checkpoint every N timesteps (default: 50000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory (default: output)",
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
        default="weights/breakout71/best.pt",
        help="Path to YOLO weights (default: weights/breakout71/best.pt)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10_000,
        help="Max steps per episode (default: 10000)",
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
    # PPO hyperparameters
    parser.add_argument("--n-steps", type=int, default=2048, help="PPO n_steps")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO batch_size")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO n_epochs")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coeff")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args(argv)


class FrameCollectionCallback:
    """SB3 callback for frame collection, oracle findings, and episode metrics.

    Collects frames via ``FrameCollector`` at the configured interval
    and logs oracle findings per episode.

    Parameters
    ----------
    collector : FrameCollector or None
        Frame collector instance.  If None, no frames are saved.
    checkpoint_interval : int
        Save PPO checkpoint every N timesteps.
    checkpoint_dir : Path
        Directory for checkpoint files.
    """

    def __init__(
        self,
        collector: Any | None = None,
        checkpoint_interval: int = 50_000,
        checkpoint_dir: Path = Path("output/checkpoints"),
    ) -> None:
        # Import at call site to avoid top-level SB3 import in CI
        from stable_baselines3.common.callbacks import BaseCallback

        self._base_class = BaseCallback
        self._collector = collector
        self._checkpoint_interval = checkpoint_interval
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._episode_count = 0
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []

    def create(self) -> Any:
        """Create and return the actual SB3 BaseCallback subclass instance.

        Returns
        -------
        BaseCallback
            The SB3-compatible callback.
        """
        from stable_baselines3.common.callbacks import BaseCallback

        collector = self._collector
        checkpoint_interval = self._checkpoint_interval
        checkpoint_dir = self._checkpoint_dir
        episode_rewards = self._episode_rewards
        episode_lengths = self._episode_lengths

        class _Callback(BaseCallback):
            """Inner SB3 callback with closure over collector state."""

            def __init__(self) -> None:
                super().__init__(verbose=0)
                self._last_checkpoint = 0
                self._episode_count = 0

            def _on_step(self) -> bool:
                # Collect frame if available
                if collector is not None:
                    infos = self.locals.get("infos", [])
                    for info in infos:
                        frame = info.get("frame")
                        if frame is not None:
                            step = info.get("step", self.num_timesteps)
                            collector.save_frame(
                                frame=frame,
                                step=step,
                                episode_id=self._episode_count,
                            )

                # Log episode completions
                dones = self.locals.get("dones", [])
                for i, done in enumerate(dones):
                    if done:
                        self._episode_count += 1
                        infos = self.locals.get("infos", [])
                        if i < len(infos):
                            ep_info = infos[i].get("episode")
                            if ep_info:
                                episode_rewards.append(ep_info.get("r", 0.0))
                                episode_lengths.append(ep_info.get("l", 0))

                # Periodic checkpointing
                if self.num_timesteps - self._last_checkpoint >= checkpoint_interval:
                    ckpt_path = checkpoint_dir / f"ppo_breakout71_{self.num_timesteps}"
                    self.model.save(str(ckpt_path))
                    logger.info(
                        "Checkpoint saved: %s (step %d)",
                        ckpt_path,
                        self.num_timesteps,
                    )
                    self._last_checkpoint = self.num_timesteps

                return True

        return _Callback()


def main(argv: list[str] | None = None) -> int:
    """Run PPO training on Breakout71Env.

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

    # Lazy imports to avoid CI failures
    from stable_baselines3 import PPO

    from scripts._smoke_utils import BrowserInstance
    from src.env.breakout71_env import Breakout71Env
    from src.game_loader.config import load_game_config
    from src.orchestrator.data_collector import FrameCollector
    from src.orchestrator.session_runner import _create_oracles
    from src.reporting import ReportGenerator

    output_dir = Path(args.output_dir)

    # Load game config
    config_path = Path(args.config)
    config = load_game_config(
        config_path.stem,
        configs_dir=config_path.parent,
    )

    # Launch browser
    browser_instance = BrowserInstance(
        url=config.url,
        settle_seconds=8.0,
        window_size=(config.window_width, config.window_height),
        browser=args.browser,
    )

    try:
        # Create environment
        oracles = _create_oracles()
        window_title = config.window_title or "Breakout"

        env = Breakout71Env(
            window_title=window_title,
            yolo_weights=args.yolo_weights,
            max_steps=args.max_steps,
            oracles=oracles,
        )

        # Create frame collector
        collector = None
        if not args.no_data_collection:
            collector = FrameCollector(
                output_dir=output_dir,
                capture_interval=args.frame_interval,
            )

        # Create callback
        checkpoint_dir = output_dir / "checkpoints"
        cb_factory = FrameCollectionCallback(
            collector=collector,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_dir=checkpoint_dir,
        )
        callback = cb_factory.create()

        # Create PPO model
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            learning_rate=args.lr,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            device=args.device,
            verbose=1,
        )

        logger.info(
            "Starting PPO training: %d timesteps, device=%s",
            args.timesteps,
            args.device,
        )
        train_start = time.perf_counter()

        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            progress_bar=True,
        )

        train_elapsed = time.perf_counter() - train_start
        logger.info("Training complete in %.1f seconds", train_elapsed)

        # Save final model
        model_path = output_dir / "models" / "ppo_breakout71_final"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        logger.info("Final model saved to %s", model_path)

        # Finalize data collection
        if collector is not None:
            frames_dir = collector.finalize()
            logger.info(
                "Collected %d frames in %s",
                collector.frame_count,
                frames_dir,
            )

        # Generate report
        report_gen = ReportGenerator(
            output_dir=output_dir / "reports",
            game_name="breakout-71",
        )
        report_path = report_gen.save()
        logger.info("Report saved to %s", report_path)

        # Print summary
        print("\n--- Training Summary ---")
        print(f"Total timesteps: {args.timesteps}")
        print(f"Training time:   {train_elapsed:.1f}s")
        print(f"Episodes:        {len(cb_factory._episode_rewards)}")
        if cb_factory._episode_rewards:
            import numpy as np

            print(f"Mean reward:     {np.mean(cb_factory._episode_rewards):.2f}")
            print(f"Mean length:     {np.mean(cb_factory._episode_lengths):.0f}")
        print(f"Model saved:     {model_path}")
        if collector is not None:
            print(f"Frames saved:    {collector.frame_count}")

    finally:
        if "env" in locals():
            env.close()
        browser_instance.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
