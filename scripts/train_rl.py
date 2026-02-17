#!/usr/bin/env python
"""CLI entry point -- PPO training with Stable Baselines3 on any game plugin.

Usage::

    python scripts/train_rl.py --game breakout71 --timesteps 200000

    # CNN policy (pixel-based, uses GPU for policy network):
    python scripts/train_rl.py --game breakout71 --policy cnn --timesteps 200000

    # Headless mode (no mouse capture, uses Selenium for I/O):
    python scripts/train_rl.py --game breakout71 --headless --timesteps 200000

    # Portrait mode (768x1024, mobile-style layout):
    python scripts/train_rl.py --game breakout71 --orientation portrait --headless

Supports two policy types for A/B comparison:

- **mlp** (default): 8-element YOLO feature vector -> MlpPolicy (CPU)
- **cnn**: 84x84 grayscale pixels -> NatureCNN -> CnnPolicy (GPU)

The CNN pipeline stacks 4 consecutive frames (``VecFrameStack``) so the
policy can infer velocity from motion.  Both pipelines use the same
reward signal and episode boundaries from YOLO detection.

Launches the dev server automatically via GameLoader.  Training runs in
real-time; at ~52 FPS native or ~2-3 FPS headless.

All runs produce two log files in ``output/``:

- ``training_<timestamp>.log`` -- human-readable, line-buffered
- ``training_<timestamp>.jsonl`` -- structured events (config, step
  summaries, episode completions, checkpoints, interrupts, final
  summary), line-buffered for crash safety
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

# Ensure project root is on sys.path so ``src.*`` and ``scripts.*`` resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Orientation presets
# ---------------------------------------------------------------------------

_ORIENTATION_PRESETS: dict[str, tuple[int, int]] = {
    "landscape": (1280, 1024),
    "portrait": (768, 1024),
}


# ---------------------------------------------------------------------------
# JSONL structured logger
# ---------------------------------------------------------------------------


class TrainingLogger:
    """Line-buffered JSONL logger for structured training events.

    Each event is a single JSON object on its own line, flushed
    immediately so data survives Ctrl-C or crashes.

    Parameters
    ----------
    jsonl_path : Path
        Path to the ``.jsonl`` output file.
    """

    def __init__(self, jsonl_path: Path) -> None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: IO[str] = open(  # noqa: SIM115
            jsonl_path, "w", encoding="utf-8", buffering=1
        )
        atexit.register(self.close)

    def log(self, event: dict[str, Any]) -> None:
        """Write a JSON event, adding a timestamp if absent.

        Parameters
        ----------
        event : dict
            Event payload.  Must include ``"event"`` key.
        """
        event.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        self._fh.write(json.dumps(event, default=str) + "\n")

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        if self._fh and not self._fh.closed:
            self._fh.flush()
            self._fh.close()


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


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
        description="Train PPO agent on a game plugin with data collection.",
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
        choices=["chrome", "edge", "firefox"],
        help="Browser to use (default: auto-select)",
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default=None,
        help=("Path to YOLO weights.  If omitted, uses the plugin's default weights."),
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
        "--data-collection",
        action="store_true",
        help="Enable frame collection for YOLO retraining (default: off)",
    )

    # -- New flags: headless / mute / orientation --------------------------
    parser.add_argument(
        "--headless",
        action="store_true",
        help=(
            "Run browser in headless mode.  Uses Selenium screenshots "
            "for capture instead of Win32 screen capture.  "
            "Much slower (~2-3 FPS) but does not capture the mouse."
        ),
    )
    parser.add_argument(
        "--no-mute",
        action="store_false",
        dest="mute",
        help="Keep game audio on (default: muted during training)",
    )
    parser.set_defaults(mute=True)
    parser.add_argument(
        "--orientation",
        type=str,
        default="portrait",
        choices=list(_ORIENTATION_PRESETS.keys()),
        help="Window orientation preset (default: portrait = 768x1024)",
    )
    parser.add_argument(
        "--landscape",
        action="store_const",
        const="landscape",
        dest="orientation",
        help="Shorthand for --orientation landscape (1280x1024)",
    )
    parser.add_argument(
        "--window-size",
        type=str,
        default=None,
        metavar="WxH",
        help="Override window size, e.g. 768x1024 (overrides --orientation)",
    )

    # -- Policy type -------------------------------------------------------
    parser.add_argument(
        "--policy",
        type=str,
        default="mlp",
        choices=["mlp", "cnn"],
        help=(
            "Policy type: 'mlp' uses 8-element YOLO feature vector, "
            "'cnn' uses 84x84 grayscale pixel observations with "
            "4-frame stacking (default: mlp)"
        ),
    )
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=4,
        help="Number of frames to stack for CNN policy (default: 4)",
    )

    # -- Resume from checkpoint --------------------------------------------
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Resume training from a saved model/checkpoint (.zip).  "
            "Loads the model weights and optimizer state.  The --timesteps "
            "value is the TOTAL target (not additional), so set it to the "
            "original target (e.g. 200000) to complete a partial run."
        ),
    )

    # -- PPO hyperparameters -----------------------------------------------
    parser.add_argument("--n-steps", type=int, default=2048, help="PPO n_steps")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO batch_size")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO n_epochs")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coeff")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device: auto, xpu, cuda, cpu (default: auto)",
    )

    # -- Logging -----------------------------------------------------------
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Log a step_summary event every N steps (default: 100)",
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=None,
        metavar="SECONDS",
        help=(
            "Stop training after N seconds (wall-clock).  Useful for "
            "time-boxed debug runs, e.g. --max-time 180 for 3 minutes."
        ),
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Stop training after N completed episodes.  Useful for quick "
            "performance checks, e.g. --max-episodes 10."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG-level) logging",
    )

    return parser.parse_args(argv)


def resolve_window_size(args: argparse.Namespace, config: Any) -> tuple[int, int]:
    """Determine the browser window size from CLI args and game config.

    Priority: ``--window-size`` > ``--orientation`` preset > game config.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    config : GameConfig
        Loaded game configuration.

    Returns
    -------
    tuple[int, int]
        ``(width, height)`` in pixels.
    """
    if args.window_size:
        parts = args.window_size.lower().split("x")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid --window-size format: {args.window_size!r}. "
                "Expected WxH, e.g. 768x1024"
            )
        try:
            w, h = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise ValueError(
                f"Invalid --window-size values: {args.window_size!r}. "
                "Width and height must be integers, e.g. 768x1024"
            ) from exc
        if w <= 0 or h <= 0:
            raise ValueError(
                f"Invalid --window-size values: {args.window_size!r}. "
                "Width and height must be positive integers"
            )
        return w, h
    if args.orientation in _ORIENTATION_PRESETS:
        return _ORIENTATION_PRESETS[args.orientation]
    return config.window_width, config.window_height


# ---------------------------------------------------------------------------
# SB3 callback with rich logging
# ---------------------------------------------------------------------------


class FrameCollectionCallback:
    """SB3 callback for frame collection, training metrics, and JSONL logging.

    Captures frames via ``FrameCollector``, logs per-step summaries and
    per-episode completions to both the Python logger and the structured
    JSONL file, and saves periodic PPO checkpoints.

    Parameters
    ----------
    collector : FrameCollector or None
        Frame collector instance.  If None, no frames are saved.
    checkpoint_interval : int
        Save PPO checkpoint every N timesteps.
    checkpoint_dir : Path
        Directory for checkpoint files.
    training_logger : TrainingLogger or None
        Structured JSONL logger.  If None, no JSONL events are written.
    log_interval : int
        Emit a ``step_summary`` JSONL event every N timesteps.
    max_time : int or None
        Maximum training wall-clock time in seconds.  When exceeded,
        ``_on_step`` returns ``False`` to cleanly stop ``model.learn()``.
    max_episodes : int or None
        Maximum number of completed episodes.  When reached,
        ``_on_step`` returns ``False`` to cleanly stop ``model.learn()``.
    game_slug : str
        Short game identifier used for checkpoint file names
        (e.g. ``"breakout71"``).
    """

    def __init__(
        self,
        collector: Any | None = None,
        checkpoint_interval: int = 50_000,
        checkpoint_dir: Path = Path("output/checkpoints"),
        training_logger: TrainingLogger | None = None,
        log_interval: int = 100,
        max_time: int | None = None,
        max_episodes: int | None = None,
        game_slug: str = "breakout71",
    ) -> None:
        # Import at call site to avoid top-level SB3 import in CI
        from stable_baselines3.common.callbacks import BaseCallback

        self._base_class = BaseCallback
        self._collector = collector
        self._checkpoint_interval = checkpoint_interval
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._training_logger = training_logger
        self._log_interval = max(1, log_interval)
        self._max_time = max_time
        self._max_episodes = max_episodes
        self._game_slug = game_slug
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
        tlog = self._training_logger
        log_interval = self._log_interval
        max_time = self._max_time
        max_episodes = self._max_episodes
        game_slug = self._game_slug

        # We need a reference back so the outer class can read episode_count
        outer = self

        class _Callback(BaseCallback):
            """Inner SB3 callback with closure over collector state."""

            def __init__(self) -> None:
                super().__init__(verbose=0)
                self._last_checkpoint = 0
                self._episode_count = 0
                self._episode_start_time = time.perf_counter()
                self._episode_step_count = 0
                self._episode_cumulative_reward = 0.0
                self._episode_fps_samples: list[float] = []
                self._last_step_time = time.perf_counter()
                self._train_start_time = time.perf_counter()
                self.training_stop_reason: str | None = None

            def _on_step(self) -> bool:
                now = time.perf_counter()
                dt = now - self._last_step_time
                fps = 1.0 / dt if dt > 0.001 else 0.0
                self._last_step_time = now
                self._episode_fps_samples.append(fps)
                self._episode_step_count += 1

                # -- Time limit check -------------------------------------
                if max_time is not None:
                    elapsed = now - self._train_start_time
                    if elapsed >= max_time:
                        logger.info(
                            "Max time reached (%.0fs >= %ds) at step %d — "
                            "stopping training",
                            elapsed,
                            max_time,
                            self.num_timesteps,
                        )
                        if tlog:
                            tlog.log(
                                {
                                    "event": "max_time_reached",
                                    "step": self.num_timesteps,
                                    "elapsed_seconds": round(elapsed, 1),
                                    "max_time_seconds": max_time,
                                }
                            )
                        # Propagate stop reason so outer code can distinguish
                        # max-time stop from normal completion.
                        self.training_stop_reason = "max_time_reached"
                        return False

                # Accumulate reward
                rewards = self.locals.get("rewards")
                step_reward = 0.0
                if rewards is not None and len(rewards) > 0:
                    step_reward = float(rewards[0])
                    self._episode_cumulative_reward += step_reward

                # -- Periodic step summary --------------------------------
                if self.num_timesteps % log_interval == 0:
                    infos = self.locals.get("infos", [])
                    info = infos[0] if infos else {}
                    actions = self.locals.get("actions")
                    action_val = (
                        float(actions[0][0])
                        if actions is not None and len(actions) > 0
                        else None
                    )
                    paddle_pos = info.get("paddle_pos")

                    step_event = {
                        "event": "step_summary",
                        "step": self.num_timesteps,
                        "episode": self._episode_count,
                        "reward": step_reward,
                        "cumulative_reward": self._episode_cumulative_reward,
                        "ball_detected": info.get("ball_pos") is not None,
                        "brick_count": info.get("brick_count", -1),
                        "paddle_x": (paddle_pos[0] if paddle_pos is not None else None),
                        "action": action_val,
                        "fps": round(fps, 1),
                        "no_ball_count": info.get("no_ball_count", 0),
                    }
                    if tlog:
                        tlog.log(step_event)
                    logger.info(
                        "Step %d | ep %d | r=%.3f | bricks=%d | fps=%.1f",
                        self.num_timesteps,
                        self._episode_count,
                        step_reward,
                        info.get("brick_count", -1),
                        fps,
                    )

                # -- Frame collection (existing logic) --------------------
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

                # -- Episode completion -----------------------------------
                dones = self.locals.get("dones", [])
                infos = self.locals.get("infos", [])
                for i, done in enumerate(dones):
                    if done:
                        ep_duration = now - self._episode_start_time
                        mean_fps = (
                            sum(self._episode_fps_samples)
                            / len(self._episode_fps_samples)
                            if self._episode_fps_samples
                            else 0.0
                        )
                        info = infos[i] if i < len(infos) else {}
                        ep_info = info.get("episode", {})

                        ep_reward = ep_info.get("r", self._episode_cumulative_reward)
                        ep_length = ep_info.get("l", self._episode_step_count)

                        episode_rewards.append(ep_reward)
                        episode_lengths.append(ep_length)

                        # Termination reason
                        if info.get("TimeLimit.truncated", False):
                            termination = "truncated"
                        else:
                            termination = "game_over"

                        ep_event = {
                            "event": "episode_end",
                            "episode": self._episode_count,
                            "steps": ep_length,
                            "total_reward": ep_reward,
                            "mean_step_reward": (ep_reward / max(ep_length, 1)),
                            "termination": termination,
                            "brick_count": info.get("brick_count", -1),
                            "duration_seconds": round(ep_duration, 2),
                            "mean_fps": round(mean_fps, 1),
                        }
                        if tlog:
                            tlog.log(ep_event)
                        logger.info(
                            "Episode %d done: %d steps, reward=%.2f, %s, fps=%.1f",
                            self._episode_count,
                            ep_length,
                            ep_reward,
                            termination,
                            mean_fps,
                        )

                        self._episode_count += 1
                        outer._episode_count = self._episode_count
                        self._episode_start_time = now
                        self._episode_step_count = 0
                        self._episode_cumulative_reward = 0.0
                        self._episode_fps_samples = []

                        # -- Episode limit check --------------------------
                        if (
                            max_episodes is not None
                            and self._episode_count >= max_episodes
                        ):
                            logger.info(
                                "Max episodes reached (%d >= %d) at step %d "
                                "— stopping training",
                                self._episode_count,
                                max_episodes,
                                self.num_timesteps,
                            )
                            if tlog:
                                tlog.log(
                                    {
                                        "event": "max_episodes_reached",
                                        "step": self.num_timesteps,
                                        "episodes": self._episode_count,
                                        "max_episodes": max_episodes,
                                    }
                                )
                            self.training_stop_reason = "max_episodes_reached"
                            return False

                # -- Periodic checkpointing -------------------------------
                if self.num_timesteps - self._last_checkpoint >= checkpoint_interval:
                    ckpt_path = checkpoint_dir / f"ppo_{game_slug}_{self.num_timesteps}"
                    self.model.save(str(ckpt_path))
                    logger.info(
                        "Checkpoint saved: %s (step %d)",
                        ckpt_path,
                        self.num_timesteps,
                    )
                    if tlog:
                        tlog.log(
                            {
                                "event": "checkpoint",
                                "step": self.num_timesteps,
                                "path": str(ckpt_path),
                            }
                        )
                    self._last_checkpoint = self.num_timesteps

                return True

        return _Callback()


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging(
    output_dir: Path,
    verbose: bool,
) -> tuple[Path, Path, TrainingLogger, logging.FileHandler]:
    """Configure console + file logging and create the JSONL logger.

    Parameters
    ----------
    output_dir : Path
        Directory for log files.
    verbose : bool
        If True, set log level to DEBUG.

    Returns
    -------
    tuple[Path, Path, TrainingLogger, logging.FileHandler]
        ``(log_path, jsonl_path, training_logger, file_handler)``
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Console handler — INFO by default, DEBUG with -v
    console_level = logging.DEBUG if verbose else logging.INFO
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(console_level)
    console.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    )

    # File handler — always DEBUG for post-mortem analysis
    log_path = output_dir / f"training_{ts}.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    )

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)

    # Silence noisy third-party loggers on console (still DEBUG in file)
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

    # JSONL structured logger
    jsonl_path = output_dir / f"training_{ts}.jsonl"
    training_logger = TrainingLogger(jsonl_path)

    return log_path, jsonl_path, training_logger, file_handler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run PPO training on a game plugin environment.

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
    output_dir = Path(args.output_dir)

    # -- Logging setup (before anything else) ------------------------------
    log_path, jsonl_path, tlog, file_handler = _setup_logging(output_dir, args.verbose)
    logger.info("Log file:  %s", log_path)
    logger.info("JSONL log: %s", jsonl_path)

    # -- Lazy imports (avoid CI failures) ----------------------------------
    from stable_baselines3 import PPO

    from scripts._smoke_utils import BrowserInstance
    from games import load_game_plugin
    from src.platform.cnn_wrapper import CnnObservationWrapper
    from src.game_loader import create_loader
    from src.game_loader.config import load_game_config
    from src.orchestrator.data_collector import FrameCollector

    # -- Load game plugin --------------------------------------------------
    plugin = load_game_plugin(args.game)
    EnvClass = plugin.env_class
    game_name = plugin.game_name  # e.g. "breakout-71"
    game_slug = args.game  # e.g. "breakout71" (directory name, for file paths)
    logger.info("Game plugin: %s (%s)", game_slug, game_name)

    # -- Resolve config / weights from plugin defaults ---------------------
    config_path = Path(args.config if args.config else plugin.default_config)
    yolo_weights = args.yolo_weights if args.yolo_weights else plugin.default_weights

    # -- Load game config --------------------------------------------------
    config = load_game_config(
        config_path.stem,
        configs_dir=config_path.parent,
    )

    # -- Resolve window size -----------------------------------------------
    win_w, win_h = resolve_window_size(args, config)
    logger.info(
        "Window: %dx%d (orientation=%s, headless=%s)",
        win_w,
        win_h,
        args.orientation,
        args.headless,
    )

    # -- Start dev server --------------------------------------------------
    loader = create_loader(config)
    logger.info("Setting up game (npm install) ...")
    loader.setup()
    logger.info("Starting dev server ...")
    loader.start()
    logger.info("Dev server ready at %s", loader.url or config.url)

    # -- Launch browser ----------------------------------------------------
    url = loader.url or config.url
    browser_instance = BrowserInstance(
        url=url,
        settle_seconds=8.0,
        window_size=(win_w, win_h),
        browser=args.browser,
        headless=args.headless,
    )

    # -- Mute game audio ---------------------------------------------------
    mute_js = getattr(plugin, "mute_js", None)
    setup_js = getattr(plugin, "setup_js", None)
    needs_refresh = False
    if args.mute and mute_js and browser_instance.driver is not None:
        try:
            browser_instance.driver.execute_script(mute_js)
            needs_refresh = True
            logger.info("Game audio muted via plugin mute_js")
        except Exception as exc:
            logger.warning("Failed to mute game audio: %s", exc)
    if setup_js and browser_instance.driver is not None:
        try:
            browser_instance.driver.execute_script(setup_js)
            needs_refresh = True
            logger.info("Game settings configured via plugin setup_js")
        except Exception as exc:
            logger.warning("Failed to apply game setup_js: %s", exc)
    if needs_refresh:
        browser_instance.driver.refresh()
        time.sleep(3)  # let page reload with new settings

    # -- Re-initialise game after refresh ----------------------------------
    # After a refresh, window.gameState may point to a stale object
    # (newGameState/Object.assign pointer mismatch).  The plugin's
    # reinit_js calls window.restart({}) to rebuild game state and
    # re-assign the live pointer.
    reinit_js = getattr(plugin, "reinit_js", None)
    if needs_refresh and reinit_js and browser_instance.driver is not None:
        try:
            browser_instance.driver.execute_script(reinit_js)
            time.sleep(2)  # let game re-initialise
            logger.info("Game re-initialised via plugin reinit_js")
        except Exception as exc:
            logger.warning("Failed to re-initialise game: %s", exc)

    # -- Log config event --------------------------------------------------
    tlog.log(
        {
            "event": "config",
            "game": game_slug,
            "game_name": game_name,
            "args": {
                "timesteps": args.timesteps,
                "config": str(config_path),
                "headless": args.headless,
                "mute": args.mute,
                "orientation": args.orientation,
                "window_size": [win_w, win_h],
                "device": args.device,
                "policy": args.policy,
                "frame_stack": args.frame_stack,
                "yolo_weights": yolo_weights,
                "max_steps": args.max_steps,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "n_epochs": args.n_epochs,
                "gamma": args.gamma,
                "lr": args.lr,
                "clip_range": args.clip_range,
                "ent_coef": args.ent_coef,
                "frame_interval": args.frame_interval,
                "data_collection": args.data_collection,
                "log_interval": args.log_interval,
                "max_time": args.max_time,
                "max_episodes": args.max_episodes,
            },
            "browser": browser_instance.name,
            "python_version": platform.python_version(),
            "platform": sys.platform,
        }
    )

    callback = None
    vec_env = None  # track for cleanup
    try:
        # -- Create environment --------------------------------------------
        window_title = config.window_title or game_name

        env = EnvClass(
            window_title=window_title,
            yolo_weights=yolo_weights,
            max_steps=args.max_steps,
            driver=browser_instance.driver,
            device=args.device,
            headless=args.headless,
        )

        # -- Wrap for CNN policy (if requested) ----------------------------
        if args.policy == "cnn":
            from stable_baselines3.common.vec_env import (
                DummyVecEnv,
                VecFrameStack,
                VecTransposeImage,
            )

            cnn_env = CnnObservationWrapper(env)
            # NOTE: lambda captures cnn_env by reference — works with
            # DummyVecEnv but would NOT work with SubprocVecEnv (not
            # picklable).  Use a factory function if multi-process
            # vectorised envs are needed in future.
            vec_env = DummyVecEnv([lambda: cnn_env])
            vec_env = VecFrameStack(vec_env, n_stack=args.frame_stack)
            vec_env = VecTransposeImage(vec_env)
            logger.info(
                "CNN pipeline: CnnObservationWrapper → DummyVecEnv → "
                "VecFrameStack(%d) → VecTransposeImage",
                args.frame_stack,
            )
            logger.info("CNN observation space: %s", vec_env.observation_space.shape)
            train_env = vec_env
            sb3_policy = "CnnPolicy"
        else:
            train_env = env
            sb3_policy = "MlpPolicy"

        # -- Frame collector -----------------------------------------------
        collector = None
        if args.data_collection:
            collector = FrameCollector(
                output_dir=output_dir,
                capture_interval=args.frame_interval,
            )

        # -- Callback ------------------------------------------------------
        checkpoint_dir = output_dir / "checkpoints"
        cb_factory = FrameCollectionCallback(
            collector=collector,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_dir=checkpoint_dir,
            training_logger=tlog,
            log_interval=args.log_interval,
            max_time=args.max_time,
            max_episodes=args.max_episodes,
            game_slug=game_slug,
        )
        callback = cb_factory.create()

        # -- PPO model -----------------------------------------------------
        # Device routing:
        # - MlpPolicy: CPU is fastest (tiny network, GPU overhead hurts)
        # - CnnPolicy: GPU benefits from NatureCNN's ~1.7M params
        #
        # SB3 supports "auto", "cpu", "cuda" but not "xpu" natively.
        # For XPU, we pass a torch.device object directly.
        sb3_device: Any = args.device
        if args.device in ("xpu", "auto"):
            if args.policy == "cnn":
                # CNN benefits from GPU — use XPU:1 (GPU.0 is for YOLO)
                try:
                    import torch

                    if torch.xpu.is_available() and torch.xpu.device_count() > 1:
                        sb3_device = torch.device("xpu:1")
                        logger.info("CnnPolicy device: xpu:1 (dedicated GPU)")
                    elif torch.xpu.is_available():
                        sb3_device = torch.device("xpu:0")
                        logger.info("CnnPolicy device: xpu:0 (single GPU)")
                    else:
                        sb3_device = "cpu"
                        logger.info("CnnPolicy device: cpu (XPU unavailable)")
                except ImportError:
                    sb3_device = "cpu"
                    logger.warning(
                        "PyTorch could not be imported; "
                        "falling back to CPU for CnnPolicy device."
                    )
            else:
                # MlpPolicy: CPU is faster than GPU for tiny networks
                sb3_device = "cpu"
        else:
            # Explicit device specified — respect user choice but log
            # how it interacts with the selected policy.
            if args.policy == "cnn":
                logger.info(
                    "CnnPolicy device: %s (explicit --device override)",
                    args.device,
                )
            else:
                logger.info(
                    "MlpPolicy device: %s (explicit --device override)",
                    args.device,
                )

        if args.resume:
            resume_path = Path(args.resume)
            if not resume_path.exists():
                # Try appending .zip if not present
                if resume_path.with_suffix(".zip").exists():
                    resume_path = resume_path.with_suffix(".zip")
                else:
                    logger.error("Resume path not found: %s", resume_path)
                    return 1
            logger.info("Resuming training from checkpoint: %s", resume_path)
            model = PPO.load(
                str(resume_path),
                env=train_env,
                device=sb3_device,
            )
            logger.info(
                "Loaded model: %d timesteps completed previously",
                model.num_timesteps,
            )
        else:
            model = PPO(
                sb3_policy,
                train_env,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                learning_rate=args.lr,
                clip_range=args.clip_range,
                ent_coef=args.ent_coef,
                device=sb3_device,
                verbose=1,
            )

        logger.info(
            "Starting PPO training: %d timesteps, policy=%s, device=%s (sb3=%s)%s",
            args.timesteps,
            sb3_policy,
            args.device,
            sb3_device,
            f", resumed from {args.resume}" if args.resume else "",
        )
        train_start = time.perf_counter()
        train_elapsed = 0.0
        interrupted = False
        model_path: Path | None = None
        partial_path: Path | None = None

        # -- Training loop (with interrupt handling) -----------------------
        try:
            model.learn(
                total_timesteps=args.timesteps,
                callback=callback,
                progress_bar=True,
                reset_num_timesteps=not bool(args.resume),
            )
        except KeyboardInterrupt:
            interrupted = True
            train_elapsed = time.perf_counter() - train_start
            num_steps = callback.num_timesteps if callback is not None else 0
            logger.warning(
                "Training interrupted by user at step %d after %.1fs",
                num_steps,
                train_elapsed,
            )

            # Save partial model
            partial_path = output_dir / "models" / f"ppo_{game_slug}_interrupted"
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                model.save(str(partial_path))
                logger.info("Partial model saved to %s", partial_path)
            except Exception as exc:
                logger.error("Failed to save partial model: %s", exc)
                partial_path = None

            tlog.log(
                {
                    "event": "interrupted",
                    "step": num_steps,
                    "episodes_completed": len(cb_factory._episode_rewards),
                    "mean_reward": (
                        float(
                            sum(cb_factory._episode_rewards)
                            / len(cb_factory._episode_rewards)
                        )
                        if cb_factory._episode_rewards
                        else 0.0
                    ),
                    "partial_model_path": (str(partial_path) if partial_path else None),
                    "training_time_seconds": round(train_elapsed, 1),
                }
            )

        # -- Normal completion ---------------------------------------------
        if not interrupted:
            train_elapsed = time.perf_counter() - train_start
            logger.info("Training complete in %.1f seconds", train_elapsed)

            model_path = output_dir / "models" / f"ppo_{game_slug}_final"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(model_path))
            logger.info("Final model saved to %s", model_path)
        else:
            model_path = partial_path

        # -- Finalize data collection --------------------------------------
        if collector is not None:
            try:
                frames_dir = collector.finalize()
                logger.info(
                    "Collected %d frames in %s",
                    collector.frame_count,
                    frames_dir,
                )
            except Exception as exc:
                logger.error("Failed to finalize frame collector: %s", exc)

        # -- Summary event -------------------------------------------------
        import numpy as np

        ep_rewards = cb_factory._episode_rewards
        ep_lengths = cb_factory._episode_lengths
        total_steps = callback.num_timesteps if callback is not None else args.timesteps
        stop_reason = getattr(callback, "training_stop_reason", None)

        summary = {
            "event": "summary",
            "total_steps": total_steps,
            "total_episodes": len(ep_rewards),
            "mean_reward": (float(np.mean(ep_rewards)) if ep_rewards else 0.0),
            "std_reward": (float(np.std(ep_rewards)) if ep_rewards else 0.0),
            "mean_episode_length": (float(np.mean(ep_lengths)) if ep_lengths else 0.0),
            "best_episode_reward": (float(np.max(ep_rewards)) if ep_rewards else 0.0),
            "worst_episode_reward": (float(np.min(ep_rewards)) if ep_rewards else 0.0),
            "training_time_seconds": round(train_elapsed, 1),
            "model_path": str(model_path) if model_path else None,
            "frames_collected": (collector.frame_count if collector else 0),
            "completed": not interrupted,
            "stop_reason": stop_reason
            or ("interrupted" if interrupted else "completed"),
        }
        tlog.log(summary)

        # -- Console summary -----------------------------------------------
        print("\n--- Training Summary ---")
        print(f"Total timesteps: {total_steps}")
        print(f"Training time:   {train_elapsed:.1f}s")
        print(f"Episodes:        {len(ep_rewards)}")
        if ep_rewards:
            print(f"Mean reward:     {np.mean(ep_rewards):.2f}")
            print(f"Std reward:      {np.std(ep_rewards):.2f}")
            print(f"Best reward:     {np.max(ep_rewards):.2f}")
            print(f"Worst reward:    {np.min(ep_rewards):.2f}")
            print(f"Mean length:     {np.mean(ep_lengths):.0f}")
        print(f"Model saved:     {model_path}")
        if collector is not None:
            print(f"Frames saved:    {collector.frame_count}")
        print(f"Log file:        {log_path}")
        print(f"JSONL log:       {jsonl_path}")
        status = stop_reason or ("INTERRUPTED" if interrupted else "COMPLETED")
        print(f"Status:          {status.upper()}")

    finally:
        if vec_env is not None:
            vec_env.close()
        elif "env" in locals():
            env.close()
        browser_instance.close()
        logger.info("Stopping dev server ...")
        loader.stop()
        tlog.close()
        logging.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
