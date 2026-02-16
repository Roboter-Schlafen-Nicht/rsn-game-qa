"""Session runner -- full game lifecycle orchestrator.

Launches the game, creates the environment with oracles, runs N episodes,
generates structured reports and an HTML dashboard.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from src.oracles import (
    BoundaryOracle,
    CrashOracle,
    EpisodeLengthOracle,
    Oracle,
    PerformanceOracle,
    PhysicsViolationOracle,
    RewardConsistencyOracle,
    ScoreAnomalyOracle,
    SoakOracle,
    StateTransitionOracle,
    StuckOracle,
    TemporalAnomalyOracle,
    VisualGlitchOracle,
)
from src.oracles.base import Finding
from src.reporting import (
    DashboardRenderer,
    EpisodeMetrics,
    EpisodeReport,
    FindingReport,
    ReportGenerator,
    SessionReport,
)

from .data_collector import FrameCollector

logger = logging.getLogger(__name__)


def _finding_to_report(
    finding: Finding,
    screenshots_dir: Path,
) -> FindingReport:
    """Convert an oracle ``Finding`` to a serialisable ``FindingReport``.

    If the finding has an attached frame (numpy array), it is saved
    as a PNG in ``screenshots_dir``.

    Parameters
    ----------
    finding : Finding
        The oracle finding to convert.
    screenshots_dir : Path
        Directory to save any attached screenshot.

    Returns
    -------
    FindingReport
        The serialisable finding report.
    """
    screenshot_path: str | None = None

    if finding.frame is not None:
        # Lazy import to avoid CI failures in Docker
        import cv2

        screenshots_dir.mkdir(parents=True, exist_ok=True)
        fname = (
            f"{finding.oracle_name}_step{finding.step}_{int(time.time() * 1000)}.png"
        )
        fpath = screenshots_dir / fname
        cv2.imwrite(str(fpath), finding.frame)
        screenshot_path = str(fpath)

    return FindingReport(
        oracle_name=finding.oracle_name,
        severity=finding.severity,
        step=finding.step,
        description=finding.description,
        data=finding.data,
        screenshot_path=screenshot_path,
    )


def _build_episode_metrics(
    step_times: list[float],
    rewards: list[float],
    episode_start: float,
) -> EpisodeMetrics:
    """Compute episode performance metrics from raw step timing data.

    Parameters
    ----------
    step_times : list[float]
        Duration of each step in seconds.
    rewards : list[float]
        Reward received at each step.
    episode_start : float
        ``time.perf_counter()`` value at episode start.

    Returns
    -------
    EpisodeMetrics
        Computed metrics dataclass.
    """
    total_duration = time.perf_counter() - episode_start

    if step_times:
        fps_values = [1.0 / max(t, 1e-9) for t in step_times]
        mean_fps = float(np.mean(fps_values))
        min_fps = float(np.min(fps_values))
    else:
        mean_fps = None
        min_fps = None

    max_reward = float(max(rewards)) if rewards else None

    return EpisodeMetrics(
        mean_fps=mean_fps,
        min_fps=min_fps,
        max_reward_per_step=max_reward,
        total_duration_seconds=total_duration,
    )


def _create_oracles() -> list[Oracle]:
    """Instantiate all 12 oracles with default settings.

    Returns
    -------
    list[Oracle]
        All oracle instances.
    """
    return [
        BoundaryOracle(),
        CrashOracle(),
        EpisodeLengthOracle(),
        PerformanceOracle(),
        PhysicsViolationOracle(),
        RewardConsistencyOracle(),
        ScoreAnomalyOracle(),
        SoakOracle(),
        StateTransitionOracle(),
        StuckOracle(),
        TemporalAnomalyOracle(),
        VisualGlitchOracle(),
    ]


class SessionRunner:
    """Full game lifecycle orchestrator.

    Launches the game via a Selenium browser, creates the
    ``Breakout71Env`` with all 12 oracles, runs N episodes, collects
    data for YOLO retraining, and generates structured JSON reports
    with an HTML dashboard.

    Parameters
    ----------
    game_config : str or Path
        Path to the game YAML config (e.g. ``configs/games/breakout-71.yaml``).
    n_episodes : int
        Number of episodes to run.
    max_steps_per_episode : int
        Maximum steps per episode (truncation threshold).
    output_dir : str or Path
        Directory for reports, screenshots, and collected frames.
    browser : str or None
        Browser to use (``"chrome"``, ``"msedge"``, ``"firefox"``).
        If None, auto-selects the first available.
    yolo_weights : str or Path
        Path to trained YOLO weights.
    frame_capture_interval : int
        Save every Nth frame for data collection.
    enable_data_collection : bool
        Whether to collect frames for YOLO retraining.
    """

    def __init__(
        self,
        game_config: str | Path = "configs/games/breakout-71.yaml",
        n_episodes: int = 3,
        max_steps_per_episode: int = 10_000,
        output_dir: str | Path = "output",
        browser: str | None = None,
        yolo_weights: str | Path = "weights/breakout71/best.pt",
        frame_capture_interval: int = 30,
        enable_data_collection: bool = True,
    ) -> None:
        self.game_config_path = Path(game_config)
        self.n_episodes = n_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.output_dir = Path(output_dir)
        self.browser = browser
        self.yolo_weights = Path(yolo_weights)
        self.frame_capture_interval = frame_capture_interval
        self.enable_data_collection = enable_data_collection

        self._browser_instance = None
        self._env = None
        self._collector: FrameCollector | None = None
        self._loader = None

    def run(self) -> SessionReport:
        """Run the full QA session: launch game, run episodes, generate report.

        Returns
        -------
        SessionReport
            The complete session report with all episodes and summary.
        """
        report_gen = ReportGenerator(
            output_dir=self.output_dir / "reports",
            game_name="breakout-71",
        )

        try:
            self._setup()

            for ep_idx in range(self.n_episodes):
                logger.info("Starting episode %d/%d", ep_idx + 1, self.n_episodes)
                episode = self._run_episode(ep_idx)
                report_gen.add_episode(episode)
                logger.info(
                    "Episode %d: %d steps, reward=%.2f, findings=%d",
                    ep_idx + 1,
                    episode.steps,
                    episode.total_reward,
                    len(episode.findings),
                )
        finally:
            self._cleanup()

        # Finalize data collection
        if self._collector is not None:
            frames_dir = self._collector.finalize()
            logger.info(
                "Collected %d frames in %s",
                self._collector.frame_count,
                frames_dir,
            )

        # Save JSON report
        report_path = report_gen.save()
        logger.info("Report saved to %s", report_path)

        # Generate HTML dashboard
        try:
            dashboard = DashboardRenderer(
                output_dir=self.output_dir / "reports",
            )
            dashboard_path = dashboard.render(report_gen.to_dict())
            logger.info("Dashboard saved to %s", dashboard_path)
        except Exception:
            logger.warning("Dashboard generation failed", exc_info=True)

        return report_gen.session

    def _setup(self) -> None:
        """Launch the game browser and create the environment."""
        # Lazy imports to avoid CI failures
        from scripts._smoke_utils import BrowserInstance
        from src.env.breakout71_env import Breakout71Env
        from src.game_loader import create_loader
        from src.game_loader.config import load_game_config

        # Load game config
        config = load_game_config(
            self.game_config_path.stem,
            configs_dir=self.game_config_path.parent,
        )

        # Start dev server via GameLoader
        self._loader = create_loader(config)
        self._loader.setup()
        self._loader.start()
        logger.info("Dev server ready at %s", self._loader.url or config.url)

        # Launch browser
        url = self._loader.url or config.url
        window_size = (config.window_width, config.window_height)
        self._browser_instance = BrowserInstance(
            url=url,
            settle_seconds=8.0,
            window_size=window_size,
            browser=self.browser,
        )

        # Create oracles
        oracles = _create_oracles()

        # Create environment with Selenium driver
        window_title = config.window_title or "Breakout"
        self._env = Breakout71Env(
            window_title=window_title,
            yolo_weights=self.yolo_weights,
            max_steps=self.max_steps_per_episode,
            oracles=oracles,
            driver=self._browser_instance.driver,
        )

        # Create frame collector
        if self.enable_data_collection:
            self._collector = FrameCollector(
                output_dir=self.output_dir,
                capture_interval=self.frame_capture_interval,
            )

        logger.info(
            "Session setup complete: browser=%s, window=%s, episodes=%d",
            self.browser,
            window_title,
            self.n_episodes,
        )

    def _run_episode(self, episode_id: int) -> EpisodeReport:
        """Run a single episode and return its report.

        Parameters
        ----------
        episode_id : int
            Sequential episode number.

        Returns
        -------
        EpisodeReport
            The completed episode report.
        """
        env = self._env
        screenshots_dir = self.output_dir / "screenshots" / f"episode_{episode_id}"

        obs, info = env.reset()
        episode_start = time.perf_counter()

        total_reward = 0.0
        step_times: list[float] = []
        rewards: list[float] = []
        terminated = False
        truncated = False

        while not terminated and not truncated:
            step_start = time.perf_counter()

            # Random policy (uniform over action space)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            step_elapsed = time.perf_counter() - step_start
            step_times.append(step_elapsed)
            rewards.append(reward)
            total_reward += reward

            # Collect frames for YOLO retraining
            if self._collector is not None and info.get("frame") is not None:
                self._collector.save_frame(
                    frame=info["frame"],
                    step=env.step_count,
                    episode_id=episode_id,
                )

        # Gather oracle findings
        findings: list[FindingReport] = []
        for oracle in env._oracles:
            for finding in oracle.get_findings():
                findings.append(_finding_to_report(finding, screenshots_dir))

        metrics = _build_episode_metrics(step_times, rewards, episode_start)

        return EpisodeReport(
            episode_id=episode_id,
            steps=env.step_count,
            total_reward=total_reward,
            terminated=terminated,
            truncated=truncated,
            findings=findings,
            metrics=metrics,
        )

    def _cleanup(self) -> None:
        """Release environment, browser, and loader resources."""
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                logger.warning("Env cleanup failed", exc_info=True)
            self._env = None

        if self._browser_instance is not None:
            try:
                self._browser_instance.close()
            except Exception:
                logger.warning("Browser cleanup failed", exc_info=True)
            self._browser_instance = None

        if self._loader is not None:
            try:
                self._loader.stop()
                logger.info("Dev server stopped")
            except Exception:
                logger.warning("Loader cleanup failed", exc_info=True)
            self._loader = None
