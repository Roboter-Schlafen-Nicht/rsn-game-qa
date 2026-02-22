"""Session runner -- full game lifecycle orchestrator.

Launches the game, creates the environment with oracles, runs N episodes,
generates structured reports and an HTML dashboard.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

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
        fname = f"{finding.oracle_name}_step{finding.step}_{int(time.time() * 1000)}.png"
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

    Launches the game via a Selenium browser, creates the game
    environment with all 12 oracles, runs N episodes, collects
    data for YOLO retraining, and generates structured JSON reports
    with an HTML dashboard.

    Parameters
    ----------
    game : str
        Game plugin name (directory under ``games/``, e.g.
        ``"breakout71"``).  Used to dynamically load the env class,
        loader class, and derive default config/weights paths.
    game_config : str or Path or None
        Path to the game YAML config.  If None, uses the plugin's
        ``default_config``.
    n_episodes : int
        Number of episodes to run.
    max_steps_per_episode : int
        Maximum steps per episode (truncation threshold).
    output_dir : str or Path
        Directory for reports, screenshots, and collected frames.
    browser : str or None
        Browser to use (``"chrome"``, ``"msedge"``, ``"firefox"``).
        If None, auto-selects the first available.
    yolo_weights : str or Path or None
        Path to trained YOLO weights.  If None, uses the plugin's
        ``default_weights``.
    frame_capture_interval : int
        Save every Nth frame for data collection.
    enable_data_collection : bool
        Whether to collect frames for YOLO retraining.
    policy_fn : callable or None
        Action-selection function with signature ``(obs) -> action``.
        When None (default), uses random sampling via
        ``env.action_space.sample()``.  Pass a trained model's predict
        function to evaluate a learned policy.
    headless : bool
        If True, launch the browser in headless mode and use
        Selenium-based frame capture instead of Win32 APIs.
    policy : str
        Observation policy type: ``"mlp"`` (default) or ``"cnn"``.
        When ``"cnn"``, the environment is wrapped with
        :class:`~src.platform.cnn_wrapper.CnnEvalWrapper` to produce
        stacked grayscale image observations matching the training
        pipeline.
    frame_stack : int
        Number of frames to stack when ``policy="cnn"``.  Default is 4.
        Ignored when ``policy="mlp"``.
    reward_mode : str
        Reward signal strategy passed to the environment.
        ``"yolo"`` (default) uses game-specific YOLO-based reward.
        ``"survival"`` uses ``+0.01`` per step, ``-5.0`` on game over.
    game_over_detector : GameOverDetector or None
        Pixel-based game-over detector.  When provided, passed to
        the environment constructor so that ``update(frame)`` is called
        every step.  If it signals game-over, the episode terminates
        without requiring DOM/JS modal checks.
    score_region : tuple[int, int, int, int] or None
        Bounding box ``(x, y, w, h)`` for OCR score reading.  Only
        used when ``reward_mode="score"``.  If None, OCR scans the
        full frame.
    score_ocr_interval : int
        Run OCR every N steps to reduce overhead.  Default is 5.
        Only used when ``reward_mode="score"``.
    score_reward_coeff : float
        Coefficient for score delta reward.  Default is 0.01.
        Only used when ``reward_mode="score"``.
    human_mode : bool
        When True, the human controls the game through the browser.
        ``apply_action()`` becomes a no-op, ``policy_fn`` is forced
        to None, and the ``EventRecorder`` captures human input
        events each step.  Default is False.
    record_demo : bool
        When True, enables enriched per-step recording via
        :class:`~src.orchestrator.demo_recorder.DemoRecorder`.
        Captures frames, human events, game state, reward, oracle
        findings, and observation hashes.  Default is False.
    """

    def __init__(
        self,
        game: str = "breakout71",
        game_config: str | Path | None = None,
        n_episodes: int = 3,
        max_steps_per_episode: int = 10_000,
        output_dir: str | Path = "output",
        browser: str | None = None,
        yolo_weights: str | Path | None = None,
        frame_capture_interval: int = 30,
        enable_data_collection: bool = True,
        policy_fn: Callable[[Any], Any] | None = None,
        headless: bool = False,
        policy: str = "mlp",
        frame_stack: int = 4,
        reward_mode: str = "yolo",
        game_over_detector: Any | None = None,
        score_region: tuple[int, int, int, int] | None = None,
        score_ocr_interval: int = 5,
        score_reward_coeff: float = 0.01,
        human_mode: bool = False,
        record_demo: bool = False,
    ) -> None:
        valid_policies = ("mlp", "cnn")
        if policy not in valid_policies:
            raise ValueError(f"policy must be one of {valid_policies}, got {policy!r}")
        if policy == "cnn" and frame_stack < 1:
            raise ValueError(f"frame_stack must be >= 1 when policy='cnn', got {frame_stack}")
        self.game = game
        self.n_episodes = n_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.output_dir = Path(output_dir)
        self.browser = browser
        self.frame_capture_interval = frame_capture_interval
        self.enable_data_collection = enable_data_collection
        self.policy_fn = policy_fn
        self.headless = headless
        self.human_mode = human_mode
        self.record_demo = record_demo

        # In human mode, the human controls the game — no policy needed
        if human_mode:
            self.policy_fn = None
        self.policy = policy
        self.frame_stack = frame_stack
        self.reward_mode = reward_mode
        self.game_over_detector = game_over_detector
        self.score_region = score_region
        self.score_ocr_interval = score_ocr_interval
        self.score_reward_coeff = score_reward_coeff

        # Load plugin to resolve defaults
        from games import load_game_plugin

        self._plugin = load_game_plugin(game)
        self.game_config_path = Path(
            game_config if game_config is not None else self._plugin.default_config
        )
        self.yolo_weights = Path(
            yolo_weights if yolo_weights is not None else self._plugin.default_weights
        )

        self._browser_instance = None
        self._env = None
        self._raw_env = None  # Unwrapped env for oracle/step_count access
        self._collector: FrameCollector | None = None
        self._demo_recorder = None
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
            game_name=self._plugin.game_name,
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

        # Finalize demo recording
        if self._demo_recorder is not None:
            demo_dir = self._demo_recorder.finalize()
            logger.info(
                "Demo recording finalized: %d steps in %s",
                self._demo_recorder.step_count,
                demo_dir,
            )

        # Save JSON report
        report_path = report_gen.save()
        logger.info("Report saved to %s", report_path)

        # Generate HTML dashboard
        try:
            dashboard = DashboardRenderer()
            dashboard_path = dashboard.render_to_file(
                report_gen.to_dict(),
                self.output_dir / "reports" / "dashboard.html",
            )
            logger.info("Dashboard saved to %s", dashboard_path)
        except Exception:
            logger.warning("Dashboard generation failed", exc_info=True)

        return report_gen.session

    def _setup(self) -> None:
        """Launch the game browser and create the environment."""
        # Lazy imports to avoid CI failures
        from scripts._smoke_utils import BrowserInstance
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
            headless=self.headless,
        )

        # -- Apply plugin JS snippets (mute, setup, reinit) ---------------
        driver = self._browser_instance.driver
        needs_refresh = False
        mute_js = getattr(self._plugin, "mute_js", None)
        setup_js = getattr(self._plugin, "setup_js", None)
        reinit_js = getattr(self._plugin, "reinit_js", None)

        if mute_js and driver is not None:
            try:
                driver.execute_script(mute_js)
                needs_refresh = True
                logger.info("Game audio muted via plugin mute_js")
            except Exception as exc:
                logger.warning("Failed to mute game audio: %s", exc)

        if setup_js and driver is not None:
            try:
                driver.execute_script(setup_js)
                needs_refresh = True
                logger.info("Game settings applied via plugin setup_js")
            except Exception as exc:
                logger.warning("Failed to apply setup_js: %s", exc)

        if needs_refresh and driver is not None:
            driver.refresh()
            time.sleep(3)  # let page reload with new settings

        if needs_refresh and reinit_js and driver is not None:
            try:
                driver.execute_script(reinit_js)
                time.sleep(2)  # let game re-initialise
                logger.info("Game re-initialised via plugin reinit_js")
            except Exception as exc:
                logger.warning("Failed to re-initialise game: %s", exc)

        # Create oracles
        oracles = _create_oracles()

        # Create environment using the plugin's env class
        EnvClass = self._plugin.env_class
        window_title = config.window_title or "Breakout"
        self._env = EnvClass(
            window_title=window_title,
            yolo_weights=self.yolo_weights,
            max_steps=self.max_steps_per_episode,
            oracles=oracles,
            driver=self._browser_instance.driver,
            headless=self.headless,
            reward_mode=self.reward_mode,
            game_over_detector=self.game_over_detector,
            browser_instance=self._browser_instance,
            score_region=self.score_region,
            score_ocr_interval=self.score_ocr_interval,
            score_reward_coeff=self.score_reward_coeff,
            human_mode=self.human_mode,
        )

        # Create frame collector
        if self.enable_data_collection:
            self._collector = FrameCollector(
                output_dir=self.output_dir,
                capture_interval=self.frame_capture_interval,
            )

        # Create demo recorder
        if self.record_demo:
            from .demo_recorder import DemoRecorder

            self._demo_recorder = DemoRecorder(
                output_dir=self.output_dir,
                game_name=self._plugin.game_name,
            )

        logger.info(
            "Session setup complete: browser=%s, window=%s, episodes=%d",
            self.browser,
            window_title,
            self.n_episodes,
        )

        # Apply CNN wrapping if requested
        self._wrap_env_for_cnn()

    def _wrap_env_for_cnn(self) -> None:
        """Wrap the environment for CNN policy evaluation if configured.

        When ``self.policy == "cnn"``, wraps ``self._env`` with
        :class:`~src.platform.cnn_wrapper.CnnEvalWrapper`.  Stores the
        original unwrapped env in ``self._raw_env`` for oracle access.

        When ``self.policy == "mlp"``, this is a no-op.
        """
        if self.policy != "cnn":
            return

        from src.platform.cnn_wrapper import CnnEvalWrapper

        self._raw_env = self._env
        self._env = CnnEvalWrapper(self._env, frame_stack=self.frame_stack)
        logger.info(
            "CNN eval wrapper applied: frame_stack=%d, obs_space=%s",
            self.frame_stack,
            self._env.observation_space.shape,
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
        # Use raw (unwrapped) env for oracle/step_count when CNN-wrapped
        raw_env = self._raw_env if self._raw_env is not None else env
        screenshots_dir = self.output_dir / "screenshots" / f"episode_{episode_id}"

        # Start demo recording for this episode
        if self._demo_recorder is not None:
            self._demo_recorder.start_episode(episode_id=episode_id)

        obs, info = env.reset()
        episode_start = time.perf_counter()

        total_reward = 0.0
        step_times: list[float] = []
        rewards: list[float] = []
        terminated = False
        truncated = False
        step_idx = 0

        while not terminated and not truncated:
            step_start = time.perf_counter()

            # Select action: use policy_fn if provided, else random
            if self.policy_fn is not None:
                action = self.policy_fn(obs)
            else:
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
                    step=raw_env.step_count,
                    episode_id=episode_id,
                )

            # Record enriched demo step
            if self._demo_recorder is not None:
                self._demo_recorder.record_step(
                    step=step_idx,
                    frame=info.get("frame"),
                    action=action,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    human_events=info.get("human_events"),
                    game_state=info.get("game_state"),
                    observation=obs,
                    oracle_findings=info.get("oracle_findings"),
                )

            step_idx += 1

        # End demo recording for this episode
        if self._demo_recorder is not None:
            self._demo_recorder.end_episode(
                terminated=terminated,
                truncated=truncated,
            )

        # Gather oracle findings from raw env (oracles live on base env)
        findings: list[FindingReport] = []
        for oracle in raw_env._oracles:
            for finding in oracle.get_findings():
                findings.append(_finding_to_report(finding, screenshots_dir))

        metrics = _build_episode_metrics(step_times, rewards, episode_start)

        return EpisodeReport(
            episode_id=episode_id,
            steps=raw_env.step_count,
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
            self._raw_env = None

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
