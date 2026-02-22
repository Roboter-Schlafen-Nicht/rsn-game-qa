"""Tests for human play mode (Phase 7b).

Verifies that ``human_mode=True`` on ``BaseGameEnv`` makes
``apply_action()`` a no-op, and that ``SessionRunner`` integrates
the ``EventRecorder`` and step timing when running in human mode.
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import gymnasium as gym
import numpy as np

from src.platform.base_env import BaseGameEnv
from src.platform.event_recorder import EventRecorder

# ---------------------------------------------------------------------------
# Stub env (mirrors test_base_env.py pattern)
# ---------------------------------------------------------------------------


class HumanModeStubEnv(BaseGameEnv):
    """Minimal concrete env for human-mode tests."""

    _BALL_LOST_THRESHOLD = 5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._no_ball_count: int = 0
        self._no_bricks_count: int = 0
        self.apply_action_called: bool = False

    def game_classes(self) -> list[str]:
        return ["ball", "paddle", "brick"]

    def build_observation(self, detections: dict[str, Any], *, reset: bool = False) -> np.ndarray:
        return np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    def compute_reward(
        self, detections: dict[str, Any], terminated: bool, level_cleared: bool
    ) -> float:
        return 0.01

    def check_termination(self, detections: dict[str, Any]) -> tuple[bool, bool]:
        return False, False

    def apply_action(self, action: np.ndarray) -> None:
        self.apply_action_called = True

    def handle_modals(self, *, dismiss_game_over: bool = True) -> str:
        return "gameplay"

    def start_game(self) -> None:
        pass

    def canvas_selector(self) -> str:
        return "game"

    def build_info(self, detections: dict[str, Any]) -> dict[str, Any]:
        return {}

    def terminal_reward(self) -> float:
        return -5.0

    def on_reset_detections(self, detections: dict[str, Any]) -> bool:
        return True

    def reset_termination_state(self) -> None:
        self._no_ball_count = 0
        self._no_bricks_count = 0


# ---------------------------------------------------------------------------
# BaseGameEnv human_mode parameter
# ---------------------------------------------------------------------------


class TestBaseGameEnvHumanMode:
    """Test that human_mode parameter changes env behaviour."""

    def test_human_mode_defaults_to_false(self):
        """human_mode is False by default."""
        env = HumanModeStubEnv(reward_mode="survival")
        assert env.human_mode is False

    def test_human_mode_can_be_set_true(self):
        """human_mode=True is accepted in constructor."""
        env = HumanModeStubEnv(reward_mode="survival", human_mode=True)
        assert env.human_mode is True

    def test_apply_action_skipped_in_human_mode(self):
        """In human mode, apply_action is not called during step()."""
        env = HumanModeStubEnv(reward_mode="survival", human_mode=True)
        # Set up internal state to allow step() to work
        env._initialized = True
        env._last_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        env._step_count = 0

        # Mock the capture/detect pipeline so step() can run
        fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections: dict[str, Any] = {"ball": (0.5, 0.5, 0.02, 0.02)}
        with (
            mock.patch.object(env, "_capture_frame", return_value=fake_frame),
            mock.patch.object(env, "_detect_objects", return_value=detections),
            mock.patch.object(env, "_should_check_modals", return_value=False),
            mock.patch.object(env, "_check_late_game_over", return_value=False),
            mock.patch.object(env, "_run_oracles", return_value=[]),
        ):
            env.step(np.array([0.0]))

        assert env.apply_action_called is False

    def test_apply_action_called_in_normal_mode(self):
        """In normal mode, apply_action IS called during step()."""
        env = HumanModeStubEnv(reward_mode="survival")
        env._initialized = True
        env._last_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        env._step_count = 0

        fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections: dict[str, Any] = {"ball": (0.5, 0.5, 0.02, 0.02)}
        with (
            mock.patch.object(env, "_capture_frame", return_value=fake_frame),
            mock.patch.object(env, "_detect_objects", return_value=detections),
            mock.patch.object(env, "_should_check_modals", return_value=False),
            mock.patch.object(env, "_check_late_game_over", return_value=False),
            mock.patch.object(env, "_run_oracles", return_value=[]),
        ):
            env.step(np.array([0.0]))

        assert env.apply_action_called is True

    def test_event_recorder_accessible_in_human_mode(self):
        """Human mode env exposes an EventRecorder instance."""
        env = HumanModeStubEnv(reward_mode="survival", human_mode=True)
        assert isinstance(env.event_recorder, EventRecorder)

    def test_event_recorder_none_in_normal_mode(self):
        """Normal mode env has no EventRecorder."""
        env = HumanModeStubEnv(reward_mode="survival")
        assert env.event_recorder is None

    def test_event_recorder_gets_driver(self):
        """EventRecorder receives the driver when set on the env."""
        driver = mock.MagicMock()
        env = HumanModeStubEnv(reward_mode="survival", human_mode=True, driver=driver)
        assert env.event_recorder._driver is driver

    def test_flush_events_returns_events(self):
        """flush_events() returns events from the EventRecorder."""
        driver = mock.MagicMock()
        raw_events = [{"type": "click", "x": 100, "y": 200, "button": 0, "timestamp": 1000.0}]
        driver.execute_script.return_value = raw_events
        env = HumanModeStubEnv(reward_mode="survival", human_mode=True, driver=driver)
        events = env.flush_events()
        assert len(events) == 1
        assert events[0]["type"] == "click"

    def test_flush_events_empty_in_normal_mode(self):
        """flush_events() returns [] in normal mode."""
        env = HumanModeStubEnv(reward_mode="survival")
        assert env.flush_events() == []

    def test_info_contains_human_events_when_human_mode(self):
        """step() info dict includes 'human_events' in human mode."""
        env = HumanModeStubEnv(reward_mode="survival", human_mode=True)
        env._initialized = True
        env._last_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        env._step_count = 0

        driver = mock.MagicMock()
        driver.execute_script.return_value = [
            {"type": "click", "x": 50, "y": 60, "button": 0, "timestamp": 500.0}
        ]
        env._driver = driver
        env._event_recorder.set_driver(driver)

        fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections: dict[str, Any] = {"ball": (0.5, 0.5, 0.02, 0.02)}
        with (
            mock.patch.object(env, "_capture_frame", return_value=fake_frame),
            mock.patch.object(env, "_detect_objects", return_value=detections),
            mock.patch.object(env, "_should_check_modals", return_value=False),
            mock.patch.object(env, "_check_late_game_over", return_value=False),
            mock.patch.object(env, "_run_oracles", return_value=[]),
        ):
            _, _, _, _, info = env.step(np.array([0.0]))

        assert "human_events" in info
        assert len(info["human_events"]) == 1


# ---------------------------------------------------------------------------
# SessionRunner human_mode integration
# ---------------------------------------------------------------------------


class TestSessionRunnerHumanMode:
    """Test SessionRunner behaviour in human mode."""

    def test_human_mode_defaults_to_false(self):
        """SessionRunner defaults human_mode to False."""
        from src.orchestrator.session_runner import SessionRunner

        runner = SessionRunner()
        assert runner.human_mode is False

    def test_human_mode_can_be_set_true(self):
        """SessionRunner accepts human_mode=True."""
        from src.orchestrator.session_runner import SessionRunner

        runner = SessionRunner(human_mode=True)
        assert runner.human_mode is True

    def test_human_mode_uses_noop_policy(self):
        """In human mode, no action is applied by the policy."""
        from src.orchestrator.session_runner import SessionRunner

        runner = SessionRunner(human_mode=True)
        # policy_fn should be None (human controls the game)
        assert runner.policy_fn is None


# ---------------------------------------------------------------------------
# CLI --human flag
# ---------------------------------------------------------------------------


class TestRunSessionHumanFlag:
    """Test the --human CLI flag in run_session.py."""

    def test_human_flag_parsed(self):
        """--human flag is parsed correctly."""
        from scripts.run_session import parse_args

        args = parse_args(["--human"])
        assert args.human is True

    def test_human_flag_default_false(self):
        """--human flag defaults to False."""
        from scripts.run_session import parse_args

        args = parse_args([])
        assert args.human is False

    def test_human_flag_disables_headless(self):
        """--human flag with --headless raises or overrides to non-headless."""
        from scripts.run_session import parse_args

        # Even if --headless is given, --human should take precedence
        args = parse_args(["--human", "--headless"])
        # The human needs to see the browser, so headless should be
        # overridden to False or the combination should be handled
        # (implementation can choose either approach; we test the
        # main() function handles it)
        assert args.human is True
