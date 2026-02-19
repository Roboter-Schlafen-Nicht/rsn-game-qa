"""Tests for browser crash recovery in BrowserInstance and BaseGameEnv.

Verifies that:
- ``BrowserInstance.is_alive()`` detects live vs dead browsers.
- ``BrowserInstance.restart()`` creates a new driver session.
- ``BaseGameEnv.step()`` survives a browser crash via forced terminal
  transition and auto-restart.
- ``BaseGameEnv.reset()`` survives a browser crash via auto-restart.

TDD convention applies — these tests cover behavioral changes to
``step()`` and ``reset()``.
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import gymnasium as gym
import numpy as np
import pytest

from src.platform.base_env import BaseGameEnv

try:
    import cv2  # noqa: F401

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

_skip_no_cv2 = pytest.mark.skipif(not _CV2_AVAILABLE, reason="cv2 not available")


# ---------------------------------------------------------------------------
# Stub env (same as test_base_env.py)
# ---------------------------------------------------------------------------


class StubEnv(BaseGameEnv):
    """Minimal concrete implementation of BaseGameEnv for tests."""

    _BALL_LOST_THRESHOLD = 5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._no_ball_count: int = 0
        self._no_bricks_count: int = 0

    def game_classes(self) -> list[str]:
        return ["ball", "paddle", "brick"]

    def build_observation(self, detections: dict[str, Any], *, reset: bool = False) -> np.ndarray:
        ball = detections.get("ball")
        bx = ball[0] if ball else 0.5
        by = ball[1] if ball else 0.5
        return np.array([bx, by, 0.5, 0.5], dtype=np.float32)

    def compute_reward(
        self,
        detections: dict[str, Any],
        terminated: bool,
        level_cleared: bool,
    ) -> float:
        if terminated:
            return -1.0
        return 0.01

    def check_termination(self, detections: dict[str, Any]) -> tuple[bool, bool]:
        if self._no_ball_count >= self._BALL_LOST_THRESHOLD:
            return True, False
        return False, False

    def apply_action(self, action: np.ndarray) -> None:
        pass

    def handle_modals(self, *, dismiss_game_over: bool = True) -> str:
        return "gameplay"

    def start_game(self) -> None:
        pass

    def canvas_selector(self) -> str:
        return "game"

    def build_info(self, detections: dict[str, Any]) -> dict[str, Any]:
        return {"ball_detected": detections.get("ball") is not None}

    def terminal_reward(self) -> float:
        return -5.0

    def on_reset_detections(self, detections: dict[str, Any]) -> bool:
        return detections.get("ball") is not None

    def reset_termination_state(self) -> None:
        self._no_ball_count = 0
        self._no_bricks_count = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detections(*, ball=(0.5, 0.5, 0.02, 0.02), bricks=None):
    """Build a minimal detections dict."""
    return {
        "ball": ball,
        "paddle": (0.5, 0.9, 0.1, 0.02),
        "bricks": bricks or [(0.1 * i, 0.1, 0.05, 0.03) for i in range(5)],
        "powerups": [],
        "raw_detections": [],
    }


def _action(val=0.0):
    return np.array([val], dtype=np.float32)


def _make_ready_env(**kwargs):
    """Return a StubEnv in post-reset state with mocked internals."""
    env = StubEnv(**kwargs)
    env._initialized = True
    env._capture = mock.MagicMock()
    env._detector = mock.MagicMock()

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    env._capture.capture_frame.return_value = frame
    env._detector.detect_to_game_state.return_value = _make_detections()
    env._last_frame = frame
    return env


# ===========================================================================
# BrowserInstance.is_alive() tests
# ===========================================================================


class TestBrowserInstanceIsAlive:
    """Tests for BrowserInstance.is_alive() method."""

    def test_is_alive_returns_true_when_driver_responds(self):
        """is_alive() returns True when driver.title succeeds."""
        from scripts._smoke_utils import BrowserInstance

        bi = BrowserInstance.__new__(BrowserInstance)
        bi._driver = mock.MagicMock()
        bi._driver.title = "Test Page"
        bi.name = "chrome"

        assert bi.is_alive() is True

    def test_is_alive_returns_false_when_driver_throws(self):
        """is_alive() returns False when driver.title raises."""
        from scripts._smoke_utils import BrowserInstance

        bi = BrowserInstance.__new__(BrowserInstance)
        bi._driver = mock.MagicMock()
        type(bi._driver).title = mock.PropertyMock(side_effect=Exception("tab crashed"))
        bi.name = "chrome"

        assert bi.is_alive() is False

    def test_is_alive_returns_false_when_driver_is_none(self):
        """is_alive() returns False when driver has been closed."""
        from scripts._smoke_utils import BrowserInstance

        bi = BrowserInstance.__new__(BrowserInstance)
        bi._driver = None
        bi.name = "chrome"

        assert bi.is_alive() is False


# ===========================================================================
# BrowserInstance.restart() tests
# ===========================================================================


class TestBrowserInstanceRestart:
    """Tests for BrowserInstance.restart() method."""

    def test_restart_creates_new_driver(self):
        """restart() closes old driver and creates a new one."""
        from scripts._smoke_utils import BrowserInstance

        bi = BrowserInstance.__new__(BrowserInstance)
        old_driver = mock.MagicMock()
        bi._driver = old_driver
        bi.name = "chrome"
        bi.url = "http://localhost:1234"
        bi._window_size = (768, 1024)
        bi._headless = True
        bi._settle_seconds = 1.0

        with mock.patch.object(BrowserInstance, "_create_driver") as mock_create:
            new_driver = mock.MagicMock()
            mock_create.return_value = new_driver

            bi.restart()

            old_driver.quit.assert_called_once()
            mock_create.assert_called_once()
            assert bi._driver is new_driver

    def test_restart_navigates_to_url(self):
        """restart() navigates the new driver to the original URL."""
        from scripts._smoke_utils import BrowserInstance

        bi = BrowserInstance.__new__(BrowserInstance)
        bi._driver = mock.MagicMock()
        bi.name = "chrome"
        bi.url = "http://localhost:1234"
        bi._window_size = (768, 1024)
        bi._headless = True
        bi._settle_seconds = 0.0

        with mock.patch.object(BrowserInstance, "_create_driver") as mock_create:
            new_driver = mock.MagicMock()
            mock_create.return_value = new_driver

            bi.restart()

            new_driver.get.assert_called_once_with("http://localhost:1234")

    def test_restart_survives_dead_old_driver(self):
        """restart() handles the case where old driver.quit() throws."""
        from scripts._smoke_utils import BrowserInstance

        bi = BrowserInstance.__new__(BrowserInstance)
        old_driver = mock.MagicMock()
        old_driver.quit.side_effect = Exception("already dead")
        bi._driver = old_driver
        bi.name = "chrome"
        bi.url = "http://localhost:1234"
        bi._window_size = (768, 1024)
        bi._headless = True
        bi._settle_seconds = 0.0

        with mock.patch.object(BrowserInstance, "_create_driver") as mock_create:
            new_driver = mock.MagicMock()
            mock_create.return_value = new_driver

            bi.restart()  # should not raise

            assert bi._driver is new_driver


# ===========================================================================
# BaseGameEnv crash recovery in step()
# ===========================================================================


class TestStepCrashRecovery:
    """Tests for browser crash recovery during step()."""

    @mock.patch("src.platform.base_env.time")
    def test_step_returns_terminal_transition_on_browser_crash(self, mock_time):
        """When browser crashes during step, return forced terminal transition."""
        browser_instance = mock.MagicMock()
        browser_instance.is_alive.return_value = False

        new_driver = mock.MagicMock()
        browser_instance.restart.return_value = new_driver

        env = _make_ready_env(
            headless=True,
            reward_mode="survival",
            browser_instance=browser_instance,
        )
        env._driver = mock.MagicMock()

        # Make capture crash (simulating browser tab crash)
        env._driver.execute_script.side_effect = Exception("tab crashed")
        env._driver.get_screenshot_as_png.side_effect = Exception("tab crashed")
        env._last_frame = None  # No cached frame

        obs, reward, terminated, truncated, info = env.step(_action())

        assert terminated is True
        assert "browser_crashed" in info
        assert info["browser_crashed"] is True

    @mock.patch("src.platform.base_env.time")
    def test_step_triggers_browser_restart_on_crash(self, mock_time):
        """When browser crashes, step() triggers browser restart."""
        browser_instance = mock.MagicMock()
        new_driver = mock.MagicMock()
        browser_instance.restart.return_value = new_driver
        browser_instance.driver = new_driver

        env = _make_ready_env(
            headless=True,
            reward_mode="survival",
            browser_instance=browser_instance,
        )
        env._driver = mock.MagicMock()

        # Make capture crash
        env._driver.execute_script.side_effect = Exception("tab crashed")
        env._driver.get_screenshot_as_png.side_effect = Exception("tab crashed")
        env._last_frame = None

        env.step(_action())

        browser_instance.restart.assert_called_once()

    @mock.patch("src.platform.base_env.time")
    def test_step_crash_without_browser_instance_raises(self, mock_time):
        """Without browser_instance, capture crash raises RuntimeError."""
        env = _make_ready_env(headless=True, reward_mode="survival")
        env._driver = mock.MagicMock()

        # Make capture crash
        env._driver.execute_script.side_effect = Exception("tab crashed")
        env._driver.get_screenshot_as_png.side_effect = Exception("tab crashed")
        env._last_frame = None

        with pytest.raises(RuntimeError):
            env.step(_action())

    @mock.patch("src.platform.base_env.time")
    def test_step_crash_uses_terminal_reward(self, mock_time):
        """Crash recovery uses terminal penalty for reward."""
        browser_instance = mock.MagicMock()
        new_driver = mock.MagicMock()
        browser_instance.restart.return_value = new_driver
        browser_instance.driver = new_driver

        env = _make_ready_env(
            headless=True,
            reward_mode="survival",
            survival_bonus=0.0,
            browser_instance=browser_instance,
        )
        env._driver = mock.MagicMock()

        env._driver.execute_script.side_effect = Exception("tab crashed")
        env._driver.get_screenshot_as_png.side_effect = Exception("tab crashed")
        env._last_frame = None

        _, reward, _, _, _ = env.step(_action())

        assert reward == pytest.approx(-5.0)

    @mock.patch("src.platform.base_env.time")
    def test_step_continues_after_crash_recovery(self, mock_time):
        """After crash recovery and reset, env can step again."""
        browser_instance = mock.MagicMock()
        new_driver = mock.MagicMock()
        browser_instance.restart.return_value = new_driver
        browser_instance.driver = new_driver

        env = _make_ready_env(
            headless=True,
            reward_mode="survival",
            browser_instance=browser_instance,
        )

        # First step: crash
        env._driver = mock.MagicMock()
        env._driver.execute_script.side_effect = Exception("tab crashed")
        env._driver.get_screenshot_as_png.side_effect = Exception("tab crashed")
        env._last_frame = None

        _, _, terminated, _, _ = env.step(_action())
        assert terminated is True

        # Verify driver was swapped
        assert env._driver is new_driver


class TestStepCrashRecoveryYoloMode:
    """Crash recovery in yolo reward mode."""

    @mock.patch("src.platform.base_env.time")
    def test_step_crash_yolo_mode_uses_terminal_reward(self, mock_time):
        """In yolo mode, crash recovery uses terminal_reward()."""
        browser_instance = mock.MagicMock()
        new_driver = mock.MagicMock()
        browser_instance.restart.return_value = new_driver
        browser_instance.driver = new_driver

        env = _make_ready_env(
            headless=True,
            reward_mode="yolo",
            browser_instance=browser_instance,
        )
        env._driver = mock.MagicMock()

        env._driver.execute_script.side_effect = Exception("tab crashed")
        env._driver.get_screenshot_as_png.side_effect = Exception("tab crashed")
        env._last_frame = None

        _, reward, terminated, _, _ = env.step(_action())

        assert terminated is True
        assert reward == pytest.approx(-5.0)  # StubEnv.terminal_reward()


# ===========================================================================
# BaseGameEnv crash recovery in step() — cached frame scenario
# ===========================================================================


class TestStepCrashRecoveryWithCachedFrame:
    """Tests for crash recovery when _last_frame is populated.

    This is the critical real-world scenario: in training, _last_frame
    is always populated after the first frame.  Before the fix,
    _capture_frame_headless() would silently return the stale cached
    frame instead of triggering crash recovery.
    """

    @_skip_no_cv2
    @mock.patch("src.platform.base_env.time")
    def test_step_detects_crash_with_cached_frame(self, mock_time):
        """When browser crashes and _last_frame exists, step() still detects the crash."""
        browser_instance = mock.MagicMock()
        browser_instance.is_alive.return_value = False
        new_driver = mock.MagicMock()
        browser_instance.restart.return_value = new_driver
        browser_instance.driver = new_driver

        env = _make_ready_env(
            headless=True,
            reward_mode="survival",
            browser_instance=browser_instance,
        )
        env._driver = mock.MagicMock()

        # _last_frame is populated (as it always is after first frame)
        env._last_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Make all capture attempts fail (tab crashed)
        env._driver.execute_script.side_effect = Exception("tab crashed")
        env._driver.get_screenshot_as_png.side_effect = Exception("tab crashed")

        obs, reward, terminated, truncated, info = env.step(_action())

        assert terminated is True
        assert info["browser_crashed"] is True
        browser_instance.restart.assert_called_once()

    @_skip_no_cv2
    @mock.patch("src.platform.base_env.time")
    def test_cached_frame_returned_when_browser_alive(self, mock_time):
        """When capture fails but browser is alive, return cached frame (transient error)."""
        browser_instance = mock.MagicMock()
        browser_instance.is_alive.return_value = True

        env = _make_ready_env(
            headless=True,
            reward_mode="survival",
            browser_instance=browser_instance,
        )
        env._driver = mock.MagicMock()

        cached_frame = np.ones((480, 640, 3), dtype=np.uint8) * 42
        env._last_frame = cached_frame.copy()

        # Capture fails but browser is alive
        env._driver.execute_script.side_effect = Exception("transient error")
        env._driver.get_screenshot_as_png.side_effect = Exception("transient error")

        obs, reward, terminated, truncated, info = env.step(_action())

        # Should NOT trigger crash recovery — just use cached frame
        browser_instance.restart.assert_not_called()
        assert info.get("browser_crashed") is not True

    @_skip_no_cv2
    @mock.patch("src.platform.base_env.time")
    def test_cached_frame_returned_without_browser_instance(self, mock_time):
        """Without browser_instance, capture failure returns cached frame."""
        env = _make_ready_env(headless=True, reward_mode="survival")
        env._driver = mock.MagicMock()

        cached_frame = np.ones((480, 640, 3), dtype=np.uint8) * 42
        env._last_frame = cached_frame.copy()

        # Capture fails, no browser_instance to check
        env._driver.execute_script.side_effect = Exception("transient error")
        env._driver.get_screenshot_as_png.side_effect = Exception("transient error")

        # Should not raise — falls back to cached frame
        obs, reward, terminated, truncated, info = env.step(_action())
        assert "browser_crashed" not in info or info.get("browser_crashed") is not True


# ===========================================================================
# _capture_frame_headless() direct tests
# ===========================================================================


class TestCaptureFrameHeadlessCrashDetection:
    """Direct tests for crash detection in _capture_frame_headless()."""

    @_skip_no_cv2
    def test_raises_when_browser_crashed_and_cached_frame_exists(self):
        """_capture_frame_headless raises RuntimeError when browser is dead."""
        browser_instance = mock.MagicMock()
        browser_instance.is_alive.return_value = False

        env = _make_ready_env(
            headless=True,
            reward_mode="survival",
            browser_instance=browser_instance,
        )
        env._driver = mock.MagicMock()
        env._last_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # All capture methods fail
        env._driver.execute_script.side_effect = Exception("tab crashed")
        env._driver.get_screenshot_as_png.side_effect = Exception("tab crashed")

        with pytest.raises(RuntimeError, match="Browser tab crashed"):
            env._capture_frame_headless()

    @_skip_no_cv2
    def test_returns_cached_frame_when_browser_alive(self):
        """_capture_frame_headless returns cached frame when browser is alive."""
        browser_instance = mock.MagicMock()
        browser_instance.is_alive.return_value = True

        env = _make_ready_env(
            headless=True,
            reward_mode="survival",
            browser_instance=browser_instance,
        )
        env._driver = mock.MagicMock()

        cached = np.ones((480, 640, 3), dtype=np.uint8) * 99
        env._last_frame = cached.copy()

        env._driver.execute_script.side_effect = Exception("transient")
        env._driver.get_screenshot_as_png.side_effect = Exception("transient")

        result = env._capture_frame_headless()
        np.testing.assert_array_equal(result, cached)

    @_skip_no_cv2
    def test_returns_cached_frame_without_browser_instance(self):
        """Without browser_instance, returns cached frame (can't check liveness)."""
        env = _make_ready_env(headless=True, reward_mode="survival")
        env._driver = mock.MagicMock()

        cached = np.ones((480, 640, 3), dtype=np.uint8) * 77
        env._last_frame = cached.copy()

        env._driver.execute_script.side_effect = Exception("error")
        env._driver.get_screenshot_as_png.side_effect = Exception("error")

        result = env._capture_frame_headless()
        np.testing.assert_array_equal(result, cached)

    @_skip_no_cv2
    def test_raises_when_no_cached_frame_and_no_browser_instance(self):
        """Without cached frame or browser_instance, raises RuntimeError."""
        env = _make_ready_env(headless=True, reward_mode="survival")
        env._driver = mock.MagicMock()
        env._last_frame = None

        env._driver.execute_script.side_effect = Exception("error")
        env._driver.get_screenshot_as_png.side_effect = Exception("error")

        with pytest.raises(RuntimeError, match="Failed to decode screenshot"):
            env._capture_frame_headless()


# ===========================================================================
# BaseGameEnv crash recovery in reset()
# ===========================================================================


class TestResetCrashRecovery:
    """Tests for browser crash recovery during reset()."""

    @mock.patch("src.platform.base_env.time")
    def test_reset_restarts_browser_on_capture_crash(self, mock_time):
        """When browser crashes during reset(), restart and retry."""
        browser_instance = mock.MagicMock()
        new_driver = mock.MagicMock()
        browser_instance.restart.return_value = new_driver
        browser_instance.driver = new_driver

        env = _make_ready_env(
            headless=True,
            reward_mode="survival",
            browser_instance=browser_instance,
        )
        env._driver = mock.MagicMock()

        # Make first capture crash, subsequent ones succeed
        call_count = {"n": 0}

        def _capture_with_crash():
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("browser crashed")
            return np.zeros((480, 640, 3), dtype=np.uint8)

        env._capture_frame = _capture_with_crash

        obs, info = env.reset()

        browser_instance.restart.assert_called_once()
        assert obs is not None


# ===========================================================================
# BaseGameEnv browser_instance parameter
# ===========================================================================


class TestBrowserInstanceParam:
    """Tests for the browser_instance parameter on BaseGameEnv."""

    def test_browser_instance_stored(self):
        """browser_instance parameter is stored on the env."""
        browser_instance = mock.MagicMock()
        env = StubEnv(browser_instance=browser_instance)
        assert env._browser_instance is browser_instance

    def test_browser_instance_default_none(self):
        """browser_instance defaults to None."""
        env = StubEnv()
        assert env._browser_instance is None

    def test_close_does_not_close_browser_instance(self):
        """close() does not close the browser instance (caller owns it)."""
        browser_instance = mock.MagicMock()
        env = _make_ready_env(browser_instance=browser_instance)
        env.close()
        browser_instance.close.assert_not_called()
