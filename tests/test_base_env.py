"""Tests for the BaseGameEnv abstract base class.

Verifies the generic lifecycle contract: ABC enforcement, step/reset flow,
modal throttling hooks, info construction, close/render, and oracle wiring.
A minimal concrete ``StubEnv`` subclass exercises the base without any
game-specific logic.
"""

from __future__ import annotations

from unittest import mock
from typing import Any

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
# Stub concrete subclass for testing
# ---------------------------------------------------------------------------


class StubEnv(BaseGameEnv):
    """Minimal concrete implementation of BaseGameEnv for tests."""

    _BALL_LOST_THRESHOLD = 5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Game-specific counters
        self._no_ball_count: int = 0
        self._no_bricks_count: int = 0

    def game_classes(self) -> list[str]:
        return ["ball", "paddle", "brick"]

    def build_observation(
        self, detections: dict[str, Any], *, reset: bool = False
    ) -> np.ndarray:
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
        pass  # no-op for tests

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
# Tests
# ===========================================================================


class TestABCEnforcement:
    """BaseGameEnv cannot be instantiated directly."""

    def test_cannot_instantiate_abc(self):
        """Instantiating BaseGameEnv raises TypeError."""
        with pytest.raises(TypeError, match="abstract"):
            BaseGameEnv()

    def test_missing_abstract_method_raises(self):
        """A subclass missing any abstract method raises TypeError."""

        class IncompleteEnv(BaseGameEnv):
            def game_classes(self):
                return []

        with pytest.raises(TypeError, match="abstract"):
            IncompleteEnv()

    def test_stub_env_instantiates(self):
        """StubEnv (all abstract methods implemented) instantiates fine."""
        env = StubEnv()
        assert isinstance(env, BaseGameEnv)
        assert isinstance(env, gym.Env)


class TestMakeInfo:
    """Tests for _make_info (adds frame + step to game-specific info)."""

    def test_make_info_adds_frame_and_step(self):
        """_make_info merges build_info with frame and step."""
        env = _make_ready_env()
        env._step_count = 42
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        env._last_frame = frame

        det = _make_detections()
        info = env._make_info(det)

        assert info["frame"] is frame
        assert info["step"] == 42
        assert info["ball_detected"] is True

    def test_make_info_step_zero(self):
        """After reset, step should be 0."""
        env = _make_ready_env()
        env._step_count = 0

        det = _make_detections()
        info = env._make_info(det)

        assert info["step"] == 0

    def test_make_info_no_ball(self):
        """build_info reports ball_detected=False when ball is None."""
        env = _make_ready_env()
        det = _make_detections(ball=None)
        info = env._make_info(det)

        assert info["ball_detected"] is False


class TestDefaultHooks:
    """Default _should_check_modals and _check_late_game_over."""

    def test_should_check_modals_default_true(self):
        """Default _should_check_modals returns True (always check)."""
        env = StubEnv()
        assert env._should_check_modals() is True

    def test_check_late_game_over_default_false(self):
        """Default _check_late_game_over returns False."""
        env = StubEnv()
        det = _make_detections()
        assert env._check_late_game_over(det) is False


class TestStep:
    """Tests for the base step() lifecycle."""

    @mock.patch("src.platform.base_env.time")
    def test_step_increments_counter(self, mock_time):
        """step() should increment _step_count."""
        env = _make_ready_env()
        assert env._step_count == 0

        env.step(_action())

        assert env._step_count == 1

    @mock.patch("src.platform.base_env.time")
    def test_step_returns_five_tuple(self, mock_time):
        """step() returns (obs, reward, terminated, truncated, info)."""
        env = _make_ready_env()

        result = env.step(_action())

        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    @mock.patch("src.platform.base_env.time")
    def test_step_normal_not_terminated(self, mock_time):
        """Normal step should not terminate."""
        env = _make_ready_env()

        _, _, terminated, truncated, _ = env.step(_action())

        assert terminated is False
        assert truncated is False

    @mock.patch("src.platform.base_env.time")
    def test_step_truncation_at_max_steps(self, mock_time):
        """step() truncates when max_steps reached."""
        env = _make_ready_env(max_steps=5)
        env._step_count = 4

        _, _, _, truncated, _ = env.step(_action())

        assert truncated is True

    @mock.patch("src.platform.base_env.time")
    def test_step_calls_apply_action(self, mock_time):
        """step() calls apply_action with the provided action."""
        env = _make_ready_env()

        with mock.patch.object(env, "apply_action") as mock_apply:
            env.step(_action(0.7))
            mock_apply.assert_called_once()

    @mock.patch("src.platform.base_env.time")
    def test_step_game_over_via_mid_modal(self, mock_time):
        """Mid-step game_over modal terminates with terminal_reward."""
        env = _make_ready_env()

        with (
            mock.patch.object(env, "_should_check_modals", return_value=True),
            mock.patch.object(env, "handle_modals", return_value="game_over"),
        ):
            _, reward, terminated, _, _ = env.step(_action())

        assert terminated is True
        assert reward == pytest.approx(-5.0)

    @mock.patch("src.platform.base_env.time")
    def test_step_perk_picker_not_terminal(self, mock_time):
        """Mid-step perk_picker modal is handled but not terminal."""
        env = _make_ready_env()

        with (
            mock.patch.object(env, "_should_check_modals", return_value=True),
            mock.patch.object(env, "handle_modals", return_value="perk_picker"),
            mock.patch.object(env, "start_game") as mock_start,
        ):
            _, _, terminated, _, _ = env.step(_action())

        assert terminated is False
        mock_start.assert_called_once()

    @mock.patch("src.platform.base_env.time")
    def test_step_late_game_over(self, mock_time):
        """Late game-over detection returns terminal_reward."""
        env = _make_ready_env()

        with mock.patch.object(env, "_check_late_game_over", return_value=True):
            _, reward, terminated, _, _ = env.step(_action())

        assert terminated is True
        assert reward == pytest.approx(-5.0)

    @mock.patch("src.platform.base_env.time")
    def test_step_late_game_over_skips_compute_reward(self, mock_time):
        """When late game-over fires, compute_reward is NOT called."""
        env = _make_ready_env()

        with (
            mock.patch.object(env, "_check_late_game_over", return_value=True),
            mock.patch.object(env, "compute_reward") as mock_cr,
        ):
            env.step(_action())
            mock_cr.assert_not_called()

    @mock.patch("src.platform.base_env.time")
    def test_step_info_includes_oracle_findings(self, mock_time):
        """step() info dict contains oracle_findings."""
        env = _make_ready_env()
        _, _, _, _, info = env.step(_action())
        assert "oracle_findings" in info

    @mock.patch("src.platform.base_env.time")
    def test_step_skips_modal_check_when_should_returns_false(self, mock_time):
        """When _should_check_modals returns False, handle_modals is not called."""
        env = _make_ready_env()

        with (
            mock.patch.object(env, "_should_check_modals", return_value=False),
            mock.patch.object(env, "handle_modals") as mock_hm,
        ):
            env.step(_action())
            mock_hm.assert_not_called()


class TestReset:
    """Tests for the base reset() lifecycle."""

    @mock.patch("src.platform.base_env.time")
    def test_reset_returns_obs_and_info(self, mock_time):
        """reset() returns (obs, info) tuple."""
        env = _make_ready_env()

        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4,)
        assert isinstance(info, dict)

    @mock.patch("src.platform.base_env.time")
    def test_reset_resets_step_count(self, mock_time):
        """reset() sets _step_count to 0."""
        env = _make_ready_env()
        env._step_count = 42

        env.reset()

        assert env._step_count == 0

    @mock.patch("src.platform.base_env.time")
    def test_reset_calls_reset_termination_state(self, mock_time):
        """reset() calls reset_termination_state."""
        env = _make_ready_env()
        env._no_ball_count = 99

        env.reset()

        assert env._no_ball_count == 0

    @mock.patch("src.platform.base_env.time")
    def test_reset_raises_after_5_invalid_attempts(self, mock_time):
        """reset() raises RuntimeError if on_reset_detections always False."""
        env = _make_ready_env()
        # Return no ball in detections
        env._detector.detect_to_game_state.return_value = _make_detections(ball=None)

        with pytest.raises(RuntimeError, match="failed to get valid detections"):
            env.reset()

    @mock.patch("src.platform.base_env.time")
    def test_reset_calls_handle_modals_and_start_game(self, mock_time):
        """reset() calls handle_modals and start_game each attempt."""
        env = _make_ready_env()

        with (
            mock.patch.object(env, "handle_modals") as mock_hm,
            mock.patch.object(env, "start_game") as mock_sg,
        ):
            env.reset()
            mock_hm.assert_called()
            mock_sg.assert_called()

    @mock.patch("src.platform.base_env.time")
    def test_reset_clears_oracles(self, mock_time):
        """reset() calls clear() then on_reset() on all oracles."""
        env = _make_ready_env()
        oracle = mock.MagicMock()
        oracle.get_findings.return_value = []
        env._oracles = [oracle]

        env.reset()

        oracle.clear.assert_called_once()
        oracle.on_reset.assert_called_once()

    @mock.patch("src.platform.base_env.time")
    def test_reset_calls_on_reset_complete(self, mock_time):
        """reset() calls on_reset_complete hook."""
        env = _make_ready_env()

        with mock.patch.object(env, "on_reset_complete") as mock_hook:
            env.reset()
            mock_hook.assert_called_once()

    @mock.patch("src.platform.base_env.time")
    def test_reset_info_has_frame_and_step(self, mock_time):
        """reset() info dict has frame and step=0."""
        env = _make_ready_env()

        _, info = env.reset()

        assert "frame" in info
        assert info["step"] == 0


class TestClose:
    """Tests for close() resource cleanup."""

    def test_close_clears_state(self):
        """close() clears capture, detector, canvas, initialized."""
        env = _make_ready_env()

        env.close()

        assert env._capture is None
        assert env._detector is None
        assert env._game_canvas is None
        assert env._canvas_size is None
        assert env._initialized is False

    def test_close_calls_capture_release(self):
        """close() calls _capture.release()."""
        env = _make_ready_env()
        mock_capture = env._capture

        env.close()

        mock_capture.release.assert_called_once()

    def test_close_without_capture_no_error(self):
        """close() when _capture is None does not raise."""
        env = StubEnv()
        env._capture = None
        env.close()  # should not raise


class TestRender:
    """Tests for render()."""

    def test_render_rgb_array(self):
        """render_mode='rgb_array' returns the last frame."""
        env = StubEnv(render_mode="rgb_array")
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        env._last_frame = frame

        result = env.render()

        assert result is frame

    def test_render_human_returns_none(self):
        """render_mode='human' returns None."""
        env = StubEnv(render_mode="human")
        env._last_frame = np.zeros((10, 10, 3), dtype=np.uint8)

        assert env.render() is None

    def test_render_no_mode_returns_none(self):
        """No render_mode returns None."""
        env = StubEnv()
        env._last_frame = np.zeros((10, 10, 3), dtype=np.uint8)

        assert env.render() is None


class TestStepCount:
    """Tests for the step_count property."""

    def test_step_count_property(self):
        """step_count is a read-only property."""
        env = StubEnv()
        env._step_count = 7

        assert env.step_count == 7


class TestRunOracles:
    """Tests for _run_oracles oracle wiring."""

    def test_calls_on_step_on_all_oracles(self):
        """_run_oracles calls on_step on every oracle."""
        env = StubEnv()
        o1 = mock.MagicMock()
        o1.get_findings.return_value = []
        o2 = mock.MagicMock()
        o2.get_findings.return_value = [{"type": "bug"}]
        env._oracles = [o1, o2]

        obs = np.zeros(4, dtype=np.float32)
        info = {"frame": None}
        findings = env._run_oracles(obs, 0.5, False, False, info)

        o1.on_step.assert_called_once_with(obs, 0.5, False, False, info)
        o2.on_step.assert_called_once_with(obs, 0.5, False, False, info)
        assert findings == [{"type": "bug"}]

    def test_no_oracles_returns_empty(self):
        """_run_oracles with no oracles returns empty list."""
        env = StubEnv()
        env._oracles = []

        obs = np.zeros(4, dtype=np.float32)
        findings = env._run_oracles(obs, 0.0, False, False, {})

        assert findings == []


class TestOptionalHooks:
    """Default optional hooks are no-ops."""

    def test_on_lazy_init_default(self):
        """on_lazy_init does nothing by default."""
        env = StubEnv()
        env.on_lazy_init()  # should not raise

    def test_on_reset_complete_default(self):
        """on_reset_complete does nothing by default."""
        env = StubEnv()
        obs = np.zeros(4, dtype=np.float32)
        env.on_reset_complete(obs, {})  # should not raise


class TestDismissAllAlerts:
    """Tests for _dismiss_all_alerts helper."""

    def test_dismiss_single_alert(self):
        """Dismisses a single alert and returns 1."""
        env = StubEnv(headless=True)
        mock_driver = mock.MagicMock()
        env._driver = mock_driver

        mock_alert = mock.MagicMock()
        mock_alert.text = "Some alert"
        # First access returns the alert, second raises (no more alerts)
        type(mock_driver.switch_to).alert = mock.PropertyMock(
            side_effect=[mock_alert, Exception("no alert")]
        )

        dismissed = env._dismiss_all_alerts()
        assert dismissed == 1
        mock_alert.dismiss.assert_called_once()

    def test_dismiss_multiple_alerts(self):
        """Dismisses multiple alerts in a loop."""
        env = StubEnv(headless=True)
        mock_driver = mock.MagicMock()
        env._driver = mock_driver

        alert1 = mock.MagicMock(text="First alert")
        alert2 = mock.MagicMock(text="Second alert")
        alert3 = mock.MagicMock(text="Third alert")
        type(mock_driver.switch_to).alert = mock.PropertyMock(
            side_effect=[alert1, alert2, alert3, Exception("no alert")]
        )

        dismissed = env._dismiss_all_alerts()
        assert dismissed == 3
        alert1.dismiss.assert_called_once()
        alert2.dismiss.assert_called_once()
        alert3.dismiss.assert_called_once()

    def test_dismiss_no_alerts(self):
        """Returns 0 when no alerts present."""
        env = StubEnv(headless=True)
        mock_driver = mock.MagicMock()
        env._driver = mock_driver

        type(mock_driver.switch_to).alert = mock.PropertyMock(
            side_effect=Exception("no alert")
        )

        dismissed = env._dismiss_all_alerts()
        assert dismissed == 0

    def test_dismiss_respects_max_attempts(self):
        """Stops after max_attempts even if alerts keep appearing."""
        env = StubEnv(headless=True)
        mock_driver = mock.MagicMock()
        env._driver = mock_driver

        # Infinite alerts — should stop at max_attempts
        mock_alert = mock.MagicMock(text="Infinite alert")
        mock_driver.switch_to.alert = mock_alert

        dismissed = env._dismiss_all_alerts(max_attempts=3)
        assert dismissed == 3
        assert mock_alert.dismiss.call_count == 3

    def test_dismiss_no_driver_returns_zero(self):
        """Returns 0 when driver is None."""
        env = StubEnv(headless=True)
        env._driver = None

        dismissed = env._dismiss_all_alerts()
        assert dismissed == 0


class TestHeadlessCapture:
    """Tests for headless frame capture."""

    def test_headless_requires_driver(self):
        """Headless mode raises if driver is None on lazy init."""
        env = StubEnv(headless=True)

        with pytest.raises(RuntimeError, match="Headless mode requires"):
            env._lazy_init()

    def test_headless_capture_no_driver_raises(self):
        """_capture_frame_headless raises if driver is None."""
        env = StubEnv(headless=True)
        env._driver = None

        with pytest.raises(RuntimeError, match="driver is None"):
            env._capture_frame_headless()

    @_skip_no_cv2
    def test_headless_capture_dismisses_alert(self):
        """_capture_frame_headless dismisses unexpected browser alerts."""
        import cv2

        env = StubEnv(headless=True)
        mock_driver = mock.MagicMock()
        env._driver = mock_driver

        # Simulate an alert present, then no more
        mock_alert = mock.MagicMock()
        mock_alert.text = "Two alerts where opened at once"
        type(mock_driver.switch_to).alert = mock.PropertyMock(
            side_effect=[mock_alert, Exception("no alert")]
        )

        # After dismissal, screenshot works
        fake_img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, png_bytes = cv2.imencode(".png", fake_img)
        mock_driver.get_screenshot_as_png.return_value = png_bytes.tobytes()

        frame = env._capture_frame_headless()
        mock_alert.dismiss.assert_called_once()
        assert frame is not None
        assert frame.shape[0] > 0

    @_skip_no_cv2
    def test_headless_capture_no_alert_present(self):
        """_capture_frame_headless works normally when no alert is present."""
        import cv2

        env = StubEnv(headless=True)
        mock_driver = mock.MagicMock()
        env._driver = mock_driver

        # No alert — accessing switch_to.alert raises
        type(mock_driver.switch_to).alert = mock.PropertyMock(
            side_effect=Exception("no alert present")
        )

        fake_img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, png_bytes = cv2.imencode(".png", fake_img)
        mock_driver.get_screenshot_as_png.return_value = png_bytes.tobytes()

        frame = env._capture_frame_headless()
        assert frame is not None

    @_skip_no_cv2
    def test_headless_capture_retries_after_screenshot_alert(self):
        """Screenshot fails with alert, dismiss-all and retry succeeds."""
        import cv2

        env = StubEnv(headless=True)
        mock_driver = mock.MagicMock()
        env._driver = mock_driver

        # First _dismiss_all_alerts: no alert.  Second (in retry): one alert.
        mock_retry_alert = mock.MagicMock(text="popup")
        type(mock_driver.switch_to).alert = mock.PropertyMock(
            side_effect=[
                Exception("no alert"),  # preemptive check
                mock_retry_alert,  # retry dismiss finds alert
                Exception("no more"),  # retry dismiss loop ends
            ]
        )

        # canvas toDataURL fails (unmocked execute_script) so we fall to screenshot
        # First screenshot raises, second (after dismiss) succeeds
        fake_img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, png_bytes = cv2.imencode(".png", fake_img)

        mock_driver.get_screenshot_as_png.side_effect = [
            Exception("alert open"),
            png_bytes.tobytes(),
        ]

        frame = env._capture_frame_headless()
        assert frame is not None
        assert mock_driver.get_screenshot_as_png.call_count == 2

    @_skip_no_cv2
    def test_headless_capture_falls_back_to_cached_frame(self):
        """Returns cached frame when all capture attempts fail."""
        env = StubEnv(headless=True)
        mock_driver = mock.MagicMock()
        env._driver = mock_driver

        # No alerts
        type(mock_driver.switch_to).alert = mock.PropertyMock(
            side_effect=Exception("no alert")
        )

        # All capture methods fail
        mock_driver.execute_script.side_effect = Exception("toDataURL failed")
        mock_driver.get_screenshot_as_png.side_effect = Exception("screenshot failed")

        # Set a cached frame
        cached = np.zeros((50, 50, 3), dtype=np.uint8)
        env._last_frame = cached

        frame = env._capture_frame_headless()
        assert frame is cached

    @_skip_no_cv2
    def test_headless_capture_raises_when_no_cached_frame(self):
        """Raises RuntimeError when all attempts fail and no cached frame."""
        env = StubEnv(headless=True)
        mock_driver = mock.MagicMock()
        env._driver = mock_driver

        # No alerts
        type(mock_driver.switch_to).alert = mock.PropertyMock(
            side_effect=Exception("no alert")
        )

        # All capture methods fail
        mock_driver.execute_script.side_effect = Exception("toDataURL failed")
        mock_driver.get_screenshot_as_png.side_effect = Exception("screenshot failed")
        env._last_frame = None

        with pytest.raises(RuntimeError, match="Failed to decode"):
            env._capture_frame_headless()


class TestLazyInitGameClasses:
    """Tests that game_classes() is wired into YoloDetector during _lazy_init."""

    def test_game_classes_passed_to_yolo_detector(self):
        """_lazy_init passes game_classes() to YoloDetector constructor."""
        env = StubEnv(headless=True)
        mock_driver = mock.MagicMock()
        env._driver = mock_driver

        with mock.patch("src.perception.yolo_detector.YoloDetector") as MockDetector:
            mock_instance = mock.MagicMock()
            MockDetector.return_value = mock_instance

            env._lazy_init()

            MockDetector.assert_called_once()
            call_kwargs = MockDetector.call_args
            assert call_kwargs.kwargs.get("classes") == ["ball", "paddle", "brick"]

    def test_game_classes_custom_values(self):
        """A subclass with different game_classes() passes them through."""

        class CustomClassesEnv(StubEnv):
            def game_classes(self) -> list[str]:
                return ["enemy", "player", "obstacle", "coin"]

        env = CustomClassesEnv(headless=True)
        mock_driver = mock.MagicMock()
        env._driver = mock_driver

        with mock.patch("src.perception.yolo_detector.YoloDetector") as MockDetector:
            mock_instance = mock.MagicMock()
            MockDetector.return_value = mock_instance

            env._lazy_init()

            call_kwargs = MockDetector.call_args
            assert call_kwargs.kwargs.get("classes") == [
                "enemy",
                "player",
                "obstacle",
                "coin",
            ]
