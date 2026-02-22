"""Tests for the BaseGameEnv abstract base class.

Verifies the generic lifecycle contract: ABC enforcement, step/reset flow,
modal throttling hooks, info construction, close/render, and oracle wiring.
A minimal concrete ``StubEnv`` subclass exercises the base without any
game-specific logic.
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
# Stub concrete subclass for testing
# ---------------------------------------------------------------------------


class StubEnv(BaseGameEnv):
    """Minimal concrete implementation of BaseGameEnv for tests."""

    _BALL_LOST_THRESHOLD = 5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Game-specific counters
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
    def test_step_perk_picker_calls_level_transition(self, mock_time):
        """Mid-step perk_picker modal routes through _handle_level_transition."""
        env = _make_ready_env(reward_mode="survival")

        with (
            mock.patch.object(env, "_should_check_modals", return_value=True),
            mock.patch.object(env, "handle_modals", return_value="perk_picker"),
            mock.patch.object(
                env, "_handle_level_transition", return_value=True
            ) as mock_transition,
        ):
            _, reward, terminated, _, _ = env.step(_action())

        assert terminated is False
        mock_transition.assert_called_once()
        # Non-terminal level clear gives +1.0 bonus in survival mode
        assert reward == pytest.approx(1.01)

    @mock.patch("src.platform.base_env.time")
    def test_step_perk_picker_fallback_when_transition_fails(self, mock_time):
        """When _handle_level_transition returns False, falls back to start_game."""
        env = _make_ready_env(reward_mode="survival")

        with (
            mock.patch.object(env, "_should_check_modals", return_value=True),
            mock.patch.object(env, "handle_modals", return_value="perk_picker"),
            mock.patch.object(env, "_handle_level_transition", return_value=False),
            mock.patch.object(env, "start_game") as mock_start,
        ):
            _, reward, terminated, _, _ = env.step(_action())

        assert terminated is False
        mock_start.assert_called_once()
        # No level clear bonus — transition failed
        assert reward == pytest.approx(0.01)

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

        type(mock_driver.switch_to).alert = mock.PropertyMock(side_effect=Exception("no alert"))

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
        type(mock_driver.switch_to).alert = mock.PropertyMock(side_effect=Exception("no alert"))

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
        type(mock_driver.switch_to).alert = mock.PropertyMock(side_effect=Exception("no alert"))

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


# ===========================================================================
# Reward mode tests
# ===========================================================================


class TestRewardModeValidation:
    """Tests for reward_mode parameter validation."""

    def test_default_reward_mode_is_yolo(self):
        """Default reward_mode is 'yolo'."""
        env = StubEnv()
        assert env.reward_mode == "yolo"

    def test_survival_mode_accepted(self):
        """reward_mode='survival' is valid."""
        env = StubEnv(reward_mode="survival")
        assert env.reward_mode == "survival"

    def test_invalid_reward_mode_raises(self):
        """Invalid reward_mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid reward_mode"):
            StubEnv(reward_mode="invalid")

    def test_yolo_mode_accepted(self):
        """reward_mode='yolo' is valid."""
        env = StubEnv(reward_mode="yolo")
        assert env.reward_mode == "yolo"


class TestSurvivalReward:
    """Tests for _compute_survival_reward and survival reward mode in step()."""

    def test_survival_reward_per_step(self):
        """Survival mode gives +0.01 per step."""
        env = StubEnv(reward_mode="survival")
        reward = env._compute_survival_reward(terminated=False, level_cleared=False)
        assert abs(reward - 0.01) < 1e-6

    def test_survival_reward_game_over(self):
        """Survival mode gives -5.0 + 0.01 = -4.99 on game over."""
        env = StubEnv(reward_mode="survival")
        reward = env._compute_survival_reward(terminated=True, level_cleared=False)
        assert abs(reward - (-4.99)) < 1e-6

    def test_survival_reward_level_clear(self):
        """Survival mode gives +5.0 + 0.01 = 5.01 on terminal level clear."""
        env = StubEnv(reward_mode="survival")
        reward = env._compute_survival_reward(terminated=True, level_cleared=True)
        assert abs(reward - 5.01) < 1e-6

    def test_survival_reward_non_terminal_level_clear(self):
        """Survival mode gives +1.0 + 0.01 = 1.01 on non-terminal level clear."""
        env = StubEnv(reward_mode="survival")
        reward = env._compute_survival_reward(terminated=False, level_cleared=True)
        assert abs(reward - 1.01) < 1e-6

    def test_survival_terminal_reward_constant(self):
        """Survival terminal reward is -5.01."""
        env = StubEnv(reward_mode="survival")
        assert abs(env._SURVIVAL_TERMINAL_REWARD - (-5.01)) < 1e-6

    def test_step_uses_survival_reward_when_mode_set(self):
        """step() uses survival reward (not game-specific) when mode='survival'."""
        env = _make_ready_env(reward_mode="survival")
        # StubEnv.compute_reward returns 0.01 for non-terminal steps
        # Survival mode should also give 0.01 (same value, but from different source)
        # The key test: if we make compute_reward return something different,
        # survival mode should still give 0.01
        env.compute_reward = lambda *a, **kw: 999.0  # noisy YOLO reward
        obs, reward, terminated, truncated, info = env.step(_action())
        assert abs(reward - 0.01) < 1e-6  # survival, not 999.0

    def test_step_uses_yolo_reward_by_default(self):
        """step() uses game-specific compute_reward() in yolo mode."""
        env = _make_ready_env(reward_mode="yolo")
        env.compute_reward = lambda *a, **kw: 42.0
        obs, reward, terminated, truncated, info = env.step(_action())
        assert abs(reward - 42.0) < 1e-6

    def test_survival_mid_step_game_over_uses_survival_terminal(self):
        """Mid-step game-over in survival mode uses _SURVIVAL_TERMINAL_REWARD."""
        env = _make_ready_env(reward_mode="survival")
        env.handle_modals = lambda **kw: "game_over"
        env._no_ball_count = 1  # triggers _should_check_modals -> True
        obs, reward, terminated, truncated, info = env.step(_action())
        assert terminated is True
        assert abs(reward - (-5.01)) < 1e-6

    def test_yolo_mid_step_game_over_uses_game_terminal(self):
        """Mid-step game-over in yolo mode uses terminal_reward()."""
        env = _make_ready_env(reward_mode="yolo")
        env.handle_modals = lambda **kw: "game_over"
        env._no_ball_count = 1
        obs, reward, terminated, truncated, info = env.step(_action())
        assert terminated is True
        assert abs(reward - (-5.0)) < 1e-6  # StubEnv.terminal_reward() = -5.0

    def test_survival_late_game_over_uses_survival_terminal(self):
        """Late game-over (ball disappears) in survival mode uses survival terminal."""
        env = _make_ready_env(reward_mode="survival")
        # Make detection return no ball
        no_ball_det = _make_detections(ball=None)
        env._detector.detect_to_game_state.return_value = no_ball_det
        # Make handle_modals return game_over when called with dismiss=False
        env.handle_modals = lambda **kw: (
            "game_over" if not kw.get("dismiss_game_over", True) else "gameplay"
        )
        obs, reward, terminated, truncated, info = env.step(_action())
        assert terminated is True
        assert abs(reward - (-5.01)) < 1e-6

    def test_step_ignores_level_cleared_in_survival_mode(self):
        """In survival mode, level_cleared from check_termination is ignored.

        YOLO brick detection is unreliable in headless mode and can
        return 0 bricks, causing false level_cleared=True.  In survival
        mode the step should continue (not terminate) even when
        check_termination reports level_cleared.
        """
        env = _make_ready_env(reward_mode="survival")
        # Make check_termination return level_cleared=True
        env.check_termination = lambda detections: (True, True)
        obs, reward, terminated, truncated, info = env.step(_action())
        # Should NOT terminate — level_cleared is ignored in survival mode
        assert terminated is False
        # Reward should be survival per-step (+0.01), not level-clear bonus
        assert abs(reward - 0.01) < 1e-6

    def test_step_respects_level_cleared_in_yolo_mode(self):
        """In yolo mode, level_cleared from check_termination is respected."""
        env = _make_ready_env(reward_mode="yolo")
        # Make check_termination return level_cleared=True
        env.check_termination = lambda detections: (True, True)
        env.compute_reward = (
            lambda detections, terminated, level_cleared: 5.0 if level_cleared else -1.0
        )
        obs, reward, terminated, truncated, info = env.step(_action())
        # Should terminate — level_cleared is respected in yolo mode
        assert terminated is True
        assert abs(reward - 5.0) < 1e-6

    def test_step_game_over_still_terminates_in_survival_mode(self):
        """In survival mode, game_over from check_termination still works.

        Only level_cleared is ignored; game_over (ball lost) should
        still terminate the episode with a negative reward.
        """
        env = _make_ready_env(reward_mode="survival")
        # Make check_termination return game_over=True, level_cleared=False
        env.check_termination = lambda detections: (True, False)
        obs, reward, terminated, truncated, info = env.step(_action())
        # Should terminate — game_over is respected
        assert terminated is True
        # Reward should be game-over penalty: 0.01 - 5.0 = -4.99
        assert abs(reward - (-4.99)) < 1e-6


class TestSurvivalResetDetections:
    """Tests for lenient reset detection validation in survival mode."""

    @mock.patch("src.platform.base_env.time")
    def test_reset_succeeds_in_survival_mode_when_no_ball_detected(self, mock_time):
        """reset() succeeds in survival mode even when YOLO can't detect ball.

        In survival mode with CNN policy, YOLO detections are not needed
        for observations or reward computation. The reset should accept
        any frame with a valid capture, bypassing on_reset_detections().
        """
        env = _make_ready_env(reward_mode="survival")
        # Simulate YOLO failing to detect ball (common in headless mode)
        env._detector.detect_to_game_state.return_value = _make_detections(ball=None)

        # Should NOT raise — survival mode is lenient
        obs, info = env.reset()
        assert obs is not None

    @mock.patch("src.platform.base_env.time")
    def test_reset_still_validates_in_yolo_mode(self, mock_time):
        """reset() still requires valid detections in yolo mode."""
        env = _make_ready_env(reward_mode="yolo")
        env._detector.detect_to_game_state.return_value = _make_detections(ball=None)

        with pytest.raises(RuntimeError, match="failed to get valid detections"):
            env.reset()

    @mock.patch("src.platform.base_env.time")
    def test_reset_prefers_valid_detections_even_in_survival_mode(self, mock_time):
        """reset() still tries for valid detections in survival mode.

        If valid detections are available, survival mode uses them.
        The leniency only matters when all 5 attempts fail.
        """
        env = _make_ready_env(reward_mode="survival")
        # First call: no ball; second call: ball detected
        env._detector.detect_to_game_state.side_effect = [
            _make_detections(ball=None),
            _make_detections(ball=(0.5, 0.5, 0.02, 0.02)),
        ]

        obs, info = env.reset()
        assert obs is not None
        # Should have tried 2 times (found valid on 2nd attempt)
        assert env._detector.detect_to_game_state.call_count == 2


# ===========================================================================
# Survival bonus parameter
# ===========================================================================


class TestSurvivalBonus:
    """Tests for the configurable survival_bonus parameter."""

    def test_default_survival_bonus_is_001(self):
        """Default survival_bonus is 0.01."""
        env = StubEnv(reward_mode="survival")
        assert env._survival_bonus == 0.01

    def test_custom_survival_bonus_zero(self):
        """survival_bonus=0.0 is accepted and stored."""
        env = StubEnv(reward_mode="survival", survival_bonus=0.0)
        assert env._survival_bonus == 0.0

    def test_custom_survival_bonus_positive(self):
        """survival_bonus=0.05 is accepted and stored."""
        env = StubEnv(reward_mode="survival", survival_bonus=0.05)
        assert env._survival_bonus == 0.05

    def test_survival_reward_per_step_uses_custom_bonus(self):
        """Per-step reward uses the configured survival_bonus."""
        env = StubEnv(reward_mode="survival", survival_bonus=0.0)
        reward = env._compute_survival_reward(terminated=False, level_cleared=False)
        assert abs(reward - 0.0) < 1e-6

    def test_survival_reward_game_over_with_zero_bonus(self):
        """Game over with survival_bonus=0.0 gives -5.0."""
        env = StubEnv(reward_mode="survival", survival_bonus=0.0)
        reward = env._compute_survival_reward(terminated=True, level_cleared=False)
        assert abs(reward - (-5.0)) < 1e-6

    def test_survival_reward_level_clear_with_zero_bonus(self):
        """Level clear with survival_bonus=0.0 gives +5.0."""
        env = StubEnv(reward_mode="survival", survival_bonus=0.0)
        reward = env._compute_survival_reward(terminated=True, level_cleared=True)
        assert abs(reward - 5.0) < 1e-6

    def test_survival_reward_non_terminal_level_clear_with_zero_bonus(self):
        """Non-terminal level clear with survival_bonus=0.0 gives +1.0."""
        env = StubEnv(reward_mode="survival", survival_bonus=0.0)
        reward = env._compute_survival_reward(terminated=False, level_cleared=True)
        assert abs(reward - 1.0) < 1e-6

    def test_terminal_reward_property_adapts_to_bonus(self):
        """_SURVIVAL_TERMINAL_REWARD adapts to survival_bonus value."""
        env_default = StubEnv(reward_mode="survival", survival_bonus=0.01)
        assert abs(env_default._SURVIVAL_TERMINAL_REWARD - (-5.01)) < 1e-6

        env_zero = StubEnv(reward_mode="survival", survival_bonus=0.0)
        assert abs(env_zero._SURVIVAL_TERMINAL_REWARD - (-5.0)) < 1e-6

        env_custom = StubEnv(reward_mode="survival", survival_bonus=0.05)
        assert abs(env_custom._SURVIVAL_TERMINAL_REWARD - (-5.05)) < 1e-6

    def test_step_uses_custom_survival_bonus(self):
        """Full step() cycle uses the configured survival_bonus."""
        env = _make_ready_env(reward_mode="survival", survival_bonus=0.0)
        env.compute_reward = lambda *a, **kw: 999.0  # would be used in yolo mode
        obs, reward, terminated, truncated, info = env.step(_action())
        assert abs(reward - 0.0) < 1e-6  # survival_bonus=0.0, not 999.0

    def test_mid_step_game_over_uses_adapted_terminal_reward(self):
        """Mid-step game-over with survival_bonus=0.0 gives -5.0 (not -5.01)."""
        env = _make_ready_env(reward_mode="survival", survival_bonus=0.0)
        env.handle_modals = lambda **kw: "game_over"
        env._no_ball_count = 1  # triggers _should_check_modals -> True
        obs, reward, terminated, truncated, info = env.step(_action())
        assert terminated is True
        assert abs(reward - (-5.0)) < 1e-6


# ===========================================================================
# Score reward mode
# ===========================================================================


class TestScoreRewardMode:
    """Tests for reward_mode='score' — OCR-based score delta reward."""

    def test_score_mode_accepted(self):
        """reward_mode='score' is a valid mode."""
        env = StubEnv(reward_mode="score")
        assert env.reward_mode == "score"

    def test_score_mode_creates_score_ocr_instance(self):
        """When reward_mode='score', a ScoreOCR is created."""
        env = StubEnv(reward_mode="score")
        assert env._score_ocr is not None

    def test_non_score_mode_has_no_score_ocr(self):
        """When reward_mode != 'score', _score_ocr is None."""
        env = StubEnv(reward_mode="survival")
        assert env._score_ocr is None
        env2 = StubEnv(reward_mode="yolo")
        assert env2._score_ocr is None

    def test_score_region_passed_to_ocr(self):
        """score_region parameter is forwarded to ScoreOCR."""
        env = StubEnv(reward_mode="score", score_region=(10, 20, 100, 30))
        assert env._score_ocr._region == (10, 20, 100, 30)

    def test_score_ocr_interval_passed_to_ocr(self):
        """score_ocr_interval parameter is forwarded to ScoreOCR."""
        env = StubEnv(reward_mode="score", score_ocr_interval=5)
        assert env._score_ocr._interval == 5

    def test_score_reward_coeff_stored(self):
        """score_reward_coeff parameter is stored on the env."""
        env = StubEnv(reward_mode="score", score_reward_coeff=0.05)
        assert env._score_reward_coeff == 0.05

    def test_default_score_reward_coeff_is_001(self):
        """Default score_reward_coeff is 0.01."""
        env = StubEnv(reward_mode="score")
        assert env._score_reward_coeff == 0.01


class TestComputeScoreReward:
    """Tests for _compute_score_reward method."""

    def test_score_reward_baseline_is_survival(self):
        """Score reward includes survival reward as baseline."""
        env = StubEnv(reward_mode="score")
        # No frame → OCR can't run → reward = survival only
        env._last_frame = None
        reward = env._compute_score_reward(terminated=False, level_cleared=False)
        # Should equal survival reward: 0.01
        assert abs(reward - 0.01) < 1e-6

    def test_score_reward_game_over_is_survival_terminal(self):
        """Score reward on game over uses survival terminal reward."""
        env = StubEnv(reward_mode="score")
        env._last_frame = None
        reward = env._compute_score_reward(terminated=True, level_cleared=False)
        # Should equal survival game-over: -5.0 + 0.01 = -4.99
        assert abs(reward - (-4.99)) < 1e-6

    def test_score_reward_adds_positive_delta(self):
        """Positive score delta is added to survival baseline."""
        env = StubEnv(reward_mode="score", score_reward_coeff=0.01)
        env._last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        env._prev_ocr_score = 100
        # Mock the ScoreOCR to return 200 (delta = 100)
        env._score_ocr = mock.MagicMock()
        env._score_ocr.read_score.return_value = 200
        reward = env._compute_score_reward(terminated=False, level_cleared=False)
        # Expected: 0.01 (survival) + 100 * 0.01 (delta) = 1.01
        assert abs(reward - 1.01) < 1e-6

    def test_score_reward_ignores_negative_delta(self):
        """Negative score delta is ignored (no penalty for score decreasing)."""
        env = StubEnv(reward_mode="score", score_reward_coeff=0.01)
        env._last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        env._prev_ocr_score = 200
        env._score_ocr = mock.MagicMock()
        env._score_ocr.read_score.return_value = 100  # delta = -100
        reward = env._compute_score_reward(terminated=False, level_cleared=False)
        # Expected: 0.01 (survival only, no negative delta applied)
        assert abs(reward - 0.01) < 1e-6

    def test_score_reward_no_previous_score_no_delta(self):
        """When prev_ocr_score is None, no delta is added."""
        env = StubEnv(reward_mode="score", score_reward_coeff=0.01)
        env._last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        env._prev_ocr_score = None  # no previous score yet
        env._score_ocr = mock.MagicMock()
        env._score_ocr.read_score.return_value = 500
        reward = env._compute_score_reward(terminated=False, level_cleared=False)
        # Expected: 0.01 (survival only, no delta because prev was None)
        assert abs(reward - 0.01) < 1e-6
        # But prev_ocr_score should now be updated
        assert env._prev_ocr_score == 500

    def test_score_reward_ocr_returns_none_uses_cached(self):
        """When OCR returns None, prev_ocr_score is not updated."""
        env = StubEnv(reward_mode="score", score_reward_coeff=0.01)
        env._last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        env._prev_ocr_score = 100
        env._score_ocr = mock.MagicMock()
        env._score_ocr.read_score.return_value = None
        reward = env._compute_score_reward(terminated=False, level_cleared=False)
        # No delta applied (current is None)
        assert abs(reward - 0.01) < 1e-6
        # prev_ocr_score should remain 100 (not overwritten with None)
        assert env._prev_ocr_score == 100

    def test_score_reward_coeff_scales_delta(self):
        """Score reward delta is multiplied by score_reward_coeff."""
        env = StubEnv(reward_mode="score", score_reward_coeff=0.1)
        env._last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        env._prev_ocr_score = 0
        env._score_ocr = mock.MagicMock()
        env._score_ocr.read_score.return_value = 50
        reward = env._compute_score_reward(terminated=False, level_cleared=False)
        # Expected: 0.01 (survival) + 50 * 0.1 (delta) = 5.01
        assert abs(reward - 5.01) < 1e-6

    def test_score_reward_level_clear_bonus(self):
        """Score reward includes level clear bonus (from survival baseline)."""
        env = StubEnv(reward_mode="score")
        env._last_frame = None
        reward = env._compute_score_reward(terminated=False, level_cleared=True)
        # Should equal survival non-terminal level clear: +1.0 + 0.01 = 1.01
        assert abs(reward - 1.01) < 1e-6


class TestScoreRewardStepIntegration:
    """Tests for score reward mode in the full step() lifecycle."""

    def test_step_uses_score_reward_when_mode_set(self):
        """step() uses score reward (not game-specific) in score mode."""
        env = _make_ready_env(reward_mode="score")
        env.compute_reward = lambda *a, **kw: 999.0  # would be used in yolo
        # No OCR score change → reward = survival baseline (0.01)
        obs, reward, terminated, truncated, info = env.step(_action())
        assert abs(reward - 0.01) < 1e-6  # score mode, not 999.0

    def test_step_score_mode_mid_step_game_over_uses_terminal(self):
        """Mid-step game-over in score mode uses _SURVIVAL_TERMINAL_REWARD."""
        env = _make_ready_env(reward_mode="score")
        env.handle_modals = lambda **kw: "game_over"
        env._no_ball_count = 1  # triggers _should_check_modals -> True
        obs, reward, terminated, truncated, info = env.step(_action())
        assert terminated is True
        assert abs(reward - (-5.01)) < 1e-6

    def test_step_score_mode_ignores_yolo_level_cleared(self):
        """In score mode, level_cleared from check_termination is ignored.

        Same YOLO suppression as survival mode — brick detection is
        unreliable in headless mode.
        """
        env = _make_ready_env(reward_mode="score")
        env.check_termination = lambda detections: (True, True)
        obs, reward, terminated, truncated, info = env.step(_action())
        # Should NOT terminate — level_cleared is suppressed
        assert terminated is False
        assert abs(reward - 0.01) < 1e-6

    def test_step_score_mode_crash_transition_uses_terminal(self):
        """Crash transition in score mode uses _SURVIVAL_TERMINAL_REWARD."""
        browser = mock.MagicMock()
        browser.driver = mock.MagicMock()
        env = _make_ready_env(reward_mode="score", browser_instance=browser)
        # Simulate crash by making apply_action raise
        env.apply_action = mock.MagicMock(side_effect=RuntimeError("Chrome crashed"))
        obs, reward, terminated, truncated, info = env.step(_action())
        assert terminated is True
        assert abs(reward - (-5.01)) < 1e-6


class TestScoreRewardResetIntegration:
    """Tests for score OCR state reset during reset()."""

    @mock.patch("src.platform.base_env.time")
    def test_reset_clears_ocr_state(self, mock_time):
        """reset() clears the OCR score reader state."""
        env = _make_ready_env(reward_mode="score")
        env._prev_ocr_score = 500
        env._score_ocr = mock.MagicMock()

        env.reset()

        env._score_ocr.reset.assert_called_once()
        assert env._prev_ocr_score is None

    @mock.patch("src.platform.base_env.time")
    def test_reset_succeeds_in_score_mode_without_valid_detections(self, mock_time):
        """reset() in score mode is lenient like survival mode.

        YOLO detections are not needed for CNN observations or
        OCR-based reward computation.
        """
        env = _make_ready_env(reward_mode="score")
        env._detector.detect_to_game_state.return_value = _make_detections(ball=None)

        # Should NOT raise — score mode is lenient
        obs, info = env.reset()
        assert obs is not None


# ===========================================================================
# GameOverDetector integration with step() and reset()
# ===========================================================================


class TestGameOverDetectorStepIntegration:
    """GameOverDetector.update() is called in step() and terminates when triggered."""

    @mock.patch("src.platform.base_env.time")
    def test_step_calls_detector_update_with_captured_frame(self, mock_time):
        """step() calls game_over_detector.update(frame) after frame capture."""
        detector = mock.MagicMock()
        detector.update.return_value = False
        env = _make_ready_env(game_over_detector=detector)

        env.step(_action())

        detector.update.assert_called_once()
        # The argument should be a numpy array (the captured frame)
        call_args = detector.update.call_args
        assert isinstance(call_args[0][0], np.ndarray)

    @mock.patch("src.platform.base_env.time")
    def test_step_terminates_when_detector_signals_game_over(self, mock_time):
        """step() returns terminated=True when detector.update() returns True."""
        detector = mock.MagicMock()
        detector.update.return_value = True
        detector.get_confidence.return_value = {"screen_freeze": 0.9}
        env = _make_ready_env(game_over_detector=detector)

        obs, reward, terminated, truncated, info = env.step(_action())

        assert terminated is True

    @mock.patch("src.platform.base_env.time")
    def test_step_does_not_terminate_when_detector_returns_false(self, mock_time):
        """step() returns terminated=False when detector.update() returns False."""
        detector = mock.MagicMock()
        detector.update.return_value = False
        env = _make_ready_env(game_over_detector=detector)

        obs, reward, terminated, truncated, info = env.step(_action())

        assert terminated is False

    @mock.patch("src.platform.base_env.time")
    def test_step_applies_terminal_reward_on_detector_game_over(self, mock_time):
        """When detector signals game-over, reward uses terminal penalty."""
        detector = mock.MagicMock()
        detector.update.return_value = True
        detector.get_confidence.return_value = {"screen_freeze": 0.9}
        env = _make_ready_env(reward_mode="survival", game_over_detector=detector)

        obs, reward, terminated, truncated, info = env.step(_action())

        assert terminated is True
        assert reward == env._SURVIVAL_TERMINAL_REWARD

    @mock.patch("src.platform.base_env.time")
    def test_step_applies_terminal_reward_on_detector_game_over_yolo_mode(self, mock_time):
        """When detector signals game-over in yolo mode, reward uses terminal_reward()."""
        detector = mock.MagicMock()
        detector.update.return_value = True
        detector.get_confidence.return_value = {"screen_freeze": 0.9}
        env = _make_ready_env(game_over_detector=detector)
        assert env.reward_mode == "yolo"

        obs, reward, terminated, truncated, info = env.step(_action())

        assert terminated is True
        assert reward == env.terminal_reward()
        assert "game_over_detector" in info

    @mock.patch("src.platform.base_env.time")
    def test_step_without_detector_behaves_normally(self, mock_time):
        """When no detector is provided, step() works as before."""
        env = _make_ready_env()  # no game_over_detector
        assert env._game_over_detector is None

        obs, reward, terminated, truncated, info = env.step(_action())

        assert terminated is False
        assert isinstance(obs, np.ndarray)

    @mock.patch("src.platform.base_env.time")
    def test_step_detector_info_includes_confidence(self, mock_time):
        """When detector signals game-over, info includes detector confidence."""
        detector = mock.MagicMock()
        detector.update.return_value = True
        detector.get_confidence.return_value = {"screen_freeze": 0.95}
        env = _make_ready_env(game_over_detector=detector)

        obs, reward, terminated, truncated, info = env.step(_action())

        assert "game_over_detector" in info
        assert info["game_over_detector"]["screen_freeze"] == 0.95

    @mock.patch("src.platform.base_env.time")
    def test_step_detector_checked_after_modal_game_over_skipped(self, mock_time):
        """When modal says game_over, detector is not also called (modal takes priority)."""
        detector = mock.MagicMock()
        detector.update.return_value = False
        env = _make_ready_env(game_over_detector=detector)

        with mock.patch.object(env, "handle_modals", return_value="game_over"):
            with mock.patch.object(env, "_should_check_modals", return_value=True):
                obs, reward, terminated, truncated, info = env.step(_action())

        # Modal-based game-over takes priority — returns early
        assert terminated is True
        # Detector should NOT be called because modal returned early
        detector.update.assert_not_called()

    @mock.patch("src.platform.base_env.time")
    def test_step_runs_oracles_on_detector_termination(self, mock_time):
        """Oracles run when detector signals game-over."""
        detector = mock.MagicMock()
        detector.update.return_value = True
        detector.get_confidence.return_value = {}
        oracle = mock.MagicMock()
        oracle.on_step.return_value = []
        env = _make_ready_env(game_over_detector=detector)
        env._oracles = [oracle]

        env.step(_action())

        oracle.on_step.assert_called_once()


class TestGameOverDetectorResetIntegration:
    """GameOverDetector.reset() is called during reset()."""

    @mock.patch("src.platform.base_env.time")
    def test_reset_calls_detector_reset(self, mock_time):
        """reset() calls game_over_detector.reset() to clear per-episode state."""
        detector = mock.MagicMock()
        env = _make_ready_env(game_over_detector=detector)

        env.reset()

        detector.reset.assert_called_once()

    @mock.patch("src.platform.base_env.time")
    def test_reset_without_detector_works(self, mock_time):
        """reset() works fine when no detector is configured."""
        env = _make_ready_env()  # no game_over_detector
        assert env._game_over_detector is None

        obs, info = env.reset()
        assert obs is not None

    @mock.patch("src.platform.base_env.time")
    def test_reset_clears_detector_before_first_step(self, mock_time):
        """After reset, detector state is clean for the new episode."""
        detector = mock.MagicMock()
        detector.update.return_value = False
        env = _make_ready_env(game_over_detector=detector)

        # First episode
        for _ in range(5):
            env.step(_action())

        # Reset
        env.reset()

        # detector.reset() should have been called once
        detector.reset.assert_called_once()
        # update calls from the 5 steps should be 5
        assert detector.update.call_count == 5


# ===========================================================================
# Savegame Injection Integration
# ===========================================================================


class TestSavegameInjectorConstruction:
    """Savegame injector is created from savegame_dir + load_save_js."""

    def test_savegame_dir_without_js_raises(self, tmp_path):
        """savegame_dir without load_save_js raises ValueError."""
        with pytest.raises(ValueError, match="load_save_js"):
            StubEnv(savegame_dir=str(tmp_path))

    def test_savegame_dir_with_js_creates_injector(self, tmp_path):
        """savegame_dir + load_save_js creates _savegame_injector."""
        env = StubEnv(
            savegame_dir=str(tmp_path),
            load_save_js="return {};",
        )
        assert env._savegame_injector is not None

    def test_no_savegame_dir_leaves_injector_none(self):
        """Without savegame_dir, _savegame_injector is None."""
        env = StubEnv()
        assert env._savegame_injector is None

    def test_injector_has_correct_js(self, tmp_path):
        """The injector stores the provided JS snippet."""
        js = "var data = arguments[0]; return {};"
        env = StubEnv(savegame_dir=str(tmp_path), load_save_js=js)
        assert env._savegame_injector.load_save_js == js


class TestSavegameInjectorResetIntegration:
    """Savegame injection happens during reset()."""

    @mock.patch("src.platform.base_env.time")
    def test_reset_injects_savegame(self, mock_time, tmp_path):
        """reset() calls inject() when savegame injector is configured."""
        (tmp_path / "save1.json").write_text('{"level": 5}', encoding="utf-8")

        env = _make_ready_env(
            savegame_dir=str(tmp_path),
            load_save_js="return {};",
        )
        env._driver = mock.MagicMock()
        env._driver.execute_script.side_effect = lambda *a: {}

        obs, info = env.reset()

        assert "savegame" in info
        assert info["savegame"]["save_file"].endswith("save1.json")
        env._driver.execute_script.assert_called()

    @mock.patch("src.platform.base_env.time")
    def test_reset_without_injector_has_no_savegame_info(self, mock_time):
        """reset() without savegame injector does not add savegame to info."""
        env = _make_ready_env()

        obs, info = env.reset()

        assert "savegame" not in info

    @mock.patch("src.platform.base_env.time")
    def test_reset_survives_injection_failure(self, mock_time, tmp_path):
        """reset() continues when injection raises RuntimeError."""
        # Empty dir -> inject() will raise RuntimeError (pool empty)
        env = StubEnv(
            savegame_dir=str(tmp_path),
            load_save_js="return {};",
        )
        env._initialized = True
        env._capture = mock.MagicMock()
        env._detector = mock.MagicMock()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        env._capture.capture_frame.return_value = frame
        env._detector.detect_to_game_state.return_value = _make_detections()
        env._last_frame = frame
        env._driver = mock.MagicMock()

        obs, info = env.reset()

        # No savegame key because injection failed gracefully
        assert "savegame" not in info
        # But reset still succeeded
        assert obs is not None

    @mock.patch("src.platform.base_env.time")
    def test_reset_without_driver_skips_injection(self, mock_time, tmp_path):
        """reset() skips injection when driver is None."""
        (tmp_path / "save.json").write_text("{}", encoding="utf-8")

        env = _make_ready_env(
            savegame_dir=str(tmp_path),
            load_save_js="return {};",
        )
        env._driver = None

        obs, info = env.reset()

        assert "savegame" not in info
        assert obs is not None
