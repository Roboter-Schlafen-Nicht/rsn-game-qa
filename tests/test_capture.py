"""Tests for the capture module (WindowCapture + InputController)."""


class TestWindowCapture:
    """Placeholder tests for WindowCapture."""

    def test_window_capture_init_without_pywin32(self):
        """WindowCapture should raise RuntimeError if pywin32 is missing."""
        # TODO: mock pywin32 unavailability and verify RuntimeError
        pass

    def test_capture_frame_returns_ndarray(self):
        """capture_frame should return a (H, W, 3) uint8 numpy array."""
        # TODO: requires a live window or mock
        pass

    def test_is_window_visible(self):
        """is_window_visible should return bool."""
        # TODO: requires a live window or mock
        pass


class TestInputController:
    """Placeholder tests for InputController."""

    def test_input_controller_init_without_pydirectinput(self):
        """InputController should raise RuntimeError if pydirectinput is missing."""
        # TODO: mock pydirectinput unavailability and verify RuntimeError
        pass

    def test_apply_action_invalid_raises(self):
        """apply_action should raise ValueError for unrecognized actions."""
        # TODO: implement once apply_action is implemented
        pass

    def test_action_constants_defined(self):
        """Action constants should be defined on the class."""
        from src.capture.input_controller import InputController

        assert hasattr(InputController, "ACTION_NOOP")
        assert hasattr(InputController, "ACTION_LEFT")
        assert hasattr(InputController, "ACTION_RIGHT")
        assert hasattr(InputController, "ACTION_FIRE")
