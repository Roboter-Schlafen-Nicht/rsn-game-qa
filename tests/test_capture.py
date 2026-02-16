"""Tests for the capture module (WindowCapture + InputController + WinCamCapture).

Tests cover:

- WindowCapture construction and pywin32 availability guard
- Window discovery (_find_window) with mocked win32gui
- Dimension updates from client rect
- Frame capture via BitBlt (mocked GDI pipeline)
- Window visibility checks
- Release lifecycle
- InputController construction and pydirectinput availability guard
- Discrete action mapping (NOOP, LEFT, RIGHT, FIRE)
- Invalid action rejection
- Normalised-to-screen coordinate conversion
- Mouse movement, click, key press, hold/release (mocked pydirectinput)
- WinCamCapture construction and wincam availability guard
- WinCamCapture frame capture via mocked DXCamera
- WinCamCapture camera lifecycle (create, resize, stop)
- WinCamCapture window visibility and release
"""

from __future__ import annotations

import sys
import types
from unittest import mock

import numpy as np
import pytest


# ── Helpers ─────────────────────────────────────────────────────────


def _make_mock_pywin32():
    """Create mock win32gui/win32ui/win32con modules.

    Returns a dict suitable for patching ``sys.modules``.
    """
    win32gui = mock.MagicMock(name="win32gui")
    win32ui = mock.MagicMock(name="win32ui")
    win32con = types.ModuleType("win32con")
    win32con.SRCCOPY = 0x00CC0020  # type: ignore[attr-defined]

    # Default window-discovery behaviour.
    win32gui.FindWindow.return_value = 12345
    win32gui.GetClientRect.return_value = (0, 0, 800, 600)
    win32gui.IsWindowVisible.return_value = True

    return {
        "win32gui": win32gui,
        "win32ui": win32ui,
        "win32con": win32con,
    }


def _make_mock_pydirectinput():
    """Create a mock pydirectinput module."""
    pdi = mock.MagicMock(name="pydirectinput")
    return {"pydirectinput": pdi}


def _import_window_capture(modules_patch: dict):
    """Import WindowCapture with mocked win32 modules.

    Forces a fresh module reload so the ``_PYWIN32_AVAILABLE`` flag
    picks up the patched (or absent) modules.
    """
    # Remove cached module so re-import picks up mocks.
    sys.modules.pop("src.capture.window_capture", None)
    with mock.patch.dict(sys.modules, modules_patch):
        from src.capture.window_capture import WindowCapture

        return WindowCapture


def _import_input_controller(modules_patch: dict):
    """Import InputController with mocked pydirectinput module."""
    sys.modules.pop("src.capture.input_controller", None)
    with mock.patch.dict(sys.modules, modules_patch):
        from src.capture.input_controller import InputController

        return InputController


# ── WindowCapture ───────────────────────────────────────────────────


class TestWindowCaptureInit:
    """Construction and availability guard tests."""

    def test_raises_without_pywin32(self):
        """WindowCapture raises RuntimeError when pywin32 is missing."""
        # Force reload with the win32 modules mapped to None so the
        # try/except ImportError in the module body fires.
        sys.modules.pop("src.capture.window_capture", None)
        with mock.patch.dict(
            sys.modules,
            {"win32gui": None, "win32ui": None, "win32con": None},
        ):
            import importlib
            import src.capture.window_capture as wc_mod

            importlib.reload(wc_mod)
            assert wc_mod._PYWIN32_AVAILABLE is False
            with pytest.raises(RuntimeError, match="pywin32 is required"):
                wc_mod.WindowCapture(window_title="Test")

    def test_init_with_hwnd(self):
        """WindowCapture accepts a direct HWND, skipping window search."""
        mocks = _make_mock_pywin32()
        WC = _import_window_capture(mocks)
        wc = WC(hwnd=99999)
        assert wc.hwnd == 99999
        assert wc.window_title == ""

    def test_init_with_title(self):
        """WindowCapture resolves HWND from window title."""
        mocks = _make_mock_pywin32()
        mocks["win32gui"].FindWindow.return_value = 42
        WC = _import_window_capture(mocks)
        wc = WC(window_title="My Game")
        assert wc.hwnd == 42
        assert wc.window_title == "My Game"

    def test_init_empty(self):
        """WindowCapture with no title or HWND leaves hwnd as 0."""
        mocks = _make_mock_pywin32()
        WC = _import_window_capture(mocks)
        wc = WC()
        assert wc.hwnd == 0


class TestWindowDiscovery:
    """Tests for _find_window."""

    def test_find_window_exact_match(self):
        """_find_window uses FindWindow for exact title match."""
        mocks = _make_mock_pywin32()
        mocks["win32gui"].FindWindow.return_value = 111
        WC = _import_window_capture(mocks)
        wc = WC(window_title="Exact Title")
        assert wc.hwnd == 111
        mocks["win32gui"].FindWindow.assert_called_with(None, "Exact Title")

    def test_find_window_substring_fallback(self):
        """_find_window falls back to EnumWindows substring match."""
        mocks = _make_mock_pywin32()
        mocks["win32gui"].FindWindow.return_value = 0  # No exact match.

        # Simulate EnumWindows calling the callback with a matching window.
        def fake_enum(callback, _extra):
            callback(222, None)

        mocks["win32gui"].EnumWindows.side_effect = fake_enum
        mocks["win32gui"].GetWindowText.return_value = "My Cool Game - v1.0"

        WC = _import_window_capture(mocks)
        wc = WC(window_title="Cool Game")
        assert wc.hwnd == 222

    def test_find_window_not_found_raises(self):
        """_find_window raises RuntimeError when no window matches."""
        mocks = _make_mock_pywin32()
        mocks["win32gui"].FindWindow.return_value = 0
        mocks["win32gui"].EnumWindows.side_effect = lambda cb, _: None  # No hits.

        WC = _import_window_capture(mocks)
        with pytest.raises(RuntimeError, match="not found"):
            WC(window_title="Nonexistent Window")


class TestUpdateDimensions:
    """Tests for _update_dimensions."""

    def test_dimensions_from_client_rect(self):
        """_update_dimensions reads width/height from GetClientRect."""
        mocks = _make_mock_pywin32()
        mocks["win32gui"].GetClientRect.return_value = (0, 0, 1024, 768)
        WC = _import_window_capture(mocks)
        wc = WC(hwnd=1)
        assert wc.width == 1024
        assert wc.height == 768

    def test_dimensions_update_error(self):
        """_update_dimensions raises RuntimeError on invalid HWND."""
        mocks = _make_mock_pywin32()
        mocks["win32gui"].FindWindow.return_value = 50
        mocks["win32gui"].GetClientRect.side_effect = Exception("bad handle")
        WC = _import_window_capture(mocks)
        with pytest.raises(RuntimeError, match="Failed to get client rect"):
            WC(window_title="Bad Window")


class TestCaptureFrame:
    """Tests for capture_frame."""

    def _setup_capture_mocks(self, mocks, width=4, height=3):
        """Wire up the full BitBlt mock pipeline."""
        mocks["win32gui"].GetClientRect.return_value = (0, 0, width, height)
        mocks["win32gui"].GetWindowDC.return_value = 100

        mfc_dc = mock.MagicMock(name="mfc_dc")
        save_dc = mock.MagicMock(name="save_dc")
        mfc_dc.CreateCompatibleDC.return_value = save_dc
        mocks["win32ui"].CreateDCFromHandle.return_value = mfc_dc

        bitmap = mock.MagicMock(name="bitmap")
        bitmap.GetHandle.return_value = 999
        # Create fake BGRA pixel data.
        bgra = np.zeros((height, width, 4), dtype=np.uint8)
        bgra[:, :, 0] = 10  # B
        bgra[:, :, 1] = 20  # G
        bgra[:, :, 2] = 30  # R
        bgra[:, :, 3] = 255  # A
        bitmap.GetBitmapBits.return_value = bgra.tobytes()
        mocks["win32ui"].CreateBitmap.return_value = bitmap

        return mfc_dc, save_dc, bitmap

    def test_capture_returns_bgr_array(self):
        """capture_frame returns a (H, W, 3) BGR uint8 array."""
        mocks = _make_mock_pywin32()
        self._setup_capture_mocks(mocks, width=4, height=3)
        WC = _import_window_capture(mocks)
        wc = WC(hwnd=1)

        frame = wc.capture_frame()

        assert isinstance(frame, np.ndarray)
        assert frame.shape == (3, 4, 3)
        assert frame.dtype == np.uint8
        # BGR values should be [10, 20, 30] (alpha dropped).
        np.testing.assert_array_equal(frame[0, 0], [10, 20, 30])

    def test_capture_no_hwnd_raises(self):
        """capture_frame raises RuntimeError if no HWND is set."""
        mocks = _make_mock_pywin32()
        WC = _import_window_capture(mocks)
        wc = WC()  # hwnd == 0
        with pytest.raises(RuntimeError, match="No window handle"):
            wc.capture_frame()

    def test_capture_zero_size_raises(self):
        """capture_frame raises RuntimeError for zero-size client area."""
        mocks = _make_mock_pywin32()
        mocks["win32gui"].GetClientRect.return_value = (0, 0, 0, 0)
        WC = _import_window_capture(mocks)
        wc = WC(hwnd=1)
        # First _update_dimensions in __init__ sets 0x0, then capture re-checks.
        with pytest.raises(RuntimeError, match="zero size"):
            wc.capture_frame()

    def test_capture_cleans_up_gdi_on_success(self):
        """GDI resources are released after a successful capture."""
        mocks = _make_mock_pywin32()
        mfc_dc, save_dc, bitmap = self._setup_capture_mocks(mocks)
        WC = _import_window_capture(mocks)
        wc = WC(hwnd=1)

        wc.capture_frame()

        mocks["win32gui"].DeleteObject.assert_called_once_with(999)
        save_dc.DeleteDC.assert_called_once()
        mfc_dc.DeleteDC.assert_called_once()
        mocks["win32gui"].ReleaseDC.assert_called_once_with(1, 100)

    def test_capture_cleans_up_gdi_on_failure(self):
        """GDI resources are released even when capture fails mid-way."""
        mocks = _make_mock_pywin32()
        mocks["win32gui"].GetClientRect.return_value = (0, 0, 4, 3)
        mocks["win32gui"].GetWindowDC.return_value = 100

        mfc_dc = mock.MagicMock(name="mfc_dc")
        save_dc = mock.MagicMock(name="save_dc")
        mfc_dc.CreateCompatibleDC.return_value = save_dc
        mocks["win32ui"].CreateDCFromHandle.return_value = mfc_dc

        bitmap = mock.MagicMock(name="bitmap")
        bitmap.GetHandle.return_value = 888
        bitmap.GetBitmapBits.side_effect = Exception("GDI fail")
        mocks["win32ui"].CreateBitmap.return_value = bitmap

        WC = _import_window_capture(mocks)
        wc = WC(hwnd=1)

        with pytest.raises(RuntimeError, match="Frame capture failed"):
            wc.capture_frame()

        # Cleanup should still happen.
        mocks["win32gui"].DeleteObject.assert_called_once_with(888)
        save_dc.DeleteDC.assert_called_once()


class TestWindowVisible:
    """Tests for is_window_visible."""

    def test_visible_returns_true(self):
        """is_window_visible returns True for a visible window."""
        mocks = _make_mock_pywin32()
        mocks["win32gui"].IsWindowVisible.return_value = True
        WC = _import_window_capture(mocks)
        wc = WC(hwnd=1)
        assert wc.is_window_visible() is True

    def test_not_visible_returns_false(self):
        """is_window_visible returns False for a hidden window."""
        mocks = _make_mock_pywin32()
        mocks["win32gui"].IsWindowVisible.return_value = False
        WC = _import_window_capture(mocks)
        wc = WC(hwnd=1)
        assert wc.is_window_visible() is False

    def test_no_hwnd_returns_false(self):
        """is_window_visible returns False when no HWND is set."""
        mocks = _make_mock_pywin32()
        WC = _import_window_capture(mocks)
        wc = WC()
        assert wc.is_window_visible() is False

    def test_exception_returns_false(self):
        """is_window_visible returns False on win32gui exception."""
        mocks = _make_mock_pywin32()
        mocks["win32gui"].IsWindowVisible.side_effect = Exception("dead")
        WC = _import_window_capture(mocks)
        wc = WC(hwnd=1)
        assert wc.is_window_visible() is False


class TestRelease:
    """Tests for release lifecycle."""

    def test_release_resets_state(self):
        """release() zeroes out HWND and dimensions."""
        mocks = _make_mock_pywin32()
        WC = _import_window_capture(mocks)
        wc = WC(hwnd=1)
        assert wc.hwnd != 0

        wc.release()
        assert wc.hwnd == 0
        assert wc.width == 0
        assert wc.height == 0

    def test_release_idempotent(self):
        """Calling release() twice does not raise."""
        mocks = _make_mock_pywin32()
        WC = _import_window_capture(mocks)
        wc = WC(hwnd=1)
        wc.release()
        wc.release()  # Should not raise.


# ── InputController ─────────────────────────────────────────────────


class TestInputControllerInit:
    """Construction and availability guard tests."""

    def test_raises_without_pydirectinput(self):
        """InputController raises RuntimeError when pydirectinput is missing."""
        sys.modules.pop("src.capture.input_controller", None)
        with mock.patch.dict(sys.modules, {"pydirectinput": None}):
            import importlib
            import src.capture.input_controller as ic_mod

            importlib.reload(ic_mod)
            assert ic_mod._PYDIRECTINPUT_AVAILABLE is False
            with pytest.raises(RuntimeError, match="pydirectinput is required"):
                ic_mod.InputController()

    def test_default_window_rect(self):
        """Default window_rect is (0, 0, 800, 600)."""
        mocks = _make_mock_pydirectinput()
        IC = _import_input_controller(mocks)
        ic = IC()
        assert ic.window_rect == (0, 0, 800, 600)

    def test_custom_window_rect(self):
        """Custom window_rect is stored."""
        mocks = _make_mock_pydirectinput()
        IC = _import_input_controller(mocks)
        ic = IC(window_rect=(100, 200, 900, 800))
        assert ic.window_rect == (100, 200, 900, 800)

    def test_action_constants_defined(self):
        """Action constants should be defined on the class."""
        mocks = _make_mock_pydirectinput()
        IC = _import_input_controller(mocks)
        assert IC.ACTION_NOOP == 0
        assert IC.ACTION_LEFT == 1
        assert IC.ACTION_RIGHT == 2
        assert IC.ACTION_FIRE == 3


class TestApplyAction:
    """Tests for apply_action."""

    def test_noop_does_nothing(self):
        """ACTION_NOOP should not call any pydirectinput functions."""
        mocks = _make_mock_pydirectinput()
        IC = _import_input_controller(mocks)
        ic = IC()
        ic.apply_action(IC.ACTION_NOOP)
        mocks["pydirectinput"].keyDown.assert_not_called()
        mocks["pydirectinput"].keyUp.assert_not_called()

    def test_left_sends_left_key(self):
        """ACTION_LEFT sends left arrow key down then up."""
        mocks = _make_mock_pydirectinput()
        IC = _import_input_controller(mocks)
        ic = IC()
        ic.apply_action(IC.ACTION_LEFT)
        mocks["pydirectinput"].keyDown.assert_called_with("left")
        mocks["pydirectinput"].keyUp.assert_called_with("left")

    def test_right_sends_right_key(self):
        """ACTION_RIGHT sends right arrow key."""
        mocks = _make_mock_pydirectinput()
        IC = _import_input_controller(mocks)
        ic = IC()
        ic.apply_action(IC.ACTION_RIGHT)
        mocks["pydirectinput"].keyDown.assert_called_with("right")
        mocks["pydirectinput"].keyUp.assert_called_with("right")

    def test_fire_sends_space_key(self):
        """ACTION_FIRE sends space key."""
        mocks = _make_mock_pydirectinput()
        IC = _import_input_controller(mocks)
        ic = IC()
        ic.apply_action(IC.ACTION_FIRE)
        mocks["pydirectinput"].keyDown.assert_called_with("space")
        mocks["pydirectinput"].keyUp.assert_called_with("space")

    def test_invalid_action_raises(self):
        """apply_action raises ValueError for unrecognised actions."""
        mocks = _make_mock_pydirectinput()
        IC = _import_input_controller(mocks)
        ic = IC()
        with pytest.raises(ValueError, match="Unrecognised action"):
            ic.apply_action(99)


class TestCoordinateConversion:
    """Tests for normalised-to-screen coordinate conversion."""

    def test_origin(self):
        """(0.0, 0.0) maps to top-left of window rect."""
        mocks = _make_mock_pydirectinput()
        IC = _import_input_controller(mocks)
        ic = IC(window_rect=(100, 200, 500, 600))
        x, y = ic._to_screen_coords(0.0, 0.0)
        assert x == 100
        assert y == 200

    def test_center(self):
        """(0.5, 0.5) maps to center of window rect."""
        mocks = _make_mock_pydirectinput()
        IC = _import_input_controller(mocks)
        ic = IC(window_rect=(0, 0, 800, 600))
        x, y = ic._to_screen_coords(0.5, 0.5)
        assert x == 400
        assert y == 300

    def test_bottom_right(self):
        """(1.0, 1.0) maps to bottom-right of window rect."""
        mocks = _make_mock_pydirectinput()
        IC = _import_input_controller(mocks)
        ic = IC(window_rect=(100, 100, 900, 700))
        x, y = ic._to_screen_coords(1.0, 1.0)
        assert x == 900
        assert y == 700


class TestMouseAndKeyboard:
    """Tests for move_mouse_to, click, press_key, hold_key, release_key."""

    def test_move_mouse_to(self):
        """move_mouse_to calls pydirectinput.moveTo with screen coords."""
        mocks = _make_mock_pydirectinput()
        IC = _import_input_controller(mocks)
        ic = IC(window_rect=(0, 0, 800, 600))
        ic.move_mouse_to(0.5, 0.5)
        mocks["pydirectinput"].moveTo.assert_called_once_with(400, 300)

    def test_click(self):
        """click calls pydirectinput.click with screen coords and button."""
        mocks = _make_mock_pydirectinput()
        IC = _import_input_controller(mocks)
        ic = IC(window_rect=(0, 0, 800, 600))
        ic.click(0.25, 0.75, button="right")
        mocks["pydirectinput"].click.assert_called_once_with(
            x=200, y=450, button="right"
        )

    def test_press_key(self):
        """press_key sends keyDown then keyUp."""
        mocks = _make_mock_pydirectinput()
        IC = _import_input_controller(mocks)
        ic = IC()
        ic.press_key("enter", duration=0.0)
        mocks["pydirectinput"].keyDown.assert_called_with("enter")
        mocks["pydirectinput"].keyUp.assert_called_with("enter")

    def test_hold_key(self):
        """hold_key sends keyDown only."""
        mocks = _make_mock_pydirectinput()
        IC = _import_input_controller(mocks)
        ic = IC()
        ic.hold_key("left")
        mocks["pydirectinput"].keyDown.assert_called_once_with("left")
        mocks["pydirectinput"].keyUp.assert_not_called()

    def test_release_key(self):
        """release_key sends keyUp only."""
        mocks = _make_mock_pydirectinput()
        IC = _import_input_controller(mocks)
        ic = IC()
        ic.release_key("left")
        mocks["pydirectinput"].keyUp.assert_called_once_with("left")
        mocks["pydirectinput"].keyDown.assert_not_called()


# ── WinCamCapture ───────────────────────────────────────────────────


def _make_mock_wincam():
    """Create mock wincam and win32gui modules for WinCamCapture tests.

    Returns a dict suitable for patching ``sys.modules``, plus the
    mock objects for assertions.
    """
    win32gui = mock.MagicMock(name="win32gui")
    win32gui.FindWindow.return_value = 12345
    win32gui.GetClientRect.return_value = (0, 0, 1264, 1016)
    win32gui.ClientToScreen.return_value = (8, 130)
    win32gui.IsWindowVisible.return_value = True

    # Build mock wincam package.
    mock_dxcamera_cls = mock.MagicMock(name="DXCamera")
    mock_desktop_cls = mock.MagicMock(name="DesktopWindow")

    wincam_mod = types.ModuleType("wincam")
    wincam_mod.DXCamera = mock_dxcamera_cls  # type: ignore[attr-defined]

    wincam_desktop = types.ModuleType("wincam.desktop")
    wincam_desktop.DesktopWindow = mock_desktop_cls  # type: ignore[attr-defined]

    modules = {
        "win32gui": win32gui,
        "win32ui": mock.MagicMock(name="win32ui"),
        "win32con": mock.MagicMock(name="win32con"),
        "wincam": wincam_mod,
        "wincam.desktop": wincam_desktop,
    }
    return modules, win32gui, mock_dxcamera_cls, mock_desktop_cls


def _import_wincam_capture(modules_patch: dict):
    """Import WinCamCapture with mocked wincam/win32gui modules."""
    sys.modules.pop("src.capture.wincam_capture", None)
    with mock.patch.dict(sys.modules, modules_patch):
        from src.capture.wincam_capture import WinCamCapture

        return WinCamCapture


class TestWinCamCaptureInit:
    """Construction and availability guard tests for WinCamCapture."""

    def test_raises_without_wincam(self):
        """WinCamCapture raises RuntimeError when wincam is missing."""
        sys.modules.pop("src.capture.wincam_capture", None)
        with mock.patch.dict(
            sys.modules,
            {"wincam": None, "wincam.desktop": None},
        ):
            import importlib
            import src.capture.wincam_capture as wc_mod

            importlib.reload(wc_mod)
            assert wc_mod._WINCAM_AVAILABLE is False
            with pytest.raises(RuntimeError, match="wincam is required"):
                wc_mod.WinCamCapture(window_title="Test")

    def test_raises_without_pywin32(self):
        """WinCamCapture raises RuntimeError when pywin32 is missing."""
        sys.modules.pop("src.capture.wincam_capture", None)
        # wincam available but win32gui not.
        mock_wincam = types.ModuleType("wincam")
        mock_wincam.DXCamera = mock.MagicMock()  # type: ignore[attr-defined]
        mock_wincam_desktop = types.ModuleType("wincam.desktop")
        mock_wincam_desktop.DesktopWindow = mock.MagicMock()  # type: ignore[attr-defined]
        with mock.patch.dict(
            sys.modules,
            {
                "wincam": mock_wincam,
                "wincam.desktop": mock_wincam_desktop,
                "win32gui": None,
            },
        ):
            import importlib
            import src.capture.wincam_capture as wc_mod

            importlib.reload(wc_mod)
            assert wc_mod._WINCAM_AVAILABLE is True
            assert wc_mod._PYWIN32_AVAILABLE is False
            with pytest.raises(RuntimeError, match="pywin32 is required"):
                wc_mod.WinCamCapture(window_title="Test")

    def test_init_with_hwnd(self):
        """WinCamCapture accepts a direct HWND, skipping window search."""
        modules, win32gui, _, _ = _make_mock_wincam()
        WCC = _import_wincam_capture(modules)
        wc = WCC(hwnd=99999)
        assert wc.hwnd == 99999
        assert wc.window_title == ""

    def test_init_with_title(self):
        """WinCamCapture resolves HWND from window title."""
        modules, win32gui, _, _ = _make_mock_wincam()
        win32gui.FindWindow.return_value = 42
        WCC = _import_wincam_capture(modules)
        wc = WCC(window_title="My Game")
        assert wc.hwnd == 42

    def test_init_empty(self):
        """WinCamCapture with no title or HWND leaves hwnd as 0."""
        modules, _, _, _ = _make_mock_wincam()
        WCC = _import_wincam_capture(modules)
        wc = WCC()
        assert wc.hwnd == 0

    def test_init_dimensions(self):
        """WinCamCapture reads width/height from GetClientRect."""
        modules, win32gui, _, _ = _make_mock_wincam()
        win32gui.GetClientRect.return_value = (0, 0, 1024, 768)
        WCC = _import_wincam_capture(modules)
        wc = WCC(hwnd=1)
        assert wc.width == 1024
        assert wc.height == 768


class TestWinCamCaptureWindowDiscovery:
    """Tests for WinCamCapture._find_window."""

    def test_find_window_exact_match(self):
        """_find_window uses FindWindow for exact title match."""
        modules, win32gui, _, _ = _make_mock_wincam()
        win32gui.FindWindow.return_value = 111
        WCC = _import_wincam_capture(modules)
        wc = WCC(window_title="Exact Title")
        assert wc.hwnd == 111
        win32gui.FindWindow.assert_called_with(None, "Exact Title")

    def test_find_window_substring_fallback(self):
        """_find_window falls back to EnumWindows substring match."""
        modules, win32gui, _, _ = _make_mock_wincam()
        win32gui.FindWindow.return_value = 0

        def fake_enum(callback, _extra):
            callback(222, None)

        win32gui.EnumWindows.side_effect = fake_enum
        win32gui.GetWindowText.return_value = "My Cool Game - v1.0"

        WCC = _import_wincam_capture(modules)
        wc = WCC(window_title="Cool Game")
        assert wc.hwnd == 222

    def test_find_window_not_found_raises(self):
        """_find_window raises RuntimeError when no window matches."""
        modules, win32gui, _, _ = _make_mock_wincam()
        win32gui.FindWindow.return_value = 0
        win32gui.EnumWindows.side_effect = lambda cb, _: None

        WCC = _import_wincam_capture(modules)
        with pytest.raises(RuntimeError, match="not found"):
            WCC(window_title="Nonexistent Window")


class TestWinCamCaptureFrame:
    """Tests for WinCamCapture.capture_frame."""

    def test_capture_returns_bgr_array(self):
        """capture_frame returns BGR (H, W, 3) uint8 array from DXCamera."""
        modules, win32gui, mock_dxcamera_cls, _ = _make_mock_wincam()
        win32gui.GetClientRect.return_value = (0, 0, 640, 480)
        win32gui.ClientToScreen.return_value = (100, 200)

        # Mock the DXCamera instance returned by DXCamera(x, y, w, h, fps=...).
        mock_camera = mock.MagicMock(name="camera_instance")
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fake_frame[:, :, 0] = 10  # B
        fake_frame[:, :, 1] = 20  # G
        fake_frame[:, :, 2] = 30  # R
        mock_camera.get_bgr_frame.return_value = (fake_frame, 0.0)
        mock_dxcamera_cls.return_value = mock_camera

        WCC = _import_wincam_capture(modules)
        wc = WCC(hwnd=1)
        frame = wc.capture_frame()

        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8
        np.testing.assert_array_equal(frame[0, 0], [10, 20, 30])

        # DXCamera was constructed with screen coords of client area.
        mock_dxcamera_cls.assert_called_once_with(100, 200, 640, 480, fps=60)
        mock_camera.__enter__.assert_called_once()

    def test_capture_no_hwnd_raises(self):
        """capture_frame raises RuntimeError if no HWND is set."""
        modules, _, _, _ = _make_mock_wincam()
        WCC = _import_wincam_capture(modules)
        wc = WCC()
        with pytest.raises(RuntimeError, match="No window handle"):
            wc.capture_frame()

    def test_capture_zero_size_raises(self):
        """capture_frame raises RuntimeError for zero-size client area."""
        modules, win32gui, _, _ = _make_mock_wincam()
        win32gui.GetClientRect.return_value = (0, 0, 0, 0)
        WCC = _import_wincam_capture(modules)
        wc = WCC(hwnd=1)
        with pytest.raises(RuntimeError, match="zero size"):
            wc.capture_frame()

    def test_camera_reused_on_same_dimensions(self):
        """DXCamera is reused when window hasn't moved or resized."""
        modules, win32gui, mock_dxcamera_cls, _ = _make_mock_wincam()
        win32gui.GetClientRect.return_value = (0, 0, 640, 480)
        win32gui.ClientToScreen.return_value = (100, 200)

        mock_camera = mock.MagicMock(name="camera_instance")
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_camera.get_bgr_frame.return_value = (fake_frame, 0.0)
        mock_dxcamera_cls.return_value = mock_camera

        WCC = _import_wincam_capture(modules)
        wc = WCC(hwnd=1)

        wc.capture_frame()
        wc.capture_frame()

        # DXCamera should only be created once.
        assert mock_dxcamera_cls.call_count == 1

    def test_camera_recreated_on_resize(self):
        """DXCamera is recreated when window dimensions change."""
        modules, win32gui, mock_dxcamera_cls, _ = _make_mock_wincam()

        # First capture: 640x480 at (100, 200).
        win32gui.GetClientRect.return_value = (0, 0, 640, 480)
        win32gui.ClientToScreen.return_value = (100, 200)

        camera1 = mock.MagicMock(name="camera1")
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        camera1.get_bgr_frame.return_value = (fake_frame, 0.0)

        camera2 = mock.MagicMock(name="camera2")
        fake_frame2 = np.zeros((600, 800, 3), dtype=np.uint8)
        camera2.get_bgr_frame.return_value = (fake_frame2, 0.0)

        mock_dxcamera_cls.side_effect = [camera1, camera2]

        WCC = _import_wincam_capture(modules)
        wc = WCC(hwnd=1)
        wc.capture_frame()

        # Simulate resize.
        win32gui.GetClientRect.return_value = (0, 0, 800, 600)
        wc.capture_frame()

        assert mock_dxcamera_cls.call_count == 2
        camera1.__exit__.assert_called_once()


class TestWinCamCaptureVisible:
    """Tests for WinCamCapture.is_window_visible."""

    def test_visible_returns_true(self):
        """is_window_visible returns True for a visible window."""
        modules, win32gui, _, _ = _make_mock_wincam()
        win32gui.IsWindowVisible.return_value = True
        WCC = _import_wincam_capture(modules)
        wc = WCC(hwnd=1)
        assert wc.is_window_visible() is True

    def test_not_visible_returns_false(self):
        """is_window_visible returns False for a hidden window."""
        modules, win32gui, _, _ = _make_mock_wincam()
        win32gui.IsWindowVisible.return_value = False
        WCC = _import_wincam_capture(modules)
        wc = WCC(hwnd=1)
        assert wc.is_window_visible() is False

    def test_no_hwnd_returns_false(self):
        """is_window_visible returns False when no HWND is set."""
        modules, _, _, _ = _make_mock_wincam()
        WCC = _import_wincam_capture(modules)
        wc = WCC()
        assert wc.is_window_visible() is False

    def test_exception_returns_false(self):
        """is_window_visible returns False on win32gui exception."""
        modules, win32gui, _, _ = _make_mock_wincam()
        win32gui.IsWindowVisible.side_effect = Exception("dead")
        WCC = _import_wincam_capture(modules)
        wc = WCC(hwnd=1)
        assert wc.is_window_visible() is False


class TestWinCamCaptureRelease:
    """Tests for WinCamCapture.release lifecycle."""

    def test_release_resets_state(self):
        """release() zeroes out HWND and dimensions."""
        modules, _, _, _ = _make_mock_wincam()
        WCC = _import_wincam_capture(modules)
        wc = WCC(hwnd=1)
        assert wc.hwnd != 0

        wc.release()
        assert wc.hwnd == 0
        assert wc.width == 0
        assert wc.height == 0

    def test_release_stops_camera(self):
        """release() calls __exit__ on the active DXCamera."""
        modules, win32gui, mock_dxcamera_cls, _ = _make_mock_wincam()
        win32gui.GetClientRect.return_value = (0, 0, 640, 480)
        win32gui.ClientToScreen.return_value = (100, 200)

        mock_camera = mock.MagicMock(name="camera_instance")
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_camera.get_bgr_frame.return_value = (fake_frame, 0.0)
        mock_dxcamera_cls.return_value = mock_camera

        WCC = _import_wincam_capture(modules)
        wc = WCC(hwnd=1)
        wc.capture_frame()  # Creates the camera.
        wc.release()

        mock_camera.__exit__.assert_called_once()

    def test_release_idempotent(self):
        """Calling release() twice does not raise."""
        modules, _, _, _ = _make_mock_wincam()
        WCC = _import_wincam_capture(modules)
        wc = WCC(hwnd=1)
        wc.release()
        wc.release()  # Should not raise.


class TestWinCamCaptureCustomFps:
    """Tests for WinCamCapture custom FPS parameter."""

    def test_custom_fps_passed_to_dxcamera(self):
        """Custom fps is forwarded to DXCamera constructor."""
        modules, win32gui, mock_dxcamera_cls, _ = _make_mock_wincam()
        win32gui.GetClientRect.return_value = (0, 0, 640, 480)
        win32gui.ClientToScreen.return_value = (100, 200)

        mock_camera = mock.MagicMock(name="camera_instance")
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_camera.get_bgr_frame.return_value = (fake_frame, 0.0)
        mock_dxcamera_cls.return_value = mock_camera

        WCC = _import_wincam_capture(modules)
        wc = WCC(hwnd=1, fps=30)
        wc.capture_frame()

        mock_dxcamera_cls.assert_called_once_with(100, 200, 640, 480, fps=30)
