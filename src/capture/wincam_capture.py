"""Fast screen capture using wincam (Direct3D11CaptureFramePool).

Provides the same interface as :class:`WindowCapture` but uses the
`wincam <https://github.com/lovettchris/wincam>`_ library for
sub-millisecond frame capture via DirectX 11 GPU async buffering.

Raises ``RuntimeError`` if wincam or pywin32 is unavailable (CI,
headless, non-Windows).  Callers should catch the error and fall
back to :class:`WindowCapture` (PrintWindow/GDI) themselves.

Key differences from WindowCapture
-----------------------------------
- ``wincam.DXCamera`` captures a *screen region* (not a window DC), so
  the target window must be visible and unoccluded.
- Frames are asynchronously buffered by C++ code on the GPU; Python reads
  from that buffer in <1 ms.
- The monitor refresh rate caps effective FPS (typically 60 Hz).
- Only one ``DXCamera`` instance is allowed at a time (library constraint).
"""

from __future__ import annotations


import numpy as np

try:
    import win32gui

    _PYWIN32_AVAILABLE = True
except ImportError:
    _PYWIN32_AVAILABLE = False

try:
    from wincam import DXCamera

    _WINCAM_AVAILABLE = True
except (ImportError, OSError):
    _WINCAM_AVAILABLE = False


class WinCamCapture:
    """Fast window capture using wincam's Direct3D11CaptureFramePool.

    Matches the :class:`WindowCapture` interface so it can be used as a
    drop-in replacement.  Finds the target window by title or HWND, then
    captures its client area via ``wincam.DXCamera``.

    Parameters
    ----------
    window_title : str
        The title (or partial title) of the target window.
    hwnd : int, optional
        Direct HWND handle.  If provided, ``window_title`` is ignored.
    fps : int, optional
        Target frames per second for wincam's internal throttle.
        Defaults to 60 (monitor refresh rate ceiling).

    Attributes
    ----------
    hwnd : int
        The resolved window handle.
    width : int
        Captured frame width in pixels.
    height : int
        Captured frame height in pixels.

    Raises
    ------
    RuntimeError
        If neither wincam nor pywin32 is available, or the target window
        cannot be found.
    """

    def __init__(
        self,
        window_title: str = "",
        hwnd: int | None = None,
        fps: int = 60,
    ) -> None:
        if not _WINCAM_AVAILABLE:
            raise RuntimeError(
                "wincam is required for WinCamCapture. "
                "Install it with: pip install wincam"
            )
        if not _PYWIN32_AVAILABLE:
            raise RuntimeError(
                "pywin32 is required for WinCamCapture (window lookup). "
                "Install it with: pip install pywin32"
            )

        self.window_title = window_title
        self.hwnd: int = hwnd if hwnd is not None else 0
        self.width: int = 0
        self.height: int = 0
        self._fps = fps

        # Lazily created on first capture_frame() call.
        self._camera: DXCamera | None = None

        # Track the region the camera was created with so we can detect
        # when the window moves/resizes without accessing DXCamera privates.
        self._camera_x: int = 0
        self._camera_y: int = 0
        self._camera_w: int = 0
        self._camera_h: int = 0

        if self.hwnd == 0 and self.window_title:
            self._find_window()

        if self.hwnd != 0:
            self._update_dimensions()

    # ------------------------------------------------------------------
    # Window lookup (same logic as WindowCapture)
    # ------------------------------------------------------------------

    def _find_window(self) -> None:
        """Locate the target window by title and store its HWND.

        Uses ``win32gui.FindWindow`` for exact title match, then falls
        back to substring search via ``EnumWindows``.

        Raises
        ------
        RuntimeError
            If no window matching ``self.window_title`` is found.
        """
        hwnd = win32gui.FindWindow(None, self.window_title)
        if hwnd != 0:
            self.hwnd = hwnd
            return

        results: list[int] = []

        def _enum_cb(h: int, _extra: object) -> None:
            title = win32gui.GetWindowText(h)
            if self.window_title.lower() in title.lower():
                results.append(h)

        win32gui.EnumWindows(_enum_cb, None)

        if not results:
            raise RuntimeError(
                f"Window '{self.window_title}' not found. "
                "Ensure the target application is running."
            )
        self.hwnd = results[0]

    def _update_dimensions(self) -> None:
        """Read client-area dimensions and screen position from HWND."""
        try:
            left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to get client rect for HWND {self.hwnd}: {exc}"
            ) from exc
        self.width = right - left
        self.height = bottom - top

    def _get_client_screen_rect(self) -> tuple[int, int, int, int]:
        """Return (x, y, width, height) of the client area in screen coords.

        wincam needs screen coordinates to know which region to capture.
        """
        left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
        w = right - left
        h = bottom - top
        # Convert client (0,0) to screen coordinates.
        screen_x, screen_y = win32gui.ClientToScreen(self.hwnd, (left, top))
        return screen_x, screen_y, w, h

    # ------------------------------------------------------------------
    # Camera lifecycle
    # ------------------------------------------------------------------

    def _ensure_camera(self) -> None:
        """Create or recreate the DXCamera if needed.

        The camera is torn down and recreated whenever the window
        dimensions change (e.g. window resize) since DXCamera is
        initialised with a fixed region.
        """
        self._update_dimensions()

        if self.width == 0 or self.height == 0:
            raise RuntimeError(
                f"Window client area has zero size ({self.width}x{self.height}). "
                "The window may be minimised."
            )

        x, y, w, h = self._get_client_screen_rect()

        # If camera already exists with matching dimensions, keep it.
        if self._camera is not None:
            if (
                self._camera_x == x
                and self._camera_y == y
                and self._camera_w == w
                and self._camera_h == h
            ):
                return
            # Dimensions changed — tear down and recreate.
            self._stop_camera()

        self._camera = DXCamera(x, y, w, h, fps=self._fps)
        self._camera.__enter__()
        self._camera_x = x
        self._camera_y = y
        self._camera_w = w
        self._camera_h = h

    def _stop_camera(self) -> None:
        """Stop and release the DXCamera."""
        if self._camera is not None:
            try:
                self._camera.__exit__(None, None, None)
            except Exception:
                pass
            self._camera = None

    # ------------------------------------------------------------------
    # Public interface (matches WindowCapture)
    # ------------------------------------------------------------------

    def capture_frame(self) -> np.ndarray:
        """Capture a single frame from the target window.

        Uses wincam's DXCamera (Direct3D11CaptureFramePool) for <1 ms
        GPU-async capture.  The first call incurs camera setup latency
        (~50-100 ms); subsequent calls read from the pre-filled buffer.

        Returns
        -------
        np.ndarray
            BGR image as a ``(height, width, 3)`` uint8 array.

        Raises
        ------
        RuntimeError
            If the window handle is invalid or capture fails.
        """
        if self.hwnd == 0:
            raise RuntimeError(
                "No window handle set. Construct WinCamCapture with "
                "window_title or hwnd before calling capture_frame()."
            )

        self._ensure_camera()
        assert self._camera is not None  # noqa: S101

        frame, _timestamp = self._camera.get_bgr_frame()
        return frame

    def is_window_visible(self) -> bool:
        """Check whether the target window is still visible/alive.

        Returns
        -------
        bool
            True if the window exists and is visible.
        """
        if self.hwnd == 0:
            return False
        try:
            return bool(win32gui.IsWindowVisible(self.hwnd))
        except Exception:
            return False

    def release(self) -> None:
        """Release wincam resources and reset state.

        Stops the DXCamera capture and resets the window handle.
        """
        self._stop_camera()
        self.hwnd = 0
        self.width = 0
        self.height = 0

    def __del__(self) -> None:
        """Ensure resources are released on garbage collection."""
        try:
            self.release()
        except Exception:
            pass
