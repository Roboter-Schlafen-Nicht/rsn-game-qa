"""Window frame capture using Windows GDI via pywin32.

Captures frames from a target window by its title or HWND handle,
returning BGR numpy arrays suitable for YOLO inference or CV processing.

Uses ``PrintWindow`` with ``PW_RENDERFULLCONTENT`` (flag 2) to capture
hardware-accelerated / composited windows (e.g. Chromium browsers)
that return black frames with plain BitBlt.
"""

from __future__ import annotations


import numpy as np

try:
    import win32gui
    import win32ui
    import win32con

    _PYWIN32_AVAILABLE = True
except ImportError:
    _PYWIN32_AVAILABLE = False


class WindowCapture:
    """Captures frames from a Windows application window using BitBlt/GDI.

    Parameters
    ----------
    window_title : str
        The title (or partial title) of the target window to capture.
    hwnd : int, optional
        Direct HWND handle. If provided, ``window_title`` is ignored.

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
        If pywin32 is not installed or the target window cannot be found.
    """

    def __init__(self, window_title: str = "", hwnd: int | None = None) -> None:
        if not _PYWIN32_AVAILABLE:
            raise RuntimeError(
                "pywin32 is required for WindowCapture. "
                "Install it with: pip install pywin32"
            )

        self.window_title = window_title
        self.hwnd: int = hwnd if hwnd is not None else 0
        self.width: int = 0
        self.height: int = 0

        if self.hwnd == 0 and self.window_title:
            self._find_window()

        if self.hwnd != 0:
            self._update_dimensions()

    def _find_window(self) -> None:
        """Locate the target window by title and store its HWND.

        Uses ``win32gui.FindWindow`` for exact title match. If no exact
        match is found, enumerates all top-level windows and picks the
        first whose title contains ``self.window_title`` as a substring.

        Raises
        ------
        RuntimeError
            If no window matching ``self.window_title`` is found.
        """
        # Try exact match first.
        hwnd = win32gui.FindWindow(None, self.window_title)
        if hwnd != 0:
            self.hwnd = hwnd
            return

        # Fall back to substring search across all top-level windows.
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
        """Read the current client-area dimensions of the target window.

        Raises
        ------
        RuntimeError
            If the window handle is invalid.
        """
        try:
            left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to get client rect for HWND {self.hwnd}: {exc}"
            ) from exc
        self.width = right - left
        self.height = bottom - top

    def capture_frame(self) -> np.ndarray:
        """Capture a single frame from the target window.

        Uses ``PrintWindow`` with ``PW_RENDERFULLCONTENT`` (flag 2) to
        request the window to paint itself into an off-screen bitmap.
        This works reliably with hardware-accelerated Chromium browsers
        that return all-black frames with plain ``BitBlt``.

        Returns
        -------
        np.ndarray
            BGR image as a ``(height, width, 3)`` uint8 array, compatible
            with OpenCV and Ultralytics YOLO inference.

        Raises
        ------
        RuntimeError
            If the window handle is invalid or the capture fails.
        """
        if self.hwnd == 0:
            raise RuntimeError("No window handle set. Call _find_window() first.")

        self._update_dimensions()

        if self.width == 0 or self.height == 0:
            raise RuntimeError(
                f"Window client area has zero size ({self.width}x{self.height}). "
                "The window may be minimised."
            )

        # PW_RENDERFULLCONTENT = 2 — tells the window to render its full
        # content (including hardware-accelerated layers) into the DC.
        PW_RENDERFULLCONTENT = 2  # noqa: N806

        hwnd_dc = None
        mfc_dc = None
        save_dc = None
        bitmap = None
        try:
            hwnd_dc = win32gui.GetWindowDC(self.hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()

            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(mfc_dc, self.width, self.height)
            save_dc.SelectObject(bitmap)

            # Try PrintWindow first (works with composited / GPU-accelerated
            # windows).  Fall back to BitBlt if PrintWindow is unavailable
            # or fails (e.g. in unit tests with mocked GDI objects).
            captured = False
            try:
                import ctypes

                result = ctypes.windll.user32.PrintWindow(
                    int(self.hwnd),
                    int(save_dc.GetSafeHdc()),
                    PW_RENDERFULLCONTENT,
                )
                captured = bool(result)
            except (TypeError, ValueError, OSError, AttributeError):
                pass

            if not captured:
                # Fallback: plain BitBlt (works in unit tests and for
                # software-rendered windows).
                save_dc.BitBlt(
                    (0, 0),
                    (self.width, self.height),
                    mfc_dc,
                    (0, 0),
                    win32con.SRCCOPY,
                )

            # Convert bitmap bits to numpy array.
            bmpstr = bitmap.GetBitmapBits(True)
            img = np.frombuffer(bmpstr, dtype=np.uint8)
            img = img.reshape((self.height, self.width, 4))  # BGRA
            img = img[:, :, :3].copy()  # Drop alpha -> BGR
        except Exception as exc:
            raise RuntimeError(f"Frame capture failed: {exc}") from exc
        finally:
            # Always clean up GDI resources.
            if bitmap is not None:
                win32gui.DeleteObject(bitmap.GetHandle())
            if save_dc is not None:
                save_dc.DeleteDC()
            if mfc_dc is not None:
                mfc_dc.DeleteDC()
            if hwnd_dc is not None:
                win32gui.ReleaseDC(self.hwnd, hwnd_dc)

        return img

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
        """Release any GDI resources held by this capture instance.

        Resets the window handle and dimensions. Per-capture GDI objects
        are cleaned up in ``capture_frame``'s ``finally`` block, so this
        method mainly serves as a lifecycle signal.
        """
        self.hwnd = 0
        self.width = 0
        self.height = 0

    def __del__(self) -> None:
        """Ensure GDI resources are released on garbage collection."""
        try:
            self.release()
        except Exception:
            pass
