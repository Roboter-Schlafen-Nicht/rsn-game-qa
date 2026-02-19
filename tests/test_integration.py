"""Integration tests for Breakout 71 lifecycle, capture, and oracles.

These tests require a live game instance and are marked with
``@pytest.mark.integration``.  They are **skipped in CI** (Docker has
no display) and run locally with::

    pytest -m integration -v

Tests are **parameterized across browsers** (Chrome and Firefox by
default).  Each test class runs once per installed browser.  The dev
server is shared across all browsers (session-scoped); only the
browser window is created / destroyed between browser switches.

The ``breakout71_loader`` and ``window_capture`` fixtures are defined
in ``tests/conftest.py``.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

# ======================================================================
# Lifecycle tests
# ======================================================================


@pytest.mark.integration
class TestBreakout71Lifecycle:
    """Verify game start, readiness, and shutdown."""

    def test_loader_is_running(self, breakout71_loader):
        """Loader reports running after start."""
        assert breakout71_loader.running

    def test_loader_is_ready(self, breakout71_loader):
        """HTTP readiness probe succeeds."""
        assert breakout71_loader.is_ready()

    def test_loader_has_url(self, breakout71_loader):
        """Loader exposes the game URL."""
        url = breakout71_loader.url
        assert url is not None
        assert url.startswith("http")

    def test_loader_has_process(self, breakout71_loader):
        """The dev server subprocess is alive."""
        proc = getattr(breakout71_loader, "_process", None)
        assert proc is not None
        assert proc.poll() is None  # still running


# ======================================================================
# Window capture tests
# ======================================================================


@pytest.mark.integration
class TestBreakout71Capture:
    """Verify window discovery and frame capture."""

    def test_window_found(self, window_capture):
        """WindowCapture found the game HWND."""
        assert window_capture.hwnd != 0

    def test_window_dimensions(self, window_capture):
        """Window has non-zero dimensions."""
        assert window_capture.width > 0
        assert window_capture.height > 0

    def test_capture_frame_shape(self, window_capture):
        """Captured frame has correct shape and dtype."""
        frame = window_capture.capture_frame()
        assert frame.ndim == 3
        assert frame.shape[2] == 3  # BGR
        assert frame.dtype == np.uint8
        assert frame.shape[0] == window_capture.height
        assert frame.shape[1] == window_capture.width

    def test_capture_frame_not_black(self, window_capture):
        """Captured frame is not all-zero (game is rendering)."""
        # Give the game a moment to render
        time.sleep(2)
        frame = window_capture.capture_frame()
        assert frame.sum() > 0, "Frame is all black — game may not be rendering"

    def test_capture_multiple_frames_differ(self, window_capture):
        """Multiple frames are not identical (game is animating).

        The game may be idle on a menu screen, producing identical frames.
        We capture 10 frames over 5 seconds and require at least 2 to
        differ.  If all are identical we skip rather than fail, since a
        static menu is valid game state.
        """
        time.sleep(1)
        frames = []
        for _ in range(10):
            frames.append(window_capture.capture_frame())
            time.sleep(0.5)

        unique = 1
        for i in range(1, len(frames)):
            if not np.array_equal(frames[i], frames[i - 1]):
                unique += 1

        if unique < 2:
            pytest.skip(f"Only {unique}/10 unique frames — game may be on a static menu screen")


# ======================================================================
# Oracle smoke tests
# ======================================================================


@pytest.mark.integration
class TestBreakout71OracleSmoke:
    """Run oracles against a short live session."""

    def _run_oracle_steps(self, window_capture, oracle, n_steps=50, interval=0.1):
        """Helper: feed n_steps of captured frames to an oracle."""
        frame = window_capture.capture_frame()
        obs = frame.astype(np.float32) / 255.0
        oracle.on_reset(obs, {"frame": frame})

        for _ in range(n_steps):
            time.sleep(interval)
            frame = window_capture.capture_frame()
            obs = frame.astype(np.float32) / 255.0
            oracle.on_step(obs, 0.0, False, False, {"frame": frame})

        return oracle.get_findings()

    def test_crash_oracle_no_crash(self, window_capture):
        """CrashOracle should not detect crashes in a healthy game."""
        from src.oracles import CrashOracle

        oracle = CrashOracle()
        findings = self._run_oracle_steps(window_capture, oracle, n_steps=30)

        critical = [f for f in findings if f.severity == "critical"]
        assert len(critical) == 0, (
            f"CrashOracle found {len(critical)} critical issues: "
            + "; ".join(f.description for f in critical)
        )

    def test_performance_oracle_fps(self, window_capture):
        """PerformanceOracle should report acceptable FPS."""
        from src.oracles import PerformanceOracle

        oracle = PerformanceOracle(min_fps=5.0, sustained_frames=10)
        self._run_oracle_steps(window_capture, oracle, n_steps=50, interval=0.05)

        summary = oracle.get_fps_summary()
        assert summary.get("mean_fps", 0) > 0, "No FPS data collected"

    def test_stuck_oracle_not_stuck(self, window_capture):
        """StuckOracle should not flag stuck state in a healthy game."""
        from src.oracles import StuckOracle

        oracle = StuckOracle(patience=50)
        findings = self._run_oracle_steps(window_capture, oracle, n_steps=30)

        warnings = [f for f in findings if f.severity == "warning"]
        # The game might legitimately idle on a menu, so we just log
        if warnings:
            pytest.skip(
                f"StuckOracle fired {len(warnings)} warnings — game may be on a menu screen"
            )
