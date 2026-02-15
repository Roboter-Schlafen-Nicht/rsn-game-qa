"""Shared pytest fixtures for the rsn-game-qa test suite.

Provides session-scoped fixtures for integration tests that require
a live Breakout 71 game instance.  Tests are parameterized across
installed browsers (Chrome and Firefox) so each test runs once per
browser.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Generator

import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Integration test helpers
# ---------------------------------------------------------------------------

_WINDOWS = sys.platform == "win32"

# Browsers to test — filtered at collection time to only include
# those actually installed on the machine.
_TARGET_BROWSERS = ["chrome", "firefox"]


def _get_test_browsers() -> list[str]:
    """Return the subset of _TARGET_BROWSERS that are installed."""
    if not _WINDOWS:
        return []
    sys.path.insert(
        0, str(__import__("pathlib").Path(__file__).resolve().parent.parent)
    )
    from scripts._smoke_utils import get_available_browsers

    available = get_available_browsers()
    return [b for b in _TARGET_BROWSERS if b in available]


# ---------------------------------------------------------------------------
# Session-scoped: dev server (shared across all browsers)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _breakout71_server():
    """Start the Breakout 71 dev server once for the entire test session.

    This is an internal fixture — tests should use
    :func:`breakout71_loader` instead (which adds a browser).
    """
    if not _WINDOWS:
        pytest.skip("Integration tests require Windows (GDI capture)")

    from src.game_loader import load_game_config, create_loader

    config = load_game_config("breakout-71")
    loader = create_loader(config)

    logger.info("Starting Breakout 71 dev server ...")
    try:
        loader.setup()
        loader.start()
    except Exception as exc:
        pytest.skip(f"Could not start Breakout 71: {exc}")

    yield loader, config

    logger.info("Stopping Breakout 71 dev server ...")
    loader.stop()


# ---------------------------------------------------------------------------
# Per-browser: parameterized browser launch
# ---------------------------------------------------------------------------


@pytest.fixture(params=_get_test_browsers() or ["chrome"], scope="class")
def breakout71_loader(request, _breakout71_server) -> Generator:
    """Launch a dedicated browser and yield the game loader.

    Parameterized over installed browsers so each test class runs
    once per browser.  The dev server is shared (session-scoped);
    only the browser window is created and destroyed per parameter.

    The current browser name is available as
    ``request.param`` and is also set as ``loader._test_browser``.
    """
    sys.path.insert(
        0, str(__import__("pathlib").Path(__file__).resolve().parent.parent)
    )
    from scripts._smoke_utils import BrowserInstance

    loader, config = _breakout71_server
    browser_name: str = request.param

    url = loader.url or config.url
    logger.info("Launching %s for integration tests: %s", browser_name, url)

    browser: BrowserInstance | None = None
    try:
        browser = BrowserInstance(
            url,
            settle_seconds=12,
            window_size=(config.window_width, config.window_height),
            browser=browser_name,
        )
        logger.info("Browser launched: %s (PID %s)", browser.name, browser.pid)
    except Exception as exc:
        # Selenium raises WebDriverException when a driver/browser is
        # unavailable; BrowserInstance raises FileNotFoundError.
        pytest.skip(f"Browser {browser_name} not available: {exc}")

    # Stash browser name on loader for test introspection
    loader._test_browser = browser_name  # type: ignore[attr-defined]

    yield loader

    # ── Teardown: close browser ───────────────────────────────────────
    logger.info("Closing %s ...", browser_name)
    if browser is not None:
        browser.close()
    # Small delay between browser teardown/setup to avoid window races
    time.sleep(1)


@pytest.fixture(scope="class")
def window_capture(breakout71_loader):
    """A ``WindowCapture`` attached to the game window.

    Depends on :func:`breakout71_loader` so the game and browser are
    guaranteed to be running before we try to find the window.
    Scoped to ``class`` so it is recreated when the browser changes.
    """
    from src.capture import WindowCapture

    window_title = breakout71_loader.config.window_title or "Breakout"

    try:
        cap = WindowCapture(window_title=window_title)
    except RuntimeError as exc:
        pytest.skip(f"Could not find game window: {exc}")
        return  # unreachable — pytest.skip raises; satisfies type checker

    yield cap
    cap.release()
