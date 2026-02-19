"""Shared utilities for smoke-test scripts.

Provides consistent logging setup, output directory management,
timestamp helpers, and common argument parsing for all ``smoke_*.py``
scripts.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "output"


def ensure_output_dir(sub: str | None = None) -> Path:
    """Create and return the output directory.

    Parameters
    ----------
    sub : str, optional
        Subdirectory under ``output/``.  Created if it doesn't exist.

    Returns
    -------
    Path
        Absolute path to the (sub)directory.
    """
    out = DEFAULT_OUTPUT_DIR
    if sub:
        out = out / sub
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_COLORS = {
    "DEBUG": "\033[90m",  # grey
    "INFO": "\033[36m",  # cyan
    "WARNING": "\033[33m",  # yellow
    "ERROR": "\033[31m",  # red
    "CRITICAL": "\033[1;31m",  # bold red
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    """Formatter that prepends a colored level tag and timestamp."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now(tz=UTC).strftime("%H:%M:%S")
        color = _COLORS.get(record.levelname, "")
        tag = f"{color}[{ts}] {record.levelname:<8}{_RESET}"
        return f"{tag} {record.getMessage()}"


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure root logger with colored output.

    Parameters
    ----------
    verbose : bool
        If ``True``, set level to ``DEBUG``; otherwise ``INFO``.

    Returns
    -------
    logging.Logger
        The root logger.
    """
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ColorFormatter())

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers on re-init
    root.handlers.clear()
    root.addHandler(handler)
    return root


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


class Timer:
    """Simple context-manager stopwatch."""

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed = time.perf_counter() - self._start


# ---------------------------------------------------------------------------
# Common CLI arguments
# ---------------------------------------------------------------------------


def base_argparser(description: str) -> argparse.ArgumentParser:
    """Return an ``ArgumentParser`` pre-loaded with common options.

    Includes ``--output-dir``, ``--verbose``, and ``--config``.
    """
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--config",
        default="breakout-71",
        help="Game config name (default: %(default)s)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: output/)",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def timestamp_str() -> str:
    """Return a filesystem-safe UTC timestamp string."""
    return datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")


def save_frame_png(frame, path: Path) -> None:
    """Save a BGR numpy frame as a PNG file.

    Parameters
    ----------
    frame : np.ndarray
        BGR uint8 image array (H, W, 3).
    path : Path
        Destination file path.
    """
    import cv2

    cv2.imwrite(str(path), frame)


# ---------------------------------------------------------------------------
# Browser launcher (Selenium WebDriver)
# ---------------------------------------------------------------------------

#: Browsers supported for integration testing.  Selenium handles
#: driver binaries, process lifecycle, and profile isolation.
_SUPPORTED_BROWSERS = ("chrome", "edge", "firefox")

logger_browser = logging.getLogger(__name__ + ".browser")


def get_available_browsers() -> list[str]:
    """Return names of browsers for which Selenium drivers are available.

    Attempts a lightweight probe for each supported browser by
    checking whether the executable exists on disk.  This avoids
    the cost of actually launching a WebDriver.

    Returns
    -------
    list[str]
        E.g. ``["chrome", "firefox"]``.
    """
    _BROWSER_PATHS: dict[str, list[str]] = {
        "chrome": [
            # Windows
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            # Linux
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
        ],
        "edge": [
            # Windows
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            # Linux
            "/usr/bin/microsoft-edge",
            "/usr/bin/microsoft-edge-stable",
        ],
        "firefox": [
            # Windows
            r"C:\Program Files\Mozilla Firefox\firefox.exe",
            # Linux
            "/usr/bin/firefox",
        ],
    }
    available: list[str] = []
    for name in _SUPPORTED_BROWSERS:
        # Check known paths first
        for p in _BROWSER_PATHS.get(name, []):
            if Path(p).is_file():
                available.append(name)
                break
        else:
            # Fall back to shutil.which() for PATH-based detection
            import shutil

            cmd = {
                "chrome": "google-chrome",
                "edge": "microsoft-edge",
                "firefox": "firefox",
            }.get(name, name)
            if shutil.which(cmd):
                available.append(name)
    return available


class BrowserInstance:
    """Manage a browser window via Selenium WebDriver.

    Uses Selenium to launch an isolated browser instance with a clean
    profile, navigate to the given URL, and set the window size.
    :meth:`close` calls ``driver.quit()`` which reliably tears down
    the entire browser process tree on all platforms.

    Parameters
    ----------
    url : str
        The URL to open.
    settle_seconds : float
        Seconds to wait after launch for the page to load.
    window_size : tuple[int, int], optional
        ``(width, height)`` in pixels.  Defaults to ``(1280, 720)``.
    browser : str, optional
        Browser to use (``"chrome"``, ``"edge"``, or ``"firefox"``).
        If ``None``, the first available is used.
    headless : bool
        If ``True``, launch the browser in headless mode.  Uses
        ``--headless=new`` for Chrome/Edge and ``--headless`` for
        Firefox.  Adds ``--disable-gpu`` for stability.
    """

    def __init__(
        self,
        url: str,
        settle_seconds: float = 6.0,
        window_size: tuple[int, int] = (1280, 720),
        browser: str | None = None,
        headless: bool = False,
    ) -> None:
        # Resolve which browser to use
        if browser is None:
            available = get_available_browsers()
            if not available:
                raise FileNotFoundError(
                    "No supported browser found. Install Chrome, Edge, or Firefox."
                )
            browser = available[0]
        if browser not in _SUPPORTED_BROWSERS:
            raise FileNotFoundError(
                f"Unknown browser {browser!r}. Supported: {list(_SUPPORTED_BROWSERS)}"
            )

        self.name: str = browser
        self.url: str = url
        self._window_size: tuple[int, int] = window_size
        self._headless: bool = headless
        self._settle_seconds: float = settle_seconds

        self._driver = self._create_driver()

        # Navigate to the URL
        self._driver.get(url)

        # Set window size explicitly (ensures consistency)
        w, h = window_size
        self._driver.set_window_size(w, h)

        logger_browser.info(
            "Browser ready — waiting %.1fs for page to settle ...",
            settle_seconds,
        )
        time.sleep(settle_seconds)

    # -- Driver creation --------------------------------------------------

    def _create_driver(self):
        """Create and return a new Selenium WebDriver instance.

        Uses the browser name, window size, and headless flag stored
        at construction time.  Does **not** navigate to any URL or
        set the window size — the caller handles that.

        Returns
        -------
        selenium.webdriver.Remote
            A fresh WebDriver instance.

        Raises
        ------
        RuntimeError
            If Selenium is not installed.
        """
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options as ChromeOptions
            from selenium.webdriver.edge.options import Options as EdgeOptions
            from selenium.webdriver.firefox.options import Options as FirefoxOptions
        except ImportError as exc:
            raise RuntimeError(
                "Selenium is required for BrowserInstance but is not installed. "
                "Install it with 'pip install selenium' or add it to "
                "environment.yml pip dependencies."
            ) from exc

        w, h = self._window_size
        logger_browser.info("Launching %s via Selenium ...", self.name)

        if self.name == "chrome":
            opts = ChromeOptions()
            opts.add_argument("--no-first-run")
            opts.add_argument("--no-default-browser-check")
            opts.add_argument("--disable-extensions")
            opts.add_argument("--disable-popup-blocking")
            opts.add_argument("--disable-translate")
            opts.add_argument("--disable-features=TranslateUI")
            opts.add_argument("--no-service-autorun")
            opts.add_argument("--password-store=basic")
            opts.add_argument(f"--window-size={w},{h}")
            if self._headless:
                opts.add_argument("--headless=new")
                opts.add_argument("--disable-gpu")
                opts.add_argument("--no-sandbox")
                opts.add_argument("--disable-dev-shm-usage")
            return webdriver.Chrome(options=opts)

        if self.name == "edge":
            opts = EdgeOptions()
            opts.add_argument("--no-first-run")
            opts.add_argument("--no-default-browser-check")
            opts.add_argument("--disable-extensions")
            opts.add_argument(f"--window-size={w},{h}")
            if self._headless:
                opts.add_argument("--headless=new")
                opts.add_argument("--disable-gpu")
            return webdriver.Edge(options=opts)

        # firefox
        opts = FirefoxOptions()
        opts.set_preference("browser.startup.homepage_override.mstone", "ignore")
        opts.set_preference("startup.homepage_welcome_url", "")
        opts.set_preference("startup.homepage_welcome_url.additional", "")
        opts.set_preference("startup.homepage_override_url", "")
        opts.set_preference("browser.startup.firstrunSkipsHomepage", True)
        opts.set_preference("browser.shell.checkDefaultBrowser", False)
        opts.set_preference("browser.shell.skipDefaultBrowserCheckOnFirstRun", True)
        opts.set_preference("datareporting.policy.dataSubmissionEnabled", False)
        opts.set_preference(
            "datareporting.policy.dataSubmissionPolicyBypassNotification",
            True,
        )
        opts.set_preference("toolkit.telemetry.reportingpolicy.firstRun", False)
        opts.set_preference("datareporting.policy.firstRunURL", "")
        opts.set_preference("browser.aboutwelcome.enabled", False)
        opts.set_preference("trailhead.firstrun.didSeeAboutWelcome", True)
        if self._headless:
            opts.add_argument("--headless")
        opts.add_argument(f"--width={w}")
        opts.add_argument(f"--height={h}")
        return webdriver.Firefox(options=opts)

    # -- Public API -----------------------------------------------------

    @property
    def pid(self) -> int | None:
        """Return the browser process PID, or ``None`` if not running.

        Selenium exposes the browser PID via the service process; this
        returns the service (driver) PID which manages the browser.
        """
        if self._driver is None:
            return None
        try:
            return self._driver.service.process.pid
        except Exception:
            return None

    @property
    def driver(self):
        """Return the underlying Selenium WebDriver instance."""
        return self._driver

    def close(self) -> None:
        """Quit the browser and clean up all processes.

        Calls ``driver.quit()`` which sends a shutdown command to the
        browser and terminates the driver service.  Selenium handles
        all process-tree cleanup reliably across Chrome, Edge, and
        Firefox.
        """
        if self._driver is None:
            return

        logger_browser.info("Closing %s via Selenium ...", self.name)
        try:
            self._driver.quit()
        except Exception:
            pass
        self._driver = None
        logger_browser.info("Browser closed.")

    def is_alive(self) -> bool:
        """Check whether the browser session is still responsive.

        Attempts to read ``driver.title`` as a lightweight health
        check.  Returns ``False`` if the driver is ``None`` or if
        the query raises any exception (tab crash, process exit,
        connection refused, etc.).

        Returns
        -------
        bool
            ``True`` if the browser is responsive.
        """
        if self._driver is None:
            return False
        try:
            _ = self._driver.title
            return True
        except Exception:
            return False

    def restart(self) -> None:
        """Close the current browser and launch a fresh session.

        Handles the case where the old driver is already dead (its
        ``quit()`` may throw).  After creating a new driver, navigates
        to the original URL, sets the window size, and waits the
        configured settle time.

        The new driver is stored in ``self._driver`` and is also
        accessible via ``self.driver``.
        """
        # Best-effort close of the old driver
        if self._driver is not None:
            try:
                self._driver.quit()
            except Exception:
                pass

        logger_browser.info("Restarting %s browser ...", self.name)

        self._driver = self._create_driver()
        self._driver.get(self.url)

        w, h = self._window_size
        self._driver.set_window_size(w, h)

        logger_browser.info(
            "Browser restarted — waiting %.1fs for page to settle ...",
            self._settle_seconds,
        )
        time.sleep(self._settle_seconds)
