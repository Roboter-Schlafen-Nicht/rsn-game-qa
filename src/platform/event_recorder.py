"""Game-agnostic human input event recorder.

Injects a JS snippet into the browser page that captures mouse,
keyboard, and wheel events into ``window.__rsn_events``.  The
platform reads and flushes this buffer each step via
:meth:`flush`.

Designed for *human play mode* (Phase 7): the human plays the game
in the Selenium-controlled browser while the platform records
input events alongside frames and game state.

Parameters
----------
driver : selenium.webdriver.remote.webdriver.WebDriver or None
    Selenium WebDriver instance.  Can be set later via
    :meth:`set_driver`.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class EventRecorder:
    """Record human input events from a Selenium-controlled browser.

    Injects a lightweight JS snippet that listens to ``mousemove``,
    ``mousedown``, ``mouseup``, ``click``, ``keydown``, ``keyup``,
    and ``wheel`` events on ``document``.  Events are buffered in
    ``window.__rsn_events`` and retrieved via :meth:`flush`.

    Mousemove events are throttled to avoid flooding the buffer
    (only recorded when the position changes beyond a small
    threshold since the last recorded move).

    Parameters
    ----------
    driver : object or None
        Selenium WebDriver instance.  Can be provided later via
        :meth:`set_driver`.
    """

    def __init__(self, driver: object | None = None) -> None:
        self._driver = driver
        self._injected: bool = False

    def set_driver(self, driver: object) -> None:
        """Set or replace the Selenium WebDriver.

        Resets the injection flag so the JS snippet is re-injected
        on the next :meth:`inject` or :meth:`flush` call.

        Parameters
        ----------
        driver : object
            Selenium WebDriver instance.
        """
        self._driver = driver
        self._injected = False

    @property
    def injection_js(self) -> str:
        """Return the JS snippet that installs event listeners.

        The snippet is idempotent: calling it multiple times does
        not add duplicate listeners (it checks for
        ``window.__rsn_events``).

        Returns
        -------
        str
            JavaScript source code.
        """
        return _INJECTION_JS

    def inject(self) -> None:
        """Inject the event recording JS into the current page.

        Idempotent: does nothing if already injected for this driver.

        Raises
        ------
        RuntimeError
            If no driver has been set.
        """
        if self._driver is None:
            raise RuntimeError(
                "EventRecorder: no driver set. "
                "Call set_driver() or pass driver to the constructor."
            )

        if self._injected:
            return

        self._driver.execute_script(self.injection_js)
        self._injected = True
        logger.debug("EventRecorder: JS snippet injected")

    def flush(self) -> list[dict]:
        """Read and clear all buffered events from the browser.

        Automatically calls :meth:`inject` if the snippet has not
        been injected yet.

        Returns
        -------
        list[dict]
            List of event dicts.  Each dict has at least ``type``
            and ``timestamp`` fields.  Empty list if no events
            occurred.

        Raises
        ------
        RuntimeError
            If no driver has been set.
        """
        if self._driver is None:
            raise RuntimeError(
                "EventRecorder: no driver set. "
                "Call set_driver() or pass driver to the constructor."
            )

        if not self._injected:
            self.inject()

        result = self._driver.execute_script(_FLUSH_JS)
        if result is None:
            return []
        return result


# -- JS snippets --------------------------------------------------------------

_INJECTION_JS = """\
(function() {
    if (window.__rsn_events) return;
    window.__rsn_events = [];
    var _lastMouseX = -1;
    var _lastMouseY = -1;
    var _lastMoveTimestamp = 0;
    var _THROTTLE_MS = 16;  // ~60 Hz max for mousemove

    function _record(e) {
        var evt = {
            type: e.type,
            timestamp: Date.now()
        };
        if (typeof e.clientX === 'number') {
            evt.x = e.clientX;
            evt.y = e.clientY;
        }
        if (typeof e.button === 'number' &&
            (e.type === 'mousedown' || e.type === 'mouseup' || e.type === 'click')) {
            evt.button = e.button;
        }
        if (typeof e.key === 'string') {
            evt.key = e.key;
            evt.code = e.code || '';
        }
        if (typeof e.deltaX === 'number') {
            evt.deltaX = e.deltaX;
            evt.deltaY = e.deltaY;
        }
        window.__rsn_events.push(evt);
    }

    function _recordMousemove(e) {
        var now = Date.now();
        if (now - _lastMoveTimestamp < _THROTTLE_MS) return;
        if (e.clientX === _lastMouseX && e.clientY === _lastMouseY) return;
        _lastMouseX = e.clientX;
        _lastMouseY = e.clientY;
        _lastMoveTimestamp = now;
        _record(e);
    }

    document.addEventListener('mousemove', _recordMousemove, true);
    document.addEventListener('mousedown', _record, true);
    document.addEventListener('mouseup', _record, true);
    document.addEventListener('click', _record, true);
    document.addEventListener('keydown', _record, true);
    document.addEventListener('keyup', _record, true);
    document.addEventListener('wheel', _record, true);
})();
"""

_FLUSH_JS = """\
var _evts = window.__rsn_events || [];
window.__rsn_events = [];
return _evts;
"""
