"""Tests for the EventRecorder platform module.

Verifies the JS event injection snippet, event flushing from the
browser, and event data parsing.  All tests mock Selenium to avoid
requiring a running browser.
"""

from __future__ import annotations

import json
from unittest import mock

import pytest

from src.platform.event_recorder import EventRecorder

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestEventRecorderConstruction:
    """Test EventRecorder initialisation."""

    def test_default_construction(self):
        """EventRecorder can be created with no arguments."""
        recorder = EventRecorder()
        assert recorder._driver is None
        assert recorder._injected is False

    def test_construction_with_driver(self):
        """EventRecorder accepts a Selenium WebDriver."""
        driver = mock.MagicMock()
        recorder = EventRecorder(driver=driver)
        assert recorder._driver is driver

    def test_set_driver_after_construction(self):
        """Driver can be set after construction."""
        recorder = EventRecorder()
        driver = mock.MagicMock()
        recorder.set_driver(driver)
        assert recorder._driver is driver
        # Injection flag resets when driver changes
        assert recorder._injected is False


# ---------------------------------------------------------------------------
# JS snippet
# ---------------------------------------------------------------------------


class TestJsSnippet:
    """Test the JS injection snippet generation."""

    def test_injection_js_is_string(self):
        """The injection JS property returns a non-empty string."""
        recorder = EventRecorder()
        js = recorder.injection_js
        assert isinstance(js, str)
        assert len(js) > 100

    def test_injection_js_creates_event_array(self):
        """The JS snippet initialises window.__rsn_events."""
        recorder = EventRecorder()
        js = recorder.injection_js
        assert "__rsn_events" in js

    def test_injection_js_listens_to_required_event_types(self):
        """The JS snippet listens to mouse, keyboard, and wheel events."""
        recorder = EventRecorder()
        js = recorder.injection_js
        for event_type in [
            "mousemove",
            "mousedown",
            "mouseup",
            "click",
            "keydown",
            "keyup",
            "wheel",
        ]:
            assert event_type in js

    def test_injection_js_records_timestamp(self):
        """Events include a timestamp field."""
        recorder = EventRecorder()
        js = recorder.injection_js
        # The JS should capture Date.now() or performance.now()
        assert "timestamp" in js.lower() or "Date.now" in js or "performance.now" in js


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------


class TestInjection:
    """Test injecting the JS snippet into the browser."""

    def test_inject_calls_execute_script(self):
        """inject() executes the JS snippet via the driver."""
        driver = mock.MagicMock()
        recorder = EventRecorder(driver=driver)
        recorder.inject()
        driver.execute_script.assert_called_once()
        call_js = driver.execute_script.call_args[0][0]
        assert "__rsn_events" in call_js
        assert recorder._injected is True

    def test_inject_idempotent(self):
        """inject() only executes the snippet once."""
        driver = mock.MagicMock()
        recorder = EventRecorder(driver=driver)
        recorder.inject()
        recorder.inject()
        assert driver.execute_script.call_count == 1

    def test_inject_raises_without_driver(self):
        """inject() raises RuntimeError if no driver is set."""
        recorder = EventRecorder()
        with pytest.raises(RuntimeError, match="driver"):
            recorder.inject()

    def test_inject_resets_on_new_driver(self):
        """Setting a new driver resets the injected flag."""
        driver1 = mock.MagicMock()
        driver2 = mock.MagicMock()
        recorder = EventRecorder(driver=driver1)
        recorder.inject()
        assert recorder._injected is True
        recorder.set_driver(driver2)
        assert recorder._injected is False
        recorder.inject()
        driver2.execute_script.assert_called_once()


# ---------------------------------------------------------------------------
# Flush events
# ---------------------------------------------------------------------------


class TestFlushEvents:
    """Test reading and clearing events from the browser."""

    def test_flush_returns_list_of_events(self):
        """flush() returns a list of event dicts."""
        driver = mock.MagicMock()
        raw_events = [
            {
                "type": "mousemove",
                "x": 100,
                "y": 200,
                "timestamp": 1000.0,
            },
            {
                "type": "click",
                "x": 150,
                "y": 250,
                "button": 0,
                "timestamp": 1050.0,
            },
        ]
        driver.execute_script.return_value = raw_events
        recorder = EventRecorder(driver=driver)
        recorder._injected = True
        events = recorder.flush()
        assert len(events) == 2
        assert events[0]["type"] == "mousemove"
        assert events[1]["type"] == "click"

    def test_flush_clears_browser_buffer(self):
        """flush() clears the window.__rsn_events array."""
        driver = mock.MagicMock()
        driver.execute_script.return_value = []
        recorder = EventRecorder(driver=driver)
        recorder._injected = True
        recorder.flush()
        call_js = driver.execute_script.call_args[0][0]
        # The flush JS should read and then clear the array
        assert "__rsn_events" in call_js

    def test_flush_returns_empty_list_when_no_events(self):
        """flush() returns [] when no events have been recorded."""
        driver = mock.MagicMock()
        driver.execute_script.return_value = []
        recorder = EventRecorder(driver=driver)
        recorder._injected = True
        events = recorder.flush()
        assert events == []

    def test_flush_returns_empty_list_when_js_returns_none(self):
        """flush() handles None return from JS gracefully."""
        driver = mock.MagicMock()
        driver.execute_script.return_value = None
        recorder = EventRecorder(driver=driver)
        recorder._injected = True
        events = recorder.flush()
        assert events == []

    def test_flush_auto_injects_if_not_injected(self):
        """flush() calls inject() first if not yet injected."""
        driver = mock.MagicMock()
        # First call: inject; second call: flush returns events
        driver.execute_script.side_effect = [None, []]
        recorder = EventRecorder(driver=driver)
        events = recorder.flush()
        assert recorder._injected is True
        assert events == []
        assert driver.execute_script.call_count == 2

    def test_flush_raises_without_driver(self):
        """flush() raises RuntimeError if no driver is set."""
        recorder = EventRecorder()
        with pytest.raises(RuntimeError, match="driver"):
            recorder.flush()


# ---------------------------------------------------------------------------
# Event parsing
# ---------------------------------------------------------------------------


class TestEventParsing:
    """Test parsing of raw JS event data."""

    def test_mouse_move_event_has_coordinates(self):
        """mousemove events have x, y fields."""
        driver = mock.MagicMock()
        raw = [{"type": "mousemove", "x": 42, "y": 84, "timestamp": 1000.0}]
        driver.execute_script.return_value = raw
        recorder = EventRecorder(driver=driver)
        recorder._injected = True
        events = recorder.flush()
        assert events[0]["x"] == 42
        assert events[0]["y"] == 84

    def test_keyboard_event_has_key_field(self):
        """keydown/keyup events include the key field."""
        driver = mock.MagicMock()
        raw = [
            {
                "type": "keydown",
                "key": "ArrowLeft",
                "code": "ArrowLeft",
                "timestamp": 2000.0,
            }
        ]
        driver.execute_script.return_value = raw
        recorder = EventRecorder(driver=driver)
        recorder._injected = True
        events = recorder.flush()
        assert events[0]["key"] == "ArrowLeft"
        assert events[0]["code"] == "ArrowLeft"

    def test_wheel_event_has_delta(self):
        """wheel events include deltaX and deltaY."""
        driver = mock.MagicMock()
        raw = [
            {
                "type": "wheel",
                "deltaX": 0,
                "deltaY": -120,
                "x": 100,
                "y": 200,
                "timestamp": 3000.0,
            }
        ]
        driver.execute_script.return_value = raw
        recorder = EventRecorder(driver=driver)
        recorder._injected = True
        events = recorder.flush()
        assert events[0]["deltaY"] == -120

    def test_click_event_has_button(self):
        """click/mousedown/mouseup events include button field."""
        driver = mock.MagicMock()
        raw = [
            {
                "type": "mousedown",
                "x": 50,
                "y": 60,
                "button": 2,
                "timestamp": 4000.0,
            }
        ]
        driver.execute_script.return_value = raw
        recorder = EventRecorder(driver=driver)
        recorder._injected = True
        events = recorder.flush()
        assert events[0]["button"] == 2

    def test_all_events_have_type_and_timestamp(self):
        """Every event dict has 'type' and 'timestamp' fields."""
        driver = mock.MagicMock()
        raw = [
            {"type": "mousemove", "x": 0, "y": 0, "timestamp": 100.0},
            {"type": "keydown", "key": "a", "code": "KeyA", "timestamp": 200.0},
            {"type": "wheel", "deltaX": 0, "deltaY": 1, "x": 0, "y": 0, "timestamp": 300.0},
        ]
        driver.execute_script.return_value = raw
        recorder = EventRecorder(driver=driver)
        recorder._injected = True
        events = recorder.flush()
        for event in events:
            assert "type" in event
            assert "timestamp" in event


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


class TestEventSerialisation:
    """Test that events are JSON-serialisable for JSONL recording."""

    def test_events_are_json_serialisable(self):
        """Flushed events can be serialised to JSON."""
        driver = mock.MagicMock()
        raw = [
            {"type": "mousemove", "x": 10, "y": 20, "timestamp": 500.0},
            {"type": "keydown", "key": "Space", "code": "Space", "timestamp": 600.0},
        ]
        driver.execute_script.return_value = raw
        recorder = EventRecorder(driver=driver)
        recorder._injected = True
        events = recorder.flush()
        serialised = json.dumps(events)
        assert isinstance(serialised, str)
        assert len(json.loads(serialised)) == 2


# ---------------------------------------------------------------------------
# Mousemove throttling
# ---------------------------------------------------------------------------


class TestMousemoveThrottling:
    """Test that the JS snippet throttles mousemove events."""

    def test_injection_js_throttles_mousemove(self):
        """The JS snippet throttles mousemove to avoid flooding."""
        recorder = EventRecorder()
        js = recorder.injection_js
        # Should contain some throttling mechanism (requestAnimationFrame,
        # setTimeout, or time delta check)
        has_throttle = (
            "throttle" in js.lower()
            or "requestAnimationFrame" in js
            or "lastMove" in js
            or "_lastMouse" in js
        )
        assert has_throttle, "JS snippet should throttle mousemove events"
