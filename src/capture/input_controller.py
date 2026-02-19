"""Input injection for Windows games using pydirectinput.

Provides high-level action methods (move paddle, click, key press) that
translate discrete/continuous RL actions into real input events sent to
the active game window.
"""

from __future__ import annotations

import time

try:
    import pydirectinput

    # Disable the built-in 100ms sleep after every pydirectinput call.
    # The default PAUSE=0.1 is a safety feature to prevent runaway input,
    # but it caps the control loop to ~10 FPS — unacceptable for RL.
    pydirectinput.PAUSE = 0

    _PYDIRECTINPUT_AVAILABLE = True
except ImportError:
    _PYDIRECTINPUT_AVAILABLE = False


class InputController:
    """Injects mouse/keyboard input into the game window.

    Uses pydirectinput (DirectInput scan-codes) which works with games that
    ignore standard Win32 messages. Falls back to pyautogui-style calls
    where DirectInput is unnecessary.

    Parameters
    ----------
    window_rect : tuple[int, int, int, int], optional
        The (left, top, right, bottom) client-area rectangle of the game
        window in screen coordinates. Used to translate normalised
        positions (0.0 - 1.0) to absolute pixel coordinates.

    Raises
    ------
    RuntimeError
        If pydirectinput is not installed.
    """

    # Discrete action mapping for Breakout-style games
    ACTION_NOOP = 0
    ACTION_LEFT = 1
    ACTION_RIGHT = 2
    ACTION_FIRE = 3

    # Maps discrete actions to pydirectinput key names.
    _ACTION_KEY_MAP: dict[int, str] = {
        ACTION_LEFT: "left",
        ACTION_RIGHT: "right",
        ACTION_FIRE: "space",
    }

    def __init__(
        self,
        window_rect: tuple[int, int, int, int] | None = None,
    ) -> None:
        if not _PYDIRECTINPUT_AVAILABLE:
            raise RuntimeError(
                "pydirectinput is required for InputController. "
                "Install it with: pip install pydirectinput"
            )

        self.window_rect = window_rect or (0, 0, 800, 600)

    # -- Helpers ---------------------------------------------------------------

    def _to_screen_coords(self, x_norm: float, y_norm: float) -> tuple[int, int]:
        """Convert normalised (0-1) position to absolute screen pixel coords.

        Parameters
        ----------
        x_norm : float
            Horizontal position in [0.0, 1.0] (left to right).
        y_norm : float
            Vertical position in [0.0, 1.0] (top to bottom).

        Returns
        -------
        tuple[int, int]
            ``(x_abs, y_abs)`` in screen pixels.
        """
        left, top, right, bottom = self.window_rect
        x_abs = int(left + x_norm * (right - left))
        y_abs = int(top + y_norm * (bottom - top))
        return x_abs, y_abs

    # -- Public API ------------------------------------------------------------

    def apply_action(self, action: int) -> None:
        """Apply a discrete action to the game.

        For movement actions, the key is held briefly (~30 ms) to produce
        a smooth one-frame movement impulse.

        Parameters
        ----------
        action : int
            One of ``ACTION_NOOP``, ``ACTION_LEFT``, ``ACTION_RIGHT``,
            ``ACTION_FIRE``.

        Raises
        ------
        ValueError
            If the action is not recognized.
        """
        if action == self.ACTION_NOOP:
            return

        key = self._ACTION_KEY_MAP.get(action)
        if key is None:
            raise ValueError(
                f"Unrecognised action {action!r}. "
                f"Expected one of NOOP({self.ACTION_NOOP}), "
                f"LEFT({self.ACTION_LEFT}), RIGHT({self.ACTION_RIGHT}), "
                f"FIRE({self.ACTION_FIRE})."
            )

        pydirectinput.keyDown(key)
        time.sleep(0.03)  # ~1-2 frames of input at 30 FPS
        pydirectinput.keyUp(key)

    def move_mouse_to(self, x_norm: float, y_norm: float) -> None:
        """Move the mouse to a normalised position within the game window.

        Parameters
        ----------
        x_norm : float
            Horizontal position in [0.0, 1.0] (left to right).
        y_norm : float
            Vertical position in [0.0, 1.0] (top to bottom).
        """
        x_abs, y_abs = self._to_screen_coords(x_norm, y_norm)
        pydirectinput.moveTo(x_abs, y_abs)

    def click(
        self, x_norm: float = 0.5, y_norm: float = 0.5, button: str = "left"
    ) -> None:
        """Click at a normalised position within the game window.

        Parameters
        ----------
        x_norm : float
            Horizontal position in [0.0, 1.0].
        y_norm : float
            Vertical position in [0.0, 1.0].
        button : str
            Mouse button: ``"left"``, ``"right"``, or ``"middle"``.
        """
        x_abs, y_abs = self._to_screen_coords(x_norm, y_norm)
        pydirectinput.click(x=x_abs, y=y_abs, button=button)

    def press_key(self, key: str, duration: float = 0.05) -> None:
        """Press and release a keyboard key.

        Parameters
        ----------
        key : str
            Key name recognised by pydirectinput (e.g. ``"left"``,
            ``"right"``, ``"space"``, ``"enter"``).
        duration : float
            How long to hold the key in seconds.
        """
        pydirectinput.keyDown(key)
        time.sleep(duration)
        pydirectinput.keyUp(key)

    def hold_key(self, key: str) -> None:
        """Press and hold a key (call ``release_key`` to let go).

        Parameters
        ----------
        key : str
            Key name recognised by pydirectinput.
        """
        pydirectinput.keyDown(key)

    def release_key(self, key: str) -> None:
        """Release a previously held key.

        Parameters
        ----------
        key : str
            Key name recognised by pydirectinput.
        """
        pydirectinput.keyUp(key)
