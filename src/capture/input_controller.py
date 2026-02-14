"""Input injection for Windows games using pydirectinput.

Provides high-level action methods (move paddle, click, key press) that
translate discrete/continuous RL actions into real input events sent to
the active game window.
"""

from __future__ import annotations

from typing import Optional

try:
    import pydirectinput

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

    def __init__(
        self,
        window_rect: Optional[tuple[int, int, int, int]] = None,
    ) -> None:
        if not _PYDIRECTINPUT_AVAILABLE:
            raise RuntimeError(
                "pydirectinput is required for InputController. "
                "Install it with: pip install pydirectinput"
            )

        self.window_rect = window_rect or (0, 0, 800, 600)

    def apply_action(self, action: int) -> None:
        """Apply a discrete action to the game.

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
        raise NotImplementedError("Discrete action injection not yet implemented")

    def move_mouse_to(self, x_norm: float, y_norm: float) -> None:
        """Move the mouse to a normalised position within the game window.

        Parameters
        ----------
        x_norm : float
            Horizontal position in [0.0, 1.0] (left to right).
        y_norm : float
            Vertical position in [0.0, 1.0] (top to bottom).
        """
        raise NotImplementedError("Mouse movement not yet implemented")

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
        raise NotImplementedError("Click injection not yet implemented")

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
        raise NotImplementedError("Key press injection not yet implemented")

    def hold_key(self, key: str) -> None:
        """Press and hold a key (call ``release_key`` to let go).

        Parameters
        ----------
        key : str
            Key name recognised by pydirectinput.
        """
        raise NotImplementedError("Key hold not yet implemented")

    def release_key(self, key: str) -> None:
        """Release a previously held key.

        Parameters
        ----------
        key : str
            Key name recognised by pydirectinput.
        """
        raise NotImplementedError("Key release not yet implemented")
