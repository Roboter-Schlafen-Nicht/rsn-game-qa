# Windows Capture & Input Spec

> Extracted from session 1 notes. This is the reference
> design for frame capture (BitBlt/GDI) and input injection (pydirectinput).

## Overview

The capture + input layer sits between the game window and the Gymnasium
environment. It provides two capabilities:

1. **Frame capture** — grab the game's client-area pixels as a numpy array
2. **Input injection** — send keyboard/mouse events to control the game

Both target native Windows desktop sessions (not headless) and require
`pywin32` and `pydirectinput`.

## Dependencies

```
pywin32       # win32gui, win32ui, win32con, win32api
pydirectinput # DirectInput keyboard/mouse
numpy         # frame array conversion
```

---

## Frame Capture (`_capture_frame`)

### Window Discovery

```python
import win32gui

def _find_window(self):
    hwnd = win32gui.FindWindow(None, self.window_title)
    if hwnd == 0:
        raise RuntimeError(f"Window '{self.window_title}' not found")
    self.hwnd = hwnd

def _focus_window(self):
    if self.hwnd is None:
        self._find_window()
    win32gui.SetForegroundWindow(self.hwnd)
```

### BitBlt Capture Implementation

Captures the client area of the game window and returns a numpy RGB array:

```python
import win32gui
import win32ui
import win32con
import numpy as np

def _capture_frame(self):
    if self.hwnd is None:
        self._find_window()

    # Get client rect (relative to client origin)
    left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
    width = right - left
    height = bottom - top

    # Translate client coords to screen coords
    client_point = win32gui.ClientToScreen(self.hwnd, (0, 0))
    screen_left, screen_top = client_point

    # Get device context for the entire window
    hwnd_dc = win32gui.GetWindowDC(self.hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()

    # Create a bitmap to store the screenshot
    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
    save_dc.SelectObject(bitmap)

    # BitBlt from window DC to memory DC
    save_dc.BitBlt(
        (0, 0), (width, height),
        mfc_dc,
        (0, 0),  # top-left of client area
        win32con.SRCCOPY
    )

    # Get bitmap bits as bytes
    bmpinfo = bitmap.GetInfo()
    bmpstr = bitmap.GetBitmapBits(True)

    # Convert to numpy array (BGRA -> BGR -> RGB)
    img = np.frombuffer(bmpstr, dtype=np.uint8)
    img = img.reshape((height, width, 4))  # 4 channels: BGRA
    img = img[:, :, :3]                    # drop alpha -> BGR
    img = img[:, :, ::-1]                  # BGR -> RGB

    # Clean up GDI objects
    win32gui.DeleteObject(bitmap.GetHandle())
    save_dc.DeleteDC()
    mfc_dc.DeleteDC()
    win32gui.ReleaseDC(self.hwnd, hwnd_dc)

    return img
```

### Game Region Cropping

If the browser UI is visible in the captured area, manually measure the game
canvas sub-rectangle and crop:

```python
# After capture:
img = img[y0:y1, x0:x1]  # hardcode offsets after manual inspection
```

### Timing

```python
def _wait_step(self):
    time.sleep(1.0 / 30.0)  # ~30 FPS equivalent
```

---

## Input Injection (`_apply_action`)

### Basic Tap Variant

```python
import pydirectinput

def _apply_action(self, action: int):
    self._focus_window()
    if action == 0:
        return  # no-op
    elif action == 1:
        pydirectinput.keyDown('left')
        pydirectinput.keyUp('left')
    elif action == 2:
        pydirectinput.keyDown('right')
        pydirectinput.keyUp('right')
```

### Smoother Motion Variant (Hold Key Briefly)

```python
def _apply_action(self, action: int):
    self._focus_window()
    key = None
    if action == 1:
        key = 'left'
    elif action == 2:
        key = 'right'
    if key is not None:
        pydirectinput.keyDown(key)
        time.sleep(0.03)  # ~1-2 frames of input
        pydirectinput.keyUp(key)
```

### Starting a New Game

```python
def _start_new_game(self):
    self._focus_window()
    pydirectinput.press('space')
```

---

## CI/GUI Session Requirements

For capture + input to work in CI, the Windows runner must:

1. Have an **interactive desktop session** (not a background service)
2. Use **auto-login** so a GUI session exists after reboot
3. Disable screen saver, sleep, and lock screen
4. Optionally use a dummy HDMI adapter for a real display

If using RDP: the session may "freeze" on disconnect. Prefer physical
console access, VNC, or Parsec-style remote tools. If RDP is necessary,
use a "caffeine"-style tool to prevent session idling.

---

## Error Handling

| Scenario                | Handling                                         |
|-------------------------|--------------------------------------------------|
| Window not found        | `RuntimeError` with clear message                |
| Window minimized        | Decide: restore or error out                     |
| GDI resource leak       | Always clean up in try/finally or `__del__`      |
| pydirectinput not found | `RuntimeError` at init time                      |
| pywin32 not found       | `RuntimeError` at init time                      |

## Source Files

- `src/capture/window_capture.py` — `WindowCapture` class
- `src/capture/input_controller.py` — `InputController` class
