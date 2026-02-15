from pathlib import Path
from controllers.adb_controller import EmulatorController


def demo():
    """
    Run a brief demonstration of EmulatorController usage.
    This function initializes an EmulatorController with a configured
    adb executable, ensures the (only) attached emulator/device is
    selected, waits briefly for the UI to stabilize, and captures a
    screenshot saved to "screens/demo_000.png".
    Returns:
        None
    Raises:
        Exceptions propagated from EmulatorController or I/O operations
        (e.g. if adb is not available, no device is found, or the screenshot
            path cannot be written).
    """
    adb_path = "C:/Users/human/AppData/Local/Android/Sdk/platform-tools/adb.exe"
    adb = EmulatorController(adb_path=adb_path)
    adb.ensure_device()

    # Wait for UI, then screenshot
    adb.sleep(1.0)
    adb.screencap(Path("screens/demo_000.png"))


if __name__ == "__main__":
    demo()
