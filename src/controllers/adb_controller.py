"""Controller for Android emulator/device interaction via ADB."""
import subprocess
import time
from typing import Optional, List, Tuple


class AdbError(Exception):
    """
    Exception raised when ADB (Android Debug Bridge) operations fail.

    This exception is raised when there are errors during ADB command
    execution, device connection issues, or other ADB-related operations
    in the emulator controller.
    """


class EmulatorController:
    """
    Minimal ADB helper for a single device/emulator.
    """

    BACK_KEY = 4
    HOME_KEY = 3

    def __init__(
        self,
        adb_path: str = "adb",
        serial: Optional[str] = None,
        default_timeout: float = 10.0,
    ):
        """
        :param adb_path: Path to adb executable.
        :param serial: Device serial (e.g. 'emulator-5554' or
                       '127.0.0.1:5555'). If None, uses the only
                       attached device.
        """
        self.adb_path = adb_path
        self.serial = serial
        self.default_timeout = default_timeout

    # ---------- low-level helpers ----------

    def _adb_base_cmd(self) -> List[str]:
        cmd = [self.adb_path]
        if self.serial:
            cmd += ["-s", self.serial]
        return cmd

    def _run(
        self,
        args: List[str],
        timeout: Optional[float] = None,
        check: bool = True,
    ) -> str:
        """Run an adb command and return stdout (decoded)."""
        if timeout is None:
            timeout = self.default_timeout
        cmd = self._adb_base_cmd() + args
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise AdbError(f"ADB timeout: {' '.join(cmd)}") from e

        out = proc.stdout.decode(errors="replace")
        err = proc.stderr.decode(errors="replace")
        if check and proc.returncode != 0:
            raise AdbError(
                f"ADB failed ({proc.returncode}): {' '.join(cmd)}\n{err}"
            )
        return out

    # ---------- device / connection ----------

    @staticmethod
    def list_devices(adb_path: str = "adb") -> List[Tuple[str, str]]:
        """
        :return: List of (serial, state) tuples.
        """
        proc = subprocess.run(
            [adb_path, "devices"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
        out = proc.stdout.decode(errors="replace").strip().splitlines()
        devices = []
        for line in out[1:]:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                devices.append((parts[0], parts[1]))
        return devices  # e.g. [('emulator-5554', 'device')]

    def connect_tcp(self, host: str = "127.0.0.1", port: int = 5555) -> None:
        """
        Connect to an emulator over TCP (if it's exposed on host:port).
        """
        proc = subprocess.run(
            [self.adb_path, "connect", f"{host}:{port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.default_timeout,
            check=False,
        )
        out = proc.stdout.decode(errors="replace")
        err = proc.stderr.decode(errors="replace")
        # Example output: "connected to 127.0.0.1:5555"
        if proc.returncode != 0:
            msg = (
                f"Failed to connect to {host}:{port} "
                f"(rc={proc.returncode}): "
                + err
            )
            raise AdbError(msg)
        if "connected to" not in out and "already connected" not in out:
            raise AdbError(f"Failed to connect to {host}:{port}: {out}{err}")
        self.serial = f"{host}:{port}"

    def ensure_device(self) -> None:
        """
        Ensure a device is selected; if not, auto-pick the single
        attached device.
        """
        if self.serial:
            return
        devices = self.list_devices(self.adb_path)
        ready = [d for d in devices if d[1] == "device"]
        if len(ready) == 0:
            raise AdbError("No ADB devices in 'device' state.")
        if len(ready) > 1:
            raise AdbError(
                "Multiple devices attached; specify serial explicitly."
            )
        self.serial = ready[0][0]

    # ---------- shell / input ----------

    def shell(self, cmd: str, timeout: Optional[float] = None) -> str:
        """
        Run a shell command: adb shell <cmd>.
        """
        # Pass the full command as a single argument to adb shell to preserve
        # quoting and complex commands.
        return self._run(["shell", cmd], timeout=timeout)

    def tap(self, x: int, y: int) -> None:
        """
        Simulate a tap at (x, y).
        """
        self.shell(f"input tap {x} {y}")

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 300,
    ) -> None:
        """
        Swipe from (x1, y1) to (x2, y2) over duration_ms milliseconds.
        """
        self.shell(f"input swipe {x1} {y1} {x2} {y2} {duration_ms}")

    def key_back(self) -> None:
        """
        Send the back key event to the emulator.

        Simulates pressing the back button by sending Android keyevent 4
        through the ADB shell command.

        Returns:
            None
        """
        self.shell(f"input keyevent {self.BACK_KEY}")

    def key_home(self) -> None:
        """
        Send the HOME key event to the emulator.

        This method simulates pressing the Android HOME button by sending
        keyevent 3 through the ADB shell command.

        Returns:
            None
        """
        self.shell(f"input keyevent {self.HOME_KEY}")

    # ---------- screencap ----------

    def screencap(
        self, timeout: Optional[float] = None
    ) -> bytes:
        """
        Capture the current screen and save as PNG.
        Uses 'adb shell screencap -p' for direct PNG bytes.
        """
        if timeout is None:
            timeout = self.default_timeout
        # Use shell to get raw PNG bytes without an extra interactive
        # shell which may mangle line endings.
        cmd = self._adb_base_cmd() + ["shell", "screencap", "-p"]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            png_bytes, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as e:
            proc.kill()
            raise AdbError("screencap timed out") from e

        if proc.returncode != 0:
            raise AdbError(f"screencap failed: {err.decode(errors='replace')}")

        if not png_bytes:
            raise AdbError("screencap returned no data")

        # Normalize CRLF -> LF (classic ADB quirk)
        png_bytes = png_bytes.replace(b"\r\n", b"\n")
        return png_bytes

    # ---------- simple helpers for your bot ----------

    def go_home_spam_back(self, n: int = 3, delay: float = 0.4) -> None:
        """
        Press back a few times, then home – good to 'reset' to main base view.
        """
        for _ in range(n):
            self.key_back()
            time.sleep(delay)
        self.key_home()
        time.sleep(delay)

    def sleep(self, seconds: float) -> None:
        """Sleep for the specified number of seconds.

        Args:
            seconds (float): The number of seconds to sleep.

        Returns:
            None
        """
        time.sleep(seconds)
