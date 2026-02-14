import random
from pathlib import Path
from typing import Optional, Dict, Tuple
from controllers.adb_controller import EmulatorController, AdbError


class LastWarController(EmulatorController):
    """
    Game-specific helper for Last War built on top of EmulatorController.
    Encapsulates known UI coordinates and common actions.
    """

    def __init__(
        self,
        adb_path: str = "adb",
        serial: Optional[str] = None,
        default_timeout: float = 10.0,
        coords: Optional[Dict[str, Tuple[int, int]]] = None,
    ):
        super().__init__(
            adb_path=adb_path, serial=serial, default_timeout=default_timeout
        )

        # Default coordinate mapping (example values, adjust to your emulator!)
        self.coords: Dict[str, Tuple[int, int]] = {
            "DAILY": (1000, 1800),  # daily/events icon in bottom/right bar
            "ALLIANCE": (100, 1800),  # alliance icon in bottom/left bar
            "MAIL": (1000, 120),  # mail icon at top-right
            "CLOSE": (1040, 80),  # generic close 'X' at top-right
            "SAFE_CENTER": (540, 960),  # safe tap in center
        }
        if coords:
            self.coords.update(coords)

    # ---------- coordinate helpers ----------

    def set_coord(self, name: str, x: int, y: int) -> None:
        self.coords[name] = (x, y)

    def get_coord(self, name: str) -> Tuple[int, int]:
        if name not in self.coords:
            raise AdbError(f"Unknown coord '{name}'")
        return self.coords[name]

    # ---------- high-level navigation ----------

    def reset_to_base(self, n_back: int = 3, delay: float = 0.4) -> None:
        """
        Try to reliably reach the main base view.
        """
        self.go_home_spam_back(n=n_back, delay=delay)

    def open_daily_menu(self, delay: float = 1.0) -> None:
        self.tap(*self.get_coord("DAILY"))
        self.sleep(delay)

    def open_alliance_menu(self, delay: float = 1.0) -> None:
        self.tap(*self.get_coord("ALLIANCE"))
        self.sleep(delay)

    def open_mail_menu(self, delay: float = 1.0) -> None:
        self.tap(*self.get_coord("MAIL"))
        self.sleep(delay)

    def close_popup(self, delay: float = 0.5) -> None:
        """
        Tap the generic close ('X') location.
        """
        self.tap(*self.get_coord("CLOSE"))
        self.sleep(delay)

    # ---------- generic state sampler for data collection ----------

    def collect_state_frames(
        self,
        state_name: str,
        n_frames: int,
        out_dir: Path,
        start_index: int = 0,
        random_taps: bool = True,
        tap_region: Tuple[int, int, int, int] = (200, 400, 880, 1500),
        min_delay: float = 0.4,
        max_delay: float = 1.0,
    ) -> int:
        """
        Collect screenshots in a given state.
        You are responsible for navigating into the state before calling.

        :param state_name: A label like 'base', 'daily', 'alliance', 'mail'.
        :param n_frames: Number of screenshots to capture.
        :param out_dir: Base output directory for frames and metadata.
        :param start_index: Starting index for naming.
        :param random_taps: If True, occasionally tap in a safe region for
                            variation.
        :param tap_region: (x_min, y_min, x_max, y_max) for random taps.
        :return: Next index after the last captured frame.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        x_min, y_min, x_max, y_max = tap_region
        idx = start_index

        for _ in range(n_frames):
            if random_taps and random.random() < 0.3:
                rx = random.randint(x_min, x_max)
                ry = random.randint(y_min, y_max)
                self.tap(rx, ry)
                self.sleep(0.3)

            png_path = out_dir / f"{state_name}_{idx:06d}.png"
            self.screencap(png_path)
            print(f"[{state_name}] saved {png_path}")
            idx += 1

            delay = random.uniform(min_delay, max_delay)
            self.sleep(delay)

        return idx

    # ---------- canned routines for your v1 loops ----------

    def collect_base_view(
        self,
        n_frames: int,
        out_dir: Path,
        start_index: int = 0,
    ) -> int:
        """
        Reset to base and collect base screenshots.
        """
        self.reset_to_base()
        return self.collect_state_frames(
            state_name="base",
            n_frames=n_frames,
            out_dir=out_dir,
            start_index=start_index,
            random_taps=True,
        )

    def collect_daily_view(
        self,
        n_frames: int,
        out_dir: Path,
        start_index: int = 0,
    ) -> int:
        """
        Navigate to daily/events menu and collect screenshots.
        """
        self.reset_to_base()
        self.open_daily_menu()
        idx = self.collect_state_frames(
            state_name="daily",
            n_frames=n_frames,
            out_dir=out_dir,
            start_index=start_index,
            random_taps=True,
        )
        self.close_popup()
        return idx

    def collect_alliance_view(
        self,
        n_frames: int,
        out_dir: Path,
        start_index: int = 0,
    ) -> int:
        """
        Navigate to alliance menu and collect screenshots.
        """
        self.reset_to_base()
        self.open_alliance_menu()
        idx = self.collect_state_frames(
            state_name="alliance",
            n_frames=n_frames,
            out_dir=out_dir,
            start_index=start_index,
            random_taps=True,
        )
        self.close_popup()
        return idx

    def collect_mail_view(
        self,
        n_frames: int,
        out_dir: Path,
        start_index: int = 0,
    ) -> int:
        """
        Navigate to mail menu and collect screenshots.
        """
        self.reset_to_base()
        self.open_mail_menu()
        idx = self.collect_state_frames(
            state_name="mail",
            n_frames=n_frames,
            out_dir=out_dir,
            start_index=start_index,
            random_taps=True,
        )
        self.close_popup()
        return idx
