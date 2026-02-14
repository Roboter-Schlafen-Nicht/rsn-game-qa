"""v1 breakout bot for testing and dataset collection.
No ML model, just a simple scripted controller."""

from pathlib import Path
import datetime as dt
import torch
from controllers.breakout_71_controller import BreakOut71Controller

ADB_PATH = "C:\\Users\\human\\AppData\\Local\\Android\\Sdk\\platform-tools\\adb.exe"

LOG_DIR = Path("data/live_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def make_run_log_dir(tag: str | None = None) -> Path:
    """_summary_

    Args:
        tag (str | None, optional): _description_. Defaults to None.

    Returns:
        Path: _description_
    """
    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S")
    suffix = f"_{tag}" if tag else ""
    run_dir = LOG_DIR / f"run_{ts}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_breakout_bot(
    dry_run: bool = True,
    log_tag: str | None = None,
):
    """
    Simple swipe loop.

    - Capture frame -> YOLO -> best HELP.
    - Apply confidence threshold, rate limiting, miss probability,
      jittered tap.
    - If dry_run=True, only logs what it would do.
    """
    device = "xpu" if torch.xpu.is_available() else "cpu"
    print(f"[ BREAKOUT 71 BOT] Using device={device}, dry_run={dry_run}")

    ctrl = BreakOut71Controller(adb_path=ADB_PATH)
    ctrl.connect_tcp()
    ctrl.ensure_device()

    log_dir = make_run_log_dir(log_tag)
    print(f"[BREAKOUT 71 BOT] Logging to {log_dir}")

    ctrl.random_continuous_slide(max_duration_s=15.0)


if __name__ == "__main__":
    run_breakout_bot(dry_run=False, log_tag="breakout_71_v1")
