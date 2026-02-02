# help_bot_loop.py
import time
import random
import torch
import cv2
import numpy as np
from controllers.last_war_controller import LastWarController
from policies.help_policy import HelpOnlyPolicy

ADB_PATH = ("C:\\Users\\human\\AppData\\Local\\Android\\Sdk\\platform-tools"
            "\\adb.exe")


def jitter_point(x: int, y: int, radius_px: int = 8) -> tuple[int, int]:
    """Return a jittered (x, y) within ±radius_px."""
    return (
        x + random.randint(-radius_px, radius_px),
        y + random.randint(-radius_px, radius_px),
    )


def run_help_bot(
    model_path: str,
    dry_run: bool = True,
    max_steps: int = 10_000,
    conf_thres: float = 0.05,
    miss_click_prob: float = 0.15,
    actions_per_minute: int = 18,
    observe_delay_range: tuple[float, float] = (0.4, 1.0),
    click_delay_range: tuple[float, float] = (0.5, 1.5),
    jitter_radius_px: int = 10,
):
    """
    Simple HELP-only loop.

    - Capture frame -> YOLO -> best HELP.
    - Apply confidence threshold, rate limiting, miss probability,
      jittered tap.
    - If dry_run=True, only logs what it would do.
    """
    device = "xpu" if torch.xpu.is_available() else "cpu"
    print(f"[HELP BOT] Using device={device}, dry_run={dry_run}")

    ctrl = LastWarController(adb_path=ADB_PATH)
    ctrl.connect_tcp()
    ctrl.ensure_device()

    policy = HelpOnlyPolicy(
        model_path=model_path,
        device=device,
        conf_thres=conf_thres,
    )

    min_interval = 60.0 / max(actions_per_minute, 1)
    last_action_ts = 0.0

    step = 0
    while step < max_steps:
        step += 1

        # Random "thinking" delay before each observation
        time.sleep(random.uniform(*observe_delay_range))
        png_bytes = ctrl.screencap()
        frame = cv2.imdecode(
            np.frombuffer(png_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        det = policy.infer(frame)

        if det is None:
            # Occasionally hit back like a human trying to recover
            if step % 200 == 0:
                msg = f"[{step}] No HELP detected; would press BACK to recover."
                print(msg)
                if not dry_run:
                    # ctrl.keyback()
                    time.sleep(random.uniform(0.6, 1.2))
            continue

        MIN_CONF = 0.5  # or 0.6, tune as you like
        if det.conf < MIN_CONF:
            print(f"[{step}] HELP detected at conf={det.conf:.2f} but below MIN_CONF={MIN_CONF:.2f}, ignoring.")
            continue

        now = time.time()
        since_last = now - last_action_ts

        # Rate limiting
        if since_last < min_interval:
            print(
                f"[{step}] HELP detected but rate-limited "
                f"(since_last={since_last:.1f}s < {min_interval:.1f}s)."
            )
            continue

        # Random miss probability (hesitation)
        if random.random() < miss_click_prob:
            print(
                f"[{step}] HELP detected at conf={det['conf']:.2f} "
                f"but skipping due to miss_click_prob."
            )
            continue

        # Jittered click point
        tx, ty = jitter_point(det["cx"], det["cy"], jitter_radius_px)

        print(
            f"[{step}] {'WOULD CLICK' if dry_run else 'CLICKING'} "
            f"HELP at ({tx}, {ty}), conf={det['conf']:.2f}, "
            f"interval={since_last:.1f}s"
        )

        if not dry_run:
            ctrl.tap(tx, ty)
            last_action_ts = time.time()
            time.sleep(random.uniform(*click_delay_range))
        else:
            # In dry-run, just simulate the timing
            last_action_ts = now


if __name__ == "__main__":
    # Start in shadow mode; switch dry_run=False once you like the behavior
    run_help_bot(
        "F:\\work\\LastWarRobot\\runs\\detect\\train12\\weights\\best.pt",
        dry_run=False)
