# help_bot_loop.py
import time
import random
import json
from pathlib import Path
import datetime as dt
import torch
import cv2
import numpy as np
from controllers.last_war_controller import LastWarController
from policies.help_policy import HelpOnlyPolicy
from policies.rl_help_policy import RLHelpPolicy

ADB_PATH = ("C:\\Users\\human\\AppData\\Local\\Android\\Sdk\\platform-tools"
            "\\adb.exe")

LOG_DIR = Path("data/live_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def jitter_point(x: int, y: int, radius_px: int = 8) -> tuple[int, int]:
    """Return a jittered (x, y) within ±radius_px."""
    return (
        x + random.randint(-radius_px, radius_px),
        y + random.randint(-radius_px, radius_px),
    )


def make_run_log_dir(tag: str | None = None) -> Path:
    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S")
    suffix = f"_{tag}" if tag else ""
    run_dir = LOG_DIR / f"run_{ts}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_event(
    frame_np: np.ndarray,
    step: int,
    det: dict | None,
    will_click: bool,
    reason: str,
    dry_run: bool,
    rl_action: int | None = None,
    rl_reason: str | None = None,
    log_dir: Path = LOG_DIR,
) -> None:
    """Save a frame + JSON metadata for v2 twin dataset building."""
    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    base = f"frame_{step:06d}_{ts}"
    img_path = log_dir / f"{base}.png"
    meta_path = log_dir / f"{base}.json"

    # Save image
    cv2.imwrite(str(img_path), frame_np)

    det_payload = None
    if det is not None:
        det_payload = {
            "cx": int(det["cx"]),
            "cy": int(det["cy"]),
            "conf": float(det["conf"]),
            "bbox": [float(x) for x in det["bbox"]],
        }

    meta = {
        "step": step,
        "timestamp": ts,
        "will_click": will_click,
        "reason": reason,
        "dry_run": dry_run,
        "det": det_payload,
        "rl_action": rl_action,
        "rl_reason": rl_reason,
    }
    meta_path.write_text(json.dumps(meta))


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
    rl_model_path: str | None = None,
    shadow_rl: bool = False,
    log_tag: str | None = None,
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

    log_dir = make_run_log_dir(log_tag)
    print(f"[HELP BOT] Logging to {log_dir}")

    policy = HelpOnlyPolicy(
        model_path=model_path,
        device=device,
        conf_thres=conf_thres,
    )

    rl_policy = None
    if rl_model_path is not None:
        rl_policy = RLHelpPolicy(
            model_path=rl_model_path,
            since_last_clip=60.0,
            device="cpu",  # PPO MLP on CPU is fine
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

        # --- RL shadow decision (does NOT act) ---
        rl_action = None
        rl_reason = None
        now = time.time()
        if rl_policy is not None:
            rl_action = rl_policy.decide(det, now)
            if rl_action == 0:
                rl_reason = "rl_noop"
            elif rl_action == 1:
                rl_reason = "rl_click_help"
            elif rl_action == 2:
                rl_reason = "rl_random_swipe"

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
        print(det)
        if det["conf"] < MIN_CONF:
            print(f"[{step}] HELP detected at conf={det['conf']:.2f} but below MIN_CONF={MIN_CONF:.2f}, ignoring.")
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

        save_event(frame, step, det, True, "click_help", dry_run,
                   rl_action, rl_reason, log_dir)

        if not dry_run:
            ctrl.tap(tx, ty)
            last_action_ts = time.time()
            # tell RL that a help was actually clicked (for its
            # internal since_last_help)
            if rl_policy is not None:
                rl_policy.notify_help_clicked(last_action_ts)

            time.sleep(random.uniform(*click_delay_range))
        else:
            # In dry-run, just simulate the timing
            last_action_ts = now


if __name__ == "__main__":
    # Start in shadow mode; switch dry_run=False once you like the behavior
    run_help_bot(
        "E:\\trainingdata\\last-war-robot\\runs\\detect\\train12\\weights\\best.pt",
        dry_run=False,
        rl_model_path="F:\\work\\LastWarRobot\\runs\\rl_help_v3\\ppo_lastwar_help.zip",
        shadow_rl=True,
        log_tag="rl_help_shadowrun",)