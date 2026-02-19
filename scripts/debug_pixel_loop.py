#!/usr/bin/env python
"""Debug script: validate the pixel-based game control pipeline end-to-end.

Launches Breakout 71, then uses Win32 APIs exclusively for runtime I/O:
- **Frame capture**: ``WindowCapture`` (PrintWindow + PW_RENDERFULLCONTENT)
- **Object detection**: ``YoloDetector`` with trained weights
- **Input control**: ``pydirectinput`` for mouse movement (absolute coords)

Selenium is kept alive for **modal handling only** (game-over and perk-
picker modals are DOM overlays, not canvas-drawn — clicking them via
Win32 coordinates is unreliable).  All runtime observation and paddle
control is purely pixel-based.

Four sequential phases
----------------------
1. **Frame capture + YOLO overlay** (~5s) — capture frames with the ball
   still stuck to the paddle (pre-start), run YOLO, draw annotated
    bounding boxes, save to ``output/debug_pixel_<timestamp>/``.
2. **Paddle tracking** (~10s) — ball is still stuck to the paddle.
   Move mouse to 5 predefined X positions, capture + YOLO after each,
   compare intended vs detected paddle position.  No risk of game over
   because the ball hasn't been released yet.
3. **Ball tracking** (~10s) — click to release the ball, then
   continuous capture + YOLO at ~60 FPS, track ball trajectory, print
   position updates.  Modal handling (game-over, perk-picker) via
   Selenium keeps the game alive.
4. **Gameplay loop** (~30s) — follow-the-ball: move paddle toward
   detected ball X.  Modal handling restarts after game-over.  Track
   FPS, detections, state transitions.

Coordinate pipeline::

    YOLO detection (normalised [0,1] relative to captured frame)
        -> frame dims (WindowCapture.width / height = client area)
        -> client pixel: x_client = norm_x * client_width
        -> screen pixel: x_screen = client_left + x_client
           (client_left from win32gui.ClientToScreen(hwnd, (0,0)))
        -> pydirectinput.moveTo(x_screen, y_screen)

Usage::

    python scripts/debug_pixel_loop.py
    python scripts/debug_pixel_loop.py --skip-setup --browser edge
    python scripts/debug_pixel_loop.py --phase 4 --duration 60
    python scripts/debug_pixel_loop.py -v
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._smoke_utils import (
    BrowserInstance,
    Timer,
    base_argparser,
    ensure_output_dir,
    setup_logging,
    timestamp_str,
)

logger = logging.getLogger(__name__)

# -- Detection colour palette (BGR) ----------------------------------------
_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "paddle": (0, 255, 0),  # green
    "ball": (0, 0, 255),  # red
    "brick": (255, 128, 0),  # blue-ish
    "powerup": (0, 255, 255),  # yellow
    "wall": (128, 128, 128),  # grey
}
_DEFAULT_COLOR = (255, 255, 255)


# -- Helpers ---------------------------------------------------------------


def _draw_detections(
    frame: np.ndarray,
    detections: list[dict],
) -> np.ndarray:
    """Draw bounding boxes and labels on a copy of the frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR image ``(H, W, 3)``.
    detections : list[dict]
        Detection dicts from ``YoloDetector.detect()``.

    Returns
    -------
    np.ndarray
        Annotated copy of the frame.
    """
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        cls = det["class_name"]
        conf = det["confidence"]
        color = _CLASS_COLORS.get(cls, _DEFAULT_COLOR)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{cls} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
    return annotated


def _detection_summary(detections: list[dict]) -> str:
    """One-line summary of detections by class."""
    counts: dict[str, int] = {}
    for det in detections:
        counts[det["class_name"]] = counts.get(det["class_name"], 0) + 1
    parts = [f"{cls}={n}" for cls, n in sorted(counts.items())]
    return ", ".join(parts) if parts else "(none)"


def _get_client_origin(hwnd: int) -> tuple[int, int]:
    """Return screen-absolute (x, y) of the client area's top-left corner.

    Parameters
    ----------
    hwnd : int
        Window handle.

    Returns
    -------
    tuple[int, int]
        ``(screen_x, screen_y)`` of client area origin.
    """
    import win32gui

    return win32gui.ClientToScreen(hwnd, (0, 0))


def _norm_to_screen(
    x_norm: float,
    y_norm: float,
    client_w: int,
    client_h: int,
    client_origin: tuple[int, int],
) -> tuple[int, int]:
    """Convert normalised [0,1] coords to absolute screen pixels.

    Parameters
    ----------
    x_norm, y_norm : float
        Normalised position within the captured frame.
    client_w, client_h : int
        Client area dimensions (from WindowCapture).
    client_origin : tuple[int, int]
        Screen-absolute (x, y) of client area top-left.

    Returns
    -------
    tuple[int, int]
        ``(screen_x, screen_y)`` absolute pixel coordinates.
    """
    x_norm_clamped = max(0.0, min(1.0, x_norm))
    y_norm_clamped = max(0.0, min(1.0, y_norm))
    x_client = min(int(x_norm_clamped * client_w), max(client_w - 1, 0))
    y_client = min(int(y_norm_clamped * client_h), max(client_h - 1, 0))
    return client_origin[0] + x_client, client_origin[1] + y_client


# -- Modal handling (Selenium) ---------------------------------------------
# Modals (game-over, perk-picker) are DOM overlays — not canvas-drawn.
# We import the canonical JS snippets from the game plugin (single
# source of truth).  This is the one place we use the browser driver
# at runtime; all observation and control is pixel-based via Win32.

from games.breakout71.modal_handler import (
    CLICK_PERK_JS as _CLICK_PERK_JS,
    DETECT_STATE_JS as _DETECT_STATE_JS,
    DISMISS_GAME_OVER_JS as _DISMISS_GAME_OVER_JS,
)


def _ensure_gameplay(
    driver,
    cap,
    client_origin: tuple[int, int],
    *,
    max_retries: int = 5,
    label: str = "",
) -> bool:
    """Dismiss any modal and ensure the game is in active gameplay.

    Uses Selenium for modal detection/dismissal (DOM elements), then
    pydirectinput to click the canvas centre to start/unpause.

    Parameters
    ----------
    driver : selenium.webdriver.Remote
        The Selenium WebDriver (kept alive for modals only).
    cap : WindowCapture
        For client area dimensions.
    client_origin : tuple[int, int]
        Screen-absolute origin of client area.
    max_retries : int
        Maximum attempts to dismiss modals and reach gameplay.
    label : str
        Context label for log messages.

    Returns
    -------
    bool
        True if gameplay state was reached.
    """
    import pydirectinput

    for attempt in range(max_retries):
        try:
            state_info = driver.execute_script(_DETECT_STATE_JS)
        except Exception as exc:
            logger.debug("State detection failed (%s): %s", label, exc)
            time.sleep(0.5)
            continue

        if state_info is None:
            state_info = {"state": "unknown"}

        state = state_info.get("state", "unknown")

        if state == "gameplay":
            return True

        if state == "game_over":
            logger.info(
                "  [%s] Game over detected — dismissing (attempt %d)",
                label,
                attempt + 1,
            )
            try:
                driver.execute_script(_DISMISS_GAME_OVER_JS)
            except Exception:
                pass
            time.sleep(1.0)
            # Click canvas centre to start new game
            cx, cy = _norm_to_screen(0.5, 0.5, cap.width, cap.height, client_origin)
            pydirectinput.click(cx, cy)
            time.sleep(1.5)

        elif state == "perk_picker":
            logger.info("  [%s] Perk picker detected — picking (attempt %d)", label, attempt + 1)
            try:
                driver.execute_script(_CLICK_PERK_JS)
            except Exception:
                pass
            time.sleep(1.0)
            # Click to unpause after perk pick
            cx, cy = _norm_to_screen(0.5, 0.5, cap.width, cap.height, client_origin)
            pydirectinput.click(cx, cy)
            time.sleep(0.5)

        else:
            # Unknown state — try clicking canvas
            logger.debug(
                "  [%s] Unknown state — clicking canvas (attempt %d)",
                label,
                attempt + 1,
            )
            cx, cy = _norm_to_screen(0.5, 0.5, cap.width, cap.height, client_origin)
            pydirectinput.click(cx, cy)
            time.sleep(1.0)

    logger.warning("  [%s] Could not reach gameplay after %d attempts", label, max_retries)
    return False


# -- Phase implementations ------------------------------------------------


def phase1_capture_and_overlay(
    cap,
    detector,
    out_dir,
    *,
    num_frames: int = 5,
    delay: float = 0.5,
) -> bool:
    """Phase 1: Capture frames, run YOLO, save annotated images.

    Returns True if at least one detection was found across all frames.
    """
    logger.info("=== Phase 1: Frame Capture + YOLO Overlay ===")
    any_detections = False

    for i in range(num_frames):
        with Timer(f"frame_{i}") as t:
            frame = cap.capture_frame()
            detections = detector.detect(frame)

        if detections:
            any_detections = True

        annotated = _draw_detections(frame, detections)

        # Save both raw and annotated
        raw_path = out_dir / f"phase1_raw_{i:02d}.png"
        ann_path = out_dir / f"phase1_annotated_{i:02d}.png"
        cv2.imwrite(str(raw_path), frame)
        cv2.imwrite(str(ann_path), annotated)

        logger.info(
            "  Frame %d: %s  (%.1fms)  -> %s",
            i,
            _detection_summary(detections),
            t.elapsed * 1000,
            ann_path.name,
        )
        if i < num_frames - 1:
            time.sleep(delay)

    if any_detections:
        logger.info("Phase 1 PASSED: YOLO detections found")
    else:
        logger.warning("Phase 1 WARNING: No detections in any frame")
    return any_detections


def phase2_paddle_tracking(
    cap,
    detector,
    out_dir,
    client_origin: tuple[int, int],
) -> bool:
    """Phase 2: Move mouse to predefined positions, verify paddle follows.

    Returns True if paddle was detected in at least 3 of 5 positions.
    """
    import pydirectinput

    logger.info("=== Phase 2: Paddle Tracking via Mouse ===")

    target_x_norms = [0.20, 0.35, 0.50, 0.65, 0.80]
    paddle_y_norm = 0.90  # paddle is near the bottom
    detections_found = 0

    for i, target_x in enumerate(target_x_norms):
        # Move mouse to target position
        sx, sy = _norm_to_screen(target_x, paddle_y_norm, cap.width, cap.height, client_origin)
        pydirectinput.moveTo(sx, sy)
        time.sleep(0.3)  # let the game react

        # Capture and detect
        frame = cap.capture_frame()
        game_state = detector.detect_to_game_state(frame, cap.width, cap.height)
        detections = game_state["raw_detections"]

        paddle = game_state["paddle"]
        if paddle is not None:
            detected_x = paddle[0]  # cx_norm
            error = abs(detected_x - target_x)
            logger.info(
                "  Pos %d: target_x=%.2f  detected_x=%.2f  error=%.3f  %s",
                i,
                target_x,
                detected_x,
                error,
                "OK" if error < 0.15 else "DRIFT",
            )
            detections_found += 1
        else:
            logger.warning("  Pos %d: target_x=%.2f  paddle NOT DETECTED", i, target_x)

        # Save annotated frame
        annotated = _draw_detections(frame, detections)
        ann_path = out_dir / f"phase2_paddle_{i:02d}_x{int(target_x * 100):03d}.png"
        cv2.imwrite(str(ann_path), annotated)

    passed = detections_found >= 3
    logger.info(
        "Phase 2 %s: paddle detected in %d/5 positions",
        "PASSED" if passed else "FAILED",
        detections_found,
    )
    return passed


def phase3_ball_tracking(
    cap,
    detector,
    out_dir,
    *,
    driver=None,
    client_origin: tuple[int, int] | None = None,
    duration: float = 10.0,
) -> bool:
    """Phase 3: Continuous capture + YOLO, track ball trajectory.

    Parameters
    ----------
    driver : selenium.webdriver.Remote, optional
        Selenium WebDriver for modal handling (game-over, perk-picker).
    client_origin : tuple[int, int], optional
        Screen-absolute origin of client area (needed for modal click).

    Returns True if ball was detected in at least 30% of frames.
    """
    logger.info("=== Phase 3: Ball Tracking (%.0fs) ===", duration)

    frame_count = 0
    ball_count = 0
    ball_positions: list[tuple[float, float, float]] = []  # (time, x, y)
    modal_recoveries = 0
    start = time.perf_counter()
    last_report = start
    last_modal_check = start

    target_fps = 60
    frame_interval = 1.0 / target_fps
    modal_check_interval = 2.0  # seconds between modal checks

    while time.perf_counter() - start < duration:
        loop_start = time.perf_counter()

        # Handle modals periodically (Selenium HTTP round-trip is ~100ms)
        if driver is not None and client_origin is not None:
            if loop_start - last_modal_check >= modal_check_interval:
                last_modal_check = loop_start
                if _ensure_gameplay(driver, cap, client_origin, label="phase3"):
                    modal_recoveries += 1
                else:
                    logger.warning("Phase 3: _ensure_gameplay failed to restore gameplay.")

        frame = cap.capture_frame()
        game_state = detector.detect_to_game_state(frame, cap.width, cap.height)
        frame_count += 1

        ball = game_state["ball"]
        if ball is not None:
            ball_count += 1
            elapsed = time.perf_counter() - start
            ball_positions.append((elapsed, ball[0], ball[1]))

        # Report every 2 seconds
        now = time.perf_counter()
        if now - last_report >= 2.0:
            elapsed = now - start
            fps = frame_count / elapsed if elapsed > 0 else 0
            if ball is not None:
                logger.info(
                    "  t=%.1fs  FPS=%.0f  ball=(%.3f, %.3f)  detected=%d/%d",
                    elapsed,
                    fps,
                    ball[0],
                    ball[1],
                    ball_count,
                    frame_count,
                )
            else:
                logger.info(
                    "  t=%.1fs  FPS=%.0f  ball=NONE  detected=%d/%d",
                    elapsed,
                    fps,
                    ball_count,
                    frame_count,
                )
            last_report = now

        # Save a frame every ~2 seconds for visual inspection
        if frame_count % (target_fps * 2) == 0:
            detections = detector.detect(frame)
            annotated = _draw_detections(frame, detections)
            save_idx = frame_count // (target_fps * 2)
            ann_path = out_dir / f"phase3_ball_{save_idx:03d}.png"
            cv2.imwrite(str(ann_path), annotated)

        # Rate limit to target FPS
        elapsed_frame = time.perf_counter() - loop_start
        sleep_time = frame_interval - elapsed_frame
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Final stats
    actual_fps = frame_count / duration if duration > 0 else 0
    detect_rate = ball_count / frame_count if frame_count > 0 else 0

    logger.info(
        "  Ball tracking: %d/%d frames (%.0f%%), avg FPS=%.0f, modal_recoveries=%d",
        ball_count,
        frame_count,
        detect_rate * 100,
        actual_fps,
        modal_recoveries,
    )

    # Compute ball movement if enough samples
    if len(ball_positions) >= 2:
        xs = [p[1] for p in ball_positions]
        ys = [p[2] for p in ball_positions]
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        logger.info(
            "  Ball movement: x_range=%.3f  y_range=%.3f  %s",
            x_range,
            y_range,
            "MOVING" if (x_range > 0.05 or y_range > 0.05) else "STATIC/STUCK",
        )

    passed = detect_rate >= 0.30
    logger.info("Phase 3 %s", "PASSED" if passed else "FAILED")
    return passed


def phase4_gameplay_loop(
    cap,
    detector,
    out_dir,
    client_origin: tuple[int, int],
    *,
    driver=None,
    duration: float = 30.0,
) -> bool:
    """Phase 4: Follow-the-ball gameplay with YOLO-only observations.

    Moves the paddle to track the ball's X position.  Tracks FPS,
    detection rates, and brick count changes.  Modal handling (game-over,
    perk-picker) via Selenium keeps the game alive across deaths.

    Parameters
    ----------
    driver : selenium.webdriver.Remote, optional
        Selenium WebDriver for modal handling.

    Returns True if the loop ran without crashing and achieved average FPS > 3.
    """
    import pydirectinput

    # Disable pydirectinput's built-in 100ms sleep after each call.
    # Default PAUSE=0.1 is a safety feature but kills FPS in tight loops.
    pydirectinput.PAUSE = 0

    logger.info("=== Phase 4: Gameplay Loop (%.0fs) ===", duration)

    target_fps = 60
    frame_interval = 1.0 / target_fps

    frame_count = 0
    ball_detected = 0
    paddle_detected = 0
    brick_counts: list[int] = []
    fps_samples: list[float] = []
    modal_recoveries = 0
    last_frame: np.ndarray | None = None

    # Per-stage timing accumulators
    t_capture_total = 0.0
    t_infer_total = 0.0
    t_modal_total = 0.0
    t_input_total = 0.0
    t_save_total = 0.0

    start = time.perf_counter()
    last_report = start
    last_frame_time = start
    last_modal_check = start
    last_paddle_x_screen: int | None = None
    modal_check_interval = 2.0  # seconds between modal checks

    while time.perf_counter() - start < duration:
        loop_start = time.perf_counter()

        # Handle modals periodically (Selenium HTTP round-trip is ~100ms)
        t_modal_start = time.perf_counter()
        if driver is not None:
            if loop_start - last_modal_check >= modal_check_interval:
                last_modal_check = loop_start
                gameplay_ok = _ensure_gameplay(driver, cap, client_origin, label="phase4")
                if gameplay_ok:
                    modal_recoveries += 1
                else:
                    logger.error(
                        "Phase 4: unable to restore gameplay via modal "
                        "handling; aborting gameplay loop."
                    )
                    break
        t_modal_total += time.perf_counter() - t_modal_start

        # Capture
        t_cap_start = time.perf_counter()
        frame = cap.capture_frame()
        t_capture_total += time.perf_counter() - t_cap_start

        last_frame = frame

        # YOLO inference
        t_inf_start = time.perf_counter()
        game_state = detector.detect_to_game_state(frame, cap.width, cap.height)
        t_infer_total += time.perf_counter() - t_inf_start

        frame_count += 1

        # Track FPS
        now = time.perf_counter()
        dt = now - last_frame_time
        last_frame_time = now
        if dt > 0:
            fps_samples.append(1.0 / dt)

        ball = game_state["ball"]
        paddle = game_state["paddle"]
        bricks = game_state["bricks"]
        brick_counts.append(len(bricks))

        if ball is not None:
            ball_detected += 1

        if paddle is not None:
            paddle_detected += 1

        # -- Control: move paddle toward ball X -------------------------
        t_input_start = time.perf_counter()
        if ball is not None:
            ball_cx = ball[0]  # normalised X of ball centre
            # Move mouse to ball's X, paddle's Y (near bottom)
            paddle_y_norm = 0.90
            sx, sy = _norm_to_screen(ball_cx, paddle_y_norm, cap.width, cap.height, client_origin)

            # Only move if position changed significantly (>5px)
            if last_paddle_x_screen is None or abs(sx - last_paddle_x_screen) > 5:
                pydirectinput.moveTo(sx, sy)
                last_paddle_x_screen = sx
        elif paddle is not None:
            # No ball detected — hold paddle position (do nothing)
            pass
        t_input_total += time.perf_counter() - t_input_start

        # -- Report every 5 seconds ------------------------------------
        if now - last_report >= 5.0:
            elapsed = now - start
            avg_fps = np.mean(fps_samples[-target_fps:]) if fps_samples else 0
            logger.info(
                "  t=%5.1fs  FPS=%.0f  ball=%d/%d  paddle=%d/%d  bricks=%d  recoveries=%d",
                elapsed,
                avg_fps,
                ball_detected,
                frame_count,
                paddle_detected,
                frame_count,
                len(bricks),
                modal_recoveries,
            )
            last_report = now

        # -- Save periodic annotated frames ----------------------------
        t_save_start = time.perf_counter()
        if frame_count % (target_fps * 5) == 0:
            raw_detections = detector.detect(frame)
            annotated = _draw_detections(frame, raw_detections)
            save_idx = frame_count // (target_fps * 5)
            ann_path = out_dir / f"phase4_gameplay_{save_idx:03d}.png"
            cv2.imwrite(str(ann_path), annotated)
        t_save_total += time.perf_counter() - t_save_start

        # Rate limit
        elapsed_frame = time.perf_counter() - loop_start
        sleep_time = frame_interval - elapsed_frame
        if sleep_time > 0:
            time.sleep(sleep_time)

    # -- Final report --------------------------------------------------
    total_elapsed = time.perf_counter() - start
    avg_fps = frame_count / total_elapsed if total_elapsed > 0 else 0
    ball_rate = ball_detected / frame_count if frame_count > 0 else 0
    paddle_rate = paddle_detected / frame_count if frame_count > 0 else 0

    logger.info("--- Phase 4 Results ---")
    logger.info("  Duration       : %.1fs", total_elapsed)
    logger.info("  Total frames   : %d", frame_count)
    logger.info("  Avg FPS        : %.0f", avg_fps)
    logger.info("  Ball detected  : %d/%d (%.0f%%)", ball_detected, frame_count, ball_rate * 100)
    logger.info(
        "  Paddle detected: %d/%d (%.0f%%)",
        paddle_detected,
        frame_count,
        paddle_rate * 100,
    )
    logger.info("  Modal recoveries: %d", modal_recoveries)

    # Per-stage timing breakdown
    if frame_count > 0:
        logger.info("--- Timing Breakdown (avg per frame) ---")
        logger.info(
            "  Capture        : %6.1fms  (%4.1f%%)",
            t_capture_total / frame_count * 1000,
            t_capture_total / total_elapsed * 100,
        )
        logger.info(
            "  YOLO inference : %6.1fms  (%4.1f%%)",
            t_infer_total / frame_count * 1000,
            t_infer_total / total_elapsed * 100,
        )
        logger.info(
            "  Modal handling : %6.1fms  (%4.1f%%)",
            t_modal_total / frame_count * 1000,
            t_modal_total / total_elapsed * 100,
        )
        logger.info(
            "  Input control  : %6.1fms  (%4.1f%%)",
            t_input_total / frame_count * 1000,
            t_input_total / total_elapsed * 100,
        )
        logger.info(
            "  Frame save     : %6.1fms  (%4.1f%%)",
            t_save_total / frame_count * 1000,
            t_save_total / total_elapsed * 100,
        )
        t_other = total_elapsed - (
            t_capture_total + t_infer_total + t_modal_total + t_input_total + t_save_total
        )
        logger.info(
            "  Other/overhead : %6.1fms  (%4.1f%%)",
            t_other / frame_count * 1000,
            t_other / total_elapsed * 100,
        )

    if brick_counts:
        logger.info(
            "  Bricks         : start=%d  end=%d  delta=%d",
            brick_counts[0],
            brick_counts[-1],
            brick_counts[0] - brick_counts[-1],
        )

    # Save final annotated frame
    if last_frame is not None:
        raw_detections = detector.detect(last_frame)
        annotated = _draw_detections(last_frame, raw_detections)
        cv2.imwrite(str(out_dir / "phase4_final.png"), annotated)

    # FPS threshold accounts for CPU-based YOLO inference (~130ms/frame).
    # On GPU/XPU the loop should easily exceed 30 FPS.
    passed = avg_fps > 3
    logger.info("Phase 4 %s (FPS > 3 required)", "PASSED" if passed else "FAILED")
    return passed


# -- Main ------------------------------------------------------------------


def main() -> int:
    parser = base_argparser("Debug pixel-based game control pipeline.")
    # Override base_argparser's --config default so plugin default kicks in
    parser.set_defaults(config=None)
    parser.add_argument(
        "--game",
        type=str,
        default="breakout71",
        help="Game plugin name (directory under games/). Default: breakout71",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip npm install step",
    )
    parser.add_argument(
        "--browser",
        type=str,
        default=None,
        help="Browser to use (chrome, edge, firefox). Default: auto-detect.",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help="Run only this phase (0 = all, default: 0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Duration for phase 4 gameplay loop in seconds (default: 30)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to YOLO weights (default: from game plugin)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        help="YOLO confidence threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device for YOLO inference: auto, xpu, cuda, cpu (default: auto)",
    )
    args = parser.parse_args()
    setup_logging(args.verbose)

    from dotenv import load_dotenv

    load_dotenv()

    # ── Setup output directory ───────────────────────────────────────
    ts = timestamp_str()
    out_dir = (
        ensure_output_dir(f"debug_pixel_{ts}") if args.output_dir is None else args.output_dir
    )
    logger.info("Output directory: %s", out_dir)

    # ── Load game plugin and resolve config ────────────────────────────
    from games import load_game_plugin

    plugin = load_game_plugin(args.game)

    # ── Load YOLO model ──────────────────────────────────────────────
    from src.perception import YoloDetector, resolve_device

    device = resolve_device(args.device)
    weights = args.weights or plugin.default_weights
    logger.info("Loading YOLO model: %s (device=%s)", weights, device)
    with Timer("yolo_load") as t:
        detector = YoloDetector(
            weights_path=weights,
            device=device,
            confidence_threshold=args.confidence,
        )
        detector.load()
    logger.info(
        "YOLO loaded in %.1fs  classes=%s",
        t.elapsed,
        detector.class_names,
    )

    # ── Start game server ────────────────────────────────────────────
    from src.game_loader import create_loader, load_game_config

    config_path = Path(args.config if args.config else plugin.default_config)
    config = load_game_config(config_path.stem, configs_dir=config_path.parent)
    loader = create_loader(config)

    if not args.skip_setup:
        logger.info("Running setup (npm install) ...")
        loader.setup()

    logger.info("Starting game server ...")
    with Timer("server_start") as t:
        loader.start()
    logger.info("Game ready at %s (%.1fs)", loader.url, t.elapsed)

    # ── Launch browser ───────────────────────────────────────────────
    url = loader.url or config.url
    logger.info("Launching browser: %s", url)
    browser = BrowserInstance(
        url,
        settle_seconds=5,
        window_size=(config.window_width, config.window_height),
        browser=args.browser,
    )

    # ── Set up screen capture ──────────────────────────────────────────
    # Prefer wincam (Direct3D11, <1ms per frame) over PrintWindow (~25ms).
    window_title = config.window_title or "Breakout"
    try:
        from src.capture.wincam_capture import WinCamCapture

        cap = WinCamCapture(window_title=window_title, fps=60)
        logger.info(
            "Capture: WinCamCapture (Direct3D11)  HWND=%s, %dx%d",
            cap.hwnd,
            cap.width,
            cap.height,
        )
    except (ImportError, RuntimeError, OSError) as exc:
        logger.warning("WinCamCapture unavailable (%s), falling back to PrintWindow", exc)
        from src.capture import WindowCapture

        cap = WindowCapture(window_title=window_title)
        logger.info(
            "Capture: WindowCapture (PrintWindow)  HWND=%s, %dx%d",
            cap.hwnd,
            cap.width,
            cap.height,
        )

    # Get client area origin in screen coordinates
    client_origin = _get_client_origin(cap.hwnd)
    logger.info("Client area origin: (%d, %d)", *client_origin)

    # ── Bring window to foreground ───────────────────────────────────
    import win32gui

    try:
        win32gui.SetForegroundWindow(cap.hwnd)
        time.sleep(0.5)
    except Exception as exc:
        logger.warning("Could not set foreground window: %s", exc)

    # Keep driver reference for modal handling in phases 3-4
    driver = browser.driver

    # NOTE: We do NOT click yet — phases 1-2 run with ball stuck to
    # paddle (pre-start).  Click happens before phase 3.
    import pydirectinput

    # Disable the built-in 100ms sleep after every pydirectinput call.
    # The default PAUSE=0.1 caps the control loop to ~10 FPS.
    pydirectinput.PAUSE = 0

    # ── Run phases ───────────────────────────────────────────────────
    results: dict[str, bool] = {}
    run_all = args.phase == 0

    try:
        # -- Phases 1-2: ball stuck to paddle (safe, no game over risk) --
        if run_all or args.phase == 1:
            results["phase1"] = phase1_capture_and_overlay(cap, detector, out_dir)

        if run_all or args.phase == 2:
            results["phase2"] = phase2_paddle_tracking(cap, detector, out_dir, client_origin)

        # -- Click canvas centre to release the ball -------------------
        # Ball was stuck to paddle through phases 1-2.  One click on the
        # canvas starts the game: play() sets running=true and
        # ballStickToPuck=false.  After this, only moveTo — mouseup
        # while running pauses the game (game.ts:247-256).
        if run_all or args.phase >= 3:
            cx, cy = _norm_to_screen(0.5, 0.5, cap.width, cap.height, client_origin)
            logger.info("Clicking canvas centre to release ball: (%d, %d)", cx, cy)
            pydirectinput.click(cx, cy)
            time.sleep(1.0)

        # -- Phases 3-4: ball released (active gameplay) ---------------
        if run_all or args.phase == 3:
            results["phase3"] = phase3_ball_tracking(
                cap,
                detector,
                out_dir,
                driver=driver,
                client_origin=client_origin,
                duration=10.0,
            )

        if run_all or args.phase == 4:
            results["phase4"] = phase4_gameplay_loop(
                cap,
                detector,
                out_dir,
                client_origin,
                driver=driver,
                duration=args.duration,
            )

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as exc:
        logger.error("Phase failed: %s", exc, exc_info=True)
    finally:
        # ── Cleanup ──────────────────────────────────────────────────
        logger.info("Cleaning up ...")
        cap.release()
        browser.close()
        loader.stop()

    # ── Final report ─────────────────────────────────────────────────
    logger.info("=== Final Results ===")
    all_passed = True
    for phase, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        logger.info("  %-8s : %s", phase, status)
        if not passed:
            all_passed = False

    if all_passed and results:
        logger.info("All phases PASSED")
    elif results:
        logger.warning("Some phases FAILED")
    else:
        logger.warning("No phases completed")

    logger.info("Output saved to: %s", out_dir)
    return 0 if all_passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as exc:
        logger.critical("FAILED: %s", exc, exc_info=True)
        sys.exit(1)
