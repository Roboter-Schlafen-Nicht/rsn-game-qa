#!/usr/bin/env python
"""Auto-annotate Breakout 71 frames for YOLO training.

Uses a dual approach for reliable object detection:

1. **Color segmentation** — HSV thresholding finds bricks (all 21 palette
   colors grouped into 11 HSV ranges), paddle (white), ball (white),
   coins (small yellow), walls (game zone boundaries).
2. **Frame differencing** — Subtracting consecutive frames reveals motion.
   Moving objects (ball, paddle, coins) light up in the diff; static UI
   elements (menu button, score text, coin counter) cancel out.

Combining both signals dramatically reduces false positives from UI
elements and improves ball/coin localization.

YOLO class indices (matching configs/training/breakout-71.yaml):
    0 = paddle
    1 = ball
    2 = brick
    3 = powerup  (coins)
    4 = wall

Usage::

    python scripts/auto_annotate.py output/dataset_20260215_171450
    python scripts/auto_annotate.py output/dataset_20260215_171450 --visualize
    python scripts/auto_annotate.py output/dataset_20260215_171450 --dry-run -v
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# YOLO class indices (must match configs/training/breakout-71.yaml order)
CLS_PADDLE = 0
CLS_BALL = 1
CLS_BRICK = 2
CLS_POWERUP = 3
CLS_WALL = 4

# Browser chrome height (tab bar + URL bar + automation banner).
# Stable for Chrome at 1280x1024.
CHROME_HEIGHT = 130

# Game UI exclusion: the "Level X/Y" / menu button sits in the top-left
# corner just below the browser chrome.  The "$N" score label sits in the
# top-right corner.  Both are static and must not be detected as game
# objects.  We define UI zones as (y_start, y_end, x_start, x_end)
# relative to the full image.  Computed dynamically for each frame.

# Minimum brick area in pixels (individual bricks are ~89x89 ≈ 7900 px²).
MIN_BRICK_AREA = 2000

# Coin/powerup area range (small yellow circles).
MAX_COIN_AREA = 800
MIN_COIN_AREA = 10

# Ball area range after dilation.
MIN_BALL_AREA = 50

# Paddle geometry.
MIN_PADDLE_WIDTH = 40
MIN_PADDLE_ASPECT = 2.5  # width / height

# Wall thickness in pixels.
WALL_THICKNESS = 8

# Gameplay-frame heuristic: need enough bricks or a paddle to be gameplay.
MIN_BRICKS_GAMEPLAY = 3

# Motion-mask threshold: pixel intensity difference above this counts as
# movement.  Tuned for 0.5 s inter-frame interval at 60 fps game.
MOTION_THRESHOLD = 25

# Motion overlap: fraction of a detection's bounding box that must
# overlap with the motion mask to be considered "moving".
MOTION_OVERLAP_BALL = 0.10  # ball is small, even a few motion pixels count
MOTION_OVERLAP_BALL_CONFIRMED = 0.50  # high-confidence ball confirmation
MOTION_OVERLAP_COIN = 0.05  # coins are tiny
MOTION_OVERLAP_PADDLE = 0.05  # paddle may move slowly

# Expected brick size in pixels (from empirical measurement).
EXPECTED_BRICK_SIZE = 89


# ---------------------------------------------------------------------------
# Motion mask
# ---------------------------------------------------------------------------


def compute_motion_mask(
    frame: np.ndarray,
    prev_frame: np.ndarray,
    threshold: int = MOTION_THRESHOLD,
) -> np.ndarray:
    """Compute a binary motion mask from two consecutive frames.

    Parameters
    ----------
    frame, prev_frame : np.ndarray
        BGR uint8 images of the same size.
    threshold : int
        Grayscale intensity difference threshold.

    Returns
    -------
    np.ndarray
        Binary mask (uint8, 0 or 255) where motion was detected.
    """
    diff = cv2.absdiff(frame, prev_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, motion = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)

    # Dilate slightly to fill gaps in moving objects
    kernel = np.ones((5, 5), np.uint8)
    motion = cv2.dilate(motion, kernel, iterations=1)

    # Zero out browser chrome — UI flicker is not game motion
    motion[:CHROME_HEIGHT, :] = 0

    return motion


def _box_motion_overlap(
    motion_mask: np.ndarray,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    img_h: int,
    img_w: int,
) -> float:
    """Compute fraction of a normalized bounding box covered by motion.

    Parameters
    ----------
    motion_mask : np.ndarray
        Binary motion mask (full image size).
    x_center, y_center, width, height : float
        YOLO-normalized coordinates.
    img_h, img_w : int
        Image dimensions.

    Returns
    -------
    float
        Fraction of box area that overlaps with motion pixels (0.0–1.0).
    """
    cx = x_center * img_w
    cy = y_center * img_h
    w = width * img_w
    h = height * img_h
    x1 = max(0, int(cx - w / 2))
    y1 = max(0, int(cy - h / 2))
    x2 = min(img_w, int(cx + w / 2))
    y2 = min(img_h, int(cy + h / 2))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi = motion_mask[y1:y2, x1:x2]
    box_area = (x2 - x1) * (y2 - y1)
    if box_area == 0:
        return 0.0
    return float(roi.sum() / 255) / box_area


# ---------------------------------------------------------------------------
# UI exclusion
# ---------------------------------------------------------------------------


def _build_ui_mask(img_h: int, img_w: int) -> np.ndarray:
    """Create a mask of UI zones to exclude from detection.

    Masks out:
    - Browser chrome (top CHROME_HEIGHT pixels)
    - Menu/level button (top-left corner below chrome)
    - Score display (top-right corner below chrome)
    - Coin counter icon near the paddle (bottom-center)

    Parameters
    ----------
    img_h, img_w : int
        Full image dimensions.

    Returns
    -------
    np.ndarray
        Binary mask (uint8, 255 = UI zone to exclude).
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    # Browser chrome
    mask[:CHROME_HEIGHT, :] = 255

    # Menu / "Level X/Y" button: top-left, ~140x45 below chrome
    mask[CHROME_HEIGHT : CHROME_HEIGHT + 45, :140] = 255

    # Score "$N": top-right, ~60x35 below chrome
    mask[CHROME_HEIGHT : CHROME_HEIGHT + 35, img_w - 60 :] = 255

    # Coin counter near paddle: the "$N" coin icon + text sits on or
    # just below the paddle.  Covers a wider area since the paddle
    # can move horizontally and the icon follows it.
    # Bottom 50px, full width — any small yellow blob at the very
    # bottom is the coin counter, not a real coin.
    mask[img_h - 50 :, :] = 255

    return mask


# ---------------------------------------------------------------------------
# Detection functions
# ---------------------------------------------------------------------------


def _detect_game_zone(gray: np.ndarray) -> tuple[int, int] | None:
    """Find the left and right boundaries of the game zone.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image (browser chrome already excluded).

    Returns
    -------
    tuple[int, int] or None
        ``(left_col, right_col)`` pixel columns, or ``None`` if not found.
    """
    col_brightness = gray.mean(axis=0)
    threshold = max(col_brightness.max() * 0.08, 5)
    game_cols = np.where(col_brightness > threshold)[0]
    if len(game_cols) < 50:
        return None
    return int(game_cols[0]), int(game_cols[-1])


def _extract_brick_contours(
    color_mask: np.ndarray,
    color_name: str,
    img_h: int,
    img_w: int,
) -> list[dict]:
    """Extract brick detections from a single color mask.

    Handles morphological separation of adjacent same-color bricks and
    grid-splitting of large merged contours.

    Parameters
    ----------
    color_mask : np.ndarray
        Binary mask (uint8, 0 or 255) for one brick color.
    color_name : str
        Label for the ``meta`` field (e.g. ``"blue"``, ``"red"``).
    img_h, img_w : int
        Full image dimensions for YOLO normalization.

    Returns
    -------
    list[dict]
        Brick detections with normalized coordinates.
    """
    detections: list[dict] = []

    # Morphological ops to separate adjacent bricks
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(color_mask, kernel, iterations=2)
    dilated = cv2.dilate(eroded, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_BRICK_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # If larger than a single brick, split into grid
        if w > EXPECTED_BRICK_SIZE * 1.5 or h > EXPECTED_BRICK_SIZE * 1.5:
            n_cols = max(1, round(w / EXPECTED_BRICK_SIZE))
            n_rows = max(1, round(h / EXPECTED_BRICK_SIZE))
            brick_w = w / n_cols
            brick_h = h / n_rows
            for row in range(n_rows):
                for col in range(n_cols):
                    bx = x + col * brick_w
                    by = y + row * brick_h
                    sub_mask = color_mask[
                        int(by) : int(by + brick_h),
                        int(bx) : int(bx + brick_w),
                    ]
                    if sub_mask.sum() / 255 > brick_w * brick_h * 0.3:
                        detections.append(
                            {
                                "cls": CLS_BRICK,
                                "x_center": (bx + brick_w / 2) / img_w,
                                "y_center": (by + brick_h / 2) / img_h,
                                "width": brick_w / img_w,
                                "height": brick_h / img_h,
                                "meta": color_name,
                            }
                        )
        else:
            detections.append(
                {
                    "cls": CLS_BRICK,
                    "x_center": (x + w / 2) / img_w,
                    "y_center": (y + h / 2) / img_h,
                    "width": w / img_w,
                    "height": h / img_h,
                    "meta": color_name,
                }
            )

    return detections


# Brick zone: bricks only appear in the upper-middle portion of the
# game area (below the browser chrome, above the paddle zone).  This
# restricts white/gray brick detection to avoid false positives from
# the paddle, ball, or background elements.
BRICK_ZONE_TOP_FRAC = 0.0  # fraction below CHROME_HEIGHT
BRICK_ZONE_BOTTOM_FRAC = 0.65  # fraction of (img_h - CHROME_HEIGHT)


def _detect_bricks(
    hsv: np.ndarray,
    ui_mask: np.ndarray,
    img_h: int,
    img_w: int,
    game_zone: tuple[int, int] | None = None,
) -> list[dict]:
    """Detect bricks by color across all 21 game palette colors.

    Bricks are static, so motion is NOT required — they are validated
    purely by color and shape.

    The game has 21 brick colors (see ``palette.json``).  Rather than
    enumerating 21 individual HSV ranges, we group them:

    1. **Blue** — H 95-125 (medium blue, sky blue)
    2. **Yellow/Gold** — H 20-35, high S
    3. **Red** — H 0-10 + H 170-180 (red, salmon, scarlet, crimson)
    4. **Orange** — H 10-20, high S
    5. **Green** — H 35-85 (lime, mint, bright green, olive)
    6. **Cyan/Aqua** — H 85-95
    7. **Purple/Violet** — H 125-140
    8. **Pink/Magenta** — H 140-175
    9. **White** — low S (<40), high V (>200), brick zone only
    10. **Gray** — low S (<40), moderate V (80-200), brick zone only
    11. **Beige/Tan** — H 10-25, low-moderate S (20-80), high V

    Parameters
    ----------
    hsv : np.ndarray
        HSV image.
    ui_mask : np.ndarray
        UI exclusion mask.
    img_h, img_w : int
        Full image dimensions for YOLO normalization.
    game_zone : tuple[int, int] or None
        ``(left_col, right_col)`` pixel columns of the game zone.

    Returns
    -------
    list[dict]
        Brick detections with normalized coordinates.
    """
    detections: list[dict] = []

    # Build a brick-zone mask: restrict certain low-contrast colors to
    # the area where bricks actually appear (upper portion of game area).
    brick_zone_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    bz_top = CHROME_HEIGHT + int((img_h - CHROME_HEIGHT) * BRICK_ZONE_TOP_FRAC)
    bz_bot = CHROME_HEIGHT + int((img_h - CHROME_HEIGHT) * BRICK_ZONE_BOTTOM_FRAC)
    brick_zone_mask[bz_top:bz_bot, :] = 255
    # Also restrict to game zone columns if known
    if game_zone is not None:
        brick_zone_mask[:, : game_zone[0]] = 0
        brick_zone_mask[:, game_zone[1] :] = 0

    # --- Saturated color masks (high S, reliable hue) ---

    # Blue bricks: H~95-125, S>40, V>80  (#6262EA, #5DA3EA)
    blue_mask = cv2.inRange(hsv, (95, 40, 80), (125, 255, 255))

    # Yellow/gold bricks: H~20-35, S>80, V>100  (#FFD300)
    yellow_mask = cv2.inRange(hsv, (20, 80, 100), (35, 255, 255))

    # Red bricks: H~0-10 + H~170-180, S>60, V>60
    # Covers #e32119, #ab0c0c, #F44848, #E67070
    red_low = cv2.inRange(hsv, (0, 60, 60), (10, 255, 255))
    red_high = cv2.inRange(hsv, (170, 60, 60), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_low, red_high)

    # Orange bricks: H~10-20, S>80, V>100  (#F29E4A)
    orange_mask = cv2.inRange(hsv, (10, 80, 100), (20, 255, 255))

    # Green bricks: H~35-85, S>40, V>40  (#59EEA3, #A1F051, #53EE53, #618227)
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))

    # Cyan/Aqua bricks: H~85-95, S>40, V>100  (#5BECEC)
    cyan_mask = cv2.inRange(hsv, (85, 40, 100), (95, 255, 255))

    # Purple/Violet bricks: H~125-140, S>40, V>80  (#A664E8)
    purple_mask = cv2.inRange(hsv, (125, 40, 80), (140, 255, 255))

    # Pink/Magenta bricks: H~140-175, S>40, V>80  (#E869E8, #E66BA8)
    pink_mask = cv2.inRange(hsv, (140, 40, 80), (175, 255, 255))

    # Beige/Tan bricks: H~10-25, S 20-80, V>150  (#e1c8b4)
    beige_mask = cv2.inRange(hsv, (10, 20, 150), (25, 80, 255))

    # --- Low-saturation masks (restricted to brick zone) ---

    # White bricks: S<40, V>200  (#FFFFFF)
    white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))
    white_mask[brick_zone_mask == 0] = 0  # restrict to brick zone

    # Gray bricks: S<40, V 80-200  (#9b9fa4)
    gray_mask = cv2.inRange(hsv, (0, 0, 80), (180, 40, 200))
    gray_mask[brick_zone_mask == 0] = 0  # restrict to brick zone

    # Collect all color masks with names
    color_masks = [
        (blue_mask, "blue"),
        (yellow_mask, "yellow"),
        (red_mask, "red"),
        (orange_mask, "orange"),
        (green_mask, "green"),
        (cyan_mask, "cyan"),
        (purple_mask, "purple"),
        (pink_mask, "pink"),
        (beige_mask, "beige"),
        (white_mask, "white"),
        (gray_mask, "gray"),
    ]

    for color_mask, color_name in color_masks:
        # Exclude UI zones
        color_mask[ui_mask == 255] = 0

        bricks = _extract_brick_contours(color_mask, color_name, img_h, img_w)
        detections.extend(bricks)

    # Deduplicate: if two detections from different color masks overlap
    # significantly, keep the one with the better (more specific) color.
    detections = _deduplicate_bricks(detections, img_h, img_w)

    return detections


def _deduplicate_bricks(
    detections: list[dict],
    img_h: int,
    img_w: int,
) -> list[dict]:
    """Remove duplicate brick detections from overlapping color masks.

    When a brick's hue falls on a boundary between two HSV ranges, it
    may be detected by both.  This removes the less-specific duplicate.

    Parameters
    ----------
    detections : list[dict]
        Raw brick detections (may contain duplicates).
    img_h, img_w : int
        Image dimensions for converting normalized coords to pixels.

    Returns
    -------
    list[dict]
        Deduplicated brick detections.
    """
    if len(detections) <= 1:
        return detections

    # Color specificity priority (lower = more specific / preferred).
    # White and gray are least specific since they match any hue.
    specificity = {
        "blue": 0,
        "yellow": 0,
        "red": 0,
        "orange": 0,
        "green": 0,
        "cyan": 0,
        "purple": 0,
        "pink": 0,
        "beige": 1,
        "white": 2,
        "gray": 2,
    }

    keep = [True] * len(detections)

    for i in range(len(detections)):
        if not keep[i]:
            continue
        di = detections[i]
        ci_x = di["x_center"] * img_w
        ci_y = di["y_center"] * img_h
        ci_w = di["width"] * img_w
        ci_h = di["height"] * img_h

        for j in range(i + 1, len(detections)):
            if not keep[j]:
                continue
            dj = detections[j]
            cj_x = dj["x_center"] * img_w
            cj_y = dj["y_center"] * img_h
            cj_w = dj["width"] * img_w
            cj_h = dj["height"] * img_h

            # Check overlap: centers within half a brick size
            if (
                abs(ci_x - cj_x) < (ci_w + cj_w) * 0.3
                and abs(ci_y - cj_y) < (ci_h + cj_h) * 0.3
            ):
                # Overlapping — keep the more specific color
                si = specificity.get(di.get("meta", ""), 1)
                sj = specificity.get(dj.get("meta", ""), 1)
                if si <= sj:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    return [d for d, k in zip(detections, keep) if k]


def _detect_paddle(
    hsv: np.ndarray,
    motion_mask: np.ndarray | None,
    img_h: int,
    img_w: int,
) -> list[dict]:
    """Detect the paddle (white, wide, near bottom of frame).

    Detection is always based on color and shape; when a motion mask is
    available it is used only as an auxiliary signal (for logging /
    diagnostics), not as a hard requirement for accepting the paddle.

    Parameters
    ----------
    hsv : np.ndarray
        HSV image.
    motion_mask : np.ndarray or None
        Binary motion mask, or None for first frame.
    img_h, img_w : int
        Full image dimensions.

    Returns
    -------
    list[dict]
        At most one paddle detection.
    """
    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 50, 255))

    # Only look in the bottom 20% of the image
    paddle_zone_start = int(img_h * 0.80)
    white_mask[:paddle_zone_start, :] = 0
    # Exclude edge columns (browser chrome remnants)
    white_mask[:, :15] = 0
    white_mask[:, img_w - 15 :] = 0

    contours, _ = cv2.findContours(
        white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(h, 1)
        if w >= MIN_PADDLE_WIDTH and aspect >= MIN_PADDLE_ASPECT and area > best_area:
            best = (x, y, w, h)
            best_area = area

    if best is None:
        return []

    x, y, w, h = best
    det = {
        "cls": CLS_PADDLE,
        "x_center": (x + w / 2) / img_w,
        "y_center": (y + h / 2) / img_h,
        "width": w / img_w,
        "height": h / img_h,
    }

    # The paddle is a reliable detection by shape alone (wide white bar
    # at the bottom), so we accept it even without motion confirmation.
    # Motion is used only as a bonus signal logged at debug level.
    if motion_mask is not None:
        overlap = _box_motion_overlap(
            motion_mask,
            det["x_center"],
            det["y_center"],
            det["width"],
            det["height"],
            img_h,
            img_w,
        )
        logger.debug("Paddle motion overlap: %.2f", overlap)

    return [det]


def _detect_ball(
    hsv: np.ndarray,
    ui_mask: np.ndarray,
    motion_mask: np.ndarray | None,
    img_h: int,
    img_w: int,
    game_zone: tuple[int, int] | None = None,
) -> list[dict]:
    """Detect the ball using color + motion.

    Strategy:
    - Find white blobs in the game area (excluding chrome, paddle zone,
      UI zones, and outside game zone).
    - If a motion mask is available, **require** that the candidate
      overlaps with motion.  This eliminates static UI text like
      "Level 1/7", "$10", etc.
    - Without motion (first frame), pick the smallest white blob in the
      game zone interior as a best guess (the actual ball, not the
      particle trail).
    - Among motion-confirmed candidates, prefer **smaller** blobs (the
      actual ball) over large dilated trail clusters.

    Parameters
    ----------
    hsv : np.ndarray
        HSV image.
    ui_mask : np.ndarray
        UI exclusion mask.
    motion_mask : np.ndarray or None
        Binary motion mask, or None for first frame.
    img_h, img_w : int
        Full image dimensions.
    game_zone : tuple[int, int] or None
        ``(left_col, right_col)`` pixel columns of the game zone.

    Returns
    -------
    list[dict]
        At most one ball detection.
    """
    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 50, 255))

    # Exclude UI zones
    white_mask[ui_mask == 255] = 0
    # Exclude paddle zone (bottom 15%)
    white_mask[int(img_h * 0.85) :, :] = 0
    # Exclude outside game zone (honeycomb background borders)
    if game_zone is not None:
        margin = 10  # small margin inside the walls
        white_mask[:, : max(0, game_zone[0] + margin)] = 0
        white_mask[:, min(img_w, game_zone[1] - margin) :] = 0

    # Dilate to merge ball + particle trail into one blob
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(white_mask, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_BALL_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(h, 1)
        if not (0.3 < aspect < 3.0):
            continue

        # Cap the bounding box size — the actual ball is small (~15-25px),
        # but dilation of particle trail can create blobs up to 60-80px.
        # Clamp to a reasonable ball size.
        max_ball_px = 30
        disp_w = min(w, max_ball_px)
        disp_h = min(h, max_ball_px)

        det = {
            "cls": CLS_BALL,
            "x_center": (x + w / 2) / img_w,
            "y_center": (y + h / 2) / img_h,
            "width": (disp_w * 0.7) / img_w,
            "height": (disp_h * 0.7) / img_h,
        }

        if motion_mask is not None:
            overlap = _box_motion_overlap(
                motion_mask,
                det["x_center"],
                det["y_center"],
                det["width"],
                det["height"],
                img_h,
                img_w,
            )
            if overlap >= MOTION_OVERLAP_BALL:
                candidates.append((area, overlap, det))
                logger.debug(
                    "Ball candidate at (%.3f, %.3f) area=%d motion=%.2f",
                    det["x_center"],
                    det["y_center"],
                    area,
                    overlap,
                )
        else:
            # No motion data — accept by area as fallback
            candidates.append((area, 1.0, det))

    if not candidates:
        return []

    # Prefer candidates with high motion overlap, then prefer SMALLER
    # area (actual ball vs. large particle trail blob).  Require at
    # least MOTION_OVERLAP_BALL_CONFIRMED to be "confirmed", then pick
    # smallest.
    confirmed = [c for c in candidates if c[1] >= MOTION_OVERLAP_BALL_CONFIRMED]
    if confirmed:
        confirmed.sort(key=lambda c: c[0])  # smallest area first
        return [confirmed[0][2]]

    # Fallback: highest motion overlap
    candidates.sort(key=lambda c: c[1], reverse=True)
    return [candidates[0][2]]


def _detect_coins(
    hsv: np.ndarray,
    ui_mask: np.ndarray,
    motion_mask: np.ndarray | None,
    img_h: int,
    img_w: int,
    brick_detections: list[dict],
    game_zone: tuple[int, int] | None = None,
) -> list[dict]:
    """Detect coins/powerups (small yellow circles).

    Coins are spawned when bricks break and fly with physics — they are
    always in motion.  The static coin counter icon near the paddle is
    excluded by the UI mask and by requiring motion overlap.

    Coins must be within the game zone boundaries.

    Parameters
    ----------
    hsv : np.ndarray
        HSV image.
    ui_mask : np.ndarray
        UI exclusion mask.
    motion_mask : np.ndarray or None
        Binary motion mask.
    img_h, img_w : int
        Full image dimensions.
    brick_detections : list[dict]
        Already-detected bricks to avoid double-counting.
    game_zone : tuple[int, int] or None
        ``(left_col, right_col)`` pixel columns of the game zone.

    Returns
    -------
    list[dict]
        Coin/powerup detections.
    """
    yellow_mask = cv2.inRange(hsv, (15, 80, 100), (35, 255, 255))
    yellow_mask[ui_mask == 255] = 0
    # Exclude outside game zone — honeycomb borders are orange/yellow
    if game_zone is not None:
        yellow_mask[:, : game_zone[0]] = 0
        yellow_mask[:, game_zone[1] :] = 0

    contours, _ = cv2.findContours(
        yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_COIN_AREA or area > MAX_COIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h

        # Skip if overlapping with a brick
        overlaps_brick = False
        for brick in brick_detections:
            if (
                abs(cx - brick["x_center"]) < brick["width"] / 2
                and abs(cy - brick["y_center"]) < brick["height"] / 2
            ):
                overlaps_brick = True
                break
        if overlaps_brick:
            continue

        det = {
            "cls": CLS_POWERUP,
            "x_center": cx,
            "y_center": cy,
            "width": w / img_w,
            "height": h / img_h,
        }

        # If motion data available, require motion overlap
        if motion_mask is not None:
            overlap = _box_motion_overlap(
                motion_mask,
                det["x_center"],
                det["y_center"],
                det["width"],
                det["height"],
                img_h,
                img_w,
            )
            if overlap < MOTION_OVERLAP_COIN:
                logger.debug(
                    "Coin rejected (no motion): (%.3f, %.3f) overlap=%.2f",
                    cx,
                    cy,
                    overlap,
                )
                continue

        detections.append(det)

    return detections


def _detect_walls(
    game_zone: tuple[int, int],
    img_h: int,
    img_w: int,
) -> list[dict]:
    """Generate wall annotations from game zone boundaries.

    Walls are static — no motion required.

    Parameters
    ----------
    game_zone : tuple[int, int]
        ``(left_col, right_col)`` from ``_detect_game_zone``.
    img_h, img_w : int
        Full image dimensions.

    Returns
    -------
    list[dict]
        Two wall detections (left and right).
    """
    left, right = game_zone
    game_y_center = (CHROME_HEIGHT + img_h) / 2 / img_h
    game_height = (img_h - CHROME_HEIGHT) / img_h

    return [
        {
            "cls": CLS_WALL,
            "x_center": left / img_w,
            "y_center": game_y_center,
            "width": WALL_THICKNESS / img_w,
            "height": game_height,
        },
        {
            "cls": CLS_WALL,
            "x_center": right / img_w,
            "y_center": game_y_center,
            "width": WALL_THICKNESS / img_w,
            "height": game_height,
        },
    ]


# ---------------------------------------------------------------------------
# Main annotation pipeline
# ---------------------------------------------------------------------------


def annotate_frame(
    img: np.ndarray,
    prev_img: np.ndarray | None = None,
) -> list[dict] | None:
    """Auto-annotate a single game frame.

    Parameters
    ----------
    img : np.ndarray
        BGR uint8 image array (H, W, 3).
    prev_img : np.ndarray or None
        Previous frame for motion detection.  If ``None``, motion-based
        filtering is disabled (first frame or standalone mode).

    Returns
    -------
    list[dict] or None
        List of detections, each with keys ``cls``, ``x_center``,
        ``y_center``, ``width``, ``height`` (all normalized to [0, 1]).
        Returns ``None`` for non-gameplay frames (menus, game-over).
    """
    img_h, img_w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Build masks
    ui_mask = _build_ui_mask(img_h, img_w)
    motion_mask = None
    if prev_img is not None and prev_img.shape == img.shape:
        motion_mask = compute_motion_mask(img, prev_img)

    # Detect game zone
    gray = cv2.cvtColor(img[CHROME_HEIGHT:], cv2.COLOR_BGR2GRAY)
    game_zone = _detect_game_zone(gray)

    # Detect objects
    bricks = _detect_bricks(hsv, ui_mask, img_h, img_w, game_zone)
    paddle = _detect_paddle(hsv, motion_mask, img_h, img_w)
    ball = _detect_ball(hsv, ui_mask, motion_mask, img_h, img_w, game_zone)
    coins = _detect_coins(
        hsv,
        ui_mask,
        motion_mask,
        img_h,
        img_w,
        bricks,
        game_zone,
    )
    walls = _detect_walls(game_zone, img_h, img_w) if game_zone else []

    # Check if this is a gameplay frame
    if len(bricks) < MIN_BRICKS_GAMEPLAY and len(paddle) == 0:
        logger.debug(
            "Skipping non-gameplay frame (bricks=%d, paddle=%d)",
            len(bricks),
            len(paddle),
        )
        return None

    return bricks + paddle + ball + coins + walls


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def detections_to_yolo(detections: list[dict]) -> str:
    """Convert detections to YOLO label format.

    Parameters
    ----------
    detections : list[dict]
        Each with ``cls``, ``x_center``, ``y_center``, ``width``,
        ``height``.

    Returns
    -------
    str
        One line per object: ``cls x_center y_center w h``.
    """
    lines = []
    for d in detections:
        lines.append(
            f"{d['cls']} {d['x_center']:.6f} {d['y_center']:.6f} "
            f"{d['width']:.6f} {d['height']:.6f}"
        )
    return "\n".join(lines)


def visualize_annotations(
    img: np.ndarray,
    detections: list[dict],
    output_path: Path,
    motion_mask: np.ndarray | None = None,
) -> None:
    """Draw bounding boxes on the image and save for visual review.

    Parameters
    ----------
    img : np.ndarray
        BGR image.
    detections : list[dict]
        Annotation detections.
    output_path : Path
        Where to save the visualization.
    motion_mask : np.ndarray or None
        If provided, overlays the motion mask as a semi-transparent
        red layer for debugging.
    """
    vis = img.copy()
    img_h, img_w = vis.shape[:2]

    # Overlay motion mask if provided
    if motion_mask is not None:
        red_overlay = np.zeros_like(vis)
        red_overlay[:, :, 2] = motion_mask  # red channel
        vis = cv2.addWeighted(vis, 0.8, red_overlay, 0.3, 0)

    colors = {
        CLS_PADDLE: (0, 255, 0),  # green
        CLS_BALL: (0, 0, 255),  # red
        CLS_BRICK: (255, 165, 0),  # orange (BGR)
        CLS_POWERUP: (0, 255, 255),  # yellow
        CLS_WALL: (255, 0, 255),  # magenta
    }
    label_names = {
        CLS_PADDLE: "paddle",
        CLS_BALL: "ball",
        CLS_BRICK: "brick",
        CLS_POWERUP: "powerup",
        CLS_WALL: "wall",
    }

    for d in detections:
        cx = d["x_center"] * img_w
        cy = d["y_center"] * img_h
        w = d["width"] * img_w
        h = d["height"] * img_h
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)

        color = colors.get(d["cls"], (255, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = label_names.get(d["cls"], str(d["cls"]))
        cv2.putText(
            vis,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
        )

    cv2.imwrite(str(output_path), vis)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    """Run auto-annotation on a captured dataset directory."""
    parser = argparse.ArgumentParser(
        description="Auto-annotate Breakout 71 frames for YOLO training.",
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Path to dataset directory containing frame PNGs",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save annotated visualizations to <dataset>/viz/",
    )
    parser.add_argument(
        "--show-motion",
        action="store_true",
        help="Overlay motion mask on visualizations (implies --visualize)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print detections without writing label files",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.show_motion:
        args.visualize = True

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.is_dir():
        logger.error("Dataset directory not found: %s", dataset_dir)
        return 1

    frames = sorted(dataset_dir.glob("frame_*.png"))
    if not frames:
        logger.error("No frame_*.png files found in %s", dataset_dir)
        return 1

    logger.info("Found %d frames in %s", len(frames), dataset_dir)

    labels_dir = dataset_dir / "labels"
    viz_dir = dataset_dir / "viz"
    if not args.dry_run:
        labels_dir.mkdir(exist_ok=True)
    if args.visualize:
        viz_dir.mkdir(exist_ok=True)

    cls_names = {
        CLS_PADDLE: "paddle",
        CLS_BALL: "ball",
        CLS_BRICK: "brick",
        CLS_POWERUP: "powerup",
        CLS_WALL: "wall",
    }
    stats = {
        "total": len(frames),
        "annotated": 0,
        "skipped": 0,
        "objects": {n: 0 for n in cls_names.values()},
    }

    prev_img = None
    for frame_path in frames:
        img = cv2.imread(str(frame_path))
        if img is None:
            logger.warning("Could not read %s", frame_path.name)
            continue

        detections = annotate_frame(img, prev_img)

        # Compute motion mask for visualization
        motion_mask = None
        if args.show_motion and prev_img is not None:
            motion_mask = compute_motion_mask(img, prev_img)

        prev_img = img  # store for next iteration

        if detections is None:
            logger.debug("Skipped %s (non-gameplay)", frame_path.name)
            stats["skipped"] += 1
            if not args.dry_run:
                label_path = labels_dir / frame_path.with_suffix(".txt").name
                label_path.write_text("")
            continue

        stats["annotated"] += 1
        for d in detections:
            name = cls_names.get(d["cls"], "unknown")
            stats["objects"][name] = stats["objects"].get(name, 0) + 1

        yolo_text = detections_to_yolo(detections)

        if args.dry_run:
            logger.info(
                "%s: %d objects (%s)",
                frame_path.name,
                len(detections),
                ", ".join(
                    f"{sum(1 for d in detections if d['cls'] == c)} {n}"
                    for c, n in cls_names.items()
                    if any(d["cls"] == c for d in detections)
                ),
            )
        else:
            label_path = labels_dir / frame_path.with_suffix(".txt").name
            label_path.write_text(yolo_text)
            logger.debug(
                "Wrote %s (%d objects)",
                label_path.name,
                len(detections),
            )

        if args.visualize:
            viz_path = viz_dir / frame_path.name
            visualize_annotations(img, detections, viz_path, motion_mask)

    # Summary
    logger.info("--- Annotation Summary ---")
    logger.info("Total frames    : %d", stats["total"])
    logger.info("Annotated       : %d", stats["annotated"])
    logger.info("Skipped (menus) : %d", stats["skipped"])
    for name, count in stats["objects"].items():
        if count > 0:
            logger.info("  %-12s : %d", name, count)

    if not args.dry_run:
        logger.info("Labels saved to : %s", labels_dir)
    if args.visualize:
        logger.info("Viz saved to    : %s", viz_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
